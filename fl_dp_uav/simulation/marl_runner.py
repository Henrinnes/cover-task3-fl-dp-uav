"""
MARLSimulationRunner — extends FullSimulationRunner with two extra methods:

  _get_snr_dict()
      Returns {uav_id: snr_db} after stepping UAV mobility, used by
      MARLFLTrainer to build agent state vectors.

  _fl_round(participants, epochs_chosen, dp_cfg, round_num)
      Runs one FL round with the participant set and per-UAV epoch counts
      chosen by the MARL agents, including:
        - Parallel DP-SGD training across 3 GPUs
        - Byzantine / free-rider detection + robust aggregation
        - PoRS proof generation + PoA blockchain block
        - Smart contract token distribution
        - Global model evaluation

      Returns (accuracy, contributions, privacy_eps) where
      contributions[uav_id] = cosine similarity of that agent's gradient
      to the aggregated model — used to shape rewards.

The actual training loop is driven by MARLFLTrainer (marl/trainer.py), which
calls _fl_round() once per round after the agents have chosen their actions.
"""
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import torch

from blockchain.block import Transaction
from differential_privacy.accountant import max_client_epsilon
from .full_runner import FullSimulationRunner


class MARLSimulationRunner(FullSimulationRunner):
    """Full-stack runner that exposes a per-round API for MARL agent control."""

    # ------------------------------------------------------------------ #
    # State helpers called by MARLFLTrainer
    # ------------------------------------------------------------------ #

    def _get_snr_dict(self) -> Dict[int, float]:
        """Step mobility and return {uav_id: snr_db} for every non-PS UAV."""
        self.mobility.step(self.network.nodes, dt=self.cfg["mobility"]["step_dt"])
        ps = self.network.nodes[0]
        return {
            node.node_id: float(self.channel.snr_db(node, ps))
            for node in self.network.nodes[1:]
        }

    # ------------------------------------------------------------------ #
    # Core FL round
    # ------------------------------------------------------------------ #

    def _fl_round(
        self,
        participants: List[int],
        epochs_chosen: List[int],
        dp_cfg: dict,
        round_num: int,
    ) -> Tuple[float, Dict[int, float], float]:
        """
        Execute one FL round driven by MARL-chosen participants/epochs.

        Parameters
        ----------
        participants  : list of UAV IDs that will train this round
        epochs_chosen : epochs_chosen[i] = local epochs for UAV i (0 = skip)
        dp_cfg        : dp section from config.yaml
        round_num     : 1-based round counter (for blockchain / logging)

        Returns
        -------
        accuracy      : global test accuracy after aggregation
        contributions : {uav_id: cos-sim contribution ∈ [0, 1]}
        privacy_eps   : cumulative DP-ε after this round
        """
        fl_default = self.cfg["fl"]["local_epochs"]

        # Distribute current global weights to all participants
        global_w = self.server.get_weights()
        for cid in participants:
            self.clients[cid].set_weights(global_w)

        training_times: Dict[int, float] = {}
        raw_updates: List[Tuple[int, dict]] = []

        def _train(cid: int):
            self.clients[cid].local_epochs = epochs_chosen[cid]
            t0 = time.time()
            w, _ = self.clients[cid].train()
            elapsed = time.time() - t0
            self.clients[cid].local_epochs = fl_default

            if cid in self.byzantine_attackers:
                w = self.byzantine_attackers[cid].corrupt(w)
            elif cid in self.freerider_attackers:
                w = self.freerider_attackers[cid].free_ride(w)

            if dp_cfg["enabled"]:
                # Each client's own accountant -- no shared state, no lock
                # needed. Adaptive clipping (if enabled) rides this same
                # per-step subsampled query -- see FLClient.
                self.client_accountants[cid].accumulate(
                    noise_multiplier=dp_cfg["noise_multiplier"],
                    sample_rate=self.clients[cid].sample_rate,
                    num_steps=self.clients[cid].last_num_steps,
                )
            return cid, w, elapsed

        with ThreadPoolExecutor(max_workers=len(participants)) as pool:
            futures = {pool.submit(_train, cid): cid for cid in participants}
            for future in as_completed(futures):
                cid, w, elapsed = future.result()
                raw_updates.append((cid, w))
                training_times[cid] = elapsed

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Security filtering
        _, byz_flagged = self.byzantine_defense.detect(raw_updates)
        genuine, fr_flagged = self.freerider_detector.detect(raw_updates)
        blocked = set(byz_flagged) | set(fr_flagged)

        clean_updates = [
            (w, self.clients[cid].num_samples)
            for cid, w in raw_updates if cid not in blocked
        ]
        if not clean_updates:
            clean_updates = [
                (w, self.clients[cid].num_samples) for cid, w in raw_updates
            ]

        # Robust aggregation
        aggregated = self.aggregation_fn(clean_updates)
        self.server.model.load_state_dict(aggregated)
        self.server.round += 1

        # Contribution scores (cosine similarity to aggregated gradient)
        contributions = self._compute_contributions(raw_updates, aggregated, blocked)

        # PoRS proofs → blockchain transactions
        for cid, w in raw_updates:
            if cid in blocked:
                continue
            proof = self.pors_prover.build_proof(
                uav_id=cid,
                fl_round=round_num,
                state_dict=w,
                num_samples=self.clients[cid].num_samples,
                compute_time_s=training_times.get(cid, 1.0),
            )
            tx = Transaction(
                tx_type="DATA_SHARE",
                sender_id=cid,
                recipient_id=0,
                fl_round=round_num,
                payload={"pors_proof": proof.to_dict()},
            )
            for vid in self.consensus.validators:
                if vid in self.chains:
                    self.chains[vid].add_transaction(tx)

        # PoA consensus → block finalisation → smart contract
        block = self.consensus.propose_and_finalise(
            self.chains, fl_round=round_num, exclude_ids=list(blocked)
        )
        if block is not None:
            self.contract.process_block(block.transactions, fl_round=round_num)

        self.chain_heights.append(self.chains[0].height)
        self.blocked_counts.append(len(blocked))

        # Evaluate global model
        _, accuracy = self.server.evaluate(self.test_loader)
        self.tracker.record(round_num, 0.0, accuracy, 0.0, len(participants))

        priv_eps = (
            max_client_epsilon(self.client_accountants, dp_cfg["target_delta"])
            if dp_cfg["enabled"] else 0.0
        )

        return float(accuracy), contributions, float(priv_eps)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _flatten_sd(sd: dict) -> torch.Tensor:
        return torch.cat([
            v.detach().cpu().float().flatten()
            for v in sd.values() if v.is_floating_point()
        ])

    def _compute_contributions(
        self,
        raw_updates: List[Tuple[int, dict]],
        aggregated: dict,
        blocked: set,
    ) -> Dict[int, float]:
        """Cosine similarity of each update to the aggregated model (reward signal)."""
        agg_vec = self._flatten_sd(aggregated)
        contribs: Dict[int, float] = {}
        for cid, w in raw_updates:
            if cid in blocked:
                contribs[cid] = 0.0
                continue
            vec = self._flatten_sd(w)
            cos = torch.nn.functional.cosine_similarity(
                vec.unsqueeze(0), agg_vec.unsqueeze(0)
            ).item()
            contribs[cid] = float(max(cos, 0.0))
        return contribs
