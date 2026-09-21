"""
Full COVER Task 3.2 simulation: FL + DP + Blockchain + Incentive + Security.

Each FL round proceeds as follows:
  1. Mobility update (UAV positions)
  2. Channel-aware participant selection
  3. Game-theoretic incentive: PS computes λ*; UAVs declare effort
  4. Local DP-SGD training (high-effort UAVs train more epochs)
  5. Security filtering: Byzantine / free-rider detection before aggregation
  6. Robust aggregation (configurable: FedAvg / median / trimmed-mean)
  7. PoRS proof generation per participant
  8. Blockchain transactions broadcast → PoA consensus → block finalised
  9. Smart contract: verify PoRS, distribute token rewards, update reputation
 10. Log metrics (accuracy, loss, ε, λ*, participation, chain height)
"""

import copy
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from blockchain.chain import Blockchain
from blockchain.consensus import PoAConsensus
from blockchain.block import Transaction
from blockchain.smart_contract import ResourceSharingContract
from differential_privacy.accountant import RDPAccountant
from federated_learning.client import FLClient
from federated_learning.server import FLServer
from federated_learning.convergence import ConvergenceTracker
from incentive.game import build_game
from incentive.proof_of_resource import PoRSProver, PoRSVerifier
from security.attacks import ByzantineAttack, FreeRiderAttack
from security.defense import (
    ByzantineDefense,
    FreeRiderDetector,
    SybilDefense,
    coordinate_wise_median,
    fedavg,
    trimmed_mean,
)
from .runner import SimulationRunner  # inherits UAV network + FL setup


class FullSimulationRunner(SimulationRunner):
    """
    Extends SimulationRunner with blockchain, incentive, and security layers.

    Extra config keys (add to config.yaml under a 'security' and 'incentive' block):

    incentive:
      quality_weight: 10.0
      cost_mode: heterogeneous   # homogeneous | heterogeneous | adversarial

    security:
      aggregation: fedavg        # fedavg | median | trimmed_mean
      byzantine_fraction: 0.1   # fraction of UAVs that are Byzantine
      byzantine_mode: sign_flip  # random | sign_flip | label_flip
      free_rider_fraction: 0.1
      free_rider_mode: noise

    blockchain:
      num_validators: 5
      fault_tolerance: 1
    """

    def __init__(self, config_path: str):
        super().__init__(config_path)
        self._build_extra_layers()

    def _build_extra_layers(self):
        cfg = self.cfg
        n = self.network.num_uavs

        # --- Blockchain ---
        bc_cfg = cfg.get("blockchain", {})
        num_validators = min(bc_cfg.get("num_validators", 5), n)
        fault_tol = bc_cfg.get("fault_tolerance", 1)

        # UAVs with highest initial reputation become validators
        validator_ids = list(range(num_validators))
        self.chains: Dict[int, Blockchain] = {
            i: Blockchain(node_id=i) for i in range(n)
        }
        self.consensus = PoAConsensus(validator_ids, fault_tolerance=fault_tol)
        self.contract = ResourceSharingContract()
        for i in range(n):
            self.contract.register_uav(i)

        # --- Incentive ---
        inc_cfg = cfg.get("incentive", {})
        self.game = build_game(
            num_uavs=n,
            cost_mode=inc_cfg.get("cost_mode", "heterogeneous"),
            quality_weight=inc_cfg.get("quality_weight", 10.0),
            rng=self.rng,
        )

        # --- PoRS ---
        self.pors_prover = PoRSProver()
        self.pors_verifier = PoRSVerifier()

        # --- Security: attacks ---
        sec_cfg = cfg.get("security", {})
        byz_frac = sec_cfg.get("byzantine_fraction", 0.1)
        fr_frac = sec_cfg.get("free_rider_fraction", 0.1)
        n_byzantine = max(0, int(byz_frac * (n - 1)))
        n_freeriders = max(0, int(fr_frac * (n - 1)))

        byz_ids = list(range(1, 1 + n_byzantine))
        fr_ids = list(range(1 + n_byzantine, 1 + n_byzantine + n_freeriders))
        self.byzantine_ids = set(byz_ids)
        self.freerider_ids = set(fr_ids)

        byz_mode = sec_cfg.get("byzantine_mode", "sign_flip")
        fr_mode = sec_cfg.get("free_rider_mode", "noise")
        self.byzantine_attackers = {
            cid: ByzantineAttack(mode=byz_mode) for cid in byz_ids
        }
        self.freerider_attackers = {
            cid: FreeRiderAttack(mode=fr_mode) for cid in fr_ids
        }

        # --- Security: defences ---
        self.sybil_defense = SybilDefense()
        self.byzantine_defense = ByzantineDefense()
        self.freerider_detector = FreeRiderDetector()

        agg = sec_cfg.get("aggregation", "fedavg")
        self.aggregation_fn = {
            "fedavg": fedavg,
            "median": coordinate_wise_median,
            "trimmed_mean": trimmed_mean,
        }.get(agg, fedavg)

        # Extended tracker fields
        self.chain_heights: List[int] = []
        self.lambda_stars: List[float] = []
        self.blocked_counts: List[int] = []
        self.token_balances_history: List[Dict[int, float]] = []

    # ------------------------------------------------------------------
    # Override run()
    # ------------------------------------------------------------------

    def run(self, output_dir: str = "./results/full") -> Dict:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        dp_cfg = self.cfg["dp"]
        sec_cfg = self.cfg.get("security", {})

        gpu_summary = ", ".join(
            f"GPU{i}:{torch.cuda.get_device_name(i).split()[-1]}"
            for i in range(torch.cuda.device_count())
        ) if torch.cuda.is_available() else "CPU"

        print("=" * 70)
        print("COVER Task 3.2 — FULL Simulation (FL + DP + Blockchain + Incentive)")
        print(f"  UAVs           : {self.network.num_uavs}")
        print(f"  GPUs in use    : {gpu_summary}")
        print(f"  Byzantine UAVs : {len(self.byzantine_ids)}")
        print(f"  Free-rider UAVs: {len(self.freerider_ids)}")
        print(f"  Aggregation    : {sec_cfg.get('aggregation', 'fedavg')}")
        print(f"  DP enabled     : {dp_cfg['enabled']}")
        print(f"  Blockchain     : PoA, {len(self.consensus.validators)} validators")
        print("=" * 70)

        for rnd in range(1, self.num_rounds + 1):
            # 1. Mobility
            self.mobility.step(self.network.nodes, dt=self.cfg["mobility"]["step_dt"])

            # 2. Participant selection (channel-aware)
            selected = self._select_participants()
            if not selected:
                continue

            # 3. Game-theoretic incentive: compute λ* and UAV efforts
            game_result = self.game.solve(fl_round=rnd)
            lambda_star = game_result["lambda_star"]

            # 4. Effort-scaled epochs per client (game-driven)
            fl_default = self.cfg["fl"]["local_epochs"]
            epoch_overrides = {
                cid: max(1, round(fl_default * game_result["efforts"].get(cid, 0.5)))
                for cid in selected
            }

            # 5. Parallel training across all 3 GPUs
            #    Distribute weights, inject attacks inside thread, collect results.
            global_w = self.server.get_weights()
            for cid in selected:
                self.clients[cid].set_weights(global_w)

            training_times: Dict[int, float] = {}
            raw_updates: List[tuple] = []

            def _train_full(cid: int):
                self.clients[cid].local_epochs = epoch_overrides[cid]
                t0 = time.time()
                w, _ = self.clients[cid].train()
                elapsed = time.time() - t0
                self.clients[cid].local_epochs = fl_default  # restore

                # Attack injection (adversarial clients)
                if cid in self.byzantine_attackers:
                    w = self.byzantine_attackers[cid].corrupt(w)
                elif cid in self.freerider_attackers:
                    w = self.freerider_attackers[cid].free_ride(w)

                if dp_cfg["enabled"]:
                    with self._accountant_lock:
                        self.accountant.accumulate(
                            noise_multiplier=dp_cfg["noise_multiplier"],
                            sample_rate=self.clients[cid].sample_rate,
                            num_steps=epoch_overrides[cid],
                        )
                return cid, w, elapsed

            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=len(selected)) as pool:
                futures = {pool.submit(_train_full, cid): cid for cid in selected}
                for future in as_completed(futures):
                    cid, w, elapsed = future.result()
                    raw_updates.append((cid, w))
                    training_times[cid] = elapsed

            # Release cached CUDA memory after each round's vmap passes to
            # prevent gradual memory fragmentation from inflating round times.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 5. Security filtering
            _, byz_flagged = self.byzantine_defense.detect(raw_updates)
            genuine, fr_flagged = self.freerider_detector.detect(raw_updates)
            blocked = set(byz_flagged) | set(fr_flagged)

            clean_updates = [
                (w, self.clients[cid].num_samples)
                for cid, w in raw_updates
                if cid not in blocked
            ]
            self.blocked_counts.append(len(blocked))

            if not clean_updates:
                clean_updates = [
                    (w, self.clients[cid].num_samples)
                    for cid, w in raw_updates
                ]

            # 6. Robust aggregation
            aggregated = self.aggregation_fn(clean_updates)
            self.server.model.load_state_dict(aggregated)
            self.server.round += 1

            # 7. PoRS proofs + blockchain transactions
            for cid, w in raw_updates:
                if cid in blocked:
                    continue
                proof = self.pors_prover.build_proof(
                    uav_id=cid,
                    fl_round=rnd,
                    state_dict=w,
                    num_samples=self.clients[cid].num_samples,
                    compute_time_s=training_times.get(cid, 1.0),
                )
                tx = Transaction(
                    tx_type="DATA_SHARE",
                    sender_id=cid,
                    recipient_id=0,
                    fl_round=rnd,
                    payload={"pors_proof": proof.to_dict()},
                )
                for vid in self.consensus.validators:
                    if vid in self.chains:
                        self.chains[vid].add_transaction(tx)

            # 8. PoA consensus: propose and finalise block
            block = self.consensus.propose_and_finalise(
                self.chains, fl_round=rnd, exclude_ids=list(blocked)
            )

            # 9. Smart contract execution
            if block is not None:
                self.contract.process_block(block.transactions, fl_round=rnd)

            # 10. Log
            self.chain_heights.append(self.chains[0].height)
            self.lambda_stars.append(lambda_star)
            self.token_balances_history.append(
                dict(self.contract.state.token_balances)
            )

            if rnd % self.log_interval == 0 or rnd == 1:
                loss, acc = self.server.evaluate(self.test_loader)
                eps = (
                    self.accountant.get_epsilon(dp_cfg["target_delta"])
                    if dp_cfg["enabled"]
                    else 0.0
                )
                self.tracker.record(rnd, loss, acc, eps, len(selected))

                top_bal = sorted(
                    self.contract.state.token_balances.items(),
                    key=lambda x: x[1], reverse=True
                )[:3]
                print(
                    f"  Rnd {rnd:4d} | acc={acc:.4f} | ε={eps:5.2f} | "
                    f"λ*={lambda_star:.3f} | blocked={len(blocked)} | "
                    f"chain={self.chains[0].height} | "
                    f"top-3 balances={[(i, f'{b:.1f}') for i,b in top_bal]}"
                )

                if dp_cfg["enabled"] and eps > dp_cfg.get("target_epsilon", float("inf")):
                    print(f"\n  Privacy budget exhausted (ε={eps:.2f}). Stopping.")
                    break

        # Save
        results_path = f"{output_dir}/full_results.json"
        self.tracker.save(results_path)
        self._save_extra(output_dir)

        summary = self.tracker.summary()
        summary["game"] = self.game.history_summary()
        summary["avg_blocked"] = (
            float(np.mean(self.blocked_counts)) if self.blocked_counts else 0.0
        )
        summary["final_chain_height"] = self.chains[0].height

        print("\n" + "=" * 70)
        print("Full Simulation Complete")
        print(f"  Convergence round  : {summary['convergence_round']}")
        print(f"  Final accuracy     : {summary['final_accuracy']:.4f}")
        print(f"  Chain height       : {summary['final_chain_height']}")
        print(f"  Avg blocked/round  : {summary['avg_blocked']:.1f}")
        print(f"  Total rewards dist : {self.contract.state.total_rewards_distributed:.1f} tokens")
        print("=" * 70)
        return summary

    def _save_extra(self, output_dir: str):
        extra = {
            "chain_heights": self.chain_heights,
            "lambda_stars": self.lambda_stars,
            "blocked_counts": self.blocked_counts,
            "game_history": self.game._history,
            "final_balances": dict(self.contract.state.token_balances),
            "final_reputations": dict(self.contract.state.reputation_scores),
            "leaderboard": self.contract.leaderboard()[:10],
            "attacker_ids": sorted(self.byzantine_ids | self.freerider_ids),
        }
        with open(f"{output_dir}/blockchain_incentive_results.json", "w") as f:
            json.dump(extra, f, indent=2)
        print(f"Blockchain/incentive results → {output_dir}/blockchain_incentive_results.json")
