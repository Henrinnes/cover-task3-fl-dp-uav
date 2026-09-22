"""
MARL-FL Trainer: orchestrates DQN agents across FL rounds.

Loop per round:
  1. Each UAV agent observes state → selects action (skip / 1ep / 3ep / 5ep).
  2. MARLSimulationRunner._fl_round() trains participating UAVs with the
     chosen epochs, runs security filtering, aggregates, executes blockchain,
     and returns (accuracy, contributions, DP-ε).
  3. Environment computes per-UAV shaped rewards from Δacc + contribution.
  4. All agents store (s, a, r, s') and run one DQN gradient update.
  5. Every `fed_q_interval` rounds: Q-net weights are FedAvg'd across
     participating agents (federated Q-learning).

Results written to results/marl/marl_results.json.
"""
import json
import os
import time
from typing import Dict, List

import numpy as np
import torch

from .agent import DQNAgent, ACTION_EPOCHS
from .environment import MARLEnvironment, STATE_DIM


class MARLFLTrainer:
    def __init__(self, runner, fed_q_interval: int = 10):
        """
        Parameters
        ----------
        runner           : MARLSimulationRunner instance
        fed_q_interval   : rounds between Q-network federation steps
        """
        self.runner         = runner
        self.fed_q_interval = fed_q_interval

        cfg        = runner.cfg
        num_uavs   = cfg["uav_network"]["num_uavs"]
        num_rounds = cfg["fl"]["num_rounds"]

        self.env = MARLEnvironment(num_uavs, num_rounds)

        # eps decays once per round per agent, but only once the replay buffer
        # holds a full batch (`update()` is a no-op before that) — so both the
        # warm-up length and the decay rate must scale with the configured
        # round budget, or short runs never decay at all (buffer never fills)
        # and long runs never reach eps_end (a fixed decay=0.99 leaves
        # eps~=0.19 after a 200-round run instead of reaching 0.05).
        eps_start, eps_end = 1.0, 0.05
        batch_size = min(32, max(8, num_rounds // 4))
        decay_steps = max(1, num_rounds - batch_size)
        eps_decay = (eps_end / eps_start) ** (1.0 / decay_steps)

        self.agents: List[DQNAgent] = [
            DQNAgent(
                agent_id=i,
                state_dim=STATE_DIM,
                n_actions=len(ACTION_EPOCHS),
                eps_start=eps_start,
                eps_end=eps_end,
                eps_decay=eps_decay,
                batch_size=batch_size,
                device=runner.gpu_devices[i % len(runner.gpu_devices)],
            )
            for i in range(num_uavs)
        ]

        self._results: Dict = {
            "rounds": [],
            "accuracy": [],
            "epsilon_greedy": [],
            "mean_reward": [],
            "participation_rate": [],
            "mean_q_loss": [],
            "privacy_epsilon": [],
        }

    # ------------------------------------------------------------------ #

    def train(self) -> Dict:
        cfg        = self.runner.cfg
        num_rounds = cfg["fl"]["num_rounds"]
        dp_cfg     = cfg["dp"]
        min_snr    = float(cfg.get("power", {}).get("min_snr_db", 5.0))
        num_uavs   = len(self.agents)

        print("\n" + "=" * 70)
        print("COVER Task 3.2 — MARL + Federated Learning (All Layers)")
        print(f"  UAVs          : {num_uavs}")
        print(f"  Rounds        : {num_rounds}")
        print(f"  Actions       : skip / 1ep / 3ep / 5ep")
        print(f"  Q-fed interval: every {self.fed_q_interval} rounds")
        print(f"  State dim     : {STATE_DIM}")
        print("=" * 70 + "\n")

        self.env.reset()
        global_acc = 0.0

        for rnd in range(num_rounds):
            t0 = time.time()

            # 1. Observe state
            snr_db         = self.runner._get_snr_dict()
            data_qualities = self._get_data_qualities()
            states         = self.env.get_states(snr_db, data_qualities, rnd, global_acc)

            # 2. Select actions
            actions       = [self.agents[i].select_action(states[i]) for i in range(num_uavs)]
            epochs_chosen = list(ACTION_EPOCHS)  # indexed copy, we'll override per-agent
            epochs_chosen = [ACTION_EPOCHS[a] for a in actions]

            # Participants: chose > 0 epochs AND meet min SNR
            participants = [
                i for i, ep in enumerate(epochs_chosen)
                if ep > 0 and snr_db.get(i, 0.0) >= min_snr
            ]
            # Guarantee at least 2 for valid aggregation
            if len(participants) < 2:
                eligible = sorted(
                    [i for i in range(num_uavs) if snr_db.get(i, 0.0) >= min_snr],
                    key=lambda i: snr_db.get(i, 0.0),
                    reverse=True,
                )[:2]
                for i in eligible:
                    if i not in participants:
                        participants.append(i)
                        epochs_chosen[i] = 1
                        actions[i] = 1

            participation_rate = len(participants) / num_uavs

            # 3. FL round (training + security + aggregation + blockchain)
            global_acc, contributions, priv_eps = self.runner._fl_round(
                participants=participants,
                epochs_chosen=epochs_chosen,
                dp_cfg=dp_cfg,
                round_num=rnd + 1,
            )

            # 4. Rewards + next states
            rewards     = self.env.compute_rewards(actions, global_acc, contributions)
            next_states = self.env.get_states(snr_db, data_qualities, rnd + 1, global_acc)
            done        = float(rnd == num_rounds - 1)

            # 5. Experience push + Q-network update for every agent
            q_losses = []
            for i in range(num_uavs):
                self.agents[i].push(
                    states[i], actions[i], float(rewards[i]), next_states[i], done
                )
                q_losses.append(self.agents[i].update())

            # 6. Federate Q-nets every K rounds
            if (rnd + 1) % self.fed_q_interval == 0:
                self._federate_q_nets(participants)

            # Logging
            elapsed  = time.time() - t0
            mean_r   = float(np.mean(rewards))
            active   = [l for l in q_losses if l > 0]
            mean_qls = float(np.mean(active)) if active else 0.0
            eps_g    = self.agents[0].eps

            self._results["rounds"].append(rnd + 1)
            self._results["accuracy"].append(global_acc)
            self._results["epsilon_greedy"].append(eps_g)
            self._results["mean_reward"].append(mean_r)
            self._results["participation_rate"].append(participation_rate)
            self._results["mean_q_loss"].append(mean_qls)
            self._results["privacy_epsilon"].append(priv_eps)

            print(
                f"  Rnd {rnd+1:3d}/{num_rounds} | acc={global_acc:.4f} | "
                f"reward={mean_r:+.4f} | part={len(participants):2d}/{num_uavs} | "
                f"ε-greedy={eps_g:.3f} | DP-ε={priv_eps:.2f} | {elapsed:.1f}s"
            )

            # Privacy budget guard
            if dp_cfg["enabled"] and priv_eps > dp_cfg.get("target_epsilon", float("inf")):
                print(f"\n  Privacy budget exhausted (DP-ε={priv_eps:.2f}). Stopping.")
                break

        self._save_results()
        return self._results

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _get_data_qualities(self) -> np.ndarray:
        """Normalised label-distribution entropy per client (data quality proxy)."""
        qualities = np.zeros(len(self.agents), dtype=np.float32)
        for i, client in enumerate(self.runner.clients):
            n      = min(200, len(client.dataset))
            labels = [client.dataset[j][1] for j in range(n)]
            counts = np.bincount(labels, minlength=10).astype(float)
            counts /= counts.sum() + 1e-8
            entropy       = -np.sum(counts * np.log(counts + 1e-8)) / np.log(10)
            qualities[i]  = float(np.clip(entropy, 0.0, 1.0))
        return qualities

    def _federate_q_nets(self, participants: List[int]):
        """FedAvg over Q-network weights of the round's participants."""
        if not participants:
            return
        weights = [self.agents[i].get_q_weights() for i in participants]
        avg_w: Dict = {}
        for key in weights[0]:
            avg_w[key] = torch.stack([w[key].cpu().float() for w in weights]).mean(0)
        for agent in self.agents:
            agent.set_q_weights(avg_w)

    def _save_results(self):
        os.makedirs("results/marl", exist_ok=True)
        path = "results/marl/marl_results.json"
        with open(path, "w") as f:
            json.dump(self._results, f, indent=2)
        print(f"\nMARL results saved → {path}")
