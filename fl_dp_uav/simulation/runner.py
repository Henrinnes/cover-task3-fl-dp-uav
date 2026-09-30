"""
Main simulation orchestrator for COVER Task 3.2.

Multi-GPU strategy
------------------
UAV clients are assigned to GPUs round-robin at construction time so each
GPU hosts roughly num_uavs/num_gpus clients.  Per-round local training is
parallelised with ThreadPoolExecutor: every selected client runs on its own
GPU concurrently.  Thread-safety is achieved by:
  - weights distributed to all clients *before* threads are launched
  - RDP accountant updates collected *after* all threads join (via a lock)
  - each client owns its own model copy (deep-copied in FLClient.__init__)

Server (aggregation + evaluation) runs on GPU 0 (the primary device).
"""

import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from data.partition import dirichlet_partition, load_dataset
from differential_privacy.accountant import RDPAccountant, max_client_epsilon
from federated_learning.client import FLClient
from federated_learning.convergence import ConvergenceTracker
from federated_learning.server import FLServer
from models.cnn import SimpleCNN
from models.mobilenet import MobileNetV2CIFAR
from uav_network.channel import UAVChannel
from uav_network.mobility import RandomWaypointMobility
from uav_network.topology import UAVNetwork


def _gpu_devices() -> List[str]:
    """Return list of available CUDA device strings, or ['cpu']."""
    n = torch.cuda.device_count()
    if n == 0:
        return ["cpu"]
    return [f"cuda:{i}" for i in range(n)]


def _num_classes(dataset) -> int:
    """Read class count off the underlying torchvision/ImageFolder dataset,
    unwrapping the Subset that load_eurosat()/load_aider() apply for their
    manual train/test split."""
    ds = dataset.dataset if isinstance(dataset, Subset) else dataset
    return len(ds.classes)


class SimulationRunner:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        seed = self.cfg["simulation"]["seed"]
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        self.gpu_devices = _gpu_devices()
        self.primary_device = self.gpu_devices[0]

        self._build()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _build(self):
        cfg = self.cfg
        net_cfg = cfg["uav_network"]
        mob_cfg = cfg["mobility"]
        ch_cfg  = cfg["channel"]
        fl_cfg  = cfg["fl"]
        dp_cfg  = cfg["dp"]
        data_cfg = cfg["data"]

        self.network = UAVNetwork(
            num_uavs=net_cfg["num_uavs"],
            area_size=tuple(net_cfg["area_size"]),
            altitude_range=(net_cfg["min_altitude"], net_cfg["max_altitude"]),
            rng=self.rng,
        )
        self.mobility = RandomWaypointMobility(
            area_size=tuple(net_cfg["area_size"]),
            altitude_range=(net_cfg["min_altitude"], net_cfg["max_altitude"]),
            max_speed=mob_cfg["max_speed"],
            pause_time=mob_cfg["pause_time"],
            rng=self.rng,
        )
        self.channel = UAVChannel(
            frequency_hz=float(ch_cfg["frequency"]),
            path_loss_exp=float(ch_cfg["path_loss_exponent"]),
            rician_k_db=float(ch_cfg["rician_k"]),
            noise_power_dbm=float(ch_cfg["noise_power_dbm"]),
            bandwidth_hz=float(ch_cfg["bandwidth_hz"]),
            rng=self.rng,
        )

        train_set, test_set = load_dataset(data_cfg["dataset"], data_cfg["data_dir"])
        self.num_classes = num_classes = _num_classes(train_set)
        # DataLoader for evaluation: pin_memory only useful when device is CUDA
        self.test_loader = DataLoader(
            test_set, batch_size=256, shuffle=False,
            num_workers=4, pin_memory=(self.primary_device != "cpu"),
        )
        client_datasets = dirichlet_partition(
            train_set,
            num_clients=self.network.num_uavs,
            alpha=data_cfg["dirichlet_alpha"],
            rng=self.rng,
            min_per_class=data_cfg.get("min_per_class", 0),
        )

        backbone = cfg.get("model", {}).get("backbone", "simplecnn")
        if backbone == "mobilenetv2":
            global_model = MobileNetV2CIFAR(num_classes=num_classes)
        elif backbone == "simplecnn":
            global_model = SimpleCNN(num_classes=num_classes)
        else:
            raise ValueError(f"Unknown model.backbone: {backbone!r}")
        # Server lives on the primary (GPU 0)
        self.server = FLServer(global_model, device=self.primary_device)

        # Distribute clients round-robin across all GPUs
        self.clients: List[FLClient] = [
            FLClient(
                client_id=i,
                model=copy.deepcopy(global_model),
                dataset=client_datasets[i],
                lr=fl_cfg["learning_rate"],
                local_epochs=fl_cfg["local_epochs"],
                batch_size=fl_cfg["local_batch_size"],
                dp_enabled=dp_cfg["enabled"],
                max_grad_norm=dp_cfg["max_grad_norm"],
                noise_multiplier=dp_cfg["noise_multiplier"],
                device=self.gpu_devices[i % len(self.gpu_devices)],
                adaptive_clipping=dp_cfg.get("adaptive_clipping", False),
                target_quantile=dp_cfg.get("target_quantile", 0.5),
                clip_lr=dp_cfg.get("clip_lr", 0.2),
                quantile_noise_multiplier=dp_cfg.get("quantile_noise_multiplier", 3.0),
                clip_min=dp_cfg.get("clip_min", 0.1),
                clip_max=dp_cfg.get("clip_max", 10.0),
            )
            for i in range(self.network.num_uavs)
        ]

        # One accountant PER CLIENT, not one shared across all of them: a
        # client's privacy loss depends only on what's revealed about their
        # own data, not on how many other clients independently also have
        # their own (eps, delta)-DP guarantee. Pooling all clients into one
        # running RDP total (the previous design) makes "eps" grow with
        # population size for reasons that have nothing to do with any
        # individual client's actual privacy cost -- confirmed directly: a
        # 100-UAV run's round-1 eps (pooled) was ~4.4x a 20-UAV run's,
        # tracking the participant-count ratio, not anything about the
        # noise each client actually adds to their own gradients.
        self.client_accountants: Dict[int, RDPAccountant] = {
            i: RDPAccountant() for i in range(self.network.num_uavs)
        }
        self.tracker = ConvergenceTracker(target_accuracy=fl_cfg.get("target_accuracy", 0.65))

        self.min_snr_db   = cfg["power"]["min_snr_db"]
        self.frac         = fl_cfg["fraction_participate"]
        self.num_rounds   = fl_cfg["num_rounds"]
        self.log_interval = cfg["simulation"]["log_interval"]

    # ------------------------------------------------------------------
    # Per-round helpers
    # ------------------------------------------------------------------

    def _select_participants(self) -> List[int]:
        """Channel-aware participant selection. UAV 0 = parameter server."""
        ps = self.network.nodes[0]
        eligible = [
            node.node_id
            for node in self.network.nodes[1:]
            if self.channel.can_communicate(node, ps, self.min_snr_db)
        ]
        k = max(1, round(len(eligible) * self.frac))
        if len(eligible) <= k:
            return eligible
        return self.rng.choice(eligible, size=k, replace=False).tolist()

    def _parallel_train(
        self, selected: List[int], dp_cfg: dict, local_epochs_override: Dict[int, int] = None
    ) -> List[Tuple[Dict, int]]:
        """
        Train all selected clients in parallel (one thread per client).
        Each client runs on its pre-assigned GPU independently.
        Returns list of (state_dict, num_samples) ready for aggregation.
        """
        fl_epochs = self.cfg["fl"]["local_epochs"]

        def _train_one(cid: int) -> Tuple[int, Dict, int, float]:
            if local_epochs_override:
                self.clients[cid].local_epochs = local_epochs_override.get(cid, fl_epochs)
            w, _ = self.clients[cid].train()
            # Restore default epochs (matters if override was applied)
            self.clients[cid].local_epochs = fl_epochs

            if dp_cfg["enabled"]:
                # Each client's own accountant -- no cross-client shared
                # state, so no lock needed (each thread only ever touches
                # its own client's entry). Adaptive clipping (if enabled)
                # rides this same per-step subsampled query -- Theorem 1
                # guarantees the joint release costs the same as this
                # nominal call, so no separate accounting is needed (see
                # FLClient).
                self.client_accountants[cid].accumulate(
                    noise_multiplier=dp_cfg["noise_multiplier"],
                    sample_rate=self.clients[cid].sample_rate,
                    num_steps=self.clients[cid].last_num_steps,
                )
            return cid, w, self.clients[cid].num_samples

        # Distribute weights before spawning threads (avoids deepcopy race)
        global_w = self.server.get_weights()
        for cid in selected:
            self.clients[cid].set_weights(global_w)

        updates = []
        with ThreadPoolExecutor(max_workers=len(selected)) as pool:
            futures = {pool.submit(_train_one, cid): cid for cid in selected}
            for future in as_completed(futures):
                _, w, n_samples = future.result()
                updates.append((w, n_samples))

        return updates

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, output_dir: str = "./results") -> Dict:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        dp_cfg = self.cfg["dp"]

        gpu_summary = ", ".join(
            f"GPU{i}:{torch.cuda.get_device_name(i).split()[-1]}"
            for i in range(torch.cuda.device_count())
        ) if torch.cuda.is_available() else "CPU"

        print("=" * 65)
        print("COVER Task 3.2 — FL + DP Simulation")
        print(f"  UAVs          : {self.network.num_uavs}")
        print(f"  FL rounds     : {self.num_rounds}")
        print(f"  GPUs in use   : {gpu_summary}")
        print(f"  DP            : σ={dp_cfg['noise_multiplier']}, C={dp_cfg['max_grad_norm']}")
        print(f"  Target ε      : {dp_cfg.get('target_epsilon', '—')}")
        # Show which client IDs land on which GPU
        for gidx, gdev in enumerate(self.gpu_devices):
            cids = [i for i in range(self.network.num_uavs) if i % len(self.gpu_devices) == gidx]
            print(f"  {gdev:<10} : clients {cids}")
        print("=" * 65)

        for rnd in range(1, self.num_rounds + 1):
            self.mobility.step(self.network.nodes, dt=self.cfg["mobility"]["step_dt"])

            selected = self._select_participants()
            if not selected:
                continue

            updates = self._parallel_train(selected, dp_cfg)

            self.server.aggregate(updates)

            if rnd % self.log_interval == 0 or rnd == 1:
                loss, acc = self.server.evaluate(self.test_loader)
                eps = (
                    max_client_epsilon(self.client_accountants, dp_cfg["target_delta"])
                    if dp_cfg["enabled"] else 0.0
                )
                self.tracker.record(rnd, loss, acc, eps, len(selected))
                print(
                    f"  Round {rnd:4d} | loss={loss:.4f} | acc={acc:.4f} | "
                    f"ε={eps:6.2f} | participants={len(selected)}"
                )
                if dp_cfg["enabled"] and eps > dp_cfg.get("target_epsilon", float("inf")):
                    print(f"\n  Privacy budget exhausted (ε={eps:.2f}). Stopping.")
                    break

        results_path = f"{output_dir}/convergence_results.json"
        self.tracker.save(results_path)
        summary = self.tracker.summary()
        print("\n" + "=" * 65)
        print(f"  Convergence round : {summary['convergence_round']}")
        print(f"  Final accuracy    : {summary['final_accuracy']:.4f}")
        print(f"  Final ε           : {summary['final_epsilon']:.2f}")
        print("=" * 65)
        return summary
