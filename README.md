# COVER Task 3.2 — Secure, Privacy-Preserved and Incentive-Based Data Sharing in UAV Networks

**EU Horizon MSCA Staff Exchanges | Grant 101086228 | Lead: Georg-August-Universität Göttingen**

> A Python simulation framework implementing all four sub-tasks of COVER Task 3.2:
> blockchain-based security, game-theoretic incentives, differential privacy, and
> federated learning convergence in 3D UAV networks.

---

## Project Overview

The [COVER project](https://trimis.ec.europa.eu/project/cooperative-and-intelligent-unmanned-aerial-vehicles-emergency-response-applications)
develops cooperative, intelligent UAV systems for emergency response (flooding, wildfires, earthquakes).

**Task 3.2** designs the data sharing layer that makes multi-UAV collaboration secure, private, and fair:

| Sub-task | Mechanism | Status |
|---|---|---|
| A — Security | Blockchain (PoA) + Byzantine/Sybil defences | Implemented |
| B — Incentives | Stackelberg game + Proof-of-Resource-Sharing | Implemented |
| C — Differential Privacy | DP-SGD + RDP accountant | Implemented |
| D — Federated Learning | FedAvg + 3D convergence analysis | Implemented |

---

## Architecture

```
fl_dp_uav/
├── uav_network/          UAV topology, 3D random-waypoint mobility, Rician A2A channel
├── federated_learning/   FedAvg server, DP-SGD client, convergence tracker
├── differential_privacy/ Gradient clipping/noise (Abadi et al. 2016), RDP accountant
├── incentive/            Stackelberg game (optimal λ*), Proof-of-Resource-Sharing
├── blockchain/           Block/Transaction, hash-linked chain, PoA consensus, smart contract
├── security/             Sybil/Byzantine/free-rider attacks + FoolsGold/median defences
├── models/               SimpleCNN (32×32 RGB, proxy for aerial image classification)
├── data/                 CIFAR-10 & EuroSAT loaders, Dirichlet non-IID partitioning
├── simulation/           SimulationRunner (FL+DP) and FullSimulationRunner (all layers)
└── evaluation/           Convergence plots, DP vs no-DP comparison, privacy–accuracy Pareto
```

### Multi-GPU parallelism

UAV clients are assigned round-robin to all available GPUs. Each FL round
spawns `ThreadPoolExecutor` workers so every client trains in parallel on its
assigned GPU. Aggregation and evaluation run on GPU 0.

---

## Requirements

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r fl_dp_uav/requirements.txt
```

Python 3.9+, PyTorch ≥ 2.0, CUDA 11.8+ recommended (3× GTX 1070 Ti tested).

---

## Quick Start

```bash
cd fl_dp_uav

# Full simulation — all 4 sub-tasks, uses all GPUs
python main.py --full

# FL + DP only (faster; good for privacy-accuracy sweep)
python main.py

# No-DP baseline
python main.py --no-dp

# Head-to-head DP vs no-DP comparison (saves overlay PDF plot)
python main.py --compare

# Privacy–accuracy Pareto sweep (varies noise_multiplier)
python main.py --sweep

# Plot from saved results
python main.py --plot-only
```

Results and plots are saved to `./results/`.

---

## Configuration (`config.yaml`)

Key parameters and their research significance:

| Parameter | Default | Effect |
|---|---|---|
| `fl.num_rounds` | 200 | Communication rounds budget |
| `dp.noise_multiplier` (σ) | 1.1 | Privacy–accuracy tradeoff: ↑σ = more private, less accurate |
| `dp.max_grad_norm` (C) | 1.0 | Gradient clipping threshold; bounds sensitivity |
| `dp.target_epsilon` | 10.0 | Stops training if ε budget exceeded |
| `data.dirichlet_alpha` | 0.5 | Non-IID degree: ↓α = more heterogeneous UAV data |
| `channel.rician_k` | 10 dB | A2A channel quality: ↑K = stronger LoS |
| `power.min_snr_db` | 5 dB | Minimum SNR for UAV to participate |
| `incentive.cost_mode` | heterogeneous | UAV cost heterogeneity in Stackelberg game |
| `security.aggregation` | fedavg | `median` or `trimmed_mean` for Byzantine robustness |
| `security.byzantine_fraction` | 0.10 | Fraction of UAVs with adversarial behaviour |

---

## Datasets

| Dataset | Relevance | Download |
|---|---|---|
| **CIFAR-10** | Standard FL benchmark | Automatic |
| **EuroSAT** | Aerial/satellite land-cover (10 classes, 64×64) | Automatic — change `data.dataset: eurosat` |
| **FloodNet** | UAV flood disaster imagery (directly on-task) | [GitHub](https://github.com/BinaLab/FloodNet-Challenge-EARTHVISION2021) |
| **AIDER** | Aerial fire/flood/collapse classification | [GitHub](https://github.com/ckyrkou/AIDER) |

For FloodNet/AIDER: place under `data/<name>/train/` and `test/`, then set `data.dataset: custom`.

---

## Key Technical Contributions

### 1. Federated Learning with DP-SGD
FedAvg over a wireless UAV network where only UAVs with sufficient channel SNR
participate. Local training uses DP-SGD (Abadi et al., 2016) with per-sample
gradient clipping and Gaussian noise injection.

### 2. Tight Privacy Accounting (RDP)
Privacy expenditure is tracked via the Rényi Differential Privacy (RDP) accountant
(Mironov, 2017; Wang et al., 2019), giving the tightest known (ε, δ)-DP guarantee
for the subsampled Gaussian mechanism with composition over multiple rounds.

### 3. Stackelberg Game Incentive Mechanism
The parameter server (leader) sets the optimal reward rate
λ* = (−1 + √(1 + 2αH)) / (2H)  where  H = Σ(1/cᵢ)
UAVs (followers) respond with effort eᵢ* = λ*/cᵢ, scaled as additional local
training epochs, producing a Nash equilibrium that maximises social welfare.

### 4. Proof-of-Resource-Sharing (PoRS)
Each participating UAV commits a cryptographic proof of (i) the gradient update
hash (SHA-256), (ii) a Merkle root over training batches, (iii) measured compute
time, and (iv) bytes transmitted. The smart contract verifies PoRS before
distributing token rewards.

### 5. Proof-of-Authority Blockchain
A lightweight PoA consensus (round-robin proposer, quorum sign-off) avoids the
energy cost of PoW, suited to battery-constrained UAVs. The on-chain smart
contract maintains token balances and reputation scores, detecting and penalising
free-riders automatically.

### 6. Multi-Layer Security
- **Sybil**: FoolsGold cosine-similarity detection + stake requirement
- **Byzantine**: Z-score norm outlier + cosine deviation from mean
- **Free-rider**: Gradient norm threshold + inter-round staleness check
- **Robust aggregation**: coordinate-wise median or trimmed mean

---

## References

1. McMahan et al. (2017), "Communication-Efficient Learning of Deep Networks from Decentralized Data." AISTATS.
2. Abadi et al. (2016), "Deep Learning with Differential Privacy." CCS.
3. Mironov (2017), "Rényi Differential Privacy." IEEE CSF.
4. Wang et al. (2019), "Subsampled Rényi DP and Analytical Moments Accountant." NeurIPS.
5. Yang et al. (2019), "Incentive Mechanism Design for Federated Learning." IEEE Trans. Wireless Commun.
6. Blanchard et al. (2017), "Machine Learning with Adversaries: Byzantine Tolerant SGD." NeurIPS.
7. Yin et al. (2018), "Byzantine-Robust Distributed Learning." ICML.
8. Fung et al. (2018), "Mitigating Sybils in Federated Learning with FoolsGold." arXiv.

---

## Acknowledgements

This work is part of the COVER project (Cooperative and Intelligent UAVs for Emergency Response Applications), funded by the European Union under the Marie Skłodowska-Curie Actions Staff Exchanges programme, Grant Agreement No. 101086228.

Partners: TU Eindhoven (coordinator), KTH, CentraleSupelec, Georg-August-Universität Göttingen, and 12 associated institutions.
