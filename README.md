# COVER Task 3.2 — Secure, Privacy-Preserved and Incentive-Based Data Sharing in UAV Networks

**EU Horizon MSCA Staff Exchanges | Grant 101086228 | Lead: Georg-August-Universität Göttingen**

> A Python simulation framework implementing all four sub-tasks of COVER Task 3.2:
> blockchain-based security, game-theoretic incentives, differential privacy, and
> federated learning convergence in 3D UAV networks.

---

## ⚠️ Privacy-accounting correction

An earlier version of this codebase **undercounted the true DP-SGD step count**
fed to the RDP accountant (it was credited with the local *epoch* count, not
the actual number of noisy minibatch steps — roughly 78× fewer than really
taken for a typical CIFAR-10 client). This meant the reported ε values
understated the true privacy cost by roughly 20–30×. It's fixed now
(`federated_learning/client.py` + the three `accumulate()` call sites in
`simulation/`), and `config.yaml`'s default `noise_multiplier` is raised from
1.1 to 6.6 — the minimum value that keeps a 200-round run within the
ε≤10 budget once accounted for correctly. At that noise level, **the model
does not learn** on CIFAR-10 (9.98% accuracy — chance level for 10 classes).
See `report/cover_task32_report.pdf` (Section 8, "Correction" box) for the
full writeup, including the diagnostic numbers that led to the fix.

---

## Project Overview

The [COVER project](https://trimis.ec.europa.eu/project/cooperative-and-intelligent-unmanned-aerial-vehicles-emergency-response-applications)
develops cooperative, intelligent UAV systems for emergency response (flooding, wildfires, earthquakes).

**Task 3.2** designs the data sharing layer that makes multi-UAV collaboration secure, private, and fair:

| Sub-task | Mechanism | Status |
|---|---|---|
| A — Security | Blockchain (PoA) + Byzantine/Sybil defences | Implemented |
| B — Incentives | Stackelberg game + Proof-of-Resource-Sharing | Implemented |
| C — Differential Privacy | DP-SGD (vmap) + RDP accountant | Implemented |
| D — Federated Learning | FedAvg + 3D convergence analysis | Implemented |
| E — Agentic Framework | MARL: DQN agents + federated Q-learning | Implemented |

---

## Architecture

```
fl_dp_uav/
├── uav_network/          UAV topology, 3D random-waypoint mobility, Rician A2A channel
├── federated_learning/   FedAvg server, DP-SGD client (vmap), convergence tracker
├── differential_privacy/ Gradient clipping/noise (Abadi et al. 2016), RDP accountant
├── incentive/            Stackelberg game (optimal λ*), Proof-of-Resource-Sharing
├── blockchain/           Block/Transaction, hash-linked chain, PoA consensus, smart contract
├── security/             Sybil/Byzantine/free-rider attacks + FoolsGold/median defences
├── marl/                 DQN agents, federated Q-learning, MARL environment + trainer
├── models/               SimpleCNN (GroupNorm, 32×32 RGB, proxy for aerial classification)
├── data/                 CIFAR-10, EuroSAT & AIDER loaders, Dirichlet non-IID partitioning
├── simulation/           SimulationRunner, FullSimulationRunner, MARLSimulationRunner
└── evaluation/           Convergence plots, DP vs no-DP, privacy–accuracy Pareto, MARL plots
```

### Multi-GPU parallelism

UAV clients are assigned round-robin to all available GPUs. Each FL round
spawns `ThreadPoolExecutor` workers so every client trains in parallel on its
assigned GPU. Aggregation and evaluation run on GPU 0.

---

## Requirements

```bash
# x86_64 + CUDA 11.8, e.g. the original 3x GTX 1070 Ti setup:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r fl_dp_uav/requirements.txt
```

Python 3.9+, PyTorch ≥ 2.0. Two setups have actually been tested:
- **3× NVIDIA GTX 1070 Ti** (x86_64, 8GB VRAM each, CUDA 11.8) — the setup
  originally targeted; exercises the multi-GPU round-robin path.
- **NVIDIA GB10** (DGX Spark-class, aarch64, Grace Blackwell, ~131GB unified
  memory) — what every result currently in `report/cover_task32_report.pdf`
  was actually produced on (PyTorch 2.14+cu130; the `cu118` wheel above
  doesn't target aarch64/Blackwell — install the matching wheel for your
  platform from [pytorch.org](https://pytorch.org/get-started/locally/)
  instead). Single-GPU, so the round-robin path wasn't exercised.

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
| `dp.noise_multiplier` (σ) | 6.6 | Privacy–accuracy tradeoff: ↑σ = more private, less accurate. Raised from 1.1 — see the privacy-accounting correction above; at this UAV-scale cohort (11 active/round) and step count, σ=6.6 is the minimum needed to keep a 200-round run within ε≤10 once accounted for correctly, and at that noise level accuracy does not rise above chance |
| `dp.max_grad_norm` (C) | 1.0 | Gradient clipping threshold; bounds sensitivity |
| `dp.target_epsilon` | 10.0 | Stops training if ε budget exceeded |
| `model.backbone` | simplecnn | `simplecnn` (~0.6M params) or `mobilenetv2` (~2.2M params, GroupNorm); the latter does not currently converge under DP-SGD at these hyperparameters — see report Section 8.6 |
| `data.dirichlet_alpha` | 0.5 | Non-IID degree: ↓α = more heterogeneous UAV data |
| `channel.rician_k` | 10 dB | A2A channel quality: ↑K = stronger LoS |
| `power.min_snr_db` | 5 dB | Minimum SNR for UAV to participate |
| `incentive.cost_mode` | heterogeneous | UAV cost heterogeneity in Stackelberg game |
| `security.aggregation` | fedavg | `median` or `trimmed_mean` for Byzantine robustness |
| `security.byzantine_fraction` | 0.10 | Fraction of UAVs with adversarial behaviour |

---

## Datasets

`runner.py` previously called the CIFAR-10 loader directly regardless of
`data.dataset`, so `eurosat`/`custom` were dead code — fixed this session
(now dispatches via `data.load_dataset()`, with the model's output layer
sized from the dataset instead of hardcoded to 10 classes).

| Dataset | Relevance | Status | Download |
|---|---|---|---|---|
| **CIFAR-10** | Standard FL benchmark | ✅ Experiments run (see report) | Automatic |
| **EuroSAT** | Aerial/satellite land-cover (10 classes, 64×64) | ⚠️ Loader fixed and usable; no experiment run yet | Automatic — set `data.dataset: eurosat` |
| **AIDER** | Real UAV disaster imagery, 5 classes (fire, flood, collapsed building, traffic incident, normal) | ⚠️ Stratified partitioning implemented and verified (fixes the zero-class problem, `data.min_per_class`), but **still does not converge** at a correctly-accounted ε≤10 budget — same root cause as CIFAR-10/MobileNetV2: noise needed for real privacy at this cohort size overwhelms the signal (report Section 8.7) | [Zenodo](https://zenodo.org/record/3888300), 263MB — set `data.dataset: aider` |
| **FloodNet** | UAV flood disaster imagery | ❌ Not used — Track 1 labels are a ~400-image Google Drive folder, impractical to automate and too small to partition across 20 clients | [GitHub](https://github.com/BinaLab/FloodNet-Challenge-EARTHVISION2021) (manual) |

For AIDER: download `AIDER.zip` from the Zenodo link above and extract into
`data/aider/`, producing `data/aider/AIDER/<class>/*.jpg`.

For any other ImageFolder-style dataset (including FloodNet, if you do your
own train/test split): place under `data/<name>/train/` and `test/`, then set
`data.dataset: custom`.

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

## Full Report & Results

[`report/cover_task32_report.pdf`](report/cover_task32_report.pdf) is the
source of truth for actual experimental numbers, figures, and honest
writeups of what worked and what didn't — this README describes the
framework's capabilities and config, not results. It includes:

- The privacy-accounting correction (Section 8, "Correction" box) and the
  real CIFAR-10 outcome under a genuinely correct ε≤10 budget (9.98%
  accuracy — chance level)
- A PoA consensus deadlock found and fixed (Section 8.3), with before/after
  blockchain behaviour
- An MARL exploration-decay bug found and fixed (Section 8.4)
- Three independent lines of evidence for the same root cause — MobileNetV2
  doesn't converge under DP-SGD at these hyperparameters (Section 8.6);
  AIDER's data-scarcity/class-imbalance problem was fixed with stratified
  partitioning (verified: 9/20 clients with zero examples of some class →
  0/20), but still doesn't converge at a correctly-accounted ε≤10 budget
  (Section 8.7). All three point to the same binding constraint: this
  framework's small UAV cohort (11–20 clients) needs adaptive clipping or
  larger effective participation before a genuine privacy budget and real
  learning can coexist — now the lead future-work item
- A literature comparison against published DP-FL results, built only from
  numbers independently verified against each paper's primary source
  (Section 8.5)

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

This work was carried out at **Northumbria University** (supervised by **Professor Nauman Aslam**) as part of the COVER project (Cooperative and Intelligent UAVs for Emergency Response Applications), funded by the European Union under the Marie Skłodowska-Curie Actions Staff Exchanges programme, Grant Agreement No. 101086228.

Task 3.2 is led by Georg-August-Universität Göttingen (UGOE) with contributions from CentraleSupelec and UESTC. Consortium: TU Eindhoven (coordinator), KTH, CentraleSupelec, Georg-August-Universität Göttingen, Northumbria University, and 12 associated institutions.
