# COVER Task 3.2: Secure, Privacy-Preserved and Incentive-Based Data Sharing in UAV Networks

**EU Horizon MSCA Staff Exchanges | Grant 101086228 | Lead: Georg-August-Universität Göttingen**

> A Python simulation framework implementing all four sub-tasks of COVER Task 3.2:
> blockchain-based security, game-theoretic incentives, differential privacy, and
> federated learning convergence in 3D UAV networks.

---

## Headline result, and two privacy-accounting corrections

**The framework learns under a genuine, correctly-accounted per-client
(ε≤10)-DP guarantee**: 23.6% final / 28.2% peak test accuracy on CIFAR-10
over 200 rounds at σ=2.11, spending ε=6.45. Getting to that number required
finding and fixing two independent accounting bugs, in sequence:

1. **Step undercounting.** The RDP accountant was credited with the local
   *epoch* count instead of the actual noisy-minibatch-step count (~78×
   fewer steps than really taken for a typical CIFAR-10 client), so reported
   ε understated the true cost by ~20–30×. (Fixed in
   `federated_learning/client.py` + the three `accumulate()` call sites.)
2. **Pooled accounting.** A *single shared* `RDPAccountant` received every
   client's cost, pooling independent clients into one budget that grows
   with population size, which has nothing to do with any one client's
   actual privacy loss. (Fixed: each client now has its own accountant;
   `max_client_epsilon` reports the worst case across them, not a sum.)

Bug 2 matters beyond bookkeeping: it made σ=6.6 look necessary when a real
per-client ε≤10 guarantee only needs **σ≈2.11**. At the over-conservative
σ=6.6 the model sat at chance (~10%), and an intermediate version of this
report concluded the UAV-scale cohort was simply too small to learn under
real privacy. **That conclusion was wrong**: an artefact of the bug, now
corrected throughout.

⚠️ **Two of four re-validated: opposite outcomes.** AIDER and EuroSAT
(report Sections 8.7, 8.9) have now been re-run at their own
correctly-calibrated per-client σ (1.94 and 2.17 respectively, both
landing in the same narrow band as CIFAR-10's 2.11 despite very different
datasets). **EuroSAT reverses**, same as CIFAR-10: its earlier rise-then-
fade becomes a clean monotonic rise to 36.2% balanced accuracy, the
strongest result of any dataset in this report. **AIDER does not
reverse**: it stays at chance (~20% balanced accuracy, 5 classes) even
with correct accounting, the one dataset where noise miscalibration
wasn't the problem; its severe class imbalance and tiny worst-case client
(61 samples) are now the live suspects. MobileNetV2 and adaptive clipping
(Sections 8.6, 8.8) remain un-re-validated at Bug-2-affected σ values;
their σ-independent findings stand, but anything attributing failure to
"noise at this cohort size" there is still open. Re-validating those two
and diagnosing AIDER's remaining failure are the top future-work items.

See `report/cover_task32_report.pdf` (Section 8, "Correction" box) for the
full writeup and diagnostic numbers.

---

## Project Overview

The [COVER project](https://trimis.ec.europa.eu/project/cooperative-and-intelligent-unmanned-aerial-vehicles-emergency-response-applications)
develops cooperative, intelligent UAV systems for emergency response (flooding, wildfires, earthquakes).

**Task 3.2** designs the data sharing layer that makes multi-UAV collaboration secure, private, and fair:

| Sub-task | Mechanism | Status |
|---|---|---|
| A: Security | Blockchain (PoA) + Byzantine/Sybil defences | Implemented |
| B: Incentives | Stackelberg game + Proof-of-Resource-Sharing | Implemented |
| C: Differential Privacy | DP-SGD (vmap) + RDP accountant | Implemented |
| D: Federated Learning | FedAvg + 3D convergence analysis | Implemented |
| E: Agentic Framework | MARL: DQN agents + federated Q-learning | Implemented |

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
- **3× NVIDIA GTX 1070 Ti** (x86_64, 8GB VRAM each, CUDA 11.8): the setup
  originally targeted; exercises the multi-GPU round-robin path.
- **NVIDIA GB10** (DGX Spark-class, aarch64, Grace Blackwell, ~131GB unified
  memory): what every result currently in `report/cover_task32_report.pdf`
  was actually produced on (PyTorch 2.14+cu130; the `cu118` wheel above
  doesn't target aarch64/Blackwell; install the matching wheel for your
  platform from [pytorch.org](https://pytorch.org/get-started/locally/)
  instead). Single-GPU, so the round-robin path wasn't exercised.

---

## Quick Start

```bash
cd fl_dp_uav

# Full simulation: all 4 sub-tasks, uses all GPUs
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
| `dp.noise_multiplier` (σ) | 2.11 | Privacy–accuracy tradeoff: ↑σ = more private, less accurate. 2.11 is the minimum σ giving a genuine **per-client** ε≤10 guarantee over 200 rounds at this batch size / local-epoch count; see the corrections above. The correct value is setting-specific (depends on client count, batch size, epochs, rounds); re-derive it if you change those |
| `dp.max_grad_norm` (C) | 1.0 | Gradient clipping threshold; bounds sensitivity |
| `dp.target_epsilon` | 10.0 | Stops training if ε budget exceeded |
| `dp.adaptive_clipping` | false | Tracks `max_grad_norm` toward the gradient-norm quantile instead of leaving it fixed (Andrew et al., 2021). Mechanism implemented and verified correct; showed no improvement when tested at the (since-superseded) σ=6.6, and has **not** been re-tested against the now-learning σ=2.11 baseline; see report Section 8.8 |
| `model.backbone` | simplecnn | `simplecnn` (~0.6M params) or `mobilenetv2` (~2.2M params, GroupNorm); `mobilenetv2` failed to learn when tested, including with DP fully disabled (so at least partly architectural rather than noise-driven), but that test predates the σ correction and is queued for re-validation; see report Section 8.6 |
| `data.dirichlet_alpha` | 0.5 | Non-IID degree: ↓α = more heterogeneous UAV data |
| `channel.rician_k` | 10 dB | A2A channel quality: ↑K = stronger LoS |
| `power.min_snr_db` | 5 dB | Minimum SNR for UAV to participate |
| `incentive.cost_mode` | heterogeneous | UAV cost heterogeneity in Stackelberg game |
| `security.aggregation` | fedavg | `median` or `trimmed_mean` for Byzantine robustness |
| `security.byzantine_fraction` | 0.10 | Fraction of UAVs with adversarial behaviour |

---

## Datasets

`runner.py` previously called the CIFAR-10 loader directly regardless of
`data.dataset`, so `eurosat`/`custom` were dead code, fixed this session
(now dispatches via `data.load_dataset()`, with the model's output layer
sized from the dataset instead of hardcoded to 10 classes).

| Dataset | Relevance | Status | Download |
|---|---|---|---|---|
| **CIFAR-10** | Standard FL benchmark | ✅ Experiments run (see report) | Automatic |
| **EuroSAT** | Aerial/satellite land-cover (10 classes, 64×64) | ✅ Re-validated at correctly-calibrated σ=2.17, **reverses**: clean monotonic rise to 36.2% balanced accuracy over 30 rounds, the strongest result of any dataset in this report (report Section 8.9) | Automatic; set `data.dataset: eurosat` |
| **AIDER** | Real UAV disaster imagery, 5 classes (fire, flood, collapsed building, traffic incident, normal) | ⚠️ Stratified partitioning implemented and verified (fixes the zero-class problem, `data.min_per_class`); re-validated at correctly-calibrated σ=1.94 but **does not reverse**: stays at chance (~20% balanced accuracy) even with correct per-client accounting. Root cause narrowed to class imbalance (68% majority class) and a tiny worst-case client (61 samples), not noise miscalibration (report Section 8.7) | [Zenodo](https://zenodo.org/record/3888300), 263MB; set `data.dataset: aider` |
| **FloodNet** | UAV flood disaster imagery | ❌ Not used: Track 1 labels are a ~400-image Google Drive folder, impractical to automate and too small to partition across 20 clients | [GitHub](https://github.com/BinaLab/FloodNet-Challenge-EARTHVISION2021) (manual) |

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
writeups of what worked and what didn't; this README describes the
framework's capabilities and config, not results. It includes:

- Both privacy-accounting corrections (Section 8, "Correction" box) and the
  real CIFAR-10 outcome under a genuinely correct per-client ε≤10 budget:
  **23.6% final / 28.2% peak**, a real learning curve (Section 8.1)
- A PoA consensus deadlock found and fixed (Section 8.3), with before/after
  blockchain behaviour
- An MARL exploration-decay bug found and fixed (Section 8.4)
- Four experiments originally run at the over-conservative σ (Sections
  8.6–8.9); two are now re-validated at their own correct per-client σ,
  with opposite outcomes (Section 8, "Correction" box): **EuroSAT
  reverses**: a clean monotonic rise to **36.2% balanced accuracy**, the
  strongest result of any dataset in this report (Section 8.9); **AIDER
  does not reverse**: it stays at chance even with correct accounting,
  ruling out noise miscalibration and narrowing its real problem to class
  imbalance and a 61-sample worst-case client (Section 8.7; its zero-class
  partitioning problem was separately diagnosed and fixed: 9/20 clients
  missing a class → 0/20). MobileNetV2 and adaptive clipping remain
  outstanding: MobileNetV2 fails to learn *even with DP fully disabled* (a
  real architectural finding, σ-independent, Section 8.6); adaptive
  clipping (Andrew et al., 2021) was implemented with two design mistakes
  caught via the accountant/real training rather than assumed (an initial
  "separate mechanism" design added ~150% ε overhead; the paper's own
  learning rate destabilised training applied per-minibatch instead of
  per-round), but whether it helps on top of the now-learning σ=2.11
  baseline is still untested (Section 8.8)
- A literature comparison against published DP-FL results, built only from
  numbers independently verified against each paper's primary source
  (Section 8.5)
- The bibliography is now a proper `.bib` file (`report/references.bib`)
  instead of a manual list, verifying each entry while converting it
  caught two more unverifiable citations sitting in an older, never-checked
  part of the report (the same class of issue as an earlier fabricated
  citation this session already caught and fixed in Section 8.5) plus a
  couple of wrong venues/years; all fixed

---

## References

Full bibliography: [`report/references.bib`](report/references.bib) (15
entries, each independently verified against its primary source; see the
note above). Key ones:

1. McMahan et al. (2017), "Communication-Efficient Learning of Deep Networks from Decentralized Data." AISTATS.
2. Abadi et al. (2016), "Deep Learning with Differential Privacy." CCS.
3. Mironov (2017), "Rényi Differential Privacy." IEEE CSF.
4. Wang et al. (2019), "Subsampled Rényi DP and Analytical Moments Accountant." AISTATS.
5. Yin et al. (2018), "Byzantine-Robust Distributed Learning." ICML.
6. Fung et al. (2018), "Mitigating Sybils in Federated Learning with FoolsGold." arXiv.
7. Andrew et al. (2021), "Differentially Private Learning with Adaptive Clipping." NeurIPS.

---

## Acknowledgements

This work was carried out at **Northumbria University** (supervised by **Professor Nauman Aslam**) as part of the COVER project (Cooperative and Intelligent UAVs for Emergency Response Applications), funded by the European Union under the Marie Skłodowska-Curie Actions Staff Exchanges programme, Grant Agreement No. 101086228.

Task 3.2 is led by Georg-August-Universität Göttingen (UGOE) with contributions from CentraleSupelec and UESTC. Consortium: TU Eindhoven (coordinator), KTH, CentraleSupelec, Georg-August-Universität Göttingen, Northumbria University, and 12 associated institutions.
