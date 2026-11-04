# COVER Task 3.2: Secure, Privacy-Preserved and Incentive-Based Data Sharing in UAV Networks

**EU Horizon MSCA Staff Exchanges | Grant 101086228 | Lead: Georg-August-Universität Göttingen**

> A Python simulation framework implementing all four sub-tasks of COVER Task 3.2:
> blockchain-based security, game-theoretic incentives, differential privacy, and
> federated learning convergence in 3D UAV networks.

---

## Headline result: uniform noise fails, per-client adaptive noise rescues it

**A single global noise level, calibrated to the smallest client in a
heterogeneous UAV fleet, is the wrong way to add differential privacy to
small-cohort federated learning.** At a genuinely valid per-client
(ε≤10)-DP guarantee, standard uniform-σ DP-SGD on CIFAR-10 barely learns:
**17.3% ± 3.9** mean final accuracy across 5 seeds, 200 rounds, often
indistinguishable from the 10% chance level. A new per-client adaptive
noise calibration mechanism, holding an **identical** formal guarantee,
raises this to **30.7% ± 6.6**, a +13.4 percentage-point improvement,
confirmed on EuroSAT too (+16.3pp across 3 seeds, winning all 3). Getting
to a baseline worth measuring that improvement against required finding
and fixing three layered accounting bugs, in sequence:

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
3. **Wrong worst-case client.** `SimulationRunner.__init__` builds the UAV
   network, mobility model, and channel, consuming random draws, *before*
   partitioning the dataset. The headline σ=2.11 figure in earlier versions
   of this README/report was derived by a script that partitioned the
   dataset directly, skipping that RNG consumption, so it computed the
   worst-case client against a different, more lenient partition than the
   real simulation produces. The true worst-case CIFAR-10 client (seed 42)
   has 1125 samples; at σ=2.11 that client's real ε over 200 rounds is
   ≈16.0, not ≤10 as claimed. **The old headline number (23.6% final /
   28.2% peak) was never a genuine per-client ε≤10 guarantee.** The
   corrected minimum is σ≈3.08 (seed-dependent, see `calibrate_sigma` in
   `differential_privacy/accountant.py`).

Bugs 1 and 2 each produced a confident but wrong intermediate conclusion
(first "the cohort is too small to learn", then "σ=2.11 is a valid
guarantee") that later turned out to be an artefact, not a finding. We
keep this history explicit rather than silently publishing the final
numbers, because the failure mode, a plausible-looking but wrong
accountant producing a result that reads as a clean finding, is easy to
repeat, and on this project's own evidence, easy to repeat more than once.

**Per-client adaptive noise calibration** (`dp.adaptive_sigma`, off by
default) is the direct response to Bug 3: since each client already has
its own independent accountant, there's no reason every client must use
the *same* σ, calibrated to the population's single smallest client. A
client with more samples has a lower DP-SGD sampling rate and can use
proportionally less noise while clearing the identical ε≤10 target. It
does **not** rescue every dataset: tested on AIDER, it makes no
difference (~20% balanced accuracy either way, chance level for 5
classes), which rules out noise magnitude as AIDER's bottleneck and
narrows its real problem to class imbalance and its very small worst-case
client (61 samples).

See `report/cover_task32_report.pdf` (Section 8, "Correction" box, and
Section 8.2, "Per-Client Adaptive Noise Calibration") for the full
writeup, all 5+3 seed results, and diagnostic numbers.

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
| `fl.class_weighted_loss` | false | Each client reweights its own local cross-entropy by inverse class frequency (computed only from its own partition, no cross-client info, no privacy cost). On AIDER: flips the adaptive-vs-uniform mean gap from -0.2pp (wash) to +1.4pp (5 seeds, all positive), a real but modest fix, not a rescue; see report Section 8.7 |
| `dp.noise_multiplier` (σ) | 3.08 | Privacy–accuracy tradeoff: ↑σ = more private, less accurate. 3.08 is the minimum σ giving the true worst-case client a genuine **per-client** ε≤10 guarantee over 200 rounds at this batch size / local-epoch count, for seed 42's partition; see the corrections above (Bug 3). The correct value is seed- and setting-specific (depends on the RNG seed, client count, batch size, epochs, rounds); re-derive it with `calibrate_sigma` if you change any of those |
| `dp.adaptive_sigma` | false | Per-client noise calibration instead of one shared σ: each client gets the smallest σ its own sample count allows, at an identical formal guarantee. Confirmed to substantially improve accuracy on CIFAR-10 (+13.4pp mean, 5 seeds) and EuroSAT (+16.3pp mean, 3 seeds); does **not** help on AIDER; see report Section 8.2 |
| `dp.max_grad_norm` (C) | 1.0 | Gradient clipping threshold; bounds sensitivity |
| `dp.target_epsilon` | 10.0 | Stops training if ε budget exceeded |
| `dp.adaptive_clipping` | false | Tracks `max_grad_norm` toward the gradient-norm quantile instead of leaving it fixed (Andrew et al., 2021). Mechanism implemented and verified correct, but re-validated against the corrected σ=3.08 uniform baseline and against adaptive σ, and **collapses training to flat chance accuracy in both cases**. Instrumented directly: clip-norm dynamics are structurally σ-independent (not a hyperparameter-mismatch problem, the old "needs re-tuning" explanation here was wrong and has been corrected). Four remediation attempts (sync, slower update rate, hard-capped ceiling, combinations) all tested, all converge to the same non-learning state; see report Section 8.8 for the full diagnosis |
| `model.backbone` | simplecnn | `simplecnn` (~0.6M params) or `mobilenetv2` (~2.2M params, GroupNorm); `mobilenetv2` fails to learn under DP-SGD regardless of σ (checked against the true worst-case client, still fails), including with DP fully disabled, so at least partly architectural rather than noise-driven; see report Section 8.6 |
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
| **EuroSAT** | Aerial/satellite land-cover (10 classes, 64×64) | ✅ Strongest result in the report: uniform σ=2.17 gives 38.2%±8.3 mean final balanced accuracy (3 seeds); **adaptive per-client σ raises this to 54.5%±4.8** (+16.3pp mean, wins all 3 seeds) (report Sections 8.9, 8.2) | Automatic; set `data.dataset: eurosat` |
| **AIDER** | Real UAV disaster imagery, 5 classes (fire, flood, collapsed building, traffic incident, normal) | ⚠️ Stratified partitioning implemented and verified (fixes the zero-class problem, `data.min_per_class`); stays at chance (~20% balanced accuracy) under both uniform **and** adaptive per-client σ, ruling out noise calibration as the cause. Class-weighted local loss (`fl.class_weighted_loss`) gives a real but modest fix: mean gap flips from -0.2pp (wash) to +1.4pp (5 seeds, all positive), not a rescue. Root cause narrowed to class imbalance (68% majority class) and a tiny worst-case client (61 samples) (report Sections 8.7, 8.2) | [Zenodo](https://zenodo.org/record/3888300), 263MB; set `data.dataset: aider` |
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

- Three privacy-accounting corrections (Section 8, "Correction" box),
  the third (wrong worst-case client) invalidating the CIFAR-10 headline
  number an earlier version of this report and README quoted; the
  corrected, genuinely valid result is **17.3%±3.9 mean final accuracy
  (uniform σ, 5 seeds)**, raised to **30.7%±6.6 (adaptive per-client σ,
  5 seeds)**, a +13.4pp improvement at an identical formal guarantee
  (Sections 8.1, 8.2)
- A PoA consensus deadlock found and fixed (Section 8.3), with before/after
  blockchain behaviour
- An MARL exploration-decay bug found and fixed (Section 8.4)
- **Per-client adaptive noise calibration** (Section 8.2, new): replaces
  one shared σ calibrated to the population's smallest client with a
  per-client σ, each client using the least noise its own sample count
  allows at an identical formal guarantee. Confirmed on CIFAR-10 (+13.4pp
  mean, 5 seeds, wins 4 of 5) and EuroSAT (+16.3pp mean, 3 seeds, wins all
  3); does **not** help on AIDER, which rules out noise magnitude as that
  dataset's problem and narrows it to class imbalance and a 61-sample
  worst-case client (Section 8.7)
- MobileNetV2 and adaptive clipping: MobileNetV2 fails to learn *even
  with DP fully disabled* (a real architectural finding, σ-independent,
  checked against the true worst-case client and still holds, Section
  8.6); adaptive clipping (Andrew et al., 2021) was implemented with two
  design mistakes caught via the accountant/real training rather than
  assumed (an initial "separate mechanism" design added ~150% ε overhead;
  the paper's own learning rate destabilised training applied
  per-minibatch instead of per-round), then re-validated against both
  the corrected σ=3.08 uniform baseline and adaptive σ, and **collapses
  training to flat chance accuracy in both cases**. Instrumented
  directly (clip-norm trajectories are near-identical at σ=6.6 and
  σ=3.08, so the flaw predates and is independent of the σ correction,
  it was just invisible without a real signal to damage). Four
  remediation attempts, cross-client sync, a 10× slower update rate, a
  hard-capped ceiling, and combinations, were each tested to full
  200-round completion and all converge to the same non-learning state
  via gradient explosion or degenerate single-class convergence; the
  mechanism's per-minibatch granularity itself, not any tunable
  hyperparameter, looks to be the cause (Section 8.8)
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
