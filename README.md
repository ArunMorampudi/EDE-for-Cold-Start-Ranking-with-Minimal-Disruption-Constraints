# Entropy-Driven Exploration for Cold-Start Ranking with Minimal-Disruption Constraints

Reproducible experimental pipeline for **Entropy-Driven Exploration (EDE)** — a safe-prefix reranking algorithm for discovering cold-start documents in ranked search while keeping disruption to existing ranking quality minimal.

Repository: `EDE-for-Cold-Start-Ranking-with-Minimal-Disruption-Constraints` (branch: `main`).

---

## Title

**Entropy-Driven Exploration for Cold-Start Ranking with Minimal-Disruption Constraints**

---

## Description

This repository accompanies a PeerJ Computer Science manuscript. It contains the full simulation, evaluation, and validation pipeline for **Entropy-Driven Exploration (EDE)** under cold-start ranking constraints, and produces all tables, figures, and reports used in the paper.

### Problem Context

**Objective**: Ranking systems must surface newly-arrived (cold-start) documents that lack interaction history, but naive exploration harms ranking quality. EDE addresses this by combining entropy-based exploration scores with safe-prefix constraints that protect the highest-quality ranking positions from disruption.

The main challenges this work addresses:
- New documents receive reduced visibility due to missing engagement signals (cold-start penalty), making organic discovery difficult
- Aggressive exploration degrades NDCG while conservative exploration fails to surface new content (discovery–quality trade-off)
- Top-ranked positions must be preserved to maintain user satisfaction (safe-prefix constraint of length *m*)
- The algorithm must perform reliably under different user click behaviors (PBM, Cascade, Noisy)
- All results are averaged over 20 random seeds per configuration, with additional real-data replay on 61,105 Yandex sessions

### How EDE Works

EDE is a **safe-prefix reranking policy** applied at positions *m+1* through *k*:

1. **Candidate gating**: Retains the top-*L* documents by adjusted base score
2. **Safe-prefix protection**: Preserves the top-*m* highest-quality results unchanged
3. **Entropy-driven reranking**: Reranks positions *m+1..k* using a composite exploration score:

$$E'(d) = \eta \cdot N(d) + (1 - \eta) \cdot R(d)$$

where:
- $N(d) = \exp(-I_d / \tau)$ is the **novelty term** (high for unexplored documents)
- $R(d) = (1 - \exp(-I_d / \tau)) \cdot H_{\text{norm}}(d)$ is the **refinement term** (click-entropy signal for explored documents)
- $\eta$ controls the novelty–refinement balance (0.65)
- $\tau$ is the exposure decay constant (50.0)

4. **Final score**: $s = s_0^{\text{norm}} + \lambda \cdot E'(d)$, where $\lambda$ is the exploration weight (0.4)

---

## Dataset Information

### Dataset Description

**Yandex Personalized Web Search Challenge** (2013 Kaggle competition)

- **Source**: Kaggle / Yandex
- **URL**: https://www.kaggle.com/c/yandex-personalized-web-search-challenge
- **Content**: Web search click logs containing session-level queries, ranked document lists (top-10), and binary click labels
- **Usage in this repository**:
  - **Synthetic experiments** (`src/ede_sanity/`): Do **not** require the Yandex dataset. All simulation data is generated programmatically.
  - **Real-data replay** (`src/ede_real_replay/`): Uses downloaded Yandex click logs for a conservative replay sanity check (61,105 sessions after filtering).

### Data Access

The Yandex dataset is obtained from Kaggle and is **not** bundled in this repository. To download it:

1. Set up Kaggle API credentials (see [Kaggle API docs](https://github.com/Kaggle/kaggle-api#api-credentials))
2. Run `python download_yandex_kaggle.py` to download the dataset
3. Run `python extract_logs.py` to decompress the gzip log files into `data/yandex/`

**Note**: The real-data replay is optional. All primary experiments (synthetic simulations) run without external data.

---

## Code Information

### Repository Structure

```
EDE-for-Cold-Start-Ranking-with-Minimal-Disruption-Constraints/
├── README.md                        # This file
├── LICENSE                          # MIT License
├── requirements.txt                 # Python dependencies (pinned versions)
├── download_yandex_kaggle.py        # Downloads Yandex logs from Kaggle
├── extract_logs.py                  # Decompresses gzip Yandex logs to .txt
├── src/
│   ├── ede_sanity/                  # Core synthetic simulation and evaluation suite
│   │   ├── config.py                # Central configuration dataclass (all parameters)
│   │   ├── simulate.py              # SyntheticEnvironment: embeddings, relevance, click models
│   │   ├── ranker.py                # BaseRanker: document scoring with noise
│   │   ├── entropy.py               # Novelty-augmented interaction entropy E'(d)
│   │   ├── policy.py                # EDE safe-prefix rerank + baseline policies
│   │   ├── metrics.py               # NDCG@k, coverage, TTF, CTR computations
│   │   ├── experiment.py            # Main simulation loop (T timesteps per run)
│   │   ├── confirm_operating_points.py   # Validates baseline & calibrated configs
│   │   ├── sweep_hard_pareto.py          # Pareto frontier: coverage vs. NDCG
│   │   ├── difficulty_sweep_extended.py  # Penalty sweep from easy to extreme
│   │   ├── baselines_suite.py            # EDE vs. random injection vs. epsilon-greedy
│   │   ├── click_model_robustness.py     # Tests under PBM, Cascade, Noisy models
│   │   ├── sensitivity_eta_m.py          # Sensitivity to eta and m parameters
│   │   ├── make_baseline_artifacts.py    # Baseline comparison plots and subset tables
│   │   ├── make_paper_artifacts.py       # Difficulty sweep plots for the paper
│   │   └── paper_audit.py               # Internal consistency checks
│   └── ede_real_replay/             # Real-data replay evaluation on Yandex logs
│       ├── __main__.py              # Package entry point
│       ├── load_yandex.py           # Parses Yandex log files into session objects
│       ├── replay_eval.py           # Replay criteria: Exact-Match, Set-Match, Click-Preservation
│       ├── run_replay_check.py      # Applies EDE to real sessions, computes acceptance rates
│       └── README.md                # Real-replay module documentation
└── outputs/                         # Pre-generated results (reports, tables, figures)
    ├── *.md                         # Markdown reports for each experiment
    ├── tables/*.csv                 # CSV result tables
    ├── figures/*.png                # Figures (PNG)
    └── real_replay/                 # Replay evaluation outputs
        ├── replay_summary.csv
        ├── replay_diagnostics.csv
        └── report_real_replay.md
```

### Key Modules and Components

**`src/ede_sanity/` — Synthetic Simulation Suite**

| Module | Purpose |
|--------|---------|
| `config.py` | Central configuration dataclass with all simulation parameters |
| `simulate.py` | `SyntheticEnvironment`: generates document embeddings, relevance labels, and simulates user clicks (PBM, Cascade, Noisy click models) |
| `ranker.py` | `BaseRanker`: scores documents with Gaussian noise to model imperfect ranking |
| `entropy.py` | `compute_entropy_scores()`: computes the novelty-augmented interaction entropy $E'(d)$ |
| `policy.py` | Core algorithm: `safe_prefix_rerank()` (EDE), plus `random_injection_rerank()` and `epsilon_greedy_rerank()` baselines |
| `metrics.py` | Computes NDCG@k, new-document coverage, time-to-first-click (TTF), CTR by position |
| `experiment.py` | `run_experiment()`: runs T timesteps of query–rank–click simulation per seed |

**`src/ede_real_replay/` — Real-Data Replay Evaluation**

| Module | Purpose |
|--------|---------|
| `load_yandex.py` | Parses Kaggle Yandex logs into `YandexSession` objects (session_id, query_id, ranked_docs, clicks) |
| `replay_eval.py` | Conservative replay criteria: Replay-Exact-Match (strictest), Replay-Set-Match, Replay-Click-Preservation (most permissive) |
| `run_replay_check.py` | Applies EDE reranking to real sessions; measures acceptance rates and CTR/NDCG deltas |

---

## Usage Instructions

### Prerequisites
- **Python 3.11+**
- **pip** (Python package manager)
- **Kaggle API credentials** (only for the optional real-data replay)

### Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ArunMorampudi/EDE-for-Cold-Start-Ranking-with-Minimal-Disruption-Constraints.git
   cd EDE-for-Cold-Start-Ranking-with-Minimal-Disruption-Constraints
   ```

2. **Create a Python virtual environment** (recommended):
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**:
   - **Windows (PowerShell)**:
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - **Windows (Command Prompt)**:
     ```cmd
     .venv\Scripts\activate.bat
     ```
   - **macOS / Linux**:
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Pipeline

The pipeline is organized into three sequential steps. Steps 1 and 2 use only synthetic data (no downloads required). Step 3 is optional and uses real Yandex logs.

#### Step 1: Run synthetic experiments

Run all six experiment scripts from the repository root:

```bash
python -m src.ede_sanity.confirm_operating_points
python -m src.ede_sanity.sweep_hard_pareto
python -m src.ede_sanity.difficulty_sweep_extended
python -m src.ede_sanity.baselines_suite
python -m src.ede_sanity.click_model_robustness
python -m src.ede_sanity.sensitivity_eta_m
```

**What each experiment does**:

| Command | Purpose |
|---------|---------|
| `confirm_operating_points` | Validates baseline (λ=0) and calibrated EDE configurations in normal and hard cold-start modes; writes confirmed operating-point summaries |
| `sweep_hard_pareto` | Maps the Pareto frontier of coverage vs. NDCG in hard mode by sweeping λ from 0.05 to 0.4; selects best configuration under an NDCG budget |
| `difficulty_sweep_extended` | Sweeps cold-start penalty from 0.10 to 0.80 (easy to extreme); validates that an easy regime (5%–30% baseline coverage) exists |
| `baselines_suite` | Compares EDE (smoothed and unsmoothed) against random-injection and epsilon-greedy baselines across multiple penalty settings |
| `click_model_robustness` | Tests all strategies under PBM, Cascade, and Noisy click models; validates stability of the Exploration Transition Function |
| `sensitivity_eta_m` | Measures sensitivity to novelty weight (η ∈ {0.35, 0.5, 0.65}) and safe-prefix length (m ∈ {1, 2, 3}) |

#### Step 2: Generate paper artifacts

These scripts depend on CSV outputs from Step 1:

```bash
python -m src.ede_sanity.make_baseline_artifacts
python -m src.ede_sanity.make_paper_artifacts
```

This generates subset tables and figures in `outputs/tables/` and `outputs/figures/`.

#### Step 3 (Optional): Real-data replay sanity check

Requires Kaggle API credentials. Set up credentials, then:

```bash
python download_yandex_kaggle.py
python extract_logs.py
python -m src.ede_real_replay.run_replay_check --data_dir data/yandex --max_sessions 200000
```

This does the following:
1. Downloads Yandex click logs from Kaggle
2. Decompresses log files to `data/yandex/`
3. Applies EDE reranking to real search sessions and evaluates against three conservative replay criteria:
   - **Replay-Exact-Match**: EDE output matches logged ranking exactly (~77% acceptance)
   - **Replay-Set-Match**: Same documents in top-10, any order (~77% acceptance)
   - **Replay-Click-Preservation**: All clicked documents preserved in top-10 (~99% acceptance)

Replay validation also monitors mean position change (target ≈ 0.66) and ensures zero mean rank shift for clicked items.

### Output Locations

| Output Type | Location |
|-------------|----------|
| Experiment reports | `outputs/*.md` |
| Result tables (CSV) | `outputs/tables/*.csv` |
| Figures (PNG) | `outputs/figures/*.png` |
| Real-replay report | `outputs/real_replay/report_real_replay.md` |
| Real-replay data | `outputs/real_replay/*.csv` |

### Understanding the Outputs

Primary metrics:
- **NDCG@10** — Ranking quality (binary relevance labels based on simulated or real clicks)
- **New-document coverage** — Fraction of new relevant documents that receive at least one click
- **Mean Time-to-First-Click (TTF)** — Average timestep when new relevant documents are first clicked
- **Unique new docs shown** — Count of distinct new documents surfaced in the top-10

Safety metrics:
- **Mean position change** — Average rank displacement introduced by EDE (target ≈ 0.66)
- **Click-Preservation acceptance rate** — Percentage of sessions where all clicked documents remain in top-10 (target ≈ 97%+)

Key reports:
- `report_paper_ready_difficulty.md` — Shows EDE boosting discovery from ~0.05 to ~0.58 coverage while keeping the NDCG drop to ~0.005
- `report_baselines_suite.md` — EDE-Smoothed achieves +0.62 mean coverage gain vs. +0.0003 for random injection and +0.037 for epsilon-greedy
- `report_hard_pareto.md` — Pareto frontier of coverage vs. NDCG at different exploration strengths
- `report_real_replay.md` — Replay-Click-Preservation acceptance of ~99% with near-zero CTR/NDCG degradation on real Yandex sessions

---

## Requirements

### System Environment (Tested)

- **OS**: Windows 11 (also compatible with macOS, Linux)
- **Python**: 3.11+
- **RAM**: 8 GB sufficient for synthetic experiments; 16 GB recommended for real-data replay with large session counts

### Python Dependencies

All required packages are listed in `requirements.txt` with pinned versions. Key dependencies include:

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | 2.4.2 | Numerical computation (scoring, entropy, simulation) |
| pandas | 3.0.0 | Data manipulation and result aggregation |
| matplotlib | 3.10.8 | Figure generation |
| scipy | 1.17.0 | Statistical functions (NDCG computation) |
| kaggle | 2.0.0 | Yandex dataset download (optional, for real-data replay only) |
| pytest | 9.0.2 | Testing |

### Installation

Install all pinned dependencies:

```bash
pip install -r requirements.txt
```

---

## Methodology

The evaluation pipeline combines controlled synthetic simulation with a conservative real-data replay check:

1. **Synthetic temporal ranking environment**: `SyntheticEnvironment` generates 1,000 documents (800 established + 200 new, arriving at timestep T₀ = 1,000) with configurable relevance scores and cold-start penalties. User clicks are simulated under PBM, Cascade, or Noisy click models over T = 2,000 timesteps across 100 queries.

2. **Policy comparison**: Each experiment pairs a baseline run (λ = 0, no exploration) with EDE and alternative strategies (random injection, epsilon-greedy) on the same random seeds. EDE reranks positions m+1 through k using combined novelty and entropy-based refinement signals.

3. **Metric evaluation**: For each configuration and seed, the pipeline records `new_cov` (discovery), `ndcg10` (quality), `ttf_mean` (speed of discovery), and `unique_shown_new` (breadth). Deltas (EDE minus baseline) are averaged over 20 seeds per configuration.

4. **Robustness checks**: Four additional experiments stress-test the algorithm:
   - Hard-mode Pareto sweep — maps the coverage–NDCG trade-off frontier by varying λ
   - Extended difficulty range — sweeps penalty from 0.10 (easy) to 0.80 (extreme)
   - Click-model robustness — verifies consistent behavior under PBM, Cascade, and Noisy models
   - η/m sensitivity — checks performance across novelty weight and safe-prefix length settings

5. **Artifact generation**: Post-processing scripts produce CSV tables, Markdown reports, and PNG figures from experiment outputs.

6. **Real-data replay validation**: EDE is applied to 61,105 real Yandex search sessions. Three replay criteria (Exact-Match, Set-Match, Click-Preservation) measure the fraction of sessions where EDE's reranking is acceptable, with bootstrap 95% confidence intervals.

### Default Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| λ (lambda) | 0.4 | Exploration weight |
| η (eta) | 0.65 | Novelty weight in E'(d) |
| m | 2 | Safe-prefix length (protected top positions) |
| top_L | 50 | Candidate gating threshold |
| α (alpha) | 0.5 | Laplace smoothing for entropy |
| τ (tau) | 50.0 | Exposure decay time constant |
| T | 2,000 | Total simulation timesteps |
| T₀ | 1,000 | New-document arrival timestep |
| k | 10 | Ranking cutoff (top-k shown to users) |
| Seeds | 20 | Random seeds per configuration |

---

## Citations

### Dataset Reference

- **Yandex** (2013). *Personalized Web Search Challenge* [Dataset]. Kaggle.
  - URL: https://www.kaggle.com/c/yandex-personalized-web-search-challenge

If this codebase or methodology is used in research, please cite the corresponding PeerJ Computer Science manuscript and include the dataset citation above.

---

## License & Contribution Guidelines

This project is released under the **MIT License**. See the [LICENSE](LICENSE) file for details.

Contributions via pull requests are welcome. Please make sure changes pass `pytest` and do not break reproducibility of the experimental outputs.
