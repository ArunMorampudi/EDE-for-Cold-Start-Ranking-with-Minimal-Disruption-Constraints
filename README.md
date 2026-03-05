# Reproducibility

## Title
Entropy-Driven Exploration for Cold-Start Ranking with Minimal-Disruption Constraints.

## Description
This repository contains code for a reproducible synthetic and real-log evaluation pipeline for Entropy-Driven Exploration (EDE) under cold-start ranking constraints. It supports controlled simulation experiments, baseline/ablation comparisons, sensitivity analysis, hard-mode Pareto analysis, and a conservative replay sanity check on Yandex click logs.

## Dataset Information
- Dataset: Yandex. 2013. Personalized Web Search Challenge. Kaggle. https://www.kaggle.com/c/yandex-personalized-web-search-challenge
- Usage in this repo:
	- Synthetic experiments in `src/ede_sanity/` do not require Yandex data.
	- Real-data replay sanity check in `src/ede_real_replay/` uses downloaded Yandex logs from Kaggle.

## Code Information
- Core simulation/evaluation package: `src/ede_sanity/`
	- `confirm_operating_points.py`
	- `sweep_hard_pareto.py`
	- `difficulty_sweep_extended.py`
	- `baselines_suite.py`
	- `click_model_robustness.py`
	- `sensitivity_eta_m.py`
	- `make_baseline_artifacts.py`
	- `make_paper_artifacts.py`
- Real-log replay package: `src/ede_real_replay/`
	- `run_replay_check.py`
	- `load_yandex.py`
	- `replay_eval.py`
- Utility scripts:
	- `download_yandex_kaggle.py`
	- `extract_logs.py`

## Usage Instructions

### 1) Environment setup
Run from repository root:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Run synthetic experiments (Step 1)
Run in this order:

```bash
python -m src.ede_sanity.confirm_operating_points
python -m src.ede_sanity.sweep_hard_pareto
python -m src.ede_sanity.difficulty_sweep_extended
python -m src.ede_sanity.baselines_suite
python -m src.ede_sanity.click_model_robustness
python -m src.ede_sanity.sensitivity_eta_m
```

### 2a) Experiment purposes
- `confirm_operating_points`: Verifies baseline and calibrated exploration configurations in both normal and hard cold-start modes; produces confirmed operating-point summaries.
- `sweep_hard_pareto`: Maps the trade-off between discovery and ranking quality in hard mode across exploration strengths.
- `difficulty_sweep_extended`: Sweeps easy-to-extreme difficulty settings and checks that an easy regime (baseline coverage 5%-30%) exists.
- `baselines_suite`: Compares EDE variants against random-injection and epsilon-greedy baselines under multiple penalty settings.
- `click_model_robustness`: Tests strategy behavior under PBM, cascade, and noisy click models, and validates stability of the Exploration Transition Function from novelty-dominant $N_d$ to refinement-weighted $R_d$.
- `sensitivity_eta_m`: Measures sensitivity to novelty weight (`eta`) and safe-prefix length (`m`).

### 3) Generate dependent paper artifacts (Step 2)

```bash
python -m src.ede_sanity.make_baseline_artifacts
python -m src.ede_sanity.make_paper_artifacts
```

### 4) Optional: run real-data replay sanity check (Step 3)
Set Kaggle credentials, then:

```bash
python download_yandex_kaggle.py
python extract_logs.py
python -m src.ede_real_replay.run_replay_check --data_dir data/yandex --max_sessions 200000
```

Replay validation includes monitoring mean position change (target ≈ 0.66) and ensuring zero mean rank shift for clicked items.

### 5) Output locations
- Reports: `outputs/*.md` and `outputs/real_replay/report_real_replay.md`
- Tables: `outputs/tables/*.csv` and `outputs/real_replay/*.csv`
- Figures: `outputs/figures/*.png`

### 6) Output details (what each artifact means)

* Primary Quality Metric: NDCG@10 (Ranking Quality).
* Discovery Metrics: New-document coverage, Mean Time-to-First-Click (TTF), and Unique new docs shown.
* Safety Metrics: Mean position change (target ≈ 0.66) and Click-Preservation acceptance rate (target ≈ 97%).
* Narrative Proof: `report_paper_ready_difficulty.md` provides the core narrative demonstrating EDE's ability to boost discovery (from ~0.05 to ~0.58 coverage) while respecting a small NDCG budget (~0.005 drop).

## Requirements

### Python
- Python 3.11+

### Core libraries
- numpy
- pandas
- matplotlib
- scipy

### Optional/data download tooling
- kaggle (for dataset download helper script)

Install all pinned dependencies using `requirements.txt`.

## Methodology
1. Build a synthetic temporal ranking environment with controlled cold-start penalties and click models.
2. Compare baseline policy (`lambda=0`) versus EDE exploration settings.
3. Evaluate key metrics across seeds: `new_cov`, `ndcg10`, `ttf_mean`, `unique_shown_new`, and derived deltas.
4. Stress-test behavior across:
	 - hard-mode Pareto sweep,
	 - extended difficulty range,
	 - click-model robustness,
	 - eta/m sensitivity.
5. Generate paper-ready summaries and visual artifacts from experiment CSV outputs.
6. Validate replay behavior on real Yandex logs using conservative acceptance criteria (Exact-Match, Set-Match, Click-Preservation).

## Citations
- Yandex. 2013. Personalized Web Search Challenge. Kaggle. https://www.kaggle.com/c/yandex-personalized-web-search-challenge

If this codebase is used in a manuscript, cite the corresponding paper and include the dataset citation above.

## License & Contribution Guidelines
- License: MIT (`LICENSE`).
