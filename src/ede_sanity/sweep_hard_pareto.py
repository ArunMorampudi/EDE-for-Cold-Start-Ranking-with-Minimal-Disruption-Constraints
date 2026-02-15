"""Hard cold-start Pareto sweep with validation and best-under-budget selection."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import Config
from .experiment import run_experiment, aggregate_metrics


# Sweep parameters - HARD MODE ONLY
LAMBDA_LIST = [0.05, 0.1, 0.2, 0.3, 0.4]

# Fixed baseline settings (must match confirm_operating_points)
BASELINE_ETA = 0.5
BASELINE_M = 2
BASELINE_TOP_L = 50
BASELINE_LAMBDA = 0.0

# Exploration settings (for lambda > 0)
EXPLORATION_ETA = 0.65
EXPLORATION_M = 1
EXPLORATION_TOP_L = 50

SEEDS = list(range(20))

# Validation thresholds (true baseline)
CONFIRMED_BASELINE_NDCG = 0.1809
CONFIRMED_BASELINE_COV = 0.0059
CONFIRMED_BASELINE_AVG_NEW_FRAC = 0.0011
CONFIRMED_BASELINE_UNIQUE_SHOWN = 1.15

TOLERANCE_COV = 0.01
TOLERANCE_AVG_NEW_FRAC = 0.005
TOLERANCE_UNIQUE_SHOWN = 2.0

MIN_COVERAGE_IMPROVEMENT = 0.03
BUDGET_NDCG_DELTA = -0.01


def _root_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def _outputs_dirs(root: Path) -> Tuple[Path, Path]:
    tables_dir = root / "outputs" / "tables"
    figures_dir = root / "outputs" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return tables_dir, figures_dir


def _run_config(seeds: List[int], params: Dict[str, float]) -> Dict[str, float]:
    per_seed: List[Dict[str, float]] = []
    base_config = Config()

    for seed in seeds:
        metrics = run_experiment(
            base_config,
            seed=seed,
            mode="hard",
            policy_params=params,
            smoothed=True,
        )
        per_seed.append(metrics)

    return aggregate_metrics(per_seed)


def _validate_results(df: pd.DataFrame, baseline: Dict[str, float]) -> None:
    errors: List[str] = []

    if abs(baseline["ndcg10_mean"] - CONFIRMED_BASELINE_NDCG) > 0.01:
        errors.append("Baseline NDCG mismatch vs confirmed")
    if abs(baseline["new_cov_mean"] - CONFIRMED_BASELINE_COV) > TOLERANCE_COV:
        errors.append("Baseline coverage mismatch vs confirmed")
    if abs(baseline["avg_new_fraction_mean"] - CONFIRMED_BASELINE_AVG_NEW_FRAC) > TOLERANCE_AVG_NEW_FRAC:
        errors.append("Baseline avg_new_fraction mismatch vs confirmed")
    if abs(baseline["unique_shown_new_mean"] - CONFIRMED_BASELINE_UNIQUE_SHOWN) > TOLERANCE_UNIQUE_SHOWN:
        errors.append("Baseline unique_shown_new mismatch vs confirmed")

    for _, row in df.iterrows():
        for key in ["ndcg10_mean", "new_cov_mean", "avg_new_fraction_mean"]:
            if row[key] < 0.0 or row[key] > 1.0:
                errors.append(f"{row['lambda']}: {key} out of [0,1]")

    if df["new_cov_mean"].max() - baseline["new_cov_mean"] < MIN_COVERAGE_IMPROVEMENT:
        errors.append("No lambda improved coverage by required threshold")

    if errors:
        raise RuntimeError("Validation failed:\n  - " + "\n  - ".join(errors))


def _plot_pareto(df: pd.DataFrame, figures_dir: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.scatter(df["ndcg10_mean"], df["new_cov_mean"], s=80)
    for _, row in df.iterrows():
        plt.text(row["ndcg10_mean"], row["new_cov_mean"], f"λ={row['lambda']:.2f}")
    plt.xlabel("NDCG@10")
    plt.ylabel("Coverage")
    plt.title("Hard Mode: Coverage vs NDCG")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "hard_pareto_cov_vs_ndcg.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(df["ndcg10_mean"], df["unique_shown_new_mean"], s=80)
    for _, row in df.iterrows():
        plt.text(row["ndcg10_mean"], row["unique_shown_new_mean"], f"λ={row['lambda']:.2f}")
    plt.xlabel("NDCG@10")
    plt.ylabel("Unique New Shown")
    plt.title("Hard Mode: Unique New vs NDCG")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "hard_pareto_unique_vs_ndcg.png", dpi=160)
    plt.close()


def _best_under_budget(df: pd.DataFrame, baseline_ndcg: float) -> Tuple[pd.Series, bool]:
    df = df.copy()
    df["ndcg_delta"] = df["ndcg10_mean"] - baseline_ndcg
    df = df[df["ndcg_delta"] >= BUDGET_NDCG_DELTA]
    if df.empty:
        return None, False
    return df.sort_values("new_cov_mean", ascending=False).iloc[0], True


def main() -> None:
    root = _root_dir()
    tables_dir, figures_dir = _outputs_dirs(root)

    print("\nRunning hard-mode Pareto sweep...")

    baseline_params = {
        "lambda": BASELINE_LAMBDA,
        "eta": BASELINE_ETA,
        "m": BASELINE_M,
        "top_L": BASELINE_TOP_L,
    }

    baseline = _run_config(SEEDS, baseline_params)

    rows: List[Dict[str, float]] = []
    for lam in LAMBDA_LIST:
        params = {
            "lambda": lam,
            "eta": EXPLORATION_ETA,
            "m": EXPLORATION_M,
            "top_L": EXPLORATION_TOP_L,
        }
        metrics = _run_config(SEEDS, params)
        row = {
            "lambda": lam,
            "eta": EXPLORATION_ETA,
            "m": EXPLORATION_M,
            "top_L": EXPLORATION_TOP_L,
        }
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    _validate_results(df, baseline)

    csv_path = tables_dir / "hard_pareto_sweep.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved: {csv_path}")

    _plot_pareto(df, figures_dir)
    print(f"✓ Saved: {figures_dir / 'hard_pareto_cov_vs_ndcg.png'}")
    print(f"✓ Saved: {figures_dir / 'hard_pareto_unique_vs_ndcg.png'}")

    best, meets_budget = _best_under_budget(df, baseline["ndcg10_mean"])
    if best is None:
        best = df.sort_values("new_cov_mean", ascending=False).iloc[0]
        print("WARNING: No lambda meets NDCG budget constraint; selecting max coverage instead.")
    report_path = root / "outputs" / "report_hard_pareto.md"
    report_lines = [
        "# Hard Pareto Sweep",
        "",
        f"Best under budget (NDCG delta >= {BUDGET_NDCG_DELTA}):",
        f"- lambda={best['lambda']:.2f}",
        f"- coverage={best['new_cov_mean']:.4f}",
        f"- ndcg={best['ndcg10_mean']:.4f}",
        "",
    ]
    if not meets_budget:
        report_lines.append("WARNING: No lambda met the NDCG budget; selected max coverage instead.")
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"✓ Saved: {report_path}")


if __name__ == "__main__":
    main()
