"""Extended difficulty sweep including easy regime where baseline discovery is non-trivial."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import Config
from .experiment import run_experiment, aggregate_metrics


PENALTY_VALUES = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.6, 0.8]
SEEDS = list(range(20))

BASELINE_CONFIG = {
    "config_name": "Baseline",
    "lambda": 0.0,
    "eta": 0.5,
    "m": 2,
    "top_L": 50,
}

EDE_CONFIG = {
    "config_name": "EDE",
    "lambda": 0.4,
    "eta": 0.65,
    "m": 2,
    "top_L": 50,
}

TARGET_COV_MIN = 0.05
TARGET_COV_MAX = 0.30


def _root_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def _outputs_dirs(root: Path) -> Tuple[Path, Path]:
    tables_dir = root / "outputs" / "tables"
    reports_dir = root / "outputs"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    return tables_dir, reports_dir


def _run_config_at_penalty(penalty: float, config_dict: Dict, seeds: List[int]) -> Dict[str, float]:
    per_seed: List[Dict[str, float]] = []
    base_config = Config()

    params = dict(config_dict)
    params["penalty"] = penalty

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


def _check_calibration(df: pd.DataFrame) -> None:
    baseline_df = df[df["config_name"] == "Baseline"].sort_values("penalty")
    found_easy = False

    for _, row in baseline_df.iterrows():
        cov = row["new_cov_mean"]
        if TARGET_COV_MIN <= cov <= TARGET_COV_MAX:
            found_easy = True
            break

    if not found_easy:
        raise RuntimeError(
            "Calibration failed: no penalty achieved baseline coverage in 5%-30% range."
        )


def _validate_ranges(df: pd.DataFrame) -> None:
    for col in ["new_cov_mean", "ndcg10_mean"]:
        if df[col].min() < 0.0 or df[col].max() > 1.0:
            raise RuntimeError(f"{col} out of [0,1] bounds.")


def _write_report(df: pd.DataFrame, report_path: Path) -> None:
    baseline_df = df[df["config_name"] == "Baseline"].sort_values("penalty")

    easy_penalties = []
    for _, row in baseline_df.iterrows():
        if TARGET_COV_MIN <= row["new_cov_mean"] <= TARGET_COV_MAX:
            easy_penalties.append(row["penalty"])

    lines = [
        "# Extended Difficulty Sweep: Easy to Extreme",
        "",
        "## Overview",
        "This extended sweep tests EDE across a broader difficulty spectrum, from easier penalties",
        "where baseline has non-trivial discovery to extreme penalties where even EDE struggles.",
        "",
        "## Sweep Configuration",
        f"- Penalty range: {PENALTY_VALUES}",
        f"- Seeds: {len(SEEDS)}",
        "- Baseline: λ=0.0, η=0.5, m=2, top_L=50",
        "- EDE: λ=0.4, η=0.65, m=2, top_L=50",
        "",
        "## Calibration",
        f"Easy regime found at penalties: {easy_penalties}",
        "",
    ]

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    print("\n" + "=" * 80)
    print("EXTENDED DIFFICULTY SWEEP")
    print("=" * 80)

    root = _root_dir()
    tables_dir, reports_dir = _outputs_dirs(root)

    rows: List[Dict] = []
    for penalty in PENALTY_VALUES:
        print(f"\nPenalty = {penalty:.2f}")

        aggregated_baseline = _run_config_at_penalty(penalty, BASELINE_CONFIG, SEEDS)
        row_baseline = {
            "penalty": penalty,
            "config_name": BASELINE_CONFIG["config_name"],
            "lambda": BASELINE_CONFIG["lambda"],
            "eta": BASELINE_CONFIG["eta"],
            "m": BASELINE_CONFIG["m"],
            "top_L": BASELINE_CONFIG["top_L"],
        }
        row_baseline.update(aggregated_baseline)
        rows.append(row_baseline)

        aggregated_ede = _run_config_at_penalty(penalty, EDE_CONFIG, SEEDS)
        row_ede = {
            "penalty": penalty,
            "config_name": EDE_CONFIG["config_name"],
            "lambda": EDE_CONFIG["lambda"],
            "eta": EDE_CONFIG["eta"],
            "m": EDE_CONFIG["m"],
            "top_L": EDE_CONFIG["top_L"],
        }
        row_ede.update(aggregated_ede)
        rows.append(row_ede)

    df = pd.DataFrame(rows)

    _validate_ranges(df)
    _check_calibration(df)

    csv_path = tables_dir / "difficulty_sweep_extended.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved: {csv_path}")

    report_path = reports_dir / "report_difficulty_sweep_extended.md"
    _write_report(df, report_path)
    print(f"✓ Saved: {report_path}")

    print("\nExtended difficulty sweep completed successfully.")


if __name__ == "__main__":
    main()
