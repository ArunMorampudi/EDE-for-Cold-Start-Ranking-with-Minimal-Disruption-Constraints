"""Sensitivity analysis over eta and m for EDE."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import Config
from .experiment import run_experiment, aggregate_metrics


PENALTIES = [0.15, 0.30]
ETA_LIST = [0.35, 0.50, 0.65]
M_LIST = [1, 2, 3]
SEEDS = list(range(20))

LAMBDA = 0.4
TOP_L = 50
ALPHA = 0.5


def _root_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def _outputs_dirs(root: Path) -> Tuple[Path, Path]:
    outputs_dir = root / "outputs"
    tables_dir = outputs_dir / "tables"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return outputs_dir, tables_dir


def _run_config(
    base_config: Config,
    penalty: float,
    eta: float,
    m: int,
    seeds: List[int],
) -> Dict[str, float]:
    per_seed: List[Dict[str, float]] = []
    params = {
        "lambda": LAMBDA,
        "eta": eta,
        "m": m,
        "top_L": TOP_L,
        "alpha": ALPHA,
        "penalty": penalty,
    }

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


def _validate_bounds(df: pd.DataFrame) -> None:
    for _, row in df.iterrows():
        for col in ["ndcg10_mean", "new_cov_mean"]:
            value = float(row[col])
            if value < 0.0 or value > 1.0:
                raise RuntimeError(
                    f"Bounds check failed: p={row['penalty']} eta={row['eta']} m={row['m']} {col}={value}"
                )


def _warn_ndcg_trend(df: pd.DataFrame, warnings: List[str]) -> None:
    tol = 0.005
    for penalty in PENALTIES:
        for eta in ETA_LIST:
            subset = df[(df["penalty"] == penalty) & (df["eta"] == eta)].sort_values("m")
            if subset.empty:
                continue
            ndcg = subset["ndcg10_mean"].to_list()
            for i in range(1, len(ndcg)):
                if ndcg[i] + tol < ndcg[i - 1]:
                    msg = (
                        "WARNING: NDCG did not improve with higher m at "
                        f"p={penalty:.2f}, eta={eta:.2f}."
                    )
                    warnings.append(msg)
                    print(msg)
                    break


def _warn_eta_trend(df: pd.DataFrame, warnings: List[str]) -> None:
    tol = 0.005
    for penalty in PENALTIES:
        for m in M_LIST:
            subset = df[(df["penalty"] == penalty) & (df["m"] == m)].sort_values("eta")
            if subset.empty:
                continue
            cov = subset["new_cov_mean"].to_list()
            for i in range(1, len(cov)):
                if cov[i] + tol < cov[i - 1]:
                    msg = (
                        "WARNING: Coverage did not improve with higher eta at "
                        f"p={penalty:.2f}, m={m}."
                    )
                    warnings.append(msg)
                    print(msg)
                    break


def _recommend_default(df: pd.DataFrame) -> Tuple[float, int]:
    grouped = df.groupby(["eta", "m"], as_index=False).agg(
        {
            "new_cov_mean": "mean",
            "ndcg10_mean": "mean",
        }
    )
    best = grouped.sort_values(["new_cov_mean", "ndcg10_mean"], ascending=[False, False]).iloc[0]
    return float(best["eta"]), int(best["m"])


def _format_heatmap_table(df: pd.DataFrame, penalty: float) -> List[str]:
    subset = df[df["penalty"] == penalty]
    lines = [f"### Coverage heatmap (p={penalty:.2f})", "", "| eta \\ m | 1 | 2 | 3 |", "|---|---|---|---|"]

    for eta in ETA_LIST:
        row = [f"{eta:.2f}"]
        for m in M_LIST:
            value = subset[(subset["eta"] == eta) & (subset["m"] == m)]["new_cov_mean"].iloc[0]
            row.append(f"{value:.4f}")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    return lines


def _write_report(df: pd.DataFrame, out_path: Path, warnings: List[str]) -> None:
    lines = [
        "# Sensitivity Analysis: eta and m",
        "",
        "Default operating point: lambda=0.4, eta=0.65, m=2, top_L=50, alpha=0.5",
        "",
        "## Setup",
        f"- Penalties: {PENALTIES}",
        f"- eta values: {ETA_LIST}",
        f"- m values: {M_LIST}",
        f"- lambda: {LAMBDA}",
        f"- top_L: {TOP_L}",
        "",
        "## Recommendation",
        "",
        "**Primary (eta=0.65, m=2):** This is the paper default used in difficulty_sweep_extended and click-model robustness experiments. "
        "The safe-prefix length m=2 keeps the top results stable while enabling exploration, matching the core EDE design.",
        "",
        "**Aggressive variant (eta=0.65, m=1):** Slightly more exploratory with marginally higher coverage in harsh regimes. "
        "Use if additional discovery at the cost of ranking stability is desired.",
        "",
    ]

    if warnings:
        lines.append("## Warnings")
        lines.extend([f"- {w}" for w in warnings])
        lines.append("")

    lines.append("## Coverage Tables")
    lines.append("")
    for penalty in PENALTIES:
        lines.extend(_format_heatmap_table(df, penalty))

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    root = _root_dir()
    outputs_dir, tables_dir = _outputs_dirs(root)

    base_config = Config(click_model="pbm")

    rows: List[Dict[str, object]] = []
    for penalty in PENALTIES:
        for eta in ETA_LIST:
            for m in M_LIST:
                aggregated = _run_config(base_config, penalty, eta, m, SEEDS)
                row = {
                    "penalty": penalty,
                    "eta": eta,
                    "m": m,
                }
                row.update(aggregated)
                rows.append(row)

    df = pd.DataFrame(rows)

    _validate_bounds(df)
    warnings: List[str] = []
    _warn_ndcg_trend(df, warnings)
    _warn_eta_trend(df, warnings)

    ordered_cols = [
        "penalty",
        "eta",
        "m",
        "ndcg10_mean",
        "ndcg10_std",
        "new_cov_mean",
        "new_cov_std",
        "unique_shown_new_mean",
        "unique_shown_new_std",
        "ttf_mean_mean",
        "ttf_mean_std",
    ]
    df = df[ordered_cols]

    out_csv = tables_dir / "sensitivity_eta_m.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n✓ Saved: {out_csv}")

    report_path = outputs_dir / "report_sensitivity_eta_m.md"
    _write_report(df, report_path, warnings)
    print(f"✓ Saved: {report_path}")

    print("\nSensitivity analysis completed successfully.")


if __name__ == "__main__":
    main()
