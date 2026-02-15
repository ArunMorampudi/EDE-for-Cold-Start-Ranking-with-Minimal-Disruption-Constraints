"""Confirm operating points in normal and hard cold-start modes."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import Config
from .experiment import run_experiment, aggregate_metrics


DEBUG = False

CONFIGS = [
    {
        "config_name": "Baseline",
        "lambda": 0.00,
        "eta": 0.50,
        "m": 2,
        "top_L": 50,
    },
    {
        "config_name": "B",
        "lambda": 0.20,
        "eta": 0.50,
        "m": 2,
        "top_L": 50,
    },
    {
        "config_name": "C",
        "lambda": 0.30,
        "eta": 0.65,
        "m": 1,
        "top_L": 50,
    },
]

SEEDS = list(range(20))
MODES = ["normal", "hard"]


def _root_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def _outputs_dirs(root: Path) -> Tuple[Path, Path]:
    tables_dir = root / "outputs" / "tables"
    reports_dir = root / "outputs"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    return tables_dir, reports_dir


def _validate_summary(summary: Dict[str, Dict[str, float]]) -> None:
    errors: List[str] = []

    baseline = summary["Baseline"]
    conf_b = summary["B"]
    conf_c = summary["C"]

    for name, cfg in summary.items():
        for key in ["ndcg10_mean", "new_cov_mean", "avg_new_fraction_mean"]:
            value = cfg[key]
            if np.isnan(value) or value < 0.0 or value > 1.0:
                errors.append(f"{name}: {key} out of [0, 1] -> {value}")

    if conf_c["avg_new_fraction_mean"] <= baseline["avg_new_fraction_mean"]:
        errors.append("C: avg_new_fraction_mean not > Baseline")

    for name, cfg in [("B", conf_b), ("C", conf_c)]:
        if cfg["unique_shown_new_mean"] <= 0:
            errors.append(f"{name}: unique_shown_new_mean <= 0")
        if cfg["total_impressions_new_mean"] <= 0:
            errors.append(f"{name}: total_impressions_new_mean <= 0")

    if errors:
        raise RuntimeError("Validation failed:\n  - " + "\n  - ".join(errors))


def _write_report(summary: Dict[str, Dict[str, float]], report_path: Path, mode: str) -> None:
    baseline = summary["Baseline"]
    conf_b = summary["B"]
    conf_c = summary["C"]

    delta_cov_c = conf_c["new_cov_mean"] - baseline["new_cov_mean"]
    delta_ndcg_c = conf_c["ndcg10_mean"] - baseline["ndcg10_mean"]

    lines = [
        f"# Confirmed Operating Points ({mode} mode)",
        "",
        "## Summary",
        f"- Baseline NDCG@10: {baseline['ndcg10_mean']:.4f}",
        f"- Baseline coverage: {baseline['new_cov_mean']:.4f}",
        f"- Config C coverage gain: {delta_cov_c:+.4f}",
        f"- Config C NDCG delta: {delta_ndcg_c:+.4f}",
        "",
        "## Configs",
        "- Baseline: λ=0.0, η=0.5, m=2, top_L=50",
        "- B: λ=0.2, η=0.5, m=2, top_L=50",
        "- C: λ=0.3, η=0.65, m=1, top_L=50",
        "",
    ]

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _compute_deltas_table(df: pd.DataFrame) -> pd.DataFrame:
    baseline_rows = df[df["config_name"] == "Baseline"]
    if len(baseline_rows) != 1:
        raise RuntimeError("Expected exactly one Baseline row in operating points table.")

    if len(df[df["lambda"] == 0.0]) != 1:
        raise RuntimeError("Lambda==0 must appear only in the Baseline row.")

    for name in ["Baseline", "B", "C"]:
        row = df[df["config_name"] == name]
        if row.empty:
            raise RuntimeError(f"Missing required config row: {name}")
        if int(row.iloc[0]["top_L"]) != 50:
            raise RuntimeError(f"top_L must be 50 for {name} unless explicitly overridden.")

    baseline = baseline_rows.iloc[0]
    baseline_ndcg = float(baseline["ndcg10_mean"])
    baseline_cov = float(baseline["new_cov_mean"])
    baseline_ttf = float(baseline["ttf_mean_mean"])
    baseline_new_frac = float(baseline["avg_new_fraction_mean"])
    baseline_unique = float(baseline["unique_shown_new_mean"])

    deltas_df = df.copy()
    deltas_df["delta_ndcg"] = deltas_df["ndcg10_mean"] - baseline_ndcg
    deltas_df["delta_cov"] = deltas_df["new_cov_mean"] - baseline_cov
    deltas_df["delta_ttf"] = baseline_ttf - deltas_df["ttf_mean_mean"]
    deltas_df["delta_new_fraction"] = (
        deltas_df["avg_new_fraction_mean"] - baseline_new_frac
    )
    deltas_df["delta_unique_shown"] = (
        deltas_df["unique_shown_new_mean"] - baseline_unique
    )

    for _, row in deltas_df.iterrows():
        if row["ndcg10_mean"] < 0.0 or row["ndcg10_mean"] > 1.0:
            raise RuntimeError("NDCG out of [0,1] after rank fix.")
        if row["new_cov_mean"] < 0.0 or row["new_cov_mean"] > 1.0:
            raise RuntimeError("Coverage out of [0,1] after rank fix.")
        if abs((row["delta_ndcg"] + baseline_ndcg) - row["ndcg10_mean"]) > 1e-9:
            raise RuntimeError("delta_ndcg consistency check failed.")
        if abs((row["delta_cov"] + baseline_cov) - row["new_cov_mean"]) > 1e-9:
            raise RuntimeError("delta_cov consistency check failed.")
        if abs((baseline_ttf - row["delta_ttf"]) - row["ttf_mean_mean"]) > 1e-9:
            raise RuntimeError("delta_ttf consistency check failed.")

    ordered_cols = [
        "config_name",
        "lambda",
        "eta",
        "m",
        "top_L",
        "ndcg10_mean",
        "ndcg10_std",
        "delta_ndcg",
        "new_cov_mean",
        "new_cov_std",
        "delta_cov",
        "ttf_mean_mean",
        "ttf_mean_std",
        "delta_ttf",
        "avg_new_fraction_mean",
        "avg_new_fraction_std",
        "delta_new_fraction",
        "unique_shown_new_mean",
        "unique_shown_new_std",
        "delta_unique_shown",
    ]
    return deltas_df[ordered_cols]


def _format_markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    rows = [headers] + df.values.tolist()

    def _fmt(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    formatted_rows = [[_fmt(val) for val in row] for row in rows]
    header = "| " + " | ".join(formatted_rows[0]) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in formatted_rows[1:]]
    return "\n".join([header, sep] + body)


def _dump_per_seed(per_seed_by_config: Dict[str, List[Dict[str, float]]], tables_dir: Path, mode: str) -> None:
    for config_name, rows in per_seed_by_config.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        out_path = tables_dir / f"confirm_per_seed_{mode}_{config_name}.csv"
        df.to_csv(out_path, index=False)


def main() -> None:
    root = _root_dir()
    tables_dir, reports_dir = _outputs_dirs(root)

    base_config = Config()

    for mode in MODES:
        print(f"\nRunning operating points for mode={mode}...")
        per_seed_by_config: Dict[str, List[Dict[str, float]]] = {}
        summary: Dict[str, Dict[str, float]] = {}
        rows: List[Dict[str, float]] = []

        for cfg in CONFIGS:
            per_seed_metrics: List[Dict[str, float]] = []
            for seed in SEEDS:
                run_metrics = run_experiment(
                    base_config,
                    seed=seed,
                    mode=mode,
                    policy_params=cfg,
                    smoothed=True,
                )
                run_metrics["seed"] = seed
                per_seed_metrics.append(run_metrics)

            per_seed_by_config[cfg["config_name"]] = per_seed_metrics
            aggregated = aggregate_metrics(per_seed_metrics)
            summary[cfg["config_name"]] = aggregated

            row = {
                "config_name": cfg["config_name"],
                "lambda": cfg["lambda"],
                "eta": cfg["eta"],
                "m": cfg["m"],
                "top_L": cfg["top_L"],
            }
            row.update(aggregated)
            rows.append(row)

        _validate_summary(summary)

        df = pd.DataFrame(rows)
        ordered_cols = [
            "config_name",
            "lambda",
            "eta",
            "m",
            "top_L",
            "ndcg10_mean",
            "ndcg10_std",
            "new_cov_mean",
            "new_cov_std",
            "ttf_mean_mean",
            "ttf_mean_std",
            "avg_new_fraction_mean",
            "avg_new_fraction_std",
            "total_impressions_new_mean",
            "total_impressions_new_std",
            "unique_shown_new_mean",
            "unique_shown_new_std",
        ]
        df = df[ordered_cols]

        out_csv = tables_dir / f"operating_points_confirmed_{mode}.csv"
        df.to_csv(out_csv, index=False)
        print(f"✓ Saved: {out_csv}")

        report_path = reports_dir / f"report_operating_points_{mode}.md"
        _write_report(summary, report_path, mode)
        print(f"✓ Saved: {report_path}")

        if mode == "hard":
            deltas_df = _compute_deltas_table(df)
            deltas_csv = tables_dir / "operating_points_confirmed_deltas_hard.csv"
            deltas_df.to_csv(deltas_csv, index=False)
            print(f"✓ Saved: {deltas_csv}")

            deltas_table = deltas_df[
                [
                    "config_name",
                    "delta_ndcg",
                    "delta_cov",
                    "delta_ttf",
                    "delta_new_fraction",
                    "delta_unique_shown",
                ]
            ]
            report_path_confirmed = reports_dir / "report_operating_points_confirmed_hard.md"
            report_lines = [
                "# Confirmed Operating Points (hard mode)",
                "",
                "## Deltas (vs Baseline)",
                _format_markdown_table(deltas_table),
                "",
                "Note: delta_ttf = baseline_ttf - ttf (positive means faster discovery).",
                "",
            ]
            report_path_confirmed.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
            print(f"✓ Saved: {report_path_confirmed}")

        if DEBUG:
            _dump_per_seed(per_seed_by_config, tables_dir, mode)

    print("\nOperating points confirmed for both modes.")


if __name__ == "__main__":
    main()
