"""Click-model robustness sweep for EDE strategies."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import Config
from .experiment import run_experiment, aggregate_metrics
from .policy import safe_prefix_rerank, random_injection_rerank, epsilon_greedy_rerank


PENALTIES = [0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.80]
CLICK_MODELS = ["pbm", "cascade", "noisy"]
SEEDS = list(range(20))

STRATEGIES = [
    {
        "strategy_name": "Baseline",
        "policy_fn": safe_prefix_rerank,
        "smoothed": True,
        "params": {
            "lambda": 0.0,
            "eta": 0.65,
            "m": 2,
            "top_L": 50,
        },
    },
    {
        "strategy_name": "EDE-Safer",
        "policy_fn": safe_prefix_rerank,
        "smoothed": True,
        "params": {
            "lambda": 0.4,
            "eta": 0.65,
            "m": 2,
            "top_L": 50,
            "alpha": 0.5,
        },
    },
    {
        "strategy_name": "Random-Injection",
        "policy_fn": random_injection_rerank,
        "smoothed": True,
        "params": {
            "lambda": 0.4,
            "eta": 0.65,
            "m": 2,
            "top_L": 50,
            "inject_prob": 0.2,
        },
    },
    {
        "strategy_name": "Epsilon-Greedy",
        "policy_fn": epsilon_greedy_rerank,
        "smoothed": True,
        "params": {
            "lambda": 0.4,
            "eta": 0.65,
            "m": 2,
            "top_L": 50,
            "epsilon": 0.2,
        },
    },
]


def _root_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def _outputs_dirs(root: Path) -> Tuple[Path, Path, Path]:
    outputs_dir = root / "outputs"
    tables_dir = outputs_dir / "tables"
    figures_dir = outputs_dir / "figures"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return outputs_dir, tables_dir, figures_dir


def _run_strategy(
    base_config: Config,
    strategy: Dict[str, object],
    penalty: float,
    seeds: List[int],
) -> Dict[str, float]:
    per_seed: List[Dict[str, float]] = []
    params = dict(strategy["params"])
    params["penalty"] = penalty

    for seed in seeds:
        metrics = run_experiment(
            base_config,
            seed=seed,
            mode="hard",
            policy_params=params,
            smoothed=bool(strategy["smoothed"]),
            policy_fn=strategy["policy_fn"],
        )
        per_seed.append(metrics)

    return aggregate_metrics(per_seed)


def _validate_bounds(df: pd.DataFrame) -> None:
    for _, row in df.iterrows():
        for col in ["ndcg10_mean", "new_cov_mean", "avg_new_fraction_mean"]:
            value = float(row[col])
            if value < 0.0 or value > 1.0:
                raise RuntimeError(
                    f"Bounds check failed: {row['strategy_name']} {row['click_model']} "
                    f"p={row['penalty']} {col}={value}"
                )


def _warn_easy_regime(df: pd.DataFrame, warnings: List[str]) -> None:
    for click_model in CLICK_MODELS:
        subset = df[(df["click_model"] == click_model) & (df["strategy_name"] == "Baseline")]
        easy = subset[(subset["penalty"].isin([0.10, 0.15]))]
        ok = ((easy["new_cov_mean"] >= 0.05) & (easy["new_cov_mean"] <= 0.30)).any()
        if not ok:
            msg = (
                f"WARNING: No easy-regime baseline coverage in [0.05,0.30] for click_model={click_model}."
            )
            warnings.append(msg)
            print(msg)


def _check_core_robustness(df: pd.DataFrame, warnings: List[str]) -> None:
    worst = None
    worst_delta = 0.0

    for click_model in CLICK_MODELS:
        for penalty in [0.10, 0.15, 0.20]:
            subset = df[(df["click_model"] == click_model) & (df["penalty"] == penalty)]
            if subset.empty:
                continue

            ede = subset[subset["strategy_name"] == "EDE-Safer"]["new_cov_mean"].iloc[0]
            rand = subset[subset["strategy_name"] == "Random-Injection"]["new_cov_mean"].iloc[0]
            eps = subset[subset["strategy_name"] == "Epsilon-Greedy"]["new_cov_mean"].iloc[0]

            target = max(rand, eps) - 0.05
            delta = ede - target
            if delta < worst_delta:
                worst_delta = delta
                worst = (click_model, penalty, ede, rand, eps)

    if worst is not None and worst_delta < 0.0:
        click_model, penalty, ede, rand, eps = worst
        msg = (
            "WARNING: Robustness gap detected at "
            f"click_model={click_model}, p={penalty:.2f} "
            f"(EDE={ede:.4f}, Random={rand:.4f}, Eps={eps:.4f})."
        )
        warnings.append(msg)
        print(msg)

        if worst_delta < -0.10:
            raise RuntimeError("Robustness check failed: EDE-Safer below baselines by >0.10.")


def _plot_coverage(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(CLICK_MODELS), figsize=(15, 4), sharey=True)
    if len(CLICK_MODELS) == 1:
        axes = [axes]

    colors = {
        "Baseline": "#1f77b4",
        "EDE-Safer": "#ff7f0e",
        "Random-Injection": "#d62728",
        "Epsilon-Greedy": "#2ca02c",
    }

    for ax, click_model in zip(axes, CLICK_MODELS):
        subset = df[df["click_model"] == click_model]
        for strat in [s["strategy_name"] for s in STRATEGIES]:
            strat_df = subset[subset["strategy_name"] == strat].sort_values("penalty")
            if strat_df.empty:
                continue
            ax.errorbar(
                strat_df["penalty"],
                strat_df["new_cov_mean"],
                yerr=strat_df["new_cov_std"],
                marker="o",
                linewidth=2,
                markersize=4,
                label=strat,
                color=colors.get(strat),
                capsize=3,
            )
        ax.set_title(f"Click model: {click_model}")
        ax.set_xlabel("Penalty (p)")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("New Coverage")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _write_report(df: pd.DataFrame, out_path: Path, warnings: List[str]) -> None:
    lines = [
        "# Click-Model Robustness",
        "",
        "Default operating point: lambda=0.4, eta=0.65, m=2, top_L=50, alpha=0.5",
        "",
        "## Setup",
        f"- Penalties: {PENALTIES}",
        f"- Click models: {CLICK_MODELS}",
        f"- Seeds: {len(SEEDS)}",
        "- Strategies: Baseline, EDE-Safer, Random-Injection, Epsilon-Greedy",
        "",
        "## Notes",
        "Coverage and NDCG are aggregated across 20 seeds in hard cold-start mode.",
        "",
    ]

    if warnings:
        lines.append("## Warnings")
        lines.extend([f"- {w}" for w in warnings])
        lines.append("")

    lines.append("## Outputs")
    lines.append("- tables/click_model_robustness.csv")
    lines.append("- figures/robust_cov_lines.png")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    root = _root_dir()
    outputs_dir, tables_dir, figures_dir = _outputs_dirs(root)

    rows: List[Dict[str, object]] = []
    for click_model in CLICK_MODELS:
        base_config = Config(click_model=click_model)
        for penalty in PENALTIES:
            for strategy in STRATEGIES:
                aggregated = _run_strategy(base_config, strategy, penalty, SEEDS)
                row = {
                    "penalty": penalty,
                    "click_model": click_model,
                    "strategy_name": strategy["strategy_name"],
                    "lambda": strategy["params"].get("lambda", 0.0),
                    "eta": strategy["params"].get("eta", base_config.eta),
                    "m": strategy["params"].get("m", base_config.m),
                    "top_L": strategy["params"].get("top_L", base_config.top_L),
                }
                row.update(aggregated)
                rows.append(row)

    df = pd.DataFrame(rows)

    _validate_bounds(df)
    warnings: List[str] = []
    _warn_easy_regime(df, warnings)
    _check_core_robustness(df, warnings)

    ordered_cols = [
        "penalty",
        "click_model",
        "strategy_name",
        "lambda",
        "eta",
        "m",
        "top_L",
        "ndcg10_mean",
        "ndcg10_std",
        "new_cov_mean",
        "new_cov_std",
        "unique_shown_new_mean",
        "unique_shown_new_std",
        "ttf_mean_mean",
        "ttf_mean_std",
        "avg_new_fraction_mean",
        "avg_new_fraction_std",
    ]
    df = df[ordered_cols]

    out_csv = tables_dir / "click_model_robustness.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n✓ Saved: {out_csv}")

    fig_path = figures_dir / "robust_cov_lines.png"
    _plot_coverage(df, fig_path)
    print(f"✓ Saved: {fig_path}")

    report_path = outputs_dir / "report_click_model_robustness.md"
    _write_report(df, report_path, warnings)
    print(f"✓ Saved: {report_path}")

    print("\nClick-model robustness sweep completed successfully.")


if __name__ == "__main__":
    main()
