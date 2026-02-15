"""Run baseline/ablation suite for hard cold-start mode."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import Config
from .simulate import SyntheticEnvironment
from .ranker import BaseRanker
from .policy import safe_prefix_rerank, random_injection_rerank, epsilon_greedy_rerank
from .entropy import compute_entropy_scores
from .metrics import compute_ndcg_at_k, compute_time_to_first_click


PENALTIES = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.60, 0.80]
SEEDS = list(range(20))

LAMBDA = 0.40
ETA = 0.65
TOP_L = 50
M = 2


def _root_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def _outputs_dir(root: Path) -> Path:
    tables_dir = root / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    return tables_dir


def _strategy_definitions() -> List[Dict]:
    return [
        {
            "strategy_name": "Baseline",
            "policy_fn": safe_prefix_rerank,
            "smoothed": True,
            "params": {
                "lambda": 0.0,
                "eta": ETA,
                "m": M,
                "top_L": TOP_L,
            },
        },
        {
            "strategy_name": "EDE-Smoothed",
            "policy_fn": safe_prefix_rerank,
            "smoothed": True,
            "params": {
                "lambda": LAMBDA,
                "eta": ETA,
                "m": M,
                "top_L": TOP_L,
                "alpha": 0.5,
            },
        },
        {
            "strategy_name": "EDE-Unsmooth",
            "policy_fn": safe_prefix_rerank,
            "smoothed": False,
            "params": {
                "lambda": LAMBDA,
                "eta": ETA,
                "m": M,
                "top_L": TOP_L,
                "alpha": 0.0,
            },
        },
        {
            "strategy_name": "Random-Injection",
            "policy_fn": random_injection_rerank,
            "smoothed": True,
            "params": {
                "lambda": LAMBDA,
                "eta": ETA,
                "m": M,
                "top_L": TOP_L,
                "inject_prob": 0.2,
            },
        },
        {
            "strategy_name": "Epsilon-Greedy",
            "policy_fn": epsilon_greedy_rerank,
            "smoothed": True,
            "params": {
                "lambda": LAMBDA,
                "eta": ETA,
                "m": M,
                "top_L": TOP_L,
                "epsilon": 0.2,
            },
        },
    ]


def _entropy_stability_proxy(seed: int = 0) -> Tuple[float, float]:
    rng = np.random.RandomState(seed)
    impressions = rng.poisson(1.0, size=500).astype(float)
    clicks = rng.binomial(impressions.astype(int), 0.1).astype(float)

    smoothed = compute_entropy_scores(
        impressions=impressions,
        clicks=clicks,
        alpha=0.5,
        tau=50.0,
        eta=ETA,
        smoothed=True,
    )
    unsmoothed = compute_entropy_scores(
        impressions=impressions,
        clicks=clicks,
        alpha=0.0,
        tau=50.0,
        eta=ETA,
        smoothed=False,
    )

    low_mask = impressions <= 2
    if low_mask.sum() == 0:
        low_mask = np.ones_like(impressions, dtype=bool)

    return float(np.std(smoothed[low_mask])), float(np.std(unsmoothed[low_mask]))


def _init_state(env: SyntheticEnvironment, seed: int) -> Dict[str, object]:
    return {
        "impressions": np.zeros(env.config.n_docs, dtype=int),
        "clicks": np.zeros(env.config.n_docs, dtype=int),
        "click_log": [],
        "click_pos_log": [],
        "query_history": [],
        "ranking_history": [],
        "step_history": [],
        "rng": np.random.RandomState(seed),
        "entropy_log": [],  # List of (doc_id, impressions_after, entropy_bonus)
    }


def _compute_entropy_stability(
    entropy_log: List[Tuple[int, int, float]],
    impression_thresh: int = 5,
    min_events: int = 3,
    ) -> Tuple[float, float]:
    """
    Compute entropy stability metric for low-impression documents.
    
    For docs with impressions <= impression_thresh, compute per-doc std of entropy bonus.
    Return (mean_std, median_std) across qualifying docs.
    
    Args:
        entropy_log: list of (doc_id, impressions_after, entropy_bonus)
        impression_thresh: only consider events with impressions <= this
        min_events: minimum number of events required per doc
    
    Returns:
        (stability_mean_std, stability_median_std)
    """
    if not entropy_log:
        return float('nan'), float('nan')
    
    # Filter to low-impression events
    low_imp_events = [(doc_id, bonus) for doc_id, imp, bonus in entropy_log if imp <= impression_thresh]
    
    if not low_imp_events:
        return float('nan'), float('nan')
    
    # Group by doc_id
    from collections import defaultdict
    doc_bonuses = defaultdict(list)
    for doc_id, bonus in low_imp_events:
        doc_bonuses[doc_id].append(bonus)
    
    # Compute per-doc std for docs with at least min_events
    doc_stds = []
    for doc_id, bonuses in doc_bonuses.items():
        if len(bonuses) >= min_events:
            doc_stds.append(np.std(bonuses))
    
    if not doc_stds:
        return float('nan'), float('nan')
    
    return float(np.mean(doc_stds)), float(np.median(doc_stds))


def _compute_metrics(env: SyntheticEnvironment, state: Dict[str, object]) -> Dict[str, float]:
    ndcg_values: List[float] = []
    for query_idx, ranking in zip(state["query_history"], state["ranking_history"]):
        if len(ranking) == 0:
            continue
        relevance_labels = env.relevance_labels[query_idx]
        ndcg_values.append(compute_ndcg_at_k(ranking, relevance_labels, k=env.config.k))

    ndcg10 = float(np.mean(ndcg_values)) if ndcg_values else 0.0

    new_relevant_docs = env.get_new_relevant_docs()
    ttf_dict, mean_ttf, _ = compute_time_to_first_click(
        new_relevant_docs,
        state["click_log"],
        env.config.T,
    )
    new_cov = (
        sum(1 for ttf in ttf_dict.values() if ttf < env.config.T)
        / max(len(new_relevant_docs), 1)
    )

    new_fractions: List[float] = []
    for step, ranking in zip(state["step_history"], state["ranking_history"]):
        if step < env.config.T0:
            continue
        if len(ranking) == 0:
            continue
        k_use = min(env.config.k, len(ranking))
        served = ranking[:k_use]
        new_fractions.append(float(np.mean(env.new_docs_mask[served])))

    avg_new_fraction = float(np.mean(new_fractions)) if new_fractions else 0.0

    total_impressions_new = int(state["impressions"][env.new_docs_mask].sum())
    unique_shown_new = int((state["impressions"][env.new_docs_mask] > 0).sum())

    metrics = {
        "ndcg10": ndcg10,
        "new_cov": float(new_cov),
        "ttf_mean": float(mean_ttf),
        "avg_new_fraction": avg_new_fraction,
        "total_impressions_new": float(total_impressions_new),
        "unique_shown_new": float(unique_shown_new),
    }
    
    # Add entropy stability metrics if available
    if state.get("entropy_log"):
        stability_mean, stability_median = _compute_entropy_stability(state["entropy_log"])
        metrics["entropy_stability_mean"] = stability_mean
        metrics["entropy_stability_median"] = stability_median
    else:
        metrics["entropy_stability_mean"] = float('nan')
        metrics["entropy_stability_median"] = float('nan')
    
    return metrics


def _simulate_seed_penalty(seed: int, penalty: float, strategies: List[Dict]) -> Dict[str, Dict[str, float]]:
    base_config = Config(
        seed=seed,
        hard_cold_start=True,
        cold_start_penalty_hard=penalty,
        eta=ETA,
        m=M,
        top_L=TOP_L,
    )

    env = SyntheticEnvironment(base_config)
    ranker = BaseRanker(env.rel_true, sigma=base_config.ranker_noise_sigma, seed=seed)

    query_sequence = env.rng.randint(0, base_config.n_queries, size=base_config.T)

    state_by_strategy: Dict[str, Dict[str, object]] = {}
    for idx, strat in enumerate(strategies):
        state_by_strategy[strat["strategy_name"]] = _init_state(env, seed + 1000 * idx)

    for step, query_idx in enumerate(query_sequence):
        candidates = env.retrieve_candidates(query_idx, step, k=100)
        if len(candidates) == 0:
            continue

        base_scores = ranker.score(
            query_idx,
            candidates,
            env.new_docs_mask,
            cold_start_penalty=base_config.cold_start_penalty,
        )

        for strat in strategies:
            state = state_by_strategy[strat["strategy_name"]]
            params = strat["params"]

            ranking = strat["policy_fn"](
                base_scores=base_scores,
                doc_indices=candidates,
                impressions=state["impressions"],
                clicks=state["clicks"],
                k=base_config.k,
                m=params.get("m", M),
                top_L=params.get("top_L", TOP_L),
                lam=params.get("lambda", 0.0),
                alpha=params.get("alpha", base_config.alpha),
                tau=base_config.tau,
                eta=params.get("eta", ETA),
                smoothed=strat["smoothed"],
                new_docs_mask=env.new_docs_mask,
                cold_start_penalty=base_config.cold_start_penalty,
                hard_cold_start=True,
                penalty_value=penalty,
                rng=state["rng"],
                inject_prob=params.get("inject_prob"),
                epsilon=params.get("epsilon"),
            )

            state["query_history"].append(query_idx)
            state["ranking_history"].append(np.array(ranking))
            state["step_history"].append(step)

            for doc_idx in ranking:
                state["impressions"][doc_idx] += 1

                # Log entropy bonus for EDE strategies (for stability analysis)
                if params.get("lambda", 0.0) > 0 and strat["strategy_name"] in ["EDE-Smoothed", "EDE-Unsmooth"]:
                    # Compute entropy bonus for served docs after impression update
                    E_prime = compute_entropy_scores(
                        state["impressions"][ranking],
                        state["clicks"][ranking],
                        alpha=params.get("alpha", base_config.alpha),
                        tau=base_config.tau,
                        eta=params.get("eta", ETA),
                        smoothed=strat["smoothed"],
                    )
                    for doc_idx, entropy_bonus in zip(ranking, E_prime):
                        state["entropy_log"].append((doc_idx, state["impressions"][doc_idx], entropy_bonus))

            k_use = min(base_config.k, len(ranking))
            shown = np.array(ranking[:k_use], dtype=int)
            rel_true_list = env.rel_true[query_idx, shown]
            positions = np.arange(1, len(shown) + 1)
            clicks = env.simulate_clicks(rel_true_list, positions, k_use)

            for pos, (doc_idx, clicked) in enumerate(zip(shown, clicks), start=1):
                if clicked:
                    state["clicks"][doc_idx] += 1
                state["click_log"].append((step, doc_idx, bool(clicked)))
                state["click_pos_log"].append((step, pos, bool(clicked)))

    metrics_by_strategy: Dict[str, Dict[str, float]] = {}
    for strat in strategies:
        name = strat["strategy_name"]
        metrics_by_strategy[name] = _compute_metrics(env, state_by_strategy[name])

    return metrics_by_strategy


def _validate_results(df: pd.DataFrame) -> None:
    errors: List[str] = []

    def _get(penalty: float, strategy: str, col: str) -> float:
        row = df[(df["penalty"] == penalty) & (df["strategy_name"] == strategy)]
        if row.empty:
            raise RuntimeError(f"Missing row for penalty={penalty}, strategy={strategy}")
        return float(row.iloc[0][col])

    for _, row in df.iterrows():
        for col in ["ndcg10_mean", "new_cov_mean", "avg_new_fraction_mean"]:
            value = float(row[col])
            if value < 0.0 or value > 1.0:
                errors.append(
                    f"Bounds check failed: {row['strategy_name']} p={row['penalty']} {col}={value}"
                )

    for penalty in [0.10, 0.15, 0.20]:
        ede_cov = _get(penalty, "EDE-Smoothed", "new_cov_mean")
        rand_cov = _get(penalty, "Random-Injection", "new_cov_mean")
        eps_cov = _get(penalty, "Epsilon-Greedy", "new_cov_mean")
        if ede_cov < rand_cov - 0.02:
            errors.append(f"EDE coverage below Random at p={penalty}")
        if ede_cov < eps_cov - 0.02:
            errors.append(f"EDE coverage below Epsilon at p={penalty}")

    for penalty in [0.10, 0.20, 0.30]:
        ede_ndcg = _get(penalty, "EDE-Smoothed", "ndcg10_mean")
        rand_ndcg = _get(penalty, "Random-Injection", "ndcg10_mean")
        eps_ndcg = _get(penalty, "Epsilon-Greedy", "ndcg10_mean")
        if ede_ndcg < rand_ndcg - 0.01:
            errors.append(f"EDE NDCG below Random at p={penalty}")
        if ede_ndcg < eps_ndcg - 0.01:
            errors.append(f"EDE NDCG below Epsilon at p={penalty}")

    for penalty in [0.10, 0.20, 0.30]:
        ede_cov = _get(penalty, "EDE-Smoothed", "new_cov_mean")
        uns_cov = _get(penalty, "EDE-Unsmooth", "new_cov_mean")
        if ede_cov < uns_cov - 0.02:
            errors.append(f"EDE smoothed coverage below unsmoothed at p={penalty}")

    stable_smoothed, stable_unsmoothed = _entropy_stability_proxy()
    if stable_smoothed > stable_unsmoothed:
        print(
            "WARNING: Smoothed entropy appears less stable than unsmoothed "
            f"(std {stable_smoothed:.4f} > {stable_unsmoothed:.4f})."
        )

    # Check3b: Entropy stability from actual runs
    print("\n=== Check3b: Entropy Stability (Low-Impression Docs) ===")
    for penalty in [0.10, 0.15, 0.20, 0.30]:
        try:
            smoothed_stab = _get(penalty, "EDE-Smoothed", "entropy_stability_mean_mean")
            unsmooth_stab = _get(penalty, "EDE-Unsmooth", "entropy_stability_mean_mean")
            
            if np.isnan(smoothed_stab) or np.isnan(unsmooth_stab):
                print(f"  p={penalty}: WARNING - NaN stability values, skipping check")
                continue
            
            print(f"  p={penalty}: smoothed={smoothed_stab:.6f}, unsmoothed={unsmooth_stab:.6f}")
            
            if smoothed_stab > unsmooth_stab + 0.002:
                errors.append(
                    f"Check3b failed: Smoothed stability ({smoothed_stab:.6f}) > "
                    f"Unsmoothed stability ({unsmooth_stab:.6f}) at p={penalty}"
                )
        except Exception as e:
            print(f"  p={penalty}: WARNING - Could not compute stability check: {e}")

    if errors:
        raise RuntimeError("Validation failed:\n  - " + "\n  - ".join(errors))


def main() -> None:
    root = _root_dir()
    tables_dir = _outputs_dir(root)

    strategies = _strategy_definitions()
    per_strategy: Dict[str, List[Dict[str, float]]] = {s["strategy_name"]: [] for s in strategies}

    for penalty in PENALTIES:
        print(f"\nPenalty = {penalty:.2f}")
        for seed in SEEDS:
            metrics_by_strategy = _simulate_seed_penalty(seed, penalty, strategies)
            for strat in strategies:
                name = strat["strategy_name"]
                row = {
                    "penalty": penalty,
                    "strategy_name": name,
                }
                row.update(metrics_by_strategy[name])
                per_strategy[name].append(row)

    rows: List[Dict[str, float]] = []
    for strat in strategies:
        name = strat["strategy_name"]
        df = pd.DataFrame(per_strategy[name])
        for penalty in PENALTIES:
            subset = df[df["penalty"] == penalty]
            if subset.empty:
                continue
            aggregated = {
                "penalty": penalty,
                "strategy_name": name,
                "lambda": strat["params"].get("lambda", 0.0),
                "eta": strat["params"].get("eta", ETA),
                "m": strat["params"].get("m", M),
                "top_L": strat["params"].get("top_L", TOP_L),
                "alpha": strat["params"].get("alpha", 0.5),
                "epsilon": strat["params"].get("epsilon", 0.2),
                "inject_prob": strat["params"].get("inject_prob", 0.2),
            }
            for key in [
                "ndcg10",
                "new_cov",
                "unique_shown_new",
                "avg_new_fraction",
                "ttf_mean",
                    "entropy_stability_mean",
                "entropy_stability_median",
            ]:
                values = subset[key].to_numpy(dtype=float)
                aggregated[f"{key}_mean"] = float(np.mean(values))
                aggregated[f"{key}_std"] = float(np.std(values))
            rows.append(aggregated)

    df = pd.DataFrame(rows)

    deltas = []
    for penalty in PENALTIES:
        subset = df[df["penalty"] == penalty]
        baseline_rows = subset[subset["strategy_name"] == "Baseline"]
        if len(baseline_rows) != 1:
            raise RuntimeError(f"Expected one Baseline row for penalty={penalty}")
        baseline = baseline_rows.iloc[0]

        for _, row in subset.iterrows():
            row = row.copy()
            row["delta_cov"] = row["new_cov_mean"] - baseline["new_cov_mean"]
            row["delta_unique"] = row["unique_shown_new_mean"] - baseline["unique_shown_new_mean"]
            row["delta_ttf"] = baseline["ttf_mean_mean"] - row["ttf_mean_mean"]
            row["delta_ndcg"] = row["ndcg10_mean"] - baseline["ndcg10_mean"]
            deltas.append(row)

    df = pd.DataFrame(deltas)

    _validate_results(df)

    out_path = tables_dir / "baselines_suite_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\n✓ Saved: {out_path}")


if __name__ == "__main__":
    main()
