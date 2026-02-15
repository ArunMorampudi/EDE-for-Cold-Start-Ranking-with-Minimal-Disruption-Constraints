"""Experiment API for reproducible EDE evaluations."""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional

import numpy as np

from .config import Config
from .simulate import SyntheticEnvironment
from .ranker import BaseRanker
from .policy import safe_prefix_rerank
from .metrics import compute_ndcg_at_k, compute_time_to_first_click, compute_ctr_by_position


def _compute_run_metrics(env: SyntheticEnvironment, config: Config) -> Dict[str, float]:
    ndcg_values: List[float] = []
    for query_idx, ranking in zip(env.query_history, env.ranking_history):
        if len(ranking) == 0:
            continue
        relevance_labels = env.relevance_labels[query_idx]
        ndcg_values.append(compute_ndcg_at_k(ranking, relevance_labels, k=config.k))

    ndcg10 = float(np.mean(ndcg_values)) if ndcg_values else 0.0

    new_relevant_docs = env.get_new_relevant_docs()
    ttf_dict, mean_ttf, _ = compute_time_to_first_click(
        new_relevant_docs,
        env.click_log,
        config.T,
    )
    new_cov = (
        sum(1 for ttf in ttf_dict.values() if ttf < config.T)
        / max(len(new_relevant_docs), 1)
    )

    new_fractions: List[float] = []
    for step, ranking in zip(env.step_history, env.ranking_history):
        if step < config.T0:
            continue
        if len(ranking) == 0:
            continue
        k_use = min(config.k, len(ranking))
        served = ranking[:k_use]
        new_fractions.append(float(np.mean(env.new_docs_mask[served])))

    avg_new_fraction = float(np.mean(new_fractions)) if new_fractions else 0.0

    total_impressions_new = int(env.impressions[env.new_docs_mask].sum())
    unique_shown_new = int((env.impressions[env.new_docs_mask] > 0).sum())

    return {
        "ndcg10": ndcg10,
        "new_cov": float(new_cov),
        "ttf_mean": float(mean_ttf),
        "avg_new_fraction": avg_new_fraction,
        "total_impressions_new": float(total_impressions_new),
        "unique_shown_new": float(unique_shown_new),
    }


def aggregate_metrics(per_seed: List[Dict[str, float]]) -> Dict[str, float]:
    aggregated: Dict[str, float] = {}
    if not per_seed:
        return aggregated

    keys = [key for key in per_seed[0].keys() if key != "seed"]
    for key in keys:
        values = np.array([run[key] for run in per_seed], dtype=float)
        aggregated[f"{key}_mean"] = float(np.mean(values))
        aggregated[f"{key}_std"] = float(np.std(values))
    return aggregated


def run_experiment(
    base_config: Config,
    seed: int,
    mode: str,
    policy_params: Dict[str, float],
    smoothed: bool = True,
    policy_fn=safe_prefix_rerank,
    baseline_check: bool = True,
) -> Dict[str, float]:
    """Run a single experiment and return metrics.

    Args:
        base_config: base configuration dataclass
        seed: random seed
        mode: "normal" or "hard"
        policy_params: policy parameters (lambda, eta, m, top_L, penalty optional)
        smoothed: whether to use smoothed entropy

    Returns:
        dict of metrics for the run
    """
    hard_cold_start = mode == "hard"

    config = replace(
        base_config,
        seed=seed,
        eta=policy_params.get("eta", base_config.eta),
        m=policy_params.get("m", base_config.m),
        top_L=policy_params.get("top_L", base_config.top_L),
        alpha=policy_params.get("alpha", base_config.alpha),
        tau=policy_params.get("tau", base_config.tau),
        hard_cold_start=hard_cold_start,
        cold_start_penalty_hard=policy_params.get(
            "penalty", base_config.cold_start_penalty_hard
        ),
    )

    lam = float(policy_params.get("lambda", 0.0))

    env = SyntheticEnvironment(config)
    ranker = BaseRanker(env.rel_true, sigma=config.ranker_noise_sigma, seed=seed)

    for step in range(config.T):
        query_idx = env.rng.randint(0, config.n_queries)
        candidates = env.retrieve_candidates(query_idx, step, k=100)
        if len(candidates) == 0:
            continue

        base_scores = ranker.score(
            query_idx,
            candidates,
            env.new_docs_mask,
            cold_start_penalty=config.cold_start_penalty,
        )

        ranking = policy_fn(
            base_scores=base_scores,
            doc_indices=candidates,
            impressions=env.impressions,
            clicks=env.clicks,
            k=config.k,
            m=config.m,
            top_L=config.top_L,
            lam=lam,
            alpha=config.alpha,
            tau=config.tau,
            eta=config.eta,
            smoothed=smoothed,
            new_docs_mask=env.new_docs_mask,
            cold_start_penalty=config.cold_start_penalty,
            hard_cold_start=hard_cold_start,
            penalty_value=config.cold_start_penalty_hard if hard_cold_start else None,
            rng=env.rng,
            inject_prob=policy_params.get("inject_prob"),
            epsilon=policy_params.get("epsilon"),
        )

        # Baseline consistency: lambda==0 should match pure base ranking
        if baseline_check and policy_fn is safe_prefix_rerank and lam == 0.0:
            if hard_cold_start:
                is_new = env.new_docs_mask[candidates].astype(np.float32)
                penalty_val = config.cold_start_penalty_hard
                scores_for_ranking = base_scores - penalty_val * is_new
            else:
                scores_for_ranking = base_scores
            expected = candidates[np.argsort(-scores_for_ranking)[: min(config.k, len(candidates))]]
            if not np.array_equal(ranking, expected):
                raise RuntimeError("Baseline consistency check failed: lambda==0 produced exploration behavior.")

        env.query_history.append(query_idx)
        env.ranking_history.append(np.array(ranking))
        env.step_history.append(step)

        for doc_idx in ranking:
            env.impressions[doc_idx] += 1

        k_use = min(config.k, len(ranking))
        shown = np.array(ranking[:k_use], dtype=int)
        rel_true_list = env.rel_true[query_idx, shown]
        positions = np.arange(1, len(shown) + 1)
        clicks = env.simulate_clicks(rel_true_list, positions, k_use)

        for pos, (doc_idx, clicked) in enumerate(zip(shown, clicks), start=1):
            if clicked:
                env.clicks[doc_idx] += 1
            env.click_log.append((step, doc_idx, bool(clicked)))
            env.click_pos_log.append((step, pos, bool(clicked)))

    metrics = _compute_run_metrics(env, config)

    ctr = compute_ctr_by_position(env.click_pos_log, max_k=10)
    ctr1 = ctr.get(1)
    ctr5 = ctr.get(5)
    ctr10 = ctr.get(10)
    if all(v is not None and not np.isnan(v) for v in [ctr1, ctr5, ctr10]):
        monotonic = ctr1 > ctr5 > ctr10
        status = "OK" if monotonic else "WARNING"
        print(f"CTR monotonicity ({status}): CTR@1={ctr1:.4f}, CTR@5={ctr5:.4f}, CTR@10={ctr10:.4f}")

    # Basic bounds checks
    if metrics["new_cov"] < 0.0 or metrics["new_cov"] > 1.0:
        raise RuntimeError(f"Coverage out of [0,1]: {metrics['new_cov']}")
    if metrics["ndcg10"] < 0.0 or metrics["ndcg10"] > 1.0:
        raise RuntimeError(f"NDCG out of [0,1]: {metrics['ndcg10']}")

    return metrics
