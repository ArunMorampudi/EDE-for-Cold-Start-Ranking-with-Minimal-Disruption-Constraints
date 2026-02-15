"""Metrics for EDE sanity checks."""
import numpy as np
from typing import Dict, List, Tuple


def compute_ndcg_at_k(
    ranking: np.ndarray,
    relevance_labels: np.ndarray,
    k: int = 10,
) -> float:
    """
    Compute NDCG@k for a single ranking.
    
    Args:
        ranking: array of doc indices (top-k)
        relevance_labels: binary relevance labels for all docs
        k: cutoff
        
    Returns:
        NDCG@k in [0, 1]
    """
    k_use = min(k, len(ranking))
    
    # DCG: sum_{i=1}^k rel[ranking[i]] / log2(i+1)
    dcg = 0.0
    for i in range(k_use):
        doc_idx = ranking[i]
        rel = relevance_labels[doc_idx]
        dcg += rel / np.log2(i + 2)  # i+2 because DCG uses 1-indexed positions
    
    # IDCG: ideal DCG with k best items
    ideal_ranking = np.argsort(-relevance_labels)[:k_use]
    idcg = 0.0
    for i in range(k_use):
        rel = relevance_labels[ideal_ranking[i]]
        idcg += rel / np.log2(i + 2)
    
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return ndcg


def compute_time_to_first_click(
    new_relevant_docs: set,
    click_log: List[Tuple[int, int, bool]],  # (step, doc, clicked)
    T: int,
) -> Tuple[Dict[int, int], float, float]:
    """
    Compute time-to-first-click for new relevant docs.
    
    Args:
        new_relevant_docs: set of doc indices that are new and relevant (binary label y=1)
        click_log: list of (step, doc_idx, clicked) tuples for all served docs
        T: total timesteps (for right-censoring)
        
    Returns:
        ttf_dict: dict mapping doc_idx -> first_click_step (or T if never clicked)
        mean_ttf: mean TTF (including T for censored)
        median_ttf: median TTF
    """
    ttf_dict = {}
    
    # Initialize all new relevant docs with censoring value
    for doc_idx in new_relevant_docs:
        ttf_dict[doc_idx] = T
    
    # Update with actual first click
    for step, doc_idx, clicked in click_log:
        if doc_idx in new_relevant_docs and clicked:
            if ttf_dict[doc_idx] == T:  # First click for this doc
                ttf_dict[doc_idx] = step
    
    if len(ttf_dict) > 0:
        ttf_values = list(ttf_dict.values())
        mean_ttf = np.mean(ttf_values)
        median_ttf = np.median(ttf_values)
    else:
        mean_ttf = T
        median_ttf = T
    
    return ttf_dict, mean_ttf, median_ttf


def compute_discovery_metrics(
    new_relevant_docs: set,
    click_log: List[Tuple[int, int, bool]],
    T: int,
) -> Dict[str, float]:
    """
    Compute discovery metrics.
    
    Args:
        new_relevant_docs: set of new relevant doc indices
        click_log: list of (step, doc_idx, clicked) tuples
        T: total timesteps
        
    Returns:
        dict with keys: 'coverage', 'mean_ttf', 'median_ttf'
    """
    ttf_dict, mean_ttf, median_ttf = compute_time_to_first_click(
        new_relevant_docs, click_log, T
    )
    
    # Coverage: number of new relevant docs clicked at least once
    coverage = sum(1 for ttf in ttf_dict.values() if ttf < T)
    
    return {
        'coverage': coverage,
        'n_new_relevant': len(new_relevant_docs),
        'coverage_rate': coverage / len(new_relevant_docs) if len(new_relevant_docs) > 0 else 0.0,
        'mean_ttf': mean_ttf,
        'median_ttf': median_ttf,
    }


def compute_ctr_by_position(
    click_pos_log: List[Tuple[int, int, bool]],
    max_k: int,
) -> Dict[int, float]:
    """
    Compute CTR by rank position from a click-position log.

    Args:
        click_pos_log: list of (step, position, clicked) tuples
        max_k: maximum position to compute

    Returns:
        dict mapping position -> CTR (NaN if no impressions at that position)
    """
    if max_k <= 0:
        return {}

    clicks = {pos: 0 for pos in range(1, max_k + 1)}
    impressions = {pos: 0 for pos in range(1, max_k + 1)}

    for _, pos, clicked in click_pos_log:
        if pos in impressions:
            impressions[pos] += 1
            if clicked:
                clicks[pos] += 1

    ctr = {}
    for pos in range(1, max_k + 1):
        if impressions[pos] == 0:
            ctr[pos] = float("nan")
        else:
            ctr[pos] = clicks[pos] / impressions[pos]

    return ctr
