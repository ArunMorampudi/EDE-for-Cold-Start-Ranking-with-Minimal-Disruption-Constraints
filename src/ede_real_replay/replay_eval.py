"""Conservative replay evaluation metrics."""
import numpy as np
from typing import List, Tuple, Dict
from .load_yandex import YandexSession


def compute_ctr_at_k(ranked_docs: List[int], clicks: List[int], k: int = 10) -> float:
    """
    Compute CTR@k (mean clicks in top-k).
    
    Args:
        ranked_docs: list of doc ids in ranking order
        clicks: 0/1 click indicators aligned to ranked_docs
        k: cutoff position
        
    Returns:
        CTR@k value
    """
    k_use = min(k, len(ranked_docs))
    if k_use == 0:
        return 0.0
    return sum(clicks[:k_use]) / k_use


def compute_click_ndcg_at_k(ranked_docs: List[int], clicks: List[int], k: int = 10) -> float:
    """
    Compute click-NDCG@k proxy.
    
    Uses clicks as binary relevance labels:
    - rel_i = click_i
    - DCG = sum(rel_i / log2(i+1))
    - IDCG = DCG of sorted rel (all clicked at top)
    - NDCG = DCG/IDCG
    
    Args:
        ranked_docs: list of doc ids in ranking order
        clicks: 0/1 click indicators aligned to ranked_docs
        k: cutoff position
        
    Returns:
        NDCG@k value
    """
    k_use = min(k, len(ranked_docs))
    if k_use == 0:
        return 0.0
    
    # Compute DCG
    dcg = 0.0
    for i in range(k_use):
        rel = clicks[i]
        dcg += rel / np.log2(i + 2)  # i+2 because positions are 0-indexed
    
    # Compute IDCG (ideal: all clicks at top)
    sorted_clicks = sorted(clicks[:k_use], reverse=True)
    idcg = 0.0
    for i, rel in enumerate(sorted_clicks):
        idcg += rel / np.log2(i + 2)
    
    # Handle zero clicks safely
    if idcg == 0.0:
        return 0.0
    
    return dcg / idcg


def check_replay_set_match(
    baseline_ranking: List[int],
    ede_ranking: List[int],
    k: int = 10,
) -> bool:
    """
    Check if EDE ranking passes Replay-Set-Match criterion.
    
    Accepts session only if EDE top-k == baseline top-k (same docs, any order).
    
    Args:
        baseline_ranking: baseline (logged) doc ids in order
        ede_ranking: EDE reranked doc ids in order
        k: cutoff position
        
    Returns:
        True if session passes Replay-Set-Match
    """
    k_use = min(k, len(baseline_ranking), len(ede_ranking))
    baseline_top_k = set(baseline_ranking[:k_use])
    ede_top_k = set(ede_ranking[:k_use])
    return baseline_top_k == ede_top_k


def check_replay_exact_match(
    baseline_ranking: List[int],
    ede_ranking: List[int],
    k: int = 10,
) -> bool:
    """
    Check if EDE ranking passes Replay-Exact-Match criterion.
    
    Accepts session only if EDE top-k matches baseline top-k in exact order.
    
    Args:
        baseline_ranking: baseline (logged) doc ids in order
        ede_ranking: EDE reranked doc ids in order
        k: cutoff position
        
    Returns:
        True if session passes Replay-Exact-Match
    """
    k_use = min(k, len(baseline_ranking), len(ede_ranking))
    return baseline_ranking[:k_use] == ede_ranking[:k_use]


def check_replay_click_preservation(
    baseline_ranking: List[int],
    baseline_clicks: List[int],
    ede_ranking: List[int],
    k: int = 10,
) -> bool:
    """
    Check if EDE ranking passes Replay-Click-Preservation criterion.
    
    Accepts session only if every clicked doc in baseline top-k is still in EDE top-k.
    
    Args:
        baseline_ranking: baseline (logged) doc ids in order
        baseline_clicks: 0/1 click indicators aligned to baseline_ranking
        ede_ranking: EDE reranked doc ids in order
        k: cutoff position
        
    Returns:
        True if session passes Replay-Click-Preservation
    """
    k_use = min(k, len(baseline_ranking), len(ede_ranking))
    
    # Find clicked docs in baseline top-k
    clicked_docs = set()
    for i in range(k_use):
        if baseline_clicks[i] == 1:
            clicked_docs.add(baseline_ranking[i])
    
    # Check if all clicked docs are in EDE top-k
    ede_top_k = set(ede_ranking[:k_use])
    return clicked_docs.issubset(ede_top_k)


def evaluate_replay_session(
    session: YandexSession,
    ede_ranking: List[int],
    k: int = 10,
) -> Dict[str, any]:
    """
    Evaluate a single session under both replay criteria.
    
    Args:
        session: YandexSession with baseline ranking and clicks
        ede_ranking: EDE reranked doc ids
        k: cutoff position
        
    Returns:
        dict with evaluation results
    """
    baseline_ranking = session.ranked_docs
    baseline_clicks = session.clicks
    
    # Check replay criteria
    exact_match_pass = check_replay_exact_match(baseline_ranking, ede_ranking, k)
    set_match_pass = check_replay_set_match(baseline_ranking, ede_ranking, k)
    click_pres_pass = check_replay_click_preservation(
        baseline_ranking, baseline_clicks, ede_ranking, k
    )
    
    # Compute metrics (we'll aggregate later based on acceptance)
    baseline_ctr = compute_ctr_at_k(baseline_ranking, baseline_clicks, k)
    baseline_ndcg = compute_click_ndcg_at_k(baseline_ranking, baseline_clicks, k)
    
    # For EDE metrics, we need to map clicks to the new ranking
    # Find where baseline docs appear in EDE ranking
    ede_clicks = [0] * len(ede_ranking)
    baseline_doc_to_click = {doc: click for doc, click in zip(baseline_ranking, baseline_clicks)}
    for i, doc in enumerate(ede_ranking):
        if doc in baseline_doc_to_click:
            ede_clicks[i] = baseline_doc_to_click[doc]
    
    ede_ctr = compute_ctr_at_k(ede_ranking, ede_clicks, k)
    ede_ndcg = compute_click_ndcg_at_k(ede_ranking, ede_clicks, k)
    
    return {
        'exact_match_pass': exact_match_pass,
        'set_match_pass': set_match_pass,
        'click_pres_pass': click_pres_pass,
        'baseline_ctr': baseline_ctr,
        'baseline_ndcg': baseline_ndcg,
        'ede_ctr': ede_ctr,
        'ede_ndcg': ede_ndcg,
    }


def aggregate_replay_results(
    results: List[Dict[str, any]],
    k: int = 10,
) -> Dict[str, any]:
    """
    Aggregate replay evaluation results across sessions.
    
    Args:
        results: list of per-session evaluation dicts
        k: cutoff position
        
    Returns:
        dict with aggregated metrics and acceptance rates
    """
    total_sessions = len(results)
    
    if total_sessions == 0:
        return {
            'total_sessions': 0,
            'exact_match_acceptance_rate': 0.0,
            'set_match_acceptance_rate': 0.0,
            'click_pres_acceptance_rate': 0.0,
            'exact_match_baseline_ctr': 0.0,
            'exact_match_baseline_ndcg': 0.0,
            'exact_match_ede_ctr': 0.0,
            'exact_match_ede_ndcg': 0.0,
            'set_match_baseline_ctr': 0.0,
            'set_match_baseline_ndcg': 0.0,
            'set_match_ede_ctr': 0.0,
            'set_match_ede_ndcg': 0.0,
            'click_pres_baseline_ctr': 0.0,
            'click_pres_baseline_ndcg': 0.0,
            'click_pres_ede_ctr': 0.0,
            'click_pres_ede_ndcg': 0.0,
        }
    
    # Filter by Replay-Exact-Match
    exact_match_results = [r for r in results if r['exact_match_pass']]
    exact_match_acceptance = len(exact_match_results) / total_sessions
    
    # Filter by Replay-Set-Match
    set_match_results = [r for r in results if r['set_match_pass']]
    set_match_acceptance = len(set_match_results) / total_sessions
    
    # Filter by Replay-Click-Preservation
    click_pres_results = [r for r in results if r['click_pres_pass']]
    click_pres_acceptance = len(click_pres_results) / total_sessions
    
    # Aggregate metrics for Replay-Exact-Match
    if exact_match_results:
        exact_match_baseline_ctr = np.mean([r['baseline_ctr'] for r in exact_match_results])
        exact_match_baseline_ndcg = np.mean([r['baseline_ndcg'] for r in exact_match_results])
        exact_match_ede_ctr = np.mean([r['ede_ctr'] for r in exact_match_results])
        exact_match_ede_ndcg = np.mean([r['ede_ndcg'] for r in exact_match_results])
    else:
        exact_match_baseline_ctr = 0.0
        exact_match_baseline_ndcg = 0.0
        exact_match_ede_ctr = 0.0
        exact_match_ede_ndcg = 0.0
    
    # Aggregate metrics for Replay-Set-Match
    if set_match_results:
        set_match_baseline_ctr = np.mean([r['baseline_ctr'] for r in set_match_results])
        set_match_baseline_ndcg = np.mean([r['baseline_ndcg'] for r in set_match_results])
        set_match_ede_ctr = np.mean([r['ede_ctr'] for r in set_match_results])
        set_match_ede_ndcg = np.mean([r['ede_ndcg'] for r in set_match_results])
    else:
        set_match_baseline_ctr = 0.0
        set_match_baseline_ndcg = 0.0
        set_match_ede_ctr = 0.0
        set_match_ede_ndcg = 0.0
    
    # Aggregate metrics for Replay-Click-Preservation
    if click_pres_results:
        click_pres_baseline_ctr = np.mean([r['baseline_ctr'] for r in click_pres_results])
        click_pres_baseline_ndcg = np.mean([r['baseline_ndcg'] for r in click_pres_results])
        click_pres_ede_ctr = np.mean([r['ede_ctr'] for r in click_pres_results])
        click_pres_ede_ndcg = np.mean([r['ede_ndcg'] for r in click_pres_results])
    else:
        click_pres_baseline_ctr = 0.0
        click_pres_baseline_ndcg = 0.0
        click_pres_ede_ctr = 0.0
        click_pres_ede_ndcg = 0.0
    
    return {
        'total_sessions': total_sessions,
        'exact_match_acceptance_rate': exact_match_acceptance,
        'exact_match_num_accepted': len(exact_match_results),
        'set_match_acceptance_rate': set_match_acceptance,
        'set_match_num_accepted': len(set_match_results),
        'click_pres_acceptance_rate': click_pres_acceptance,
        'click_pres_num_accepted': len(click_pres_results),
        'exact_match_baseline_ctr': exact_match_baseline_ctr,
        'exact_match_baseline_ndcg': exact_match_baseline_ndcg,
        'exact_match_ede_ctr': exact_match_ede_ctr,
        'exact_match_ede_ndcg': exact_match_ede_ndcg,
        'set_match_baseline_ctr': set_match_baseline_ctr,
        'set_match_baseline_ndcg': set_match_baseline_ndcg,
        'set_match_ede_ctr': set_match_ede_ctr,
        'set_match_ede_ndcg': set_match_ede_ndcg,
        'click_pres_baseline_ctr': click_pres_baseline_ctr,
        'click_pres_baseline_ndcg': click_pres_baseline_ndcg,
        'click_pres_ede_ctr': click_pres_ede_ctr,
        'click_pres_ede_ndcg': click_pres_ede_ndcg,
    }


def bootstrap_ci(deltas: np.ndarray, B: int = 1000, seed: int = 0) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for the mean delta.

    Args:
        deltas: 1D array of per-session deltas
        B: number of bootstrap resamples
        seed: RNG seed

    Returns:
        (point_estimate, ci_low, ci_high)
    """
    values = np.asarray(deltas, dtype=np.float64)
    if values.size == 0:
        return 0.0, 0.0, 0.0

    point = float(np.mean(values))
    rng = np.random.default_rng(seed)
    n = values.size

    # Memory-efficient bootstrap
    bootstrap_means = np.empty(B, dtype=np.float64)
    for b in range(B):
        sample_idx = rng.integers(0, n, size=n)
        bootstrap_means[b] = np.mean(values[sample_idx])

    ci_low = float(np.percentile(bootstrap_means, 2.5))
    ci_high = float(np.percentile(bootstrap_means, 97.5))
    return point, ci_low, ci_high


def validate_metrics(aggregated: Dict[str, any]) -> Dict[str, any]:
    """
    Validate aggregated metrics and return warnings.
    
    Args:
        aggregated: dict from aggregate_replay_results
        
    Returns:
        dict with validation results and warnings
    """
    validation = {
        'passed': True,
        'warnings': [],
    }
    
    # Check acceptance rates
    exact_rate = aggregated['exact_match_acceptance_rate']
    set_rate = aggregated['set_match_acceptance_rate']
    click_pres_rate = aggregated['click_pres_acceptance_rate']
    
    # Check ordering: exact <= set <= click_pres
    if not (exact_rate <= set_rate <= click_pres_rate + 1e-6):
        validation['warnings'].append(
            f"Acceptance rate ordering violated: exact={exact_rate:.4f}, set={set_rate:.4f}, click_pres={click_pres_rate:.4f}"
        )
    
    if exact_rate < 0.001:  # 0.1%
        validation['warnings'].append(
            f"Replay-Exact-Match acceptance rate very low: {exact_rate*100:.3f}% - order changes are common"
        )
    
    if set_rate < 0.001:  # 0.1%
        validation['warnings'].append(
            f"Replay-Set-Match acceptance rate very low: {set_rate*100:.3f}% (expected >0.1%)"
        )
    
    if click_pres_rate < 0.01:  # 1%
        validation['warnings'].append(
            f"Replay-Click-Preservation acceptance rate low: {click_pres_rate*100:.2f}% (expected >1%)"
        )
    
    # Check metric bounds [0,1]
    metrics_to_check = [
        ('exact_match_baseline_ctr', aggregated['exact_match_baseline_ctr']),
        ('exact_match_baseline_ndcg', aggregated['exact_match_baseline_ndcg']),
        ('exact_match_ede_ctr', aggregated['exact_match_ede_ctr']),
        ('exact_match_ede_ndcg', aggregated['exact_match_ede_ndcg']),
        ('set_match_baseline_ctr', aggregated['set_match_baseline_ctr']),
        ('set_match_baseline_ndcg', aggregated['set_match_baseline_ndcg']),
        ('set_match_ede_ctr', aggregated['set_match_ede_ctr']),
        ('set_match_ede_ndcg', aggregated['set_match_ede_ndcg']),
        ('click_pres_baseline_ctr', aggregated['click_pres_baseline_ctr']),
        ('click_pres_baseline_ndcg', aggregated['click_pres_baseline_ndcg']),
        ('click_pres_ede_ctr', aggregated['click_pres_ede_ctr']),
        ('click_pres_ede_ndcg', aggregated['click_pres_ede_ndcg']),
    ]
    
    for metric_name, value in metrics_to_check:
        if not (0.0 <= value <= 1.0):
            validation['passed'] = False
            validation['warnings'].append(
                f"Metric {metric_name} out of bounds: {value:.4f} (expected [0,1])"
            )
    
    return validation
