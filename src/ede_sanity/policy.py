"""Ranking policy for EDE (safe-prefix reranking with entropy bonus)."""
import numpy as np
from .entropy import compute_entropy_scores


DEBUG = False


def safe_prefix_rerank(
    base_scores: np.ndarray,
    doc_indices: np.ndarray,
    impressions: np.ndarray,
    clicks: np.ndarray,
    k: int = 10,
    m: int = 2,
    top_L: int = 100,
    lam: float = 0.1,
    alpha: float = 0.5,
    tau: float = 50.0,
    eta: float = 0.5,
    smoothed: bool = True,
    new_docs_mask: np.ndarray = None,
    cold_start_penalty: float = 0.5,
    hard_cold_start: bool = True,
    penalty_value: float = None,
    rng: np.random.RandomState = None,
    inject_prob: float = None,
    epsilon: float = None,
) -> np.ndarray:
    """
    Rerank documents using safe-prefix strategy with novelty-augmented entropy bonus.
    
    Strategy:
    1. Gate candidates: keep only top_L by (base_score + eta*N(d))
       When hard_cold_start=True, apply penalty to new docs before gating to reduce their visibility
    2. Keep top-m from base ranker (within gated candidates) fixed
    3. For positions m+1..k, rerank by s = s0_norm + lambda * E'(d), where E'(d) includes novelty
    
    E'(d) = eta * N(d) + (1-eta) * R(d), where:
    - N(d) = exp(-I_d / tau): novelty term
    - R(d) = (1 - exp(-I_d / tau)) * H_norm(d): refinement term
    
    Args:
        base_scores: scores from base ranker (shape n_candidates, no penalty applied)
        doc_indices: indices of candidate docs in global doc array
        impressions: global impression counts (shape n_docs)
        clicks: global click counts (shape n_docs)
        k: ranking cutoff
        m: safe-prefix length (keep top-m from base ranker)
        top_L: candidate gating threshold (only rerank within top_L)
        lam: entropy bonus weight
        alpha: entropy smoothing constant
        tau: entropy exposure weighting constant
        eta: novelty weight in E'(d) = eta*N(d) + (1-eta)*R(d)
        smoothed: whether to use smoothed entropy
        new_docs_mask: boolean mask indicating new docs
        cold_start_penalty: penalty for new docs (used only if not hard_cold_start)
        hard_cold_start: if True, apply penalty before gating to increase discovery difficulty
        penalty_value: explicit penalty value to use (overrides defaults if provided)
        
    Returns:
        ranking: array of shape (k,) with reranked doc indices
    """
    n_candidates = len(doc_indices)
    k_use = min(k, n_candidates)
    
    if new_docs_mask is None:
        new_docs_mask = np.zeros(len(impressions), dtype=bool)
    
    # Determine penalty to apply based on mode
    if penalty_value is not None:
        penalty_val = penalty_value
    else:
        penalty_val = 0.3 if hard_cold_start else cold_start_penalty
    
    # BASELINE MODE: lambda=0 means NO EXPLORATION
    # Pure base ranking with no entropy, no novelty terms
    if lam == 0.0:
        # Apply penalty to new docs if hard_cold_start
        if hard_cold_start:
            is_new = new_docs_mask[doc_indices].astype(np.float32)
            scores_for_ranking = base_scores - penalty_val * is_new
        else:
            scores_for_ranking = base_scores
        
        # Sort purely by base scores (with penalty if applicable)
        sorted_indices = np.argsort(-scores_for_ranking)
        return doc_indices[sorted_indices[:k_use]]
    
    # EXPLORATION MODE: lambda > 0
    # Step 1: Apply penalty to new docs for gating decision if hard_cold_start
    if hard_cold_start:
        # Penalize new docs before gating: base_score - penalty if new
        is_new = new_docs_mask[doc_indices].astype(np.float32)
        scores_for_gating = base_scores - penalty_val * is_new
    else:
        scores_for_gating = base_scores
    
    # Step 2: Compute novelty term for all candidates
    N = np.exp(-impressions[doc_indices] / tau)
    
    # Gate candidates using adjusted scores
    adjusted_for_gating = scores_for_gating + eta * N
    
    top_L_use = min(top_L, n_candidates)
    gated_indices = np.argsort(-adjusted_for_gating)[:top_L_use]
    gated_docs = doc_indices[gated_indices]
    gated_scores = base_scores[gated_indices]  # Use unpenalized base scores in gated set
    gated_N = N[gated_indices]
    
    # Step 2: Get top-m from gated candidates
    # NOTE: We use base_scores for safe-prefix to maintain stability, but this means
    # the safe-prefix is NOT protected if it would filter out all new docs.
    # Alternative: use adjusted_for_gating scores for safe-prefix too (less conservative)
    # For now, using base_scores but accepting that new docs need to beat positions m+1..k
    top_m_indices_in_gated = np.argsort(-gated_scores)[:min(m, len(gated_docs))]
    top_m_docs = gated_docs[top_m_indices_in_gated]
    
    # Step 3: Remaining gated candidates (below top-m within gated set)
    remaining_mask = np.ones(len(gated_docs), dtype=bool)
    remaining_mask[top_m_indices_in_gated] = False
    remaining_indices_in_gated = np.where(remaining_mask)[0]
    remaining_docs = gated_docs[remaining_indices_in_gated]
    remaining_scores = gated_scores[remaining_indices_in_gated]
    
    # Step 4: Compute novelty-augmented entropy scores for remaining docs
    E_prime = compute_entropy_scores(
        impressions[remaining_docs],
        clicks[remaining_docs],
        alpha=alpha,
        tau=tau,
        eta=eta,
        smoothed=smoothed,
    )
    
    # Normalize base scores in remaining pool by rank
    order = np.argsort(-remaining_scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(remaining_docs) + 1)

    if DEBUG:
        if ranks.min() != 1 or ranks.max() != len(remaining_docs):
            raise RuntimeError("Rank normalization failed: ranks out of bounds.")
        if len(np.unique(ranks)) != len(remaining_docs):
            raise RuntimeError("Rank normalization failed: ranks are not a permutation.")

    s0_norm = np.zeros_like(remaining_scores, dtype=np.float32)
    for i, rank in enumerate(ranks):
        if rank <= len(remaining_docs):
            s0_norm[i] = 1 - (rank - 1) / (len(remaining_docs) - 1) if len(remaining_docs) > 1 else 1.0
    
    # Composite score: s = s0_norm + lambda * E'(d)
    composite = s0_norm + lam * E_prime
    
    # Step 5: Rerank remaining by composite score
    reranked_indices = remaining_docs[np.argsort(-composite)]
    
    # Step 6: Assemble final ranking
    final_ranking = np.concatenate([top_m_docs, reranked_indices])
    final_ranking = final_ranking[:k_use]
    
    return final_ranking


def random_injection_rerank(
    base_scores: np.ndarray,
    doc_indices: np.ndarray,
    impressions: np.ndarray,
    clicks: np.ndarray,
    k: int = 10,
    m: int = 2,
    top_L: int = 50,
    lam: float = 0.1,
    alpha: float = 0.5,
    tau: float = 50.0,
    eta: float = 0.5,
    smoothed: bool = True,
    new_docs_mask: np.ndarray = None,
    cold_start_penalty: float = 0.5,
    hard_cold_start: bool = True,
    penalty_value: float = None,
    rng: np.random.RandomState = None,
    inject_prob: float = None,
    epsilon: float = None,
) -> np.ndarray:
    """Randomly inject a gated candidate into the non-safe-prefix positions."""
    n_candidates = len(doc_indices)
    if n_candidates == 0:
        return np.array([], dtype=int)

    k_use = min(k, n_candidates)
    if new_docs_mask is None:
        new_docs_mask = np.zeros(len(impressions), dtype=bool)

    if penalty_value is not None:
        penalty_val = penalty_value
    else:
        penalty_val = 0.3 if hard_cold_start else cold_start_penalty

    if hard_cold_start:
        is_new = new_docs_mask[doc_indices].astype(np.float32)
        scores_for_gating = base_scores - penalty_val * is_new
    else:
        scores_for_gating = base_scores

    top_L_use = min(top_L, n_candidates)
    gated_indices = np.argsort(-scores_for_gating)[:top_L_use]
    gated_docs = doc_indices[gated_indices]
    gated_scores = scores_for_gating[gated_indices]

    if len(gated_docs) == 0:
        return np.array([], dtype=int)

    top_m_indices = np.argsort(-gated_scores)[:min(m, len(gated_docs))]
    top_m_docs = gated_docs[top_m_indices]

    remaining_mask = np.ones(len(gated_docs), dtype=bool)
    remaining_mask[top_m_indices] = False
    remaining_docs = gated_docs[remaining_mask]
    remaining_scores = gated_scores[remaining_mask]

    order = np.argsort(-remaining_scores)
    remaining_list = list(remaining_docs[order])

    if rng is None:
        rng = np.random.RandomState(0)
    inject_p = inject_prob if inject_prob is not None else lam

    if len(remaining_list) > 1 and rng.uniform() < inject_p:
        pick_pos = int(rng.randint(0, len(remaining_list)))
        replace_pos = int(rng.randint(0, len(remaining_list)))
        chosen = remaining_list.pop(pick_pos)
        remaining_list.insert(replace_pos, chosen)

    final_ranking = np.concatenate([top_m_docs, np.array(remaining_list, dtype=int)])
    return final_ranking[:k_use]


def epsilon_greedy_rerank(
    base_scores: np.ndarray,
    doc_indices: np.ndarray,
    impressions: np.ndarray,
    clicks: np.ndarray,
    k: int = 10,
    m: int = 2,
    top_L: int = 50,
    lam: float = 0.1,
    alpha: float = 0.5,
    tau: float = 50.0,
    eta: float = 0.5,
    smoothed: bool = True,
    new_docs_mask: np.ndarray = None,
    cold_start_penalty: float = 0.5,
    hard_cold_start: bool = True,
    penalty_value: float = None,
    rng: np.random.RandomState = None,
    inject_prob: float = None,
    epsilon: float = None,
) -> np.ndarray:
    """Epsilon-greedy injection into non-safe-prefix positions."""
    n_candidates = len(doc_indices)
    if n_candidates == 0:
        return np.array([], dtype=int)

    k_use = min(k, n_candidates)
    if new_docs_mask is None:
        new_docs_mask = np.zeros(len(impressions), dtype=bool)

    if penalty_value is not None:
        penalty_val = penalty_value
    else:
        penalty_val = 0.3 if hard_cold_start else cold_start_penalty

    if hard_cold_start:
        is_new = new_docs_mask[doc_indices].astype(np.float32)
        scores_for_gating = base_scores - penalty_val * is_new
    else:
        scores_for_gating = base_scores

    top_L_use = min(top_L, n_candidates)
    gated_indices = np.argsort(-scores_for_gating)[:top_L_use]
    gated_docs = doc_indices[gated_indices]
    gated_scores = scores_for_gating[gated_indices]

    if len(gated_docs) == 0:
        return np.array([], dtype=int)

    top_m_indices = np.argsort(-gated_scores)[:min(m, len(gated_docs))]
    top_m_docs = gated_docs[top_m_indices]

    remaining_mask = np.ones(len(gated_docs), dtype=bool)
    remaining_mask[top_m_indices] = False
    remaining_docs = gated_docs[remaining_mask]
    remaining_scores = gated_scores[remaining_mask]

    order = np.argsort(-remaining_scores)
    remaining_list = list(remaining_docs[order])

    if rng is None:
        rng = np.random.RandomState(0)
    eps = epsilon if epsilon is not None else 0.2

    chosen = []
    while remaining_list and len(chosen) < (k_use - len(top_m_docs)):
        if rng.uniform() < eps:
            pick_pos = int(rng.randint(0, len(remaining_list)))
            chosen.append(remaining_list.pop(pick_pos))
        else:
            chosen.append(remaining_list.pop(0))

    final_ranking = np.concatenate([top_m_docs, np.array(chosen, dtype=int)])
    return final_ranking[:k_use]
