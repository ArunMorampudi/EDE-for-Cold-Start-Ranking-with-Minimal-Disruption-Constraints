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
    """Return revised EDE with the original baseline prefix preserved exactly."""
    from .revision_policy import rerank
    if new_docs_mask is None:
        new_docs_mask = np.zeros(len(impressions), dtype=bool)
    penalty = (penalty_value if penalty_value is not None else 0.3) if hard_cold_start else 0.0
    return rerank(base_scores, doc_indices, impressions, clicks, new_docs_mask,
                  penalty=penalty, lam=lam, eta=eta, m=m, top_L=top_L,
                  alpha=alpha if smoothed else 0.0, tau=tau, rng=rng, k=k)



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
