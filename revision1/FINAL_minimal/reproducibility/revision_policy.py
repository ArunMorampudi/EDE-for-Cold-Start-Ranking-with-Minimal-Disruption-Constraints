"""Revision-one policy with an invariant prefix of the penalized base ranking.

Comparators are explicitly operational heuristics, not reproductions of named
bandit algorithms with theoretical guarantees. All share the same gate/prefix.
"""
import numpy as np


def rerank(scores, docs, impressions, clicks, new_mask, penalty=0.3,
           lam=0.4, eta=0.65, m=2, top_L=50, alpha=0.5, tau=50.,
           strategy="EDE", rng=None, k=10):
    if tau <= 0 or alpha < 0 or not 0 <= m <= k or top_L < k:
        raise ValueError("Require tau>0, alpha>=0, 0<=m<=k<=top_L")
    docs = np.asarray(docs, dtype=int)
    if len(docs) == 0:
        return docs
    adjusted = np.asarray(scores) - penalty * new_mask[docs]
    base = np.argsort(-adjusted, kind="stable")
    if lam == 0 or strategy == "Baseline":
        return docs[base[:k]]
    protected = base[:m]
    I = impressions[docs]
    C = clicks[docs]
    novelty = np.exp(-I / tau)
    if strategy == "No-decay":
        novelty = np.ones_like(novelty)
    denom = I + 2 * alpha
    prob = np.divide(C + alpha, denom, out=np.full(len(I), .5), where=denom > 0)
    prob = np.clip(prob, 1e-12, 1 - 1e-12)
    entropy = -(prob * np.log2(prob) + (1-prob) * np.log2(1-prob))
    refined = (1-novelty) * entropy
    bonus = eta * novelty + (1-eta) * refined
    if strategy == "Novelty-only": bonus = novelty
    if strategy == "Entropy-only": bonus = entropy
    if strategy == "Unrefined": bonus = eta * novelty + (1-eta) * entropy
    if strategy == "Recency": bonus = new_mask[docs].astype(float)
    if strategy == "UCB":
        bonus = np.clip(prob + np.sqrt(2*np.log(max(2, I.sum()))/(I+1)), 0, 1)
    if strategy == "Thompson":
        bonus = rng.beta(C+1, np.maximum(I-C, 0)+1)
    # Gate is common to comparators to isolate the scoring rule.
    gate_score = adjusted + eta * novelty
    if strategy == "No-gate": top_L = len(docs)
    pool_order = np.argsort(-gate_score, kind="stable")
    pool_order = pool_order[~np.isin(pool_order, protected)][:max(0, top_L-len(protected))]
    pool_order = pool_order[np.argsort(-adjusted[pool_order], kind="stable")]
    normalized = np.linspace(1, 0, len(pool_order)) if len(pool_order)>1 else np.ones(len(pool_order))
    if strategy == "No-normalization": normalized = adjusted[pool_order]
    composite = normalized + lam * bonus[pool_order]
    chosen = pool_order[np.argsort(-composite, kind="stable")]
    if strategy == "Random":
        chosen = pool_order.copy()
        if len(chosen)>1 and rng.random()<.2:
            a,b = rng.choice(len(chosen),2,replace=False)
            chosen[a],chosen[b] = chosen[b],chosen[a]
    if strategy == "Epsilon":
        left = list(pool_order); selected=[]
        while left and len(selected)<k-m:
            pick = int(rng.randint(len(left))) if rng.random()<.2 else 0
            selected.append(left.pop(pick))
        chosen=np.asarray(selected,dtype=int)
    if strategy == "Local-swap":
        chosen = pool_order.copy()
        # Disjoint neighboring swaps only; each item moves at most one place.
        for j in range(0,len(chosen)-1,2):
            if composite[j+1] > composite[j]: chosen[j],chosen[j+1]=chosen[j+1],chosen[j]
    if strategy == "Exposure-quota":
        # At least one eligible new item in the tail, if available.
        chosen = pool_order.copy()
        tail_size=max(0,k-m)
        if tail_size and not new_mask[docs[chosen[:tail_size]]].any():
            eligible=chosen[new_mask[docs[chosen]]]
            if len(eligible):
                target=eligible[0]
                chosen=np.concatenate([chosen[:tail_size-1],[target],chosen[tail_size-1:][chosen[tail_size-1:]!=target]])
    return docs[np.concatenate([protected,chosen])[:min(k,len(docs))]]
