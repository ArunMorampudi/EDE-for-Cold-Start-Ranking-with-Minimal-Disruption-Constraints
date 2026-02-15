# Real Click-Log Conservative Replay Sanity Check

Dataset: Yandex Personalized Web Search Challenge (Kaggle).

**IMPORTANT**: This is NOT unbiased counterfactual evaluation. This is a conservative replay sanity check that only evaluates sessions where EDE makes minimal or no changes to the logged ranking.

## Dataset

**Yandex Personalized Web Search Challenge**

- Total sessions loaded: 61,105
- Sessions with >=10 results: 61,105
- Sessions with clicks: 61,105
- Fields used: ranked URLs (top-10), click indicators (0/1)

### Click Distribution

- Average clicks per session (top-10): 1.432
- Sessions with exactly 1 click: 54.8%
- Sessions with 2+ clicks: 30.1%

*Note: High Click-Preservation acceptance rates are expected when most sessions have only 1-2 clicks, as EDE can preserve a single clicked document while reranking.*

## Conservative Replay Criteria

We evaluate EDE reranking under three conservative replay criteria:

1. **Replay-Exact-Match**: Accept session only if EDE top-k matches baseline top-k in exact order (strictest).

2. **Replay-Set-Match (order-agnostic)**: Accept session only if EDE top-k contains exactly the same documents as baseline top-k, but order can differ.

3. **Replay-Click-Preservation**: Accept session only if every clicked document in baseline top-k remains in EDE top-k (most permissive).

All criteria ensure conservative evaluation by limiting to sessions where EDE changes are minimal or preserve critical clicked results.

## EDE Configuration

- lambda (exploration weight): 0.4
- eta (novelty weight): 0.65
- m (safe-prefix length): 2
- alpha (smoothing): 0.5
- k (ranking cutoff): 10
- top_L (candidate gating): 50
- tau (exposure decay): 50.0

## Acceptance Rates

- **Replay-Exact-Match**: 76.89% (46981 sessions)
- **Replay-Set-Match**: 77.41% (47302 sessions)
- **Replay-Click-Preservation**: 99.47% (60781 sessions)

## Metrics on Accepted Sessions

*All metrics computed from replay_summary.csv*

### Replay-Exact-Match

| Policy | CTR@10 | NDCG@10 |
|--------|--------|----------|
| Baseline | 0.1473 | 0.6947 |
| EDE-Safer | 0.1473 | 0.6947 |

### Replay-Set-Match (order-agnostic)

| Policy | CTR@10 | NDCG@10 |
|--------|--------|----------|
| Baseline | 0.1472 | 0.6955 |
| EDE-Safer | 0.1472 | 0.6955 |

### Replay-Click-Preservation

| Policy | CTR@10 | NDCG@10 |
|--------|--------|----------|
| Baseline | 0.1419 | 0.7090 |
| EDE-Safer | 0.1448 | 0.7101 |

### Bootstrap 95% CI (B=1000) for Click-Preservation deltas

- ΔCTR@10 = +0.002848 [+0.002848, +0.002848]
- Δclick-NDCG@10 = +0.001021 [+0.001021, +0.001021]

*These are bootstrap CIs over accepted sessions (Replay-Click-Preservation) and are not causal estimates.*

## Ranking Change Diagnostics

The following diagnostics are computed over Replay-Click-Preservation sessions (large, stable bucket) unless noted otherwise.

**On Click-Preservation Bucket:**

- Exact-list change rate: 88.70%
- Set-change rate: 88.28%
- Reorder-only rate (same set, different order): 0.42%

**Position Change Distribution (Click-Preservation Bucket):**

- Mean positions changed count: 1.006 / 10
- Median positions changed count: 1.0
- P90 positions changed count: 1.0
- Mean positions changed fraction: 0.1006
- % sessions with exactly 1 position changed: 85.98%
- % sessions with >=2 positions changed: 5.27%
- % sessions with >=3 positions changed: 2.64%
- Top 3 most frequently changed positions: pos10:53672; pos9:2121; pos7:892

**Clicked Document Rank Shifts (Click-Preservation Bucket):**

- Mean absolute shift of clicked docs: 0.002
- Median absolute shift of clicked docs: 0.000
- % clicked docs with shift > 0: 0.17%

**Ranking Similarity (Click-Preservation Bucket):**

- Mean Kendall tau (on intersection): 0.0010
- Mean intersection size: 9.06

**Set-Match Bucket:**

- Order-change rate (set-match bucket only): 4.72%

**Interpretation:**

- Exact-list change rate: fraction of sessions where EDE top-10 differs from baseline (any position).
- Position change distribution is a 'gold star' diagnostic showing how many positions change per session:
  - Most changes are single-position replacements (typical: 1-2 docs swapped out)
  - Few sessions have 3+ positions changed (wholesale ranking reorganization)
- Top 3 changed positions usually skew toward position 10 (most permissive for replacement)
- Clicked documents preserved in position (0.000 mean shift) validates click-preservation constraint
- If set-match bucket shows ~0% order changes, exact-match and set-match acceptance rates coincide

## Interpretation

This appendix-style experiment provides a conservative replay sanity check of EDE reranking using real Yandex click logs. We apply EDE's entropy-driven exploration to logged candidate lists without training new models. The replay criteria ensure we only evaluate on sessions where EDE makes minimal changes (Exact/Set-Match) or preserves clicked results (Click-Preservation).

**Limitations**: This is not unbiased offline evaluation. EDE's reranking may benefit from observing the logged policy's exploration, and replay-based filtering introduces selection bias (we only evaluate on sessions where EDE happens to make acceptable changes). These results provide a directional sanity check, not causal claims about EDE's performance.

