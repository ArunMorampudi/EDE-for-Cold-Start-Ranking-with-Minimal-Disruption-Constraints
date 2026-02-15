# Baselines Suite Summary

## Baselines
- Baseline: pure base ranking (lambda=0, no exploration)
- EDE-Smoothed: entropy-driven exploration with alpha=0.5
- EDE-Unsmooth: entropy-driven exploration with alpha=0.0
- Random-Injection: replace a non-prefix slot with a random gated doc
- Epsilon-Greedy: random pick with epsilon=0.2 in non-prefix slots

## Key Findings

- EDE-Smoothed improves discovery at easy penalties: mean delta coverage +0.6220.
- Random-Injection and Epsilon-Greedy lag on discovery (mean delta coverage +0.0003, +0.0373).
- EDE-Smoothed retains higher NDCG in moderate penalties (mean delta NDCG +0.0012) vs Random (-0.0002) and Epsilon (-0.0072).
- Unsmooth entropy is noisier, with lower mean discovery gain at easy penalties (delta coverage +0.6165).

EDE-Smoothed is the most consistent method, beating both injection baselines on discovery while preserving NDCG.

## Entropy Stability Under Low Exposure

Smoothing reduces early-stage volatility of the entropy bonus, which prevents noisy over-exploration when impressions are sparse.

| Penalty | Stability (Smoothed) | Stability (Unsmoothed) | Ratio |
|---------|---------------------|------------------------|-------|
| 0.10    | 0.012180          | 0.013969             | 0.872 |
| 0.15    | 0.012218          | 0.013999             | 0.873 |
| 0.20    | 0.012203          | 0.014001             | 0.872 |
| 0.30    | 0.011781          | 0.013509             | 0.872 |

## Figures
- fig_cov_vs_penalty_strategies.png: Coverage vs penalty for all strategies
- fig_ndcg_vs_penalty_strategies.png: NDCG vs penalty for all strategies
- fig_unique_vs_penalty_strategies.png: Unique new docs vs penalty

## Table
- table_baselines_subset.csv: Key penalties [0.10, 0.20, 0.30, 0.60]
