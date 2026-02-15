# Click-Model Robustness

Default operating point: lambda=0.4, eta=0.65, m=2, top_L=50, alpha=0.5

## Setup
- Penalties: [0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8]
- Click models: ['pbm', 'cascade', 'noisy']
- Seeds: 20
- Strategies: Baseline, EDE-Safer, Random-Injection, Epsilon-Greedy

## Notes
Coverage and NDCG are aggregated across 20 seeds in hard cold-start mode.

## Outputs
- tables/click_model_robustness.csv
- figures/robust_cov_lines.png
