# Paper-Ready Difficulty Sweep Summary

Default operating point: lambda=0.4, eta=0.65, m=2, top_L=50, alpha=0.5

## Experimental Setup

We evaluate entropy-driven exploration (EDE) across a **broad difficulty spectrum** by varying the penalty parameter $p$ in hard cold-start mode. The penalty controls how aggressively the ranker avoids showing new documents (at $p=0$, new docs are unrestricted; at $p=1$, they incur maximum penalty). For each penalty level, we run 20 independent seeds with two configurations: (1) **Baseline**: a pure ranking policy with no exploration ($\lambda=0$), and (2) **EDE**: entropy-driven exploration ($\lambda=0.4$, $\eta=0.65$, $m=2$). We measure coverage (fraction of new relevant docs discovered) and NDCG@10 (ranking quality).

## Key Findings

**1. Easy Regime:** At lower penalties (p=0.10, p=0.15), the problem remains non-trivial even without exploration. For example, at $p=0.10$, baseline achieves only 20.1% coverage despite pure ranking quality (NDCG=0.18). This validates that discovery is a **genuine challenge**, not an artifact of overly-favorable settings. EDE dramatically improves performance, reaching 73.3% coverage (+53.2 percentage points) and surfacing 171 unique new documents vs. baseline's 43.

**2. Moderate Regime:** At $p = 0.20$–$0.30$, baseline coverage drops sharply (0.6%–4.6%), while EDE maintains strong improvements (coverage 39%–67%). This regime reveals the **core exploration challenge**: many high-quality new documents exist but are masked by the penalty, requiring entropy-driven search to uncover.

**3. Extreme Regime:** At harsh penalties ($p\geq 0.60$), even baseline approaches zero coverage, and EDE improvement plateaus due to the severity of the constraint. At $p=0.6$, EDE achieves only 0.70% coverage (+0.70 percentage points vs. baseline). This demonstrates **realistic limitations**: beyond a certain difficulty, no exploration strategy can overcome an overly-restrictive penalty.

**4. Consistent Benefit:** Across all difficulties, EDE consistently outperforms baseline (coverage always higher, unique selection always higher). The magnitude of benefit scales naturally with difficulty: largest at moderate penalties where the problem is challenging but solvable, tapering at extremes where the problem becomes intractable.

**5. NDCG Trade-off:** EDE exhibits a small NDCG@10 reduction ($\Delta \approx -0.01$ across most penalties) due to prioritizing discovery over ranking quality. This trade-off is intentional and acceptable in a discovery-focused application.

## Figures and Tables

**`figures/fig_cov_vs_penalty.png`** shows coverage vs. penalty across the easy-to-extreme spectrum. The green shaded region (5%–30%) marks the easy regime where baseline achieves non-trivial discovery. EDE (orange line) significantly outperforms baseline (blue line) everywhere, with the largest *absolute* gains in the moderate difficulty range.

**`figures/fig_ndcg_vs_penalty.png`** displays NDCG@10 over the same penalty range. Both configurations maintain stable ranking quality (≈0.17–0.18) across most penalties, with EDE showing minimal degradation. Error bars reflect ±1 standard deviation over 20 seeds.

**`tables/table_difficulty_subset.csv`** summarizes key metrics at five representative penalties (easy, moderate, harsh, extreme). For each, we report baseline/EDE coverage and NDCG, absolute gains, and unique document count.

