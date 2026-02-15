# Real-Data Replay Check

This module performs a conservative replay evaluation of EDE reranking using real click logs from the Yandex Personalized Web Search Challenge.

## Dataset

**Yandex Personalized Web Search Challenge**
- Download from: [Kaggle](https://www.kaggle.com/c/yandex-personalized-web-search-challenge) or original sources
- Format: TSV files with columns: SessionId, TimePassed, TypeOfAction, QueryId, RegionId, UrlId
- Required: At least one log file (.txt or .tsv) in the data directory

## Quick Start

```bash
# Install package
cd ede_sanity
pip install -e .

# Run replay check (basic)
python -m ede_real_replay.run_replay_check --data_dir /path/to/yandex/logs

# Run with custom parameters
python -m ede_real_replay.run_replay_check \
    --data_dir /path/to/yandex/logs \
    --output_dir outputs/real_replay \
    --k 10 \
    --lam 0.4 \
    --eta 0.65 \
    --m 2
```

## Output Files

After running, you'll find:
- `outputs/real_replay/replay_summary.csv` - Tabular results with acceptance rates and metrics
- `outputs/real_replay/report_real_replay.md` - Paper-ready summary with interpretation

## Replay Criteria

**Replay-Strict**: Only evaluate sessions where EDE top-k contains exactly the same documents as baseline top-k (any order). Most conservative.

**Replay-Click-Preservation**: Only evaluate sessions where every clicked document in baseline top-k remains in EDE top-k. Moderate conservative approach.

## Expected Results

- **Acceptance Rates**: 
  - Replay-Strict: typically 0.1-5% (very conservative)
  - Replay-Click-Preservation: typically 1-20% (more permissive)

- **Metrics**: CTR@10 and click-NDCG@10 computed on accepted sessions for both baseline (logged) and EDE policies

## Validation

The script includes automatic validation checks:
- ✓ At least 50k sessions loaded (warns if less)
- ✓ At least 70% sessions have >=10 results (warns if less)
- ✓ Click alignment >99% required (fails if violated)
- ✓ Acceptance rates above minimum thresholds (warns if very low)
- ✓ All metrics in [0,1] bounds (fails if violated)

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_dir` | (required) | Path to Yandex log directory |
| `--output_dir` | `outputs/real_replay` | Output directory |
| `--k` | 10 | Ranking cutoff |
| `--min_results` | 10 | Minimum results per session |
| `--lam` | 0.4 | EDE exploration weight |
| `--eta` | 0.65 | EDE novelty weight |
| `--m` | 2 | EDE safe-prefix length |
| `--alpha` | 0.5 | EDE entropy smoothing |
| `--top_L` | 50 | EDE candidate gating |
| `--tau` | 50.0 | EDE exposure decay |

## Testing

```bash
# Run unit tests
python -m unittest tests.test_real_replay -v
```

## Notes

- No model training required - EDE is applied directly to logged rankings
- Per-doc impression/click counters maintained in time order
- Conservative replay ensures minimal deviation from logged behavior
- Suitable as an appendix-style sanity check for paper validation
