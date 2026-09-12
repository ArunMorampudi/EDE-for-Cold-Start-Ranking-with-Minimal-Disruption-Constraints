# Reproducibility evidence

This folder is a compact evidence bundle for the corrected revision. It keeps
the focused revision policy and experiment/replay scripts, the pinned runtime
requirements, the validation/test protocol, and both summary and per-seed
results. The raw Yandex log is deliberately excluded; `fetch_replay_data.py`
documents how to obtain it with the author's own Kaggle access.

Validation seeds are 100-109. Held-out test seeds are 1000-1019. The frozen
operating point is lambda=0.4, eta=0.65, m=2, top_L=50, alpha=0.5, and tau=50.
The replay manifest identifies the logged-support sample and its bootstrap
settings. These materials support reproducibility checks but do not turn the
offline replay into causal or real-world cold-start evidence.
