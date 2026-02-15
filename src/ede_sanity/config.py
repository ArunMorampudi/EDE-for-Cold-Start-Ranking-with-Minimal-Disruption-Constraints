"""Configuration dataclass for EDE sanity checks."""
from dataclasses import dataclass
from typing import List


@dataclass
class Config:
    """Central configuration for sanity check experiments."""
    
    # Random seed for reproducibility
    seed: int = 42
    
    # Environment parameters
    n_queries: int = 100
    n_docs: int = 1000
    n_new_docs: int = 200
    T: int = 2000  # total timesteps
    T0: int = 1000  # new docs arrive after T0
    
    # Ranking parameters
    k: int = 10  # ranking cutoff
    m: int = 2   # safe-prefix length
    top_L: int = 50  # Default chosen to match paper experiments (top_L=50).
    
    # Entropy parameters
    alpha: float = 0.5  # smoothing constant for entropy
    tau: float = 50.0   # exposure weighting time constant
    eta: float = 0.5    # novelty weight in E'(d) = eta*N(d) + (1-eta)*R(d)
    
    # Click model parameters
    a: float = 3.0      # relevance scaling
    b: float = 0.35     # position bias

    # Click model robustness
    click_model: str = "pbm"  # {"pbm","cascade","noisy"}
    cascade_stop_prob: float = 0.8
    noise_rate: float = 0.1
    noise_click_one: bool = True  # if True, noisy user clicks exactly one random doc; else can click none
    
    # Base ranker noise
    ranker_noise_sigma: float = 0.15
    
    # Cold-start penalty for new docs
    cold_start_penalty: float = 0.5
    
    # Hard cold-start mode: penalize new docs before candidate gating to increase discovery difficulty
    hard_cold_start: bool = True
    cold_start_penalty_hard: float = 0.3  # Moderate penalty (was 0.8); EDE will help discovery
    cold_start_extra_gate: bool = True    # Apply penalty during candidate gating for hard mode
    max_new_fraction_cap: float = None    # Optional cap on new fraction (None = no cap)

    # Baseline exploration settings
    epsilon_greedy: float = 0.2
    inject_prob: float = 0.2
    
    # NDCG relevance threshold (binary labels for NDCG)
    relevance_threshold: float = 0.5
    
    # Exploration lambdas to sweep
    lambdas: List[float] = None
    
    # Embedding dimension
    embedding_dim: int = 16
    
    # Number of relevant docs per query
    n_relevant_per_query: int = 10
    
    # Entropy stability logging
    log_entropy_timeseries: bool = False
    
    def __post_init__(self):
        """Set defaults for mutable fields."""
        if self.lambdas is None:
            self.lambdas = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]


PAPER_DEFAULTS = {
    "lambda": 0.4,
    "eta": 0.65,
    "m": 2,
    "top_L": 50,
    "alpha": 0.5,
}
