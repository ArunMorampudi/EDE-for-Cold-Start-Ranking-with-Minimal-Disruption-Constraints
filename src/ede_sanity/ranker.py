"""Base ranker for EDE sanity checks."""
import numpy as np


class BaseRanker:
    """Ranker that scores documents based on noisy relevance signals."""
    
    def __init__(self, rel_true: np.ndarray, sigma: float = 0.15, seed: int = 42):
        """
        Initialize base ranker.
        
        Args:
            rel_true: array of shape (n_queries, n_docs) with ground-truth continuous relevance
            sigma: noise standard deviation
            seed: random seed
        """
        self.rel_true = rel_true
        self.sigma = sigma
        self.seed = seed
    
    def score(
        self,
        query_idx: int,
        doc_indices: np.ndarray,
        new_docs_mask: np.ndarray,
        cold_start_penalty: float = 0.5,
    ) -> np.ndarray:
        """
        Score documents for a query.
        
        Args:
            query_idx: query index (for noise seeding consistency)
            doc_indices: array of doc indices to score
            new_docs_mask: boolean array indicating which docs are "new" (arrive after T0)
            cold_start_penalty: NOT USED (penalty now handled in policy for better control)
            
        Returns:
            scores: array of shape (len(doc_indices),) with raw scores (no penalty)
        """
        n_docs_to_score = len(doc_indices)
        
        # Get true relevances for these docs (query_idx is row, doc_indices are columns)
        true_rels = self.rel_true[query_idx, doc_indices]
        
        # Add Gaussian noise (deterministic per query)
        rng = np.random.RandomState(self.seed + query_idx)
        noise = rng.normal(0, self.sigma, size=n_docs_to_score)
        scores = true_rels + noise
        
        # NOTE: cold-start penalty is now handled in the policy, not here
        # This allows the policy to selectively apply or override the penalty
        
        return scores


def normalize_scores_by_rank(scores: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Normalize scores using rank-based normalization.
    
    Maps rank r in [1, k] to s_norm = 1 - (r-1)/(k-1).
    
    Args:
        scores: array of scores
        k: ranking cutoff
        
    Returns:
        normalized scores in [0, 1]
    """
    # Get ranking (descending order)
    ranks = np.argsort(-scores) + 1  # 1-indexed ranks
    
    # Initialize normalized scores
    s_norm = np.zeros_like(scores, dtype=np.float32)
    
    # Only normalize up to rank k
    for i, rank in enumerate(ranks):
        if rank <= k:
            s_norm[i] = 1 - (rank - 1) / (k - 1) if k > 1 else 1.0
        else:
            s_norm[i] = 0.0
    
    return s_norm
