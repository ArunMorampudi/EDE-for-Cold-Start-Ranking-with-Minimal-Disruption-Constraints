"""Synthetic environment simulation for EDE sanity checks."""
import numpy as np
from typing import Tuple, Dict, List, Set
from scipy.special import expit  # sigmoid function


class SyntheticEnvironment:
    """Temporal search environment with document arrivals and click model."""
    
    def __init__(self, config):
        """
        Initialize synthetic environment.
        
        Args:
            config: Config dataclass with parameters
        """
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        
        # Generate embeddings and relevance labels
        self._generate_data()
    
    def _generate_data(self):
        """Generate synthetic query/doc embeddings and relevance labels."""
        config = self.config
        
        # Query and doc embeddings: shape (n_queries, embedding_dim) and (n_docs, embedding_dim)
        self.query_embeddings = self.rng.normal(0, 1, (config.n_queries, config.embedding_dim))
        self.doc_embeddings = self.rng.normal(0, 1, (config.n_docs, config.embedding_dim))
        
        # Normalize embeddings
        self.query_embeddings /= np.linalg.norm(self.query_embeddings, axis=1, keepdims=True) + 1e-8
        self.doc_embeddings /= np.linalg.norm(self.doc_embeddings, axis=1, keepdims=True) + 1e-8
        
        # Compute continuous relevance: cosine similarity
        # rel_true[q, d] = cos(query_q, doc_d)
        self.rel_true = np.dot(self.query_embeddings, self.doc_embeddings.T)
        # Shift to [0, 1]
        self.rel_true = (self.rel_true + 1) / 2
        
        # Ensure each query has roughly n_relevant_per_query relevant docs
        # Binary relevance labels based on threshold
        self.relevance_labels = (self.rel_true >= config.relevance_threshold).astype(float)
        
        # Adjust relevance to ensure each query has target number of relevant docs
        for q in range(config.n_queries):
            n_relevant = self.relevance_labels[q].sum()
            target = config.n_relevant_per_query
            
            if n_relevant < target:
                # Need more relevant docs: set top-(target-n_relevant) non-relevant to relevant
                non_relevant_indices = np.where(self.relevance_labels[q] == 0)[0]
                scores_for_nonrel = self.rel_true[q, non_relevant_indices]
                top_nonrel = non_relevant_indices[np.argsort(-scores_for_nonrel)[:target - int(n_relevant)]]
                self.relevance_labels[q, top_nonrel] = 1
            elif n_relevant > target:
                # Too many relevant: set lowest-(n_relevant-target) relevant to non-relevant
                relevant_indices = np.where(self.relevance_labels[q] == 1)[0]
                scores_for_rel = self.rel_true[q, relevant_indices]
                bottom_rel = relevant_indices[np.argsort(scores_for_rel)[:int(n_relevant) - target]]
                self.relevance_labels[q, bottom_rel] = 0
        
        # Document availability: docs 0..799 are always available; docs 800..999 arrive after T0
        self.new_docs_mask = np.zeros(config.n_docs, dtype=bool)
        self.new_docs_mask[config.n_docs - config.n_new_docs:] = True
        
        # Interaction logs
        self.impressions = np.zeros(config.n_docs, dtype=int)
        self.clicks = np.zeros(config.n_docs, dtype=int)
        self.click_log: List[Tuple[int, int, bool]] = []  # (step, doc_idx, clicked)
        self.click_pos_log: List[Tuple[int, int, bool]] = []  # (step, position, clicked)
        # Ranking history for sanity checks
        self.query_history: List[int] = []
        self.ranking_history: List[np.ndarray] = []
        self.step_history: List[int] = []
    
    def get_available_docs(self, step: int) -> np.ndarray:
        """Get list of available docs at a given timestep."""
        available = np.arange(self.config.n_docs)
        # Before T0, only old docs available; after T0, all docs available
        if step < self.config.T0:
            available = available[~self.new_docs_mask]
        return available
    
    def get_relevant_docs_for_query(self, query_idx: int) -> np.ndarray:
        """Get doc indices relevant to a query."""
        return np.where(self.relevance_labels[query_idx] > 0)[0]
    
    def get_new_relevant_docs(self) -> Set[int]:
        """Get set of docs that are new AND relevant for at least one query."""
        new_relevant = set()
        for doc_idx in np.where(self.new_docs_mask)[0]:
            # Check if this doc is relevant for any query
            if (self.relevance_labels[:, doc_idx] > 0).any():
                new_relevant.add(doc_idx)
        return new_relevant
    
    def retrieve_candidates(self, query_idx: int, step: int, k: int = 100) -> np.ndarray:
        """
        Retrieve candidate docs for a query (all available docs).
        
        Args:
            query_idx: query index
            step: timestep (for document availability)
            k: max number of candidates to retrieve
            
        Returns:
            candidate doc indices
        """
        available = self.get_available_docs(step)
        # Return top-k by a simple heuristic (or return all if fewer than k)
        if len(available) <= k:
            return available
        # Simple heuristic: return random sample or all
        return available
    
    def simulate_click(
        self,
        query_idx: int,
        doc_idx: int,
        position: int,
    ) -> bool:
        """
        Simulate click using position-biased model.
        
        Args:
            query_idx: query index
            doc_idx: document index
            position: 1-indexed position in ranking
            
        Returns:
            True if click, False otherwise
        """
        rel_true = self.rel_true[query_idx, doc_idx]
        clicks = self.simulate_clicks(
            rel_true_list=np.array([rel_true], dtype=float),
            positions=np.array([position], dtype=int),
            k=1,
        )
        return bool(clicks[0])

    def simulate_clicks(
        self,
        rel_true_list: np.ndarray,
        positions: np.ndarray,
        k: int,
    ) -> np.ndarray:
        """
        Simulate clicks for a ranked list under the configured click model.

        Args:
            rel_true_list: array of true relevance values for shown docs
            positions: array of 1-indexed positions aligned with rel_true_list
            k: number of shown docs to simulate

        Returns:
            clicks: 0/1 array aligned to the shown docs
        """
        if k <= 0:
            return np.zeros(0, dtype=int)

        n = min(k, len(rel_true_list), len(positions))
        rel_true_list = rel_true_list[:n]
        positions = positions[:n]

        model = self.config.click_model
        clicks = np.zeros(n, dtype=int)

        if model == "noisy":
            if self.rng.uniform() < self.config.noise_rate:
                if self.config.noise_click_one:
                    pick = int(self.rng.randint(0, n))
                    clicks[pick] = 1
                else:
                    if self.rng.uniform() >= 0.5:
                        pick = int(self.rng.randint(0, n))
                        clicks[pick] = 1
                return clicks

        # PBM click probability for each position
        logits = self.config.a * (rel_true_list - 0.5) - self.config.b * positions
        p_click = expit(logits)

        if model == "cascade":
            for i, p in enumerate(p_click):
                clicked = self.rng.uniform() < p
                clicks[i] = 1 if clicked else 0
                if clicked and (self.rng.uniform() < self.config.cascade_stop_prob):
                    break
            return clicks

        # Default: PBM (independent per position)
        clicks = (self.rng.uniform(size=n) < p_click).astype(int)
        return clicks
    
    def run_episode(self, ranker, policy, lam: float = 0.1, smoothed: bool = True, hard_cold_start: bool = True) -> None:
        """
        Run a single episode (one query at each timestep).
        
        Args:
            ranker: BaseRanker instance
            policy: reranking policy function
            lam: entropy bonus weight
            smoothed: use smoothed entropy
            hard_cold_start: whether to apply hard cold-start penalty before gating
        """
        for step in range(self.config.T):
            # Sample a query uniformly
            query_idx = self.rng.randint(0, self.config.n_queries)
            
            # Retrieve candidates
            candidates = self.retrieve_candidates(query_idx, step, k=100)
            if len(candidates) == 0:
                continue
            
            # Score candidates with base ranker
            base_scores = ranker.score(
                query_idx,
                candidates,
                self.new_docs_mask,
                cold_start_penalty=self.config.cold_start_penalty,
            )
            
            # Apply reranking policy
            ranking = policy(
                base_scores=base_scores,
                doc_indices=candidates,
                impressions=self.impressions,
                clicks=self.clicks,
                k=self.config.k,
                m=self.config.m,
                top_L=self.config.top_L,
                lam=lam,
                alpha=self.config.alpha,
                tau=self.config.tau,
                eta=self.config.eta,
                smoothed=smoothed,
                new_docs_mask=self.new_docs_mask,
                cold_start_penalty=self.config.cold_start_penalty,
                hard_cold_start=hard_cold_start,
                penalty_value=self.config.cold_start_penalty_hard if hard_cold_start else None,
            )

            # Store served ranking + query for sanity checks
            self.query_history.append(query_idx)
            self.ranking_history.append(np.array(ranking))
            self.step_history.append(step)
            
            # Log impressions for served docs
            for doc_idx in ranking:
                self.impressions[doc_idx] += 1
            
            # Simulate clicks
            k_use = min(self.config.k, len(ranking))
            shown = np.array(ranking[:k_use], dtype=int)
            rel_true_list = self.rel_true[query_idx, shown]
            positions = np.arange(1, len(shown) + 1)
            clicks = self.simulate_clicks(rel_true_list, positions, k_use)

            for pos, (doc_idx, clicked) in enumerate(zip(shown, clicks), start=1):
                if clicked:
                    self.clicks[doc_idx] += 1
                self.click_log.append((step, doc_idx, bool(clicked)))
                self.click_pos_log.append((step, pos, bool(clicked)))
    
    def reset_interaction_logs(self):
        """Reset impressions and clicks for a new run."""
        self.impressions = np.zeros(self.config.n_docs, dtype=int)
        self.clicks = np.zeros(self.config.n_docs, dtype=int)
        self.click_log = []
        self.click_pos_log = []
        self.query_history = []
        self.ranking_history = []
        self.step_history = []
