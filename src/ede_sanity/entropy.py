"""Entropy computation for EDE."""
import numpy as np
from typing import Tuple


def compute_entropy_scores(
    impressions: np.ndarray,
    clicks: np.ndarray,
    alpha: float = 0.5,
    tau: float = 50.0,
    eta: float = 0.5,
    smoothed: bool = True,
) -> np.ndarray:
    """
    Compute novelty-augmented, exposure-aware interaction entropy.
    
    Combines:
    - N(d) = exp(-I_d / tau): novelty term (high for unexplored docs)
    - R(d) = (1 - exp(-I_d / tau)) * H_norm(d): refinement term (entropy when doc is explored)
    - E'(d) = eta * N(d) + (1 - eta) * R(d): composite score
    
    Args:
        impressions: array of shape (n_docs,) with impression counts
        clicks: array of shape (n_docs,) with click counts
        alpha: Laplace smoothing constant
        tau: exposure weighting time constant
        eta: novelty weight (0.5 = balanced between exploration and exploitation)
        smoothed: if True, apply Laplace smoothing; if False, use eps for numerical stability
        
    Returns:
        E_prime(d): novelty-augmented entropy scores of shape (n_docs,)
    """
    eps = 1e-8
    
    # Compute 0/1 counts
    n0 = impressions - clicks  # non-clicks
    n1 = clicks                 # clicks
    
    if smoothed:
        # Laplace smoothing
        p1 = (n1 + alpha) / (impressions + 2 * alpha)
        p0 = (n0 + alpha) / (impressions + 2 * alpha)
    else:
        # Without smoothing, use eps for zero-count handling
        p1 = (n1 + eps) / (impressions + eps)
        p0 = (n0 + eps) / (impressions + eps)
        # Clamp to [0, 1]
        p1 = np.clip(p1, 0, 1)
        p0 = np.clip(p0, 0, 1)
    
    # Binary entropy: H = -p0*log(p0) - p1*log(p1)
    # Replace 0*log(0) = 0
    H = np.zeros_like(p1)
    mask_p0 = p0 > eps
    mask_p1 = p1 > eps
    H = np.where(mask_p0, -p0 * np.log(p0), 0) + np.where(mask_p1, -p1 * np.log(p1), 0)
    
    # Normalize by log(K) where K=2 for binary entropy
    H_norm = H / np.log(2)
    
    # Compute novelty and refinement terms
    # N(d) = exp(-I_d / tau): high for unexplored, low for heavily explored
    N = np.exp(-impressions / tau)
    
    # w = 1 - exp(-I_d / tau): exploration weight (inverse of N)
    w = 1 - N
    
    # R(d) = w * H_norm: refined entropy when doc is explored
    R = w * H_norm
    
    # E'(d) = eta * N(d) + (1 - eta) * R(d)
    E_prime = eta * N + (1 - eta) * R
    
    return E_prime


def binary_entropy(p: float) -> float:
    """
    Compute binary entropy for a single probability.
    
    Args:
        p: probability in [0, 1]
        
    Returns:
        H = -p*log(p) - (1-p)*log(1-p), handling edge cases
    """
    eps = 1e-8
    p = np.clip(p, eps, 1 - eps)
    return -p * np.log(p) - (1 - p) * np.log(1 - p)
