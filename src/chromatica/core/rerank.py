"""
Reranking module for Chromatica using Sinkhorn-EMD distance.

This module implements the second stage of the search pipeline, providing
high-fidelity reranking using the Sinkhorn-approximated Earth Mover's Distance.
It supports both single-query and batched operations for efficiency.

Key Features:
- Sinkhorn-EMD distance computation
- Batched reranking operations
- L2 distance fallback for fast mode
- Pre-computed cost matrices for efficiency
"""

import numpy as np
import logging
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
import ot  # Python Optimal Transport

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """Result from reranking a single candidate."""

    image_id: str
    distance: float
    confidence: float = 1.0


def create_cost_matrix(bins: int = 1152) -> np.ndarray:
    """
    Create the cost matrix for EMD computation.

    Args:
        bins: Number of histogram bins (default: 1152 for 8x12x12 Lab grid)

    Returns:
        Cost matrix for EMD computation
    """
    # Create indices for bin centers
    idx = np.arange(bins)

    # Compute pairwise squared distances
    cost_matrix = np.zeros((bins, bins), dtype=np.float32)
    for i in range(bins):
        cost_matrix[i] = (idx - i) ** 2

    return cost_matrix


def compute_sinkhorn_distance(
    hist1: np.ndarray,
    hist2: np.ndarray,
    cost_matrix: Optional[np.ndarray] = None,
    reg: float = 0.1,
    numItermax: int = 100,
    min_threshold: float = 1e-10,
    **kwargs,
) -> float:
    """
    Compute Sinkhorn-approximated Earth Mover's Distance between two histograms.

    Args:
        hist1: First histogram (query)
        hist2: Second histogram (candidate)
        cost_matrix: Pre-computed cost matrix (if None, will be computed)
        reg: Regularization parameter for Sinkhorn algorithm
        numItermax: Maximum number of Sinkhorn iterations
        min_threshold: Minimum value for histogram entries to avoid div by zero
        **kwargs: Additional arguments passed to ot.sinkhorn2

    Returns:
        Computed EMD distance (float)
    """
    try:
        # Ensure histograms are properly shaped and normalized
        hist1 = hist1.astype(np.float64).flatten()
        hist2 = hist2.astype(np.float64).flatten()

        # Add small constant to avoid divide by zero
        hist1 = np.maximum(hist1, min_threshold)
        hist2 = np.maximum(hist2, min_threshold)

        # Normalize histograms
        hist1 = hist1 / hist1.sum()
        hist2 = hist2 / hist2.sum()

        # Create or use cost matrix
        if cost_matrix is None:
            cost_matrix = create_cost_matrix(len(hist1))

        try:
            # Try Sinkhorn with stabilization
            distance = ot.sinkhorn2(
                hist1,
                hist2,
                cost_matrix,
                reg,
                method="sinkhorn_stabilized",  # Use stabilized version
                numItermax=numItermax,
                stopThr=1e-6,
                verbose=False,
                **kwargs,
            )
            return float(distance)
        except Exception as sink_err:
            logger.warning(
                f"Stabilized Sinkhorn failed, trying epsilon scaling: {sink_err}"
            )
            # Try with epsilon scaling if stabilized version fails
            distance = ot.sinkhorn2(
                hist1,
                hist2,
                cost_matrix,
                reg,
                method="sinkhorn_epsilon_scaling",
                numItermax=numItermax,
                stopThr=1e-6,
                verbose=False,
                **kwargs,
            )
            return float(distance)

    except Exception as e:
        logger.error(f"All Sinkhorn methods failed: {e}, falling back to L2")
        return float(np.sqrt(np.sum((hist1 - hist2) ** 2)))


def rerank_candidates(
    query_histogram: np.ndarray,
    candidate_histograms: np.ndarray,
    batch_size: int = 10,
    reg: float = 0.1,
    numItermax: int = 100,
    cost_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Rerank candidates using Sinkhorn-EMD distance.

    Args:
        query_histogram: Query histogram (1152-dimensional)
        candidate_histograms: Array of candidate histograms (N x 1152)
        batch_size: Size of batches for processing
        reg: Regularization parameter for Sinkhorn
        numItermax: Maximum number of Sinkhorn iterations
        cost_matrix: Pre-computed cost matrix (if None, will be computed)

    Returns:
        Array of EMD distances for each candidate
    """
    if cost_matrix is None:
        cost_matrix = create_cost_matrix()

    # Ensure inputs are correct shape and type
    query_histogram = query_histogram.astype(np.float64)
    candidate_histograms = candidate_histograms.astype(np.float64)

    # Initialize results array
    n_candidates = len(candidate_histograms)
    distances = np.zeros(n_candidates)

    # Process in batches
    for i in range(0, n_candidates, batch_size):
        batch_end = min(i + batch_size, n_candidates)
        batch = candidate_histograms[i:batch_end]

        # Compute EMD for each histogram in batch
        for j, hist in enumerate(batch):
            distances[i + j] = compute_sinkhorn_distance(
                query_histogram, hist, cost_matrix, reg=reg, numItermax=numItermax
            )

    return distances


def rerank_with_confidence(
    query_histogram: np.ndarray,
    candidate_histograms: np.ndarray,
    candidate_ids: List[str],
    batch_size: int = 10,
) -> List[RerankResult]:
    """
    Rerank candidates and compute confidence scores.

    Args:
        query_histogram: Query histogram
        candidate_histograms: Candidate histograms
        candidate_ids: List of candidate image IDs
        batch_size: Size of batches for processing

    Returns:
        List of RerankResult objects with distances and confidence scores
    """
    # Compute EMD distances
    distances = rerank_candidates(
        query_histogram, candidate_histograms, batch_size=batch_size
    )

    # Compute confidence scores (1 / (1 + distance))
    confidence_scores = 1 / (1 + distances)

    # Create results
    results = []
    for i in range(len(candidate_ids)):
        results.append(
            RerankResult(
                image_id=candidate_ids[i],
                distance=float(distances[i]),
                confidence=float(confidence_scores[i]),
            )
        )

    # Sort by distance
    results.sort(key=lambda x: x.distance)
    return results


def sinkhorn_rerank(
    candidates: np.ndarray,
    query: np.ndarray,
    epsilon: float = 0.1,
    max_iter: int = 100,
    min_threshold: float = 1e-10,
) -> np.ndarray:
    """
    Rerank candidates using stabilized Sinkhorn algorithm.
    """
    # Ensure inputs are properly normalized and non-zero
    query = np.maximum(query, min_threshold)
    query = query / query.sum()

    candidates = np.maximum(candidates, min_threshold)
    candidates = candidates / candidates.sum(axis=1, keepdims=True)

    try:
        distances = ot.sinkhorn2(
            query,
            candidates,
            M,
            epsilon,
            numItermax=max_iter,
            stopThr=1e-6,
            verbose=False,
            log=False,
        )
        return distances
    except Exception as e:
        logger.error(f"Sinkhorn algorithm failed: {e}")
        # Fallback to L2 distance if Sinkhorn fails
        return np.linalg.norm(candidates - query, axis=1)
