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
from skimage import color as skcolor

logger = logging.getLogger(__name__)

try:
    from ..utils.config import L_BINS, A_BINS, B_BINS, TOTAL_BINS, LAB_RANGES 
    from ..utils.config import CHROMA_CUTOFF, CHROMA_SIGMA, RERANK_ALPHA_L1
    from ..utils.config import SINKHORN_EPSILON, SINKHORN_MAX_ITER
    from ..utils.config import LIGHTNESS_SUPPRESS_THRESHOLD, LIGHTNESS_SIGMA, HUE_SIGMA
    from ..utils.config import HUE_MIXTURE_TOP_K, HUE_MIXTURE_GAIN
    from ..utils.config import M
except ImportError:
    # Fallback/placeholder definitions if config is not available, which you should remove if unnecessary.
    L_BINS = 8
    A_BINS = 12
    B_BINS = 12
    TOTAL_BINS = L_BINS * A_BINS * B_BINS  # 1152
    LAB_RANGES = [[0, 100], [-86, 98], [-108, 95]]
    CHROMA_CUTOFF = 10.0
    CHROMA_SIGMA = 25.0
    RERANK_ALPHA_L1 = 0.15
    LIGHTNESS_SUPPRESS_THRESHOLD = 85.0
    LIGHTNESS_SIGMA = 8.0
    HUE_SIGMA = 20.0

@dataclass
class RerankResult:
    """Result from reranking a single candidate."""

    image_id: str
    file_path: str
    distance: float
    rank: int
    ann_score: float  # Original ANN distance score
    confidence: float = 1.0
    image_url: Optional[str] = None  # URL of the image

# --- Add this helper function immediately before 'create_cost_matrix' ---
def _get_lab_bin_centers() -> Optional[np.ndarray]:
    """
    Calculates the 3D Lab coordinates for the center of each of the 1152 bins.
    The order must match the histogram generation order (L -> A -> B).
    """
    try:
        # Assuming LAB_RANGES is correctly imported from config.py
        l_min, l_max = LAB_RANGES[0]
        a_min, a_max = LAB_RANGES[1]
        b_min, b_max = LAB_RANGES[2]

        # Calculate bin centers for each dimension
        l_centers = np.linspace(l_min, l_max, L_BINS + 1)[:-1] + (l_max - l_min) / (2 * L_BINS)
        a_centers = np.linspace(a_min, a_max, A_BINS + 1)[:-1] + (a_max - a_min) / (2 * A_BINS)
        b_centers = np.linspace(b_min, b_max, B_BINS + 1)[:-1] + (b_max - b_min) / (2 * B_BINS)

        # Create the 3D grid and flatten it, ensuring the flattening order 
        # matches the build_histogram function (B_BINS is the fastest changing index)
        L, A, B = np.meshgrid(l_centers, a_centers, b_centers, indexing='ij')
        
        # Stack and reshape into an array of (TOTAL_BINS, 3) where each row is [L, a, b]
        lab_centers = np.stack([L.ravel(), A.ravel(), B.ravel()], axis=1)

        if lab_centers.shape != (TOTAL_BINS, 3):
            logger.error(f"Bin center shape mismatch: {lab_centers.shape}")
            return None
            
        return lab_centers
    except Exception as e:
        logger.error(f"Error generating Lab bin centers: {e}")
        return None

_CHROMA_MASK_CACHE: dict = {}
_LIGHTNESS_MASK_CACHE: dict = {}
# Module-level cached cost matrix to avoid recomputation on every search
_CACHED_COST_MATRIX: Optional[np.ndarray] = None

def get_chroma_mask(cutoff: float = CHROMA_CUTOFF, sigma: float = CHROMA_SIGMA) -> np.ndarray:
    """
    Per-bin chroma weights based on bin centers.
    - Set bins with C* < cutoff to 0 (neutral suppression)
    - Others weighted by 1 - exp(-(C*^2)/(2*sigma^2))
    """
    key = (float(cutoff), float(sigma))
    if key in _CHROMA_MASK_CACHE:
        return _CHROMA_MASK_CACHE[key]

    centers = _get_lab_bin_centers()
    if centers is None:
        mask = np.ones(TOTAL_BINS, dtype=np.float64)
        _CHROMA_MASK_CACHE[key] = mask
        return mask

    a = centers[:, 1]
    b = centers[:, 2]
    chroma = np.sqrt(a * a + b * b)
    weight = 1.0 - np.exp(-(chroma * chroma) / (2.0 * (sigma ** 2)))
    weight[chroma < cutoff] = 0.0
    mask = weight.astype(np.float64)
    _CHROMA_MASK_CACHE[key] = mask
    return mask

def get_lightness_mask(threshold: float = LIGHTNESS_SUPPRESS_THRESHOLD, sigma_l: float = LIGHTNESS_SIGMA) -> np.ndarray:
    """Down-weight very bright bins: for L* > threshold, weight = exp(-((L-thr)^2)/(2*sigma_l^2))."""
    key = (float(threshold), float(sigma_l))
    if key in _LIGHTNESS_MASK_CACHE:
        return _LIGHTNESS_MASK_CACHE[key]

    centers = _get_lab_bin_centers()
    if centers is None:
        mask = np.ones(TOTAL_BINS, dtype=np.float64)
        _LIGHTNESS_MASK_CACHE[key] = mask
        return mask

    L_vals = centers[:, 0]
    mask = np.ones_like(L_vals, dtype=np.float64)
    over = np.maximum(L_vals - threshold, 0.0)
    # weights in (0,1], approach 0 as L gets very large above threshold
    mask = np.exp(-(over * over) / (2.0 * (sigma_l ** 2)))
    _LIGHTNESS_MASK_CACHE[key] = mask
    return mask

def get_hue_proximity_mask(query_hist: np.ndarray, sigma_h: float = HUE_SIGMA) -> np.ndarray:
    """
    Weight bins by proximity in (a*, b*) to the query's mean (a*, b*).
    This is a simple unimodal emphasis; sufficient for most 1–2 color queries.
    """
    centers = _get_lab_bin_centers()
    if centers is None:
        return np.ones(TOTAL_BINS, dtype=np.float64)

    q = np.asarray(query_hist, dtype=np.float64).flatten()
    if q.sum() <= 0:
        return np.ones(TOTAL_BINS, dtype=np.float64)

    q = q / q.sum()
    a_bins = centers[:, 1]
    b_bins = centers[:, 2]
    a0 = float(np.dot(q, a_bins))
    b0 = float(np.dot(q, b_bins))

    da = a_bins - a0
    db = b_bins - b0
    d2 = da * da + db * db
    mask = np.exp(-(d2) / (2.0 * (sigma_h ** 2)))
    return mask.astype(np.float64)

def get_hue_mixture_mask(
    query_hist: np.ndarray,
    sigma_h: float = HUE_SIGMA,
    top_k: int = HUE_MIXTURE_TOP_K,
    gain: float = HUE_MIXTURE_GAIN,
) -> np.ndarray:
    """Emphasize multiple hue peaks using a Gaussian mixture over (a*, b*).

    This better preserves multi-color queries by centering Gaussians on the
    top-K query bins (by mass) rather than a single mean.
    """
    centers = _get_lab_bin_centers()
    if centers is None:
        return np.ones(TOTAL_BINS, dtype=np.float64)

    q = np.asarray(query_hist, dtype=np.float64).flatten()
    if q.sum() <= 0:
        return np.ones(TOTAL_BINS, dtype=np.float64)
    q = q / q.sum()

    # Select top-K bins by probability mass
    top_indices = np.argpartition(q, -int(max(1, top_k)))[-int(max(1, top_k)):]
    a_bins = centers[:, 1]
    b_bins = centers[:, 2]

    mix = np.zeros(TOTAL_BINS, dtype=np.float64)
    for idx in top_indices:
        a0 = float(centers[idx, 1])
        b0 = float(centers[idx, 2])
        da = a_bins - a0
        db = b_bins - b0
        d2 = da * da + db * db
        mix += np.exp(-(d2) / (2.0 * (sigma_h ** 2))) * float(q[idx])

    # Normalize and apply gain; keep within [0,1]
    if mix.max() > 0:
        mix = mix / mix.max()
    mix = np.clip(mix * float(gain), 0.0, 1.0)
    return mix.astype(np.float64)


def histogram_top_hex_colors(hist: np.ndarray, top_n: int = 5) -> List[str]:
    """Return top-N approximate hex colors from a histogram using bin centers.

    This avoids decoding images at runtime by mapping peak bins to their Lab
    centers and converting to sRGB.
    """
    try:
        hist = np.asarray(hist, dtype=np.float64).flatten()
        if hist.sum() <= 0 or top_n <= 0:
            return []
        centers = _get_lab_bin_centers()
        if centers is None:
            return []
        n = int(max(1, top_n))
        idx = np.argpartition(hist, -n)[-n:]
        # Sort by probability descending
        idx = idx[np.argsort(-hist[idx])]
        lab = centers[idx]
        # skimage expects Lab with L in [0,100], a,b in roughly [-128,127]
        rgb = skcolor.lab2rgb(lab.reshape(1, n, 3)).reshape(n, 3)
        rgb = np.clip(np.round(rgb * 255), 0, 255).astype(np.uint8)
        return [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in rgb]
    except Exception:
        return []


def get_query_peak_centers(
    query_hist: np.ndarray,
    top_k: int,
) -> np.ndarray:
    """Return Lab centers (a*, b*) for the top-K bins in the query histogram."""
    centers = _get_lab_bin_centers()
    if centers is None:
        return np.zeros((0, 2), dtype=np.float64)
    q = np.asarray(query_hist, dtype=np.float64).flatten()
    if q.sum() <= 0:
        return np.zeros((0, 2), dtype=np.float64)
    k = int(max(1, top_k))
    idx = np.argpartition(q, -k)[-k:]
    idx = idx[np.argsort(-q[idx])]
    return centers[idx][:, 1:3]  # a*, b*


def compute_per_peak_hue_masses(
    candidate_hist: np.ndarray,
    peak_centers_ab: np.ndarray,
    sigma_h: float,
) -> np.ndarray:
    """Compute mass near each query peak using Gaussian windows around (a*, b*)."""
    centers = _get_lab_bin_centers()
    if centers is None or peak_centers_ab.size == 0:
        return np.zeros(0, dtype=np.float64)
    a_bins = centers[:, 1]
    b_bins = centers[:, 2]
    h = np.asarray(candidate_hist, dtype=np.float64).flatten()
    masses: List[float] = []
    for a0, b0 in peak_centers_ab:
        da = a_bins - float(a0)
        db = b_bins - float(b0)
        w = np.exp(-(da * da + db * db) / (2.0 * (sigma_h ** 2)))
        w = w / max(w.sum(), 1e-12)
        masses.append(float(np.dot(h, w)))
    return np.asarray(masses, dtype=np.float64)

def create_cost_matrix(bins: int = TOTAL_BINS) -> np.ndarray:
    """
    Create the perceptually accurate cost matrix for EMD computation 
    using the squared Euclidean distance (L2) between CIE Lab bin centers.
    """
    # Try to calculate the accurate 3D Lab cost matrix
    lab_centers = _get_lab_bin_centers()
    
    if lab_centers is None:
        logger.warning("Using fallback index distance cost matrix (less accurate).")
        # FALLBACK: Original simple index distance
        idx = np.arange(bins)
        cost_matrix = (idx[:, None] - idx[None, :]) ** 2
    else:
        logger.info("Using ACCURATE 3D Lab Euclidean distance for EMD cost matrix.")
        # C[i, j] = ||center_i - center_j||^2. Computed efficiently.
        c_i_sq = np.sum(lab_centers ** 2, axis=1, keepdims=True)
        c_j_sq = np.sum(lab_centers ** 2, axis=1, keepdims=False)
        
        cost_matrix = c_i_sq + c_j_sq - 2 * np.dot(lab_centers, lab_centers.T)
        cost_matrix = np.maximum(cost_matrix, 0.0) # Ensure non-negative

    # Normalization for numerical stability in Sinkhorn
    max_cost = cost_matrix.max()
    if max_cost > 0:
        cost_matrix = cost_matrix / max_cost
    
    return cost_matrix


def get_cost_matrix(bins: int = TOTAL_BINS) -> np.ndarray:
    """Return a cached EMD cost matrix, computing it once per process."""
    global _CACHED_COST_MATRIX
    if _CACHED_COST_MATRIX is not None and _CACHED_COST_MATRIX.shape == (bins, bins):
        return _CACHED_COST_MATRIX
    _CACHED_COST_MATRIX = create_cost_matrix(bins)
    return _CACHED_COST_MATRIX


def compute_sinkhorn_distance(
    hist1: np.ndarray,
    hist2: np.ndarray,
    cost_matrix: Optional[np.ndarray] = None,
    reg: float = 0.1,
    numItermax: int = 100,
    min_threshold: float = 1e-10,
    chroma_mask: Optional[np.ndarray] = None,
    alpha_l1: float = 0.0,
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

        # Apply chroma mask if provided; then re-normalize
        if chroma_mask is not None and chroma_mask.shape[0] == hist1.shape[0]:
            hist1 = np.maximum(hist1 * chroma_mask, min_threshold)
            hist2 = np.maximum(hist2 * chroma_mask, min_threshold)
            hist1 = hist1 / hist1.sum()
            hist2 = hist2 / hist2.sum()

        # Create or use cost matrix
        if cost_matrix is None:
            cost_matrix = get_cost_matrix(len(hist1))

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
            base = float(distance)
            if alpha_l1 > 0.0:
                base += float(alpha_l1) * float(np.sum(np.abs(hist1 - hist2)))
            return base
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
            base = float(distance)
            if alpha_l1 > 0.0:
                base += float(alpha_l1) * float(np.sum(np.abs(hist1 - hist2)))
            return base

    except Exception as e:
        logger.error(f"All Sinkhorn methods failed: {e}, falling back to L2")
        base = float(np.sqrt(np.sum((hist1 - hist2) ** 2)))
        if alpha_l1 > 0.0:
            base += float(alpha_l1) * float(np.sum(np.abs(hist1 - hist2)))
        return base


def rerank_candidates(
    query_histogram: np.ndarray,
    candidate_histograms: np.ndarray,
    batch_size: int = 10,
    reg: float = SINKHORN_EPSILON,
    numItermax: int = SINKHORN_MAX_ITER,
    cost_matrix: Optional[np.ndarray] = None,
    chroma_mask: Optional[np.ndarray] = None,
    alpha_l1: float = 0.0,
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
        cost_matrix = get_cost_matrix()

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
                query_histogram, hist, cost_matrix, reg=reg, numItermax=numItermax,
                chroma_mask=chroma_mask, alpha_l1=alpha_l1
            )

    return distances


def rerank_with_confidence(
    query_histogram: np.ndarray,
    candidate_histograms: np.ndarray,
    candidate_ids: List[str],
    batch_size: int = 10,
    chroma_mask: Optional[np.ndarray] = None,
    alpha_l1: float = RERANK_ALPHA_L1,
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
        query_histogram, candidate_histograms, batch_size=batch_size,
        chroma_mask=chroma_mask, alpha_l1=alpha_l1
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
