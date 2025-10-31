"""
Main search module for the Chromatica color search engine.

This module implements the complete two-stage search pipeline that combines:
1. Fast Approximate Nearest Neighbor (ANN) search using FAISS HNSW index
2. High-fidelity reranking using Sinkhorn-approximated Earth Mover's Distance

The search process follows the architecture specified in the critical instructions:
- First stage: Use FAISS to retrieve top-K candidates efficiently
- Second stage: Fetch raw histograms and rerank using Sinkhorn-EMD
- Comprehensive logging for performance monitoring and debugging

Key Components:
- find_similar(): Main search function that orchestrates the entire pipeline
- Performance monitoring with separate timing for each stage
- Error handling and graceful degradation
- Integration with existing histogram generation and reranking modules

This implementation provides the core search functionality for the Chromatica API
and serves as the foundation for the color search engine's query processing.
"""

import logging
import time
import numpy as np
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
from functools import lru_cache
import threading

from .indexing.store import AnnIndex, MetadataStore, hellinger_transform
from .core.rerank import (
    rerank_with_confidence,
    RerankResult,
    get_chroma_mask,
    get_lightness_mask,
    get_hue_proximity_mask,
    get_hue_mixture_mask,
    histogram_top_hex_colors,
    get_query_peak_centers,
    compute_per_peak_hue_masses,
)
from .utils.config import (
    QUERY_SHARPEN_EXPONENT,
    CHROMA_CUTOFF,
    CHROMA_SIGMA,
    RERANK_ALPHA_L1,
    LIGHTNESS_SUPPRESS_THRESHOLD,
    LIGHTNESS_SIGMA,
    HUE_SIGMA,
)
from .utils.config import RERANK_K
from .utils.config import PREFILTER_HUE_MIN_MASS, PREFILTER_MIN_KEEP
from .utils.config import RERANK_BATCH_SIZE
from .utils.config import VERBOSE_SEARCH_LOGS, LOG_TOP_COLORS_N
from .utils.config import PERCOLOR_MIN_MASS, PERCOLOR_TOP_K, PERCOLOR_ENFORCE_STRICT

from .api.models import SearchResult


# Configure logging for this module
logger = logging.getLogger(__name__)


# Global query cache for avoiding repeated computations
_query_cache: Dict[str, List[SearchResult]] = {}
_cache_lock = threading.Lock()
_cache_max_size = 1000  # Maximum number of cached queries

# --- Search Engine Constants (Required for API contract and limiting search) ---
MAX_SEARCH_RESULTS = 200  # As defined in previous project steps
RERANK_K = 500  # Default number of candidates to rerank

# Performance monitoring
_performance_stats = {
    "total_searches": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "avg_ann_time": 0.0,
    "avg_metadata_time": 0.0,
    "avg_rerank_time": 0.0,
    "avg_total_time": 0.0,
}
_performance_lock = threading.Lock()


def _generate_query_hash(
    query_histogram: np.ndarray,
    k: int,
    max_rerank: int,
    use_approximate_reranking: bool,
) -> str:
    """
    Generate a hash for a query to use as cache key.

    Args:
        query_histogram: Query histogram
        k: Number of candidates to retrieve
        max_rerank: Maximum number of candidates to rerank
        use_approximate_reranking: Whether to use approximate reranking

    Returns:
        Hash string for the query
    """
    # Create a hash from the query parameters
    query_data = {
        "histogram": query_histogram.tobytes(),
        "k": k,
        "max_rerank": max_rerank,
        "use_approximate_reranking": use_approximate_reranking,
    }

    # Convert to string and hash
    query_str = str(query_data)
    return hashlib.md5(query_str.encode()).hexdigest()


def _create_search_results(
    image_ids: List[str], distances: np.ndarray, store: MetadataStore, k: int
) -> List[SearchResult]:
    """Helper to create SearchResult objects from image_ids and distances (Fast Mode/Fallback)."""
    results = []

    num_results = min(k, len(image_ids))

    for i in range(num_results):
        image_id = image_ids[i]

        # Get metadata (assuming it contains 'file_path' and 'image_url')
        metadata = store.get_image_info(image_id) or {}

        # Try to extract dominant colors from histogram if available
        # Note: We'll extract from histograms after creating all results
        results.append(
            SearchResult(
                image_id=image_id,
                file_path=metadata.get("file_path", "N/A"),  # <-- NEW
                distance=float(distances[i]),
                rank=i + 1,  # <-- NEW
                ann_score=float(distances[i]),  # <-- NEW (ANN distance for fallback)
                confidence=1 / (1 + float(distances[i])),
                image_url=metadata.get("image_url"),  # <-- NEW
                dominant_colors=[],  # Will be populated from histograms below
                # metadata=metadata or {}, # Re-add if you still use metadata field
            )
        )
    return results


def _get_cached_results(query_hash: str) -> Optional[List[SearchResult]]:
    """
    Get cached results for a query hash.

    Args:
        query_hash: Hash of the query

    Returns:
        Cached results or None if not found
    """
    with _cache_lock:
        return _query_cache.get(query_hash)


def _cache_results(query_hash: str, results: List[SearchResult]) -> None:
    """
    Cache results for a query hash.

    Args:
        query_hash: Hash of the query
        results: Search results to cache
    """
    with _cache_lock:
        # Simple LRU eviction: remove oldest entries if cache is full
        if len(_query_cache) >= _cache_max_size:
            # Remove the first (oldest) entry
            oldest_key = next(iter(_query_cache))
            del _query_cache[oldest_key]
            logger.debug(f"Evicted query {oldest_key} from cache")

        _query_cache[query_hash] = results.copy()
        logger.debug(
            f"Cached query {query_hash} with {len(results)} results (cache size: {len(_query_cache)})"
        )


def _clear_query_cache() -> None:
    """Clear the query cache."""
    with _cache_lock:
        _query_cache.clear()
        logger.info("Query cache cleared")


def _update_performance_stats(
    ann_time: float,
    metadata_time: float,
    rerank_time: float,
    total_time: float,
    cache_hit: bool,
) -> None:
    """
    Update performance statistics.

    Args:
        ann_time: ANN search time in seconds
        metadata_time: Metadata retrieval time in seconds
        rerank_time: Reranking time in seconds
        total_time: Total search time in seconds
        cache_hit: Whether this was a cache hit
    """
    with _performance_lock:
        _performance_stats["total_searches"] += 1

        if cache_hit:
            _performance_stats["cache_hits"] += 1
        else:
            _performance_stats["cache_misses"] += 1

        # Update running averages
        n = _performance_stats["total_searches"]
        _performance_stats["avg_ann_time"] = (
            _performance_stats["avg_ann_time"] * (n - 1) + ann_time
        ) / n
        _performance_stats["avg_metadata_time"] = (
            _performance_stats["avg_metadata_time"] * (n - 1) + metadata_time
        ) / n
        _performance_stats["avg_rerank_time"] = (
            _performance_stats["avg_rerank_time"] * (n - 1) + rerank_time
        ) / n
        _performance_stats["avg_total_time"] = (
            _performance_stats["avg_total_time"] * (n - 1) + total_time
        ) / n


def get_performance_stats() -> Dict[str, Union[int, float]]:
    """
    Get current performance statistics.

    Returns:
        Dictionary containing performance metrics
    """
    with _performance_lock:
        stats = _performance_stats.copy()

        # Calculate cache hit rate
        if stats["total_searches"] > 0:
            stats["cache_hit_rate"] = stats["cache_hits"] / stats["total_searches"]
        else:
            stats["cache_hit_rate"] = 0.0

        return stats


def _get_adaptive_parameters(
    dataset_size: int, query_complexity: float
) -> Dict[str, Union[bool, int, float]]:
    """
    Get adaptive search parameters based on dataset size and query complexity.

    Args:
        dataset_size: Number of images in the dataset
        query_complexity: Measure of query complexity (0.0 to 1.0)

    Returns:
        Dictionary with adaptive parameters
    """
    from ..utils.config import get_adaptive_search_params

    # Get base parameters from dataset size
    base_params = get_adaptive_search_params(dataset_size)

    # Adjust based on query complexity
    if query_complexity > 0.8:
        # Complex queries: use more conservative parameters
        base_params["rerank_k"] = min(base_params["rerank_k"] * 2, dataset_size)
        base_params["use_parallel"] = True
        base_params["streaming_mode"] = False
    elif query_complexity < 0.3:
        # Simple queries: use faster parameters
        base_params["rerank_k"] = max(base_params["rerank_k"] // 2, 10)
        base_params["use_parallel"] = False
        base_params["streaming_mode"] = True

    return base_params


def _apply_query_adjustments(query_hist: np.ndarray) -> np.ndarray:
    """Apply chroma weighting and sharpening to the query histogram for ANN stage.

    - Apply smooth chroma weighting (no hard cutoff - preserve query colors)
    - Sharpen with exponent and renormalize
    """
    # Get smooth chroma weights only (no cutoff) - preserves low-chroma query colors
    mask = get_chroma_mask(cutoff=0.0, sigma=CHROMA_SIGMA)  # No cutoff for query
    q = query_hist.astype(np.float64)
    if mask is not None and mask.shape[0] == q.shape[0]:
        q = q * mask
    if QUERY_SHARPEN_EXPONENT and QUERY_SHARPEN_EXPONENT > 1.0:
        q = np.power(np.maximum(q, 0.0), QUERY_SHARPEN_EXPONENT)
    s = q.sum()
    if s > 0:
        q = q / s
    return q


def find_similar(
    query_histogram: np.ndarray,
    index: AnnIndex,
    store: MetadataStore,
    k: int,
    fast_mode: bool = False,
    max_rerank: int = RERANK_K,  # <-- FIX: Corrected argument name to match the API call
) -> List[SearchResult]:
    """
    Performs the two-stage color search with robust fallback.
    """
    start_time = time.time()

    # 1. Stage 1: Fast ANN Search (Always performed)
    # Use max_rerank (which is RERANK_K) for normal mode candidates
    num_ann_candidates = max_rerank if not fast_mode else k

    # Preprocess query to suppress near-neutrals and sharpen peaks
    preprocessed_query = _apply_query_adjustments(query_histogram)

    # D: distances (Hellinger L2), I: indices (FAISS IDs)
    # NOTE: Do NOT pre-transform here; the index applies Hellinger internally.
    D, I = index.search(preprocessed_query, num_ann_candidates)

    # Check if any results were returned
    if I.size == 0 or I[0].size == 0:
        logger.warning("FAISS search returned no candidates.")
        return []

    ann_candidate_faiss_ids = I.flatten().astype(int)
    ann_distances = D.flatten()
    # -------------------------------------------------------------------------------

    # Map FAISS IDs to image IDs and keep distances aligned; drop any missing IDs
    ann_candidate_ids: List[str] = []
    ann_candidate_distances: List[float] = []
    for faiss_id, dist in zip(ann_candidate_faiss_ids, ann_distances):
        img_id = store.faiss_id_to_image_id(faiss_id)
        if img_id is not None:
            ann_candidate_ids.append(img_id)
            ann_candidate_distances.append(float(dist))

    # Overwrite distances with filtered, aligned list
    ann_distances = np.array(ann_candidate_distances, dtype=float)

    stage1_time = time.time()
    logger.info(
        f"Stage 1 (FAISS) completed in {stage1_time - start_time:.4f}s. "
        f"Retrieved {len(ann_candidate_ids)} candidates."
    )

    # ann_candidate_ids already filtered and aligned with ann_distances

    # 2. Stage 2: Reranking (Only in Normal Search)
    if fast_mode:
        logger.info("Fast Mode: Skipping Stage 2 (Reranking).")
        # Use the helper to create the final, required SearchResult objects
        results = _create_search_results(ann_candidate_ids, ann_distances, store, k)

        # Extract dominant colors from histograms for fast mode results
        if results:
            ids_for_hist = [r.image_id for r in results]
            try:
                hists = store.get_raw_histograms(ids_for_hist)
                logger.info(f"Retrieved {len(hists)} histograms for fast mode results")
                # Update results with dominant colors extracted from histograms
                for r, h in zip(results, hists):
                    if h is not None and h.sum() > 0:
                        tops = histogram_top_hex_colors(h, top_n=LOG_TOP_COLORS_N)
                        r.dominant_colors = tops if tops else []  # Always set, even if empty
                        logger.info(
                            f"Fast result: id={r.image_id}, d={r.distance:.3f}, "
                            f"hist_sum={h.sum():.6f}, colors={len(tops) if tops else 0}: {tops}"
                        )
                    else:
                        hist_sum = h.sum() if h is not None else 0
                        logger.warning(
                            f"No valid histogram for fast mode result: {r.image_id}, hist_sum={hist_sum:.6f}"
                        )
                        r.dominant_colors = []  # Explicitly set empty list
            except Exception as e:
                logger.error(f"Error extracting dominant colors in fast mode: {e}", exc_info=True)
                # Ensure all results have dominant_colors set
                for r in results:
                    if not hasattr(r, 'dominant_colors') or r.dominant_colors is None:
                        r.dominant_colors = []

        return results

    else:
        logger.info(
            f"Normal Search: Starting Stage 2 (Reranking {len(ann_candidate_ids)} candidates)."
        )

        # --- FIX: ROBUST TRY/EXCEPT BLOCK FOR FAILURE ---
        try:
            # 2.1 CRITICAL: Retrieve RAW histograms from DuckDB
            # raw_histograms = store.get_histogram(ann_candidate_ids)

            # Retrieve raw histograms from the provided metadata store
            raw_histograms = store.get_raw_histograms(ann_candidate_ids)
            
            logger.info(f"Retrieved {len(raw_histograms)} histograms from store")
            
            # Check histogram validity
            histogram_stats = []
            for i, hist in enumerate(raw_histograms):
                if hist is not None:
                    hist_sum = np.sum(hist)
                    hist_max = np.max(hist) if hist.size > 0 else 0
                    histogram_stats.append({
                        'index': i,
                        'id': ann_candidate_ids[i],
                        'sum': hist_sum,
                        'max': hist_max,
                        'shape': hist.shape
                    })
                    if hist_sum <= 0:
                        logger.warning(
                            f"Histogram {i} ({ann_candidate_ids[i]}) has zero sum: "
                            f"sum={hist_sum:.6f}, max={hist_max:.6f}, shape={hist.shape}"
                        )
            
            # Filter out missing or empty histograms (sum == 0 implies not loaded)
            valid_indices = [
                i
                for i, hist in enumerate(raw_histograms)
                if hist is not None and np.sum(hist) > 0
            ]
            valid_ids = [ann_candidate_ids[i] for i in valid_indices]
            valid_hists = np.array([raw_histograms[i] for i in valid_indices])
            
            logger.info(
                f"Valid histograms: {len(valid_hists)}/{len(raw_histograms)} "
                f"(filtered out {len(raw_histograms) - len(valid_hists)} with zero sum)"
            )

            # FIX: If your database fails, raw_histograms might be None or empty. Handle it defensively.
            if raw_histograms is None or raw_histograms.size == 0:
                logger.error(
                    "Reranking failed: No histograms retrieved. Falling back to FAISS results."
                )
                # Return the results from the first stage (if they exist)
                return _create_search_results(
                    ann_candidate_ids, ann_distances, store, k
                )

            if len(valid_hists) == 0:
                raise RuntimeError(
                    "DuckDB returned no valid raw histograms for reranking."
                )

            logger.info(
                f"Successfully retrieved {len(valid_hists)} raw histograms for reranking."
            )

            # 2.1.1 Enforce color presence: filter by hue mass under query hue mixture
            hue_window = get_hue_mixture_mask(query_histogram)
            hw = hue_window.astype(np.float64)
            hw_sum = hw.sum()
            if hw_sum > 0:
                hw = hw / hw_sum
            hue_masses = np.dot(valid_hists, hw)
            keep_mask = hue_masses >= float(PREFILTER_HUE_MIN_MASS)
            kept_ids = [vid for vid, km in zip(valid_ids, keep_mask) if km]
            kept_hists = np.array([vh for vh, km in zip(valid_hists, keep_mask) if km])

            # Additional per-color presence requirement
            peak_centers = get_query_peak_centers(query_histogram, top_k=PERCOLOR_TOP_K)
            if peak_centers.size > 0:
                percolor_masses = [
                    compute_per_peak_hue_masses(hist, peak_centers, HUE_SIGMA)
                    for hist in kept_hists
                ]
                percolor_keep = [
                    bool((m >= float(PERCOLOR_MIN_MASS)).all()) for m in percolor_masses
                ]
                if any(percolor_keep):
                    # Keep only candidates passing all per-peak thresholds
                    if not all(percolor_keep):
                        logger.info(
                            f"Per-color filter: kept {sum(percolor_keep)} / {len(kept_ids)} (min_mass={PERCOLOR_MIN_MASS})."
                        )
                    kept_ids = [cid for cid, ok in zip(kept_ids, percolor_keep) if ok]
                    kept_hists = np.array(
                        [h for h, ok in zip(kept_hists, percolor_keep) if ok]
                    )
                else:
                    # No strict matches; choose best by summed per-peak mass if enforcement enabled
                    sums = (
                        np.array([float(m.sum()) for m in percolor_masses])
                        if len(percolor_masses)
                        else np.zeros(0)
                    )
                    if PERCOLOR_ENFORCE_STRICT and sums.size > 0:
                        order = np.argsort(-sums)
                        top_n = min(max(k, PREFILTER_MIN_KEEP), len(kept_ids))
                        kept_ids = [kept_ids[i] for i in order[:top_n]]
                        kept_hists = np.array([kept_hists[i] for i in order[:top_n]])
                        logger.info(
                            "Per-color filter: no strict matches; selected top by per-peak mass (strict fallback)."
                        )
            # If too few remain, keep top-N by hue mass
            if len(kept_ids) < max(k, PREFILTER_MIN_KEEP) and len(valid_ids) > 0:
                order = np.argsort(-hue_masses)
                top_n = min(max(k, PREFILTER_MIN_KEEP), len(valid_ids))
                kept_ids = [valid_ids[i] for i in order[:top_n]]
                kept_hists = np.array([valid_hists[i] for i in order[:top_n]])
                logger.info(
                    f"Hue prefilter kept top {len(kept_ids)} by hue mass (threshold too strict)."
                )
            else:
                logger.info(
                    f"Hue prefilter: kept {len(kept_ids)} / {len(valid_ids)} candidates (min_mass={PREFILTER_HUE_MIN_MASS})."
                )

            # 2.2 Rerank candidates using Sinkhorn-EMD with confidence scores
            chroma_mask = get_chroma_mask()
            lightness_mask = get_lightness_mask()
            # Use multi-peak hue emphasis for better adherence to query colors
            hue_mask = get_hue_mixture_mask(query_histogram)
            combined_mask = chroma_mask * lightness_mask * hue_mask
            rerank_results = rerank_with_confidence(
                query_histogram=query_histogram,
                candidate_histograms=kept_hists,
                candidate_ids=kept_ids,
                batch_size=RERANK_BATCH_SIZE,
                chroma_mask=combined_mask,
                alpha_l1=RERANK_ALPHA_L1,
            )

            rerank_time = time.time()
            logger.info(
                f"Stage 2 (Sinkhorn) completed in {rerank_time - stage1_time:.4f}s. "
            )

            # 2.3 Format and return top k results
            final_results = []

            # Create a dictionary mapping candidate image_id to its initial ANN distance
            ann_scores_map = dict(zip(ann_candidate_ids, ann_distances))

            # Build maps for verbose logs
            # Ensure kept_hists is properly indexed - if it's a 2D array, each row is a histogram
            id_to_hist = {}
            logger.info(
                f"Building id_to_hist from {len(kept_ids)} IDs and kept_hists shape: {kept_hists.shape}"
            )

            for idx, cid in enumerate(kept_ids):
                if idx < len(kept_hists):
                    # Extract the histogram row correctly
                    if len(kept_hists.shape) == 2:
                        hist_row = kept_hists[idx].copy()
                        # Ensure it's a 1D array
                        if len(hist_row.shape) > 1:
                            hist_row = hist_row.flatten()
                    elif len(kept_hists.shape) == 1 and idx == 0:
                        # Single histogram case
                        hist_row = kept_hists.copy()
                    else:
                        logger.warning(
                            f"Unexpected histogram shape for index {idx}: {kept_hists.shape}"
                        )
                        hist_row = np.zeros_like(query_histogram)

                    # Validate histogram
                    hist_sum = hist_row.sum() if hist_row is not None else 0
                    hist_max = hist_row.max() if hist_row is not None else 0
                    hist_min = hist_row.min() if hist_row is not None else 0

                    if hist_sum <= 0:
                        logger.warning(
                            f"Histogram for {cid} has zero or negative sum: {hist_sum:.6f}, "
                            f"min={hist_min:.6f}, max={hist_max:.6f}, shape={hist_row.shape}"
                        )
                    else:
                        logger.debug(
                            f"Mapped {cid} -> histogram with sum={hist_sum:.6f}, "
                            f"min={hist_min:.6f}, max={hist_max:.6f}, shape={hist_row.shape}"
                        )

                    id_to_hist[cid] = hist_row
                else:
                    logger.warning(
                        f"Index {idx} out of range for kept_hists (length: {len(kept_hists)})"
                    )
                    id_to_hist[cid] = np.zeros_like(query_histogram)

            id_to_hue_mass = {cid: float(m) for cid, m in zip(valid_ids, hue_masses)}

            for i, result in enumerate(rerank_results):
                if i >= k:
                    break

                metadata = store.get_image_info(result.image_id) or {}

                # 💡 NEW: Calculate dominant colors from the raw histogram 💡
                # Get histogram for this result - try multiple sources
                hist = None
                
                # Method 1: Get from id_to_hist map (from kept_hists)
                if result.image_id in id_to_hist:
                    hist = id_to_hist[result.image_id]
                    hist_sum_check = hist.sum() if hist is not None else 0
                    logger.debug(
                        f"Found histogram in id_to_hist for image_id: {result.image_id}, "
                        f"sum={hist_sum_check:.6f}"
                    )
                
                # Method 2: If not found, fetch directly from store
                if hist is None or (hist is not None and hist.sum() <= 0):
                    logger.warning(
                        f"Histogram missing or invalid in id_to_hist for image_id: {result.image_id}, "
                        f"fetching from store..."
                    )
                    try:
                        hist_arrays = store.get_raw_histograms([result.image_id])
                        if (
                            hist_arrays is not None
                            and len(hist_arrays) > 0
                            and hist_arrays[0] is not None
                        ):
                            # Get the first histogram (should be the only one for single image_id)
                            fetched_hist = hist_arrays[0]
                            # Ensure it's 1D
                            if len(fetched_hist.shape) > 1:
                                fetched_hist = fetched_hist.flatten()
                            
                            hist_sum = fetched_hist.sum() if fetched_hist is not None else 0
                            logger.info(
                                f"Retrieved histogram from store for image_id: {result.image_id}, "
                                f"sum={hist_sum:.6f}, shape={fetched_hist.shape if fetched_hist is not None else None}"
                            )
                            
                            if hist_sum > 0:
                                hist = fetched_hist
                            else:
                                logger.warning(
                                    f"Retrieved histogram has zero sum for image_id: {result.image_id}"
                                )
                        else:
                            logger.warning(
                                f"Could not retrieve histogram from store for image_id: {result.image_id}, "
                                f"hist_arrays: {hist_arrays}"
                            )
                    except Exception as e:
                        logger.error(f"Error retrieving histogram from store: {e}", exc_info=True)
                
                # Method 3: Final fallback - use zeros (will trigger file extraction)
                if hist is None or (hist is not None and hist.sum() <= 0):
                    logger.warning(
                        f"No valid histogram found for image_id: {result.image_id}, "
                        f"using zeros as fallback (will try file extraction)"
                    )
                    hist = np.zeros_like(query_histogram)

                # Ensure histogram is valid before extracting colors
                hist_sum = hist.sum() if hist is not None else 0
                hist_shape = hist.shape if hist is not None else None

                logger.info(
                    f"Extracting dominant colors for image_id: {result.image_id}, "
                    f"hist_sum={hist_sum:.6f}, hist_shape={hist_shape}"
                )

                tops = histogram_top_hex_colors(
                    hist,
                    top_n=LOG_TOP_COLORS_N,
                )

                # Log extraction result for debugging
                logger.info(
                    f"Extracted {len(tops)} dominant colors for image_id: {result.image_id}, "
                    f"hist_sum={hist_sum:.6f}, colors: {tops}"
                )

                if not tops:
                    logger.warning(
                        f"No dominant colors extracted for image_id: {result.image_id}, "
                        f"hist_sum={hist.sum() if hist is not None else 0:.6f}, "
                        f"hist_shape={hist.shape if hist is not None else 'None'}, "
                        f"hist_min={hist.min() if hist is not None else 'None'}, "
                        f"hist_max={hist.max() if hist is not None else 'None'}"
                    )
                    
                    # Try multiple fallback strategies
                    # 1. Try metadata dominant colors first
                    metadata_dominant_colors = metadata.get("dominant_colors", [])
                    if metadata_dominant_colors and isinstance(
                        metadata_dominant_colors, list
                    ):
                        tops = metadata_dominant_colors[:LOG_TOP_COLORS_N]
                        logger.info(
                            f"Using metadata dominant colors as fallback: {tops}"
                        )
                    # 2. If still empty, try extracting from image file (slow but works)
                    elif metadata.get("file_path") and Path(metadata.get("file_path")).exists():
                        try:
                            from ..api.visualization_3d import extract_dominant_colors as extract_from_file
                            file_path = metadata.get("file_path")
                            logger.info(f"Attempting to extract colors from file: {file_path}")
                            file_colors = extract_from_file(file_path, num_colors=LOG_TOP_COLORS_N)
                            if file_colors:
                                tops = file_colors
                                logger.info(f"Extracted {len(tops)} colors from image file: {tops}")
                        except Exception as e:
                            logger.warning(f"Failed to extract colors from file: {e}")
                    else:
                        logger.warning(
                            f"No fallback available for image_id: {result.image_id}, "
                            f"file_path: {metadata.get('file_path')}"
                        )

                # Ensure tops is always a list, never None
                if tops is None:
                    tops = []
                elif not isinstance(tops, list):
                    logger.warning(f"tops is not a list: {type(tops)}, converting to list")
                    tops = list(tops) if tops else []
                
                final_results.append(
                    SearchResult(
                        image_id=result.image_id,
                        file_path=metadata.get("file_path", "N/A"),
                        distance=result.distance,
                        rank=i + 1,
                        ann_score=float(
                            ann_scores_map.get(result.image_id, 0.0)
                        ),  # Get the original score
                        confidence=result.confidence,
                        image_url=metadata.get("image_url"),
                        dominant_colors=tops,  # Always a list, never None
                        # metadata=metadata or {}, # Re-add if needed
                    )
                )
                
                # Verify the SearchResult was created correctly
                if not hasattr(final_results[-1], 'dominant_colors') or final_results[-1].dominant_colors is None:
                    logger.error(f"SearchResult for {result.image_id} missing dominant_colors field!")
                    final_results[-1].dominant_colors = []

                if VERBOSE_SEARCH_LOGS:
                    tops = histogram_top_hex_colors(
                        id_to_hist.get(result.image_id, np.zeros_like(query_histogram)),
                        top_n=LOG_TOP_COLORS_N,
                    )
                    hm = id_to_hue_mass.get(result.image_id, 0.0)
                    percolor = []
                    if peak_centers.size > 0:
                        masses = compute_per_peak_hue_masses(
                            id_to_hist.get(
                                result.image_id, np.zeros_like(query_histogram)
                            ),
                            peak_centers,
                            HUE_SIGMA,
                        )
                        percolor = [round(float(x), 3) for x in masses.tolist()]
                    logger.info(
                        f"Reranked result: id={result.image_id}, d={result.distance:.3f}, ann={ann_scores_map.get(result.image_id, 0.0):.3f}, hue_mass={hm:.3f}, percolor={percolor}, tops={tops}"
                    )

            return final_results

        except Exception as e:
            # GRACEFUL FALLBACK: If anything fails (DuckDB or Sinkhorn), use Stage 1 results
            logger.error(
                f"Normal search (Reranking stage) FAILED: {type(e).__name__}: {e}. "
                f"FALLING BACK TO FAST MODE RESULTS."
            )
            # Use the entire set of candidates retrieved from FAISS for the fallback
            return _create_search_results(ann_candidate_ids, ann_distances, store, k)


def validate_search_system(index: AnnIndex, store: MetadataStore) -> bool:
    """
    Validate the complete search system by running a test search.

    This function performs a comprehensive validation of the search pipeline
    by running a test search with synthetic data. It verifies that all
    components work together correctly and provides performance benchmarks.

    Args:
        index: FAISS HNSW index instance
        store: DuckDB metadata store instance

    Returns:
        bool: True if validation passes, False otherwise

    Example:
        >>> from chromatica.indexing.store import AnnIndex, MetadataStore
        >>> index = AnnIndex()
        >>> store = MetadataStore("test.db")
        >>> is_valid = validate_search_system(index, store)
        >>> print(f"Search system validation: {'PASSED' if is_valid else 'FAILED'}")
    """
    logger.info("Validating complete search system...")

    try:
        # Check if we have data to search
        if index.get_total_vectors() == 0:
            logger.warning("Index is empty, cannot perform validation search")
            return False

        if store.get_image_count() == 0:
            logger.warning("Metadata store is empty, cannot perform validation search")
            return False

        # Create a synthetic query histogram
        np.random.seed(42)  # For reproducible results
        query_hist = np.random.random(1152).astype(np.float64)
        query_hist = query_hist / query_hist.sum()  # Normalize

        logger.info(
            f"Running validation search with {index.get_total_vectors()} indexed vectors"
        )

        # Perform test search
        results = find_similar(
            query_histogram=query_hist,
            k=min(50, index.get_total_vectors()),
            max_rerank=min(200, index.get_total_vectors()),
            index=index,
            store=store,
            fast_mode=False,
        )

        # Validate results
        if not results:
            logger.error("Validation search returned no results")
            return False

        # Check result structure - Use dot notation for dataclass access
        for result in results:
            try:
                # Check if required attributes are present (implicitly)
                _ = result.image_id
                _ = result.distance
            except AttributeError:
                logger.error("Search result missing required attributes")
                return False

        # Check ranking consistency
        distances = [r.distance for r in results]

        if distances != sorted(distances):
            logger.error("Search results not properly sorted by distance")
            return False

        logger.info(f"Search system validation PASSED: {len(results)} results returned")
        return True

    except Exception as e:
        logger.error(f"Search system validation failed: {e}")
        return False


# Auto-validate the system when module is imported
if __name__ == "__main__":
    # Set up logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("🔍 Chromatica Search Module")
    print("=" * 50)
    print("This module provides the complete two-stage search pipeline.")
    print("Use find_similar() to search for similar images.")
    print("Use validate_search_system() to test the complete pipeline.")
    print("=" * 50)
else:
    # Log module import
    logger.debug("Search module imported successfully")
