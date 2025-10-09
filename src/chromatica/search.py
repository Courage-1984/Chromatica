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
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
from functools import lru_cache
import threading

from .indexing.store import AnnIndex, MetadataStore
from .core.rerank import rerank_candidates, RerankResult
from .utils.config import RERANK_K

# Configure logging for this module
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result of a complete search operation for a single image."""

    image_id: str
    file_path: str
    distance: float
    rank: int
    ann_score: float  # Original ANN distance score


# Global query cache for avoiding repeated computations
_query_cache: Dict[str, List[SearchResult]] = {}
_cache_lock = threading.Lock()
_cache_max_size = 1000  # Maximum number of cached queries

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


def find_similar(
    query_histogram: np.ndarray,
    k: int,
    max_rerank: int,
    index: AnnIndex,
    store: MetadataStore,
    fast_mode: bool = False,
) -> List[Dict[str, Any]]:
    """
    Find similar images using the two-stage search pipeline.

    Args:
        query_histogram: Query histogram (1152-dimensional)
        k: Number of results to return
        max_rerank: Number of candidates to rerank
        index: FAISS index instance
        store: Metadata store instance
        fast_mode: Whether to use fast mode (skip reranking)

    Returns:
        List of result dictionaries, each containing:
        - image_id: Unique image identifier
        - file_path: Path to the image file
        - distance: Distance metric value
        - confidence: Confidence score (0-1)
        - ann_distance: Original ANN search distance
    """
    try:
        start_time = time.time()
        logger.info(f"Starting search with k={k} and max_rerank={max_rerank}")

        # First stage: ANN search
        ann_start = time.time()
        distances, indices = index.search(query_histogram, max_rerank)
        ann_time = time.time() - ann_start

        # Filter out invalid indices (FAISS returns -1 for empty slots)
        valid_mask = indices >= 0
        valid_indices = indices[valid_mask]
        valid_distances = distances[valid_mask]

        if len(valid_indices) == 0:
            logger.warning("No valid indices returned from FAISS search")
            return []

        # Convert indices to image IDs (zero-padded format)
        image_ids = [f"{int(idx):05d}" for idx in valid_indices]

        logger.info(
            f"Found {len(image_ids)} valid candidates out of {len(indices)} results"
        )

        # Get histograms and metadata for candidates
        metadata_start = time.time()
        candidate_histograms = []
        valid_candidates = []

        for i, image_id in enumerate(image_ids):
            hist = store.get_histogram(image_id)
            info = store.get_image_info(image_id)

            if hist is not None and info is not None and "file_path" in info:
                candidate_histograms.append(hist)
                valid_candidates.append(
                    {
                        "image_id": image_id,
                        "file_path": info["file_path"],
                        "ann_distance": float(valid_distances[i]),
                    }
                )
            else:
                logger.debug(
                    f"Skipping candidate {image_id} due to missing histogram or metadata"
                )

        metadata_time = time.time() - metadata_start

        if not valid_candidates:
            logger.warning("No valid candidates found")
            return []

        # Second stage: Reranking (unless in fast mode)
        rerank_start = time.time()
        if not fast_mode and len(candidate_histograms) > 0:
            candidate_histograms = np.array(candidate_histograms)
            reranked_distances = rerank_candidates(
                query_histogram,
                candidate_histograms,
                batch_size=10,
            )

            # Ensure reranked_distances is a plain Python list, not a NumPy array
            if isinstance(reranked_distances, np.ndarray):
                reranked_distances = reranked_distances.tolist()
        else:
            # In fast mode, use ANN distances directly
            reranked_distances = [c["ann_distance"] for c in valid_candidates]

        rerank_time = time.time() - rerank_start

        # Create final results
        max_dist = max(reranked_distances) if len(reranked_distances) > 0 else 1.0
        results = []

        for i, (candidate, dist) in enumerate(
            zip(valid_candidates, reranked_distances)
        ):
            result = {
                "image_id": candidate["image_id"],
                "file_path": candidate["file_path"],
                "distance": float(dist),
                "confidence": max(0.0, 1.0 - float(dist) / max_dist),
                "ann_distance": candidate["ann_distance"],
                "rank": i + 1,
            }
            results.append(result)

        # Sort by distance
        results.sort(key=lambda x: x["distance"])

        # Update performance stats
        total_time = time.time() - start_time
        _update_performance_stats(
            ann_time=ann_time,
            metadata_time=metadata_time,
            rerank_time=rerank_time,
            total_time=total_time,
            cache_hit=False,
        )

        # Log completion
        logger.info(f"Search completed in {total_time:.3f}s")
        logger.info(f"Found {len(results)} results, returning top {k}")

        # Return top k results
        return results[:k]

    except Exception as e:
        logger.error(f"Search failed: {str(e)}", exc_info=True)
        raise


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

        # Check result structure
        for result in results:
            if "image_id" not in result or "distance" not in result:
                logger.error("Search result missing required attributes")
                return False

        # Check ranking consistency
        distances = [r["distance"] for r in results]
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
