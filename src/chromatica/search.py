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
from typing import List, Dict, Tuple, Optional, Union
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
    index: AnnIndex,
    store: MetadataStore,
    k: Optional[int] = None,
    max_rerank: Optional[int] = None,
    use_approximate_reranking: bool = False,
    rerank_batch_size: int = 10,
) -> List[SearchResult]:
    """
    Perform a complete two-stage search for similar images.

    This function implements the full search pipeline as specified in the critical
    instructions document:

    1. **ANN Search Stage**: Use FAISS HNSW index to retrieve top-K candidates
       - Applies Hellinger transform automatically for L2 compatibility
       - Returns distances and indices for the most similar vectors

    2. **Metadata Retrieval Stage**: Fetch raw histograms for reranking
       - Retrieves original histograms from DuckDB metadata store
       - Maps FAISS indices to actual image IDs and file paths

    3. **Reranking Stage**: Use Sinkhorn-EMD for high-fidelity ranking
       - Computes perceptually accurate distances using optimal transport
       - Re-sorts candidates by Sinkhorn distance for final ranking

    Performance monitoring is built-in with separate timing for each stage,
    enabling detailed performance analysis and optimization.

    Args:
        query_histogram: Query histogram (normalized probability distribution)
                        Shape: (TOTAL_BINS,) where TOTAL_BINS = 1152
        index: FAISS HNSW index instance for fast ANN search
        store: DuckDB metadata store for histogram retrieval
        k: Number of candidates to retrieve from ANN stage
           Default: RERANK_K from configuration (200)
        max_rerank: Maximum number of candidates to rerank
                   If None, reranks all retrieved candidates
                   If specified, limits reranking to top candidates

    Returns:
        List[SearchResult]: Ranked list of similar images with comprehensive
                           information including file paths, distances, and ranks.
                           Results are sorted by Sinkhorn distance (ascending).

    Raises:
        ValueError: If input validation fails (e.g., invalid histogram shape)
        RuntimeError: If any stage of the search pipeline fails
        TypeError: If inputs have incorrect types

    Example:
        >>> from chromatica.core.histogram import generate_histogram
        >>> from chromatica.indexing.store import AnnIndex, MetadataStore
        >>>
        >>> # Load query image and generate histogram
        >>> query_hist = generate_histogram("query_image.jpg")
        >>>
        >>> # Initialize search components
        >>> index = AnnIndex()
        >>> store = MetadataStore("metadata.db")
        >>>
        >>> # Perform search
        >>> results = find_similar(query_hist, index, store)
        >>>
        >>> # Display top results
        >>> for result in results[:5]:
        ...     print(f"Rank {result.rank}: {result.image_id} "
        ...           f"(distance: {result.distance:.6f})")

    Performance Characteristics:
        - ANN stage: ~1-5ms for 200 candidates (depending on index size)
        - Metadata retrieval: ~10-50ms for 200 histograms
        - Reranking stage: ~100-500ms for 200 candidates using Sinkhorn-EMD
        - Total search time: ~150-600ms for typical queries

    Memory Usage:
        - Query histogram: ~9KB (1152 * 8 bytes)
        - Candidate histograms: ~1.8MB for 200 candidates
        - Temporary arrays: ~2-5MB during computation
        - Total memory: ~3-7MB for typical searches
    """
    # Input validation
    if not isinstance(query_histogram, np.ndarray):
        raise TypeError(
            f"Query histogram must be numpy array, got {type(query_histogram)}"
        )

    if query_histogram.ndim != 1:
        raise ValueError(
            f"Query histogram must be 1D, got shape {query_histogram.shape}"
        )

    if query_histogram.shape[0] != 1152:  # TOTAL_BINS
        raise ValueError(
            f"Query histogram must have 1152 dimensions, got {query_histogram.shape[0]}"
        )

    if not isinstance(index, AnnIndex):
        raise TypeError(f"Index must be AnnIndex instance, got {type(index)}")

    if not isinstance(store, MetadataStore):
        raise TypeError(f"Store must be MetadataStore instance, got {type(store)}")

    # Use configuration defaults if not specified
    if k is None:
        k = RERANK_K

    if max_rerank is None:
        max_rerank = k

    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    if max_rerank <= 0:
        raise ValueError(f"max_rerank must be positive, got {max_rerank}")

    if max_rerank > k:
        logger.warning(f"max_rerank ({max_rerank}) > k ({k}), setting max_rerank = k")
        max_rerank = k

    logger.info(f"Starting two-stage search: k={k}, max_rerank={max_rerank}")
    logger.debug(f"Query histogram shape: {query_histogram.shape}")
    logger.debug(f"Query histogram sum: {query_histogram.sum():.6f}")

    # Check cache first
    query_hash = _generate_query_hash(
        query_histogram, k, max_rerank, use_approximate_reranking
    )
    cached_results = _get_cached_results(query_hash)

    if cached_results is not None:
        logger.info(
            f"Cache hit for query {query_hash[:8]}... returning {len(cached_results)} cached results"
        )
        # Update performance stats for cache hit
        _update_performance_stats(
            0.0, 0.0, 0.0, 0.001, True
        )  # Very fast for cache hits
        return cached_results

    logger.debug(f"Cache miss for query {query_hash[:8]}..., proceeding with search")

    # Stage 1: ANN Search
    ann_start_time = time.time()
    try:
        logger.info("Stage 1: Performing ANN search with FAISS HNSW index")

        # Perform ANN search to get top-k candidates
        distances, indices = index.search(query_histogram, k)

        ann_time = time.time() - ann_start_time
        logger.info(f"ANN search completed in {ann_time:.3f}s")
        logger.info(f"Retrieved {len(indices[0])} candidates from FAISS index")
        logger.debug(
            f"ANN distance range: [{distances[0].min():.6f}, {distances[0].max():.6f}]"
        )

    except Exception as e:
        ann_time = time.time() - ann_start_time
        logger.error(f"ANN search failed after {ann_time:.3f}s: {e}")
        raise RuntimeError(f"ANN search stage failed: {e}")

    # Stage 2: Metadata Retrieval (Optimized)
    metadata_start_time = time.time()
    try:
        logger.info("Stage 2: Retrieving metadata and raw histograms (optimized)")

        # Get image IDs in insertion order (matching FAISS indices)
        image_ids = store.get_image_ids_in_order()

        if not image_ids:
            logger.error("No image IDs found in metadata store")
            raise RuntimeError("Metadata store is empty")

        if len(image_ids) != index.get_total_vectors():
            logger.warning(
                f"Index vector count ({index.get_total_vectors()}) doesn't match "
                f"metadata count ({len(image_ids)})"
            )

        # Extract candidate image IDs from FAISS results
        candidate_image_ids = []
        ann_scores = []

        for i, (distance, index_idx) in enumerate(zip(distances[0], indices[0])):
            if index_idx < len(image_ids):
                candidate_image_ids.append(image_ids[index_idx])
                ann_scores.append(distance)
            else:
                logger.warning(
                    f"Index {index_idx} out of range for image_ids (len={len(image_ids)})"
                )

        if not candidate_image_ids:
            logger.error("No valid candidates found after metadata mapping")
            raise RuntimeError("Failed to map FAISS results to metadata")

        # Load only the histograms we need (selective loading)
        logger.info(f"Loading histograms for {len(candidate_image_ids)} candidates")
        hist_start = time.time()
        candidate_histograms_dict = store.get_histograms_by_ids(candidate_image_ids)
        hist_time = time.time() - hist_start
        logger.info(f"Histogram loading took {hist_time:.3f}s")

        # Convert to ordered list matching the FAISS results
        candidate_histograms = []
        candidate_ids = []

        for image_id in candidate_image_ids:
            if image_id in candidate_histograms_dict:
                candidate_histograms.append(candidate_histograms_dict[image_id])
                candidate_ids.append(image_id)
            else:
                logger.warning(f"Histogram not found for image {image_id}")

        if not candidate_histograms:
            logger.error("No valid histograms retrieved for candidates")
            raise RuntimeError("Failed to retrieve candidate histograms")

        metadata_time = time.time() - metadata_start_time
        logger.info(f"Metadata retrieval completed in {metadata_time:.3f}s")
        logger.info(
            f"Loaded {len(candidate_histograms)} histograms for {len(candidate_image_ids)} candidates"
        )

    except Exception as e:
        metadata_time = time.time() - metadata_start_time
        logger.error(f"Metadata retrieval failed after {metadata_time:.3f}s: {e}")
        raise RuntimeError(f"Metadata retrieval stage failed: {e}")

    # Stage 3: Reranking
    rerank_start_time = time.time()
    try:
        logger.info("Stage 3: Reranking candidates using Sinkhorn-EMD")

        # Limit candidates for reranking if specified
        if max_rerank < len(candidate_histograms):
            candidate_histograms = candidate_histograms[:max_rerank]
            candidate_ids = candidate_ids[:max_rerank]
            ann_scores = ann_scores[:max_rerank]
            logger.info(f"Limited reranking to top {max_rerank} candidates")

        # Perform Sinkhorn-EMD reranking with optimization options
        rerank_start = time.time()
        rerank_results = rerank_candidates(
            query_histogram,
            candidate_histograms,
            candidate_ids,
            use_approximate=use_approximate_reranking,
            batch_size=rerank_batch_size,
            use_parallel=False,  # Disable parallel processing to avoid memory issues
            streaming_mode=True,  # Enable streaming mode for memory efficiency
            early_termination_count=min(
                max_rerank,
                len(candidate_histograms),  # Use max_rerank for early termination limit
            ),  # Respect requested rerank count
            early_termination_threshold=(
                0.1 if use_approximate_reranking else 0.05
            ),  # Adjust threshold based on mode
        )

        rerank_time = time.time() - rerank_start_time
        rerank_actual_time = time.time() - rerank_start
        logger.info(
            f"Reranking completed in {rerank_time:.3f}s (actual rerank: {rerank_actual_time:.3f}s)"
        )
        logger.info(f"Successfully reranked {len(rerank_results)} candidates")

    except Exception as e:
        rerank_time = time.time() - rerank_start_time
        logger.error(f"Reranking failed after {rerank_time:.3f}s: {e}")
        raise RuntimeError(f"Reranking stage failed: {e}")

    # Stage 4: Result Assembly
    try:
        logger.info("Stage 4: Assembling final search results")

        # Create final search results with comprehensive information
        search_results = []

        for rerank_result in rerank_results:
            # Find the corresponding ANN score
            try:
                ann_score_idx = candidate_ids.index(rerank_result.candidate_id)
                ann_score = ann_scores[ann_score_idx]
            except ValueError:
                logger.warning(
                    f"Could not find ANN score for {rerank_result.candidate_id}"
                )
                ann_score = float("inf")

            # Get file path from metadata store
            try:
                metadata = store.get_image_info(rerank_result.candidate_id)
                if metadata and metadata.get("file_path"):
                    file_path = metadata["file_path"]
                else:
                    logger.warning(
                        f"No file path found for {rerank_result.candidate_id}"
                    )
                    file_path = "unknown"
            except Exception as e:
                logger.warning(
                    f"Could not retrieve file path for {rerank_result.candidate_id}: {e}"
                )
                file_path = "unknown"

            # Create search result
            search_result = SearchResult(
                image_id=rerank_result.candidate_id,
                file_path=file_path,
                distance=rerank_result.distance,
                rank=rerank_result.rank,
                ann_score=ann_score,
            )

            search_results.append(search_result)

        # Final timing and logging
        total_time = time.time() - ann_start_time

        logger.info("Search pipeline completed successfully:")
        logger.info(f"  - Total time: {total_time:.3f}s")
        logger.info(f"  - ANN stage: {ann_time:.3f}s ({ann_time/total_time*100:.1f}%)")
        logger.info(
            f"  - Metadata stage: {metadata_time:.3f}s ({metadata_time/total_time*100:.1f}%)"
        )
        logger.info(
            f"  - Reranking stage: {rerank_time:.3f}s ({rerank_time/total_time*100:.1f}%)"
        )
        logger.info(f"  - Results returned: {len(search_results)}")

        if search_results:
            min_dist = search_results[0].distance
            max_dist = search_results[-1].distance
            logger.info(f"  - Final distance range: [{min_dist:.6f}, {max_dist:.6f}]")

        # Cache the results
        _cache_results(query_hash, search_results)
        logger.debug(f"Cached search results for query {query_hash[:8]}...")

        # Update performance statistics
        _update_performance_stats(
            ann_time, metadata_time, rerank_time, total_time, False
        )

        return search_results

    except Exception as e:
        total_time = time.time() - ann_start_time
        logger.error(f"Result assembly failed after {total_time:.3f}s: {e}")
        raise RuntimeError(f"Result assembly stage failed: {e}")


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
            query_hist, index, store, k=min(50, index.get_total_vectors())
        )

        # Validate results
        if not results:
            logger.error("Validation search returned no results")
            return False

        # Check result structure
        for result in results:
            if not hasattr(result, "image_id") or not hasattr(result, "distance"):
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
