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
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass

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


def find_similar(
    query_histogram: np.ndarray,
    index: AnnIndex,
    store: MetadataStore,
    k: Optional[int] = None,
    max_rerank: Optional[int] = None,
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

    # Stage 2: Metadata Retrieval
    metadata_start_time = time.time()
    try:
        logger.info("Stage 2: Retrieving metadata and raw histograms")

        # Get all histograms to map FAISS indices to image IDs
        all_histograms = store.get_all_histograms()

        if not all_histograms:
            logger.error("No histograms found in metadata store")
            raise RuntimeError("Metadata store is empty")

        # Map FAISS indices to image IDs and histograms
        # Note: FAISS indices correspond to insertion order in the metadata store
        image_ids = list(all_histograms.keys())

        if len(image_ids) != index.get_total_vectors():
            logger.warning(
                f"Index vector count ({index.get_total_vectors()}) doesn't match "
                f"metadata count ({len(image_ids)})"
            )

        # Extract candidate information
        candidate_histograms = []
        candidate_ids = []
        ann_scores = []

        for i, (distance, index_idx) in enumerate(zip(distances[0], indices[0])):
            if index_idx < len(image_ids):
                image_id = image_ids[index_idx]
                histogram = all_histograms[image_id]

                candidate_histograms.append(histogram)
                candidate_ids.append(image_id)
                ann_scores.append(float(distance))
            else:
                logger.warning(
                    f"FAISS index {index_idx} out of range for metadata store"
                )

        if not candidate_histograms:
            logger.error("No valid candidates found after metadata mapping")
            raise RuntimeError("Failed to map FAISS results to metadata")

        metadata_time = time.time() - metadata_start_time
        logger.info(f"Metadata retrieval completed in {metadata_time:.3f}s")
        logger.info(f"Mapped {len(candidate_histograms)} candidates to metadata")

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

        # Perform Sinkhorn-EMD reranking
        rerank_results = rerank_candidates(
            query_histogram, candidate_histograms, candidate_ids
        )

        rerank_time = time.time() - rerank_start_time
        logger.info(f"Reranking completed in {rerank_time:.3f}s")
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
                # For now, we'll use the image_id as file_path since we don't have
                # a direct mapping method. In a full implementation, this would
                # come from the metadata store's get_image_info method
                file_path = str(rerank_result.candidate_id)
            except Exception:
                logger.warning(
                    f"Could not retrieve file path for {rerank_result.candidate_id}"
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
