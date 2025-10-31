import time
import uuid
import logging
from typing import Optional, List
from pathlib import Path
from dataclasses import asdict  # Needed for final conversion

from fastapi import APIRouter, HTTPException, Query
from ..models import (
    SearchResponse,
    QueryColors,
    SearchMetadata,
)  # Import Pydantic models
from ...search import (
    find_similar,
    MAX_SEARCH_RESULTS,
    RERANK_K,
)  # Import core search logic
from ...core.query import (
    create_query_histogram,
    TOTAL_BINS,
)  # Import histogram creation
from ...utils.config import get_global_state  # CRITICAL: We'll define this next


# --- CRITICAL: Get global state components ---
state = get_global_state()
index = state.index
store = state.store
increment_concurrent_searches = state.increment_concurrent_searches
decrement_concurrent_searches = state.decrement_concurrent_searches
update_performance_stats = state.update_performance_stats
# ---------------------------------------------



# Define the router instance
router = APIRouter(
    prefix="/search",
    tags=["Search"],
)


search_logger = logging.getLogger("chromatica.search")


@router.get("", response_model=SearchResponse)
async def search_images(
    colors: str = Query(...),
    weights: str = Query(...),
    k: int = Query(50),
    n_results: Optional[int] = Query(None),
    fuzz: float = Query(1.0),
    fast_mode: bool = Query(False),
    batch_size: int = Query(5),
):
    """
    Search for images based on color similarity using the two-stage search pipeline.

    This endpoint implements the complete search functionality as specified in Section H:
    1. Parse and validate color and weight parameters
    2. Create a query histogram using tri-linear soft assignment
    3. Perform ANN search using FAISS HNSW index
    4. Rerank candidates using Sinkhorn-EMD
    5. Return results in the exact JSON structure specified

    Args:
        colors: Comma-separated hex color codes (e.g., "ea6a81,f6d727")
        weights: Comma-separated weights (e.g., "0.49,0.51")
        k: Number of results to return (default: 50, max: 200)
        n_results: Alternative parameter for number of results (overrides k if provided)
        fuzz: Query fuzziness multiplier (default: 1.0)
        fast_mode: Use fast approximate reranking (default: False)
        batch_size: Batch size for reranking (default: 5)

    Returns:
        SearchResponse: Complete search results with performance metadata

    Raises:
        HTTPException: If parameters are invalid or search fails
    """

    # --- CRITICAL: Get global state components ---
    state = get_global_state()
    index = state.index
    store = state.store
    increment_concurrent_searches = state.increment_concurrent_searches
    decrement_concurrent_searches = state.decrement_concurrent_searches
    update_performance_stats = state.update_performance_stats
    # ---------------------------------------------

    # Use n_results if provided, otherwise use k
    result_count = min(n_results if n_results is not None else k, MAX_SEARCH_RESULTS)

    # Calculate max_rerank based on the requested results
    max_rerank = min(RERANK_K, result_count * 5)  # Use 5x multiplier for better results

    # Log search request
    search_logger.info(f"=== SEARCH REQUEST START ===")
    search_logger.info(f"Colors: {colors}")
    search_logger.info(f"Weights: {weights}")
    search_logger.info(f"Results count (requested): {result_count}")
    search_logger.info(f"Fast mode: {fast_mode}")
    search_logger.info(f"Batch size: {batch_size}")
    search_logger.info(f"Fuzz: {fuzz}")

    # Validate that search components are initialized
    if index is None or store is None:
        search_logger.error("Search components not initialized - returning 503")
        raise HTTPException(
            status_code=503,
            detail="Search system is not available. Please try again later.",
        )

    # Generate unique query ID
    query_id = str(uuid.uuid4())
    search_logger.info(f"Query ID: {query_id}")

    # Start timing for total search
    total_start_time = time.time()

    try:
        # Parse and validate query parameters
        try:
            # Parse colors
            color_list = [c.strip() for c in colors.split(",") if c.strip()]
            if not color_list:
                raise ValueError("At least one color must be specified")

            # Validate hex color format and strip '#' prefix if present
            processed_colors = []
            for color in color_list:
                # Strip '#' prefix if present
                color = color.lstrip("#")

                if not all(c in "0123456789ABCDEFabcdef" for c in color):
                    raise ValueError(f"Invalid hex color format: {color}")
                if len(color) not in [3, 6]:
                    raise ValueError(f"Hex color must be 3 or 6 characters: {color}")
                processed_colors.append(color)

            # Update color_list with processed colors
            color_list = processed_colors

            # Parse weights
            weight_list = [float(w.strip()) for w in weights.split(",") if w.strip()]
            if not weight_list:
                raise ValueError("At least one weight must be specified")

            # Validate weights
            if len(color_list) != len(weight_list):
                raise ValueError("Number of colors must match number of weights")

            if not all(w > 0 for w in weight_list):
                raise ValueError("All weights must be positive")

            # Normalize weights to sum to 1.0
            weight_sum = sum(weight_list)
            weight_list = [w / weight_sum for w in weight_list]

            search_logger.info(
                f"Parsed {len(color_list)} colors and weights successfully"
            )
            search_logger.info(
                f"Normalized weights: {[f'{w:.3f}' for w in weight_list]}"
            )

        except ValueError as e:
            search_logger.error(f"Parameter validation failed: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid parameters: {str(e)}")

        # Create query histogram
        try:
            search_logger.info("Creating query histogram...")
            histogram_start = time.time()
            query_histogram = create_query_histogram(color_list, weight_list)
            histogram_time = time.time() - histogram_start

            # Validate the generated histogram
            if query_histogram.shape != (TOTAL_BINS,):
                raise ValueError(f"Invalid histogram shape: {query_histogram.shape}")

            search_logger.info(
                f"Query histogram created successfully in {histogram_time:.3f}s"
            )
            search_logger.info(f"Histogram shape: {query_histogram.shape}")
            search_logger.info(f"Histogram sum: {query_histogram.sum():.6f}")

        except Exception as e:
            search_logger.error(f"Failed to create query histogram: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to process query colors: {str(e)}"
            )

        # Perform the search asynchronously
        try:
            mode_str = (
                "FAST MODE (L2 distance)" if fast_mode else "NORMAL MODE (Sinkhorn-EMD)"
            )
            search_logger.info(f"Starting search with k={result_count} in {mode_str}")

            # Track concurrent searches
            increment_concurrent_searches()

            # For fast mode, we need to request more results than we'll actually rerank
            # to ensure we have enough candidates
            # In both modes, search for more candidates than we need for better results
            search_k = max(
                result_count * 2, 50
            )  # Get more candidates for better quality

            # In fast mode, we can rerank all results since L2 distance is fast
            rerank_k = result_count  # Rerank exactly what was requested
            search_logger.info(
                f"{mode_str}: Will rerank top {rerank_k} results out of {search_k} candidates"
            )

            search_start = time.time()
            # Perform the search with correct parameters for both modes
            results = find_similar(
                query_histogram=query_histogram,
                k=result_count,
                max_rerank=max_rerank,  # Pass the calculated max_rerank value
                index=index,
                store=store,
                fast_mode=fast_mode,
            )

            # Log results safely using attribute access
            for i, result in enumerate(results):
                fp = getattr(result, "file_path", None)
                dist = getattr(result, "distance", None)
                url = getattr(result, "image_url", None)
                search_logger.info(
                    f"Result {i+1}: Image: {fp}, Distance: {dist}, URL: {url}"
                )

            search_time = time.time() - search_start

            # Limit to requested result count
            if len(results) > result_count:
                search_logger.info(
                    f"Limiting {len(results)} results to requested {result_count}"
                )
                results = results[:result_count]

            # Update performance stats
            update_performance_stats(search_time)
            decrement_concurrent_searches()

            if not results:
                search_logger.warning("Search returned no results")
                # Return empty results instead of error
                return SearchResponse(
                    query_id=query_id,
                    query=QueryColors(colors=color_list, weights=weight_list),
                    results_count=0,
                    results=[],
                    metadata=SearchMetadata(
                        ann_time_ms=0,
                        rerank_time_ms=0,
                        total_time_ms=0,
                        index_size=store.get_image_count() if store else 0,
                    ),
                )

            search_logger.info(f"Search completed successfully in {search_time:.3f}s")
            search_logger.info(f"Found {len(results)} results")

        except Exception as e:
            search_logger.error(f"Search operation failed: {e}")
            raise HTTPException(
                status_code=500, detail=f"Search operation failed: {str(e)}"
            )

        # Format results according to Section H specification
        try:
            # The 'results' list contains fully formed SearchResult dataclass objects
            # from the search layer. We convert them directly to dictionaries for the
            # SearchResponse Pydantic model to ensure all required fields are included.
            from dataclasses import asdict

            formatted_results = [asdict(r) for r in results]

            # Calculate timing metadata
            total_time = time.time() - total_start_time
            total_time_ms = int(total_time * 1000)

            # For now, we'll use placeholder timing values
            # NOTE: In a complete system, these values would come from the search.find_similar function
            ann_time_ms = int(total_time_ms * 0.3)  # Placeholder: 30% of total time
            rerank_time_ms = int(total_time_ms * 0.7)  # Placeholder: 70% of total time

            # Create the response
            response = SearchResponse(
                query_id=query_id,
                query=QueryColors(colors=color_list, weights=weight_list),
                results_count=len(formatted_results),
                results=formatted_results,
                metadata=SearchMetadata(
                    ann_time_ms=ann_time_ms,
                    rerank_time_ms=rerank_time_ms,
                    total_time_ms=total_time_ms,
                    index_size=store.get_image_count() if store else 0,
                ),
            )

            search_logger.info(f"=== SEARCH REQUEST COMPLETE ===")
            search_logger.info(f"Query ID: {query_id}")
            search_logger.info(f"Total time: {total_time:.3f}s")
            search_logger.info(f"Results returned: {len(formatted_results)}")
            search_logger.info(f"Index size: {store.get_image_count() if store else 0}")

            return response

        except Exception as e:
            search_logger.error(f"Failed to format search results: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to format search results: {str(e)}"
            )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Catch any other unexpected errors
        search_logger.error(f"Unexpected error in search endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
