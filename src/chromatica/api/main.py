"""
FastAPI application for the Chromatica color search engine.

This module provides the REST API endpoint for color-based image search as specified
in Section H of the critical instructions. The API loads the FAISS index and DuckDB
metadata store on startup and exposes the search functionality via a GET /search endpoint.

Key Features:
- FastAPI application with comprehensive logging
- Automatic loading of FAISS index and DuckDB store on startup
- GET /search endpoint with color and weight query parameters
- Integration with existing search pipeline (find_similar)
- Exact JSON response structure as specified in Section H
- Performance timing and logging for all operations
- NEW: Visualization endpoints for query and result visualization
- NEW: Web interface for interactive color picking

The API follows the two-stage search architecture:
1. ANN search using FAISS HNSW index
2. Reranking using Sinkhorn-EMD for high-fidelity results
"""

import logging
import os
import time
import uuid
from typing import List, Optional, Dict, Any, Tuple  # Added Tuple
from pathlib import Path
import base64
import io
from datetime import datetime
from contextlib import asynccontextmanager
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import lru_cache
import subprocess

from dataclasses import asdict

from .routers import search as search_router
from .routers import stats as stats_router
from .routers import system as system_router
from .routers import extract as extract_router
from .routers import advanced as advanced_router

from ..utils.config import (
    LOG_DIR,
    INDEX_FILE,
    DB_FILE,
    get_index_path,
    get_db_path,
    set_global_state,  # 💡 NEW: Import the state setter
)

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import JSONResponse, Response, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import cv2
from sklearn.cluster import KMeans


# CORRECTED IMPORTS
from ..core.query import create_query_histogram, hex_to_lab, normalize_weights
from ..search import find_similar
from ..indexing.store import AnnIndex, MetadataStore
from ..indexing.pipeline import process_image  # Added process_image import
from ..utils.config import (
    TOTAL_BINS,
    RERANK_K,
    MAX_SEARCH_RESULTS,
    MAX_COLOR_COUNT,
    USE_APPROXIMATE_RERANKING,
)
from ..utils.logging_config import setup_logging
from ..visualization import create_query_visualization, create_results_collage
from .visualization_3d import router as visualization_3d_router, set_search_components

from ..utils.color_utils import (
    extract_dominant_colors,
    extract_dominant_colors_with_weights,
)

# ADD THESE IMPORTS:
from .models import (
    QueryColors, SearchResponse, SearchMetadata,
    ParallelSearchRequest, ParallelSearchResponse,
    VisualizationResponse,
    CommandRequest, CommandResponse,
    PerformanceStatsResponse,
    SearchResult # Note: May need to confirm if SearchResult is directly used or imported from search.py
)
# The `dataclass` and `asdict` imports may no longer be needed if they were only for the models.
# Keep `from typing import List, Optional, Dict, Any, Tuple` if still used elsewhere.

import json

# Initialize logging
api_logger, webui_logger, search_logger, visualization_logger = setup_logging()
logger = api_logger  # Keep backward compatibility

START_TIME = time.time()


# Add new imports
from typing import Tuple
from ..core.query import hex_to_lab, normalize_weights
from ..indexing.pipeline import process_image
from ..utils.config import MAX_SEARCH_RESULTS, MAX_COLOR_COUNT

# 庁 NEW: Use thread-safe global state for performance tracking (Keep this section)
max_concurrent_searches = 2

PERFORMANCE_STATS = {
    "total_searches": 0,
    "total_time": 0.0,
    "concurrent_searches": 0,
    "max_concurrent_searches": max_concurrent_searches,
    "search_times": [],
    "lock": threading.Lock(),
}

performance_stats = {
    "total_searches": 0,
    "total_time": 0.0,
    "concurrent_searches": 0,
    "max_concurrent_searches": max_concurrent_searches,
    "search_times": [],
    "lock": threading.Lock(),
}


# Global variables for the search components
index: Optional[AnnIndex] = None
store: Optional[MetadataStore] = None

# Thread pool for CPU-intensive operations
thread_pool: Optional[ThreadPoolExecutor] = None

# Thread lock for stats updates
stats_lock = threading.Lock()


def update_performance_stats(duration: float):
    """Adds a search duration to the performance statistics."""
    with PERFORMANCE_STATS["lock"]:
        PERFORMANCE_STATS["total_searches"] += 1
        PERFORMANCE_STATS["total_time"] += duration

        # 💡 CRITICAL: Update the new 'search_times' list
        PERFORMANCE_STATS["search_times"].append(duration)

        # Keep the list size manageable for a rolling average (e.g., last 100 searches)
        if len(PERFORMANCE_STATS["search_times"]) > 100:
            PERFORMANCE_STATS["search_times"].pop(0)


def increment_concurrent_searches():
    # Keep this as a helper function within main.py
    with PERFORMANCE_STATS["lock"]:
        if PERFORMANCE_STATS["concurrent_searches"] >= MAX_CONCURRENT_SEARCHES:
            raise HTTPException(
                status_code=503, detail="Server busy, please try again."
            )
        PERFORMANCE_STATS["concurrent_searches"] += 1


def decrement_concurrent_searches():
    # Keep this as a helper function within main.py
    with PERFORMANCE_STATS["lock"]:
        PERFORMANCE_STATS["concurrent_searches"] = max(
            0, PERFORMANCE_STATS["concurrent_searches"] - 1
        )


def increment_concurrent_searches() -> None:
    """Increment concurrent search counter."""
    global performance_stats
    with stats_lock:
        performance_stats["concurrent_searches"] += 1
        performance_stats["max_concurrent_searches"] = max(
            performance_stats["max_concurrent_searches"],
            performance_stats["concurrent_searches"],
        )


def decrement_concurrent_searches() -> None:
    """Decrement concurrent search counter."""
    global performance_stats
    with stats_lock:
        performance_stats["concurrent_searches"] = max(
            0, performance_stats["concurrent_searches"] - 1
        )


# Configuration constants derived from environment or defaults
INDEX_DIR = os.environ.get(
    "INDEX_DIR", Path(__file__).parent.parent.parent.parent / "index"
)
FAISS_PATH = Path(INDEX_DIR) / "chromatica_index.faiss"
DB_PATH = Path(INDEX_DIR) / "chromatica_metadata.db"


# Helper function for thread pool execution
def _run_in_executor(func, *args, **kwargs):
    """Utility to run a blocking function in the thread pool executor."""
    if not thread_pool:
        raise RuntimeError("Thread pool is not initialized.")
    return thread_pool.submit(func, *args, **kwargs)


# Replace the existing perform_search_async function
async def perform_search_async(
    query_histogram: np.ndarray,
    k: int,
    max_rerank: int,
    fast_mode: bool = False,
    batch_size: int = 10,
) -> List[Dict[str, Any]]:
    """
    Asynchronously executes the two-stage color search pipeline.

    Args:
        query_histogram: Numpy array of the query histogram
        k: Number of candidates to retrieve from FAISS
        max_rerank: Number of results to rerank
        fast_mode: Whether to use fast approximate reranking
        batch_size: Batch size for reranking operations

    Returns:
        List of search results
    """
    if index is None or store is None:
        raise HTTPException(status_code=503, detail="Search engine is not initialized.")

    try:
        # Ensure the query histogram is properly normalizedf
        query_histogram = query_histogram.astype(np.float32)
        if query_histogram.sum() > 0:
            query_histogram /= query_histogram.sum()

        # Run the blocking search function in the thread pool
        search_task = _run_in_executor(
            find_similar, query_histogram, k, max_rerank, index, store, fast_mode
        )

        # Get search results
        results = await asyncio.wrap_future(search_task)

        # Convert SearchResult dataclass objects to dictionaries for JSON serialization
        response_data = [asdict(r) for r in results]

        # Return results directly - they already have the file_path key
        # No need to re-fetch from the store or reformat
        return response_data

    except Exception as e:
        logger.error(f"ANN search stage failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"ANN search stage failed: {str(e)}"
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan with proper startup and shutdown."""
    global index, store, thread_pool

    logger.info("Starting up Chromatica API...")

    # Initialize ThreadPool for blocking I/O
    thread_pool = ThreadPoolExecutor(max_workers=os.cpu_count() * 2)
    logger.info(
        f"ThreadPoolExecutor initialized with {thread_pool._max_workers} workers."
    )

    # --- Loading Index and Store ---
    global global_index, global_store

    try:
        # Define index directory from environment variable with fallback
        INDEX_DIR = Path(os.getenv("CHROMATICA_INDEX_DIR", "./index")).resolve()
        FAISS_PATH = INDEX_DIR / "chromatica_index.faiss"
        DB_PATH = INDEX_DIR / "chromatica_metadata.db"

        logger.info(f"Attempting to load index files from: {INDEX_DIR}")
        logger.info(f"FAISS path: {FAISS_PATH}")
        logger.info(f"DuckDB path: {DB_PATH}")

        # Check if index files exist - handle gracefully
        if not FAISS_PATH.exists() or not DB_PATH.exists():
            logger.warning("=" * 80)
            logger.warning("⚠️  WARNING: Index files not found!")
            logger.warning(f"   FAISS index: {FAISS_PATH.exists()} at {FAISS_PATH}")
            logger.warning(f"   DuckDB database: {DB_PATH.exists()} at {DB_PATH}")
            logger.warning("=" * 80)
            logger.warning("📋 To build the index, run:")
            logger.warning(f"   python scripts/build_index.py <image_directory> --output-dir {INDEX_DIR}")
            logger.warning("=" * 80)
            logger.warning("🔄 Application will start in LIMITED MODE (advanced features only)")
            logger.warning("   Search functionality will be disabled until index is built.")
            logger.warning("=" * 80)
            
            # Set index and store to None - search will be disabled but app can start
            index = None
            store = None
        else:
            # Initialize search components if index exists
            index = AnnIndex(index_path=str(FAISS_PATH))
            store = MetadataStore(db_path=str(DB_PATH))

        # Initialize components if they exist
        if index is not None and store is not None:
            # Add diagnostic logging
            logger.info(f"FAISS index loaded with {index.get_total_vectors()} vectors")

            # Check DuckDB records
            store.check_stored_ids(limit=5)

            # 庁 CRITICAL: Set the global state for the routers to access
            # PASS THE LOADED 'index' AND 'store' OBJECTS
            set_global_state(
                index=index,
                store=store,
                increment_func=increment_concurrent_searches,
                decrement_func=decrement_concurrent_searches,
                update_func=update_performance_stats,
                performance_stats=PERFORMANCE_STATS,  # 庁 PASS THE STATS DICT
                start_time=START_TIME,
            )

            # Verify counts match
            faiss_count = index.get_total_vectors()
            duckdb_count = store.get_image_count()
            if faiss_count != duckdb_count:
                logger.warning(
                    f"Count mismatch! FAISS: {faiss_count}, DuckDB: {duckdb_count}"
                )
        else:
            # Set global state with None values - advanced features will still work
            set_global_state(
                index=None,
                store=None,
                increment_func=increment_concurrent_searches,
                decrement_func=decrement_concurrent_searches,
                update_func=update_performance_stats,
                performance_stats=PERFORMANCE_STATS,
                start_time=START_TIME,
            )
            logger.info("✅ Application started in LIMITED MODE - advanced features available")
            logger.info("   Use color palette tools, harmony analysis, gradients, and statistics")
            logger.info("   Search functionality disabled until index is built")

    except Exception as e:
        logger.error(f"Failed to load search components: {e}")
        raise

    yield

    # Cleanup
    logger.info("Shutting down Chromatica API...")
    if store:
        store.close()
    if thread_pool:
        thread_pool.shutdown(wait=True)

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Chromatica Color Search Engine",
    description="A two-stage color-based image search engine using color space and Sinkhorn-EMD reranking with visual enhancements",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(search_router.router)
app.include_router(stats_router.router)
app.include_router(system_router.router)
app.include_router(extract_router.router)
app.include_router(advanced_router.router)

# Get absolute path to static directory for mounting
static_dir = Path(__file__).parent / "static"
logger.info(f"Static directory path: {static_dir}")

# Create static directory if it doesn't exist
static_dir.mkdir(parents=True, exist_ok=True)

# Mount static files with explicit type checking
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    logger.info(f"Static files mounted from: {static_dir}")
else:
    logger.error(f"Static directory not found: {static_dir}")


# Add explicit route for JavaScript files
@app.get("/static/js/{file_path:path}")
async def serve_js(file_path: str):
    """Serve JavaScript files with proper MIME type."""
    js_path = static_dir / "js" / file_path
    if not js_path.exists():
        raise HTTPException(
            status_code=404, detail=f"JavaScript file {file_path} not found"
        )
    return FileResponse(str(js_path), media_type="application/javascript")


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint with web interface."""
    try:
        # Try to serve the web interface
        static_dir = Path(__file__).parent / "static"
        index_path = static_dir / "index.html"

        if index_path.exists():
            with open(index_path, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        else:
            # Fallback to API info if web interface not available
            return {
                "message": "Chromatica Color Search Engine API",
                "version": "1.0.0",
                "endpoints": {
                    "search": "/search - Color-based image search",
                    "visualize_query": "/visualize/query - Generate query visualization",
                    "visualize_results": "/visualize/results - Generate results collage",
                    "docs": "/docs - API documentation",
                },
                "status": (
                    "ready"
                    if index is not None and store is not None
                    else "initializing"
                ),
            }
    except Exception as e:
        logger.error(f"Failed to serve web interface: {e}")
        # Return API info as fallback
        return {
            "message": "Chromatica Color Search Engine API",
            "version": "1.0.0",
            "endpoints": {
                "search": "/search - Color-based image search",
                "visualize_query": "/visualize/query - Generate query visualization",
                "visualize_results": "/visualize/results - Generate results collage",
                "docs": "/docs - API documentation",
            },
            "status": (
                "ready" if index is not None and store is not None else "initializing"
            ),
        }


@app.get("/health")
async def health_check():
    """Health check endpoint for server status."""
    global index, store

    # Test if components are actually working
    faiss_status = "not_loaded"
    metadata_status = "not_loaded"

    try:
        if index is not None:
            # Log index details
            api_logger.info(f"FAISS index stats - ntotal: {index.index.ntotal}")
            faiss_status = "loaded"
    except Exception as e:
        api_logger.warning(f"FAISS index test failed: {e}")

    try:
        if store is not None:
            # Test metadata store by getting image count
            count = store.get_image_count()
            api_logger.info(f"Metadata store stats - image count: {count}")
            metadata_status = "loaded"
    except Exception as e:
        api_logger.warning(f"Metadata store test failed: {e}")

    # Get actual component status
    components_healthy = faiss_status == "loaded" and metadata_status == "loaded"

    return {
        "status": "healthy" if components_healthy else "unhealthy",
        "message": "Chromatica Color Search Engine API",
        "version": "1.0.0",
        "components": {"faiss_index": faiss_status, "metadata_store": metadata_status},
        "timestamp": time.time(),
    }


@app.get("/api/info")
async def api_info():
    """API information endpoint."""
    return {
        "message": "Chromatica Color Search Engine API",
        "version": "1.0.0",
        "endpoints": {
            "search": "/search - Color-based image search",
            "visualize_query": "/visualize/query - Generate query visualization",
            "visualize_results": "/visualize/results - Generate results collage",
            "docs": "/docs - API documentation",
        },
        "status": (
            "ready" if index is not None and store is not None else "initializing"
        ),
    }


@app.get("/image/{image_id}")
async def get_image(image_id: str):
    """
    Serve an image file by its ID.

    This endpoint retrieves the image file path from the metadata store
    and serves the actual image file to the client.

    Args:
        image_id: The unique identifier for the image

    Returns:
        The image file as a response

    Raises:
        HTTPException: If image not found or file doesn't exist
    """
    global store

    if store is None:
        raise HTTPException(status_code=503, detail="Metadata store is not available")

    try:
        # Get the image metadata from the store - use get_image_info instead of get_image_metadata
        image_info = store.get_image_info(image_id)
        if not image_info:
            raise HTTPException(
                status_code=404, detail=f"Image with ID {image_id} not found"
            )

        file_path = image_info.get("file_path")
        if not file_path or not Path(file_path).exists():
            raise HTTPException(
                status_code=404, detail=f"Image file not found at {file_path}"
            )

        # Determine content type based on file extension
        content_type = "image/jpeg"  # default
        if file_path.lower().endswith(".png"):
            content_type = "image/png"
        elif file_path.lower().endswith(".gif"):
            content_type = "image/gif"
        elif file_path.lower().endswith(".webp"):
            content_type = "image/webp"

        # Read and return the image file
        with open(file_path, "rb") as f:
            image_data = f.read()

        return Response(content=image_data, media_type=content_type)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to serve image {image_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to serve image: {str(e)}")


@app.get("/visualize/query")
async def visualize_query(
    colors: str = Query(..., description="Comma-separated list of hex color codes"),
    weights: str = Query(..., description="Comma-separated list of float weights"),
):
    """
    Generate a visual representation of the color query.
    """
    visualization_logger.info("=== QUERY VISUALIZATION REQUEST ===")
    visualization_logger.info(f"Colors: {colors}")
    visualization_logger.info(f"Weights: {weights}")

    try:
        # Parse parameters
        color_list = [c.strip() for c in colors.split(",") if c.strip()]
        weight_list = [float(w.strip()) for w in weights.split(",") if w.strip()]

        if len(color_list) != len(weight_list):
            visualization_logger.error("Number of colors must match number of weights")
            raise HTTPException(
                status_code=400, detail="Number of colors must match number of weights"
            )

        visualization_logger.info(
            f"Generating query visualization for {len(color_list)} colors"
        )

        # Generate visualization
        viz_start = time.time()
        viz_path = create_query_visualization(
            color_list, weight_list, "temp_query_viz.png"
        )
        viz_time = time.time() - viz_start

        visualization_logger.info(f"Query visualization generated in {viz_time:.3f}s")

        # Read the generated image
        with open(viz_path, "rb") as f:
            image_data = f.read()

        # Clean up temporary file
        Path(viz_path).unlink(missing_ok=True)

        visualization_logger.info("Query visualization request completed successfully")

        # Return the image
        return Response(content=image_data, media_type="image/png")

    except Exception as e:
        visualization_logger.error(f"Failed to generate query visualization: {e}")
        raise HTTPException(
            status_code=500, detail=f"Visualization generation failed: {str(e)}"
        )


@app.get("/visualize/results")
async def visualize_results(
    colors: str = Query(..., description="Comma-separated list of hex color codes"),
    weights: str = Query(..., description="Comma-separated list of float weights"),
    k: int = Query(
        10, description="Number of results to include in collage", ge=1, le=50
    ),
):
    """
    Generate a visual collage of search results.

    Creates a grid collage of the top-k search results with distance annotations.

    Args:
        colors: Comma-separated hex color codes
        weights: Comma-separated weights
        k: Number of results to include

    Returns:
        PNG image of the results collage
    """
    global index, store

    if index is None or store is None:
        raise HTTPException(status_code=503, detail="Search system not available")

    try:
        # Parse parameters
        color_list = [c.strip() for c in colors.split(",") if c.strip()]
        weight_list = [float(w.strip()) for w in weights.split(",") if w.strip()]

        if len(color_list) != len(weight_list):
            raise HTTPException(
                status_code=400, detail="Number of colors must match number of weights"
            )

        # Perform search to get results - use more candidates for visualization quality
        query_histogram = create_query_histogram(color_list, weight_list)
        search_k = max(k * 2, 50)  # Get more candidates for better quality
        search_results = await perform_search_async(
            query_histogram=query_histogram,
            k=search_k,
            max_rerank=k,  # Rerank exactly what was requested
            fast_mode=False,  # Always use high-fidelity for visualization
            batch_size=10,
        )

        if not search_results:
            raise HTTPException(status_code=404, detail="No search results found")

        # Extract image paths and distances
        image_paths = []
        distances = []

        for result in search_results:
            # Get image path from metadata store
            try:
                image_info = store.get_image_info(result.image_id)
                if image_info and "file_path" in image_info:
                    image_paths.append(image_info["file_path"])
                    distances.append(result.distance)
            except Exception as e:
                logger.warning(f"Failed to get image info for {result.image_id}: {e}")

        if not image_paths:
            raise HTTPException(
                status_code=500, detail="Failed to retrieve image paths"
            )

        # Generate collage
        collage_path = create_results_collage(
            image_paths, distances, "temp_results_collage.png"
        )

        # Read the generated image
        with open(collage_path, "rb") as f:
            image_data = f.read()

        # Clean up temporary file
        Path(collage_path).unlink(missing_ok=True)

        # Return the image
        return Response(content=image_data, media_type="image/png")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate results collage: {e}")
        raise HTTPException(
            status_code=500, detail=f"Collage generation failed: {str(e)}"
        )


@app.post("/search/parallel", response_model=ParallelSearchResponse)
async def parallel_search(request: ParallelSearchRequest):
    """
    Perform multiple search queries in parallel.

    This endpoint allows processing multiple search queries concurrently,
    significantly improving throughput for batch operations. Each query
    is processed independently and results are returned together.

    Args:
        request: ParallelSearchRequest containing list of queries and concurrency limit

    Returns:
        ParallelSearchResponse with results for all queries
    """
    global index, store

    if index is None or store is None:
        raise HTTPException(
            status_code=503,
            detail="Search system is not available. Please try again later.",
        )

    search_logger.info(f"=== PARALLEL SEARCH REQUEST START ===")
    search_logger.info(f"Total queries: {len(request.queries)}")
    search_logger.info(f"Max concurrent: {request.max_concurrent}")

    start_time = time.time()
    results = []
    successful_queries = 0
    failed_queries = 0

    # Process queries in parallel with semaphore to limit concurrency
    semaphore = asyncio.Semaphore(request.max_concurrent)

    async def process_single_query(
        query_data: Dict[str, Any], query_id: str
    ) -> Dict[str, Any]:
        """Process a single search query."""
        async with semaphore:
            try:
                # Extract query parameters
                colors = query_data.get("colors", "")
                weights = query_data.get("weights", "")
                k = query_data.get("k", 50)
                fast_mode = query_data.get("fast_mode", False)
                batch_size = query_data.get("batch_size", 5)

                # Parse and validate parameters
                color_list = [c.strip() for c in colors.split(",") if c.strip()]
                weight_list = [
                    float(w.strip()) for w in weights.split(",") if w.strip()
                ]

                if len(color_list) != len(weight_list):
                    raise ValueError("Number of colors must match number of weights")

                # Normalize weights
                weight_sum = sum(weight_list)
                weight_list = [w / weight_sum for w in weight_list]

                # Create query histogram
                query_histogram = create_query_histogram(color_list, weight_list)

                # In both modes, search for more candidates than needed for better quality
                search_k = max(k * 2, 50)

                # Perform search
                search_results = await perform_search_async(
                    query_histogram=query_histogram,
                    k=search_k,
                    max_rerank=k,
                    fast_mode=fast_mode,
                    batch_size=batch_size,
                )

                # Format results
                formatted_results = []
                for result in search_results:
                    formatted_results.append(
                        {
                            "image_id": result.image_id,
                            "distance": float(result.distance),
                            "file_path": result.file_path,
                        }
                    )

                return {
                    "query_id": query_id,
                    "success": True,
                    "results": formatted_results,
                    "results_count": len(formatted_results),
                }

            except Exception as e:
                search_logger.error(f"Query {query_id} failed: {e}")
                return {
                    "query_id": query_id,
                    "success": False,
                    "error": str(e),
                    "results": [],
                    "results_count": 0,
                }

    # Create tasks for all queries
    tasks = []
    for i, query_data in enumerate(request.queries):
        query_id = f"parallel_query_{i}_{uuid.uuid4().hex[:8]}"
        task = process_single_query(query_data, query_id)
        tasks.append(task)

    # Execute all queries concurrently
    try:
        query_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for result in query_results:
            if isinstance(result, Exception):
                failed_queries += 1
                results.append(
                    {
                        "query_id": f"failed_{uuid.uuid4().hex[:8]}",
                        "success": False,
                        "error": str(result),
                        "results": [],
                        "results_count": 0,
                    }
                )
            else:
                if result["success"]:
                    successful_queries += 1
                else:
                    failed_queries += 1
                results.append(result)

    except Exception as e:
        search_logger.error(f"Parallel search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Parallel search failed: {str(e)}")

    total_time = time.time() - start_time
    total_time_ms = int(total_time * 1000)

    search_logger.info(f"=== PARALLEL SEARCH REQUEST COMPLETE ===")
    search_logger.info(f"Total time: {total_time:.3f}s")
    search_logger.info(f"Successful: {successful_queries}, Failed: {failed_queries}")

    return ParallelSearchResponse(
        total_queries=len(request.queries),
        successful_queries=successful_queries,
        failed_queries=failed_queries,
        total_time_ms=total_time_ms,
        results=results,
    )


@app.get("/performance/stats", response_model=PerformanceStatsResponse)
async def get_performance_stats():
    """
    Get current performance statistics.

    Returns real-time performance metrics including search counts,
    timing information, and concurrency statistics.
    """
    global performance_stats

    with stats_lock:
        return PerformanceStatsResponse(
            total_searches=performance_stats["total_searches"],
            concurrent_searches=performance_stats["concurrent_searches"],
            max_concurrent_searches=performance_stats["max_concurrent_searches"],
            average_search_time=performance_stats["average_search_time"],
            recent_search_times=performance_stats["search_times"][
                -100:
            ],  # Last 100 searches
        )


@app.post("/restart")
async def restart_server():
    """
    Restart the server by killing all Python processes and restarting.

    This endpoint will:
    1. Kill all running Python processes (including this server)
    2. Start a new server instance
    3. Return success status

    Returns:
        JSON response with restart status
    """
    import subprocess
    import os
    import sys
    import signal
    import psutil

    try:
        logger.info("Server restart requested")

        # Get current process ID
        current_pid = os.getpid()
        logger.info(f"Current server PID: {current_pid}")

        # Find and kill any Python processes using the database
        # Use psutil to find and kill processes instead of shell commands
        python_processes = []
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                if proc.info["name"] and "python" in proc.info["name"].lower():
                    cmdline = proc.info["cmdline"]
                    if cmdline and any(
                        "chromatica" in str(arg).lower() for arg in cmdline
                    ):
                        python_processes.append(proc.info["pid"])
                        logger.info(
                            f"Found Chromatica Python process: PID {proc.info['pid']}"
                        )
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

        # Kill all related Python processes (except current one)
        killed_processes = []
        for pid in python_processes:
            if pid != current_pid:
                try:
                    proc = psutil.Process(pid)
                    proc.terminate()
                    killed_processes.append(pid)
                    logger.info(f"Terminated process PID {pid}")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

        # Wait a moment for processes to terminate
        import time

        time.sleep(2)

        # Force kill any remaining processes
        for pid in killed_processes:
            try:
                proc = psutil.Process(pid)
                if proc.is_running():
                    proc.kill()
                    logger.info(f"Force killed process PID {pid}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # Start new server instance
        project_root = Path(__file__).parent.parent.parent.parent
        python_path = (
            str(project_root / "venv311" / "Scripts" / "python.exe")
            if os.name == "nt"
            else str(project_root / "venv311" / "bin" / "python")
        )

        # Start new server in background
        subprocess.Popen(
            [python_path, "-m", "src.chromatica.api.main"],
            cwd=str(project_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )

        logger.info("New server instance started")

        return {
            "success": True,
            "message": "Server restart initiated successfully",
            "killed_processes": killed_processes,
            "new_server_started": True,
        }

    except Exception as e:
        logger.error(f"Failed to restart server: {e}")
        return {
            "success": False,
            "message": f"Failed to restart server: {str(e)}",
            "error": str(e),
        }


@app.post("/api/execute-command", response_model=CommandResponse)
async def execute_command(request: CommandRequest):
    """
    Execute a command for Quick Test functionality.
    """
    webui_logger.info("=== QUICK TEST COMMAND EXECUTION ===")
    webui_logger.info(f"Command: {request.command}")
    webui_logger.info(f"Arguments: {request.args}")
    webui_logger.info(f"Tool type: {request.tool_type}")
    webui_logger.info(f"Dataset: {request.dataset}")

    try:
        import subprocess
        import os

        # Get the project root directory
        project_root = Path(__file__).parent.parent.parent.parent
        webui_logger.info(f"Project root: {project_root}")

        # Build the full command with absolute paths
        if request.command == "python":
            # Activate virtual environment for Python commands
            if os.name == "nt":  # Windows
                python_path = str(project_root / "venv311" / "Scripts" / "python.exe")
                # Convert relative script paths to absolute paths
                script_path = (
                    str(project_root / request.args[0]) if request.args else ""
                )
                script_args = request.args[1:] if len(request.args) > 1 else []
                full_command = [python_path, script_path] + script_args
            else:  # Unix/Linux
                python_path = str(project_root / "venv311" / "bin" / "python")
                # Convert relative script paths to absolute paths
                script_path = (
                    str(project_root / request.args[0]) if request.args else ""
                )
                script_args = request.args[1:] if len(request.args) > 1 else []
                full_command = [python_path, script_path] + script_args
        else:
            full_command = [request.command] + request.args

        webui_logger.info(f"Executing command: {' '.join(full_command)}")
        webui_logger.info(f"Working directory: {project_root}")

        cmd_start = time.time()
        result = subprocess.run(
            full_command,
            capture_output=True,
            text=True,
            timeout=60,  # 60 second timeout
            cwd=str(project_root),
        )
        cmd_time = time.time() - cmd_start

        if result.returncode == 0:
            # Command succeeded
            webui_logger.info(
                f"Quick Test command executed successfully in {cmd_time:.3f}s"
            )
            webui_logger.info(f"Tool type: {request.tool_type}")
            webui_logger.info(f"Output length: {len(result.stdout)} characters")

            return CommandResponse(
                success=True,
                output=(
                    result.stdout
                    if result.stdout
                    else "Command executed successfully with no output"
                ),
            )
        else:
            # Command failed
            error_msg = (
                result.stderr
                if result.stderr
                else f"Command failed with return code {result.returncode}"
            )
            webui_logger.error(f"Quick Test command failed: {error_msg}")
            webui_logger.error(f"Return code: {result.returncode}")
            webui_logger.error(f"Execution time: {cmd_time:.3f}s")

            return CommandResponse(
                success=False,
                output=result.stdout if result.stdout else "No output",
                error=error_msg,
            )

    except subprocess.TimeoutExpired:
        error_msg = "Command execution timed out after 60 seconds"
        webui_logger.error(f"Quick Test command timeout: {request.tool_type}")
        return CommandResponse(
            success=False, output="Command execution timed out", error=error_msg
        )
    except Exception as e:
        error_msg = f"Failed to execute command: {str(e)}"
        webui_logger.error(f"Quick Test command execution error: {error_msg}")
        return CommandResponse(
            success=False, output="Command execution failed", error=error_msg
        )


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Chromatica API server...")
    uvicorn.run(
        "src.chromatica.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
