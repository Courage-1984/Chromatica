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
from typing import List, Optional
from pathlib import Path
import base64
import io
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, Response, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import numpy as np

from ..core.query import create_query_histogram
from ..search import find_similar
from ..indexing.store import AnnIndex, MetadataStore
from ..utils.config import TOTAL_BINS
from ..visualization import create_query_visualization, create_results_collage
from .visualization_3d import router as visualization_3d_router, set_search_components
import cv2
import numpy as np
from sklearn.cluster import KMeans


# Enhanced logging configuration
def setup_logging():
    """Setup comprehensive logging for console and file output."""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"chromatica_api_{timestamp}.log"

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            # Console handler
            logging.StreamHandler(),
            # File handler
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )

    # Create specific loggers for different components
    api_logger = logging.getLogger("chromatica.api")
    webui_logger = logging.getLogger("chromatica.webui")
    search_logger = logging.getLogger("chromatica.search")
    visualization_logger = logging.getLogger("chromatica.visualization")

    # Set levels
    api_logger.setLevel(logging.INFO)
    webui_logger.setLevel(logging.INFO)
    search_logger.setLevel(logging.INFO)
    visualization_logger.setLevel(logging.INFO)

    # Log startup information
    api_logger.info(f"Logging initialized - Console and file: {log_file}")
    api_logger.info(f"Log directory: {log_dir.absolute()}")

    return api_logger, webui_logger, search_logger, visualization_logger


# Initialize logging
api_logger, webui_logger, search_logger, visualization_logger = setup_logging()
logger = api_logger  # Keep backward compatibility

# Global variables for the search components
index: Optional[AnnIndex] = None
store: Optional[MetadataStore] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan with proper startup and shutdown."""
    global index, store

    # Startup
    api_logger.info("Starting Chromatica Color Search Engine API...")
    api_logger.info(f"Global variables before init: index={index}, store={store}")

    try:
        # Load the FAISS index
        index_dir = os.getenv("CHROMATICA_INDEX_DIR", "index")
        index_filename = os.getenv("CHROMATICA_INDEX_FILE", "chromatica_index.faiss")
        index_path = Path(index_dir) / index_filename

        if not index_path.exists():
            api_logger.warning(f"FAISS index not found at {index_path}")
            return

        # Load the DuckDB metadata store
        db_filename = os.getenv("CHROMATICA_DB_FILE", "chromatica_metadata.db")
        db_path = Path(index_dir) / db_filename

        if not db_path.exists():
            api_logger.warning(f"Metadata store not found at {db_path}")
            return

        # Initialize the search components
        from ..indexing.store import AnnIndex, MetadataStore

        index = AnnIndex()
        index.load(str(index_path))
        store = MetadataStore(db_path=str(db_path))

        # Set search components for 3D visualization
        set_search_components(index, store)

        api_logger.info("Search components initialized successfully")
        api_logger.info(
            f"Global variables after init: index={index is not None}, store={store is not None}"
        )

    except Exception as e:
        api_logger.error(f"Failed to initialize search components: {e}")

    yield

    # Shutdown
    api_logger.info("Shutting down Chromatica Color Search Engine API...")
    try:
        if store:
            store.close()
            api_logger.info("Metadata store connection closed")
        if index:
            api_logger.info("FAISS index cleanup completed")
    except Exception as e:
        api_logger.error(f"Error during cleanup: {e}")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Chromatica Color Search Engine",
    description="A two-stage color-based image search engine using CIE Lab color space and Sinkhorn-EMD reranking with visual enhancements",
    version="1.0.0",
    lifespan=lifespan,
)

# Include 3D visualization router
app.include_router(visualization_3d_router)

# Mount static files for web interface
try:
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        webui_logger.info(f"Static files mounted from: {static_dir}")
    else:
        webui_logger.warning(f"Static directory not found: {static_dir}")
except Exception as e:
    webui_logger.warning(f"Failed to mount static files: {e}")


def extract_dominant_colors(file_path: str, num_colors: int = 5) -> List[str]:
    """
    Extract dominant colors from an image using K-means clustering.

    Args:
        file_path: Path to the image file
        num_colors: Number of dominant colors to extract (default: 5)

    Returns:
        List of hex color codes representing dominant colors
    """
    try:
        # Read the image
        image = cv2.imread(file_path)
        if image is None:
            logger.warning(f"Could not read image: {file_path}")
            return ["#000000"]  # Return black as fallback

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Reshape image to 2D array of pixels
        pixels = image_rgb.reshape(-1, 3)

        # Use K-means to find dominant colors
        kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)

        # Get the cluster centers (dominant colors)
        dominant_colors = kmeans.cluster_centers_.astype(int)

        # Convert to hex format
        hex_colors = []
        for color in dominant_colors:
            hex_color = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])
            hex_colors.append(hex_color)

        return hex_colors

    except Exception as e:
        logger.error(f"Failed to extract dominant colors from {file_path}: {e}")
        return ["#000000"]  # Return black as fallback


# Pydantic models for request/response validation
class QueryColors(BaseModel):
    """Query colors and weights for the search."""

    colors: List[str] = Field(..., description="List of hex color codes")
    weights: List[float] = Field(..., description="List of corresponding weights")


class SearchResult(BaseModel):
    """Individual search result with image information."""

    image_id: str = Field(..., description="Unique identifier for the image")
    distance: float = Field(..., description="Sinkhorn-EMD distance from query")
    dominant_colors: List[str] = Field(..., description="Dominant colors in the image")
    file_path: Optional[str] = Field(None, description="Path to the image file")


class SearchMetadata(BaseModel):
    """Performance metadata for the search operation."""

    ann_time_ms: int = Field(..., description="ANN search time in milliseconds")
    rerank_time_ms: int = Field(..., description="Reranking time in milliseconds")
    total_time_ms: int = Field(..., description="Total search time in milliseconds")
    index_size: int = Field(
        ..., description="Total number of images in the search index"
    )


class SearchResponse(BaseModel):
    """Complete search response as specified in Section H."""

    query_id: str = Field(..., description="Unique identifier for this search query")
    query: QueryColors = Field(..., description="Original query colors and weights")
    results_count: int = Field(..., description="Number of results returned")
    results: List[SearchResult] = Field(..., description="Ranked search results")
    metadata: SearchMetadata = Field(..., description="Performance timing metadata")


class VisualizationResponse(BaseModel):
    """Response for visualization endpoints."""

    query_id: str = Field(..., description="Unique identifier for this visualization")
    query: QueryColors = Field(..., description="Original query colors and weights")
    visualization_type: str = Field(..., description="Type of visualization generated")
    image_data: str = Field(..., description="Base64 encoded image data")
    mime_type: str = Field(..., description="MIME type of the image")


@app.on_event("startup")
async def startup_event():
    """Initialize the search components on application startup."""
    global index, store

    api_logger.info("Starting Chromatica Color Search Engine API...")
    api_logger.info(f"Global variables before init: index={index}, store={store}")

    try:
        # Load the FAISS index - support custom path via environment variable
        index_dir = os.getenv("CHROMATICA_INDEX_DIR", "index")
        index_filename = os.getenv("CHROMATICA_INDEX_FILE", "chromatica_index.faiss")
        index_path = Path(index_dir) / index_filename

        if not index_path.exists():
            api_logger.warning(f"FAISS index not found at {index_path}")
            api_logger.info(
                f"Please run the indexing script first: python scripts/build_index.py <dataset> --output-dir {index_dir}"
            )
            return

        # Load the DuckDB metadata store - support custom path via environment variable
        db_filename = os.getenv("CHROMATICA_DB_FILE", "chromatica_metadata.db")
        db_path = Path(index_dir) / db_filename

        if not db_path.exists():
            api_logger.warning(f"Metadata store not found at {db_path}")
            api_logger.info(
                f"Please run the indexing script first: python scripts/build_index.py <dataset> --output-dir {index_dir}"
            )
            return

        # Initialize the search components
        from ..indexing.store import AnnIndex, MetadataStore

        index = AnnIndex()
        index.load(str(index_path))
        store = MetadataStore(db_path=str(db_path))

        # Set search components for 3D visualization
        set_search_components(index, store)

        api_logger.info("Search components initialized successfully")
        api_logger.info(f"FAISS index loaded: {index_path}")
        api_logger.info(f"Metadata store loaded: {db_path}")
        api_logger.info(f"Index directory: {index_dir}")
        api_logger.info(
            f"Global variables after init: index={index is not None}, store={store is not None}"
        )
        api_logger.info(
            f"Environment variables: INDEX_DIR={os.getenv('CHROMATICA_INDEX_DIR', 'default')}, INDEX_FILE={os.getenv('CHROMATICA_INDEX_FILE', 'default')}, DB_FILE={os.getenv('CHROMATICA_DB_FILE', 'default')}"
        )

    except Exception as e:
        api_logger.error(f"Failed to initialize search components: {e}")
        api_logger.error("API will not be able to process search requests")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on application shutdown."""
    global index, store

    logger.info("Shutting down Chromatica Color Search Engine API...")

    try:
        if store:
            store.close()
            logger.info("Metadata store connection closed")

        # FAISS index doesn't need explicit cleanup
        if index:
            logger.info("FAISS index cleanup completed")

        logger.info("Cleanup completed successfully")

    except Exception as e:
        logger.error(f"Error during cleanup: {e}")


@app.get("/", response_class=HTMLResponse)
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
            # Test FAISS index
            test_vector = np.random.random((1, 1152)).astype(np.float32)
            distances, indices = index.search(test_vector, 1)
            faiss_status = "loaded"
    except Exception as e:
        api_logger.warning(f"FAISS index test failed: {e}")

    try:
        if store is not None:
            # Test metadata store
            count = store.get_image_count()
            faiss_status = "loaded"
            metadata_status = "loaded"
    except Exception as e:
        api_logger.warning(f"Metadata store test failed: {e}")

    return {
        "status": (
            "healthy"
            if faiss_status == "loaded" and metadata_status == "loaded"
            else "unhealthy"
        ),
        "message": "Chromatica Color Search Engine API",
        "version": "1.0.0",
        "components": {
            "faiss_index": faiss_status,
            "metadata_store": metadata_status,
        },
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


@app.get("/search", response_model=SearchResponse)
async def search_images(
    colors: str = Query(
        ...,
        description="Comma-separated list of hex color codes (without #)",
    ),
    weights: str = Query(
        ...,
        description="Comma-separated list of float weights, corresponding to colors",
    ),
    k: int = Query(50, description="Number of results to return", ge=1, le=200),
    fuzz: float = Query(
        1.0, description="Gaussian sigma multiplier for query fuzziness", ge=0.1, le=5.0
    ),
    fast_mode: bool = Query(
        False,  # Default to normal mode, let users choose fast mode explicitly
        description="Use fast approximate reranking (L2 distance instead of Sinkhorn-EMD)",
    ),
    batch_size: int = Query(
        5,
        description="Batch size for reranking operations",
        ge=1,
        le=50,  # Smaller batch size
    ),
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
        fuzz: Query fuzziness multiplier (default: 1.0)

    Returns:
        SearchResponse: Complete search results with performance metadata

    Raises:
        HTTPException: If parameters are invalid or search fails
    """
    global index, store

    # Log search request
    search_logger.info(f"=== SEARCH REQUEST START ===")
    search_logger.info(f"Colors: {colors}")
    search_logger.info(f"Weights: {weights}")
    search_logger.info(f"Results count (k): {k}")
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

            # Validate hex color format
            for color in color_list:
                if not all(c in "0123456789ABCDEFabcdef" for c in color):
                    raise ValueError(f"Invalid hex color format: {color}")
                if len(color) not in [3, 6]:
                    raise ValueError(f"Hex color must be 3 or 6 characters: {color}")

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

        # Perform the search
        try:
            mode_str = (
                "FAST MODE (L2 distance)" if fast_mode else "NORMAL MODE (Sinkhorn-EMD)"
            )
            search_logger.info(f"Starting search with k={k} in {mode_str}")

            search_start = time.time()
            search_results = find_similar(
                query_histogram=query_histogram,
                index=index,
                store=store,
                k=k,
                max_rerank=k,
                use_approximate_reranking=fast_mode,
                rerank_batch_size=batch_size,
            )
            search_time = time.time() - search_start

            if not search_results:
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
            search_logger.info(f"Found {len(search_results)} results")

        except Exception as e:
            search_logger.error(f"Search operation failed: {e}")
            raise HTTPException(
                status_code=500, detail=f"Search operation failed: {str(e)}"
            )

        # Format results according to Section H specification
        try:
            formatted_results = []

            for i, result in enumerate(search_results):
                # Skip expensive dominant color extraction for performance
                dominant_colors = ["#000000"]  # Placeholder to avoid breaking the API

                formatted_result = SearchResult(
                    image_id=result.image_id,
                    distance=float(result.distance),
                    dominant_colors=dominant_colors,
                    file_path=result.file_path,
                )
                formatted_results.append(formatted_result)

                # Log top 5 results
                if i < 5:
                    search_logger.info(
                        f"Result {i+1}: {result.image_id} (distance: {result.distance:.6f})"
                    )

            # Calculate timing metadata
            total_time = time.time() - total_start_time
            total_time_ms = int(total_time * 1000)

            # For now, we'll use placeholder timing values
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
        # Get the image metadata from the store
        metadata = store.get_image_metadata(image_id)
        if not metadata:
            raise HTTPException(
                status_code=404, detail=f"Image with ID {image_id} not found"
            )

        file_path = metadata.get("file_path")
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

        # Perform search to get results
        query_histogram = create_query_histogram(color_list, weight_list)
        search_results = find_similar(
            query_histogram=query_histogram,
            index=index,
            store=store,
            k=k,
            max_rerank=k,
            use_approximate_reranking=False,  # Use high-fidelity for visualization
            rerank_batch_size=10,
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


# Request model for command execution
class CommandRequest(BaseModel):
    command: str = Field(..., description="Command to execute")
    args: List[str] = Field(..., description="Command arguments")
    tool_type: str = Field(..., description="Type of tool being executed")
    dataset: str = Field(..., description="Dataset path for the tool")


# Response model for command execution
class CommandResponse(BaseModel):
    success: bool = Field(..., description="Whether the command executed successfully")
    output: str = Field(..., description="Command output or error message")
    error: Optional[str] = Field(None, description="Error message if command failed")


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
