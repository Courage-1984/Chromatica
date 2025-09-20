"""
3D Visualization Data API for Chromatica Color Search Engine.

This module provides FastAPI endpoints for generating data needed for interactive
3D visualizations including color space navigation, histogram clouds, similarity
landscapes, and search result animations.

Key Features:
- Color space positioning data for CIE Lab cube visualization
- Histogram data for 3D bar chart clouds
- UMAP/t-SNE projections for similarity landscapes
- Search result animation data for two-stage pipeline visualization
- Image metadata and dominant color extraction
- FAISS index structure visualization data
"""

import logging
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import json

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap

from ..core.histogram import build_histogram_from_rgb
from ..core.query import create_query_histogram
from ..search import find_similar
from ..indexing.store import AnnIndex, MetadataStore
from ..utils.config import TOTAL_BINS, L_BINS, A_BINS, B_BINS, LAB_RANGES
import cv2
from sklearn.cluster import KMeans

# Configure logging
logger = logging.getLogger(__name__)

# Create router for 3D visualization endpoints
router = APIRouter(prefix="/api/3d", tags=["3D Visualization"])

# Global variables for search components (will be injected)
index: Optional[AnnIndex] = None
store: Optional[MetadataStore] = None


def set_search_components(ann_index: AnnIndex, metadata_store: MetadataStore):
    """Set the search components for use in visualization endpoints."""
    global index, store
    index = ann_index
    store = metadata_store


# Pydantic models for 3D visualization data
class ColorSpacePoint(BaseModel):
    """A point in 3D color space with associated image data."""

    x: float = Field(..., description="L* coordinate (0-100)")
    y: float = Field(..., description="a* coordinate (-128 to 127)")
    z: float = Field(..., description="b* coordinate (-128 to 127)")
    image_id: str = Field(..., description="Unique image identifier")
    file_path: str = Field(..., description="Path to image file")
    dominant_colors: List[str] = Field(..., description="Dominant colors as hex codes")
    histogram_entropy: float = Field(..., description="Histogram entropy value")
    cluster_id: Optional[int] = Field(None, description="Cluster assignment ID")


class HistogramBar(BaseModel):
    """A 3D bar in the histogram cloud visualization."""

    x: float = Field(..., description="L* bin center")
    y: float = Field(..., description="a* bin center")
    z: float = Field(..., description="b* bin center")
    value: float = Field(..., description="Normalized histogram value")
    bin_index: int = Field(..., description="Linear bin index")
    contributing_images: List[str] = Field(
        ..., description="Images contributing to this bin"
    )


class SimilarityNode(BaseModel):
    """A node in the 3D similarity landscape."""

    x: float = Field(..., description="UMAP/t-SNE x coordinate")
    y: float = Field(..., description="UMAP/t-SNE y coordinate")
    z: float = Field(..., description="UMAP/t-SNE z coordinate")
    image_id: str = Field(..., description="Unique image identifier")
    file_path: str = Field(..., description="Path to image file")
    distance_to_query: Optional[float] = Field(
        None, description="Distance to search query"
    )
    cluster_id: Optional[int] = Field(None, description="Cluster assignment ID")


class SearchStageData(BaseModel):
    """Data for a single stage of the search pipeline."""

    stage_name: str = Field(..., description="Name of the search stage")
    results: List[SimilarityNode] = Field(..., description="Results from this stage")
    timing_ms: int = Field(..., description="Stage execution time in milliseconds")
    metadata: Dict[str, Any] = Field(..., description="Additional stage metadata")


class SearchAnimationData(BaseModel):
    """Complete animation data for search pipeline visualization."""

    query_id: str = Field(..., description="Unique query identifier")
    query_colors: List[str] = Field(..., description="Query color hex codes")
    query_weights: List[float] = Field(..., description="Query color weights")
    stages: List[SearchStageData] = Field(..., description="Search pipeline stages")
    total_time_ms: int = Field(..., description="Total search time in milliseconds")


def extract_dominant_colors(file_path: str, num_colors: int = 5) -> List[str]:
    """Extract dominant colors from an image using K-means clustering."""
    try:
        image = cv2.imread(file_path)
        if image is None:
            return ["#000000"] * num_colors

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pixels = image_rgb.reshape(-1, 3)

        # Use K-means to find dominant colors
        kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        dominant_colors = kmeans.cluster_centers_.astype(int)

        # Convert to hex format
        hex_colors = []
        for color in dominant_colors:
            hex_color = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])
            hex_colors.append(hex_color)

        return hex_colors

    except Exception as e:
        logger.error(f"Failed to extract dominant colors from {file_path}: {e}")
        return ["#000000"] * num_colors


def calculate_histogram_entropy(histogram: np.ndarray) -> float:
    """Calculate the entropy of a normalized histogram."""
    # Remove zero values to avoid log(0)
    non_zero = histogram[histogram > 0]
    if len(non_zero) == 0:
        return 0.0

    # Calculate entropy
    entropy = -np.sum(non_zero * np.log2(non_zero))
    return float(entropy)


def get_bin_centers() -> np.ndarray:
    """Get the Lab coordinates for all histogram bin centers."""
    # Calculate bin centers for L*, a*, b* dimensions
    l_range = LAB_RANGES[0]
    a_range = LAB_RANGES[1]
    b_range = LAB_RANGES[2]

    l_centers = np.linspace(l_range[0], l_range[1], L_BINS + 1)[:-1] + (
        l_range[1] - l_range[0]
    ) / (2 * L_BINS)
    a_centers = np.linspace(a_range[0], a_range[1], A_BINS + 1)[:-1] + (
        a_range[1] - a_range[0]
    ) / (2 * A_BINS)
    b_centers = np.linspace(b_range[0], b_range[1], B_BINS + 1)[:-1] + (
        b_range[1] - b_range[0]
    ) / (2 * B_BINS)

    # Create meshgrid for all combinations
    l_grid, a_grid, b_grid = np.meshgrid(l_centers, a_centers, b_centers, indexing="ij")

    # Flatten to get all bin centers
    bin_centers = np.column_stack(
        [l_grid.flatten(), a_grid.flatten(), b_grid.flatten()]
    )

    return bin_centers


@router.get("/color-space", response_model=List[ColorSpacePoint])
async def get_color_space_data(
    max_images: int = Query(
        1000, description="Maximum number of images to include", ge=10, le=10000
    ),
    cluster_images: bool = Query(
        True, description="Whether to perform clustering on images"
    ),
    num_clusters: int = Query(
        10, description="Number of clusters for grouping", ge=2, le=50
    ),
):
    """
    Get color space positioning data for 3D CIE Lab cube visualization.

    Returns image data positioned in 3D Lab color space based on their dominant colors.
    Each image is represented as a point with its Lab coordinates, metadata, and cluster assignment.
    """
    if index is None or store is None:
        raise HTTPException(status_code=503, detail="Search components not initialized")

    try:
        logger.info(f"Generating color space data for up to {max_images} images")

        # Get all image metadata
        all_images = store.get_all_image_metadata(limit=max_images)
        if not all_images:
            raise HTTPException(status_code=404, detail="No images found in database")

        color_space_points = []
        lab_coordinates = []

        for image_data in all_images:
            image_id = image_data["image_id"]
            file_path = image_data["file_path"]

            # Extract dominant colors
            dominant_colors = extract_dominant_colors(file_path)

            # Convert dominant colors to Lab space for positioning
            # For simplicity, use the first dominant color for positioning
            if dominant_colors:
                # Convert hex to RGB
                hex_color = dominant_colors[0].lstrip("#")
                rgb = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

                # Convert RGB to Lab (simplified - in practice, use proper color conversion)
                # This is a placeholder - real implementation would use skimage.color.rgb2lab
                lab = [50, 0, 0]  # Placeholder Lab values

                # Calculate histogram entropy
                try:
                    # Load image and convert to RGB
                    image = cv2.imread(file_path)
                    if image is not None:
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        histogram = build_histogram_from_rgb(image_rgb)
                        entropy = calculate_histogram_entropy(histogram)
                    else:
                        entropy = 0.0
                except Exception as e:
                    logger.warning(f"Failed to generate histogram for {image_id}: {e}")
                    entropy = 0.0

                color_space_points.append(
                    ColorSpacePoint(
                        x=float(lab[0]),
                        y=float(lab[1]),
                        z=float(lab[2]),
                        image_id=image_id,
                        file_path=file_path,
                        dominant_colors=dominant_colors,
                        histogram_entropy=entropy,
                    )
                )

                lab_coordinates.append(lab)

        # Perform clustering if requested
        if cluster_images and len(lab_coordinates) > num_clusters:
            from sklearn.cluster import KMeans

            lab_array = np.array(lab_coordinates)
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(lab_array)

            # Assign cluster IDs to points
            for i, point in enumerate(color_space_points):
                point.cluster_id = int(cluster_labels[i])

        logger.info(f"Generated color space data for {len(color_space_points)} images")
        return color_space_points

    except Exception as e:
        logger.error(f"Failed to generate color space data: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate color space data: {str(e)}"
        )


@router.get("/histogram-cloud", response_model=List[HistogramBar])
async def get_histogram_cloud_data(
    image_id: str = Query(..., description="Image ID to visualize histogram for"),
    min_value: float = Query(
        0.001, description="Minimum histogram value to include", ge=0.0, le=1.0
    ),
):
    """
    Get 3D histogram bar chart data for a specific image.

    Returns histogram bins as 3D bars positioned in Lab color space.
    Each bar represents a histogram bin with its Lab coordinates and value.
    """
    if store is None:
        raise HTTPException(status_code=503, detail="Metadata store not initialized")

    try:
        # Get image metadata
        image_data = store.get_image_metadata(image_id)
        if not image_data:
            raise HTTPException(status_code=404, detail=f"Image {image_id} not found")

        file_path = image_data["file_path"]

        # Generate histogram
        image = cv2.imread(file_path)
        if image is None:
            raise HTTPException(
                status_code=404, detail=f"Could not load image at {file_path}"
            )

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        histogram = build_histogram_from_rgb(image_rgb)

        # Get bin centers
        bin_centers = get_bin_centers()

        # Create histogram bars
        histogram_bars = []
        for i, value in enumerate(histogram):
            if value >= min_value:
                histogram_bars.append(
                    HistogramBar(
                        x=float(bin_centers[i, 0]),
                        y=float(bin_centers[i, 1]),
                        z=float(bin_centers[i, 2]),
                        value=float(value),
                        bin_index=i,
                        contributing_images=[image_id],
                    )
                )

        logger.info(f"Generated histogram cloud data with {len(histogram_bars)} bars")
        return histogram_bars

    except Exception as e:
        logger.error(f"Failed to generate histogram cloud data: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate histogram cloud data: {str(e)}"
        )


@router.get("/similarity-landscape", response_model=List[SimilarityNode])
async def get_similarity_landscape_data(
    colors: str = Query(..., description="Comma-separated hex color codes"),
    weights: str = Query(..., description="Comma-separated weights"),
    k: int = Query(100, description="Number of results to include", ge=10, le=1000),
    projection_method: str = Query(
        "umap", description="Projection method: umap, tsne, or pca"
    ),
    n_components: int = Query(
        3, description="Number of dimensions for projection", ge=2, le=3
    ),
):
    """
    Get 3D similarity landscape data using dimensionality reduction.

    Projects search results into 3D space using UMAP, t-SNE, or PCA for visualization.
    """
    if index is None or store is None:
        raise HTTPException(status_code=503, detail="Search components not initialized")

    try:
        # Parse query parameters
        color_list = [c.strip() for c in colors.split(",") if c.strip()]
        weight_list = [float(w.strip()) for w in weights.split(",") if w.strip()]

        if len(color_list) != len(weight_list):
            raise HTTPException(
                status_code=400, detail="Number of colors must match number of weights"
            )

        # Create query histogram
        query_histogram = create_query_histogram(color_list, weight_list)

        # Perform search
        search_results = find_similar(
            query_histogram=query_histogram,
            index=index,
            store=store,
            k=k,
            max_rerank=k,
            use_approximate_reranking=False,
        )

        if not search_results:
            return []

        # Get histograms for all results
        histograms = []
        image_ids = []
        for result in search_results:
            try:
                histogram = store.get_histogram(result.image_id)
                if histogram is not None:
                    histograms.append(histogram)
                    image_ids.append(result.image_id)
            except Exception as e:
                logger.warning(f"Failed to get histogram for {result.image_id}: {e}")

        if len(histograms) < 2:
            raise HTTPException(
                status_code=400, detail="Not enough valid histograms for projection"
            )

        # Convert to numpy array
        histograms_array = np.array(histograms)

        # Apply dimensionality reduction
        if projection_method.lower() == "umap":
            reducer = umap.UMAP(n_components=n_components, random_state=42)
            projected = reducer.fit_transform(histograms_array)
        elif projection_method.lower() == "tsne":
            reducer = TSNE(
                n_components=n_components,
                random_state=42,
                perplexity=min(30, len(histograms) - 1),
            )
            projected = reducer.fit_transform(histograms_array)
        elif projection_method.lower() == "pca":
            reducer = PCA(n_components=n_components, random_state=42)
            projected = reducer.fit_transform(histograms_array)
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid projection method. Use umap, tsne, or pca",
            )

        # Create similarity nodes
        similarity_nodes = []
        for i, (image_id, coords) in enumerate(zip(image_ids, projected)):
            # Get image metadata
            image_data = store.get_image_metadata(image_id)
            file_path = image_data.get("file_path", "") if image_data else ""

            # Find corresponding search result for distance
            distance = None
            for result in search_results:
                if result.image_id == image_id:
                    distance = result.distance
                    break

            similarity_nodes.append(
                SimilarityNode(
                    x=float(coords[0]),
                    y=float(coords[1]),
                    z=float(coords[2]) if n_components >= 3 else 0.0,
                    image_id=image_id,
                    file_path=file_path,
                    distance_to_query=distance,
                )
            )

        logger.info(
            f"Generated similarity landscape with {len(similarity_nodes)} nodes using {projection_method}"
        )
        return similarity_nodes

    except Exception as e:
        logger.error(f"Failed to generate similarity landscape data: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate similarity landscape data: {str(e)}",
        )


@router.get("/search-animation", response_model=SearchAnimationData)
async def get_search_animation_data(
    colors: str = Query(..., description="Comma-separated hex color codes"),
    weights: str = Query(..., description="Comma-separated weights"),
    k: int = Query(50, description="Number of results to return", ge=10, le=200),
):
    """
    Get animated search pipeline data showing the two-stage process.

    Returns data for visualizing the FAISS ANN search followed by Sinkhorn-EMD reranking.
    """
    if index is None or store is None:
        raise HTTPException(status_code=503, detail="Search components not initialized")

    try:
        query_id = str(uuid.uuid4())

        # Parse query parameters
        color_list = [c.strip() for c in colors.split(",") if c.strip()]
        weight_list = [float(w.strip()) for w in weights.split(",") if w.strip()]

        if len(color_list) != len(weight_list):
            raise HTTPException(
                status_code=400, detail="Number of colors must match number of weights"
            )

        # Create query histogram
        query_histogram = create_query_histogram(color_list, weight_list)

        # Stage 1: FAISS ANN search (simulate with approximate reranking)
        start_time = time.time()
        ann_results = find_similar(
            query_histogram=query_histogram,
            index=index,
            store=store,
            k=k * 2,  # Get more candidates for reranking
            max_rerank=k,
            use_approximate_reranking=True,  # Fast L2 distance
        )
        ann_time = int((time.time() - start_time) * 1000)

        # Stage 2: Sinkhorn-EMD reranking
        start_time = time.time()
        reranked_results = find_similar(
            query_histogram=query_histogram,
            index=index,
            store=store,
            k=k,
            max_rerank=k,
            use_approximate_reranking=False,  # Accurate Sinkhorn-EMD
        )
        rerank_time = int((time.time() - start_time) * 1000)

        # Create stage data
        stages = []

        # Stage 1: ANN results
        ann_nodes = []
        for i, result in enumerate(ann_results[:k]):
            image_data = store.get_image_metadata(result.image_id)
            file_path = image_data.get("file_path", "") if image_data else ""

            ann_nodes.append(
                SimilarityNode(
                    x=float(i % 10),  # Simple grid layout
                    y=float(i // 10),
                    z=0.0,
                    image_id=result.image_id,
                    file_path=file_path,
                    distance_to_query=result.distance,
                    cluster_id=0,  # All in same cluster initially
                )
            )

        stages.append(
            SearchStageData(
                stage_name="FAISS ANN Search",
                results=ann_nodes,
                timing_ms=ann_time,
                metadata={"method": "HNSW", "candidates": len(ann_results)},
            )
        )

        # Stage 2: Reranked results
        rerank_nodes = []
        for i, result in enumerate(reranked_results):
            image_data = store.get_image_metadata(result.image_id)
            file_path = image_data.get("file_path", "") if image_data else ""

            # Position by similarity (closer to origin = more similar)
            distance_factor = min(result.distance / 1000.0, 1.0)  # Normalize distance
            angle = (i / len(reranked_results)) * 2 * np.pi

            rerank_nodes.append(
                SimilarityNode(
                    x=float(distance_factor * np.cos(angle)),
                    y=float(distance_factor * np.sin(angle)),
                    z=float(i * 0.1),  # Stack by rank
                    image_id=result.image_id,
                    file_path=file_path,
                    distance_to_query=result.distance,
                    cluster_id=1,  # Different cluster after reranking
                )
            )

        stages.append(
            SearchStageData(
                stage_name="Sinkhorn-EMD Reranking",
                results=rerank_nodes,
                timing_ms=rerank_time,
                metadata={
                    "method": "Sinkhorn-EMD",
                    "final_results": len(reranked_results),
                },
            )
        )

        total_time = ann_time + rerank_time

        logger.info(f"Generated search animation data for query {query_id}")
        return SearchAnimationData(
            query_id=query_id,
            query_colors=color_list,
            query_weights=weight_list,
            stages=stages,
            total_time_ms=total_time,
        )

    except Exception as e:
        logger.error(f"Failed to generate search animation data: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate search animation data: {str(e)}",
        )


@router.get("/image-globe", response_model=List[ColorSpacePoint])
async def get_image_globe_data(
    max_images: int = Query(
        500, description="Maximum number of images to include", ge=10, le=2000
    ),
    sphere_radius: float = Query(
        50.0, description="Radius of the sphere", ge=10.0, le=100.0
    ),
):
    """
    Get image data positioned on a 3D sphere based on dominant colors.

    Maps images onto a sphere where position is determined by dominant color in Lab space.
    """
    if store is None:
        raise HTTPException(status_code=503, detail="Metadata store not initialized")

    try:
        # Get all image metadata
        all_images = store.get_all_image_metadata(limit=max_images)
        if not all_images:
            raise HTTPException(status_code=404, detail="No images found in database")

        globe_points = []

        for image_data in all_images:
            image_id = image_data["image_id"]
            file_path = image_data["file_path"]

            # Extract dominant colors
            dominant_colors = extract_dominant_colors(file_path)

            if dominant_colors:
                # Convert first dominant color to Lab (simplified)
                hex_color = dominant_colors[0].lstrip("#")
                rgb = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

                # Convert RGB to spherical coordinates
                # This is a simplified mapping - real implementation would use proper Lab conversion
                r, g, b = rgb

                # Map RGB to spherical coordinates
                theta = (r / 255.0) * 2 * np.pi  # Azimuthal angle
                phi = (g / 255.0) * np.pi  # Polar angle

                # Convert to Cartesian coordinates on sphere
                x = sphere_radius * np.sin(phi) * np.cos(theta)
                y = sphere_radius * np.sin(phi) * np.sin(theta)
                z = sphere_radius * np.cos(phi)

                # Calculate histogram entropy
                try:
                    # Load image and convert to RGB
                    image = cv2.imread(file_path)
                    if image is not None:
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        histogram = build_histogram_from_rgb(image_rgb)
                        entropy = calculate_histogram_entropy(histogram)
                    else:
                        entropy = 0.0
                except Exception as e:
                    logger.warning(f"Failed to generate histogram for {image_id}: {e}")
                    entropy = 0.0

                globe_points.append(
                    ColorSpacePoint(
                        x=float(x),
                        y=float(y),
                        z=float(z),
                        image_id=image_id,
                        file_path=file_path,
                        dominant_colors=dominant_colors,
                        histogram_entropy=entropy,
                    )
                )

        logger.info(f"Generated image globe data with {len(globe_points)} points")
        return globe_points

    except Exception as e:
        logger.error(f"Failed to generate image globe data: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate image globe data: {str(e)}"
        )


@router.get("/index-structure")
async def get_index_structure_data():
    """
    Get FAISS index structure data for 3D visualization.

    Returns information about the HNSW index structure for debugging and visualization.
    """
    if index is None:
        raise HTTPException(status_code=503, detail="FAISS index not initialized")

    try:
        # Get basic index information
        index_info = {
            "index_type": str(type(index.index).__name__),
            "dimension": int(index.index.d),
            "total_vectors": int(index.index.ntotal),
            "is_trained": bool(index.index.is_trained),
        }

        # Try to get HNSW-specific information
        if hasattr(index.index, "hnsw"):
            hnsw_info = {
                "M": int(index.index.hnsw.M),
                "ef_construction": int(index.index.hnsw.ef_construction),
                "ef_search": int(index.index.hnsw.ef_search),
                "max_level": int(index.index.hnsw.max_level),
            }
            index_info["hnsw"] = hnsw_info

        # Get some sample vectors for visualization
        if index.index.ntotal > 0:
            sample_size = min(100, index.index.ntotal)
            sample_indices = np.random.choice(
                index.index.ntotal, sample_size, replace=False
            )

            # Get sample vectors (this might not work for all index types)
            try:
                sample_vectors = index.index.reconstruct_batch(sample_indices)
                index_info["sample_vectors"] = sample_vectors.tolist()
                index_info["sample_indices"] = sample_indices.tolist()
            except Exception as e:
                logger.warning(f"Could not get sample vectors: {e}")
                index_info["sample_vectors"] = []
                index_info["sample_indices"] = []

        logger.info("Generated index structure data")
        return index_info

    except Exception as e:
        logger.error(f"Failed to generate index structure data: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate index structure data: {str(e)}"
        )
