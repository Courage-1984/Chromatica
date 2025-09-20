"""
Configuration module for the Chromatica color search engine.

This module contains all global constants and configuration parameters used throughout
the Chromatica application. These constants are derived from the algorithmic specifications
defined in the critical instructions document and ensure consistency across all components
of the color search engine.

Key Configuration Areas:
- Color space binning parameters for CIE Lab color space
- Histogram generation constants
- Search and reranking parameters
- Performance tuning constants

All constants are defined as module-level variables to ensure they can be imported
and used consistently across the application without modification.
"""

from typing import Dict

# =============================================================================
# COLOR SPACE BINNING PARAMETERS
# =============================================================================

# Number of bins for each dimension in the CIE Lab color space
# These values are optimized for the typical sRGB gamut coverage in Lab space
L_BINS = 8  # Lightness dimension: 8 bins over range [0, 100]
A_BINS = 12  # Green-Red dimension: 12 bins over range [-86, 98]
B_BINS = 12  # Blue-Yellow dimension: 12 bins over range [-108, 95]

# Total number of histogram dimensions
# This creates a 1,152-dimensional feature vector for each image
TOTAL_BINS = L_BINS * A_BINS * B_BINS

# =============================================================================
# LAB COLOR SPACE RANGES
# =============================================================================

# Valid ranges for each dimension in CIE Lab color space (D65 illuminant)
# These ranges define the boundaries for histogram binning
LAB_RANGES = [
    [0.0, 100.0],  # L* (Lightness): 0 = black, 100 = white
    [-86.0, 98.0],  # a* (Green-Red): negative = green, positive = red
    [-108.0, 95.0],  # b* (Blue-Yellow): negative = blue, positive = yellow
]

# =============================================================================
# SEARCH AND RERANKING PARAMETERS
# =============================================================================

# Number of candidates to retrieve from the ANN index for reranking
# This balances search speed with reranking accuracy
RERANK_K = 200

# =============================================================================
# PERFORMANCE TUNING CONSTANTS
# =============================================================================

# Sinkhorn regularization parameter for Earth Mover's Distance approximation
# Higher values provide more stable EMD but may be less accurate
# Increased from 0.1 to 1.0 for better numerical stability
SINKHORN_EPSILON = 1.0

# Performance optimization parameters
# Early termination threshold for very similar histograms (L2 distance)
EARLY_TERMINATION_THRESHOLD = 1e-6

# Batch size for reranking operations
RERANK_BATCH_SIZE = 10

# Default reranking mode (False = Sinkhorn-EMD, True = approximate L2)
USE_APPROXIMATE_RERANKING = False

# FAISS IndexIVFPQ parameters for memory-efficient indexing
# These parameters control the Product Quantization (PQ) compression
# Optimized for speed/accuracy trade-off in color search applications

# Number of Voronoi cells (clusters) for coarse quantization
# More clusters = better accuracy but more memory usage
# Note: FAISS recommends at least nlist * 39 training points
# Optimized for faster search with good accuracy
IVFPQ_NLIST = 32  # Reduced from 50 for faster search

# Number of subquantizers for Product Quantization
# Each subquantizer handles dimension/M components
# Must divide the total dimension evenly (1152 / M must be integer)
IVFPQ_M = 8  # 1152 / 8 = 144 dimensions per subquantizer

# Number of bits per subquantizer (2^nbits centroids per subquantizer)
# Higher values = better accuracy but more memory
IVFPQ_NBITS = 8  # 2^8 = 256 centroids per subquantizer

# Number of clusters to probe during search
# Higher values = better recall but slower search
# Optimized for speed while maintaining good accuracy
IVFPQ_NPROBE = 8  # Reduced from 10 for faster search

# Additional FAISS optimization parameters
# Enable GPU acceleration if available (future enhancement)
FAISS_USE_GPU = False

# Enable memory mapping for large indices
FAISS_USE_MMAP = True

# Search timeout in milliseconds (prevents hanging on large indices)
FAISS_SEARCH_TIMEOUT_MS = 5000


# Adaptive search parameters based on dataset size
def get_adaptive_search_params(dataset_size: int) -> Dict[str, int]:
    """
    Get adaptive FAISS search parameters based on dataset size.

    This function optimizes search parameters based on the size of the dataset
    to provide the best speed/accuracy trade-off for different scales.

    Args:
        dataset_size: Number of images in the dataset

    Returns:
        Dictionary with optimized search parameters
    """
    if dataset_size < 1000:
        # Small datasets: use simple index for best accuracy
        return {
            "use_simple_index": True,
            "nlist": 0,
            "nprobe": 0,
            "rerank_k": min(50, dataset_size),
        }
    elif dataset_size < 10000:
        # Medium datasets: balanced parameters
        return {
            "use_simple_index": False,
            "nlist": 16,
            "nprobe": 4,
            "rerank_k": min(100, dataset_size),
        }
    elif dataset_size < 100000:
        # Large datasets: optimized for speed
        return {
            "use_simple_index": False,
            "nlist": 32,
            "nprobe": 8,
            "rerank_k": min(200, dataset_size),
        }
    else:
        # Very large datasets: maximum speed optimization
        return {
            "use_simple_index": False,
            "nlist": 64,
            "nprobe": 16,
            "rerank_k": min(500, dataset_size),
        }


# =============================================================================
# IMAGE PROCESSING CONSTANTS
# =============================================================================

# Maximum image dimension for processing
# Images are resized to maintain this as the maximum side length
# This balances processing speed with color detail preservation
MAX_IMAGE_DIMENSION = 256

# =============================================================================
# VALIDATION CONSTANTS
# =============================================================================


def validate_config():
    """
    Validates that all configuration constants are consistent and valid.

    Raises:
        ValueError: If any configuration constants are invalid.
    """
    if TOTAL_BINS != L_BINS * A_BINS * B_BINS:
        raise ValueError(
            f"TOTAL_BINS ({TOTAL_BINS}) must equal L_BINS * A_BINS * B_BINS ({L_BINS * A_BINS * B_BINS})"
        )

    if RERANK_K <= 0:
        raise ValueError(f"RERANK_K must be positive, got {RERANK_K}")

    if SINKHORN_EPSILON <= 0:
        raise ValueError(f"SINKHORN_EPSILON must be positive, got {SINKHORN_EPSILON}")

    if MAX_IMAGE_DIMENSION <= 0:
        raise ValueError(
            f"MAX_IMAGE_DIMENSION must be positive, got {MAX_IMAGE_DIMENSION}"
        )

    # Validate IVFPQ parameters
    if IVFPQ_NLIST <= 0:
        raise ValueError(f"IVFPQ_NLIST must be positive, got {IVFPQ_NLIST}")

    if IVFPQ_M <= 0:
        raise ValueError(f"IVFPQ_M must be positive, got {IVFPQ_M}")

    if TOTAL_BINS % IVFPQ_M != 0:
        raise ValueError(
            f"TOTAL_BINS ({TOTAL_BINS}) must be divisible by IVFPQ_M ({IVFPQ_M})"
        )

    if IVFPQ_NBITS <= 0 or IVFPQ_NBITS > 16:
        raise ValueError(f"IVFPQ_NBITS must be between 1 and 16, got {IVFPQ_NBITS}")

    if IVFPQ_NPROBE <= 0 or IVFPQ_NPROBE > IVFPQ_NLIST:
        raise ValueError(
            f"IVFPQ_NPROBE must be between 1 and {IVFPQ_NLIST}, got {IVFPQ_NPROBE}"
        )

    # Validate LAB ranges
    dim_names = ["L*", "a*", "b*"]
    for dim_name, (min_val, max_val) in zip(dim_names, LAB_RANGES):
        if min_val >= max_val:
            raise ValueError(f"{dim_name} range invalid: {min_val} >= {max_val}")

    print("Configuration validation passed successfully!")


# Auto-validate configuration when module is imported
if __name__ == "__main__":
    validate_config()
else:
    # Validate configuration on import (but only print if there's an error)
    try:
        validate_config()
    except ValueError as e:
        print(f"Configuration validation failed: {e}")
        raise
