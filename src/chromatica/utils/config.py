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
# Lower values provide more accurate EMD but may be less stable
SINKHORN_EPSILON = 0.1

# FAISS HNSW index parameters
# M=32 specifies the number of neighbors in the HNSW graph
# This balances search speed with index quality
HNSW_M = 32

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
