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

import os
from pathlib import Path
from dataclasses import dataclass

from typing import Callable, Optional

# CRITICAL FIX: Import the types needed for the GlobalState dataclass
from ..indexing.store import AnnIndex, MetadataStore

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
RERANK_K = 500

# Max number of final search results to return from the API endpoint
MAX_SEARCH_RESULTS = 50

# Max number of color-weight pairs allowed in a single query
MAX_COLOR_COUNT = 5

# =============================================================================
# PERFORMANCE TUNING CONSTANTS
# =============================================================================

# Sinkhorn regularization parameter for Earth Mover's Distance approximation
# Higher values provide more stable EMD but may be less accurate
# Increased from 0.1 to 1.0 for better numerical stability
SINKHORN_EPSILON = 0.05

# Maximum iterations for Sinkhorn (speed/convergence tradeoff)
# Reduced from a likely default of 100 to 50 for faster execution while maintaining convergence.
SINKHORN_MAX_ITER = 50

# Performance optimization parameters
# Early termination threshold for very similar histograms (L2 distance)
EARLY_TERMINATION_THRESHOLD = 1e-6

# Batch size for reranking operations
RERANK_BATCH_SIZE = 10

# Default reranking mode (False = Sinkhorn-EMD, True = approximate L2)
USE_APPROXIMATE_RERANKING = False

# --- Chroma weighting and query shaping (bias control) ---
# C* cutoff below which bins are suppressed (set to 0) at search-time only
CHROMA_CUTOFF = 15.0  # in Lab chroma units
# Sigma controlling growth of chroma weight: w = 1 - exp(-(C*^2)/(2*sigma^2))
CHROMA_SIGMA = 15.0
# Query sharpening exponent (>1 concentrates mass around peaks); 1.0 disables
QUERY_SHARPEN_EXPONENT = 2.2
# Additional L1 alignment penalty in reranking to preserve multi-color balance
RERANK_ALPHA_L1 = 0.45

# Lightness suppression for very bright bins (reduces white bias) during rerank only
LIGHTNESS_SUPPRESS_THRESHOLD = 82.0  # L* above this gets down-weighted
LIGHTNESS_SIGMA = 6.0

# Hue proximity weighting (in rerank): emphasize bins close to query mean (a*, b*)
HUE_SIGMA = 15.0

# Multi-peak hue emphasis configuration (for complex multi-color queries)
# Number of top query bins to center Gaussians on
HUE_MIXTURE_TOP_K = 3
# Global gain applied to hue mixture mask (1.0 keeps absolute scale)
HUE_MIXTURE_GAIN = 1.3

PREFILTER_MIN_MASS = 0.03

# Candidate prefiltering (enforce presence of query hues before reranking)
# Minimum mass that must fall under the query hue-mixture window
PREFILTER_HUE_MIN_MASS = 0.05  # 5% of histogram probability
# If too strict, keep at least this many best-by-hue-mass
PREFILTER_MIN_KEEP = 60

# Verbose search logging
VERBOSE_SEARCH_LOGS = True
LOG_TOP_COLORS_N = 5

# Per-color presence enforcement (for multi-color queries)
# Minimum mass each of the top-K query peaks must have in a candidate
PERCOLOR_MIN_MASS = 0.03
PERCOLOR_TOP_K = 2  # usually number of query colors
PERCOLOR_ENFORCE_STRICT = True

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
IVFPQ_NPROBE = 10

# Additional FAISS optimization parameters
# Enable GPU acceleration if available (future enhancement)
FAISS_USE_GPU = False

# Enable memory mapping for large indices
FAISS_USE_MMAP = True

# Search timeout in milliseconds (prevents hanging on large indices)
FAISS_SEARCH_TIMEOUT_MS = 5000

# --- Directory Constants ---
# These are the configuration variables needed by main.py and indexing scripts
LOG_DIR = Path(os.getenv("CHROMATICA_LOG_DIR", "logs"))
INDEX_FILE = "faiss_index.bin"
DB_FILE = "metadata.db"


# --- Path Utility Functions ---
def get_index_path(output_dir: Path) -> Path:
    """Returns the full path to the FAISS index file."""
    return output_dir / INDEX_FILE


def get_db_path(output_dir: Path) -> Path:
    """Returns the full path to the DuckDB metadata file."""
    return output_dir / DB_FILE


# Define a dataclass to hold the global state components
@dataclass
class GlobalState:
    index: Optional[AnnIndex]
    store: Optional[MetadataStore]
    increment_concurrent_searches: Callable[[], None]
    decrement_concurrent_searches: Callable[[], None]
    update_performance_stats: Callable[[float], None]
    # Add any other global state variables needed by the router here


# This function will be defined and populated in main.py's startup
global_state_container: GlobalState = GlobalState(
    index=None,
    store=None,
    increment_concurrent_searches=lambda: None,
    decrement_concurrent_searches=lambda: None,
    update_performance_stats=lambda x: None,
)


def set_global_state(index, store, increment_func, decrement_func, update_func):
    """Sets the global state container during application startup."""
    global_state_container.index = index
    global_state_container.store = store
    global_state_container.increment_concurrent_searches = increment_func
    global_state_container.decrement_concurrent_searches = decrement_func
    global_state_container.update_performance_stats = update_func


def get_global_state() -> GlobalState:
    """Retrieves the global state container for routers."""
    return global_state_container


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

    if MAX_SEARCH_RESULTS <= 0:
        raise ValueError(
            f"MAX_SEARCH_RESULTS must be positive, got {MAX_SEARCH_RESULTS}"
        )

    if MAX_COLOR_COUNT <= 0:
        raise ValueError(f"MAX_COLOR_COUNT must be positive, got {MAX_COLOR_COUNT}")

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
