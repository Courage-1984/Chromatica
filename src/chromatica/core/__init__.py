"""
Core module for the Chromatica color search engine.

This module contains the fundamental components for color processing and analysis:
- Histogram generation from images
- Query processing from color specifications
- Reranking using optimal transport methods

All components follow the algorithmic specifications from the critical instructions
and are designed to work together seamlessly in the search pipeline.
"""

from .histogram import build_histogram
from .query import hex_to_lab, create_query_histogram, validate_query_histogram
from .rerank import compute_sinkhorn_distance

__all__ = [
    "build_histogram",
    "hex_to_lab",
    "create_query_histogram",
    "validate_query_histogram",
    "compute_sinkhorn_distance",
]
