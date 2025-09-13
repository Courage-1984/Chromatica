"""
Visualization module for the Chromatica color search engine.

This module provides visual representations of color queries and search results,
including weighted color bars, query color palettes, and result image collages.
"""

from .query_viz import (
    QueryVisualizer,
    ResultCollageBuilder,
    create_query_visualization,
    create_results_collage,
)

__all__ = [
    "QueryVisualizer",
    "ResultCollageBuilder",
    "create_query_visualization",
    "create_results_collage",
]
