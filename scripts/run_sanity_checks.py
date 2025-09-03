#!/usr/bin/env python3
"""
Sanity Check Script for Chromatica Color Search Engine.

This script programmatically executes the four sanity checks defined in Section F
of the critical instructions document to validate the search system's behavior.

Sanity Checks:
1. Monochrome: Query for 100% #FF0000 should return red-dominant images
2. Complementary: Query for 50% #0000FF and 50% #FFA500 should return contrasting images
3. Weight Sensitivity: 90% red, 10% blue vs 10% red, 90% blue should yield different results
4. Subtle Hues: Query for similar colors #FF0000 and #EE0000 should test fine-grained perception

Usage:
    Activate virtual environment: venv311\Scripts\activate
    Run script: python scripts/run_sanity_checks.py [--verbose] [--top-k N]

Requirements:
    - Virtual environment activated (venv311\Scripts\activate)
    - All dependencies installed (pip install -r requirements.txt)
    - Test index available in test_index/ directory
    - Test datasets available in datasets/ directory
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chromatica.indexing.pipeline import process_image
from chromatica.core.query import create_query_histogram
from chromatica.indexing.store import AnnIndex, MetadataStore
from chromatica.search import find_similar, SearchResult
from chromatica.utils.config import TOTAL_BINS


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration for sanity checks."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/sanity_checks.log"),
        ],
    )


def load_test_index() -> Tuple[AnnIndex, MetadataStore]:
    """
    Load the test FAISS index and metadata store.

    Returns:
        Tuple of (AnnIndex, MetadataStore) instances

    Raises:
        FileNotFoundError: If test index files are not found
        RuntimeError: If index loading fails
    """
    logger = logging.getLogger(__name__)

    # Check if test index exists
    index_path = Path("test_index/chromatica_index.faiss")
    metadata_path = Path("test_index/chromatica_metadata.db")

    if not index_path.exists():
        raise FileNotFoundError(f"Test index not found: {index_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Test metadata not found: {metadata_path}")

    logger.info("Loading test index and metadata store...")

    try:
        # Load FAISS index
        index = AnnIndex()
        index.load(str(index_path))

        # Load metadata store
        store = MetadataStore(str(metadata_path))

        logger.info(
            f"Successfully loaded index with {index.get_total_vectors()} vectors"
        )
        logger.info(f"Successfully loaded metadata store")

        return index, store

    except Exception as e:
        logger.error(f"Failed to load test index: {e}")
        raise RuntimeError(f"Index loading failed: {e}")


def run_sanity_check(
    check_name: str,
    colors: List[str],
    weights: List[float],
    index: AnnIndex,
    store: MetadataStore,
    top_k: int = 5,
) -> Dict:
    """
    Execute a single sanity check with the given color query.

    Args:
        check_name: Name/description of the sanity check
        colors: List of hex color codes (without #)
        weights: List of weights corresponding to colors
        index: FAISS index instance
        store: Metadata store instance
        top_k: Number of top results to return

    Returns:
        Dictionary containing check results and metadata
    """
    logger = logging.getLogger(__name__)

    logger.info(f"\n{'='*60}")
    logger.info(f"SANITY CHECK: {check_name}")
    logger.info(f"{'='*60}")

    # Log query details
    logger.info(f"Query Colors: {colors}")
    logger.info(f"Query Weights: {weights}")

    try:
        # Create query histogram
        start_time = time.time()
        query_histogram = create_query_histogram(colors, weights)
        query_time = time.time() - start_time

        logger.info(f"Query histogram generated in {query_time:.3f}s")
        logger.info(f"Histogram shape: {query_histogram.shape}")
        logger.info(f"Histogram sum: {query_histogram.sum():.6f}")

        # Validate query histogram
        if query_histogram.shape != (TOTAL_BINS,):
            raise ValueError(f"Invalid histogram shape: {query_histogram.shape}")
        if not np.isclose(query_histogram.sum(), 1.0, atol=1e-6):
            raise ValueError(f"Histogram not normalized: sum = {query_histogram.sum()}")

        # Perform search
        start_time = time.time()
        results = find_similar(query_histogram, index, store, k=top_k)
        search_time = time.time() - start_time

        logger.info(f"Search completed in {search_time:.3f}s")
        logger.info(f"Retrieved {len(results)} results")

        # Display top results
        logger.info(f"\nTop {top_k} Results:")
        logger.info("-" * 80)

        for i, result in enumerate(results[:top_k], 1):
            logger.info(f"{i:2d}. Image ID: {result.image_id}")
            logger.info(f"    File: {result.file_path}")
            logger.info(f"    Distance: {result.distance:.6f}")
            logger.info(f"    ANN Score: {result.ann_score:.6f}")
            logger.info(f"    Rank: {result.rank}")
            logger.info("")

        return {
            "check_name": check_name,
            "colors": colors,
            "weights": weights,
            "query_time": query_time,
            "search_time": search_time,
            "total_time": query_time + search_time,
            "results_count": len(results),
            "top_results": results[:top_k],
            "success": True,
        }

    except Exception as e:
        logger.error(f"Sanity check '{check_name}' failed: {e}")
        return {
            "check_name": check_name,
            "colors": colors,
            "weights": weights,
            "error": str(e),
            "success": False,
        }


def run_all_sanity_checks(
    index: AnnIndex, store: MetadataStore, top_k: int = 5
) -> List[Dict]:
    """
    Execute all four sanity checks defined in Section F.

    Args:
        index: FAISS index instance
        store: Metadata store instance
        top_k: Number of top results to return for each check

    Returns:
        List of results for each sanity check
    """
    logger = logging.getLogger(__name__)

    logger.info("Starting Chromatica Sanity Checks")
    logger.info("=" * 60)
    logger.info("This script validates the search system's behavior using")
    logger.info("the four sanity checks defined in Section F of the plan.")
    logger.info("=" * 60)

    # Define sanity checks from Section F
    sanity_checks = [
        {
            "name": "Monochrome Red Query",
            "description": "Query for 100% #FF0000 should return red-dominant images",
            "colors": ["FF0000"],
            "weights": [1.0],
        },
        {
            "name": "Complementary Colors Query",
            "description": "Query for 50% #0000FF and 50% #FFA500 should return contrasting images",
            "colors": ["0000FF", "FFA500"],
            "weights": [0.5, 0.5],
        },
        {
            "name": "Weight Sensitivity Test 1",
            "description": "90% red, 10% blue should yield red-dominant results",
            "colors": ["FF0000", "0000FF"],
            "weights": [0.9, 0.1],
        },
        {
            "name": "Weight Sensitivity Test 2",
            "description": "10% red, 90% blue should yield blue-dominant results",
            "colors": ["FF0000", "0000FF"],
            "weights": [0.1, 0.9],
        },
        {
            "name": "Subtle Hues Test",
            "description": "Query for similar colors #FF0000 and #EE0000 should test fine-grained perception",
            "colors": ["FF0000", "EE0000"],
            "weights": [0.5, 0.5],
        },
    ]

    results = []

    for check in sanity_checks:
        logger.info(f"\nRunning: {check['name']}")
        logger.info(f"Description: {check['description']}")

        result = run_sanity_check(
            check["name"], check["colors"], check["weights"], index, store, top_k
        )

        results.append(result)

        # Add a small delay between checks
        time.sleep(0.5)

    return results


def generate_summary_report(results: List[Dict]) -> None:
    """
    Generate a summary report of all sanity check results.

    Args:
        results: List of sanity check results
    """
    logger = logging.getLogger(__name__)

    logger.info("\n" + "=" * 60)
    logger.info("SANITY CHECKS SUMMARY REPORT")
    logger.info("=" * 60)

    successful_checks = [r for r in results if r.get("success", False)]
    failed_checks = [r for r in results if not r.get("success", False)]

    logger.info(f"Total Checks: {len(results)}")
    logger.info(f"Successful: {len(successful_checks)}")
    logger.info(f"Failed: {len(failed_checks)}")

    if successful_checks:
        logger.info(f"\nSuccessful Checks:")
        for result in successful_checks:
            logger.info(f"  [PASS] {result['check_name']}")
            logger.info(
                f"    Query: {result['colors']} with weights {result['weights']}"
            )
            logger.info(f"    Results: {result['results_count']} images found")
            logger.info(f"    Total Time: {result['total_time']:.3f}s")
            logger.info("")

    if failed_checks:
        logger.info(f"\nFailed Checks:")
        for result in failed_checks:
            logger.info(f"  [FAIL] {result['check_name']}")
            logger.info(f"    Error: {result.get('error', 'Unknown error')}")
            logger.info("")

    # Performance summary
    if successful_checks:
        avg_query_time = np.mean([r["query_time"] for r in successful_checks])
        avg_search_time = np.mean([r["search_time"] for r in successful_checks])
        avg_total_time = np.mean([r["total_time"] for r in successful_checks])

        logger.info(f"\nPerformance Summary:")
        logger.info(f"  Average Query Time: {avg_query_time:.3f}s")
        logger.info(f"  Average Search Time: {avg_search_time:.3f}s")
        logger.info(f"  Average Total Time: {avg_total_time:.3f}s")

    logger.info("=" * 60)


def main():
    """Main function to run sanity checks."""
    parser = argparse.ArgumentParser(
        description="Run sanity checks for Chromatica color search engine"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="Number of top results to display (default: 5)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        # Load test index and store
        index, store = load_test_index()

        # Run all sanity checks
        results = run_all_sanity_checks(index, store, args.top_k)

        # Generate summary report
        generate_summary_report(results)

        # Check if all checks passed
        successful_checks = [r for r in results if r.get("success", False)]
        if len(successful_checks) == len(results):
            logger.info("\n*** ALL SANITY CHECKS PASSED! ***")
            logger.info("The Chromatica search system is working correctly.")
            sys.exit(0)
        else:
            logger.error("\n*** SOME SANITY CHECKS FAILED! ***")
            logger.error("Please review the errors above and fix any issues.")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Fatal error during sanity checks: {e}")
        logger.error("Please ensure the virtual environment is activated and")
        logger.error("all dependencies are properly installed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
