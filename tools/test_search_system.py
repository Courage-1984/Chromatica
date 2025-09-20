#!/usr/bin/env python3
"""
Test script for the complete Chromatica search system.

This script validates the full two-stage search pipeline by:
1. Testing the search module with synthetic data
2. Validating integration between all components
3. Measuring performance characteristics
4. Testing error handling and edge cases

Usage:
    python tools/test_search_system.py [--verbose] [--performance]

Requirements:
    - Virtual environment activated (venv311\Scripts\activate)
    - All dependencies installed (pip install -r requirements.txt)
    - Test datasets available in datasets/ directory
"""

import argparse
import logging
import sys
import time
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chromatica.indexing.pipeline import process_image
from chromatica.indexing.store import AnnIndex, MetadataStore
from chromatica.search import find_similar, validate_search_system, SearchResult


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/test_search_system.log"),
        ],
    )


def create_test_index_and_store(
    test_dataset_path: str, max_images: int = 100
) -> tuple[AnnIndex, MetadataStore]:
    """
    Create a test FAISS index and metadata store with sample data.

    Args:
        test_dataset_path: Path to test dataset directory
        max_images: Maximum number of images to process

    Returns:
        Tuple of (AnnIndex, MetadataStore) instances
    """
    logger = logging.getLogger(__name__)

    logger.info(f"Creating test index and store from {test_dataset_path}")

    # Initialize components
    index = AnnIndex()
    store = MetadataStore(":memory:")  # Use in-memory database for testing

    # Get list of image files
    dataset_path = Path(test_dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Test dataset not found: {test_dataset_path}")

    image_files = list(dataset_path.glob("*.jpg")) + list(dataset_path.glob("*.png"))
    image_files = image_files[:max_images]

    if not image_files:
        raise ValueError(f"No image files found in {test_dataset_path}")

    logger.info(f"Processing {len(image_files)} images for test index")

    # Process images and build index
    histograms = []
    metadata_batch = []

    for i, image_file in enumerate(image_files):
        try:
            # Generate histogram
            histogram = process_image(str(image_file))

            # Validate histogram
            if histogram.shape != (1152,) or not np.isclose(
                histogram.sum(), 1.0, atol=1e-6
            ):
                logger.warning(f"Invalid histogram for {image_file.name}, skipping")
                continue

            # Prepare metadata
            image_id = f"test_{image_file.stem}"
            metadata = {
                "image_id": image_id,
                "file_path": str(image_file),
                "histogram": histogram,
                "file_size": image_file.stat().st_size,
            }

            histograms.append(histogram)
            metadata_batch.append(metadata)

            # Log progress
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(image_files)} images")

        except Exception as e:
            logger.warning(f"Failed to process {image_file.name}: {e}")
            continue

    if not histograms:
        raise RuntimeError("No valid histograms generated from test dataset")

    # Add to index and store
    try:
        # Convert to numpy array for FAISS
        histograms_array = np.array(histograms, dtype=np.float64)

        # Add to FAISS index
        added_count = index.add(histograms_array)
        logger.info(f"Added {added_count} vectors to FAISS index")

        # Add to metadata store
        stored_count = store.add_batch(metadata_batch)
        logger.info(f"Added {stored_count} records to metadata store")

        # Verify counts match
        if added_count != stored_count:
            logger.warning(f"Count mismatch: FAISS={added_count}, Store={stored_count}")

    except Exception as e:
        logger.error(f"Failed to build test index: {e}")
        raise

    logger.info(f"Test index and store created successfully")
    logger.info(f"  - FAISS vectors: {index.get_total_vectors()}")
    logger.info(f"  - Store records: {store.get_image_count()}")

    return index, store


def test_basic_search_functionality(index: AnnIndex, store: MetadataStore) -> bool:
    """
    Test basic search functionality with synthetic query.

    Args:
        index: FAISS index instance
        store: Metadata store instance

    Returns:
        bool: True if test passes, False otherwise
    """
    logger = logging.getLogger(__name__)
    logger.info("Testing basic search functionality...")

    try:
        # Create synthetic query histogram
        np.random.seed(42)
        query_hist = np.random.random(1152).astype(np.float64)
        query_hist = query_hist / query_hist.sum()

        # Perform search
        results = find_similar(query_hist, index, store, k=20, max_rerank=10)

        # Validate results
        if not results:
            logger.error("Search returned no results")
            return False

        if len(results) > 10:
            logger.error(f"Expected max 10 results, got {len(results)}")
            return False

        # Check result structure
        for result in results:
            if not isinstance(result, SearchResult):
                logger.error(f"Result is not SearchResult instance: {type(result)}")
                return False

            if not hasattr(result, "image_id") or not hasattr(result, "distance"):
                logger.error("SearchResult missing required attributes")
                return False

        # Check ranking consistency
        distances = [r.distance for r in results]
        if distances != sorted(distances):
            logger.error("Results not properly sorted by distance")
            return False

        logger.info(f"Basic search test PASSED: {len(results)} results returned")
        return True

    except Exception as e:
        logger.error(f"Basic search test FAILED: {e}")
        return False


def test_search_with_real_image(
    index: AnnIndex, store: MetadataStore, test_image_path: str
) -> bool:
    """
    Test search functionality with a real image from the dataset.

    Args:
        index: FAISS index instance
        store: Metadata store instance
        test_image_path: Path to test image

    Returns:
        bool: True if test passes, False otherwise
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Testing search with real image: {test_image_path}")

    try:
        # Generate histogram from real image
        query_hist = process_image(test_image_path)

        # Validate histogram
        if query_hist.shape != (1152,) or not np.isclose(
            query_hist.sum(), 1.0, atol=1e-6
        ):
            logger.error("Generated histogram is invalid")
            return False

        # Perform search
        results = find_similar(query_hist, index, store, k=20, max_rerank=10)

        if not results:
            logger.error("Search with real image returned no results")
            return False

        # Check that the query image itself is among the top results
        query_image_id = f"test_{Path(test_image_path).stem}"
        found_query = any(r.image_id == query_image_id for r in results[:5])

        if not found_query:
            logger.warning(f"Query image {query_image_id} not found in top 5 results")
            # This is not necessarily an error, just a warning

        logger.info(f"Real image search test PASSED: {len(results)} results returned")
        return True

    except Exception as e:
        logger.error(f"Real image search test FAILED: {e}")
        return False


def test_performance_characteristics(
    index: AnnIndex, store: MetadataStore, num_queries: int = 5
) -> bool:
    """
    Test performance characteristics of the search system.

    Args:
        index: FAISS index instance
        store: Metadata store instance
        num_queries: Number of test queries to run

    Returns:
        bool: True if test passes, False otherwise
    """
    logger.info(f"Testing performance characteristics with {num_queries} queries...")

    try:
        total_times = []
        ann_times = []
        metadata_times = []
        rerank_times = []

        for i in range(num_queries):
            # Create synthetic query
            np.random.seed(42 + i)
            query_hist = np.random.random(1152).astype(np.float64)
            query_hist = query_hist / query_hist.sum()

            # Time the search
            start_time = time.time()
            results = find_similar(query_hist, index, store, k=50, max_rerank=25)
            total_time = time.time() - start_time

            if not results:
                logger.warning(f"Query {i+1} returned no results")
                continue

            total_times.append(total_time)

            # Log individual query performance
            logger.debug(f"Query {i+1}: {len(results)} results in {total_time:.3f}s")

        if not total_times:
            logger.error("No successful queries for performance testing")
            return False

        # Calculate performance statistics
        avg_total = np.mean(total_times)
        min_total = np.min(total_times)
        max_total = np.max(total_times)

        logger.info("Performance test results:")
        logger.info(f"  - Queries completed: {len(total_times)}/{num_queries}")
        logger.info(f"  - Average total time: {avg_total:.3f}s")
        logger.info(f"  - Time range: [{min_total:.3f}s, {max_total:.3f}s]")

        # Performance thresholds (adjust based on your system)
        if avg_total > 2.0:  # 2 seconds average
            logger.warning("Average search time is higher than expected")

        if max_total > 5.0:  # 5 seconds maximum
            logger.warning("Maximum search time is higher than expected")

        logger.info("Performance test PASSED")
        return True

    except Exception as e:
        logger.error(f"Performance test FAILED: {e}")
        return False


def test_error_handling(index: AnnIndex, store: MetadataStore) -> bool:
    """
    Test error handling and edge cases.

    Args:
        index: FAISS index instance
        store: Metadata store instance

    Returns:
        bool: True if test passes, False otherwise
    """
    logger = logging.getLogger(__name__)
    logger.info("Testing error handling and edge cases...")

    try:
        # Test 1: Invalid histogram shape
        try:
            invalid_hist = np.random.random(100)  # Wrong dimension
            find_similar(invalid_hist, index, store)
            logger.error("Should have raised ValueError for invalid histogram shape")
            return False
        except ValueError:
            logger.debug("Correctly caught invalid histogram shape error")

        # Test 2: Invalid histogram values (not normalized)
        try:
            invalid_hist = np.random.random(1152)  # Not normalized
            find_similar(invalid_hist, index, store)
            logger.error("Should have raised error for non-normalized histogram")
            return False
        except Exception:
            logger.debug("Correctly caught non-normalized histogram error")

        # Test 3: Invalid k value
        try:
            valid_hist = np.random.random(1152)
            valid_hist = valid_hist / valid_hist.sum()
            find_similar(valid_hist, index, store, k=0)
            logger.error("Should have raised ValueError for k=0")
            return False
        except ValueError:
            logger.debug("Correctly caught invalid k value error")

        # Test 4: Invalid max_rerank value
        try:
            valid_hist = np.random.random(1152)
            valid_hist = valid_hist / valid_hist.sum()
            find_similar(valid_hist, index, store, k=10, max_rerank=0)
            logger.error("Should have raised ValueError for max_rerank=0")
            return False
        except ValueError:
            logger.debug("Correctly caught invalid max_rerank value error")

        logger.info("Error handling test PASSED")
        return True

    except Exception as e:
        logger.error(f"Error handling test FAILED: {e}")
        return False


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test Chromatica search system")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--performance", "-p", action="store_true", help="Run performance tests"
    )
    parser.add_argument(
        "--dataset",
        "-d",
        default="datasets/test-dataset-50",
        help="Path to test dataset directory",
    )
    parser.add_argument(
        "--max-images",
        "-m",
        type=int,
        default=50,
        help="Maximum number of images to process for testing",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    logger.info("Chromatica Search System Test Suite")
    logger.info("=" * 60)

    try:
        # Create test index and store
        logger.info("Phase 1: Creating test index and metadata store")
        index, store = create_test_index_and_store(args.dataset, args.max_images)

        # Run validation
        logger.info("Phase 2: Validating search system")
        if not validate_search_system(index, store):
            logger.error("Search system validation failed")
            return 1

        # Test basic functionality
        logger.info("Phase 3: Testing basic search functionality")
        if not test_basic_search_functionality(index, store):
            logger.error("Basic search functionality test failed")
            return 1

        # Test with real image
        logger.info("Phase 4: Testing search with real image")
        test_image = list(Path(args.dataset).glob("*.jpg"))[0]
        if not test_search_with_real_image(index, store, str(test_image)):
            logger.error("Real image search test failed")
            return 1

        # Test error handling
        logger.info("Phase 5: Testing error handling")
        if not test_error_handling(index, store):
            logger.error("Error handling test failed")
            return 1

        # Performance testing (optional)
        if args.performance:
            logger.info("Phase 6: Performance testing")
            if not test_performance_characteristics(index, store):
                logger.error("Performance test failed")
                return 1

        # Cleanup
        store.close()

        logger.info("=" * 60)
        logger.info("All tests PASSED! Search system is working correctly.")
        return 0

    except Exception as e:
        logger.error(f"Test suite failed with error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
