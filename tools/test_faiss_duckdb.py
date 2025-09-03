#!/usr/bin/env python3
"""
Test script for FAISS and DuckDB wrapper classes.

This script tests the AnnIndex and MetadataStore classes to ensure they
work correctly with the existing histogram generation pipeline. It creates
sample histograms, indexes them in FAISS, stores metadata in DuckDB, and
performs a test search to validate the complete workflow.

Usage:
    python tools/test_faiss_duckdb.py

Requirements:
    - Virtual environment must be activated: venv311\Scripts\activate
    - All dependencies installed: pip install -r requirements.txt
"""

import sys
import os
import logging
import numpy as np
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chromatica.core.histogram import build_histogram
from chromatica.indexing.store import AnnIndex, MetadataStore
from chromatica.utils.config import TOTAL_BINS, RERANK_K

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_sample_histograms(n_samples: int = 10) -> np.ndarray:
    """
    Create sample histograms for testing.

    Args:
        n_samples: Number of sample histograms to create.

    Returns:
        Array of shape (n_samples, TOTAL_BINS) containing normalized histograms.
    """
    logger.info(f"Creating {n_samples} sample histograms...")

    # Create random histograms with different characteristics
    histograms = []

    for i in range(n_samples):
        # Create a random histogram with some structure
        if i % 3 == 0:
            # Bright images (high L* values)
            hist = np.random.exponential(0.1, TOTAL_BINS)
            hist[: TOTAL_BINS // 2] *= 0.1  # Suppress dark regions
        elif i % 3 == 1:
            # Colorful images (high a* and b* values)
            hist = np.random.exponential(0.1, TOTAL_BINS)
            hist[TOTAL_BINS // 2 :] *= 2.0  # Enhance color regions
        else:
            # Balanced images
            hist = np.random.exponential(0.1, TOTAL_BINS)

        # Normalize to sum to 1.0
        hist = hist / hist.sum()
        histograms.append(hist)

    histograms_array = np.array(histograms, dtype=np.float32)
    logger.info(f"Created histograms with shape: {histograms_array.shape}")

    return histograms_array


def test_faiss_index(histograms: np.ndarray) -> AnnIndex:
    """
    Test the FAISS index functionality.

    Args:
        histograms: Array of histograms to index.

    Returns:
        Configured AnnIndex instance.
    """
    logger.info("Testing FAISS index functionality...")

    # Create and configure the index
    ann_index = AnnIndex(dimension=TOTAL_BINS)

    # Add histograms to the index
    added_count = ann_index.add(histograms)
    logger.info(f"Added {added_count} histograms to FAISS index")

    # Verify the index has the expected number of vectors
    total_vectors = ann_index.get_total_vectors()
    assert total_vectors == len(
        histograms
    ), f"Expected {len(histograms)} vectors, got {total_vectors}"

    # Test search functionality
    query_histogram = histograms[0]  # Use first histogram as query
    k = min(5, len(histograms))

    distances, indices = ann_index.search(query_histogram, k)
    logger.info(
        f"Search results - distances: {distances[0][:3]}, indices: {indices[0][:3]}"
    )

    # Verify search results
    assert distances.shape == (
        1,
        k,
    ), f"Expected distances shape (1, {k}), got {distances.shape}"
    assert indices.shape == (
        1,
        k,
    ), f"Expected indices shape (1, {k}), got {indices.shape}"

    # The first result should be the query vector itself (distance = 0)
    assert (
        distances[0][0] == 0.0
    ), f"Expected first distance to be 0.0, got {distances[0][0]}"
    assert indices[0][0] == 0, f"Expected first index to be 0, got {indices[0][0]}"

    logger.info("FAISS index test passed successfully!")
    return ann_index


def test_duckdb_store(histograms: np.ndarray) -> MetadataStore:
    """
    Test the DuckDB metadata store functionality.

    Args:
        histograms: Array of histograms to store.

    Returns:
        Configured MetadataStore instance.
    """
    logger.info("Testing DuckDB metadata store functionality...")

    # Create metadata store (in-memory for testing)
    metadata_store = MetadataStore(db_path=":memory:")

    # Create sample metadata
    metadata_batch = []
    for i, hist in enumerate(histograms):
        metadata_batch.append(
            {
                "image_id": f"test_image_{i:03d}",
                "file_path": f"/path/to/test_image_{i:03d}.jpg",
                "histogram": hist,
                "file_size": 1024 * (i + 1),  # Simulate different file sizes
            }
        )

    # Add metadata to the store
    inserted_count = metadata_store.add_batch(metadata_batch)
    logger.info(f"Inserted {inserted_count} metadata records")

    # Verify the store has the expected number of images
    image_count = metadata_store.get_image_count()
    assert image_count == len(
        histograms
    ), f"Expected {len(histograms)} images, got {image_count}"

    # Test histogram retrieval
    image_ids = [f"test_image_{i:03d}" for i in range(min(5, len(histograms)))]
    retrieved_histograms = metadata_store.get_histograms_by_ids(image_ids)

    logger.info(f"Retrieved {len(retrieved_histograms)} histograms")

    # Verify retrieved histograms match original ones
    for image_id in image_ids:
        assert (
            image_id in retrieved_histograms
        ), f"Image ID {image_id} not found in retrieved histograms"

        original_idx = int(image_id.split("_")[-1])
        original_hist = histograms[original_idx]
        retrieved_hist = retrieved_histograms[image_id]

        # Check that histograms are identical (within floating point precision)
        assert np.allclose(
            original_hist, retrieved_hist, rtol=1e-6
        ), f"Histogram mismatch for {image_id}"

    logger.info("DuckDB metadata store test passed successfully!")
    return metadata_store


def test_integration(
    ann_index: AnnIndex, metadata_store: MetadataStore, histograms: np.ndarray
):
    """
    Test the integration between FAISS index and DuckDB store.

    Args:
        ann_index: Configured AnnIndex instance.
        metadata_store: Configured MetadataStore instance.
        histograms: Original histograms for verification.
    """
    logger.info("Testing FAISS-DuckDB integration...")

    # Perform a search query
    query_histogram = histograms[1]  # Use second histogram as query
    k = min(RERANK_K, len(histograms))

    # Stage 1: ANN search
    distances, indices = ann_index.search(query_histogram, k)
    logger.info(f"ANN search returned {len(indices[0])} candidates")

    # Stage 2: Retrieve raw histograms for reranking
    candidate_ids = [f"test_image_{idx:03d}" for idx in indices[0]]
    candidate_histograms = metadata_store.get_histograms_by_ids(candidate_ids)

    logger.info(
        f"Retrieved {len(candidate_histograms)} candidate histograms for reranking"
    )

    # Verify that we can reconstruct the search pipeline
    assert len(candidate_histograms) == len(
        indices[0]
    ), f"Number of retrieved histograms ({len(candidate_histograms)}) doesn't match search results ({len(indices[0])})"

    # Verify histogram integrity through the pipeline
    for i, (image_id, candidate_hist) in enumerate(candidate_histograms.items()):
        # Extract the index from the image ID (e.g., "test_image_000" -> 0)
        extracted_idx = int(image_id.split("_")[-1])
        original_hist = histograms[extracted_idx]

        # Histograms should be identical (no transformation applied in storage)
        assert np.allclose(
            original_hist, candidate_hist, rtol=1e-6
        ), f"Histogram integrity check failed for {image_id}"

    logger.info("Integration test passed successfully!")


def main():
    """Main test function."""
    logger.info("Starting FAISS and DuckDB wrapper tests...")

    try:
        # Create sample histograms
        histograms = create_sample_histograms(n_samples=20)

        # Test FAISS index
        ann_index = test_faiss_index(histograms)

        # Test DuckDB store
        metadata_store = test_duckdb_store(histograms)

        # Test integration
        test_integration(ann_index, metadata_store, histograms)

        # Test persistence (optional)
        logger.info("Testing index persistence...")
        test_index_path = "test_faiss_index.bin"
        ann_index.save(test_index_path)

        # Create new index and load from file
        new_index = AnnIndex(dimension=TOTAL_BINS)
        new_index.load(test_index_path)

        # Verify loaded index has same content
        assert (
            new_index.get_total_vectors() == ann_index.get_total_vectors()
        ), "Loaded index has different number of vectors"

        # Clean up test file
        if os.path.exists(test_index_path):
            os.remove(test_index_path)

        logger.info("Index persistence test passed!")

        # Clean up
        metadata_store.close()

        logger.info("All tests passed successfully! ✅")
        logger.info("FAISS and DuckDB wrappers are working correctly.")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
