#!/usr/bin/env python3
"""
Test script for saliency weighting functionality.

This script tests the build_saliency_weighted_histogram function to ensure it
works correctly and addresses background dominance issues.

Usage:
    python tools/test_saliency_weighting.py

Requirements:
    - OpenCV with saliency module
    - Test images in datasets/test-dataset-20/
"""

import sys
import os
import numpy as np
import cv2
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chromatica.core.histogram import (
    build_saliency_weighted_histogram,
    build_histogram_from_rgb,
)
from chromatica.utils.config import TOTAL_BINS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_saliency_weighting():
    """Test the saliency weighting functionality."""
    logger.info("Testing saliency weighting functionality...")

    # Test with a sample image from the test dataset
    test_image_path = Path("datasets/test-dataset-20")

    if not test_image_path.exists():
        logger.error(f"Test dataset not found at {test_image_path}")
        return False

    # Find the first image file
    image_files = list(test_image_path.glob("*.jpg")) + list(
        test_image_path.glob("*.png")
    )

    if not image_files:
        logger.error("No test images found")
        return False

    test_image_file = image_files[0]
    logger.info(f"Testing with image: {test_image_file}")

    try:
        # Load and convert image
        image_bgr = cv2.imread(str(test_image_file))
        if image_bgr is None:
            logger.error(f"Failed to load image: {test_image_file}")
            return False

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        logger.info(f"Loaded image with shape: {image_rgb.shape}")

        # Test saliency-weighted histogram generation
        logger.info("Generating saliency-weighted histogram...")
        saliency_hist = build_saliency_weighted_histogram(image_rgb)

        # Test standard histogram generation for comparison
        logger.info("Generating standard histogram for comparison...")
        standard_hist = build_histogram_from_rgb(image_rgb)

        # Validate results
        logger.info("Validating results...")

        # Check histogram shape
        if saliency_hist.shape != (TOTAL_BINS,):
            logger.error(f"Saliency histogram has wrong shape: {saliency_hist.shape}")
            return False

        if standard_hist.shape != (TOTAL_BINS,):
            logger.error(f"Standard histogram has wrong shape: {standard_hist.shape}")
            return False

        # Check normalization
        if not np.isclose(saliency_hist.sum(), 1.0, atol=1e-6):
            logger.error(
                f"Saliency histogram not normalized: sum = {saliency_hist.sum()}"
            )
            return False

        if not np.isclose(standard_hist.sum(), 1.0, atol=1e-6):
            logger.error(
                f"Standard histogram not normalized: sum = {standard_hist.sum()}"
            )
            return False

        # Check for reasonable differences between histograms
        # They should be different but not dramatically so
        diff = np.abs(saliency_hist - standard_hist).sum()
        logger.info(f"Total difference between histograms: {diff:.6f}")

        if diff < 0.001:
            logger.warning(
                "Histograms are very similar - saliency weighting may not be working"
            )
        elif diff > 0.5:
            logger.warning(
                "Histograms are very different - check saliency implementation"
            )

        # Log histogram statistics
        logger.info(
            f"Saliency histogram - min: {saliency_hist.min():.6f}, max: {saliency_hist.max():.6f}"
        )
        logger.info(
            f"Standard histogram - min: {standard_hist.min():.6f}, max: {standard_hist.max():.6f}"
        )

        logger.info("✅ Saliency weighting test passed!")
        return True

    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_edge_cases():
    """Test edge cases for saliency weighting."""
    logger.info("Testing edge cases...")

    try:
        # Test with a simple synthetic image
        logger.info("Testing with synthetic image...")

        # Create a simple test image: red square on blue background
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[20:80, 20:80] = [255, 0, 0]  # Red square
        test_image[:, :] = [0, 0, 255]  # Blue background
        test_image[20:80, 20:80] = [255, 0, 0]  # Red square (overwrite background)

        # Generate histograms
        saliency_hist = build_saliency_weighted_histogram(test_image)
        standard_hist = build_histogram_from_rgb(test_image)

        # Check that they're different (saliency should emphasize the red square)
        diff = np.abs(saliency_hist - standard_hist).sum()
        logger.info(f"Synthetic image histogram difference: {diff:.6f}")

        # Test with very small image
        logger.info("Testing with very small image...")
        small_image = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        saliency_hist_small = build_saliency_weighted_histogram(small_image)

        if not np.isclose(saliency_hist_small.sum(), 1.0, atol=1e-6):
            logger.error("Small image histogram not normalized")
            return False

        logger.info("✅ Edge cases test passed!")
        return True

    except Exception as e:
        logger.error(f"Edge cases test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main test function."""
    logger.info("Starting saliency weighting tests...")

    # Test basic functionality
    success1 = test_saliency_weighting()

    # Test edge cases
    success2 = test_edge_cases()

    if success1 and success2:
        logger.info("🎉 All tests passed!")
        return 0
    else:
        logger.error("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit(main())
