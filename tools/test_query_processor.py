#!/usr/bin/env python3
"""
Test script for the Chromatica query processor module.

This script tests the query processing functionality including:
- Hex color to Lab conversion
- Query histogram generation with soft assignment
- Histogram validation
- Error handling for invalid inputs

Usage:
    python tools/test_query_processor.py

The script will run comprehensive tests and provide detailed output
about the performance and correctness of the query processor.
"""

import sys
import os
import time
import logging
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from chromatica.core.query import (
    hex_to_lab,
    create_query_histogram,
    validate_query_histogram,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_hex_to_lab_conversion():
    """Test hex color to Lab conversion functionality."""
    logger.info("Testing hex to Lab conversion...")

    # Test cases: (hex_color, expected_l_range, expected_a_range, expected_b_range)
    test_cases = [
        ("#FF0000", (50, 60), (75, 85), (60, 75)),  # Red
        ("#00FF00", (85, 95), (-80, -70), (80, 90)),  # Green
        ("#0000FF", (25, 35), (60, 80), (-110, -90)),  # Blue
        ("#FFFF00", (95, 100), (-10, 10), (90, 100)),  # Yellow
        ("#FF00FF", (50, 60), (80, 90), (-20, 10)),  # Magenta
        ("#00FFFF", (90, 100), (-50, -30), (-10, 10)),  # Cyan
        ("#000000", (0, 5), (-5, 5), (-5, 5)),  # Black
        ("#FFFFFF", (95, 100), (-5, 5), (-5, 5)),  # White
        ("#808080", (50, 60), (-5, 5), (-5, 5)),  # Gray
    ]

    passed = 0
    total = len(test_cases)

    for hex_color, l_range, a_range, b_range in test_cases:
        try:
            l_val, a_val, b_val = hex_to_lab(hex_color)

            # Check if values are within expected ranges
            l_ok = l_range[0] <= l_val <= l_range[1]
            a_ok = a_range[0] <= a_val <= a_range[1]
            b_ok = b_range[0] <= b_val <= b_range[1]

            if l_ok and a_ok and b_ok:
                logger.info(
                    f"✓ {hex_color} -> L*={l_val:.2f}, a*={a_val:.2f}, b*={b_val:.2f}"
                )
                passed += 1
            else:
                logger.warning(
                    f"⚠ {hex_color} -> L*={l_val:.2f} (expected {l_range}), "
                    f"a*={a_val:.2f} (expected {a_range}), "
                    f"b*={b_val:.2f} (expected {b_range})"
                )

        except Exception as e:
            logger.error(f"✗ {hex_color} conversion failed: {str(e)}")

    logger.info(f"Hex to Lab conversion: {passed}/{total} tests passed")
    return passed == total


def test_query_histogram_generation():
    """Test query histogram generation with various color combinations."""
    logger.info("Testing query histogram generation...")

    # Test cases: (colors, weights, description)
    test_cases = [
        (["#FF0000"], [1.0], "Single red color"),
        (["#FF0000", "#00FF00"], [0.5, 0.5], "Red and green, equal weights"),
        (
            ["#FF0000", "#00FF00", "#0000FF"],
            [0.6, 0.3, 0.1],
            "RGB with varying weights",
        ),
        (
            ["#FF0000", "#00FF00", "#0000FF", "#FFFF00"],
            [0.4, 0.3, 0.2, 0.1],
            "Four colors",
        ),
        (["#808080"], [1.0], "Single gray color"),
        (["#000000", "#FFFFFF"], [0.7, 0.3], "Black and white"),
    ]

    passed = 0
    total = len(test_cases)

    for colors, weights, description in test_cases:
        try:
            start_time = time.time()
            histogram = create_query_histogram(colors, weights)
            end_time = time.time()

            # Validate the histogram
            is_valid = validate_query_histogram(histogram)

            if is_valid:
                logger.info(
                    f"✓ {description}: shape={histogram.shape}, "
                    f"sum={histogram.sum():.6f}, "
                    f"time={(end_time-start_time)*1000:.2f}ms"
                )
                passed += 1
            else:
                logger.error(f"✗ {description}: histogram validation failed")

        except Exception as e:
            logger.error(f"✗ {description}: histogram generation failed: {str(e)}")

    logger.info(f"Query histogram generation: {passed}/{total} tests passed")
    return passed == total


def test_error_handling():
    """Test error handling for invalid inputs."""
    logger.info("Testing error handling...")

    # Test cases: (test_func, args, expected_error, description)
    test_cases = [
        # hex_to_lab error cases
        (hex_to_lab, ("invalid",), ValueError, "Invalid hex color"),
        (hex_to_lab, ("#GG0000",), ValueError, "Invalid hex characters"),
        (hex_to_lab, ("12345",), ValueError, "Wrong hex length"),
        (hex_to_lab, ("",), ValueError, "Empty hex string"),
        (hex_to_lab, (123,), ValueError, "Non-string input"),
        # create_query_histogram error cases
        (create_query_histogram, (["#FF0000"], []), ValueError, "Mismatched lengths"),
        (create_query_histogram, ([], [1.0]), ValueError, "Empty colors list"),
        (
            create_query_histogram,
            (["#FF0000"], ["invalid"]),
            ValueError,
            "Non-numeric weight",
        ),
        (create_query_histogram, (["#FF0000"], [-1.0]), ValueError, "Negative weight"),
        (create_query_histogram, (["#FF0000"], [0.0]), ValueError, "Zero weight sum"),
        (
            create_query_histogram,
            (["invalid"], [1.0]),
            RuntimeError,
            "Invalid color in list",
        ),
    ]

    passed = 0
    total = len(test_cases)

    for test_func, args, expected_error, description in test_cases:
        try:
            test_func(*args)
            logger.error(
                f"✗ {description}: Expected {expected_error.__name__} but no error raised"
            )
        except expected_error:
            logger.info(f"✓ {description}: Correctly raised {expected_error.__name__}")
            passed += 1
        except Exception as e:
            logger.warning(
                f"⚠ {description}: Raised {type(e).__name__} instead of {expected_error.__name__}"
            )

    logger.info(f"Error handling: {passed}/{total} tests passed")
    return passed == total


def test_performance():
    """Test performance of query histogram generation."""
    logger.info("Testing performance...")

    # Test with different numbers of colors
    color_counts = [1, 5, 10, 20, 50]

    # Generate test colors (simple pattern)
    base_colors = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF"]

    for count in color_counts:
        # Repeat colors to reach desired count
        colors = (base_colors * (count // len(base_colors) + 1))[:count]
        weights = [1.0 / count] * count  # Equal weights

        # Measure performance
        times = []
        for _ in range(5):  # Run 5 times for averaging
            start_time = time.time()
            try:
                histogram = create_query_histogram(colors, weights)
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to ms
            except Exception as e:
                logger.error(f"Performance test failed for {count} colors: {str(e)}")
                break

        if times:
            avg_time = np.mean(times)
            std_time = np.std(times)
            logger.info(f"  {count:2d} colors: {avg_time:.2f} ± {std_time:.2f} ms")

    return True


def test_histogram_properties():
    """Test specific properties of generated histograms."""
    logger.info("Testing histogram properties...")

    colors = ["#FF0000", "#00FF00", "#0000FF"]
    weights = [0.5, 0.3, 0.2]

    try:
        histogram = create_query_histogram(colors, weights)

        # Test properties
        tests = [
            (
                "Shape",
                histogram.shape == (1152,),
                f"Expected (1152,), got {histogram.shape}",
            ),
            (
                "Data type",
                histogram.dtype == np.float64,
                f"Expected float64, got {histogram.dtype}",
            ),
            (
                "L1 normalization",
                np.allclose(histogram.sum(), 1.0, atol=1e-6),
                f"Sum = {histogram.sum():.6f}, expected 1.0",
            ),
            ("Non-negative", np.all(histogram >= 0), "Contains negative values"),
            (
                "Finite values",
                np.all(np.isfinite(histogram)),
                "Contains NaN or infinite values",
            ),
            ("Non-zero elements", np.any(histogram > 0), "All elements are zero"),
        ]

        passed = 0
        total = len(tests)

        for test_name, condition, message in tests:
            if condition:
                logger.info(f"✓ {test_name}: {condition}")
                passed += 1
            else:
                logger.error(f"✗ {test_name}: {message}")

        logger.info(f"Histogram properties: {passed}/{total} tests passed")
        return passed == total

    except Exception as e:
        logger.error(f"Histogram properties test failed: {str(e)}")
        return False


def main():
    """Run all tests for the query processor module."""
    logger.info("=" * 60)
    logger.info("CHROMATICA QUERY PROCESSOR TEST SUITE")
    logger.info("=" * 60)

    # Run all test suites
    test_suites = [
        ("Hex to Lab Conversion", test_hex_to_lab_conversion),
        ("Query Histogram Generation", test_query_histogram_generation),
        ("Error Handling", test_error_handling),
        ("Performance", test_performance),
        ("Histogram Properties", test_histogram_properties),
    ]

    results = {}
    total_passed = 0
    total_tests = len(test_suites)

    for suite_name, test_func in test_suites:
        logger.info(f"\n--- {suite_name} ---")
        try:
            result = test_func()
            results[suite_name] = result
            if result:
                total_passed += 1
        except Exception as e:
            logger.error(f"Test suite '{suite_name}' failed with exception: {str(e)}")
            results[suite_name] = False

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    for suite_name, result in results.items():
        status = "PASSED" if result else "FAILED"
        logger.info(f"{suite_name:25s}: {status}")

    logger.info(f"\nOverall: {total_passed}/{total_tests} test suites passed")

    if total_passed == total_tests:
        logger.info("🎉 All tests passed! Query processor is working correctly.")
        return 0
    else:
        logger.error("❌ Some tests failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
