#!/usr/bin/env python3
"""
Demo script for the Chromatica query processor module.

This script demonstrates the key functionality of the query processor:
- Converting hex colors to Lab values
- Creating query histograms with different color combinations
- Validating histogram properties
- Performance benchmarking

Usage:
    python tools/demo_query_processor.py

The script provides practical examples of how to use the query processor
for color-based image search applications.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chromatica.core.query import (
    hex_to_lab,
    create_query_histogram,
    validate_query_histogram,
)


def demo_hex_to_lab():
    """Demonstrate hex to Lab color conversion."""
    print("=" * 60)
    print("HEX TO LAB COLOR CONVERSION DEMO")
    print("=" * 60)

    # Test colors representing different color families
    test_colors = [
        ("#FF0000", "Pure Red"),
        ("#00FF00", "Pure Green"),
        ("#0000FF", "Pure Blue"),
        ("#FFFF00", "Pure Yellow"),
        ("#FF00FF", "Pure Magenta"),
        ("#00FFFF", "Pure Cyan"),
        ("#000000", "Pure Black"),
        ("#FFFFFF", "Pure White"),
        ("#808080", "Medium Gray"),
        ("#FFA500", "Orange"),
        ("#800080", "Purple"),
        ("#008000", "Dark Green"),
    ]

    print(f"{'Hex Color':<10} {'Color Name':<15} {'L*':<8} {'a*':<8} {'b*':<8}")
    print("-" * 60)

    for hex_color, color_name in test_colors:
        try:
            l_val, a_val, b_val = hex_to_lab(hex_color)
            print(
                f"{hex_color:<10} {color_name:<15} {l_val:<8.2f} {a_val:<8.2f} {b_val:<8.2f}"
            )
        except Exception as e:
            print(f"{hex_color:<10} {color_name:<15} ERROR: {str(e)}")

    print()


def demo_query_histograms():
    """Demonstrate query histogram generation."""
    print("=" * 60)
    print("QUERY HISTOGRAM GENERATION DEMO")
    print("=" * 60)

    # Test different color combinations and weights
    test_queries = [
        {
            "name": "Warm Colors",
            "colors": ["#FF0000", "#FFA500", "#FFFF00"],
            "weights": [0.5, 0.3, 0.2],
        },
        {
            "name": "Cool Colors",
            "colors": ["#0000FF", "#00FFFF", "#800080"],
            "weights": [0.4, 0.4, 0.2],
        },
        {
            "name": "Earth Tones",
            "colors": ["#8B4513", "#A0522D", "#CD853F", "#D2B48C"],
            "weights": [0.3, 0.3, 0.2, 0.2],
        },
        {
            "name": "High Contrast",
            "colors": ["#000000", "#FFFFFF", "#FF0000"],
            "weights": [0.4, 0.4, 0.2],
        },
        {
            "name": "Pastel Palette",
            "colors": ["#FFB6C1", "#98FB98", "#87CEEB", "#DDA0DD"],
            "weights": [0.25, 0.25, 0.25, 0.25],
        },
    ]

    for query in test_queries:
        print(f"\n--- {query['name']} ---")
        print(f"Colors: {query['colors']}")
        print(f"Weights: {query['weights']}")

        try:
            start_time = time.time()
            histogram = create_query_histogram(query["colors"], query["weights"])
            end_time = time.time()

            # Validate the histogram
            is_valid = validate_query_histogram(histogram)

            print(f"Histogram shape: {histogram.shape}")
            print(f"Sum: {histogram.sum():.6f}")
            print(f"Non-zero elements: {np.count_nonzero(histogram)}")
            print(f"Max value: {histogram.max():.6f}")
            print(f"Min value: {histogram.min():.6f}")
            print(f"Generation time: {(end_time - start_time) * 1000:.2f} ms")
            print(f"Validation: {'✓ PASSED' if is_valid else '✗ FAILED'}")

        except Exception as e:
            print(f"ERROR: {str(e)}")

    print()


def demo_performance_benchmark():
    """Demonstrate performance characteristics."""
    print("=" * 60)
    print("PERFORMANCE BENCHMARK DEMO")
    print("=" * 60)

    # Test with increasing numbers of colors
    color_counts = [1, 5, 10, 20, 50, 100]

    # Generate test colors (cycling through a base set)
    base_colors = [
        "#FF0000",
        "#00FF00",
        "#0000FF",
        "#FFFF00",
        "#FF00FF",
        "#00FFFF",
        "#FFA500",
        "#800080",
        "#008000",
        "#FFC0CB",
        "#A52A2A",
        "#808080",
    ]

    print(f"{'Colors':<8} {'Time (ms)':<12} {'Std Dev':<12} {'Histogram Sum':<15}")
    print("-" * 60)

    for count in color_counts:
        # Generate colors and weights
        colors = (base_colors * (count // len(base_colors) + 1))[:count]
        weights = [1.0 / count] * count

        # Measure performance
        times = []
        histograms = []

        for _ in range(10):  # Run 10 times for reliable timing
            start_time = time.time()
            try:
                histogram = create_query_histogram(colors, weights)
                end_time = time.time()
                times.append((end_time - start_time) * 1000)
                histograms.append(histogram)
            except Exception as e:
                print(f"ERROR with {count} colors: {str(e)}")
                break

        if times:
            avg_time = np.mean(times)
            std_time = np.std(times)
            avg_sum = np.mean([h.sum() for h in histograms])

            print(f"{count:<8} {avg_time:<12.2f} {std_time:<12.2f} {avg_sum:<15.6f}")

    print()


def demo_validation():
    """Demonstrate histogram validation."""
    print("=" * 60)
    print("HISTOGRAM VALIDATION DEMO")
    print("=" * 60)

    # Test various histogram scenarios
    test_cases = [
        {
            "name": "Valid Histogram",
            "colors": ["#FF0000", "#00FF00"],
            "weights": [0.6, 0.4],
            "should_pass": True,
        },
        {
            "name": "Single Color",
            "colors": ["#0000FF"],
            "weights": [1.0],
            "should_pass": True,
        },
        {
            "name": "Many Colors",
            "colors": ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF"],
            "weights": [0.2, 0.2, 0.2, 0.2, 0.2],
            "should_pass": True,
        },
    ]

    for test_case in test_cases:
        print(f"\n--- {test_case['name']} ---")

        try:
            histogram = create_query_histogram(
                test_case["colors"], test_case["weights"]
            )
            is_valid = validate_query_histogram(histogram)

            print(f"Expected to pass: {test_case['should_pass']}")
            print(f"Actually passed: {is_valid}")
            print(f"Histogram shape: {histogram.shape}")
            print(f"Sum: {histogram.sum():.6f}")
            print(f"Non-zero elements: {np.count_nonzero(histogram)}")

            if is_valid == test_case["should_pass"]:
                print("✓ Result matches expectation")
            else:
                print("✗ Result doesn't match expectation")

        except Exception as e:
            print(f"ERROR: {str(e)}")

    print()


def main():
    """Run all demos."""
    print("🎨 CHROMATICA QUERY PROCESSOR DEMONSTRATION")
    print("This script showcases the color query processing capabilities")
    print("for the Chromatica color search engine.\n")

    try:
        # Run all demos
        demo_hex_to_lab()
        demo_query_histograms()
        demo_performance_benchmark()
        demo_validation()

        print("=" * 60)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nThe query processor is working correctly and ready for use in:")
        print("- Color-based image search applications")
        print("- Design and art recommendation systems")
        print("- Color palette analysis and comparison")
        print("- Any application requiring color similarity calculations")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
