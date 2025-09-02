#!/usr/bin/env python3
"""
Demonstration script for the Histogram Generation Testing Tool.

This script shows how to use the tool programmatically and demonstrates
various features without requiring command-line arguments.

Run this script from the project root directory:
    python tools/demo.py
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tools.test_histogram_generation import HistogramTester


def demo_single_image():
    """Demonstrate single image testing."""
    print("🔍 Single Image Testing Demo")
    print("=" * 50)

    # Initialize tester
    tester = HistogramTester(output_format="json", visualize=True)

    # Test a single image
    image_path = "datasets/test-dataset-50/test.jpg"
    if os.path.exists(image_path):
        print(f"Testing image: {image_path}")
        result = tester.test_single_image(image_path)

        if result.get("success", True):
            print(f"✅ Success! Histogram shape: {result['histogram']['shape']}")
            print(f"   Entropy: {result['validation']['metrics']['entropy']:.4f}")
            print(f"   Processing time: {result['performance']['mean_time_ms']:.2f} ms")
            print(f"   Output directory: {result['output_directory']}")
        else:
            print(f"❌ Failed: {result.get('error', 'Unknown error')}")
    else:
        print(f"⚠️  Image not found: {image_path}")
        print("   Make sure you have the test dataset available")


def demo_batch_processing():
    """Demonstrate batch directory processing."""
    print("\n📁 Batch Processing Demo")
    print("=" * 50)

    # Initialize tester
    tester = HistogramTester(output_format="both", visualize=False)

    # Test a directory (limit to first 5 images for demo)
    directory_path = "datasets/test-dataset-50"
    if os.path.exists(directory_path):
        print(f"Testing directory: {directory_path}")

        # For demo purposes, we'll just test a few images
        # In practice, you'd call tester.test_directory(directory_path)
        print("   (Demo mode: would process all images in directory)")
        print(
            "   Use: python tools/test_histogram_generation.py --directory datasets/test-dataset-50/"
        )
    else:
        print(f"⚠️  Directory not found: {directory_path}")


def demo_validation_metrics():
    """Demonstrate histogram validation metrics."""
    print("\n📊 Validation Metrics Demo")
    print("=" * 50)

    print("The tool validates histograms for:")
    print("  • Shape: Correct 1152 dimensions (8×12×12)")
    print("  • Normalization: Sum equals 1.0")
    print("  • Bounds: All values ≥ 0")
    print("  • Quality: Entropy, sparsity, distribution")

    print("\nPerformance metrics include:")
    print("  • Processing time (mean, std, min/max)")
    print("  • Memory usage estimation")
    print("  • Pixels processed per second")
    print("  • Comparison between full and fast methods")


def demo_output_formats():
    """Demonstrate output format options."""
    print("\n📤 Output Format Options")
    print("=" * 50)

    print("Available output formats:")
    print("  • JSON: Detailed results with all metadata")
    print("  • CSV: Flattened format for analysis")
    print("  • Both: Generate both formats")

    print("\nGenerated files:")
    print("  • Histogram data (.npy files) → histograms/ folder")
    print("  • Visualization plots (.png files) → histograms/ folder")
    print("  • Results summary (JSON/CSV) → reports/ folder")
    print("  • Batch processing reports → reports/ folder")

    print("\n📁 File organization:")
    print("  • histograms/ folder: Contains .npy and .png files")
    print("  • reports/ folder: Contains .json and .csv files")


def demo_usage_examples():
    """Show usage examples."""
    print("\n💡 Usage Examples")
    print("=" * 50)

    print("Basic usage:")
    print("  python tools/test_histogram_generation.py --image path/to/image.jpg")
    print("  python tools/test_histogram_generation.py --directory path/to/images/")

    print("\nAdvanced options:")
    print(
        "  python tools/test_histogram_generation.py --image image.jpg --output-format both"
    )
    print(
        "  python tools/test_histogram_generation.py --directory images/ --no-visualize"
    )
    print("  python tools/test_histogram_generation.py --image image.jpg --verbose")

    print("\nHelp:")
    print("  python tools/test_histogram_generation.py --help")


def main():
    """Run all demonstrations."""
    print("🎨 Chromatica Histogram Generation Testing Tool Demo")
    print("=" * 60)

    # Check if we're in the right directory
    if not os.path.exists("src/chromatica"):
        print(
            "❌ Error: Please run this script from the Chromatica project root directory"
        )
        print("   Current directory:", os.getcwd())
        return

    # Run demonstrations
    demo_single_image()
    demo_batch_processing()
    demo_validation_metrics()
    demo_output_formats()
    demo_usage_examples()

    print("\n" + "=" * 60)
    print("🎉 Demo completed!")
    print("\nTo get started:")
    print("1. Ensure your virtual environment is activated")
    print("2. Install required dependencies: pip install -r tools/requirements.txt")
    print("3. Test with a single image or directory")
    print("4. Check the generated histograms/ subdirectory for results")


if __name__ == "__main__":
    main()
