#!/usr/bin/env python3
"""
Histogram Generation Testing Tool for Chromatica

This tool provides comprehensive testing capabilities for the histogram generation module.
It can process single images or entire directories, generating histograms and providing
detailed analysis, validation, and visualization.

Features:
- Single image and batch directory processing
- Automatic image loading and Lab color space conversion
- Histogram validation and quality checks
- Performance benchmarking
- Visualization of histogram distributions
- Comprehensive output in multiple formats

Usage:
    # Test single image
    python tools/test_histogram_generation.py --image path/to/image.jpg

    # Test directory of images
    python tools/test_histogram_generation.py --directory path/to/images/

    # Test with specific options
    python tools/test_histogram_generation.py --image path/to/image.jpg --output-format json --visualize

Author: Chromatica Development Team
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import cv2
from skimage import color
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import seaborn as sns

from chromatica.core.histogram import (
    build_histogram,
    build_histogram_fast,
    get_bin_centers,
    get_bin_grid,
)
from chromatica.utils.config import (
    L_BINS,
    A_BINS,
    B_BINS,
    TOTAL_BINS,
    LAB_RANGES,
    MAX_IMAGE_DIMENSION,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class HistogramTester:
    """Main class for testing histogram generation functionality."""

    def __init__(self, output_format: str = "json", visualize: bool = True):
        """
        Initialize the histogram tester.

        Args:
            output_format: Output format for results ('json', 'csv', 'both')
            visualize: Whether to generate visualization plots
        """
        self.output_format = output_format
        self.visualize = visualize
        self.results = []

        # Set up matplotlib for non-interactive backend
        plt.switch_backend("Agg")

    def load_and_convert_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load an image and convert it to Lab color space.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (original_image, lab_pixels)
        """
        logger.info(f"Loading image: {image_path}")

        # Load image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Convert BGR to RGB (OpenCV loads in BGR format)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize image if necessary (maintain aspect ratio)
        height, width = image_rgb.shape[:2]
        if max(height, width) > MAX_IMAGE_DIMENSION:
            scale = MAX_IMAGE_DIMENSION / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image_rgb = cv2.resize(
                image_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA
            )
            logger.info(
                f"Resized image from {width}x{height} to {new_width}x{new_height}"
            )

        # Convert to Lab color space using scikit-image
        # Note: skimage.color.rgb2lab expects RGB in [0, 1] range
        image_rgb_normalized = image_rgb.astype(np.float32) / 255.0
        image_lab = color.rgb2lab(image_rgb_normalized, illuminant="D65")

        # Reshape to (N, 3) array of Lab pixels
        lab_pixels = image_lab.reshape(-1, 3)

        logger.info(f"Converted image to Lab: {lab_pixels.shape[0]} pixels")
        return image_rgb, lab_pixels

    def validate_histogram(
        self, histogram: np.ndarray, image_path: str
    ) -> Dict[str, Any]:
        """
        Validate the generated histogram for correctness and quality.

        Args:
            histogram: The generated histogram
            image_path: Path to the source image for reference

        Returns:
            Dictionary containing validation results
        """
        validation = {
            "image_path": image_path,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "validation_passed": True,
            "errors": [],
            "warnings": [],
            "metrics": {},
        }

        # Check shape
        if histogram.shape != (TOTAL_BINS,):
            validation["validation_passed"] = False
            validation["errors"].append(
                f"Histogram shape {histogram.shape} != expected {TOTAL_BINS}"
            )

        # Check normalization
        hist_sum = histogram.sum()
        if not np.isclose(hist_sum, 1.0, atol=1e-6):
            validation["validation_passed"] = False
            validation["errors"].append(f"Histogram sum {hist_sum:.8f} != 1.0")

        # Check bounds
        if histogram.min() < 0:
            validation["validation_passed"] = False
            validation["errors"].append(
                f"Histogram contains negative values: min = {histogram.min():.8f}"
            )

        if histogram.max() > 1.0:
            validation["warnings"].append(
                f"Histogram max value {histogram.max():.8f} > 1.0"
            )

        # Calculate metrics
        validation["metrics"] = {
            "shape": histogram.shape,
            "sum": float(hist_sum),
            "min": float(histogram.min()),
            "max": float(histogram.max()),
            "mean": float(histogram.mean()),
            "std": float(histogram.std()),
            "sparsity": float(np.count_nonzero(histogram) / TOTAL_BINS),
            "entropy": float(-np.sum(histogram * np.log2(histogram + 1e-10))),
        }

        # Check for reasonable entropy (should be > 0 for non-uniform histograms)
        if validation["metrics"]["entropy"] < 0.1:
            validation["warnings"].append(
                f"Very low entropy {validation['metrics']['entropy']:.4f} - histogram may be too uniform"
            )

        return validation

    def benchmark_performance(
        self, func, lab_pixels: np.ndarray, num_runs: int = 5
    ) -> Dict[str, float]:
        """
        Benchmark the performance of histogram generation functions.

        Args:
            func: Function to benchmark (build_histogram or build_histogram_fast)
            lab_pixels: Lab pixel array to process
            num_runs: Number of runs for averaging

        Returns:
            Dictionary containing performance metrics
        """
        times = []
        memory_usage = []

        for _ in range(num_runs):
            # Time the function
            start_time = time.perf_counter()
            histogram = func(lab_pixels)
            end_time = time.perf_counter()

            times.append(end_time - start_time)

            # Estimate memory usage (rough approximation)
            memory_usage.append(histogram.nbytes / 1024)  # KB

        return {
            "mean_time_ms": np.mean(times) * 1000,
            "std_time_ms": np.std(times) * 1000,
            "min_time_ms": np.min(times) * 1000,
            "max_time_ms": np.max(times) * 1000,
            "mean_memory_kb": np.mean(memory_usage),
            "pixels_per_second": lab_pixels.shape[0] / np.mean(times),
        }

    def create_visualization(
        self, histogram: np.ndarray, image_path: str, output_dir: str
    ) -> str:
        """
        Create visualization plots for the histogram.

        Args:
            histogram: The generated histogram
            image_path: Path to the source image
            output_dir: Directory to save visualization files

        Returns:
            Path to the saved visualization file
        """
        if not self.visualize:
            return ""

        try:
            # Reshape histogram to 3D for visualization
            hist_3d = histogram.reshape(L_BINS, A_BINS, B_BINS)

            # Create figure with multiple subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(
                f"Histogram Analysis: {os.path.basename(image_path)}", fontsize=16
            )

            # 1. 3D scatter plot of non-zero bins
            ax1 = fig.add_subplot(2, 2, 1, projection="3d")
            non_zero_indices = np.where(hist_3d > 0)
            if len(non_zero_indices[0]) > 0:
                l_coords, a_coords, b_coords = non_zero_indices
                values = hist_3d[non_zero_indices]

                # Normalize values for color mapping
                norm_values = (values - values.min()) / (
                    values.max() - values.min() + 1e-10
                )
                colors = plt.cm.viridis(norm_values)

                scatter = ax1.scatter(
                    l_coords,
                    a_coords,
                    b_coords,
                    c=norm_values,
                    cmap="viridis",
                    s=values * 1000,
                    alpha=0.7,
                )
                ax1.set_xlabel("L* Bins")
                ax1.set_ylabel("a* Bins")
                ax1.set_zlabel("b* Bins")
                ax1.set_title("3D Histogram Distribution (Non-zero bins)")
                plt.colorbar(scatter, ax=ax1, label="Normalized Values")

            # 2. 2D projection (L* vs a*)
            ax2 = axes[0, 1]
            l_a_projection = np.sum(hist_3d, axis=2)
            im2 = ax2.imshow(
                l_a_projection.T,
                cmap="viridis",
                aspect="auto",
                origin="lower",
                extent=[0, L_BINS, 0, A_BINS],
            )
            ax2.set_xlabel("L* Bins")
            ax2.set_ylabel("a* Bins")
            ax2.set_title("L* vs a* Projection")
            plt.colorbar(im2, ax=ax2)

            # 3. 2D projection (L* vs b*)
            ax3 = axes[1, 0]
            l_b_projection = np.sum(hist_3d, axis=1)
            im3 = ax3.imshow(
                l_b_projection.T,
                cmap="viridis",
                aspect="auto",
                origin="lower",
                extent=[0, L_BINS, 0, B_BINS],
            )
            ax3.set_xlabel("L* Bins")
            ax3.set_ylabel("b* Bins")
            ax3.set_title("L* vs b* Projection")
            plt.colorbar(im3, ax=ax3)

            # 4. Histogram distribution
            ax4 = axes[1, 1]
            non_zero_values = histogram[histogram > 0]
            if len(non_zero_values) > 0:
                ax4.hist(
                    non_zero_values,
                    bins=50,
                    alpha=0.7,
                    color="skyblue",
                    edgecolor="black",
                )
                ax4.set_xlabel("Histogram Values")
                ax4.set_ylabel("Frequency")
                ax4.set_title("Distribution of Non-zero Histogram Values")
                ax4.axvline(
                    non_zero_values.mean(),
                    color="red",
                    linestyle="--",
                    label=f"Mean: {non_zero_values.mean():.4f}",
                )
                ax4.legend()

            plt.tight_layout()

            # Save visualization
            base_name = Path(image_path).stem
            viz_path = os.path.join(output_dir, f"{base_name}_histogram_analysis.png")
            plt.savefig(viz_path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Visualization saved: {viz_path}")
            return viz_path

        except Exception as e:
            logger.warning(f"Failed to create visualization: {e}")
            return ""

    def save_results(
        self, results: List[Dict], output_dir: str, base_name: str = "histogram_test"
    ):
        """
        Save test results in the specified format.

        Args:
            results: List of result dictionaries
            output_dir: Directory to save results
            base_name: Base name for output files
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Create separate folders for different file types
        # PNG and NPY files stay in histograms folder
        # JSON, CSV, and other text files go to reports folder
        reports_dir = os.path.join(os.path.dirname(output_dir), "reports")
        os.makedirs(reports_dir, exist_ok=True)

        if self.output_format in ["json", "both"]:
            json_path = os.path.join(reports_dir, f"{base_name}_{timestamp}.json")
            with open(json_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"JSON results saved: {json_path}")

        if self.output_format in ["csv", "both"]:
            csv_path = os.path.join(reports_dir, f"{base_name}_{timestamp}.csv")
            # Flatten results for CSV with comprehensive information
            flattened_results = []
            for result in results:
                flat_result = {
                    "image_path": result["image_path"],
                    "image_name": os.path.basename(result["image_path"]),
                    "original_size": f"{result['image_info']['original_size'][0]}x{result['image_info']['original_size'][1]}",
                    "lab_pixels": result["image_info"]["lab_pixels"],
                    "resized": result["image_info"]["resized"],
                    "validation_passed": result["validation"]["validation_passed"],
                    "validation_errors": "; ".join(result["validation"]["errors"]),
                    "validation_warnings": "; ".join(result["validation"]["warnings"]),
                    "histogram_shape": str(result["validation"]["metrics"]["shape"]),
                    "histogram_sum": result["validation"]["metrics"]["sum"],
                    "histogram_min": result["validation"]["metrics"]["min"],
                    "histogram_max": result["validation"]["metrics"]["max"],
                    "histogram_mean": result["validation"]["metrics"]["mean"],
                    "histogram_std": result["validation"]["metrics"]["std"],
                    "sparsity": result["validation"]["metrics"]["sparsity"],
                    "entropy": result["validation"]["metrics"]["entropy"],
                    "mean_time_ms": result["performance"]["mean_time_ms"],
                    "std_time_ms": result["performance"]["std_time_ms"],
                    "min_time_ms": result["performance"]["min_time_ms"],
                    "max_time_ms": result["performance"]["max_time_ms"],
                    "mean_memory_kb": result["performance"]["mean_memory_kb"],
                    "pixels_per_second": result["performance"]["pixels_per_second"],
                    "histogram_data_path": result["histogram"]["data_path"],
                    "visualization_path": result.get("visualization", ""),
                    "output_directory": result["output_directory"],
                }
                flattened_results.append(flat_result)

            # Write CSV with comprehensive headers
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                if flattened_results:
                    writer = csv.DictWriter(f, fieldnames=flattened_results[0].keys())
                    writer.writeheader()
                    writer.writerows(flattened_results)
            logger.info(f"CSV results saved: {csv_path}")

    def test_single_image(self, image_path: str) -> Dict[str, Any]:
        """
        Test histogram generation on a single image.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary containing test results
        """
        logger.info(f"Testing single image: {image_path}")

        try:
            # Load and convert image
            image, lab_pixels = self.load_and_convert_image(image_path)

            # Generate histograms using both methods
            start_time = time.perf_counter()
            histogram = build_histogram(lab_pixels)
            end_time = time.perf_counter()

            histogram_fast = build_histogram_fast(lab_pixels)

            # Validate histograms
            validation = self.validate_histogram(histogram, image_path)
            validation_fast = self.validate_histogram(histogram_fast, image_path)

            # Benchmark performance
            performance = self.benchmark_performance(build_histogram, lab_pixels)
            performance_fast = self.benchmark_performance(
                build_histogram_fast, lab_pixels
            )

            # Create output directory
            image_dir = os.path.dirname(image_path)
            output_dir = os.path.join(image_dir, "histograms")
            os.makedirs(output_dir, exist_ok=True)

            # Create visualization
            viz_path = self.create_visualization(histogram, image_path, output_dir)

            # Save histogram data
            base_name = Path(image_path).stem
            hist_data_path = os.path.join(output_dir, f"{base_name}_histogram.npy")
            np.save(hist_data_path, histogram)

            # Compile results
            result = {
                "image_path": image_path,
                "image_info": {
                    "original_size": image.shape,
                    "lab_pixels": lab_pixels.shape[0],
                    "resized": image.shape != (lab_pixels.shape[0] // 3, 3),
                },
                "histogram": {
                    "data_path": hist_data_path,
                    "shape": histogram.shape,
                    "dtype": str(histogram.dtype),
                },
                "validation": validation,
                "validation_fast": validation_fast,
                "performance": performance,
                "performance_fast": performance_fast,
                "visualization": viz_path,
                "output_directory": output_dir,
            }

            logger.info(f"Single image test completed successfully")
            return result

        except Exception as e:
            logger.error(f"Error testing image {image_path}: {e}")
            return {"image_path": image_path, "error": str(e), "success": False}

    def test_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Test histogram generation on all images in a directory.

        Args:
            directory_path: Path to the directory containing images

        Returns:
            List of test result dictionaries
        """
        logger.info(f"Testing directory: {directory_path}")

        # Supported image extensions
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

        # Find all image files (case-insensitive, avoid duplicates)
        image_files = set()
        for ext in image_extensions:
            # Use case-insensitive pattern matching
            pattern = f"*{ext}"
            image_files.update(Path(directory_path).glob(pattern))
            image_files.update(Path(directory_path).glob(pattern.upper()))

        # Convert to list and sort for consistent processing order
        image_files = sorted(list(image_files))

        if not image_files:
            logger.warning(f"No image files found in directory: {directory_path}")
            return []

        logger.info(f"Found {len(image_files)} unique image files")

        # Process each image
        results = []
        for image_file in image_files:
            logger.info(f"Processing: {image_file.name}")
            result = self.test_single_image(str(image_file))
            results.append(result)

            # Add summary statistics
            if result.get("success", True):
                logger.info(
                    f"✓ {image_file.name}: {result['validation']['metrics']['entropy']:.4f} entropy"
                )
            else:
                logger.error(
                    f"✗ {image_file.name}: {result.get('error', 'Unknown error')}"
                )

        # Create summary output
        output_dir = os.path.join(directory_path, "histograms")
        os.makedirs(output_dir, exist_ok=True)

        # Save summary results
        self.save_results(results, output_dir, "batch_histogram_test")

        # Generate comprehensive reports
        successful_results = [r for r in results if r.get("success", True)]
        if successful_results:
            self._generate_summary_report(successful_results, output_dir)
            self._generate_detailed_analysis_report(successful_results, output_dir)
            self._generate_validation_summary_report(successful_results, output_dir)
            self._generate_performance_analysis_report(successful_results, output_dir)
            self._generate_quality_metrics_report(successful_results, output_dir)

        logger.info(f"Directory test completed: {len(results)} images processed")
        return results

    def _generate_summary_report(self, results: List[Dict], output_dir: str):
        """Generate a summary report of batch processing results."""
        try:
            # Calculate summary statistics
            entropies = [r["validation"]["metrics"]["entropy"] for r in results]
            times = [r["performance"]["mean_time_ms"] for r in results]
            sparsities = [r["validation"]["metrics"]["sparsity"] for r in results]

            summary = {
                "total_images": len(results),
                "successful_images": len(results),
                "failed_images": 0,
                "summary_statistics": {
                    "entropy": {
                        "mean": float(np.mean(entropies)),
                        "std": float(np.std(entropies)),
                        "min": float(np.min(entropies)),
                        "max": float(np.max(entropies)),
                    },
                    "processing_time_ms": {
                        "mean": float(np.mean(times)),
                        "std": float(np.std(times)),
                        "min": float(np.min(times)),
                        "max": float(np.max(times)),
                    },
                    "sparsity": {
                        "mean": float(np.mean(sparsities)),
                        "std": float(np.std(sparsities)),
                        "min": float(np.min(sparsities)),
                        "max": float(np.max(sparsities)),
                    },
                },
                "image_paths": [r["image_path"] for r in results],
            }

            # Save summary to reports folder (not histograms folder)
            reports_dir = os.path.join(os.path.dirname(output_dir), "reports")
            os.makedirs(reports_dir, exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            summary_path = os.path.join(reports_dir, f"summary_report_{timestamp}.json")
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)

            logger.info(f"Summary report saved: {summary_path}")

        except Exception as e:
            logger.warning(f"Failed to generate summary report: {e}")

    def _generate_detailed_analysis_report(self, results: List[Dict], output_dir: str):
        """Generate a detailed analysis report of histogram characteristics."""
        try:
            all_histograms = [r["histogram"]["data_path"] for r in results]
            all_entropies = [r["validation"]["metrics"]["entropy"] for r in results]
            all_sparsities = [r["validation"]["metrics"]["sparsity"] for r in results]

            # Calculate overall statistics
            overall_stats = {
                "total_images": len(all_histograms),
                "total_pixels": sum(r["image_info"]["lab_pixels"] for r in results),
                "total_resized_images": sum(r["image_info"]["resized"] for r in results),
                "mean_entropy": float(np.mean(all_entropies)),
                "std_entropy": float(np.std(all_entropies)),
                "min_entropy": float(np.min(all_entropies)),
                "max_entropy": float(np.max(all_entropies)),
                "mean_sparsity": float(np.mean(all_sparsities)),
                "std_sparsity": float(np.std(all_sparsities)),
                "min_sparsity": float(np.min(all_sparsities)),
                "max_sparsity": float(np.max(all_sparsities)),
            }

            # Save detailed analysis report
            reports_dir = os.path.join(os.path.dirname(output_dir), "reports")
            os.makedirs(reports_dir, exist_ok=True)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            detailed_path = os.path.join(reports_dir, f"detailed_analysis_{timestamp}.json")
            with open(detailed_path, "w") as f:
                json.dump(overall_stats, f, indent=2, default=str)

            logger.info(f"Detailed analysis report saved: {detailed_path}")

        except Exception as e:
            logger.warning(f"Failed to generate detailed analysis report: {e}")

    def _generate_validation_summary_report(self, results: List[Dict], output_dir: str):
        """Generate a summary report of validation results."""
        try:
            validation_results = []
            for r in results:
                validation_results.append({
                    "image_path": r["image_path"],
                    "validation_passed": r["validation"]["validation_passed"],
                    "errors": "; ".join(r["validation"]["errors"]),
                    "warnings": "; ".join(r["validation"]["warnings"]),
                })

            # Save validation summary report
            reports_dir = os.path.join(os.path.dirname(output_dir), "reports")
            os.makedirs(reports_dir, exist_ok=True)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            validation_path = os.path.join(reports_dir, f"validation_summary_{timestamp}.json")
            with open(validation_path, "w") as f:
                json.dump(validation_results, f, indent=2, default=str)

            logger.info(f"Validation summary report saved: {validation_path}")

        except Exception as e:
            logger.warning(f"Failed to generate validation summary report: {e}")

    def _generate_performance_analysis_report(self, results: List[Dict], output_dir: str):
        """Generate a performance analysis report of histogram generation."""
        try:
            performance_results = []
            for r in results:
                performance_results.append({
                    "image_path": r["image_path"],
                    "mean_time_ms": r["performance"]["mean_time_ms"],
                    "std_time_ms": r["performance"]["std_time_ms"],
                    "min_time_ms": r["performance"]["min_time_ms"],
                    "max_time_ms": r["performance"]["max_time_ms"],
                    "mean_memory_kb": r["performance"]["mean_memory_kb"],
                    "pixels_per_second": r["performance"]["pixels_per_second"],
                })

            # Save performance analysis report
            reports_dir = os.path.join(os.path.dirname(output_dir), "reports")
            os.makedirs(reports_dir, exist_ok=True)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            performance_path = os.path.join(reports_dir, f"performance_analysis_{timestamp}.json")
            with open(performance_path, "w") as f:
                json.dump(performance_results, f, indent=2, default=str)

            logger.info(f"Performance analysis report saved: {performance_path}")

        except Exception as e:
            logger.warning(f"Failed to generate performance analysis report: {e}")

    def _generate_quality_metrics_report(self, results: List[Dict], output_dir: str):
        """Generate a quality metrics report of histogram characteristics."""
        try:
            quality_results = []
            for r in results:
                quality_results.append({
                    "image_path": r["image_path"],
                    "entropy": r["validation"]["metrics"]["entropy"],
                    "sparsity": r["validation"]["metrics"]["sparsity"],
                    "mean_time_ms": r["performance"]["mean_time_ms"],
                    "pixels_per_second": r["performance"]["pixels_per_second"],
                })

            # Save quality metrics report
            reports_dir = os.path.join(os.path.dirname(output_dir), "reports")
            os.makedirs(reports_dir, exist_ok=True)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            quality_path = os.path.join(reports_dir, f"quality_metrics_{timestamp}.json")
            with open(quality_path, "w") as f:
                json.dump(quality_results, f, indent=2, default=str)

            logger.info(f"Quality metrics report saved: {quality_path}")

        except Exception as e:
            logger.warning(f"Failed to generate quality metrics report: {e}")


def main():
    """Main entry point for the histogram testing tool."""
    parser = argparse.ArgumentParser(
        description="Test histogram generation for Chromatica color search engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test single image
  python tools/test_histogram_generation.py --image data/test.jpg
  
  # Test directory of images
  python tools/test_histogram_generation.py --directory data/images/
  
  # Test with specific options
  python tools/test_histogram_generation.py --image data/test.jpg --output-format both --no-visualize
        """,
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image", "-i", type=str, help="Path to single image file to test"
    )
    input_group.add_argument(
        "--directory",
        "-d",
        type=str,
        help="Path to directory containing images to test",
    )

    # Output options
    parser.add_argument(
        "--output-format",
        "-f",
        choices=["json", "csv", "both"],
        default="json",
        help="Output format for results (default: json)",
    )

    parser.add_argument(
        "--no-visualize",
        "-nv",
        action="store_true",
        help="Disable visualization generation",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize tester
    tester = HistogramTester(
        output_format=args.output_format, visualize=not args.no_visualize
    )

    try:
        if args.image:
            # Test single image
            result = tester.test_single_image(args.image)
            if result.get("success", True):
                print(f"\n✅ Single image test completed successfully!")
                print(f"   Image: {args.image}")
                print(f"   Histogram shape: {result['histogram']['shape']}")
                print(f"   Output directory: {result['output_directory']}")
                if result["visualization"]:
                    print(f"   Visualization: {result['visualization']}")
                
                # Inform about file organization
                image_dir = os.path.dirname(args.image)
                print(f"   📁 File organization:")
                print(f"      • Histograms & Visualizations: {os.path.join(image_dir, 'histograms')}")
                print(f"      • Reports & Data: {os.path.join(image_dir, 'reports')}")
            else:
                print(f"\n❌ Single image test failed!")
                print(f"   Error: {result.get('error', 'Unknown error')}")
                sys.exit(1)

        elif args.directory:
            # Test directory
            results = tester.test_directory(args.directory)
            successful = sum(1 for r in results if r.get("success", True))
            total = len(results)

            print(f"\n✅ Directory test completed!")
            print(f"   Directory: {args.directory}")
            print(f"   Total images: {total}")
            print(f"   Successful: {successful}")
            print(f"   Failed: {total - successful}")
            
            # Inform about file organization
            print(f"   📁 File organization:")
            print(f"      • Histograms & Visualizations: {os.path.join(args.directory, 'histograms')}")
            print(f"      • Reports & Data: {os.path.join(args.directory, 'reports')}")

            if successful < total:
                sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        logger.exception("Unexpected error occurred")
        sys.exit(1)


if __name__ == "__main__":
    main()
