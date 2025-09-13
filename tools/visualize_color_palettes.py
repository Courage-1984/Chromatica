#!/usr/bin/env python3
"""
Color Palette Visualization Tool for Chromatica.

This tool provides comprehensive visualization of color palettes extracted from images,
including dominant colors, color distribution analysis, and palette comparison features.

Features:
- Extract and visualize dominant colors from images
- Generate color palette charts and swatches
- Compare color palettes between multiple images
- Analyze color distribution patterns
- Export palette visualizations

Usage:
    python tools/visualize_color_palettes.py --image path/to/image.jpg
    python tools/visualize_color_palettes.py --compare image1.jpg image2.jpg
    python tools/visualize_color_palettes.py --batch dataset_directory
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import cv2
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import seaborn as sns

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chromatica.indexing.pipeline import process_image
from chromatica.utils.config import TOTAL_BINS, LAB_RANGES

# Set up matplotlib for better-looking plots
plt.style.use("default")
sns.set_palette("husl")


class ColorPaletteVisualizer:
    """Comprehensive color palette visualization and analysis tool."""

    def __init__(self, num_colors: int = 8):
        """
        Initialize the color palette visualizer.

        Args:
            num_colors: Number of dominant colors to extract
        """
        self.num_colors = num_colors
        self.colormap = plt.cm.viridis

    def extract_dominant_colors(
        self, image_path: str
    ) -> Tuple[List[str], List[float], np.ndarray]:
        """
        Extract dominant colors from an image using K-means clustering.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (hex_colors, percentages, color_values)
        """
        try:
            # Load and resize image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Reshape image for clustering
            pixels = image_rgb.reshape(-1, 3)

            # Apply K-means clustering
            kmeans = KMeans(n_clusters=self.num_colors, random_state=42, n_init=10)
            kmeans.fit(pixels)

            # Get cluster centers and labels
            colors = kmeans.cluster_centers_
            labels = kmeans.labels_

            # Calculate percentage of each color
            unique_labels, counts = np.unique(labels, return_counts=True)
            percentages = (counts / len(labels)) * 100

            # Convert RGB to hex
            hex_colors = [
                f"#{int(r):02x}{int(g):02x}{int(b):02x}" for r, g, b in colors
            ]

            return hex_colors, percentages.tolist(), colors

        except Exception as e:
            print(f"Error extracting colors from {image_path}: {e}")
            return [], [], np.array([])

    def create_palette_swatch(
        self,
        hex_colors: List[str],
        percentages: List[float],
        title: str = "Color Palette",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Create a color palette swatch visualization.

        Args:
            hex_colors: List of hex color codes
            percentages: List of color percentages
            title: Title for the visualization
            save_path: Optional path to save the image
        """
        if not hex_colors:
            print("No colors to visualize")
            return

        fig, ax = plt.subplots(figsize=(12, 8))

        # Create color swatches
        for i, (color, percentage) in enumerate(zip(hex_colors, percentages)):
            # Create color patch
            rect = patches.Rectangle(
                (i * 1.2, 0), 1, 1, facecolor=color, edgecolor="black", linewidth=2
            )
            ax.add_patch(rect)

            # Add percentage label
            ax.text(
                i * 1.2 + 0.5,
                1.1,
                f"{percentage:.1f}%",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )

            # Add hex code label
            ax.text(
                i * 1.2 + 0.5,
                -0.1,
                color,
                ha="center",
                va="top",
                fontsize=10,
                fontfamily="monospace",
            )

        # Set up the plot
        ax.set_xlim(-0.5, len(hex_colors) * 1.2 - 0.5)
        ax.set_ylim(-0.3, 1.3)
        ax.set_aspect("equal")
        ax.axis("off")

        # Add title
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Palette swatch saved to: {save_path}")

        plt.show()

    def create_color_distribution_chart(
        self,
        hex_colors: List[str],
        percentages: List[float],
        title: str = "Color Distribution",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Create a pie chart showing color distribution.

        Args:
            hex_colors: List of hex color codes
            percentages: List of color percentages
            title: Title for the visualization
            save_path: Optional path to save the image
        """
        if not hex_colors:
            print("No colors to visualize")
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            percentages,
            labels=hex_colors,
            autopct="%1.1f%%",
            colors=hex_colors,
            startangle=90,
        )

        # Style the text
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")

        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Distribution chart saved to: {save_path}")

        plt.show()

    def create_histogram_visualization(
        self, image_path: str, save_path: Optional[str] = None
    ) -> None:
        """
        Create a visualization of the color histogram.

        Args:
            image_path: Path to the image file
            save_path: Optional path to save the image
        """
        try:
            # Generate histogram
            histogram = process_image(image_path)

            # Reshape histogram to 3D for visualization
            l_bins, a_bins, b_bins = 8, 12, 12
            hist_3d = histogram.reshape(l_bins, a_bins, b_bins)

            # Create subplots for different views
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(
                f"Color Histogram Analysis: {Path(image_path).name}",
                fontsize=16,
                fontweight="bold",
            )

            # L* vs a* projection
            im1 = axes[0, 0].imshow(
                np.sum(hist_3d, axis=2), cmap="viridis", aspect="auto"
            )
            axes[0, 0].set_title("L* vs a* Projection")
            axes[0, 0].set_xlabel("a* (Green-Red)")
            axes[0, 0].set_ylabel("L* (Lightness)")
            plt.colorbar(im1, ax=axes[0, 0])

            # L* vs b* projection
            im2 = axes[0, 1].imshow(
                np.sum(hist_3d, axis=1), cmap="viridis", aspect="auto"
            )
            axes[0, 1].set_title("L* vs b* Projection")
            axes[0, 1].set_xlabel("b* (Blue-Yellow)")
            axes[0, 1].set_ylabel("L* (Lightness)")
            plt.colorbar(im2, ax=axes[0, 1])

            # a* vs b* projection
            im3 = axes[1, 0].imshow(
                np.sum(hist_3d, axis=0), cmap="viridis", aspect="auto"
            )
            axes[1, 0].set_title("a* vs b* Projection")
            axes[1, 0].set_xlabel("b* (Blue-Yellow)")
            axes[1, 0].set_ylabel("a* (Green-Red)")
            plt.colorbar(im3, ax=axes[1, 0])

            # Overall distribution
            axes[1, 1].plot(histogram)
            axes[1, 1].set_title("Overall Histogram Distribution")
            axes[1, 1].set_xlabel("Bin Index")
            axes[1, 1].set_ylabel("Normalized Frequency")
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                print(f"Histogram visualization saved to: {save_path}")

            plt.show()

        except Exception as e:
            print(f"Error creating histogram visualization: {e}")

    def compare_palettes(
        self, image_paths: List[str], save_path: Optional[str] = None
    ) -> None:
        """
        Compare color palettes from multiple images.

        Args:
            image_paths: List of image file paths
            save_path: Optional path to save the image
        """
        if len(image_paths) < 2:
            print("Need at least 2 images for comparison")
            return

        # Extract palettes from all images
        all_palettes = []
        image_names = []

        for image_path in image_paths:
            hex_colors, percentages, _ = self.extract_dominant_colors(image_path)
            if hex_colors:
                all_palettes.append((hex_colors, percentages))
                image_names.append(Path(image_path).name)

        if not all_palettes:
            print("No valid palettes found")
            return

        # Create comparison visualization
        fig, axes = plt.subplots(
            len(all_palettes), 1, figsize=(12, 4 * len(all_palettes))
        )
        if len(all_palettes) == 1:
            axes = [axes]

        for i, ((hex_colors, percentages), image_name) in enumerate(
            zip(all_palettes, image_names)
        ):
            ax = axes[i]

            # Create color bars
            for j, (color, percentage) in enumerate(zip(hex_colors, percentages)):
                rect = patches.Rectangle(
                    (j * 1.2, 0),
                    1,
                    percentage / 100,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=1,
                )
                ax.add_patch(rect)

                # Add percentage label
                ax.text(
                    j * 1.2 + 0.5,
                    percentage / 100 + 0.02,
                    f"{percentage:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

            ax.set_xlim(-0.5, len(hex_colors) * 1.2 - 0.5)
            ax.set_ylim(0, 1.1)
            ax.set_title(f"{image_name}", fontsize=12, fontweight="bold")
            ax.set_ylabel("Percentage")
            ax.grid(True, alpha=0.3)

        # Add x-axis labels to bottom subplot
        axes[-1].set_xlabel("Color Index")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Palette comparison saved to: {save_path}")

        plt.show()

    def create_palette_report(
        self, image_path: str, output_dir: str = "palette_reports"
    ) -> None:
        """
        Create a comprehensive palette analysis report.

        Args:
            image_path: Path to the image file
            output_dir: Directory to save the report
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        image_name = Path(image_path).stem

        # Extract colors
        hex_colors, percentages, color_values = self.extract_dominant_colors(image_path)

        if not hex_colors:
            print(f"Could not extract colors from {image_path}")
            return

        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 16))

        # Main palette swatch
        ax1 = plt.subplot(2, 3, 1)
        for i, (color, percentage) in enumerate(zip(hex_colors, percentages)):
            rect = patches.Rectangle(
                (i * 1.2, 0), 1, 1, facecolor=color, edgecolor="black", linewidth=2
            )
            ax1.add_patch(rect)
            ax1.text(
                i * 1.2 + 0.5,
                1.1,
                f"{percentage:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
        ax1.set_xlim(-0.5, len(hex_colors) * 1.2 - 0.5)
        ax1.set_ylim(-0.3, 1.3)
        ax1.set_aspect("equal")
        ax1.axis("off")
        ax1.set_title("Dominant Colors", fontsize=14, fontweight="bold")

        # Pie chart
        ax2 = plt.subplot(2, 3, 2)
        wedges, texts, autotexts = ax2.pie(
            percentages,
            labels=hex_colors,
            autopct="%1.1f%%",
            colors=hex_colors,
            startangle=90,
        )
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")
        ax2.set_title("Color Distribution", fontsize=14, fontweight="bold")

        # Color values table
        ax3 = plt.subplot(2, 3, 3)
        ax3.axis("off")
        table_data = []
        for i, (color, percentage, rgb) in enumerate(
            zip(hex_colors, percentages, color_values)
        ):
            table_data.append(
                [
                    f"Color {i+1}",
                    color,
                    f"{percentage:.1f}%",
                    f"RGB({int(rgb[0])}, {int(rgb[1])}, {int(rgb[2])})",
                ]
            )

        table = ax3.table(
            cellText=table_data,
            colLabels=["Index", "Hex", "Percentage", "RGB"],
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax3.set_title("Color Details", fontsize=14, fontweight="bold")

        # Histogram visualization
        try:
            histogram = process_image(image_path)
            hist_3d = histogram.reshape(8, 12, 12)

            # L* vs a* projection
            ax4 = plt.subplot(2, 3, 4)
            im4 = ax4.imshow(np.sum(hist_3d, axis=2), cmap="viridis", aspect="auto")
            ax4.set_title("L* vs a* Projection")
            ax4.set_xlabel("a* (Green-Red)")
            ax4.set_ylabel("L* (Lightness)")
            plt.colorbar(im4, ax=ax4)

            # L* vs b* projection
            ax5 = plt.subplot(2, 3, 5)
            im5 = ax5.imshow(np.sum(hist_3d, axis=1), cmap="viridis", aspect="auto")
            ax5.set_title("L* vs b* Projection")
            ax5.set_xlabel("b* (Blue-Yellow)")
            ax5.set_ylabel("L* (Lightness)")
            plt.colorbar(im5, ax=ax5)

            # Overall distribution
            ax6 = plt.subplot(2, 3, 6)
            ax6.plot(histogram)
            ax6.set_title("Overall Histogram")
            ax6.set_xlabel("Bin Index")
            ax6.set_ylabel("Frequency")
            ax6.grid(True, alpha=0.3)

        except Exception as e:
            print(f"Error creating histogram visualizations: {e}")

        plt.suptitle(
            f"Color Palette Analysis Report: {image_name}",
            fontsize=18,
            fontweight="bold",
        )
        plt.tight_layout()

        # Save the report
        report_path = output_path / f"{image_name}_palette_report.png"
        plt.savefig(report_path, dpi=300, bbox_inches="tight")
        print(f"Palette report saved to: {report_path}")

        plt.show()


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="Color Palette Visualization Tool")
    parser.add_argument("--image", type=str, help="Path to single image for analysis")
    parser.add_argument("--compare", nargs="+", help="Paths to images for comparison")
    parser.add_argument(
        "--batch", type=str, help="Directory containing images for batch analysis"
    )
    parser.add_argument(
        "--num-colors", type=int, default=8, help="Number of dominant colors to extract"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="palette_reports",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--save", action="store_true", help="Save visualizations to files"
    )

    args = parser.parse_args()

    if not any([args.image, args.compare, args.batch]):
        parser.print_help()
        return

    # Initialize visualizer
    visualizer = ColorPaletteVisualizer(num_colors=args.num_colors)

    if args.image:
        print(f"Analyzing single image: {args.image}")

        # Extract colors
        hex_colors, percentages, _ = visualizer.extract_dominant_colors(args.image)

        if hex_colors:
            print(f"Extracted {len(hex_colors)} dominant colors")

            # Create visualizations
            visualizer.create_palette_swatch(
                hex_colors, percentages, f"Color Palette: {Path(args.image).name}"
            )
            visualizer.create_color_distribution_chart(
                hex_colors, percentages, f"Color Distribution: {Path(args.image).name}"
            )
            visualizer.create_histogram_visualization(args.image)

            if args.save:
                output_path = Path(args.output)
                output_path.mkdir(exist_ok=True)
                base_name = Path(args.image).stem

                visualizer.create_palette_swatch(
                    hex_colors,
                    percentages,
                    f"Color Palette: {Path(args.image).name}",
                    output_path / f"{base_name}_swatch.png",
                )
                visualizer.create_color_distribution_chart(
                    hex_colors,
                    percentages,
                    f"Color Distribution: {Path(args.image).name}",
                    output_path / f"{base_name}_distribution.png",
                )
                visualizer.create_histogram_visualization(
                    args.image, output_path / f"{base_name}_histogram.png"
                )
                visualizer.create_palette_report(args.image, args.output)

    elif args.compare:
        print(f"Comparing {len(args.compare)} images")
        visualizer.compare_palettes(args.compare)

        if args.save:
            output_path = Path(args.output)
            output_path.mkdir(exist_ok=True)
            visualizer.compare_palettes(
                args.compare, output_path / "palette_comparison.png"
            )

    elif args.batch:
        print(f"Batch analyzing images in: {args.batch}")
        batch_path = Path(args.batch)

        if not batch_path.exists():
            print(f"Directory not found: {args.batch}")
            return

        # Find image files
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        image_files = [
            f for f in batch_path.iterdir() if f.suffix.lower() in image_extensions
        ]

        if not image_files:
            print(f"No image files found in {args.batch}")
            return

        print(f"Found {len(image_files)} images")

        # Process each image
        for image_file in image_files[:10]:  # Limit to first 10 for demo
            print(f"Processing: {image_file.name}")

            if args.save:
                visualizer.create_palette_report(str(image_file), args.output)
            else:
                hex_colors, percentages, _ = visualizer.extract_dominant_colors(
                    str(image_file)
                )
                if hex_colors:
                    visualizer.create_palette_swatch(
                        hex_colors, percentages, f"Color Palette: {image_file.name}"
                    )


if __name__ == "__main__":
    main()
