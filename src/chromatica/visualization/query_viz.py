"""
Query visualization module for the Chromatica color search engine.

This module provides visual representations of color queries and search results,
including weighted color bars, query color palettes, and result image collages.

Key Features:
- Generate weighted color bars representing query colors and weights
- Create color palette visualizations
- Build result image collages with proper sizing and arrangement
- Distance-based visual indicators
- Interactive color picker components
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import hex2color
from PIL import Image, ImageDraw, ImageFont
import io
import base64
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class QueryVisualizer:
    """Visualizes color queries with weighted color representations."""

    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the query visualizer.

        Args:
            figsize: Figure size for matplotlib plots
        """
        self.figsize = figsize
        # Set matplotlib backend for non-interactive use
        plt.switch_backend("Agg")

    def create_weighted_color_bar(
        self,
        colors: List[str],
        weights: List[float],
        height: int = 200,
        width: int = 800,
    ) -> np.ndarray:
        """
        Create a weighted color bar image representing the query.

        Args:
            colors: List of hex color codes
            weights: List of corresponding weights
            height: Height of the output image
            width: Width of the output image

        Returns:
            numpy array representing the image
        """
        # Normalize weights to sum to 1.0
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        # Create image array
        img_array = np.zeros((height, width, 3), dtype=np.uint8)

        # Calculate cumulative positions
        cumulative_width = 0
        for color, weight in zip(colors, normalized_weights):
            # Convert hex to RGB
            hex_color = color if color.startswith("#") else f"#{color}"
            rgb_color = hex2color(hex_color)
            rgb_color = [int(c * 255) for c in rgb_color]

            # Calculate segment width
            segment_width = int(width * weight)
            start_x = cumulative_width
            end_x = min(start_x + segment_width, width)

            # Fill the segment with the color
            img_array[:, start_x:end_x] = rgb_color

            cumulative_width = end_x

            # Break if we've filled the entire width
            if cumulative_width >= width:
                break

        return img_array

    def create_color_palette(
        self, colors: List[str], weights: List[float], size: int = 400
    ) -> np.ndarray:
        """
        Create a color palette visualization with weights.

        Args:
            colors: List of hex color codes
            weights: List of corresponding weights
            size: Size of the output image (square)

        Returns:
            numpy array representing the image
        """
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        # Create image array
        img_array = np.zeros((size, size, 3), dtype=np.uint8)

        # Calculate circle parameters
        center_x, center_y = size // 2, size // 2
        max_radius = size // 2 - 20

        # Draw weighted color circles
        current_angle = 0
        for color, weight in zip(colors, normalized_weights):
            # Convert hex to RGB
            hex_color = color if color.startswith("#") else f"#{color}"
            rgb_color = hex2color(hex_color)
            rgb_color = [int(c * 255) for c in rgb_color]

            # Calculate arc angle based on weight
            arc_angle = 2 * np.pi * weight

            # Create a mask for this segment
            y, x = np.ogrid[:size, :size]
            mask = np.zeros((size, size), dtype=bool)

            # Calculate distance from center
            dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

            # Calculate angle from center
            angle_from_center = np.arctan2(y - center_y, x - center_x)
            # Normalize angle to [0, 2π]
            angle_from_center = np.where(
                angle_from_center < 0, angle_from_center + 2 * np.pi, angle_from_center
            )

            # Create mask for this segment
            angle_mask = (angle_from_center >= current_angle) & (
                angle_from_center <= current_angle + arc_angle
            )
            radius_mask = dist_from_center <= max_radius

            mask = angle_mask & radius_mask

            # Apply color to masked region
            img_array[mask] = rgb_color

            current_angle += arc_angle

        return img_array

    def create_query_summary_image(
        self,
        colors: List[str],
        weights: List[float],
        width: int = 800,
        height: int = 600,
    ) -> np.ndarray:
        """
        Create a comprehensive query summary image.

        Args:
            colors: List of hex color codes
            weights: List of corresponding weights
            width: Width of the output image
            height: Height of the output image

        Returns:
            numpy array representing the image
        """
        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
            2, 2, figsize=(width / 100, height / 100)
        )
        fig.suptitle("Color Query Visualization", fontsize=16, fontweight="bold")

        # 1. Weighted color bar
        color_bar = self.create_weighted_color_bar(
            colors, weights, height=100, width=width // 2 - 20
        )
        ax1.imshow(color_bar)
        ax1.set_title("Weighted Color Distribution")
        ax1.axis("off")

        # 2. Color palette
        palette = self.create_color_palette(
            colors, weights, size=min(width // 2 - 20, height // 2 - 20)
        )
        ax2.imshow(palette)
        ax2.set_title("Color Palette")
        ax2.axis("off")

        # 3. Color information table
        ax3.axis("off")
        table_data = []
        for i, (color, weight) in enumerate(zip(colors, weights)):
            hex_color = color if color.startswith("#") else f"#{color}"
            rgb_color = hex2color(hex_color)
            rgb_color = [int(c * 255) for c in rgb_color]
            table_data.append(
                [
                    f"Color {i+1}",
                    hex_color,
                    f"{weight:.2f}",
                    f"RGB({rgb_color[0]},{rgb_color[1]},{rgb_color[2]})",
                ]
            )

        table = ax3.table(
            cellText=table_data,
            colLabels=["", "Hex", "Weight", "RGB"],
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # 4. Weight distribution pie chart
        normalized_weights = [w / sum(weights) for w in weights]
        ax4.pie(
            normalized_weights,
            labels=[f"{w:.1%}" for w in normalized_weights],
            autopct="%1.1f%%",
            startangle=90,
        )
        ax4.set_title("Weight Distribution")

        plt.tight_layout()

        # Convert matplotlib figure to numpy array
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        img = Image.open(buf)
        img_array = np.array(img)
        plt.close()

        return img_array

    def save_image(self, img_array: np.ndarray, filepath: str) -> None:
        """
        Save an image array to file.

        Args:
            img_array: Image array to save
            filepath: Path to save the image
        """
        img = Image.fromarray(img_array)
        img.save(filepath)
        logger.info(f"Saved visualization to: {filepath}")


class ResultCollageBuilder:
    """Creates visual collages of search results."""
    
    def __init__(self, max_images_per_row: int = 3):
        """
        Initialize the collage builder.
        
        Args:
            max_images_per_row: Maximum number of images per row (fixed at 3)
        """
        self.max_images_per_row = max_images_per_row
    
    def create_results_collage(
        self, 
        image_paths: List[str], 
        distances: List[float],
        output_size: Tuple[int, int] = (1200, 800)
    ) -> np.ndarray:
        """
        Create a collage of search result images in a fixed 3x3 grid.
        
        Args:
            image_paths: List of paths to result images
            distances: List of corresponding distance scores
            output_size: Size of the output collage
            
        Returns:
            numpy array representing the collage
        """
        # Always create a 3x3 grid (9 total positions)
        n_cols = 3
        n_rows = 3
        total_positions = n_cols * n_rows
        
        # Create output array with white background
        collage = np.full((*output_size, 3), 255, dtype=np.uint8)
        
        # Calculate individual image size to fill the entire output
        img_width = output_size[0] // n_cols
        img_height = output_size[1] // n_rows
        
        # Place images in the 3x3 grid
        for i in range(total_positions):
            row = i // n_cols
            col = i % n_cols
            y_start = row * img_height
            x_start = col * img_width
            
            if i < len(image_paths):
                # We have an image to place
                try:
                    img_path = image_paths[i]
                    distance = distances[i] if i < len(distances) else 0.0
                    
                    # Load and resize image
                    img = Image.open(img_path)
                    img = img.resize((img_width, img_height), Image.Resampling.LANCZOS)
                    img_array = np.array(img)
                    
                    # Ensure image fits in the allocated space
                    actual_height, actual_width = img_array.shape[:2]
                    y_end = min(y_start + actual_height, output_size[1])
                    x_end = min(x_start + actual_width, output_size[0])
                    
                    # Place image in collage
                    collage[y_start:y_end, x_start:x_end] = img_array[
                        : y_end - y_start, : x_end - x_start
                    ]
                    
                except Exception as e:
                    logger.warning(f"Failed to load image {img_path}: {e}")
                    # Fill with placeholder color (light gray)
                    collage[
                        y_start : y_start + img_height, x_start : x_start + img_width
                    ] = [200, 200, 200]
            else:
                # No more images, fill with placeholder
                collage[
                    y_start : y_start + img_height, x_start : x_start + img_width
                ] = [240, 240, 240]  # Very light gray for empty positions
        
        return collage

    def create_dynamic_collage(
        self, 
        image_paths: List[str], 
        distances: List[float],
        max_width: int = 1200,
        min_height: int = 800
    ) -> np.ndarray:
        """
        Create a dynamic collage that maintains a fixed 3x3 grid layout.
        
        Args:
            image_paths: List of paths to result images
            distances: List of corresponding distance scores
            max_width: Maximum width of the collage
            min_height: Minimum height of the collage (adjusted for 3x3 grid)
            
        Returns:
            numpy array representing the collage
        """
        # Fixed 3x3 grid
        n_cols = 3
        n_rows = 3
        
        # Calculate individual image size
        img_width = max_width // n_cols
        img_height = img_width  # Keep images square
        
        # Calculate total height needed for 3x3 grid
        total_height = n_rows * img_height
        
        # Ensure minimum height
        if total_height < min_height:
            total_height = min_height
        
        output_size = (max_width, total_height)
        
        # Create the collage with the calculated size
        return self.create_distance_annotated_collage(image_paths, distances, output_size)

    def create_distance_annotated_collage(
        self, 
        image_paths: List[str], 
        distances: List[float],
        output_size: Tuple[int, int] = (1200, 800)
    ) -> np.ndarray:
        """
        Create a collage with distance annotations in a fixed 3x3 grid.
        
        Args:
            image_paths: List of paths to result images
            distances: List of corresponding distance scores
            output_size: Size of the output collage
            
        Returns:
            numpy array representing the annotated collage
        """
        # Create base collage
        collage = self.create_results_collage(image_paths, distances, output_size)
        
        # Add distance annotations
        annotated_collage = collage.copy()
        
        # Fixed 3x3 grid
        n_cols = 3
        n_rows = 3
        
        img_width = output_size[0] // n_cols
        img_height = output_size[1] // n_rows
        
        # Add distance text overlay for all 9 positions
        for i in range(n_cols * n_rows):
            row = i // n_cols
            col = i % n_cols
            x = col * img_width + img_width // 2
            y = row * img_height + img_height - 30  # Move text up slightly for better visibility
            
            if i < len(distances):
                # We have a distance value
                distance = distances[i]
                text = f"d={distance:.3f}"
            else:
                # No more distances, show placeholder
                text = "N/A"
            
            # Convert to PIL for text drawing
            pil_img = Image.fromarray(annotated_collage)
            draw = ImageDraw.Draw(pil_img)
            
            # Try to use a default font, fall back to basic if needed
            try:
                font = ImageFont.load_default()
            except:
                font = None
            
            # Draw text with outline for visibility
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx != 0 or dy != 0:
                        draw.text((x + dx, y + dy), text, fill=(0, 0, 0), font=font)
            draw.text((x, y), text, fill=(255, 255, 255), font=font)
            
            annotated_collage = np.array(pil_img)
        
        return annotated_collage


def create_query_visualization(
    colors: List[str],
    weights: List[float],
    output_path: str = "query_visualization.png",
) -> str:
    """
    Create and save a query visualization.

    Args:
        colors: List of hex color codes
        weights: List of corresponding weights
        output_path: Path to save the visualization

    Returns:
        Path to the saved visualization
    """
    visualizer = QueryVisualizer()

    # Create comprehensive visualization
    viz_img = visualizer.create_query_summary_image(colors, weights)

    # Save the image
    visualizer.save_image(viz_img, output_path)

    return output_path


def create_results_collage(
    image_paths: List[str],
    distances: List[float],
    output_path: str = "results_collage.png",
) -> str:
    """
    Create and save a results collage.

    Args:
        image_paths: List of paths to result images
        distances: List of corresponding distance scores
        output_path: Path to save the collage

    Returns:
        Path to the saved collage
    """
    builder = ResultCollageBuilder()

    # Create dynamic collage that adjusts size based on number of images
    collage = builder.create_dynamic_collage(image_paths, distances)

    # Save the collage
    img = Image.fromarray(collage)
    img.save(output_path)

    return output_path
