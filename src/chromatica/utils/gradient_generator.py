"""
Color gradient generation utilities for Chromatica.

This module provides functions to generate color gradients from palettes or search results:
- Generate gradients from color palettes
- Create CSS gradient strings
- Export gradient images
- Multiple gradient types (linear, radial, conic)

Key Features:
- Multiple gradient types and directions
- CSS export support
- Image generation capabilities
"""

import logging
import numpy as np
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw
import io
import base64

logger = logging.getLogger(__name__)


def generate_linear_gradient_css(
    colors: List[str],
    weights: Optional[List[float]] = None,
    direction: str = "to right",
    stops: Optional[List[float]] = None
) -> str:
    """
    Generate CSS linear gradient string.

    Args:
        colors: List of hex color codes
        weights: Optional list of weights for stop positions
        direction: Gradient direction ("to right", "to left", "to bottom", "to top", or angle)
        stops: Optional explicit stop positions (0-100%)

    Returns:
        CSS gradient string
    """
    if not colors:
        return ""
    
    if len(colors) == 1:
        return colors[0]
    
    # Calculate stop positions from weights or use equal spacing
    if stops:
        stop_positions = stops
    elif weights:
        # Normalize weights and convert to percentages
        total = sum(weights)
        cumulative = 0
        stop_positions = []
        for w in weights:
            cumulative += w
            stop_positions.append(cumulative / total * 100)
    else:
        # Equal spacing
        stop_positions = [i * 100 / (len(colors) - 1) for i in range(len(colors))]
    
    # Format colors with stops
    color_stops = []
    for i, color in enumerate(colors):
        color = color.lstrip("#").upper()
        stop = stop_positions[i] if i < len(stop_positions) else 100
        color_stops.append(f"#{color} {stop:.1f}%")
    
    return f"linear-gradient({direction}, {', '.join(color_stops)})"


def generate_radial_gradient_css(
    colors: List[str],
    weights: Optional[List[float]] = None,
    shape: str = "circle",
    position: str = "center"
) -> str:
    """
    Generate CSS radial gradient string.

    Args:
        colors: List of hex color codes
        weights: Optional list of weights for stop positions
        shape: Gradient shape ("circle" or "ellipse")
        position: Gradient position ("center", "top", "bottom", etc.)

    Returns:
        CSS gradient string
    """
    if not colors:
        return ""
    
    if len(colors) == 1:
        return colors[0]
    
    # Calculate stop positions
    if weights:
        total = sum(weights)
        cumulative = 0
        stop_positions = []
        for w in weights:
            cumulative += w
            stop_positions.append(cumulative / total * 100)
    else:
        stop_positions = [i * 100 / (len(colors) - 1) for i in range(len(colors))]
    
    color_stops = []
    for i, color in enumerate(colors):
        color = color.lstrip("#").upper()
        stop = stop_positions[i] if i < len(stop_positions) else 100
        color_stops.append(f"#{color} {stop:.1f}%")
    
    return f"radial-gradient({shape} at {position}, {', '.join(color_stops)})"


def generate_gradient_image(
    colors: List[str],
    weights: Optional[List[float]] = None,
    width: int = 800,
    height: int = 200,
    direction: str = "horizontal"
) -> bytes:
    """
    Generate a gradient image.

    Args:
        colors: List of hex color codes
        weights: Optional list of weights
        width: Image width in pixels
        height: Image height in pixels
        direction: Gradient direction ("horizontal", "vertical", "diagonal")

    Returns:
        PNG image bytes
    """
    if not colors:
        # Default to black if no colors
        colors = ["#000000"]
    
    if len(colors) == 1:
        # Single color - solid fill
        img = Image.new("RGB", (width, height), colors[0].lstrip("#"))
    else:
        # Create gradient image
        img = Image.new("RGB", (width, height))
        draw = ImageDraw.Draw(img)
        
        # Calculate stop positions
        if weights:
            total = sum(weights)
            cumulative = 0
            stops = []
            for w in weights:
                cumulative += w
                stops.append(cumulative / total)
        else:
            stops = [i / (len(colors) - 1) for i in range(len(colors))]
        
        # Convert hex to RGB
        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip("#").upper()
            if len(hex_color) == 3:
                hex_color = "".join([c * 2 for c in hex_color])
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        rgb_colors = [hex_to_rgb(c) for c in colors]
        
        # Draw gradient
        if direction == "horizontal":
            for x in range(width):
                # Find which segment this x position belongs to
                pos = x / width
                for i in range(len(stops) - 1):
                    if stops[i] <= pos <= stops[i + 1]:
                        # Interpolate between colors
                        t = (pos - stops[i]) / (stops[i + 1] - stops[i])
                        r = int(rgb_colors[i][0] * (1 - t) + rgb_colors[i + 1][0] * t)
                        g = int(rgb_colors[i][1] * (1 - t) + rgb_colors[i + 1][1] * t)
                        b = int(rgb_colors[i][2] * (1 - t) + rgb_colors[i + 1][2] * t)
                        for y in range(height):
                            draw.point((x, y), (r, g, b))
                        break
        
        elif direction == "vertical":
            for y in range(height):
                pos = y / height
                for i in range(len(stops) - 1):
                    if stops[i] <= pos <= stops[i + 1]:
                        t = (pos - stops[i]) / (stops[i + 1] - stops[i])
                        r = int(rgb_colors[i][0] * (1 - t) + rgb_colors[i + 1][0] * t)
                        g = int(rgb_colors[i][1] * (1 - t) + rgb_colors[i + 1][1] * t)
                        b = int(rgb_colors[i][2] * (1 - t) + rgb_colors[i + 1][2] * t)
                        for x in range(width):
                            draw.point((x, y), (r, g, b))
                        break
        
        # Diagonal gradient (top-left to bottom-right)
        else:
            for y in range(height):
                for x in range(width):
                    # Calculate distance from diagonal
                    pos = (x + y) / (width + height)
                    for i in range(len(stops) - 1):
                        if stops[i] <= pos <= stops[i + 1]:
                            t = (pos - stops[i]) / (stops[i + 1] - stops[i])
                            r = int(rgb_colors[i][0] * (1 - t) + rgb_colors[i + 1][0] * t)
                            g = int(rgb_colors[i][1] * (1 - t) + rgb_colors[i + 1][1] * t)
                            b = int(rgb_colors[i][2] * (1 - t) + rgb_colors[i + 1][2] * t)
                            draw.point((x, y), (r, g, b))
                            break
    
    # Convert to bytes
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def generate_gradient_image_base64(
    colors: List[str],
    weights: Optional[List[float]] = None,
    width: int = 800,
    height: int = 200,
    direction: str = "horizontal"
) -> str:
    """
    Generate gradient image as base64 string.

    Args:
        colors: List of hex color codes
        weights: Optional list of weights
        width: Image width
        height: Image height
        direction: Gradient direction

    Returns:
        Base64 encoded image string
    """
    img_bytes = generate_gradient_image(colors, weights, width, height, direction)
    return base64.b64encode(img_bytes).decode("utf-8")

