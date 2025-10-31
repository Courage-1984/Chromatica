"""
Color filter utilities for Chromatica.

This module provides functions to filter images based on color properties:
- Color temperature (warm/cool)
- Brightness range
- Saturation range
- Dominant color count
- Negative color filtering (exclude colors)

Key Features:
- Comprehensive color property analysis
- Filter implementation
- Temperature classification
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from skimage.color import rgb2lab
import colorsys

logger = logging.getLogger(__name__)


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip("#").upper()
    if len(hex_color) == 3:
        hex_color = "".join([c * 2 for c in hex_color])
    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (r, g, b)
    except ValueError:
        return (0, 0, 0)


def rgb_to_hsv(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """Convert RGB to HSV."""
    r_norm = r / 255.0
    g_norm = g / 255.0
    b_norm = b / 255.0
    h, s, v = colorsys.rgb_to_hsv(r_norm, g_norm, b_norm)
    return (h * 360, s * 100, v * 100)


def calculate_color_temperature(hex_color: str) -> Dict[str, Any]:
    """
    Calculate color temperature (warm/cool).

    Args:
        hex_color: Hex color code

    Returns:
        Dictionary with temperature classification and numeric value
    """
    rgb = hex_to_rgb(hex_color)
    h, s, v = rgb_to_hsv(rgb[0], rgb[1], rgb[2])
    
    # Temperature classification based on hue
    # Warm colors: reds, oranges, yellows (0-60° and 300-360°)
    # Cool colors: blues, greens, cyans (120-240°)
    # Neutral: everything else
    
    if h < 60 or h > 300:
        temp_class = "warm"
        temp_value = 1.0 - (h / 360) if h < 60 else (360 - h) / 60
    elif 120 <= h <= 240:
        temp_class = "cool"
        temp_value = 1.0 - abs(h - 180) / 60
    else:
        temp_class = "neutral"
        temp_value = 0.5
    
    # Calculate temperature in Kelvin (approximation)
    # Warmer colors have higher Kelvin (3000-5000K), cooler have lower (5000-10000K)
    if temp_class == "warm":
        kelvin = 3000 + temp_value * 2000
    elif temp_class == "cool":
        kelvin = 7000 + temp_value * 3000
    else:
        kelvin = 5500
    
    return {
        "temperature_class": temp_class,
        "temperature_value": temp_value,
        "kelvin": kelvin,
        "hue": h
    }


def calculate_brightness(hex_color: str) -> float:
    """
    Calculate color brightness (0-1).

    Args:
        hex_color: Hex color code

    Returns:
        Brightness value between 0 and 1
    """
    rgb = hex_to_rgb(hex_color)
    # Use relative luminance formula (WCAG)
    r, g, b = [c / 255.0 for c in rgb]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return luminance


def calculate_saturation(hex_color: str) -> float:
    """
    Calculate color saturation (0-1).

    Args:
        hex_color: Hex color code

    Returns:
        Saturation value between 0 and 1
    """
    rgb = hex_to_rgb(hex_color)
    h, s, v = rgb_to_hsv(rgb[0], rgb[1], rgb[2])
    return s / 100.0


def filter_by_temperature(
    colors: List[str],
    temperature_filter: str = "all"
) -> List[str]:
    """
    Filter colors by temperature.

    Args:
        colors: List of hex color codes
        temperature_filter: "warm", "cool", "neutral", or "all"

    Returns:
        Filtered list of colors
    """
    if temperature_filter == "all":
        return colors
    
    filtered = []
    for color in colors:
        temp_info = calculate_color_temperature(color)
        if temp_info["temperature_class"] == temperature_filter:
            filtered.append(color)
    
    return filtered


def filter_by_brightness_range(
    colors: List[str],
    min_brightness: float = 0.0,
    max_brightness: float = 1.0
) -> List[str]:
    """
    Filter colors by brightness range.

    Args:
        colors: List of hex color codes
        min_brightness: Minimum brightness (0-1)
        max_brightness: Maximum brightness (0-1)

    Returns:
        Filtered list of colors
    """
    filtered = []
    for color in colors:
        brightness = calculate_brightness(color)
        if min_brightness <= brightness <= max_brightness:
            filtered.append(color)
    
    return filtered


def filter_by_saturation_range(
    colors: List[str],
    min_saturation: float = 0.0,
    max_saturation: float = 1.0
) -> List[str]:
    """
    Filter colors by saturation range.

    Args:
        colors: List of hex color codes
        min_saturation: Minimum saturation (0-1)
        max_saturation: Maximum saturation (0-1)

    Returns:
        Filtered list of colors
    """
    filtered = []
    for color in colors:
        saturation = calculate_saturation(color)
        if min_saturation <= saturation <= max_saturation:
            filtered.append(color)
    
    return filtered


def filter_exclude_colors(
    colors: List[str],
    exclude_colors: List[str],
    threshold: float = 0.1
) -> List[str]:
    """
    Filter out colors that are similar to exclude colors.

    Args:
        colors: List of hex color codes to filter
        exclude_colors: List of colors to exclude
        threshold: Color similarity threshold (0-1)

    Returns:
        Filtered list of colors (excluding similar ones)
    """
    def color_distance(hex1: str, hex2: str) -> float:
        """Calculate color distance in RGB space."""
        rgb1 = hex_to_rgb(hex1)
        rgb2 = hex_to_rgb(hex2)
        distance = np.sqrt(
            (rgb1[0] - rgb2[0])**2 +
            (rgb1[1] - rgb2[1])**2 +
            (rgb1[2] - rgb2[2])**2
        )
        # Normalize to 0-1 (max distance is sqrt(3*255^2))
        max_distance = np.sqrt(3 * 255**2)
        return distance / max_distance
    
    filtered = []
    for color in colors:
        exclude = False
        for exclude_color in exclude_colors:
            distance = color_distance(color, exclude_color)
            if distance < threshold:
                exclude = True
                break
        
        if not exclude:
            filtered.append(color)
    
    return filtered


def get_dominant_color_count(colors: List[str], threshold: float = 0.05) -> int:
    """
    Count the number of dominant/distinct colors in a list.

    Args:
        colors: List of hex color codes
        threshold: Minimum difference threshold for distinct colors

    Returns:
        Number of distinct dominant colors
    """
    if not colors:
        return 0
    
    def color_distance(hex1: str, hex2: str) -> float:
        rgb1 = hex_to_rgb(hex1)
        rgb2 = hex_to_rgb(hex2)
        distance = np.sqrt(
            (rgb1[0] - rgb2[0])**2 +
            (rgb1[1] - rgb2[1])**2 +
            (rgb1[2] - rgb2[2])**2
        )
        max_distance = np.sqrt(3 * 255**2)
        return distance / max_distance
    
    distinct_colors = []
    for color in colors:
        is_distinct = True
        for distinct_color in distinct_colors:
            if color_distance(color, distinct_color) < threshold:
                is_distinct = False
                break
        
        if is_distinct:
            distinct_colors.append(color)
    
    return len(distinct_colors)


def analyze_color_properties(colors: List[str]) -> Dict[str, Any]:
    """
    Analyze color properties for a palette.

    Args:
        colors: List of hex color codes

    Returns:
        Dictionary with comprehensive color analysis
    """
    if not colors:
        return {
            "temperature": "neutral",
            "average_brightness": 0.5,
            "average_saturation": 0.5,
            "dominant_color_count": 0
        }
    
    temperatures = [calculate_color_temperature(c) for c in colors]
    brightnesses = [calculate_brightness(c) for c in colors]
    saturations = [calculate_saturation(c) for c in colors]
    
    # Determine dominant temperature
    temp_classes = [t["temperature_class"] for t in temperatures]
    temp_counts = {cls: temp_classes.count(cls) for cls in set(temp_classes)}
    dominant_temp = max(temp_counts, key=temp_counts.get) if temp_counts else "neutral"
    
    return {
        "temperature": dominant_temp,
        "average_brightness": float(np.mean(brightnesses)),
        "average_saturation": float(np.mean(saturations)),
        "dominant_color_count": get_dominant_color_count(colors),
        "brightness_range": (float(np.min(brightnesses)), float(np.max(brightnesses))),
        "saturation_range": (float(np.min(saturations)), float(np.max(saturations)))
    }

