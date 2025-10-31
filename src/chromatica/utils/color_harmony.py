"""
Color harmony analysis utilities for Chromatica.

This module provides functions to analyze color harmonies and suggest improvements:
- Detect harmony types (complementary, analogous, triadic, etc.)
- Suggest color improvements
- Calculate harmony scores

Key Features:
- Multiple harmony type detection
- Color theory-based analysis
- Harmony scoring and suggestions
"""

import logging
import math
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


def hex_to_hsv(hex_color: str) -> Tuple[float, float, float]:
    """
    Convert hex color to HSV.

    Args:
        hex_color: Hex color code (with or without #)

    Returns:
        Tuple of (H, S, V) values (0-360, 0-100, 0-100)
    """
    hex_color = hex_color.lstrip("#").upper()
    
    if len(hex_color) == 3:
        hex_color = "".join([c * 2 for c in hex_color])
    
    if len(hex_color) != 6:
        return (0, 0, 0)
    
    try:
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        
        # Convert RGB to HSV
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        delta = max_val - min_val
        
        # Value
        v = max_val * 100
        
        # Saturation
        if max_val == 0:
            s = 0
        else:
            s = (delta / max_val) * 100
        
        # Hue
        if delta == 0:
            h = 0
        elif max_val == r:
            h = 60 * (((g - b) / delta) % 6)
        elif max_val == g:
            h = 60 * (((b - r) / delta) + 2)
        else:  # max_val == b
            h = 60 * (((r - g) / delta) + 4)
        
        if h < 0:
            h += 360
        
        return (h, s, v)
    except ValueError:
        return (0, 0, 0)


def hsv_to_hex(h: float, s: float, v: float) -> str:
    """
    Convert HSV to hex color.

    Args:
        h: Hue (0-360)
        s: Saturation (0-100)
        v: Value (0-100)

    Returns:
        Hex color code (without #)
    """
    h = h / 360.0
    s = s / 100.0
    v = v / 100.0
    
    if s == 0:
        r = g = b = v
    else:
        i = int(h * 6)
        f = (h * 6) - i
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))
        
        i %= 6
        
        if i == 0:
            r, g, b = v, t, p
        elif i == 1:
            r, g, b = q, v, p
        elif i == 2:
            r, g, b = p, v, t
        elif i == 3:
            r, g, b = p, q, v
        elif i == 4:
            r, g, b = t, p, v
        else:
            r, g, b = v, p, q
    
    r = max(0, min(255, int(r * 255)))
    g = max(0, min(255, int(g * 255)))
    b = max(0, min(255, int(b * 255)))
    
    return f"{r:02x}{g:02x}{b:02x}".upper()


def detect_harmony_type(colors: List[str]) -> Dict[str, Any]:
    """
    Detect the type of color harmony present in the palette.

    Args:
        colors: List of hex color codes

    Returns:
        Dictionary with harmony type and details
    """
    if len(colors) < 2:
        return {
            "type": "unknown",
            "confidence": 0.0,
            "description": "Not enough colors to detect harmony"
        }
    
    # Convert to HSV
    hsv_colors = [hex_to_hsv(c) for c in colors]
    hues = [h for h, s, v in hsv_colors]
    
    # Sort hues
    sorted_hues = sorted(hues)
    
    # Calculate hue differences
    hue_diffs = []
    for i in range(len(sorted_hues)):
        diff = (sorted_hues[(i + 1) % len(sorted_hues)] - sorted_hues[i]) % 360
        hue_diffs.append(diff)
    
    # Detect harmony types
    results = []
    
    # Complementary (2 colors, 180° apart)
    if len(colors) == 2:
        diff = min(abs(hues[0] - hues[1]), 360 - abs(hues[0] - hues[1]))
        if abs(diff - 180) < 15:  # Within 15 degrees
            results.append({
                "type": "complementary",
                "confidence": 1.0 - (abs(diff - 180) / 15),
                "description": "Two colors opposite on the color wheel"
            })
    
    # Analogous (3+ colors, similar hues)
    if len(colors) >= 3:
        avg_diff = np.mean(hue_diffs)
        if avg_diff < 30:  # Colors are close together
            results.append({
                "type": "analogous",
                "confidence": 1.0 - (avg_diff / 30),
                "description": "Colors adjacent on the color wheel"
            })
    
    # Triadic (3 colors, 120° apart)
    if len(colors) == 3:
        diffs = [min(abs(hue_diffs[i] - 120), 360 - abs(hue_diffs[i] - 120)) for i in range(3)]
        avg_diff_error = np.mean(diffs)
        if avg_diff_error < 15:
            results.append({
                "type": "triadic",
                "confidence": 1.0 - (avg_diff_error / 15),
                "description": "Three colors evenly spaced (120° apart)"
            })
    
    # Split-complementary (base color + two colors 150° from its complement)
    if len(colors) == 3:
        # Check if one color is opposite to the average of the other two
        base_hue = hues[0]
        other_hues = hues[1:]
        avg_other = np.mean(other_hues)
        
        diff_to_avg = min(abs(base_hue - avg_other), 360 - abs(base_hue - avg_other))
        if abs(diff_to_avg - 180) < 20:
            # Check if other two are close to each other
            other_diff = min(abs(other_hues[0] - other_hues[1]), 360 - abs(other_hues[0] - other_hues[1]))
            if other_diff < 30:
                results.append({
                    "type": "split-complementary",
                    "confidence": 0.8,
                    "description": "Base color with two colors adjacent to its complement"
                })
    
    # Tetradic (4 colors, rectangle on color wheel)
    if len(colors) == 4:
        # Check if colors form two complementary pairs
        pair1_diff = min(abs(sorted_hues[0] - sorted_hues[2]), 360 - abs(sorted_hues[0] - sorted_hues[2]))
        pair2_diff = min(abs(sorted_hues[1] - sorted_hues[3]), 360 - abs(sorted_hues[1] - sorted_hues[3]))
        
        if abs(pair1_diff - 180) < 20 and abs(pair2_diff - 180) < 20:
            results.append({
                "type": "tetradic",
                "confidence": 0.8,
                "description": "Four colors forming a rectangle on the color wheel"
            })
    
    # Return best match
    if results:
        best = max(results, key=lambda x: x["confidence"])
        return best
    else:
        return {
            "type": "custom",
            "confidence": 0.5,
            "description": "Custom color combination"
        }


def suggest_harmony_improvements(colors: List[str], target_harmony: Optional[str] = None) -> Dict[str, Any]:
    """
    Suggest improvements to achieve a specific harmony type.

    Args:
        colors: List of current hex color codes
        target_harmony: Target harmony type ("complementary", "analogous", "triadic", etc.)

    Returns:
        Dictionary with suggested colors and improvements
    """
    hsv_colors = [hex_to_hsv(c) for c in colors]
    
    suggestions = {
        "current_harmony": detect_harmony_type(colors),
        "suggestions": []
    }
    
    if target_harmony is None:
        target_harmony = "complementary" if len(colors) == 2 else "triadic" if len(colors) == 3 else "analogous"
    
    if target_harmony == "complementary" and len(colors) >= 1:
        # Suggest complementary color
        base_h = hsv_colors[0][0]
        comp_h = (base_h + 180) % 360
        comp_color = hsv_to_hex(comp_h, hsv_colors[0][1], hsv_colors[0][2])
        suggestions["suggestions"].append({
            "type": "complementary",
            "colors": [colors[0], f"#{comp_color}"],
            "description": "Add complementary color"
        })
    
    elif target_harmony == "triadic" and len(colors) >= 1:
        # Suggest triadic colors
        base_h = hsv_colors[0][0]
        tri1_h = (base_h + 120) % 360
        tri2_h = (base_h + 240) % 360
        tri1_color = hsv_to_hex(tri1_h, hsv_colors[0][1], hsv_colors[0][2])
        tri2_color = hsv_to_hex(tri2_h, hsv_colors[0][1], hsv_colors[0][2])
        suggestions["suggestions"].append({
            "type": "triadic",
            "colors": [colors[0], f"#{tri1_color}", f"#{tri2_color}"],
            "description": "Add triadic colors (120° apart)"
        })
    
    elif target_harmony == "analogous" and len(colors) >= 1:
        # Suggest analogous colors
        base_h = hsv_colors[0][0]
        s = hsv_colors[0][1]
        v = hsv_colors[0][2]
        
        analog1_h = (base_h - 30) % 360
        analog2_h = (base_h + 30) % 360
        analog1_color = hsv_to_hex(analog1_h, s, v)
        analog2_color = hsv_to_hex(analog2_h, s, v)
        suggestions["suggestions"].append({
            "type": "analogous",
            "colors": [colors[0], f"#{analog1_color}", f"#{analog2_color}"],
            "description": "Add analogous colors (30° apart)"
        })
    
    return suggestions


def calculate_harmony_score(colors: List[str]) -> float:
    """
    Calculate a harmony score for the color palette (0-1).

    Args:
        colors: List of hex color codes

    Returns:
        Harmony score between 0 and 1
    """
    if len(colors) < 2:
        return 0.5
    
    harmony_info = detect_harmony_type(colors)
    
    # Base score from harmony type confidence
    base_score = harmony_info["confidence"]
    
    # Adjust based on saturation and value balance
    hsv_colors = [hex_to_hsv(c) for c in colors]
    sats = [s for h, s, v in hsv_colors]
    vals = [v for h, s, v in hsv_colors]
    
    # Penalize if saturation or value varies too much
    sat_variance = np.std(sats) / 100.0
    val_variance = np.std(vals) / 100.0
    
    # Lower variance = higher score
    variance_penalty = (sat_variance + val_variance) / 2.0
    adjusted_score = base_score * (1.0 - variance_penalty * 0.3)
    
    return max(0.0, min(1.0, adjusted_score))

