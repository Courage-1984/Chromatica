"""
Color palette export utilities for Chromatica.

This module provides functions to export color palettes in various formats:
- CSS variables
- SCSS/SASS
- JSON
- Adobe Swatch (.ase)
- Sketch format
- Figma format

Key Features:
- Multiple export format support
- Palette validation and normalization
- Error handling and logging
- Comprehensive format specifications
"""

import json
import struct
import logging
from typing import List, Dict, Any, Tuple
from pathlib import Path
import io

logger = logging.getLogger(__name__)


def export_css_variables(colors: List[str], weights: List[float] = None, prefix: str = "color") -> str:
    """
    Export colors as CSS variables.

    Args:
        colors: List of hex color codes (with or without #)
        weights: Optional list of weights (for generating variable names)
        prefix: Prefix for CSS variable names (default: "color")

    Returns:
        CSS string with variables
    """
    css_lines = [":root {"]
    
    for i, color in enumerate(colors):
        # Normalize color format
        color = color.lstrip("#").upper()
        var_name = f"{prefix}-{i+1}"
        
        if weights and i < len(weights):
            # Include weight in variable name or comment
            css_lines.append(f"  --{var_name}: #{color}; /* weight: {weights[i]*100:.1f}% */")
        else:
            css_lines.append(f"  --{var_name}: #{color};")
    
    css_lines.append("}")
    return "\n".join(css_lines)


def export_scss_variables(colors: List[str], weights: List[float] = None, prefix: str = "color") -> str:
    """
    Export colors as SCSS/SASS variables.

    Args:
        colors: List of hex color codes
        weights: Optional list of weights
        prefix: Prefix for SCSS variable names

    Returns:
        SCSS string with variables
    """
    scss_lines = []
    
    for i, color in enumerate(colors):
        color = color.lstrip("#").upper()
        var_name = f"${prefix}-{i+1}"
        
        if weights and i < len(weights):
            scss_lines.append(f"{var_name}: #{color}; // weight: {weights[i]*100:.1f}%")
        else:
            scss_lines.append(f"{var_name}: #{color};")
    
    return "\n".join(scss_lines)


def export_json_palette(colors: List[str], weights: List[float] = None, metadata: Dict[str, Any] = None) -> str:
    """
    Export colors as JSON palette.

    Args:
        colors: List of hex color codes
        weights: Optional list of weights
        metadata: Optional metadata dictionary

    Returns:
        JSON string representation
    """
    palette = {
        "name": metadata.get("name", "Chromatica Palette") if metadata else "Chromatica Palette",
        "colors": []
    }
    
    if metadata:
        palette.update({k: v for k, v in metadata.items() if k != "name"})
    
    for i, color in enumerate(colors):
        color_entry = {
            "hex": color.lstrip("#").upper(),
            "index": i
        }
        
        if weights and i < len(weights):
            color_entry["weight"] = float(weights[i])
            color_entry["percentage"] = float(weights[i] * 100)
        
        # Add RGB values
        rgb = hex_to_rgb(color)
        if rgb:
            color_entry["rgb"] = rgb
        
        palette["colors"].append(color_entry)
    
    return json.dumps(palette, indent=2)


def export_adobe_swatch(colors: List[str], weights: List[float] = None, filename: str = "palette.ase") -> bytes:
    """
    Export colors as Adobe Swatch (.ase) file.

    Args:
        colors: List of hex color codes
        weights: Optional list of weights (not used in ASE format)
        filename: Output filename (for metadata)

    Returns:
        Bytes representation of ASE file
    """
    # ASE file format structure
    # Header: "ASEF" + version + block count
    # Each color block: type + name length + name + color mode + color values
    
    buffer = io.BytesIO()
    
    # Write header
    buffer.write(b"ASEF")  # Signature
    buffer.write(struct.pack(">HH", 1, 0))  # Version 1.0
    buffer.write(struct.pack(">I", len(colors)))  # Block count
    
    # Write color blocks
    for i, color in enumerate(colors):
        # Block type: 0x0001 = color entry
        buffer.write(struct.pack(">H", 0x0001))
        
        # Color name (UTF-16BE)
        color_name = f"Color {i+1}"
        name_bytes = color_name.encode("utf-16-be")
        buffer.write(struct.pack(">H", len(name_bytes)))
        buffer.write(name_bytes)
        
        # Color mode: "RGB" = 0x0000
        buffer.write(struct.pack(">H", 0x0000))
        
        # RGB values (0-65535 range, 16-bit per channel)
        rgb = hex_to_rgb(color)
        if rgb:
            r = int(rgb[0] * 65535 / 255)
            g = int(rgb[1] * 65535 / 255)
            b = int(rgb[2] * 65535 / 255)
        else:
            r = g = b = 0
        
        buffer.write(struct.pack(">HHH", r, g, b))
        
        # Color type: 0 = Global
        buffer.write(struct.pack(">H", 0))
    
    return buffer.getvalue()


def export_sketch_palette(colors: List[str], weights: List[float] = None, name: str = "Chromatica Palette") -> Dict[str, Any]:
    """
    Export colors as Sketch palette format.

    Args:
        colors: List of hex color codes
        weights: Optional list of weights
        name: Palette name

    Returns:
        Dictionary in Sketch palette format
    """
    # Sketch uses a JSON format with specific structure
    palette = {
        "compatibleVersion": "2.0",
        "pluginVersion": "2.0",
        "colors": []
    }
    
    for i, color in enumerate(colors):
        color = color.lstrip("#").upper()
        rgb = hex_to_rgb(color)
        
        if rgb:
            sketch_color = {
                "red": rgb[0] / 255.0,
                "green": rgb[1] / 255.0,
                "blue": rgb[2] / 255.0,
                "alpha": 1.0
            }
            palette["colors"].append(sketch_color)
    
    return palette


def export_figma_palette(colors: List[str], weights: List[float] = None, name: str = "Chromatica Palette") -> Dict[str, Any]:
    """
    Export colors as Figma palette format.

    Args:
        colors: List of hex color codes
        weights: Optional list of weights
        name: Palette name

    Returns:
        Dictionary in Figma format
    """
    # Figma uses a JSON format
    palette = {
        "name": name,
        "colors": []
    }
    
    for i, color in enumerate(colors):
        color = color.lstrip("#").upper()
        rgb = hex_to_rgb(color)
        
        if rgb:
            figma_color = {
                "name": f"Color {i+1}",
                "color": {
                    "r": rgb[0] / 255.0,
                    "g": rgb[1] / 255.0,
                    "b": rgb[2] / 255.0
                }
            }
            
            if weights and i < len(weights):
                figma_color["weight"] = weights[i]
            
            palette["colors"].append(figma_color)
    
    return palette


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """
    Convert hex color to RGB tuple.

    Args:
        hex_color: Hex color code (with or without #)

    Returns:
        Tuple of (R, G, B) values 0-255
    """
    hex_color = hex_color.lstrip("#").upper()
    
    if len(hex_color) == 3:
        # Expand shorthand hex (e.g., "F00" -> "FF0000")
        hex_color = "".join([c * 2 for c in hex_color])
    
    if len(hex_color) != 6:
        logger.warning(f"Invalid hex color format: {hex_color}")
        return (0, 0, 0)
    
    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (r, g, b)
    except ValueError:
        logger.warning(f"Failed to parse hex color: {hex_color}")
        return (0, 0, 0)


def export_palette(colors: List[str], weights: List[float] = None, format_type: str = "css", **kwargs) -> str:
    """
    Export palette in the specified format.

    Args:
        colors: List of hex color codes
        weights: Optional list of weights
        format_type: Export format ("css", "scss", "json", "ase", "sketch", "figma")
        **kwargs: Additional format-specific parameters

    Returns:
        String or bytes representation of the palette
    """
    format_type = format_type.lower()
    
    if format_type == "css":
        return export_css_variables(colors, weights, kwargs.get("prefix", "color"))
    elif format_type == "scss" or format_type == "sass":
        return export_scss_variables(colors, weights, kwargs.get("prefix", "color"))
    elif format_type == "json":
        return export_json_palette(colors, weights, kwargs.get("metadata"))
    elif format_type == "ase":
        return export_adobe_swatch(colors, weights, kwargs.get("filename", "palette.ase"))
    elif format_type == "sketch":
        palette_dict = export_sketch_palette(colors, weights, kwargs.get("name", "Chromatica Palette"))
        return json.dumps(palette_dict, indent=2)
    elif format_type == "figma":
        palette_dict = export_figma_palette(colors, weights, kwargs.get("name", "Chromatica Palette"))
        return json.dumps(palette_dict, indent=2)
    else:
        raise ValueError(f"Unsupported export format: {format_type}")

