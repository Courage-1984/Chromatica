"""
Advanced features router for Chromatica.

This module provides API endpoints for advanced features:
- Palette export in various formats
- Color harmony analysis
- Gradient generation
- Color statistics
- Favorite palettes
- Advanced filters

Key Features:
- Multiple export formats
- Color theory analysis
- Gradient generation
- Statistics dashboard
"""

import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from pathlib import Path
import json
import time

from ..models import (
    PaletteExportRequest,
    ColorHarmonyAnalysis,
    GradientGenerationRequest,
    ColorStatistics,
    FavoritePalette,
    AdvancedSearchFilters
)
from ...utils.palette_export import export_palette
from ...utils.color_harmony import (
    detect_harmony_type,
    suggest_harmony_improvements,
    calculate_harmony_score
)
from ...utils.gradient_generator import (
    generate_linear_gradient_css,
    generate_gradient_image,
    generate_gradient_image_base64
)
from ...utils.color_filters import (
    analyze_color_properties,
    calculate_color_temperature,
    calculate_brightness,
    calculate_saturation
)
from fastapi.responses import Response, JSONResponse

router = APIRouter(
    prefix="/advanced",
    tags=["Advanced Features"],
)

logger = logging.getLogger(__name__)


@router.post("/palette/export")
async def export_palette_endpoint(request: PaletteExportRequest):
    """
    Export color palette in various formats.

    Supported formats:
    - css: CSS variables
    - scss/sass: SCSS/SASS variables
    - json: JSON format
    - ase: Adobe Swatch file
    - sketch: Sketch palette format
    - figma: Figma palette format

    Args:
        request: Palette export request

    Returns:
        Exported palette in requested format
    """
    try:
        logger.info(f"Exporting palette in {request.format_type} format")
        
        # Export palette
        result = export_palette(
            colors=request.colors,
            weights=request.weights,
            format_type=request.format_type,
            **({"metadata": request.metadata} if request.metadata else {})
        )
        
        # Handle binary formats (ASE)
        if request.format_type == "ase":
            return Response(
                content=result,
                media_type="application/octet-stream",
                headers={
                    "Content-Disposition": f'attachment; filename="palette.ase"'
                }
            )
        
        # Return text-based formats
        return JSONResponse(
            content={
                "format": request.format_type,
                "data": result,
                "colors_count": len(request.colors)
            }
        )
    
    except ValueError as e:
        logger.error(f"Invalid export format: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to export palette: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@router.post("/palette/harmony")
async def analyze_harmony(
    colors: str = Query(..., description="Comma-separated hex color codes"),
    suggest: bool = Query(False, description="Include improvement suggestions")
):
    """
    Analyze color harmony in a palette.

    Args:
        colors: Comma-separated hex color codes
        suggest: Whether to include improvement suggestions

    Returns:
        Color harmony analysis with suggestions if requested
    """
    try:
        # Parse colors
        color_list = [c.strip().lstrip("#").upper() for c in colors.split(",") if c.strip()]
        if not color_list:
            raise ValueError("At least one color must be provided")
        
        # Detect harmony type
        harmony_info = detect_harmony_type(color_list)
        
        # Calculate harmony score
        score = calculate_harmony_score(color_list)
        
        # Get suggestions if requested
        suggestions = None
        if suggest:
            suggestions_data = suggest_harmony_improvements(color_list)
            suggestions = suggestions_data.get("suggestions", [])
        
        return ColorHarmonyAnalysis(
            harmony_type=harmony_info["type"],
            confidence=float(harmony_info["confidence"]),
            description=harmony_info["description"],
            suggestions=suggestions
        )
    
    except Exception as e:
        logger.error(f"Failed to analyze harmony: {e}")
        raise HTTPException(status_code=500, detail=f"Harmony analysis failed: {str(e)}")


@router.post("/gradient/generate")
async def generate_gradient(request: GradientGenerationRequest):
    """
    Generate a color gradient from a palette.

    Args:
        request: Gradient generation request

    Returns:
        Gradient image as PNG bytes or CSS string
    """
    try:
        logger.info(f"Generating {request.gradient_type} gradient")
        
        if request.gradient_type == "css" or request.gradient_type == "linear":
            # Return CSS gradient string
            css = generate_linear_gradient_css(
                colors=request.colors,
                weights=request.weights,
                direction="to right"
            )
            return JSONResponse(
                content={
                    "type": "css",
                    "gradient": css,
                    "format": "linear-gradient"
                }
            )
        else:
            # Generate gradient image
            img_bytes = generate_gradient_image(
                colors=request.colors,
                weights=request.weights,
                width=request.width or 800,
                height=request.height or 200,
                direction=request.direction or "horizontal"
            )
            
            # Return as base64 for JSON or raw bytes
            import base64
            base64_img = base64.b64encode(img_bytes).decode("utf-8")
            
            return JSONResponse(
                content={
                    "type": "image",
                    "format": "png",
                    "width": request.width or 800,
                    "height": request.height or 200,
                    "data": base64_img,
                    "data_url": f"data:image/png;base64,{base64_img}"
                }
            )
    
    except Exception as e:
        logger.error(f"Failed to generate gradient: {e}")
        raise HTTPException(status_code=500, detail=f"Gradient generation failed: {str(e)}")


@router.post("/statistics/analyze")
async def analyze_color_statistics(
    colors: str = Query(..., description="Comma-separated hex color codes")
):
    """
    Analyze color statistics for a palette.

    Args:
        colors: Comma-separated hex color codes

    Returns:
        Color statistics including distribution, brightness, saturation, etc.
    """
    try:
        # Parse colors
        color_list = [c.strip().lstrip("#").upper() for c in colors.split(",") if c.strip()]
        if not color_list:
            raise ValueError("At least one color must be provided")
        
        # Analyze color properties
        properties = analyze_color_properties(color_list)
        
        # Count color frequencies (if duplicates exist)
        from collections import Counter
        color_counter = Counter(color_list)
        most_common = [
            {"color": f"#{color}", "frequency": count}
            for color, count in color_counter.most_common(10)
        ]
        
        # Build color distribution
        distribution = {f"#{color}": count for color, count in color_counter.items()}
        
        # Temperature distribution
        temp_distribution = {"warm": 0, "cool": 0, "neutral": 0}
        for color in color_list:
            temp_info = calculate_color_temperature(f"#{color}")
            temp_distribution[temp_info["temperature_class"]] += 1
        
        return ColorStatistics(
            most_common_colors=most_common,
            color_distribution=distribution,
            average_brightness=properties["average_brightness"],
            average_saturation=properties["average_saturation"],
            temperature_distribution=temp_distribution
        )
    
    except Exception as e:
        logger.error(f"Failed to analyze color statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Statistics analysis failed: {str(e)}")


# Favorite palettes storage (in-memory for now, can be moved to database)
_favorite_palettes: Dict[str, FavoritePalette] = {}


@router.post("/favorites/save")
async def save_favorite(request: FavoritePalette):
    """
    Save a favorite color palette.

    Args:
        request: Favorite palette data

    Returns:
        Saved favorite palette with ID
    """
    try:
        import uuid
        
        # Generate ID if not provided
        if not request.id:
            palette_id = str(uuid.uuid4())
        else:
            palette_id = request.id
        
        # Create favorite palette
        favorite = FavoritePalette(
            id=palette_id,
            name=request.name,
            colors=request.colors,
            weights=request.weights,
            created_at=request.created_at or time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # Store favorite
        _favorite_palettes[palette_id] = favorite
        
        logger.info(f"Saved favorite palette: {palette_id}")
        
        return favorite
    
    except Exception as e:
        logger.error(f"Failed to save favorite: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save favorite: {str(e)}")


@router.get("/favorites/list")
async def list_favorites():
    """
    List all favorite color palettes.

    Returns:
        List of favorite palettes
    """
    try:
        return {
            "favorites": list(_favorite_palettes.values()),
            "count": len(_favorite_palettes)
        }
    except Exception as e:
        logger.error(f"Failed to list favorites: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list favorites: {str(e)}")


@router.delete("/favorites/{favorite_id}")
async def delete_favorite(favorite_id: str):
    """
    Delete a favorite color palette.

    Args:
        favorite_id: Favorite palette ID

    Returns:
        Success status
    """
    try:
        if favorite_id not in _favorite_palettes:
            raise HTTPException(status_code=404, detail="Favorite not found")
        
        del _favorite_palettes[favorite_id]
        logger.info(f"Deleted favorite palette: {favorite_id}")
        
        return {"status": "success", "message": "Favorite deleted"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete favorite: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete favorite: {str(e)}")

