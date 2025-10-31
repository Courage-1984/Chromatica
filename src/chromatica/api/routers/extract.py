import logging
from typing import List
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from ..models import ColorExtractionResponse

from ...utils.color_utils import extract_dominant_colors_with_weights

router = APIRouter(
    tags=["Color Extraction"],
)

extract_logger = logging.getLogger("chromatica.extract")


@router.post("/extract_colors", response_model=ColorExtractionResponse)
async def extract_colors(
    image: UploadFile = File(
        ..., description="Image file to analyze (JPG, PNG, GIF, WebP)."
    ),
    num_colors: int = Form(
        5, description="Number of dominant colors to extract (1-10)."
    ),
):
    """
    Analyzes an uploaded image, extracts its dominant colors, and returns them
    with corresponding weights, suitable for a color search query.
    """
    try:
        # 1. Validation
        if num_colors < 1 or num_colors > 10:
            raise HTTPException(
                status_code=400, detail="num_colors must be between 1 and 10."
            )

        allowed_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
        if image.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"File type not supported: {image.content_type}. Must be one of: {', '.join(allowed_types)}",
            )

        # 2. Read Image Data
        image_bytes = await image.read()

        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Image file is empty.")

        extract_logger.info(
            f"Extracting {num_colors} colors from uploaded image: {image.filename}"
        )

        # 3. Core Extraction Logic
        colors, weights = extract_dominant_colors_with_weights(image_bytes, num_colors)

        if not colors:
            raise HTTPException(
                status_code=500,
                detail="Failed to extract colors. Image may be invalid or too small.",
            )

        extract_logger.info(f"Extracted {len(colors)} colors: {colors}")
        extract_logger.info(f"Color weights: {weights}")

        return ColorExtractionResponse(
            colors=colors, weights=weights, num_colors=len(colors), status="ok"
        )

    except HTTPException:
        # Re-raise explicit HTTP exceptions
        raise
    except Exception as e:
        extract_logger.error(f"Error extracting colors from image: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process image for color extraction: {str(e)}",
        )
