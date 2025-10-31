import logging
from typing import List, Tuple
import io
import numpy as np
import cv2  # OpenCV for image processing
from skimage.color import rgb2lab
from collections import Counter
from sklearn.cluster import KMeans

logger = logging.getLogger("chromatica.api")


def extract_dominant_colors_with_weights(
    image_bytes: bytes, num_colors: int = 5
) -> Tuple[List[str], List[float]]:
    """
    Extracts the most dominant colors and their relative weights from an image.

    Args:
        image_bytes: The image content as raw bytes.
        num_colors: The maximum number of dominant colors to extract.

    Returns:
        A tuple containing:
        - A list of color hex codes (e.g., ['#FF0000', '#00FF00'])
        - A list of relative weights (e.g., [70.0, 30.0])
    """
    try:
        # Decode image using OpenCV
        img_np = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Failed to decode image bytes.")

        # Resize image to speed up clustering
        h, w, _ = img.shape
        if h > 300 or w > 300:
            scale = min(300 / h, 300 / w)
            img = cv2.resize(
                img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )

        # Convert BGR (OpenCV default) to RGB and reshape for clustering
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pixels = img_rgb.reshape((-1, 3))

        # Use KMeans for color clustering
        # Ensure k is not greater than the number of unique pixels
        k = min(num_colors, len(np.unique(pixels, axis=0)))

        if k == 0:
            return ["#000000"], [100.0]

        logger.debug(f"Running KMeans with k={k}")

        kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42, max_iter=300).fit(
            pixels
        )

        # Get the color centers (RGB)
        centers = kmeans.cluster_centers_.astype(int)

        # Get the count of pixels in each cluster (weights)
        labels = kmeans.labels_
        counts = Counter(labels)
        total_pixels = len(labels)

        # Calculate weights as percentages and sort
        color_data = []
        for i, center in enumerate(centers):
            weight = (counts[i] / total_pixels) * 100
            hex_color = "#{:02x}{:02x}{:02x}".format(center[0], center[1], center[2])
            color_data.append((hex_color, weight))

        # Sort by weight in descending order
        color_data.sort(key=lambda x: x[1], reverse=True)

        # Split into separate lists
        colors = [item[0] for item in color_data]
        weights = [round(item[1], 2) for item in color_data]

        # Normalize weights to ensure they sum up to 100 (due to rounding)
        if weights:
            current_sum = sum(weights)
            if current_sum != 100.0:
                difference = 100.0 - current_sum
                # Add/subtract the difference from the largest weight
                max_index = weights.index(max(weights))
                weights[max_index] = round(weights[max_index] + difference, 2)

        return colors, weights

    except Exception as e:
        logger.error(f"Error in extract_dominant_colors_with_weights: {e}")
        # Fallback: Return a single black color if extraction fails
        return ["#000000"], [100.0]


def extract_dominant_colors(image_bytes: bytes, num_colors: int = 5) -> List[str]:
    """
    Simple wrapper to return only the color hex codes.
    """
    colors, _ = extract_dominant_colors_with_weights(image_bytes, num_colors)
    return colors

