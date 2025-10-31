"""
Test Search Quality Tool
========================

Purpose:
    Run both Normal (Sinkhorn-EMD) and Fast (L2) searches for a given multi-color
    query, then extract and print the dominant colors for each returned image.

Usage (PowerShell on Windows):
    venv311\Scripts\activate
    # Ensure CHROMATICA_INDEX_DIR points to your index folder containing
    # chromatica_index.faiss and chromatica_metadata.db
    $env:CHROMATICA_INDEX_DIR = "C:\\path\\to\\index"

    # Single color
    python tools/test_search_quality.py --colors "#ff00ff" --weights "1.0" --k 9

    # Two colors
    python tools/test_search_quality.py --colors "#ff00ff,#00ff00" --weights "0.65,0.35" --k 12

    # Save JSON report
    python tools/test_search_quality.py --colors "#ff00ff,#00ff00" --weights "0.5,0.5" --k 12 --out report.json

Notes:
    - Respects CHROMATICA_INDEX_DIR to locate FAISS and DuckDB files.
    - Computes dominant colors from image files using OpenCV k-means in Lab space
      for better perceptual clustering, then reports hex in sRGB.
"""

from __future__ import annotations

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import cv2
from skimage import color as skcolor

# Local imports
import sys
from pathlib import Path as _PathForSys

# Ensure repo root (so 'src' package is importable) is on sys.path
_repo_root = _PathForSys(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.chromatica.indexing.store import AnnIndex, MetadataStore
from src.chromatica.core.query import create_query_histogram
from src.chromatica.search import find_similar


logger = logging.getLogger("chromatica.tools.test_search_quality")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_hex_list(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def parse_float_list(value: str) -> List[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def rgb_to_hex(rgb: np.ndarray) -> str:
    r, g, b = np.clip(np.round(rgb * 255), 0, 255).astype(np.uint8)
    return f"#{r:02x}{g:02x}{b:02x}"


def extract_dominant_colors(image_path: str, num_colors: int = 5, sample: int = 256) -> List[Tuple[str, float]]:
    """
    Extract dominant colors using k-means clustering in Lab space.

    Returns list of (hex, proportion) sorted by proportion desc.
    """
    if not image_path or not Path(image_path).exists():
        return []

    # Load BGR image
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return []

    # Resize for speed
    h, w = img_bgr.shape[:2]
    scale = sample / max(h, w)
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # Convert to RGB (0..1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) / 255.0

    # Convert to Lab for perceptual clustering
    img_lab = skcolor.rgb2lab(img_rgb)
    data = img_lab.reshape(-1, 3).astype(np.float32)

    # k-means
    K = max(1, int(num_colors))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    attempts = 3
    _compact, labels, centers = cv2.kmeans(data, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)

    # Proportions
    labels = labels.reshape(-1)
    counts = np.bincount(labels, minlength=K).astype(np.float64)
    proportions = counts / counts.sum()

    # Convert cluster centers Lab -> RGB -> HEX
    centers_lab = centers.astype(np.float64)
    centers_rgb = skcolor.lab2rgb(centers_lab.reshape(1, K, 3)).reshape(K, 3)
    results = [(rgb_to_hex(centers_rgb[i]), float(proportions[i])) for i in range(K)]
    # Sort by proportion desc
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def load_components(index_dir: Path) -> Tuple[AnnIndex, MetadataStore]:
    faiss_path = index_dir / "chromatica_index.faiss"
    db_path = index_dir / "chromatica_metadata.db"
    if not faiss_path.exists():
        raise FileNotFoundError(f"FAISS index not found at {faiss_path}")
    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB database not found at {db_path}")
    index = AnnIndex(index_path=str(faiss_path))
    store = MetadataStore(db_path=str(db_path))
    return index, store


def run_search_and_report(colors: List[str], weights: List[float], k: int, mode: str, index: AnnIndex, store: MetadataStore, top_colors: int) -> List[Dict[str, Any]]:
    assert mode in ("fast", "normal")
    query_hist = create_query_histogram(colors, weights)
    fast_mode = mode == "fast"
    results = find_similar(query_hist, index=index, store=store, k=k, fast_mode=fast_mode)

    report: List[Dict[str, Any]] = []
    for r in results:
        file_path = getattr(r, "file_path", None)
        dom = extract_dominant_colors(file_path, num_colors=top_colors) if file_path else []
        report.append({
            "image_id": getattr(r, "image_id", None),
            "distance": float(getattr(r, "distance", 0.0)),
            "ann_score": float(getattr(r, "ann_score", 0.0)),
            "file_path": file_path,
            "image_url": getattr(r, "image_url", None),
            "dominant_colors": [{"hex": c, "p": round(p, 4)} for c, p in dom],
        })
    return report


def main():
    parser = argparse.ArgumentParser(description="Test Fast vs Normal search and inspect dominant colors")
    parser.add_argument("--colors", required=True, help="Comma-separated hex colors, e.g. #ff00ff,#00ff00")
    parser.add_argument("--weights", required=True, help="Comma-separated weights, e.g. 0.6,0.4")
    parser.add_argument("--k", type=int, default=9, help="Number of results to return")
    parser.add_argument("--index-dir", type=str, default=os.getenv("CHROMATICA_INDEX_DIR", "./index"))
    parser.add_argument("--top-colors", type=int, default=5, help="Top dominant colors to extract for each result")
    parser.add_argument("--out", type=str, default=None, help="Optional path to save JSON report")
    args = parser.parse_args()

    colors = parse_hex_list(args.colors)
    weights = parse_float_list(args.weights)
    if len(colors) != len(weights):
        raise SystemExit("colors and weights must have the same length")

    index_dir = Path(args.index_dir).resolve()
    index, store = load_components(index_dir)

    # Run both modes
    normal_report = run_search_and_report(colors, weights, args.k, "normal", index, store, args.top_colors)
    fast_report = run_search_and_report(colors, weights, args.k, "fast", index, store, args.top_colors)

    # Pretty print
    def print_section(title: str, data: List[Dict[str, Any]]):
        print("\n" + title)
        print("=" * len(title))
        for i, item in enumerate(data, 1):
            dom_str = ", ".join([f"{d['hex']} ({d['p']*100:.1f}%)" for d in item.get("dominant_colors", [])])
            print(f"{i:2d}. id={item['image_id']}  d={item['distance']:.4f}  ann={item['ann_score']:.4f}  {dom_str}")

    print_section("Normal Mode (Sinkhorn-EMD)", normal_report)
    print_section("Fast Mode (L2)", fast_report)

    if args.out:
        payload = {
            "query": {"colors": colors, "weights": weights, "k": args.k},
            "normal": normal_report,
            "fast": fast_report,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved report to {args.out}")


if __name__ == "__main__":
    main()


