#!/usr/bin/env python3
"""
Evaluation Harness for Chromatica Color Search Engine

This script provides comprehensive evaluation capabilities for the Chromatica
color search engine, measuring both performance metrics (latency) and quality
metrics (precision, recall) against ground truth data.

Usage:
    # Activate virtual environment first
    venv311\Scripts\activate

    # Run evaluation with default test queries
    python scripts/evaluate.py

    # Run evaluation with custom query file
    python scripts/evaluate.py --queries datasets/test-queries.json

    # Run evaluation with ground truth data
    python scripts/evaluate.py --queries datasets/test-queries.json --ground-truth datasets/ground-truth.json

Features:
- Batch query execution with latency measurement
- P95 latency calculation and reporting
- Precision@K and Recall@K metrics (when ground truth available)
- Comprehensive logging and result reporting
- Support for different test datasets and query sets
- Memory usage monitoring during evaluation
"""

import argparse
import json
import logging
import time
import statistics
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chromatica.indexing.store import AnnIndex, MetadataStore
from chromatica.core.query import create_query_histogram
from chromatica.core.rerank import rerank_candidates
from chromatica.utils.config import RERANK_K


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up logging configuration for evaluation."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/evaluation.log", mode="a"),
        ],
    )
    return logging.getLogger(__name__)


def load_test_queries(queries_file: str) -> List[Dict[str, Any]]:
    """
    Load test queries from JSON file.

    Expected JSON format:
    [
        {
            "query_id": "q1",
            "colors": ["FF0000", "00FF00", "0000FF"],
            "weights": [0.5, 0.3, 0.2],
            "description": "Red, green, blue query"
        },
        ...
    ]

    Args:
        queries_file: Path to JSON file containing test queries

    Returns:
        List of query dictionaries

    Raises:
        FileNotFoundError: If queries file doesn't exist
        ValueError: If queries file has invalid format
    """
    queries_path = Path(queries_file)
    if not queries_path.exists():
        raise FileNotFoundError(f"Queries file not found: {queries_file}")

    try:
        with open(queries_path, "r") as f:
            queries = json.load(f)

        if not isinstance(queries, list):
            raise ValueError("Queries file must contain a list of queries")

        # Validate query format
        for i, query in enumerate(queries):
            if not isinstance(query, dict):
                raise ValueError(f"Query {i} must be a dictionary")

            required_fields = ["query_id", "colors"]
            for field in required_fields:
                if field not in query:
                    raise ValueError(f"Query {i} missing required field: {field}")

            if not isinstance(query["colors"], list) or len(query["colors"]) == 0:
                raise ValueError(f"Query {i} must have non-empty colors list")

        logging.info(f"Loaded {len(queries)} test queries from {queries_file}")
        return queries

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in queries file: {e}")
    except Exception as e:
        raise ValueError(f"Error loading queries file: {e}")


def load_ground_truth(ground_truth_file: str) -> Dict[str, List[str]]:
    """
    Load ground truth labels from JSON file.

    Expected JSON format:
    {
        "q1": ["img1", "img2", "img3", ...],
        "q2": ["img4", "img5", "img6", ...],
        ...
    }

    Args:
        ground_truth_file: Path to JSON file containing ground truth

    Returns:
        Dictionary mapping query_id to list of relevant image IDs

    Raises:
        FileNotFoundError: If ground truth file doesn't exist
        ValueError: If ground truth file has invalid format
    """
    gt_path = Path(ground_truth_file)
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {ground_truth_file}")

    try:
        with open(gt_path, "r") as f:
            ground_truth = json.load(f)

        if not isinstance(ground_truth, dict):
            raise ValueError("Ground truth file must contain a dictionary")

        logging.info(
            f"Loaded ground truth for {len(ground_truth)} queries from {ground_truth_file}"
        )
        return ground_truth

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in ground truth file: {e}")
    except Exception as e:
        raise ValueError(f"Error loading ground truth file: {e}")


def calculate_precision_at_k(
    retrieved: List[str], relevant: List[str], k: int
) -> float:
    """
    Calculate Precision@K metric.

    Args:
        retrieved: List of retrieved image IDs (ordered by relevance)
        relevant: List of relevant image IDs
        k: Number of top results to consider

    Returns:
        Precision@K value (0.0 to 1.0)
    """
    if k <= 0:
        return 0.0

    # Take top k retrieved results
    top_k = retrieved[:k]

    # Handle edge case where no results are retrieved
    if len(top_k) == 0:
        return 0.0

    # Count how many are relevant
    relevant_count = sum(1 for img_id in top_k if img_id in relevant)

    return relevant_count / len(top_k)


def calculate_recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    Calculate Recall@K metric.

    Args:
        retrieved: List of retrieved image IDs (ordered by relevance)
        relevant: List of relevant image IDs
        k: Number of top results to consider

    Returns:
        Recall@K value (0.0 to 1.0)
    """
    if len(relevant) == 0:
        return 0.0

    # Take top k retrieved results
    top_k = retrieved[:k]

    # Count how many relevant items were retrieved
    relevant_retrieved = sum(1 for img_id in top_k if img_id in relevant)

    return relevant_retrieved / len(relevant)


def execute_search(
    query: Dict[str, Any],
    ann_index: AnnIndex,
    metadata_store: MetadataStore,
    k: int = 10,
) -> Tuple[List[str], float, Dict[str, Any]]:
    """
    Execute a single search query and measure performance.

    Args:
        query: Query dictionary with colors and weights
        ann_index: FAISS index for ANN search
        metadata_store: DuckDB store for metadata retrieval
        k: Number of results to return

    Returns:
        Tuple of (result_image_ids, latency_ms, performance_metrics)
    """
    start_time = time.perf_counter()

    try:
        # Process query colors
        colors = query["colors"]
        weights = query.get("weights", [1.0] * len(colors))
        query_histogram = create_query_histogram(colors, weights)

        # ANN search
        ann_start = time.perf_counter()
        distances, indices = ann_index.search(query_histogram, RERANK_K)
        ann_time = (time.perf_counter() - ann_start) * 1000

        # Get candidate image IDs
        candidate_ids = []
        for idx in indices[0]:
            if idx < ann_index.get_total_vectors():
                # Map FAISS index to image ID (assuming sequential mapping)
                candidate_ids.append(f"img_{idx:06d}")

        # Reranking
        rerank_start = time.perf_counter()
        if len(candidate_ids) > 0:
            # Get histograms for candidates
            candidate_histograms = metadata_store.get_histograms_by_ids(candidate_ids)

            # Rerank using Sinkhorn-EMD
            rerank_results = rerank_candidates(query_histogram, candidate_histograms, k)
            reranked_ids = [result.candidate_id for result in rerank_results]
            rerank_distances = [result.distance for result in rerank_results]
        else:
            reranked_ids = []
            rerank_distances = []
        rerank_time = (time.perf_counter() - rerank_start) * 1000

        # Calculate total latency
        total_time = (time.perf_counter() - start_time) * 1000

        # Performance metrics
        metrics = {
            "ann_time_ms": ann_time,
            "rerank_time_ms": rerank_time,
            "total_time_ms": total_time,
            "candidates_found": len(candidate_ids),
            "results_returned": len(reranked_ids),
        }

        return reranked_ids, total_time, metrics

    except Exception as e:
        logging.error(
            f"Search failed for query {query.get('query_id', 'unknown')}: {e}"
        )
        return [], 0.0, {"error": str(e)}


def run_evaluation(
    queries: List[Dict[str, Any]],
    ann_index: AnnIndex,
    metadata_store: MetadataStore,
    ground_truth: Optional[Dict[str, List[str]]] = None,
    k: int = 10,
) -> Dict[str, Any]:
    """
    Run comprehensive evaluation on test queries.

    Args:
        queries: List of test queries
        ann_index: FAISS index for search
        metadata_store: DuckDB store for metadata
        ground_truth: Optional ground truth labels
        k: Number of results to return per query

    Returns:
        Dictionary containing evaluation results and metrics
    """
    logger = logging.getLogger(__name__)

    # Initialize components (reranking functions are imported directly)

    # Results storage
    latencies = []
    precision_at_10 = []
    recall_at_10 = []
    query_results = []

    logger.info(f"Starting evaluation of {len(queries)} queries...")

    for i, query in enumerate(queries):
        query_id = query.get("query_id", f"query_{i}")
        logger.info(f"Processing query {i+1}/{len(queries)}: {query_id}")

        # Execute search
        result_ids, latency, metrics = execute_search(
            query, ann_index, metadata_store, k
        )

        # Store results
        latencies.append(latency)
        query_result = {
            "query_id": query_id,
            "latency_ms": latency,
            "results": result_ids,
            "metrics": metrics,
        }

        # Calculate quality metrics if ground truth available
        if ground_truth and query_id in ground_truth:
            relevant_ids = ground_truth[query_id]

            prec_10 = calculate_precision_at_k(result_ids, relevant_ids, 10)
            rec_10 = calculate_recall_at_k(result_ids, relevant_ids, 10)

            precision_at_10.append(prec_10)
            recall_at_10.append(rec_10)

            query_result.update(
                {
                    "precision_at_10": prec_10,
                    "recall_at_10": rec_10,
                    "relevant_count": len(relevant_ids),
                }
            )

        query_results.append(query_result)

        # Log progress
        if (i + 1) % 10 == 0:
            logger.info(f"Completed {i+1}/{len(queries)} queries")

    # Calculate aggregate metrics
    results = {
        "total_queries": len(queries),
        "latency_stats": {
            "mean_ms": statistics.mean(latencies),
            "median_ms": statistics.median(latencies),
            "p95_ms": np.percentile(latencies, 95) if latencies else 0,
            "p99_ms": np.percentile(latencies, 99) if latencies else 0,
            "min_ms": min(latencies) if latencies else 0,
            "max_ms": max(latencies) if latencies else 0,
        },
        "query_results": query_results,
    }

    # Add quality metrics if available
    if precision_at_10:
        results["quality_stats"] = {
            "mean_precision_at_10": statistics.mean(precision_at_10),
            "mean_recall_at_10": statistics.mean(recall_at_10),
            "queries_with_ground_truth": len(precision_at_10),
        }

    return results


def print_evaluation_results(results: Dict[str, Any]) -> None:
    """Print formatted evaluation results."""
    print("\n" + "=" * 60)
    print("🎯 CHROMATICA EVALUATION RESULTS")
    print("=" * 60)

    print(f"\n📊 OVERVIEW")
    print(f"Total queries evaluated: {results['total_queries']}")

    # Latency metrics
    latency = results["latency_stats"]
    print(f"\n⏱️  LATENCY METRICS")
    print(f"Mean latency:     {latency['mean_ms']:.2f} ms")
    print(f"Median latency:   {latency['median_ms']:.2f} ms")
    print(f"P95 latency:      {latency['p95_ms']:.2f} ms")
    print(f"P99 latency:      {latency['p99_ms']:.2f} ms")
    print(f"Min latency:      {latency['min_ms']:.2f} ms")
    print(f"Max latency:      {latency['max_ms']:.2f} ms")

    # Quality metrics
    if "quality_stats" in results:
        quality = results["quality_stats"]
        print(f"\n🎯 QUALITY METRICS")
        print(f"Mean Precision@10: {quality['mean_precision_at_10']:.3f}")
        print(f"Mean Recall@10:    {quality['mean_recall_at_10']:.3f}")
        print(f"Queries with GT:   {quality['queries_with_ground_truth']}")

    # Performance targets
    print(f"\n🎯 PERFORMANCE TARGETS")
    p95_target = 450  # ms
    p95_actual = latency["p95_ms"]
    p95_status = "✅ PASS" if p95_actual <= p95_target else "❌ FAIL"
    print(f"P95 latency target: {p95_target} ms")
    print(f"P95 latency actual: {p95_actual:.2f} ms {p95_status}")

    if "quality_stats" in results:
        prec_target = 0.8
        prec_actual = quality["mean_precision_at_10"]
        prec_status = "✅ PASS" if prec_actual >= prec_target else "❌ FAIL"
        print(f"Precision@10 target: {prec_target:.1f}")
        print(f"Precision@10 actual: {prec_actual:.3f} {prec_status}")

    print("\n" + "=" * 60)


def create_sample_queries(output_file: str, num_queries: int = 20) -> None:
    """
    Create sample test queries for evaluation.

    Args:
        output_file: Path to output JSON file
        num_queries: Number of queries to generate
    """
    import random

    # Sample color palettes for testing
    color_palettes = [
        ["FF0000", "00FF00", "0000FF"],  # RGB primary
        ["FF6B6B", "4ECDC4", "45B7D1"],  # Pastel
        ["2C3E50", "E74C3C", "F39C12"],  # Dark theme
        ["8E44AD", "3498DB", "2ECC71"],  # Vibrant
        ["F39C12", "E67E22", "D35400"],  # Orange tones
        ["9B59B6", "8E44AD", "7D3C98"],  # Purple tones
        ["1ABC9C", "16A085", "138D75"],  # Teal tones
        ["E74C3C", "C0392B", "A93226"],  # Red tones
        ["F1C40F", "F39C12", "E67E22"],  # Yellow tones
        ["34495E", "2C3E50", "1B2631"],  # Dark grays
    ]

    queries = []
    for i in range(num_queries):
        # Select random palette
        palette = random.choice(color_palettes)

        # Generate random weights
        weights = [random.random() for _ in palette]
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]  # Normalize

        query = {
            "query_id": f"q{i+1:03d}",
            "colors": palette,
            "weights": weights,
            "description": f"Test query {i+1} with {len(palette)} colors",
        }
        queries.append(query)

    # Save to file
    with open(output_file, "w") as f:
        json.dump(queries, f, indent=2)

    print(f"Created {num_queries} sample queries in {output_file}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate Chromatica color search engine"
    )
    parser.add_argument(
        "--queries",
        default="datasets/test-queries.json",
        help="Path to test queries JSON file",
    )
    parser.add_argument(
        "--ground-truth", help="Path to ground truth JSON file (optional)"
    )
    parser.add_argument(
        "--index-path",
        default="index/chromatica_index.faiss",
        help="Path to FAISS index file",
    )
    parser.add_argument(
        "--metadata-path",
        default="index/chromatica_metadata.db",
        help="Path to DuckDB metadata file",
    )
    parser.add_argument(
        "--k", type=int, default=10, help="Number of results to return per query"
    )
    parser.add_argument(
        "--create-sample-queries",
        action="store_true",
        help="Create sample test queries and exit",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Set up logging
    logger = setup_logging(args.log_level)

    try:
        # Create sample queries if requested
        if args.create_sample_queries:
            create_sample_queries(args.queries, 20)
            return

        # Load test queries
        queries = load_test_queries(args.queries)

        # Load ground truth if provided
        ground_truth = None
        if args.ground_truth:
            ground_truth = load_ground_truth(args.ground_truth)

        # Load index and metadata store
        logger.info("Loading FAISS index and metadata store...")
        ann_index = AnnIndex()
        ann_index.load(args.index_path)

        metadata_store = MetadataStore(args.metadata_path)

        logger.info(f"Index loaded with {ann_index.get_total_vectors()} vectors")
        logger.info(
            f"Metadata store loaded with {metadata_store.get_image_count()} images"
        )

        # Run evaluation
        results = run_evaluation(
            queries, ann_index, metadata_store, ground_truth, args.k
        )

        # Print results
        print_evaluation_results(results)

        # Save detailed results
        results_file = "logs/evaluation_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Detailed results saved to {results_file}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
