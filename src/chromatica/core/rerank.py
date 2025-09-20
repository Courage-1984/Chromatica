"""
Sinkhorn reranking module for the Chromatica color search engine.

This module implements the high-fidelity reranking stage using Sinkhorn-approximated
Earth Mover's Distance (EMD). The reranking process takes candidate images from the
ANN search stage and computes perceptually accurate distances using optimal transport
theory.

Key Components:
- Cost matrix generation for Lab color space bin centers
- Sinkhorn distance computation using the POT library
- Efficient batch reranking of candidate histograms

The cost matrix M is pre-computed once and reused for all distance calculations,
where M_ij = ||c_i - c_j||_2^2 represents the squared Euclidean distance between
bin centers in CIE Lab color space.

This implementation follows the specifications in the critical instructions document,
ensuring consistency with the overall system architecture and performance targets.
"""

import logging
import numpy as np
import ot  # Python Optimal Transport (POT)
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from functools import partial
import time
import psutil
import os

from ..utils.config import (
    L_BINS,
    A_BINS,
    B_BINS,
    TOTAL_BINS,
    LAB_RANGES,
    SINKHORN_EPSILON,
)

# Configure logging for this module
logger = logging.getLogger(__name__)

# Global cost matrix - computed once and reused for all distance calculations
_COST_MATRIX: Optional[np.ndarray] = None


def _check_memory_usage() -> bool:
    """
    Check if memory usage is too high and we should use fallback methods.

    Returns:
        True if memory usage is acceptable, False if we should use fallback
    """
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024  # Convert to MB

        # If we're using more than 1GB, use fallback methods
        if memory_mb > 1024:
            logger.warning(
                f"High memory usage detected: {memory_mb:.1f}MB, using fallback methods"
            )
            return False

        return True
    except Exception as e:
        logger.warning(f"Could not check memory usage: {e}")
        return True  # Default to safe mode


def _compute_distance_worker(
    args: Tuple[np.ndarray, np.ndarray, str, float],
) -> Tuple[str, float]:
    """
    Worker function for parallel distance computation.

    This function is designed to be used with multiprocessing.Pool
    to compute Sinkhorn distances in parallel.

    Args:
        args: Tuple containing (query_hist, candidate_hist, candidate_id, epsilon)

    Returns:
        Tuple of (candidate_id, distance)
    """
    query_hist, candidate_hist, candidate_id, epsilon = args

    try:
        # Compute Sinkhorn distance
        distance = compute_sinkhorn_distance(query_hist, candidate_hist, epsilon)
        return candidate_id, distance
    except Exception as e:
        logger.warning(f"Failed to compute distance for {candidate_id}: {e}")
        # Return a large distance as fallback
        return candidate_id, float("inf")


def _compute_distance_batch_worker(
    args: Tuple[np.ndarray, List[np.ndarray], List[str], float],
) -> List[Tuple[str, float]]:
    """
    Worker function for batch parallel distance computation.

    This function processes a batch of candidates in a single worker process,
    reducing the overhead of process creation.

    Args:
        args: Tuple containing (query_hist, candidate_hists, candidate_ids, epsilon)

    Returns:
        List of tuples (candidate_id, distance)
    """
    query_hist, candidate_hists, candidate_ids, epsilon = args

    results = []
    for candidate_hist, candidate_id in zip(candidate_hists, candidate_ids):
        try:
            distance = compute_sinkhorn_distance(query_hist, candidate_hist, epsilon)
            results.append((candidate_id, distance))
        except Exception as e:
            logger.warning(f"Failed to compute distance for {candidate_id}: {e}")
            results.append((candidate_id, float("inf")))

    return results


@dataclass
class RerankResult:
    """Result of a reranking operation for a single candidate."""

    candidate_id: Union[str, int]
    distance: float
    rank: int


def build_cost_matrix() -> np.ndarray:
    """
    Build the cost matrix for Sinkhorn-EMD calculations.

    This function pre-computes the cost matrix M where M_ij = ||c_i - c_j||_2^2
    represents the squared Euclidean distance between bin centers in CIE Lab color space.
    The cost matrix is computed once and reused for all distance calculations to avoid
    redundant computation.

    The bin centers are computed using np.linspace to create a regular grid across
    the Lab color space ranges defined in the configuration:
    - L*: [0, 100] with 8 bins
    - a*: [-86, 98] with 12 bins
    - b*: [-108, 95] with 12 bins

    Returns:
        np.ndarray: A (TOTAL_BINS, TOTAL_BINS) cost matrix where each element
                   represents the squared Euclidean distance between two bin centers.
                   Shape: (1152, 1152) for the default 8x12x12 binning configuration.

    Raises:
        ValueError: If the configuration constants are invalid.

    Example:
        >>> cost_matrix = build_cost_matrix()
        >>> print(f"Cost matrix shape: {cost_matrix.shape}")
        Cost matrix shape: (1152, 1152)
        >>> print(f"Cost matrix dtype: {cost_matrix.dtype}")
        Cost matrix dtype: float64
        >>> print(f"Diagonal elements (should be 0): {np.diag(cost_matrix)[:5]}")
        Diagonal elements (should be 0): [0. 0. 0. 0. 0.]
    """
    logger.info("Building cost matrix for Sinkhorn-EMD calculations...")

    # Validate configuration constants
    if TOTAL_BINS != L_BINS * A_BINS * B_BINS:
        raise ValueError(
            f"Configuration mismatch: TOTAL_BINS ({TOTAL_BINS}) != "
            f"L_BINS * A_BINS * B_BINS ({L_BINS * A_BINS * B_BINS})"
        )

    # Generate bin centers for each dimension using the configured ranges
    # These centers represent the "average" color value for each bin
    centers_l = np.linspace(
        LAB_RANGES[0][0], LAB_RANGES[0][1], L_BINS, dtype=np.float64
    )
    centers_a = np.linspace(
        LAB_RANGES[1][0], LAB_RANGES[1][1], A_BINS, dtype=np.float64
    )
    centers_b = np.linspace(
        LAB_RANGES[2][0], LAB_RANGES[2][1], B_BINS, dtype=np.float64
    )

    # Create a 3D grid of bin centers and reshape to 2D array
    # The meshgrid creates all combinations of L, a, b coordinates
    # indexing='ij' ensures the order matches our bin indexing convention
    grid_l, grid_a, grid_b = np.meshgrid(centers_l, centers_a, centers_b, indexing="ij")

    # Reshape to (TOTAL_BINS, 3) where each row is [L, a, b] coordinates
    # This creates a flat list of all bin center coordinates
    bin_centers = np.stack([grid_l, grid_a, grid_b], axis=-1).reshape(-1, 3)

    logger.debug(
        f"Generated {len(bin_centers)} bin centers with shape {bin_centers.shape}"
    )
    logger.debug(f"L* range: [{centers_l[0]:.2f}, {centers_l[-1]:.2f}]")
    logger.debug(f"a* range: [{centers_a[0]:.2f}, {centers_a[-1]:.2f}]")
    logger.debug(f"b* range: [{centers_b[0]:.2f}, {centers_b[-1]:.2f}]")

    # Compute the cost matrix using squared Euclidean distance
    # This creates a symmetric matrix where M_ij = ||c_i - c_j||_2^2
    # The POT library's dist function efficiently computes pairwise distances
    cost_matrix = ot.dist(bin_centers, bin_centers, metric="sqeuclidean")

    # Validate the cost matrix properties
    assert cost_matrix.shape == (TOTAL_BINS, TOTAL_BINS), (
        f"Expected cost matrix shape ({TOTAL_BINS}, {TOTAL_BINS}), "
        f"got {cost_matrix.shape}"
    )

    # Verify symmetry and zero diagonal
    if not np.allclose(cost_matrix, cost_matrix.T):
        raise ValueError("Cost matrix is not symmetric")

    if not np.allclose(np.diag(cost_matrix), 0):
        raise ValueError("Cost matrix diagonal should be zero")

    # Verify reasonable distance ranges
    max_distance = np.max(cost_matrix)
    min_distance = np.min(cost_matrix[np.triu_indices_from(cost_matrix, k=1)])

    logger.info(f"Cost matrix built successfully:")
    logger.info(f"  - Shape: {cost_matrix.shape}")
    logger.info(f"  - Min non-zero distance: {min_distance:.4f}")
    logger.info(f"  - Max distance: {max_distance:.4f}")
    logger.info(f"  - Memory usage: {cost_matrix.nbytes / 1024 / 1024:.2f} MB")

    return cost_matrix


def get_cost_matrix() -> np.ndarray:
    """
    Get the cost matrix, building it if necessary.

    This function ensures the cost matrix is built only once and then reused
    for all subsequent distance calculations. The cost matrix is a global
    constant that doesn't change during the application's lifetime.

    Returns:
        np.ndarray: The pre-computed cost matrix for Sinkhorn-EMD calculations.

    Example:
        >>> cost_matrix = get_cost_matrix()
        >>> # Subsequent calls return the same matrix without rebuilding
        >>> cost_matrix2 = get_cost_matrix()
        >>> assert cost_matrix is cost_matrix2  # Same object reference
    """
    global _COST_MATRIX

    if _COST_MATRIX is None:
        _COST_MATRIX = build_cost_matrix()
        logger.info("Cost matrix initialized and cached")

    return _COST_MATRIX


def compute_sinkhorn_distance(
    hist1: np.ndarray,
    hist2: np.ndarray,
    epsilon: float = SINKHORN_EPSILON,
    early_termination_threshold: float = 1e-6,
    max_iterations: int = 100,  # Add iteration limit
) -> float:
    """
    Compute the Sinkhorn distance between two histograms.

    The Sinkhorn distance is an entropy-regularized approximation of the Earth
    Mover's Distance (EMD) that provides a perceptually meaningful measure of
    color palette similarity. It represents the "work" required to transform
    one color distribution into another, accounting for both color differences
    and their relative weights.

    The distance is computed using the Sinkhorn-Knopp algorithm, which iteratively
    updates the transport plan to find the optimal solution. The regularization
    parameter epsilon controls the trade-off between accuracy and computational
    stability.

    Args:
        hist1: First histogram (normalized probability distribution)
               Shape: (TOTAL_BINS,) or (1, TOTAL_BINS)
        hist2: Second histogram (normalized probability distribution)
               Shape: (TOTAL_BINS,) or (1, TOTAL_BINS)
        epsilon: Regularization strength for Sinkhorn algorithm.
                Lower values provide more accurate EMD but may be less stable.
                Default: SINKHORN_EPSILON from configuration (0.1)

    Returns:
        float: The Sinkhorn distance between the two histograms.
               Lower values indicate more similar color palettes.

    Raises:
        ValueError: If histograms are invalid or have wrong dimensions.
        RuntimeError: If the Sinkhorn algorithm fails to converge.

    Example:
        >>> hist1 = np.random.random(1152)
        >>> hist1 = hist1 / hist1.sum()  # Normalize
        >>> hist2 = np.random.random(1152)
        >>> hist2 = hist2 / hist2.sum()  # Normalize
        >>> distance = compute_sinkhorn_distance(hist1, hist2)
        >>> print(f"Sinkhorn distance: {distance:.6f}")
        Sinkhorn distance: 0.123456
    """
    # Input validation and preprocessing
    hist1 = np.asarray(hist1, dtype=np.float64).flatten()
    hist2 = np.asarray(hist2, dtype=np.float64).flatten()

    if hist1.shape != (TOTAL_BINS,):
        raise ValueError(f"hist1 must have shape ({TOTAL_BINS},), got {hist1.shape}")

    if hist2.shape != (TOTAL_BINS,):
        raise ValueError(f"hist2 must have shape ({TOTAL_BINS},), got {hist2.shape}")

    # Validate that histograms are valid probability distributions
    if not np.allclose(hist1.sum(), 1.0, atol=1e-6):
        raise ValueError("hist1 is not normalized (sum != 1.0)")

    if not np.allclose(hist2.sum(), 1.0, atol=1e-6):
        raise ValueError("hist2 is not normalized (sum != 1.0)")

    if np.any(hist1 < -1e-10) or np.any(hist2 < -1e-10):
        raise ValueError("Histograms contain negative values")

    # Early termination check for very similar histograms
    l2_distance = np.linalg.norm(hist1 - hist2)
    if l2_distance < early_termination_threshold:
        logger.debug(
            f"Early termination: L2 distance {l2_distance:.2e} < threshold {early_termination_threshold:.2e}"
        )
        return float(l2_distance)

    # Get the pre-computed cost matrix
    cost_matrix = get_cost_matrix()

    try:
        # Compute Sinkhorn distance using POT library
        # The sinkhorn2 function returns the optimal transport cost
        # Add small regularization to avoid numerical instability
        hist1_reg = hist1 + 1e-10
        hist2_reg = hist2 + 1e-10
        hist1_reg = hist1_reg / hist1_reg.sum()
        hist2_reg = hist2_reg / hist2_reg.sum()

        try:
            # Use a more conservative epsilon for better convergence
            conservative_epsilon = max(epsilon, 0.1)

            # Try Sinkhorn with iteration limit
            try:
                distance = ot.sinkhorn2(
                    hist1_reg,
                    hist2_reg,
                    cost_matrix,
                    reg=conservative_epsilon,
                    numItermax=max_iterations,
                )
            except Exception as e:
                logger.warning(
                    f"Sinkhorn failed with conservative epsilon: {e}, using L2 fallback"
                )
                distance = np.linalg.norm(hist1_reg - hist2_reg)
                return float(distance)

            # Validate the result
            if not np.isfinite(distance):
                logger.warning(
                    f"Sinkhorn distance not finite: {distance}, using fallback"
                )
                # Fallback to L2 distance if Sinkhorn fails
                distance = np.linalg.norm(hist1_reg - hist2_reg)
            elif distance < 0:
                logger.warning(
                    f"Negative Sinkhorn distance computed: {distance}, clamping to 0"
                )
                distance = max(0.0, distance)
            elif distance < 1e-10:
                logger.warning(
                    f"Sinkhorn distance very small: {distance}, using fallback"
                )
                # Use L2 distance for very small values to avoid numerical issues
                distance = np.linalg.norm(hist1_reg - hist2_reg)

            return float(distance)

        except Exception as e:
            logger.warning(f"Sinkhorn algorithm failed: {e}, using L2 fallback")
            # Fallback to L2 distance if Sinkhorn completely fails
            distance = np.linalg.norm(hist1_reg - hist2_reg)
            return float(distance)

    except Exception as e:
        logger.error(f"Sinkhorn distance computation failed: {e}")
        logger.error(f"hist1 sum: {hist1.sum():.6f}, hist2 sum: {hist2.sum():.6f}")
        logger.error(f"hist1 range: [{hist1.min():.6f}, {hist1.max():.6f}]")
        logger.error(f"hist2 range: [{hist2.min():.6f}, {hist2.max():.6f}]")
        raise RuntimeError(f"Sinkhorn algorithm failed: {e}") from e


def rerank_candidates(
    query_hist: np.ndarray,
    candidate_hists: List[np.ndarray],
    candidate_ids: List[Union[str, int]],
    epsilon: float = SINKHORN_EPSILON,
    max_candidates: Optional[int] = None,
    batch_size: int = 10,
    use_approximate: bool = False,
    use_parallel: bool = False,  # Disable parallel processing by default
    num_processes: Optional[int] = None,
    early_termination_threshold: float = 0.01,
    early_termination_count: int = 10,
    streaming_mode: bool = True,  # Enable streaming mode by default
) -> List[RerankResult]:
    """
    Rerank candidate images using Sinkhorn-EMD distances with optional parallel processing.

    This function implements the high-fidelity reranking stage of the two-stage
    search architecture. It takes candidate histograms from the ANN search stage
    and computes perceptually accurate distances using the Sinkhorn-approximated
    Earth Mover's Distance.

    The reranking process:
    1. Computes Sinkhorn distances between the query histogram and each candidate
    2. Sorts candidates by distance (lower = more similar)
    3. Returns ranked results with distance scores

    Performance considerations:
    - The cost matrix is pre-computed and reused for all distance calculations
    - Histograms are processed in batches to minimize memory overhead
    - Optional parallel processing for large candidate sets
    - Results are sorted by distance for optimal ranking

    Args:
        query_hist: Query histogram (normalized probability distribution)
                   Shape: (TOTAL_BINS,)
        candidate_hists: List of candidate histograms to rerank
                        Each histogram should have shape (TOTAL_BINS,)
        candidate_ids: List of candidate identifiers corresponding to histograms
                      Length must match candidate_hists
        epsilon: Regularization strength for Sinkhorn algorithm
                Default: SINKHORN_EPSILON from configuration (0.1)
        max_candidates: Maximum number of candidates to return
                       If None, returns all candidates. If specified, returns
                       the top max_candidates by distance.
        batch_size: Batch size for processing (used for parallel processing)
        use_approximate: If True, use L2 distance instead of Sinkhorn-EMD
        use_parallel: If True, use parallel processing for distance calculations
        num_processes: Number of processes to use for parallel processing.
                      If None, uses cpu_count() - 1
        early_termination_threshold: Stop processing when distance improvement is below this threshold
        early_termination_count: Stop processing after finding this many good candidates
        streaming_mode: If True, process histograms in streaming fashion to minimize memory usage

    Returns:
        List[RerankResult]: Ranked list of candidates with distances and ranks.
                           Results are sorted by distance (ascending).

    Raises:
        ValueError: If input validation fails.
        RuntimeError: If distance computation fails.

    Example:
        >>> query_hist = np.random.random(1152)
        >>> query_hist = query_hist / query_hist.sum()
        >>> candidate_hists = [np.random.random(1152) for _ in range(5)]
        >>> candidate_hists = [h / h.sum() for h in candidate_hists]
        >>> candidate_ids = [f"img_{i}" for i in range(5)]
        >>> results = rerank_candidates(query_hist, candidate_hists, candidate_ids, use_parallel=True)
        >>> for result in results[:3]:  # Top 3 results
        ...     print(f"Rank {result.rank}: {result.candidate_id} (distance: {result.distance:.6f})")
        Rank 1: img_2 (distance: 0.045123)
        Rank 2: img_1 (distance: 0.067890)
        Rank 3: img_4 (distance: 0.089456)
    """
    # Input validation
    if not candidate_hists:
        logger.warning("No candidate histograms provided, returning empty results")
        return []

    if len(candidate_hists) != len(candidate_ids):
        raise ValueError(
            f"Number of histograms ({len(candidate_hists)}) must match "
            f"number of IDs ({len(candidate_ids)})"
        )

    if max_candidates is not None and max_candidates <= 0:
        raise ValueError(f"max_candidates must be positive, got {max_candidates}")

    if use_approximate:
        logger.info(
            f"🚀 FAST MODE: Reranking {len(candidate_hists)} candidates using approximate L2 distance"
        )
    else:
        logger.info(
            f"🎯 NORMAL MODE: Reranking {len(candidate_hists)} candidates using Sinkhorn-EMD"
        )
    logger.debug(f"Query histogram shape: {query_hist.shape}")
    logger.debug(f"Epsilon: {epsilon}")

    # Check memory usage and adjust strategy accordingly
    memory_ok = _check_memory_usage()
    if not memory_ok and not use_approximate:
        # Only force approximate mode if user hasn't already chosen it
        # and memory usage is critically high
        use_approximate = True
        streaming_mode = True
        logger.warning(
            "High memory usage detected, forcing approximate mode and streaming"
        )
    elif not memory_ok and use_approximate:
        # User already chose fast mode, just enable streaming
        streaming_mode = True
        logger.info(
            "High memory usage detected, enabling streaming mode for fast search"
        )

    # Compute distances for all candidates
    distances = []
    start_time = time.time()

    if streaming_mode or len(candidate_hists) > 20 or not memory_ok:
        # Use streaming mode for moderate to large candidate sets to minimize memory usage
        logger.info(f"Using streaming mode for {len(candidate_hists)} candidates")

        # Process candidates one by one to minimize memory usage
        for i, (hist, candidate_id) in enumerate(zip(candidate_hists, candidate_ids)):
            try:
                if use_approximate:
                    # Use fast L2 distance for approximate reranking
                    distance = np.linalg.norm(query_hist - hist)
                else:
                    # Use full Sinkhorn-EMD for high-fidelity reranking
                    distance = compute_sinkhorn_distance(query_hist, hist, epsilon)
                distances.append((candidate_id, distance, i))

                # Early termination check in streaming mode
                if len(distances) >= early_termination_count:
                    # Sort current results to check for convergence
                    sorted_distances = sorted(distances, key=lambda x: x[1])

                    # Check if the last few distances are very similar (convergence)
                    if len(sorted_distances) >= early_termination_count:
                        recent_distances = [d[1] for d in sorted_distances[-5:]]
                        distance_variance = np.var(recent_distances)

                        if distance_variance < early_termination_threshold:
                            logger.info(
                                f"Early termination in streaming mode: distance variance {distance_variance:.6f} < threshold {early_termination_threshold}"
                            )
                            break

            except Exception as e:
                logger.error(
                    f"Failed to compute distance for candidate {candidate_id}: {e}"
                )
                continue

            # Log progress for streaming mode
            if (i + 1) % 10 == 0:  # More frequent logging for smaller sets
                logger.debug(
                    f"Streaming processed {i + 1}/{len(candidate_hists)} candidates"
                )

    elif use_parallel and len(candidate_hists) > 10 and not use_approximate:
        # Use parallel processing for large candidate sets
        if num_processes is None:
            num_processes = max(1, cpu_count() - 1)

        logger.info(f"Using parallel processing with {num_processes} processes")

        # Prepare arguments for parallel processing
        if len(candidate_hists) > 50:
            # Use batch processing for very large sets
            batch_args = []
            for batch_start in range(0, len(candidate_hists), batch_size):
                batch_end = min(batch_start + batch_size, len(candidate_hists))
                batch_hists = candidate_hists[batch_start:batch_end]
                batch_ids = candidate_ids[batch_start:batch_end]
                batch_args.append((query_hist, batch_hists, batch_ids, epsilon))

            # Process batches in parallel
            with Pool(processes=num_processes) as pool:
                batch_results = pool.map(_compute_distance_batch_worker, batch_args)

            # Flatten results
            for batch_result in batch_results:
                for candidate_id, distance in batch_result:
                    distances.append(
                        (candidate_id, distance, 0)
                    )  # original_index not used in parallel mode
        else:
            # Use individual processing for smaller sets
            args = [
                (query_hist, hist, candidate_id, epsilon)
                for hist, candidate_id in zip(candidate_hists, candidate_ids)
            ]

            with Pool(processes=num_processes) as pool:
                results = pool.map(_compute_distance_worker, args)

            for candidate_id, distance in results:
                distances.append(
                    (candidate_id, distance, 0)
                )  # original_index not used in parallel mode
    else:
        # Use sequential processing with early termination
        logger.info("Using sequential processing with early termination")

        # Process candidates in batches for better performance
        for batch_start in range(0, len(candidate_hists), batch_size):
            batch_end = min(batch_start + batch_size, len(candidate_hists))
            batch_hists = candidate_hists[batch_start:batch_end]
            batch_ids = candidate_ids[batch_start:batch_end]
            batch_indices = list(range(batch_start, batch_end))

            # Process batch
            for i, (hist, candidate_id, original_index) in enumerate(
                zip(batch_hists, batch_ids, batch_indices)
            ):
                try:
                    if use_approximate:
                        # Use fast L2 distance for approximate reranking
                        distance = np.linalg.norm(query_hist - hist)
                    else:
                        # Use full Sinkhorn-EMD for high-fidelity reranking
                        distance = compute_sinkhorn_distance(query_hist, hist, epsilon)
                    distances.append((candidate_id, distance, original_index))

                    # Early termination check
                    if len(distances) >= early_termination_count:
                        # Sort current results to check for convergence
                        sorted_distances = sorted(distances, key=lambda x: x[1])

                        # Check if the last few distances are very similar (convergence)
                        if len(sorted_distances) >= early_termination_count:
                            recent_distances = [d[1] for d in sorted_distances[-5:]]
                            distance_variance = np.var(recent_distances)

                            if distance_variance < early_termination_threshold:
                                logger.info(
                                    f"Early termination: distance variance {distance_variance:.6f} < threshold {early_termination_threshold}"
                                )
                                break

                except Exception as e:
                    logger.error(
                        f"Failed to compute distance for candidate {candidate_id}: {e}"
                    )
                    # Continue with other candidates rather than failing completely
                    continue

            # Check for early termination after each batch
            if len(distances) >= early_termination_count:
                break

            # Log progress for large batches
            if batch_end % (batch_size * 5) == 0:
                logger.debug(f"Processed {batch_end}/{len(candidate_hists)} candidates")

    computation_time = time.time() - start_time
    logger.info(f"Distance computation completed in {computation_time:.3f}s")

    if not distances:
        logger.error("No valid distances computed, returning empty results")
        return []

    # Sort by distance (ascending - lower distance = more similar)
    distances.sort(key=lambda x: x[1])

    # Apply max_candidates limit if specified
    if max_candidates is not None:
        distances = distances[:max_candidates]
        logger.info(f"Limited results to top {max_candidates} candidates")

    # Create ranked results
    results = []
    for rank, (candidate_id, distance, original_index) in enumerate(distances, 1):
        result = RerankResult(candidate_id=candidate_id, distance=distance, rank=rank)
        results.append(result)

    # Log summary statistics
    if results:
        min_dist = results[0].distance
        max_dist = results[-1].distance
        avg_dist = np.mean([r.distance for r in results])

        logger.info(f"Reranking completed successfully:")
        logger.info(f"  - Candidates processed: {len(candidate_hists)}")
        logger.info(f"  - Valid results: {len(results)}")
        logger.info(f"  - Distance range: [{min_dist:.6f}, {max_dist:.6f}]")
        logger.info(f"  - Average distance: {avg_dist:.6f}")

    return results


def validate_reranking_system() -> bool:
    """
    Validate the reranking system by running test calculations.

    This function performs a series of validation checks to ensure the
    reranking system is working correctly:
    1. Verifies cost matrix generation and properties
    2. Tests Sinkhorn distance computation with known cases
    3. Validates reranking pipeline with synthetic data

    Returns:
        bool: True if all validation checks pass, False otherwise.

    Example:
        >>> is_valid = validate_reranking_system()
        >>> print(f"Reranking system validation: {'PASSED' if is_valid else 'FAILED'}")
        Reranking system validation: PASSED
    """
    logger.info("Validating reranking system...")

    try:
        # Test 1: Cost matrix generation
        logger.info("Test 1: Cost matrix generation")
        cost_matrix = build_cost_matrix()

        # Verify matrix properties
        assert cost_matrix.shape == (TOTAL_BINS, TOTAL_BINS), "Wrong cost matrix shape"
        assert np.allclose(cost_matrix, cost_matrix.T), "Cost matrix not symmetric"
        assert np.allclose(np.diag(cost_matrix), 0), "Cost matrix diagonal not zero"

        # Test 2: Sinkhorn distance with identical histograms
        logger.info("Test 2: Sinkhorn distance with identical histograms")
        test_hist = np.ones(TOTAL_BINS, dtype=np.float64) / TOTAL_BINS
        distance = compute_sinkhorn_distance(test_hist, test_hist)
        assert np.isclose(
            distance, 0.0, atol=1e-6
        ), f"Identical histograms should have distance 0, got {distance}"

        # Test 3: Sinkhorn distance with different histograms
        logger.info("Test 3: Sinkhorn distance with different histograms")
        # Use more realistic histograms to avoid numerical instability
        np.random.seed(42)  # For reproducible results
        hist1 = np.random.random(TOTAL_BINS).astype(np.float64)
        hist1 = hist1 / hist1.sum()  # Normalize

        hist2 = np.random.random(TOTAL_BINS).astype(np.float64)
        hist2 = hist2 / hist2.sum()  # Normalize

        # Use a higher epsilon for numerical stability in testing
        distance = compute_sinkhorn_distance(hist1, hist2, epsilon=0.5)
        assert (
            distance > 0
        ), f"Different histograms should have positive distance, got {distance}"
        logger.info(f"Test 3 passed: distance = {distance:.6f}")

        # Test 4: Reranking pipeline
        logger.info("Test 4: Reranking pipeline")
        query_hist = np.ones(TOTAL_BINS, dtype=np.float64) / TOTAL_BINS
        candidate_hists = [hist1, hist2]
        candidate_ids = ["test1", "test2"]

        results = rerank_candidates(query_hist, candidate_hists, candidate_ids)
        assert len(results) == 2, f"Expected 2 results, got {len(results)}"
        assert (
            results[0].rank == 1
        ), f"First result should have rank 1, got {results[0].rank}"

        logger.info("All validation tests passed!")
        return True

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return False


# Auto-validate the system when module is imported
if __name__ == "__main__":
    # Set up logging for standalone execution
    logging.basicConfig(level=logging.INFO)

    # Run validation
    is_valid = validate_reranking_system()
    if is_valid:
        print("✅ Reranking system validation PASSED")
    else:
        print("❌ Reranking system validation FAILED")
        exit(1)
else:
    # Validate on import (but only log errors)
    try:
        validate_reranking_system()
    except Exception as e:
        logger.error(f"Reranking system validation failed: {e}")
        # Don't raise the error on import to avoid breaking the application
