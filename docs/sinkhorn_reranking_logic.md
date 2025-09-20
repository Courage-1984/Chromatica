# Sinkhorn Reranking Logic Documentation

## Overview

The Sinkhorn reranking module (`src/chromatica/core/rerank.py`) implements the high-fidelity reranking stage of the Chromatica color search engine. This module provides perceptually accurate distance calculations using Sinkhorn-approximated Earth Mover's Distance (EMD) to rerank candidate images from the ANN search stage.

## Key Features

- **Pre-computed Cost Matrix**: Efficient distance calculations using a pre-computed cost matrix based on Lab color space bin centers
- **Sinkhorn-EMD Distance**: Perceptually meaningful color palette similarity using optimal transport theory
- **Robust Error Handling**: Comprehensive fallback mechanisms for numerical stability
- **Batch Processing**: Efficient reranking of multiple candidates with performance monitoring
- **Validation System**: Built-in validation functions to ensure system reliability

## Architecture

### Core Components

1. **Cost Matrix Generation** (`build_cost_matrix`)

   - Pre-computes squared Euclidean distances between Lab color space bin centers
   - Creates a symmetric (1152×1152) matrix for efficient distance calculations
   - Computed once and cached for reuse across all distance calculations

2. **Sinkhorn Distance Computation** (`compute_sinkhorn_distance`)

   - Implements entropy-regularized Earth Mover's Distance using the POT library
   - Provides perceptually accurate color palette similarity measurements
   - Includes numerical stability safeguards and fallback mechanisms

3. **Candidate Reranking** (`rerank_candidates`)
   - Processes batches of candidate histograms from ANN search results
   - Computes Sinkhorn distances and ranks candidates by similarity
   - Returns structured results with distances and rankings

## Algorithm Details

### Cost Matrix Construction

The cost matrix M is constructed as follows:

```python
# Generate bin centers for each Lab dimension
centers_l = np.linspace(0, 100, 8)      # L* dimension
centers_a = np.linspace(-86, 98, 12)    # a* dimension
centers_b = np.linspace(-108, 95, 12)   # b* dimension

# Create 3D grid of all bin center combinations
bin_centers = np.meshgrid(centers_l, centers_a, centers_b, indexing='ij')
bin_centers = np.stack(bin_centers, axis=-1).reshape(-1, 3)

# Compute squared Euclidean distance matrix
cost_matrix = ot.dist(bin_centers, bin_centers, metric='sqeuclidean')
```

**Mathematical Properties:**

- **Symmetry**: M[i,j] = M[j,i] for all i,j
- **Zero Diagonal**: M[i,i] = 0 for all i
- **Non-negative**: M[i,j] ≥ 0 for all i,j
- **Triangle Inequality**: M[i,j] ≤ M[i,k] + M[k,j] for all i,j,k

### Sinkhorn Distance Calculation

The Sinkhorn distance is computed using the entropy-regularized optimal transport:

```
W_ε(h_q, h_c) = min_{P ∈ U(h_q, h_c)} ⟨P, M⟩ - ε H(P)
```

Where:

- `h_q` is the query histogram (normalized probability distribution)
- `h_c` is the candidate histogram (normalized probability distribution)
- `M` is the pre-computed cost matrix
- `P` is the optimal transport plan
- `ε` is the regularization parameter (default: 1.0)
- `H(P)` is the entropy of the transport plan

### Numerical Stability

The implementation includes several safeguards for numerical stability:

1. **Regularization**: Small values (1e-10) added to histograms to avoid zero probabilities
2. **Fallback Mechanism**: L2 distance used when Sinkhorn algorithm fails
3. **Validation**: Comprehensive input validation and result verification
4. **Error Handling**: Graceful handling of convergence failures

## Configuration Parameters

### Key Constants

```python
# From src/chromatica/utils/config.py
SINKHORN_EPSILON = 1.0        # Regularization strength
TOTAL_BINS = 1152             # Histogram dimensions (8×12×12)
LAB_RANGES = [                # Lab color space ranges
    [0.0, 100.0],            # L* range
    [-86.0, 98.0],           # a* range
    [-108.0, 95.0]           # b* range
]
```

### Performance Tuning

- **Epsilon (ε)**: Controls regularization strength

  - Lower values: More accurate EMD but less stable
  - Higher values: More stable but less accurate
  - Default: 1.0 (balanced for production use)

- **Batch Size**: Number of candidates processed simultaneously
  - Larger batches: Better memory efficiency
  - Smaller batches: Lower memory usage
  - Default: Process all candidates in single batch

## Usage Examples

### Basic Reranking

```python
from src.chromatica.core.rerank import rerank_candidates
import numpy as np

# Create normalized histograms
query_hist = np.random.random(1152)
query_hist = query_hist / query_hist.sum()

candidate_hists = [np.random.random(1152) for _ in range(10)]
candidate_hists = [h / h.sum() for h in candidate_hists]

candidate_ids = [f"img_{i}" for i in range(10)]

# Rerank candidates
results = rerank_candidates(query_hist, candidate_hists, candidate_ids)

# Display top 3 results
for result in results[:3]:
    print(f"Rank {result.rank}: {result.candidate_id} (distance: {result.distance:.6f})")
```

### Distance Computation

```python
from src.chromatica.core.rerank import compute_sinkhorn_distance

# Compute distance between two histograms
hist1 = np.random.random(1152)
hist1 = hist1 / hist1.sum()

hist2 = np.random.random(1152)
hist2 = hist2 / hist2.sum()

distance = compute_sinkhorn_distance(hist1, hist2)
print(f"Sinkhorn distance: {distance:.6f}")
```

### Cost Matrix Access

```python
from src.chromatica.core.rerank import get_cost_matrix

# Get the pre-computed cost matrix
cost_matrix = get_cost_matrix()
print(f"Cost matrix shape: {cost_matrix.shape}")
print(f"Memory usage: {cost_matrix.nbytes / 1024 / 1024:.2f} MB")
```

## Performance Characteristics

### Computational Complexity

- **Cost Matrix**: O(TOTAL_BINS²) - computed once at startup
- **Sinkhorn Distance**: O(TOTAL_BINS² × iterations) per pair
- **Reranking**: O(K × TOTAL_BINS² × iterations) for K candidates

### Memory Usage

- **Cost Matrix**: ~10.2 MB (1152×1152 × 8 bytes)
- **Per Distance Calculation**: ~10.2 MB (reuses cost matrix)
- **Batch Reranking**: ~10.2 MB + O(K × 1152 × 4 bytes) for K candidates

### Performance Targets

- **Distance Calculation**: <10ms per pair (typical)
- **Reranking 200 Candidates**: <300ms (target from critical instructions)
- **Memory Overhead**: <15MB total (cost matrix + working memory)

## Error Handling

### Input Validation

- **Histogram Shape**: Must be (1152,) for all histograms
- **Normalization**: Histograms must sum to 1.0 (probability distributions)
- **Non-negative Values**: Histograms cannot contain negative values
- **Candidate Count**: Number of histograms must match number of IDs

### Numerical Stability

- **Convergence Failures**: Automatic fallback to L2 distance
- **Overflow/Underflow**: Regularization prevents numerical issues
- **Invalid Results**: Validation and clamping of distance values
- **Error Recovery**: Continue processing other candidates on individual failures

### Logging and Monitoring

- **Debug Logging**: Detailed progress information for large batches
- **Warning Messages**: Numerical issues and fallback usage
- **Error Logging**: Complete failure information with context
- **Performance Metrics**: Timing and memory usage statistics

## Integration with Search Pipeline

### Two-Stage Architecture

1. **ANN Search Stage**: Fast approximate search using FAISS HNSW index

   - Returns top K candidates (default: 200)
   - Uses Hellinger-transformed histograms for L2 compatibility

2. **Reranking Stage**: High-fidelity distance calculation using Sinkhorn-EMD
   - Processes raw histograms from ANN candidates
   - Computes perceptually accurate distances
   - Returns final ranked results

### Data Flow

```
Query Histogram → ANN Search → Candidate Histograms → Sinkhorn Reranking → Final Results
     ↓              ↓              ↓                    ↓                ↓
  Raw Histogram  Hellinger     Raw Histograms      Sinkhorn-EMD    Ranked Results
  (1152 dims)    Transform     (1152 dims each)    Distances       (with scores)
```

## Validation and Testing

### Built-in Validation

The module includes comprehensive validation functions:

```python
from src.chromatica.core.rerank import validate_reranking_system

# Run complete validation suite
is_valid = validate_reranking_system()
print(f"Validation result: {is_valid}")
```

### Validation Tests

1. **Cost Matrix Properties**: Symmetry, zero diagonal, correct shape
2. **Identical Histograms**: Distance should be 0.0
3. **Different Histograms**: Distance should be positive
4. **Reranking Pipeline**: End-to-end functionality test

### Test Datasets

Use the project's test datasets for validation:

- **test-dataset-20**: Quick validation (20 images)
- **test-dataset-50**: Small-scale testing (50 images)
- **test-dataset-200**: Medium-scale testing (200 images)
- **test-dataset-5000**: Production-scale testing (5,000 images)

## Troubleshooting

### Common Issues

1. **Numerical Warnings**

   - **Cause**: Sinkhorn algorithm convergence issues
   - **Solution**: Implementation includes automatic fallback to L2 distance
   - **Prevention**: Use appropriate epsilon values and input validation

2. **Memory Issues**

   - **Cause**: Large cost matrix (10.2 MB) + batch processing
   - **Solution**: Process candidates in smaller batches
   - **Prevention**: Monitor memory usage and adjust batch sizes

3. **Performance Issues**
   - **Cause**: Large number of candidates or numerical instability
   - **Solution**: Increase epsilon for faster convergence
   - **Prevention**: Use appropriate candidate limits and regularization

### Debug Information

Enable debug logging for detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now run reranking operations to see detailed logs
results = rerank_candidates(query_hist, candidate_hists, candidate_ids)
```

## Future Enhancements

### Potential Improvements

1. **Adaptive Epsilon**: Dynamic regularization based on histogram characteristics
2. **Parallel Processing**: Multi-threaded distance calculations for large batches
3. **Memory Optimization**: Sparse cost matrix for reduced memory usage
4. **Alternative Metrics**: Support for other optimal transport variants

### Research Directions

1. **Faster Algorithms**: Investigation of approximate EMD methods
2. **Learned Metrics**: Machine learning-based distance functions
3. **Hierarchical Processing**: Multi-scale distance calculations
4. **GPU Acceleration**: CUDA-based implementations for large-scale processing

## References

1. **Cuturi, M. (2013)**: "Sinkhorn Distances: Lightspeed Computation of Optimal Transport"
2. **Rubner, Y. et al. (2000)**: "The Earth Mover's Distance as a Metric for Image Retrieval"
3. **POT Library**: Python Optimal Transport - https://pythonot.github.io/
4. **Critical Instructions**: Chromatica project specifications in `docs/.cursor/critical_instructions.md`

## API Reference

### Functions

#### `build_cost_matrix() -> np.ndarray`

Builds the cost matrix for Sinkhorn-EMD calculations.

**Returns:**

- `np.ndarray`: (1152, 1152) cost matrix with squared Euclidean distances

#### `get_cost_matrix() -> np.ndarray`

Gets the cached cost matrix, building it if necessary.

**Returns:**

- `np.ndarray`: The pre-computed cost matrix

#### `compute_sinkhorn_distance(hist1, hist2, epsilon=1.0) -> float`

Computes Sinkhorn distance between two histograms.

**Parameters:**

- `hist1` (np.ndarray): First histogram (1152,)
- `hist2` (np.ndarray): Second histogram (1152,)
- `epsilon` (float): Regularization strength (default: 1.0)

**Returns:**

- `float`: Sinkhorn distance between histograms

#### `rerank_candidates(query_hist, candidate_hists, candidate_ids, epsilon=1.0, max_candidates=None) -> List[RerankResult]`

Reranks candidate images using Sinkhorn-EMD distances.

**Parameters:**

- `query_hist` (np.ndarray): Query histogram (1152,)
- `candidate_hists` (List[np.ndarray]): List of candidate histograms
- `candidate_ids` (List[Union[str, int]]): List of candidate identifiers
- `epsilon` (float): Regularization strength (default: 1.0)
- `max_candidates` (Optional[int]): Maximum results to return

**Returns:**

- `List[RerankResult]`: Ranked list of candidates with distances

#### `validate_reranking_system() -> bool`

Validates the reranking system with comprehensive tests.

**Returns:**

- `bool`: True if all validation tests pass

### Classes

#### `RerankResult`

Result of a reranking operation for a single candidate.

**Attributes:**

- `candidate_id` (Union[str, int]): Candidate identifier
- `distance` (float): Sinkhorn distance from query
- `rank` (int): Final ranking (1-based)

## Conclusion

The Sinkhorn reranking module provides a robust, efficient, and perceptually accurate solution for the high-fidelity reranking stage of the Chromatica color search engine. The implementation follows the specifications in the critical instructions document and includes comprehensive error handling, validation, and performance monitoring.

The module is production-ready and integrates seamlessly with the existing histogram generation and ANN search components, completing the two-stage search architecture for high-quality color-based image retrieval.
