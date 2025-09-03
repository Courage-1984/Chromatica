# Two-Stage Search Logic

## Chromatica Color Search Engine

---

## Overview

The Two-Stage Search Logic implements the core search algorithm that combines fast approximate nearest neighbor (ANN) search with high-fidelity reranking. This architecture provides the optimal balance between search speed and result quality, making it suitable for production-scale color image search.

### Key Features

- **Stage 1 - FAISS ANN**: Fast candidate retrieval using HNSW index
- **Stage 2 - Sinkhorn Reranking**: High-quality distance computation using optimal transport
- **Configurable Parameters**: Adjustable search depth and reranking limits
- **Performance Monitoring**: Comprehensive timing and accuracy metrics
- **Memory Efficiency**: Optimized for large-scale datasets

---

## Architecture

### System Design

```
┌─────────────────────────────────────────────────────────────┐
│                    Query Histogram                          │
├─────────────────────────────────────────────────────────────┤
│                Stage 1: FAISS ANN Search                   │
│              (Fast Candidate Retrieval)                    │
├─────────────────────────────────────────────────────────────┤
│                Stage 2: Sinkhorn Reranking                 │
│              (High-Fidelity Distance)                      │
├─────────────────────────────────────────────────────────────┤
│                    Final Results                            │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Input**: Normalized query histogram (1152 dimensions)
2. **Stage 1**: FAISS HNSW search returns top K candidates
3. **Stage 2**: Sinkhorn-EMD reranking of candidates
4. **Output**: Ranked list of similar images with distances

---

## Implementation

### Core Search Function

```python
from chromatica.search import find_similar
from chromatica.indexing.store import AnnIndex, MetadataStore
from chromatica.core.rerank import SinkhornReranker

def perform_two_stage_search(
    query_histogram: np.ndarray,
    index: AnnIndex,
    store: MetadataStore,
    reranker: SinkhornReranker,
    k: int = 50,
    rerank_k: int = 200
) -> List[SearchResult]:
    """Perform complete two-stage search."""

    # Stage 1: FAISS ANN search
    distances, indices = index.search(query_histogram, k=rerank_k)

    # Get candidate metadata and histograms
    candidate_ids = [f"img_{idx}" for idx in indices[0]]
    candidate_histograms = store.get_histograms(candidate_ids)

    # Stage 2: Sinkhorn reranking
    reranked_results = reranker.rerank_candidates(
        query_histogram=query_histogram,
        candidate_histograms=candidate_histograms,
        candidate_ids=candidate_ids
    )

    # Convert to search results
    search_results = []
    for result in reranked_results[:k]:
        search_results.append(SearchResult(
            image_id=result.image_id,
            file_path=store.get_image_path(result.image_id),
            distance=result.distance,
            rank=result.final_rank
        ))

    return search_results
```

### Search Result Structure

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class SearchResult:
    """Result of a two-stage search operation."""

    image_id: str
    file_path: str
    distance: float
    rank: int
    ann_score: Optional[float] = None
    rerank_score: Optional[float] = None

    def __post_init__(self):
        """Validate result data."""
        if self.distance < 0:
            raise ValueError("Distance must be non-negative")
        if self.rank < 1:
            raise ValueError("Rank must be positive")
```

---

## Usage Examples

### Basic Search

```python
from chromatica.search import find_similar
import numpy as np

# Create query histogram
query_hist = np.random.random(1152)
query_hist = query_hist / query_hist.sum()

# Perform search
results = find_similar(
    query_histogram=query_hist,
    index=index,
    store=store,
    k=50,
    max_rerank=200
)

# Display results
print(f"Found {len(results)} similar images:")
for i, result in enumerate(results[:10]):
    print(f"{i+1}. {result.image_id}: {result.distance:.6f}")
```

### Advanced Search with Monitoring

```python
def search_with_monitoring(
    query_histogram: np.ndarray,
    index: AnnIndex,
    store: MetadataStore,
    reranker: SinkhornReranker,
    k: int = 50,
    rerank_k: int = 200
) -> Tuple[List[SearchResult], Dict[str, float]]:
    """Perform search with comprehensive monitoring."""

    import time

    # Stage 1 timing
    stage1_start = time.time()
    distances, indices = index.search(query_histogram, k=rerank_k)
    stage1_time = time.time() - stage1_start

    # Get candidates
    candidate_ids = [f"img_{idx}" for idx in indices[0]]
    candidate_histograms = store.get_histograms(candidate_ids)

    # Stage 2 timing
    stage2_start = time.time()
    reranked_results = reranker.rerank_candidates(
        query_histogram=query_histogram,
        candidate_histograms=candidate_histograms,
        candidate_ids=candidate_ids
    )
    stage2_time = time.time() - stage2_start

    # Create results
    search_results = []
    for result in reranked_results[:k]:
        search_results.append(SearchResult(
            image_id=result.image_id,
            file_path=store.get_image_path(result.image_id),
            distance=result.distance,
            rank=result.final_rank
        ))

    # Performance metrics
    metrics = {
        'stage1_time': stage1_time,
        'stage2_time': stage2_time,
        'total_time': stage1_time + stage2_time,
        'candidates_retrieved': rerank_k,
        'candidates_reranked': len(reranked_results),
        'final_results': len(search_results)
    }

    return search_results, metrics
```

### Batch Search

```python
def batch_search(
    query_histograms: List[np.ndarray],
    index: AnnIndex,
    store: MetadataStore,
    reranker: SinkhornReranker,
    k: int = 50,
    rerank_k: int = 200
) -> List[List[SearchResult]]:
    """Perform batch search for multiple queries."""

    all_results = []

    for i, query_hist in enumerate(query_histograms):
        print(f"Processing query {i+1}/{len(query_histograms)}")

        try:
            results = find_similar(
                query_histogram=query_hist,
                index=index,
                store=store,
                k=k,
                max_rerank=rerank_k
            )
            all_results.append(results)

        except Exception as e:
            print(f"Query {i+1} failed: {e}")
            all_results.append([])

    return all_results
```

---

## Performance Optimization

### Parameter Tuning

```python
def optimize_search_parameters(
    query_histogram: np.ndarray,
    index: AnnIndex,
    store: MetadataStore,
    reranker: SinkhornReranker,
    target_time_ms: float = 500
) -> Dict[str, int]:
    """Find optimal search parameters for time constraints."""

    # Test different rerank_k values
    rerank_k_values = [50, 100, 200, 300, 500]
    results = {}

    for rerank_k in rerank_k_values:
        start_time = time.time()

        try:
            search_results = find_similar(
                query_histogram=query_histogram,
                index=index,
                store=store,
                k=50,
                max_rerank=rerank_k
            )

            total_time = (time.time() - start_time) * 1000  # Convert to ms
            results[rerank_k] = {
                'time_ms': total_time,
                'results_count': len(search_results),
                'within_target': total_time <= target_time_ms
            }

        except Exception as e:
            results[rerank_k] = {'error': str(e)}

    # Find best rerank_k
    valid_results = {k: v for k, v in results.items()
                    if 'error' not in v and v['within_target']}

    if valid_results:
        best_rerank_k = max(valid_results.keys())
        return {'rerank_k': best_rerank_k, 'k': 50}
    else:
        # Use fastest option
        fastest = min(results.keys(),
                     key=lambda k: results[k].get('time_ms', float('inf')))
        return {'rerank_k': fastest, 'k': 50}
```

### Memory Management

```python
def memory_efficient_search(
    query_histogram: np.ndarray,
    index: AnnIndex,
    store: MetadataStore,
    reranker: SinkhornReranker,
    k: int = 50,
    max_memory_gb: float = 4.0
) -> List[SearchResult]:
    """Perform search with memory constraints."""

    # Calculate optimal rerank_k based on memory
    memory_per_candidate_mb = 0.005  # 5KB per histogram
    available_memory_mb = max_memory_gb * 1024
    reserved_memory_mb = 1024  # 1GB for system

    max_candidates = int((available_memory_mb - reserved_memory_mb) /
                        memory_per_candidate_mb)

    # Clamp to reasonable range
    rerank_k = min(max_candidates, 500)
    rerank_k = max(rerank_k, 100)

    print(f"Using rerank_k={rerank_k} for memory constraint")

    return find_similar(
        query_histogram=query_histogram,
        index=index,
        store=store,
        k=k,
        max_rerank=rerank_k
    )
```

---

## Testing and Validation

### Search Quality Tests

```python
def test_search_quality(
    query_histogram: np.ndarray,
    index: AnnIndex,
    store: MetadataStore,
    reranker: SinkhornReranker
) -> Dict[str, float]:
    """Test search result quality."""

    # Perform search
    results = find_similar(
        query_histogram=query_histogram,
        index=index,
        store=store,
        k=50,
        max_rerank=200
    )

    if not results:
        return {'error': 'No results returned'}

    # Analyze distance distribution
    distances = [r.distance for r in results]

    quality_metrics = {
        'total_results': len(results),
        'min_distance': min(distances),
        'max_distance': max(distances),
        'mean_distance': np.mean(distances),
        'std_distance': np.std(distances),
        'distance_range': max(distances) - min(distances)
    }

    # Check ranking consistency
    ranks = [r.rank for r in results]
    if ranks == list(range(1, len(results) + 1)):
        quality_metrics['ranking_consistent'] = True
    else:
        quality_metrics['ranking_consistent'] = False

    return quality_metrics
```

### Performance Benchmarking

```python
def benchmark_search_performance(
    query_histograms: List[np.ndarray],
    index: AnnIndex,
    store: MetadataStore,
    reranker: SinkhornReranker,
    rerank_k_values: List[int] = [100, 200, 300]
) -> Dict[int, Dict[str, float]]:
    """Benchmark search performance with different parameters."""

    benchmark_results = {}

    for rerank_k in rerank_k_values:
        print(f"Benchmarking rerank_k={rerank_k}")

        stage1_times = []
        stage2_times = []
        total_times = []

        for query_hist in query_histograms:
            start_time = time.time()

            # Stage 1
            stage1_start = time.time()
            distances, indices = index.search(query_hist, k=rerank_k)
            stage1_time = time.time() - stage1_start

            # Stage 2
            candidate_ids = [f"img_{idx}" for idx in indices[0]]
            candidate_histograms = store.get_histograms(candidate_ids)

            stage2_start = time.time()
            reranked_results = reranker.rerank_candidates(
                query_histogram=query_hist,
                candidate_histograms=candidate_histograms,
                candidate_ids=candidate_ids
            )
            stage2_time = time.time() - stage2_start

            total_time = time.time() - start_time

            stage1_times.append(stage1_time)
            stage2_times.append(stage2_time)
            total_times.append(total_time)

        # Calculate statistics
        benchmark_results[rerank_k] = {
            'stage1_avg_ms': np.mean(stage1_times) * 1000,
            'stage2_avg_ms': np.mean(stage2_times) * 1000,
            'total_avg_ms': np.mean(total_times) * 1000,
            'throughput_per_sec': 1.0 / np.mean(total_times)
        }

    return benchmark_results
```

---

## Integration

### With Query Processor

```python
def search_by_image_file(
    image_path: Path,
    index: AnnIndex,
    store: MetadataStore,
    reranker: SinkhornReranker,
    processor: QueryProcessor
) -> List[SearchResult]:
    """Complete search pipeline from image file."""

    # Process query image
    query_histogram = processor.process_image_file(image_path)

    # Perform search
    results = find_similar(
        query_histogram=query_histogram,
        index=index,
        store=store,
        k=50,
        max_rerank=200
    )

    return results
```

### With API System

```python
def api_search_endpoint(
    query_histogram: np.ndarray,
    index: AnnIndex,
    store: MetadataStore,
    reranker: SinkhornReranker,
    k: int = 50,
    max_rerank: int = 200
) -> Dict[str, Any]:
    """API endpoint for search functionality."""

    try:
        # Perform search
        results = find_similar(
            query_histogram=query_histogram,
            index=index,
            store=store,
            k=k,
            max_rerank=max_rerank
        )

        # Format results for API
        formatted_results = []
        for result in results:
            formatted_results.append({
                'image_id': result.image_id,
                'file_path': result.file_path,
                'distance': result.distance,
                'rank': result.rank
            })

        return {
            'status': 'success',
            'results': formatted_results,
            'total_results': len(formatted_results),
            'query_parameters': {
                'k': k,
                'max_rerank': max_rerank
            }
        }

    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'query_parameters': {
                'k': k,
                'max_rerank': max_rerank
            }
        }
```

---

## Conclusion

The Two-Stage Search Logic provides an optimal balance between search speed and result quality through its innovative architecture:

- **Efficiency**: Fast FAISS ANN search for initial candidate retrieval
- **Quality**: High-fidelity Sinkhorn-EMD reranking for accurate results
- **Scalability**: Memory-efficient processing for large datasets
- **Flexibility**: Configurable parameters for different use cases

The system successfully implements the two-stage search architecture specified in the critical instructions document, delivering production-ready color image search capabilities.

For more information, see:

- [FAISS and DuckDB Wrappers](faiss_duckdb_wrappers.md)
- [Sinkhorn Reranking Logic](sinkhorn_reranking_logic.md)
- [Query Processor](query_processor.md)
- [FastAPI Endpoint](fastapi_endpoint.md)
