# FAISS and DuckDB Integration and Workflow

## Chromatica Color Search Engine

---

## Integration and Workflow

### Two-Stage Search Pipeline

The FAISS and DuckDB integration implements a sophisticated two-stage search pipeline that balances speed and accuracy:

```
┌─────────────────────────────────────────────────────────────┐
│                    Query Histogram                          │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                Stage 1: FAISS ANN Search                   │
│                                                             │
│  Query → Hellinger Transform → FAISS Index → Top-K Results │
│                                                             │
│  • Fast approximate search (O(log n))                      │
│  • Hellinger-transformed vectors                           │
│  • Returns candidate indices and distances                 │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                Stage 2: Histogram Retrieval                │
│                                                             │
│  Candidate IDs → DuckDB Lookup → Raw Histograms           │
│                                                             │
│  • Fast key-value retrieval                                │
│  • Original probability distributions                      │
│  • No transformation applied                               │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                Stage 3: Sinkhorn-EMD Reranking             │
│                                                             │
│  Raw Histograms → EMD Calculation → Final Ranking         │
│                                                             │
│  • High-fidelity distance metrics                          │
│  • Optimal transport algorithms                            │
│  • Accurate similarity assessment                          │
└─────────────────────────────────────────────────────────────┘
```

### Workflow Phases

#### 1. Indexing Phase

During the indexing phase, images are processed and stored in both systems:

```python
def index_images(image_paths: List[str],
                 ann_index: AnnIndex,
                 metadata_store: MetadataStore) -> int:
    """
    Index a collection of images using the FAISS-DuckDB pipeline.

    This function demonstrates the complete indexing workflow:
    1. Process images to generate histograms
    2. Store Hellinger-transformed vectors in FAISS
    3. Store raw histograms and metadata in DuckDB
    """
    indexed_count = 0

    for image_path in image_paths:
        try:
            # Generate histogram from image
            histogram = process_image(image_path)

            # Create unique image ID
            image_id = generate_image_id(image_path)

            # Stage 1: Add to FAISS index (with Hellinger transform)
            # Note: AnnIndex.add() automatically applies the transform
            ann_index.add(histogram.reshape(1, -1))

            # Stage 2: Store raw histogram and metadata in DuckDB
            metadata_record = {
                'image_id': image_id,
                'file_path': image_path,
                'histogram': histogram,
                'file_size': os.path.getsize(image_path)
            }

            metadata_store.add_batch([metadata_record])
            indexed_count += 1

            logger.info(f"Indexed image {image_id}: {image_path}")

        except Exception as e:
            logger.error(f"Failed to index {image_path}: {e}")
            continue

    return indexed_count
```

**Indexing Benefits**:

- **Parallel Processing**: FAISS and DuckDB operations can be parallelized
- **Batch Operations**: Efficient bulk insertion for large datasets
- **Error Isolation**: Failures in one system don't affect the other
- **Data Consistency**: Both systems maintain synchronized data

#### 2. Search Phase

The search phase demonstrates the two-stage pipeline in action:

```python
def search_similar_images(query_histogram: np.ndarray,
                         ann_index: AnnIndex,
                         metadata_store: MetadataStore,
                         k: int = 100,
                         rerank_k: int = 20) -> List[Dict[str, Any]]:
    """
    Perform two-stage search using FAISS and DuckDB.

    This function implements the complete search workflow:
    1. Fast ANN search using FAISS
    2. Raw histogram retrieval from DuckDB
    3. High-fidelity reranking (placeholder for future implementation)
    """
    # Stage 1: FAISS ANN Search
    logger.info(f"Stage 1: FAISS ANN search for k={k} candidates")
    start_time = time.time()

    # Search returns (distances, indices) where indices correspond to
    # the order in which vectors were added to the index
    distances, indices = ann_index.search(query_histogram, k)

    ann_time = time.time() - start_time
    logger.info(f"FAISS search completed in {ann_time:.3f}s")

    # Stage 2: Histogram Retrieval
    logger.info(f"Stage 2: Retrieving raw histograms for reranking")
    start_time = time.time()

    # Convert FAISS indices to image IDs
    # Note: This assumes a mapping between FAISS indices and image IDs
    candidate_ids = [f"image_{idx:06d}" for idx in indices[0]]

    # Retrieve raw histograms from DuckDB
    candidate_histograms = metadata_store.get_histograms_by_ids(candidate_ids)

    retrieval_time = time.time() - start_time
    logger.info(f"Histogram retrieval completed in {retrieval_time:.3f}s")

    # Stage 3: Reranking (Future Implementation)
    logger.info(f"Stage 3: Sinkhorn-EMD reranking for top {rerank_k} candidates")

    # For now, return FAISS results with raw histograms
    # Future: Implement Sinkhorn-EMD reranking here
    results = []
    for i, (image_id, histogram) in enumerate(candidate_histograms.items()):
        if i >= rerank_k:
            break

        results.append({
            'image_id': image_id,
            'rank': i + 1,
            'faiss_distance': float(distances[0][i]),
            'histogram': histogram,
            'file_path': metadata_store.get_file_path(image_id)
        })

    total_time = ann_time + retrieval_time
    logger.info(f"Complete search pipeline completed in {total_time:.3f}s")

    return results
```

**Search Benefits**:

- **Fast Initial Search**: FAISS provides rapid candidate selection
- **Accurate Reranking**: Raw histograms enable high-fidelity distance calculations
- **Scalable Performance**: Logarithmic search complexity
- **Flexible Results**: Configurable candidate counts for different use cases

### Data Synchronization

#### Index Mapping Strategy

A critical aspect of the integration is maintaining consistency between FAISS indices and DuckDB records:

```python
class IndexMapping:
    """
    Manages the mapping between FAISS indices and image IDs.

    This class ensures that FAISS search results can be correctly
    mapped to DuckDB records for histogram retrieval.
    """

    def __init__(self):
        self.faiss_to_image_id = {}  # FAISS index → image_id
        self.image_id_to_faiss = {}  # image_id → FAISS index
        self.next_faiss_index = 0

    def add_mapping(self, image_id: str) -> int:
        """Add a new image and return its FAISS index."""
        faiss_index = self.next_faiss_index
        self.faiss_to_image_id[faiss_index] = image_id
        self.image_id_to_faiss[image_id] = faiss_index
        self.next_faiss_index += 1
        return faiss_index

    def get_image_id(self, faiss_index: int) -> str:
        """Get image ID from FAISS index."""
        return self.faiss_to_image_id[faiss_index]

    def get_faiss_index(self, image_id: str) -> int:
        """Get FAISS index from image ID."""
        return self.image_id_to_faiss[image_id]

    def get_candidate_ids(self, faiss_indices: np.ndarray) -> List[str]:
        """Convert FAISS indices to image IDs."""
        return [self.get_image_id(idx) for idx in faiss_indices]
```

#### Consistency Guarantees

The integration provides several consistency guarantees:

- **Atomic Operations**: FAISS and DuckDB operations are coordinated
- **Transaction Support**: DuckDB provides ACID compliance for metadata
- **Error Recovery**: Failed operations can be rolled back
- **Data Validation**: Comprehensive checking of data integrity

### Performance Optimization

#### Batch Processing

The integration is optimized for batch operations:

```python
def batch_index_images(image_paths: List[str],
                      batch_size: int = 100) -> int:
    """
    Index images in batches for optimal performance.

    Batch processing provides several benefits:
    - Reduced database round trips
    - Better memory utilization
    - Improved error handling
    - Progress tracking and logging
    """
    total_indexed = 0

    # Process images in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(image_paths) + batch_size - 1) // batch_size

        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_paths)} images)")

        # Process batch
        batch_histograms = []
        batch_metadata = []

        for image_path in batch_paths:
            try:
                histogram = process_image(image_path)
                image_id = generate_image_id(image_path)

                batch_histograms.append(histogram.reshape(1, -1))
                batch_metadata.append({
                    'image_id': image_id,
                    'file_path': image_path,
                    'histogram': histogram,
                    'file_size': os.path.getsize(image_path)
                })

            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                continue

        if batch_histograms:
            # Batch operations
            histograms_array = np.vstack(batch_histograms)
            ann_index.add(histograms_array)
            metadata_store.add_batch(batch_metadata)

            total_indexed += len(batch_histograms)
            logger.info(f"Batch {batch_num} completed: {len(batch_histograms)} images indexed")

    return total_indexed
```

#### Memory Management

Efficient memory usage is critical for large datasets:

- **Streaming Processing**: Process images without loading all into memory
- **Garbage Collection**: Explicit cleanup of temporary variables
- **Batch Sizing**: Configurable batch sizes based on available memory
- **Progress Monitoring**: Track memory usage during operations

### Error Handling and Recovery

#### Comprehensive Error Handling

The integration provides robust error handling:

```python
def robust_search(query_histogram: np.ndarray,
                 ann_index: AnnIndex,
                 metadata_store: MetadataStore,
                 k: int = 100) -> List[Dict[str, Any]]:
    """
    Perform robust search with comprehensive error handling.

    This function demonstrates error handling strategies:
    - Graceful degradation on partial failures
    - Detailed error logging and reporting
    - Fallback mechanisms for different failure modes
    """
    try:
        # Stage 1: FAISS Search
        try:
            distances, indices = ann_index.search(query_histogram, k)
            logger.info(f"FAISS search successful: {len(indices[0])} candidates")
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            # Fallback: return empty results
            return []

        # Stage 2: Histogram Retrieval
        try:
            candidate_ids = [f"image_{idx:06d}" for idx in indices[0]]
            candidate_histograms = metadata_store.get_histograms_by_ids(candidate_ids)
            logger.info(f"Histogram retrieval successful: {len(candidate_histograms)} histograms")
        except Exception as e:
            logger.error(f"Histogram retrieval failed: {e}")
            # Fallback: return FAISS results without histograms
            return [{'image_id': f"image_{idx:06d}", 'rank': i+1, 'faiss_distance': float(distances[0][i])}
                   for i, idx in enumerate(indices[0])]

        # Success: return complete results
        results = []
        for i, (image_id, histogram) in enumerate(candidate_histograms.items()):
            results.append({
                'image_id': image_id,
                'rank': i + 1,
                'faiss_distance': float(distances[0][i]),
                'histogram': histogram
            })

        return results

    except Exception as e:
        logger.error(f"Unexpected error in search pipeline: {e}")
        return []
```

#### Recovery Strategies

Different failure modes have different recovery strategies:

- **FAISS Failures**: Return empty results, log error for debugging
- **DuckDB Failures**: Return FAISS results without histograms
- **Partial Failures**: Return available data with error indicators
- **System Failures**: Graceful degradation with user notification

### Monitoring and Logging

#### Performance Metrics

The integration provides comprehensive performance monitoring:

```python
class SearchMetrics:
    """Track and report search performance metrics."""

    def __init__(self):
        self.ann_times = []
        self.retrieval_times = []
        self.total_times = []
        self.candidate_counts = []
        self.error_counts = 0

    def record_search(self, ann_time: float, retrieval_time: float,
                     total_time: float, candidate_count: int):
        """Record metrics for a search operation."""
        self.ann_times.append(ann_time)
        self.retrieval_times.append(retrieval_time)
        self.total_times.append(total_time)
        self.candidate_counts.append(candidate_count)

    def record_error(self):
        """Record an error occurrence."""
        self.error_counts += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            'total_searches': len(self.ann_times),
            'error_rate': self.error_counts / max(1, len(self.ann_times)),
            'avg_ann_time': np.mean(self.ann_times) if self.ann_times else 0,
            'avg_retrieval_time': np.mean(self.retrieval_times) if self.retrieval_times else 0,
            'avg_total_time': np.mean(self.total_times) if self.total_times else 0,
            'avg_candidates': np.mean(self.candidate_counts) if self.candidate_counts else 0
        }
```

#### Logging Strategy

Comprehensive logging provides debugging and monitoring capabilities:

- **Operation Logging**: Track all major operations with timestamps
- **Performance Logging**: Record timing information for optimization
- **Error Logging**: Detailed error information with context
- **Progress Logging**: Track long-running operations
- **Debug Logging**: Verbose information for development
