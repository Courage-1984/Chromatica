# FAISS and DuckDB Integration Guide

## Chromatica Color Search Engine

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [FAISS Index Implementation](#faiss-index-implementation)
4. [DuckDB Metadata Store](#duckdb-metadata-store)
5. [Integration and Workflow](#integration-and-workflow)
6. [Usage Examples](#usage-examples)
7. [Testing and Validation](#testing-and-validation)
8. [Performance Considerations](#performance-considerations)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Features](#advanced-features)

---

## Overview

This document provides comprehensive documentation for the FAISS and DuckDB integration in the Chromatica color search engine. The system implements a two-stage search pipeline where:

- **FAISS HNSW Index**: Provides fast approximate nearest neighbor search using Hellinger-transformed histograms
- **DuckDB Metadata Store**: Manages image metadata and raw histograms for high-fidelity reranking

The integration enables efficient color-based image search with the following key features:

- **Automatic Hellinger Transform**: Converts normalized histograms to L2-compatible vectors
- **Fast ANN Search**: HNSW algorithm provides excellent speed-accuracy trade-offs
- **Raw Histogram Preservation**: Maintains original distributions for Sinkhorn-EMD reranking
- **Batch Operations**: Efficient processing of large image collections
- **Persistence**: Save/load capabilities for long-term storage

### Key Components

- **`AnnIndex` Class**: Wraps `faiss.IndexHNSWFlat` with automatic Hellinger transformation
- **`MetadataStore` Class**: Manages DuckDB database operations and histogram storage
- **Integration Pipeline**: Seamless workflow from indexing to search and reranking

### Technology Stack

- **FAISS**: Facebook AI Similarity Search library for vector indexing
- **DuckDB**: Embedded analytical database for metadata management
- **NumPy**: Numerical operations and array management
- **Python 3.10+**: Modern Python with type hints and comprehensive error handling

---

## Architecture

### System Design

The FAISS and DuckDB integration follows a layered architecture designed for scalability and performance:

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
├─────────────────────────────────────────────────────────────┤
│                 Search and Query Layer                      │
├─────────────────────────────────────────────────────────────┤
│                Reranking Layer (Future)                    │
├─────────────────────────────────────────────────────────────┤
│              FAISS Index + DuckDB Store                    │
├─────────────────────────────────────────────────────────────┤
│                Histogram Generation                        │
├─────────────────────────────────────────────────────────────┤
│                   Image Processing                         │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Indexing Phase**:

   - Images → Histograms → Hellinger Transform → FAISS Index
   - Images → Histograms → Raw Storage → DuckDB

2. **Search Phase**:
   - Query → Histogram → Hellinger Transform → FAISS Search
   - FAISS Results → Raw Histograms → Sinkhorn-EMD Reranking

### Component Responsibilities

#### FAISS Index (`AnnIndex`)

- **Vector Storage**: Hellinger-transformed histograms
- **Fast Search**: Approximate nearest neighbor retrieval
- **Distance Metrics**: L2 distance compatibility
- **Scalability**: Efficient indexing of large datasets

#### DuckDB Store (`MetadataStore`)

- **Metadata Management**: Image IDs, file paths, file sizes
- **Raw Histogram Storage**: Original probability distributions
- **Fast Retrieval**: Key-value lookups for reranking
- **Batch Operations**: Efficient bulk data processing

### Hellinger Transform

The Hellinger transform is a mathematical operation that converts probability distributions to vectors compatible with L2 distance metrics:

```
φ(h) = √h

Where:
- h: Normalized histogram (sum = 1.0)
- φ(h): Hellinger-transformed vector
```

**Benefits**:

- Maintains relative similarity relationships
- Enables L2 distance calculations in FAISS
- Preserves histogram structure for accurate search
- Standard technique in information retrieval

**Mathematical Properties**:

- Preserves order relationships between histograms
- Maintains clustering structure
- Enables efficient L2-based indexing
- Compatible with cosine similarity measures

---

## FAISS Index Implementation

### AnnIndex Class

The `AnnIndex` class provides a high-level wrapper around FAISS's HNSW (Hierarchical Navigable Small World) index, automatically handling Hellinger transformations and providing a clean interface for vector operations.

#### Class Overview

```python
class AnnIndex:
    """
    Wrapper class for FAISS HNSW index with automatic Hellinger transformation.

    This class manages a FAISS Hierarchical Navigable Small World (HNSW) index
    that stores color histograms transformed using the Hellinger transform.
    The Hellinger transform (element-wise square root) converts normalized
    histograms into vectors compatible with L2 distance metrics used by FAISS.
    """
```

#### Key Attributes

- **`index`**: The underlying FAISS HNSW index (`faiss.IndexHNSWFlat`)
- **`dimension`**: Vector dimensionality (1152 for Lab color histograms)
- **`total_vectors`**: Current number of indexed vectors
- **`is_trained`**: Training status (always `True` for HNSW)

#### Initialization

```python
def __init__(self, dimension: int = TOTAL_BINS):
    """
    Initialize the FAISS HNSW index.

    Args:
        dimension: The dimensionality of the vectors to be indexed.
                  Defaults to TOTAL_BINS (1152) for Lab color histograms.

    Raises:
        ValueError: If dimension is not positive.
    """
    if dimension <= 0:
        raise ValueError(f"Dimension must be positive, got {dimension}")

    self.dimension = dimension
    self.total_vectors = 0

    # Create HNSW index with M=32 neighbors for optimal performance
    # HNSW doesn't require training, so is_trained is always True
    self.index = faiss.IndexHNSWFlat(dimension, HNSW_M)
    self.is_trained = True

    logger.info(f"Initialized FAISS HNSW index with dimension {dimension}, M={HNSW_M}")
```

**Configuration Constants**:

- **`HNSW_M = 32`**: Number of neighbors in the HNSW graph (optimized for performance)
- **`TOTAL_BINS = 1152`**: Total histogram dimensions (8×12×12 Lab color bins)

### Core Methods

#### 1. Vector Addition (`add`)

The `add` method automatically applies the Hellinger transform before indexing vectors:

```python
def add(self, vectors: np.ndarray) -> int:
    """
    Add vectors to the index after applying Hellinger transformation.

    This method applies the Hellinger transform (element-wise square root)
    to the input histograms before adding them to the FAISS index. The
    transform ensures compatibility with the L2 distance metric used by
    the index while preserving the relative relationships between histograms.

    Args:
        vectors: Array of shape (n_vectors, dimension) containing normalized
                histograms to be indexed. Must be float32 or float64.

    Returns:
        The number of vectors successfully added to the index.

    Raises:
        ValueError: If vectors have incorrect shape or dtype.
        RuntimeError: If FAISS index addition fails.
    """
    # Input validation
    if not isinstance(vectors, np.ndarray):
        raise ValueError(f"Vectors must be numpy array, got {type(vectors)}")

    if vectors.ndim != 2:
        raise ValueError(f"Vectors must be 2D array, got shape {vectors.shape}")

    if vectors.shape[1] != self.dimension:
        raise ValueError(
            f"Vector dimension {vectors.shape[1]} doesn't match index dimension {self.dimension}"
        )

    # Ensure vectors are float32 for optimal FAISS performance
    vectors_float32 = vectors.astype(np.float32)

    # Apply Hellinger transform: φ(h) = √h
    # This makes histograms compatible with L2 distance metrics
    vectors_hellinger = np.sqrt(vectors_float32)

    # Add transformed vectors to the index
    try:
        self.index.add(vectors_hellinger)
        added_count = vectors.shape[0]
        self.total_vectors += added_count

        logger.info(
            f"Added {added_count} vectors to FAISS index (total: {self.total_vectors})"
        )
        return added_count

    except Exception as e:
        logger.error(f"Failed to add vectors to FAISS index: {e}")
        raise RuntimeError(f"FAISS index addition failed: {e}")
```

**Key Features**:

- **Automatic Transformation**: Hellinger transform applied transparently
- **Type Optimization**: Converts to float32 for optimal FAISS performance
- **Error Handling**: Comprehensive validation and error reporting
- **Logging**: Detailed operation tracking for debugging

#### 2. Vector Search (`search`)

The search method automatically transforms query vectors and returns nearest neighbors:

```python
def search(self, query_vector: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search for the k nearest neighbors of a query vector.

    The query vector is automatically transformed using the Hellinger transform
    to ensure compatibility with the indexed vectors. Returns both the
    distances and indices of the k nearest neighbors.

    Args:
        query_vector: Query histogram of shape (dimension,) or (1, dimension).
                     Must be a normalized histogram (sum = 1.0).
        k: Number of nearest neighbors to retrieve.

    Returns:
        Tuple of (distances, indices) where:
        - distances: Array of shape (1, k) containing L2 distances
        - indices: Array of shape (1, k) containing vector indices

    Raises:
        ValueError: If query_vector has incorrect shape or k is invalid.
        RuntimeError: If FAISS search fails.
    """
    # Input validation and reshaping
    if not isinstance(query_vector, np.ndarray):
        raise ValueError(f"Query vector must be numpy array, got {type(query_vector)}")

    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)
    elif query_vector.ndim != 2 or query_vector.shape[0] != 1:
        raise ValueError(
            f"Query vector must be 1D or (1, dimension), got shape {query_vector.shape}"
        )

    if query_vector.shape[1] != self.dimension:
        raise ValueError(
            f"Query vector dimension {query_vector.shape[1]} doesn't match index dimension {self.dimension}"
        )

    if k <= 0 or k > self.total_vectors:
        raise ValueError(f"k must be positive and <= {self.total_vectors}, got {k}")

    # Apply Hellinger transform to query vector
    query_hellinger = np.sqrt(query_vector.astype(np.float32))

    # Perform search
    try:
        distances, indices = self.index.search(query_hellinger, k)

        logger.debug(
            f"FAISS search completed: k={k}, min_distance={distances.min():.4f}, max_distance={distances.max():.4f}"
        )
        return distances, indices

    except Exception as e:
        logger.error(f"FAISS search failed: {e}")
        raise RuntimeError(f"FAISS search failed: {e}")
```

**Search Features**:

- **Automatic Transformation**: Query vectors transformed to match indexed vectors
- **Flexible Input**: Accepts both 1D and 2D query vectors
- **Comprehensive Validation**: Shape, dimension, and range checking
- **Performance Logging**: Distance statistics for optimization

#### 3. Utility Methods

Additional methods provide index management and information:

```python
def get_total_vectors(self) -> int:
    """Get the total number of vectors currently indexed."""
    return self.total_vectors

def save(self, filepath: str) -> None:
    """Save the FAISS index to disk."""
    try:
        faiss.write_index(self.index, filepath)
        logger.info(f"FAISS index saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save FAISS index to {filepath}: {e}")
        raise RuntimeError(f"Failed to save FAISS index: {e}")

def load(self, filepath: str) -> None:
    """Load a FAISS index from disk."""
    if not Path(filepath).exists():
        raise FileNotFoundError(f"FAISS index file not found: {filepath}")

    try:
        self.index = faiss.read_index(filepath)
        self.dimension = self.index.d
        self.total_vectors = self.index.ntotal
        self.is_trained = True

        logger.info(
            f"FAISS index loaded from {filepath} with {self.total_vectors} vectors"
        )
    except Exception as e:
        logger.error(f"Failed to load FAISS index from {filepath}: {e}")
        raise RuntimeError(f"Failed to load FAISS index: {e}")
```

### HNSW Algorithm Benefits

The Hierarchical Navigable Small World algorithm provides several advantages:

- **No Training Required**: Unlike some FAISS indices, HNSW works immediately
- **Excellent Accuracy**: Near-optimal search results with minimal approximation
- **Scalable Performance**: Logarithmic search complexity
- **Memory Efficient**: Compact graph representation
- **Configurable**: M parameter controls accuracy vs. speed trade-off

### Performance Characteristics

- **Indexing Speed**: O(n log n) complexity for n vectors
- **Search Speed**: O(log n) average case complexity
- **Memory Usage**: Approximately 4 bytes per vector × M neighbors
- **Accuracy**: Near-exact nearest neighbor results
- **Scalability**: Efficient for datasets up to millions of vectors

---

## DuckDB Metadata Store

### MetadataStore Class

The `MetadataStore` class provides a high-level interface for managing image metadata and raw histograms in DuckDB. It's designed for efficient batch operations and fast retrieval during the reranking phase.

#### Class Overview

```python
class MetadataStore:
    """
    Manages DuckDB database for storing image metadata and raw histograms.

    This class provides a high-level interface for storing and retrieving
    image metadata (IDs, file paths) and raw color histograms. The raw
    histograms are essential for the second stage of search (Sinkhorn-EMD
    reranking) since they preserve the original probability distributions
    without the Hellinger transformation applied to the FAISS index.

    The database uses efficient batch operations for indexing and provides
    fast key-value lookups for retrieving histograms by image IDs during
    the reranking phase.
    """
```

#### Key Attributes

- **`db_path`**: Path to the DuckDB database file (or `:memory:` for in-memory)
- **`connection`**: Active DuckDB database connection
- **`table_name`**: Name of the main metadata table (`image_metadata`)

#### Initialization

```python
def __init__(self, db_path: str = ":memory:"):
    """
    Initialize the DuckDB metadata store.

    Args:
        db_path: Path to the DuckDB database file. Defaults to ":memory:"
                for in-memory database (useful for testing).
    """
    self.db_path = db_path
    self.table_name = "image_metadata"

    # Initialize connection
    self.connection = duckdb.connect(db_path)

    # Set up the database schema
    self.setup_table()

    logger.info(f"Initialized DuckDB metadata store at {db_path}")
```

**Database Options**:

- **`:memory:`**: In-memory database for testing and temporary storage
- **File Path**: Persistent database file for long-term storage
- **Network**: Remote DuckDB server connection (future enhancement)

### Database Schema

The metadata store uses a simple but efficient schema designed for fast lookups and batch operations:

```sql
CREATE TABLE IF NOT EXISTS image_metadata (
    image_id VARCHAR PRIMARY KEY,
    file_path VARCHAR NOT NULL,
    histogram BLOB NOT NULL,
    file_size BIGINT
);
```

#### Schema Details

- **`image_id`**: Unique identifier for each image (primary key)
- **`file_path`**: Path to the image file on disk
- **`histogram`**: Raw color histogram stored as BLOB (binary data)
- **`file_size`**: Size of the image file in bytes (optional)

#### Indexes

The store automatically creates indexes for optimal performance:

```sql
CREATE INDEX IF NOT EXISTS idx_file_path ON image_metadata(file_path);
```

**Index Benefits**:

- **Fast File Lookups**: Efficient searches by file path
- **Duplicate Detection**: Quick identification of existing images
- **Batch Operations**: Optimized for bulk insertions and updates

### Core Methods

#### 1. Table Setup (`setup_table`)

The setup method creates the database schema and indexes:

```python
def setup_table(self) -> None:
    """
    Create the main metadata table if it doesn't exist.

    This method creates a table with the following schema:
    - image_id: Unique identifier for each image
    - file_path: Path to the image file
    - histogram: Raw color histogram as a BLOB (1152 dimensions)
    - file_size: Size of the image file in bytes

    The histogram is stored as a BLOB to preserve the exact
    floating-point values and maintain compatibility with DuckDB's
    binary data support.
    """
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {self.table_name} (
        image_id VARCHAR PRIMARY KEY,
        file_path VARCHAR NOT NULL,
        histogram BLOB NOT NULL,
        file_size BIGINT
    );
    """

    try:
        self.connection.execute(create_table_sql)

        # Create index on file_path for faster lookups
        index_sql = f"CREATE INDEX IF NOT EXISTS idx_file_path ON {self.table_name}(file_path);"
        self.connection.execute(index_sql)

        logger.info(f"Database table '{self.table_name}' setup completed")

    except Exception as e:
        logger.error(f"Failed to setup database table: {e}")
        raise RuntimeError(f"Database setup failed: {e}")
```

**Setup Features**:

- **Automatic Creation**: Tables and indexes created if they don't exist
- **Error Handling**: Comprehensive error reporting for debugging
- **Logging**: Detailed operation tracking
- **Index Optimization**: Automatic performance optimization

#### 2. Batch Insertion (`add_batch`)

Efficiently insert multiple image records in a single operation:

```python
def add_batch(self, metadata_batch: List[Dict[str, Any]]) -> int:
    """
    Add multiple image metadata records in a single batch operation.

    This method efficiently inserts multiple image records using DuckDB's
    batch insertion capabilities. Each record should contain image_id,
    file_path, and histogram data. The histogram must be a 1D numpy array
    that will be converted to BLOB for storage.

    Args:
        metadata_batch: List of dictionaries, each containing:
                       - image_id: Unique image identifier
                       - file_path: Path to the image file
                       - histogram: 1D numpy array of histogram values
                       - file_size: Optional file size in bytes

    Returns:
        The number of records successfully inserted.

    Raises:
        ValueError: If metadata_batch is empty or contains invalid data.
        RuntimeError: If database insertion fails.
    """
    if not metadata_batch:
        raise ValueError("Metadata batch cannot be empty")

    # Prepare batch data for insertion
    insert_data = []
    for record in metadata_batch:
        if not all(key in record for key in ["image_id", "file_path", "histogram"]):
            raise ValueError(f"Record missing required fields: {record.keys()}")

        # Convert histogram to bytes for BLOB storage
        histogram_bytes = (
            record["histogram"].tobytes()
            if isinstance(record["histogram"], np.ndarray)
            else np.array(record["histogram"]).tobytes()
        )

        insert_data.append((
            record["image_id"],
            record["file_path"],
            histogram_bytes,
            record.get("file_size", 0),
        ))

    # Perform batch insertion
    insert_sql = f"""
    INSERT OR REPLACE INTO {self.table_name} (image_id, file_path, histogram, file_size)
    VALUES (?, ?, ?, ?)
    """

    try:
        self.connection.executemany(insert_sql, insert_data)
        inserted_count = len(insert_data)

        logger.info(f"Successfully inserted {inserted_count} metadata records")
        return inserted_count

    except Exception as e:
        logger.error(f"Failed to insert metadata batch: {e}")
        raise RuntimeError(f"Database insertion failed: {e}")
```

**Batch Features**:

- **Efficient Insertion**: Single SQL statement for multiple records
- **Data Validation**: Comprehensive field checking
- **Type Conversion**: Automatic numpy array to BLOB conversion
- **UPSERT Logic**: `INSERT OR REPLACE` handles duplicates gracefully
- **Error Handling**: Detailed error reporting for failed operations

#### 3. Histogram Retrieval (`get_histograms_by_ids`)

Fast retrieval of raw histograms for reranking:

```python
def get_histograms_by_ids(self, image_ids: List[str]) -> Dict[str, np.ndarray]:
    """
    Retrieve raw histograms for a list of image IDs.

    This method is crucial for the reranking stage, as it provides
    the original, non-transformed histograms needed for accurate
    Sinkhorn-EMD distance calculations. The histograms are returned
    as a dictionary mapping image_id to numpy array.

    Args:
        image_ids: List of image IDs to retrieve histograms for.

    Returns:
        Dictionary mapping image_id to histogram numpy array.
        Only includes IDs that were found in the database.

    Raises:
        ValueError: If image_ids is empty.
        RuntimeError: If database query fails.
    """
    if not image_ids:
        raise ValueError("Image IDs list cannot be empty")

    # Create placeholders for the IN clause
    placeholders = ",".join(["?" for _ in image_ids])

    query_sql = f"""
    SELECT image_id, histogram
    FROM {self.table_name}
    WHERE image_id IN ({placeholders})
    """

    try:
        result = self.connection.execute(query_sql, image_ids).fetchall()

        # Convert results to dictionary
        histograms = {}
        for image_id, histogram_blob in result:
            # Convert BLOB back to numpy array
            histogram_array = np.frombuffer(histogram_blob, dtype=np.float32)
            histograms[image_id] = histogram_array

        logger.debug(
            f"Retrieved {len(histograms)} histograms for {len(image_ids)} requested IDs"
        )
        return histograms

    except Exception as e:
        logger.error(f"Failed to retrieve histograms: {e}")
        raise RuntimeError(f"Database query failed: {e}")
```

**Retrieval Features**:

- **Fast Lookups**: Optimized IN clause queries
- **BLOB Conversion**: Automatic conversion back to numpy arrays
- **Partial Results**: Returns only found histograms
- **Type Preservation**: Maintains original float32 precision
- **Performance Logging**: Detailed operation tracking

#### 4. Utility Methods

Additional methods provide database management and information:

```python
def get_image_count(self) -> int:
    """Get the total number of images in the metadata store."""
    try:
        result = self.connection.execute(
            f"SELECT COUNT(*) FROM {self.table_name}"
        ).fetchone()
        return result[0] if result else 0
    except Exception as e:
        logger.error(f"Failed to get image count: {e}")
        return 0

def close(self) -> None:
    """Close the database connection."""
    if self.connection:
        self.connection.close()
        logger.info("Database connection closed")

def __enter__(self):
    """Context manager entry point."""
    return self

def __exit__(self, exc_type, exc_val, exc_tb):
    """Context manager exit point."""
    self.close()
```

### Data Storage Strategy

#### BLOB vs. JSON Storage

The store uses BLOB storage for histograms instead of JSON for several reasons:

**BLOB Advantages**:

- **Precision**: Preserves exact floating-point values
- **Performance**: Faster serialization/deserialization
- **Size**: More compact storage format
- **Type Safety**: Maintains numpy array data types
- **Efficiency**: Optimized for binary data operations

**JSON Alternatives**:

- **Human Readable**: Easy to inspect and debug
- **Flexibility**: Schema evolution and metadata support
- **Compatibility**: Standard format for data exchange
- **Querying**: SQL JSON functions for analysis

#### Memory Management

The store implements efficient memory management:

- **Batch Operations**: Minimize memory overhead during bulk operations
- **Connection Pooling**: Efficient database connection management
- **Context Managers**: Automatic resource cleanup with `with` statements
- **Error Recovery**: Graceful handling of memory constraints

### Performance Characteristics

- **Insertion Speed**: O(1) per record with batch operations
- **Retrieval Speed**: O(log n) with indexed lookups
- **Memory Usage**: Minimal overhead beyond histogram storage
- **Scalability**: Efficient for datasets up to millions of images
- **Concurrency**: Single connection design for simplicity

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
