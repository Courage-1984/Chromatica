"""
FAISS and DuckDB wrapper classes for the Chromatica color search engine.

This module provides high-level abstractions for managing the FAISS IndexIVFPQ index
and DuckDB metadata store. The AnnIndex class handles vector indexing with
automatic Hellinger transformation and Product Quantization compression, while the
MetadataStore class manages image metadata and raw histogram storage for the reranking stage.

Key Components:
- AnnIndex: Wraps faiss.IndexIVFPQ with Hellinger transform and Product Quantization
- MetadataStore: Manages DuckDB connection and operations
- Batch operations for efficient indexing and retrieval

The Hellinger transform is applied automatically to make histograms compatible
with the L2-based FAISS index, while raw histograms are preserved in DuckDB
for high-fidelity Sinkhorn-EMD reranking. Product Quantization significantly
reduces memory usage by compressing vectors into compact codes.
"""

import logging
import numpy as np
import faiss
import duckdb
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from functools import lru_cache
import threading
import os

from ..utils.config import TOTAL_BINS, IVFPQ_NLIST, IVFPQ_M, IVFPQ_NBITS, IVFPQ_NPROBE

# Configure logging for this module
logger = logging.getLogger(__name__)


class AnnIndex:
    """
    Wrapper for FAISS index used in Chromatica.
    Supports HNSW, Flat, and IVFPQ indices.
    """

    def __init__(
        self,
        dimension: int,
        use_simple_index: bool = False,
        index_path: str = None,
        nlist: int = 16384,
        M: int = 32,
        nbits: int = 8,
    ):
        """
        Initialize the FAISS index.

        Args:
            dimension: Number of dimensions for the vectors.
            use_simple_index: If True, use IndexFlatL2 or HNSW; else use IVFPQ.
            index_path: Optional path to load an existing index.
            nlist: Number of coarse clusters (IVFPQ only).
            M: Number of subquantizers (IVFPQ only).
            nbits: Number of bits per subquantizer (IVFPQ only).
        """
        self.dimension = dimension
        self.use_simple_index = use_simple_index
        self.index_type = None

        if index_path and os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            self.index_type = type(self.index).__name__
            logging.getLogger(__name__).info(
                f"Loaded existing FAISS index from {index_path}"
            )
        else:
            if use_simple_index:
                self.index = faiss.IndexHNSWFlat(dimension, M)
                self.index_type = "IndexHNSWFlat"
            else:
                quantizer = faiss.IndexFlatL2(dimension)
                self.index = faiss.IndexIVFPQ(quantizer, dimension, nlist, M, nbits)
                self.index_type = "IndexIVFPQ"
                self.M = M
                self.nbits = nbits
                self.nlist = nlist
            logging.getLogger(__name__).info(
                f"Created new FAISS {self.index_type} index"
            )

    @property
    def is_trained(self) -> bool:
        """
        Returns True if the underlying FAISS index is trained, False otherwise.
        """
        return getattr(self.index, "is_trained", True)

    @property
    def total_vectors(self) -> int:
        """Get total number of vectors in the index."""
        return self.index.ntotal

    def train(self, data: np.ndarray) -> None:
        """
        Train the FAISS index if required.

        Args:
            data: Training data (N, D)
        """
        import faiss

        logger = logging.getLogger(__name__)
        if hasattr(self.index, "is_trained") and not self.index.is_trained:
            logger.info(f"Training {self.index_type} with {len(data)} vectors...")
            try:
                self.index.train(data)
                # Only log memory usage for IVFPQ
                if self.index_type == "IndexIVFPQ":
                    logger.info(
                        f"Memory usage per vector: ~{self.M * self.nbits / 8} bytes "
                        f"(M={self.M}, nbits={self.nbits})"
                    )
            except Exception as e:
                raise RuntimeError(f"Index training failed: {e}")
        else:
            logger.info(
                f"{self.index_type} does not require training or is already trained."
            )

    def add(self, vectors: np.ndarray) -> int:
        """
        Add vectors to the index after applying Hellinger transformation.

        This method applies the Hellinger transform (element-wise square root)
        to the input histograms before adding them to the FAISS index. The
        transform ensures compatibility with the L2 distance metric used by
        the index while preserving the relative relationships between histograms.

        The vectors are then compressed using Product Quantization, dramatically
        reducing memory usage while maintaining good search quality.

        Args:
            vectors: Array of shape (n_vectors, dimension) containing normalized
                    histograms to be indexed. Must be float32 or float64.

        Returns:
            The number of vectors successfully added to the index.

        Raises:
            ValueError: If vectors have incorrect shape or dtype, or index not trained.
            RuntimeError: If FAISS index addition fails.
        """
        if not self.is_trained:
            raise ValueError(
                "Index must be trained before adding vectors. Call train() first."
            )

        if not isinstance(vectors, np.ndarray):
            raise ValueError(f"Vectors must be numpy array, got {type(vectors)}")

        if vectors.ndim != 2:
            raise ValueError(f"Vectors must be 2D array, got shape {vectors.shape}")

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension {vectors.shape[1]} doesn't match index dimension {self.dimension}"
            )

        vectors_float32 = vectors.astype(np.float32)
        vectors_hellinger = np.sqrt(vectors_float32)

        try:
            self.index.add(vectors_hellinger)
            added_count = vectors.shape[0]
            logger.info(
                f"Added {added_count} vectors to IndexIVFPQ (total: {self.get_total_vectors()})"
            )
            return added_count

        except Exception as e:
            logger.error(f"Failed to add vectors to IndexIVFPQ: {e}")
            raise RuntimeError(f"IndexIVFPQ addition failed: {e}")

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
        if not isinstance(query_vector, np.ndarray):
            raise ValueError(
                f"Query vector must be numpy array, got {type(query_vector)}"
            )

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
                f"IndexIVFPQ search completed: k={k}, min_distance={distances.min():.4f}, max_distance={distances.max():.4f}"
            )
            return distances, indices

        except Exception as e:
            logger.error(f"IndexIVFPQ search failed: {e}")
            raise RuntimeError(f"IndexIVFPQ search failed: {e}")

    def get_total_vectors(self) -> int:
        """
        Returns the total number of vectors currently in the FAISS index.
        """
        return getattr(self.index, "ntotal", 0)

    def get_memory_usage_estimate(self) -> Dict[str, float]:
        """
        Estimate memory usage of the IndexIVFPQ.

        Returns:
            Dictionary containing memory usage estimates in bytes:
            - total_vectors: Number of vectors
            - memory_per_vector: Estimated memory per vector in bytes
            - total_memory: Estimated total memory usage
            - compression_ratio: Compression ratio vs full vectors
        """
        # Memory per vector: M * nbits / 8 bytes (PQ codes)
        memory_per_vector = self.M * self.nbits / 8

        # Total memory estimate
        total_memory = self.total_vectors * memory_per_vector

        # Compression ratio vs full vectors (float32)
        full_vector_memory = self.dimension * 4  # 4 bytes per float32
        compression_ratio = full_vector_memory / memory_per_vector

        return {
            "total_vectors": self.total_vectors,
            "memory_per_vector": memory_per_vector,
            "total_memory": total_memory,
            "compression_ratio": compression_ratio,
            "full_vector_memory": full_vector_memory,
        }

    def save(self, filepath: str) -> None:
        """
        Save the FAISS index to disk.

        Args:
            filepath: Path where the index should be saved.

        Raises:
            RuntimeError: If saving fails.
        """
        try:
            faiss.write_index(self.index, filepath)
            logger.info(f"IndexIVFPQ saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save IndexIVFPQ to {filepath}: {e}")
            raise RuntimeError(f"Failed to save IndexIVFPQ: {e}")

    def load(self, filepath: str) -> None:
        """
        Load a FAISS index from disk.

        Args:
            filepath: Path to the saved index file.

        Raises:
            FileNotFoundError: If the index file doesn't exist.
            RuntimeError: If loading fails.
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"IndexIVFPQ file not found: {filepath}")

        try:
            self.index = faiss.read_index(filepath)
            self.dimension = self.index.d
            self.total_vectors = self.index.ntotal
            self.is_trained = True

            # Extract IVFPQ parameters from loaded index
            if hasattr(self.index, "nlist"):
                self.nlist = self.index.nlist
            if hasattr(self.index, "pq"):
                self.M = self.index.pq.M
                self.nbits = self.index.pq.nbits
            if hasattr(self.index, "nprobe"):
                self.nprobe = self.index.nprobe

            logger.info(
                f"IndexIVFPQ loaded from {filepath} with {self.total_vectors} vectors"
            )
        except Exception as e:
            logger.error(f"Failed to load IndexIVFPQ from {filepath}: {e}")
            raise RuntimeError(f"Failed to load IndexIVFPQ: {e}")


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
    the reranking phase. Includes histogram caching for improved performance.

    Attributes:
        db_path: Path to the DuckDB database file
        connection: Active DuckDB connection
        table_name: Name of the main metadata table
        _histogram_cache: LRU cache for frequently accessed histograms
        _cache_lock: Thread lock for cache operations
    """

    def __init__(self, db_path: str = ":memory:", cache_size: int = 1000):
        """
        Initialize the DuckDB metadata store.

        Args:
            db_path: Path to the DuckDB database file. Defaults to ":memory:"
                    for in-memory database (useful for testing).
            cache_size: Maximum number of histograms to cache in memory.
        """
        self.db_path = db_path
        self.table_name = "image_metadata"
        self.cache_size = cache_size

        # Initialize connection
        self.connection = duckdb.connect(db_path)

        # Initialize histogram cache with thread safety
        self._cache_lock = threading.Lock()
        self._histogram_cache: Dict[str, np.ndarray] = {}

        # Set up the database schema
        self.setup_table()

        logger.info(
            f"Initialized DuckDB metadata store at {db_path} with cache size {cache_size}"
        )

    def _get_histogram_from_cache(self, image_id: str) -> Optional[np.ndarray]:
        """
        Get histogram from cache if available.

        Args:
            image_id: Image identifier

        Returns:
            Cached histogram or None if not in cache
        """
        with self._cache_lock:
            return self._histogram_cache.get(image_id)

    def _add_histogram_to_cache(self, image_id: str, histogram: np.ndarray) -> None:
        """
        Add histogram to cache with LRU eviction.

        Args:
            image_id: Image identifier
            histogram: Histogram to cache
        """
        with self._cache_lock:
            # If cache is full, remove oldest entry (simple FIFO for now)
            if len(self._histogram_cache) >= self.cache_size:
                # Remove the first (oldest) entry
                oldest_key = next(iter(self._histogram_cache))
                del self._histogram_cache[oldest_key]
                logger.debug(f"Evicted histogram {oldest_key} from cache")

            self._histogram_cache[image_id] = histogram.copy()
            logger.debug(
                f"Added histogram {image_id} to cache (cache size: {len(self._histogram_cache)})"
            )

    def _load_histogram_from_db(self, image_id: str) -> Optional[np.ndarray]:
        """
        Load histogram from database.

        Args:
            image_id: Image identifier

        Returns:
            Histogram array or None if not found
        """
        query_sql = f"""
        SELECT histogram
        FROM {self.table_name}
        WHERE image_id = ?
        """

        try:
            result = self.connection.execute(query_sql, [image_id]).fetchone()
            if result:
                histogram_blob = result[0]
                histogram_array = np.frombuffer(histogram_blob, dtype=np.float32)
                return histogram_array
            else:
                return None
        except Exception as e:
            logger.error(f"Failed to load histogram for {image_id}: {e}")
            return None

    def get_histogram(self, image_id: str) -> Optional[np.ndarray]:
        """
        Get histogram for a single image ID with caching.

        Args:
            image_id: Image identifier

        Returns:
            Histogram array or None if not found
        """
        # Try cache first
        cached_histogram = self._get_histogram_from_cache(image_id)
        if cached_histogram is not None:
            logger.debug(f"Cache hit for histogram {image_id}")
            return cached_histogram

        # Load from database
        histogram = self._load_histogram_from_db(image_id)
        if histogram is not None:
            # Add to cache
            self._add_histogram_to_cache(image_id, histogram)
            logger.debug(f"Cache miss for histogram {image_id}, loaded from DB")
        else:
            logger.warning(f"Histogram not found for {image_id}")

        return histogram

    def setup_table(self) -> None:
        """
        Create the main metadata table if it doesn't exist.

        This method creates a table with the following schema:
        - image_id: Unique identifier for each image
        - file_path: Path to the image file
        - histogram: Raw color histogram as a JSON array (1152 dimensions)
        - file_size: Size of the image file in bytes
        - created_at: Timestamp when the record was created

        The histogram is stored as a JSON array to preserve the exact
        floating-point values and maintain compatibility with DuckDB's
        JSON type support.
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

    def add_batch(self, metadata_batch: List[Dict[str, Any]]) -> int:
        """
        Add multiple image metadata records in a single batch operation.

        This method efficiently inserts multiple image records using DuckDB's
        batch insertion capabilities. Each record should contain image_id,
        file_path, and histogram data. The histogram must be a 1D numpy array
        that will be converted to JSON for storage.

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

            insert_data.append(
                (
                    record["image_id"],
                    record["file_path"],
                    histogram_bytes,
                    record.get("file_size", 0),
                )
            )

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

    def get_histograms_by_ids(self, image_ids: List[str]) -> Dict[str, np.ndarray]:
        """
        Retrieve raw histograms for a list of image IDs with caching.

        This method is crucial for the reranking stage, as it provides
        the original, non-transformed histograms needed for accurate
        Sinkhorn-EMD distance calculations. The histograms are returned
        as a dictionary mapping image_id to numpy array.

        Uses intelligent caching to avoid repeated database queries for
        frequently accessed histograms.

        Args:
            image_ids: List of image IDs to retrieve histograms for.

        Returns:
            Dictionary mapping image_id to histogram numpy array.
            Only includes IDs that were found in the database or cache.

        Raises:
            ValueError: If image_ids is empty.
            RuntimeError: If database query fails.
        """
        if not image_ids:
            raise ValueError("Image IDs list cannot be empty")

        histograms = {}
        uncached_ids = []

        # Check cache first
        for image_id in image_ids:
            cached_histogram = self._get_histogram_from_cache(image_id)
            if cached_histogram is not None:
                histograms[image_id] = cached_histogram
                logger.debug(f"Cache hit for histogram {image_id}")
            else:
                uncached_ids.append(image_id)

        # Load uncached histograms from database
        if uncached_ids:
            logger.debug(
                f"Loading {len(uncached_ids)} uncached histograms from database"
            )

            # Create placeholders for the IN clause
            placeholders = ",".join(["?" for _ in uncached_ids])

            query_sql = f"""
            SELECT image_id, histogram
            FROM {self.table_name}
            WHERE image_id IN ({placeholders})
            """

            try:
                result = self.connection.execute(query_sql, uncached_ids).fetchall()

                # Convert results to dictionary and add to cache
                for image_id, histogram_blob in result:
                    # Convert BLOB back to numpy array
                    histogram_array = np.frombuffer(histogram_blob, dtype=np.float32)
                    histograms[image_id] = histogram_array

                    # Add to cache
                    self._add_histogram_to_cache(image_id, histogram_array)
                    logger.debug(f"Cache miss for histogram {image_id}, loaded from DB")

                logger.debug(
                    f"Retrieved {len(histograms)} histograms for {len(image_ids)} requested IDs "
                    f"({len(uncached_ids)} from DB, {len(image_ids) - len(uncached_ids)} from cache)"
                )

            except Exception as e:
                logger.error(f"Failed to retrieve histograms: {e}")
                raise RuntimeError(f"Database query failed: {e}")

        return histograms

    def get_all_histograms(self) -> Dict[str, np.ndarray]:
        """
        Retrieve all histograms from the metadata store.

        This method returns all histograms in the database, ordered by
        insertion order. This is useful for mapping FAISS indices to
        actual histograms during the reranking stage.

        Returns:
            Dictionary mapping image_id to histogram numpy array.
            Histograms are ordered by insertion order (matching FAISS indices).

        Raises:
            RuntimeError: If database query fails.
        """
        query_sql = f"""
        SELECT image_id, histogram
        FROM {self.table_name}
        ORDER BY ROWID
        """

        try:
            result = self.connection.execute(query_sql).fetchall()

            # Convert results to dictionary
            histograms = {}
            for image_id, histogram_blob in result:
                # Convert BLOB back to numpy array
                histogram_array = np.frombuffer(histogram_blob, dtype=np.float32)
                histograms[image_id] = histogram_array

            logger.debug(f"Retrieved {len(histograms)} total histograms")
            return histograms

        except Exception as e:
            logger.error(f"Failed to retrieve all histograms: {e}")
            raise RuntimeError(f"Database query failed: {e}")

    def get_image_count(self) -> int:
        """
        Get the total number of images in the metadata store.

        Returns:
            Total count of image records in the database.
        """

    def get_image_info(self, image_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve image information including file path and metadata.

        Args:
            image_id: Unique identifier for the image.

        Returns:
            Dictionary containing image information (image_id, file_path, file_size)
            or None if the image is not found.

        Raises:
            RuntimeError: If database query fails.
        """
        query_sql = f"""
        SELECT image_id, file_path, file_size
        FROM {self.table_name}
        WHERE image_id = ?
        """

        try:
            result = self.connection.execute(query_sql, [image_id]).fetchone()

            if result:
                return {
                    "image_id": result[0],
                    "file_path": result[1],
                    "file_size": result[2],
                }
            else:
                return None

        except Exception as e:
            logger.error(f"Failed to retrieve image info for {image_id}: {e}")
            raise RuntimeError(f"Database query failed: {e}")

    def get_image_metadata(self, image_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve image metadata by image ID.

        This method is an alias for get_image_info to maintain compatibility
        with the API interface.

        Args:
            image_id: Unique identifier for the image.

        Returns:
            Dictionary containing image metadata or None if not found.
        """
        return self.get_image_info(image_id)

    def get_image_count(self) -> int:
        """
        Get the total number of images in the metadata store.

        Returns:
            Total count of image records in the database.
        """
        try:
            result = self.connection.execute(
                f"SELECT COUNT(*) FROM {self.table_name}"
            ).fetchone()
            return result[0] if result else 0
        except Exception as e:
            logger.error(f"Failed to get image count: {e}")
            return 0

    def get_image_ids_in_order(self) -> List[str]:
        """
        Get all image IDs in insertion order (matching FAISS indices).

        This method returns image IDs in the same order they were inserted,
        which corresponds to the FAISS index order. This is crucial for
        mapping FAISS indices to actual image IDs.

        Returns:
            List of image IDs in insertion order.

        Raises:
            RuntimeError: If database query fails.
        """
        query_sql = f"""
        SELECT image_id
        FROM {self.table_name}
        ORDER BY ROWID
        """

        try:
            result = self.connection.execute(query_sql).fetchall()
            image_ids = [row[0] for row in result]

            logger.debug(f"Retrieved {len(image_ids)} image IDs in insertion order")
            return image_ids

        except Exception as e:
            logger.error(f"Failed to retrieve image IDs in order: {e}")
            raise RuntimeError(f"Database query failed: {e}")

    def get_all_image_metadata(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Get metadata for all images in the store.

        Args:
            limit: Maximum number of images to return (default: 1000).

        Returns:
            List of dictionaries containing image metadata.
        """
        try:
            query = f"""
                SELECT image_id, file_path, file_size
                FROM {self.table_name}
                LIMIT ?
            """
            result = self.connection.execute(query, (limit,)).fetchall()

            # Convert to list of dictionaries
            images = []
            for row in result:
                images.append(
                    {"image_id": row[0], "file_path": row[1], "file_size": row[2]}
                )

            logger.info(f"Retrieved {len(images)} image metadata records")
            return images

        except Exception as e:
            logger.error(f"Failed to get all image metadata: {e}")
            return []

    def close(self) -> None:
        """
        Close the database connection.

        This method should be called when the MetadataStore is no longer needed
        to properly clean up database resources.
        """
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")

    def __enter__(self):
        """Context manager entry point."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point."""
        self.close()
