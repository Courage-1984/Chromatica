"""
FAISS and DuckDB wrapper classes for the Chromatica color search engine.

This module provides high-level abstractions for managing the FAISS HNSW index
and DuckDB metadata store. The AnnIndex class handles vector indexing with
automatic Hellinger transformation, while the MetadataStore class manages
image metadata and raw histogram storage for the reranking stage.

Key Components:
- AnnIndex: Wraps faiss.IndexHNSWFlat with Hellinger transform
- MetadataStore: Manages DuckDB connection and operations
- Batch operations for efficient indexing and retrieval

The Hellinger transform is applied automatically to make histograms compatible
with the L2-based FAISS index, while raw histograms are preserved in DuckDB
for high-fidelity Sinkhorn-EMD reranking.
"""

import logging
import numpy as np
import faiss
import duckdb
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

from ..utils.config import TOTAL_BINS, HNSW_M

# Configure logging for this module
logger = logging.getLogger(__name__)


class AnnIndex:
    """
    Wrapper class for FAISS HNSW index with automatic Hellinger transformation.

    This class manages a FAISS Hierarchical Navigable Small World (HNSW) index
    that stores color histograms transformed using the Hellinger transform.
    The Hellinger transform (element-wise square root) converts normalized
    histograms into vectors compatible with L2 distance metrics used by FAISS.

    The HNSW index provides fast approximate nearest neighbor search with
    excellent accuracy-to-speed trade-offs, making it ideal for the first
    stage of our two-stage search pipeline.

    Attributes:
        index: The underlying FAISS HNSW index
        dimension: The dimensionality of the indexed vectors (1152 for Lab histograms)
        total_vectors: The total number of vectors currently indexed
        is_trained: Whether the index has been trained (HNSW doesn't require training)
    """

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

        logger.info(
            f"Initialized FAISS HNSW index with dimension {dimension}, M={HNSW_M}"
        )

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
                f"FAISS search completed: k={k}, min_distance={distances.min():.4f}, max_distance={distances.max():.4f}"
            )
            return distances, indices

        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            raise RuntimeError(f"FAISS search failed: {e}")

    def get_total_vectors(self) -> int:
        """
        Get the total number of vectors currently indexed.

        Returns:
            The total number of vectors in the index.
        """
        return self.total_vectors

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
            logger.info(f"FAISS index saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save FAISS index to {filepath}: {e}")
            raise RuntimeError(f"Failed to save FAISS index: {e}")

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

    Attributes:
        db_path: Path to the DuckDB database file
        connection: Active DuckDB connection
        table_name: Name of the main metadata table
    """

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
                    "file_size": result[2]
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
