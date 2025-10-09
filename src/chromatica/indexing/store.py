"""
FAISS and DuckDB wrapper classes for the Chromatica color search engine.

(Docstring truncated for brevity)
"""

import logging
import numpy as np
import faiss
import duckdb
from typing import List, Tuple, Optional, Dict, Any, Set
from pathlib import Path
from functools import lru_cache
import threading
import os
from typing import Union

# Assuming these are available in your environment
try:
    from ..utils.config import (
        TOTAL_BINS,
        IVFPQ_NLIST,
        IVFPQ_M,
        IVFPQ_NBITS,
        IVFPQ_NPROBE,
    )
except ImportError:
    # Placeholder values if config is not available for testing
    TOTAL_BINS = 1152
    IVFPQ_NLIST = 16384
    IVFPQ_M = 32
    IVFPQ_NBITS = 8
    IVFPQ_NPROBE = 1

# Configure logging for this module
logger = logging.getLogger(__name__)


def hellinger_transform(vectors: np.ndarray) -> np.ndarray:
    """
    Apply the Hellinger transform: f(x) = sqrt(x).

    This transform is crucial for making the L2 distance metric
    (used by FAISS) a good approximation of the Hellinger distance
    (suitable for histograms).

    Args:
        vectors: A numpy array of histogram vectors (assumed to be L1 normalized).

    Returns:
        The Hellinger-transformed vectors.
    """
    # Ensure no negative values before taking sqrt
    vectors[vectors < 0] = 0
    return np.sqrt(vectors)


class AnnIndex:
    """
    Wrapper for FAISS index used in Chromatica.
    Supports HNSW, Flat, and IVFPQ indices.
    """

    def __init__(
        self,
        dimension: int = TOTAL_BINS,
        use_simple_index: bool = True,  # True for HNSW/Flat, False for IVFPQ
        index_path: Optional[str] = None,
        # HNSW parameters
        M: int = 32,
        # IVFPQ parameters (only used if use_simple_index is False)
        nlist: int = IVFPQ_NLIST,
        nprobe: int = IVFPQ_NPROBE,
        M_pq: int = IVFPQ_M,
        nbits: int = IVFPQ_NBITS,
    ):
        """
        Initialize the FAISS index, either by loading an existing one or creating a new one.
        """
        self.dimension = dimension
        self.index: Optional[faiss.Index] = None
        self.index_path = index_path
        self.M = M
        self.nlist = nlist
        self.nprobe = nprobe
        self.M_pq = M_pq
        self.nbits = nbits
        self.is_trained = False

        if index_path and Path(index_path).exists():
            # Load existing index
            try:
                self.index = faiss.read_index(index_path)
                self.is_trained = self.index.is_trained
                logger.info(
                    f"Loaded existing FAISS index from {index_path} with "
                    f"{self.index.ntotal} vectors. Trained: {self.is_trained}"
                )
            except Exception as e:
                logger.error(f"Failed to load FAISS index from {index_path}: {e}")
                self._create_new_index(use_simple_index)
        else:
            # Create a new index
            self._create_new_index(use_simple_index)

    def _create_new_index(self, use_simple_index: bool):
        """Create a new FAISS index based on configuration."""
        if use_simple_index:
            # HNSW index for smaller/moderate datasets (high recall, higher memory)
            self.index = faiss.IndexHNSWFlat(self.dimension, self.M)
            logger.info(f"Created new FAISS HNSW index (M={self.M})")
            # HNSW does not require explicit training
            self.is_trained = True
        else:
            # IVFPQ index for very large datasets (lower memory, requires training)
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFPQ(
                quantizer, self.dimension, self.nlist, self.M_pq, self.nbits
            )
            self.index.nprobe = self.nprobe
            self.is_trained = False
            logger.info(
                f"Created new FAISS IndexIVFPQ (nlist={self.nlist}, M={self.M_pq}, nbits={self.nbits}). Needs training."
            )

    def train(self, training_vectors: np.ndarray) -> None:
        """
        Train the index (required for IVFPQ).
        """
        if self.index and not self.index.is_trained:
            logger.info(
                f"Starting FAISS training with {len(training_vectors)} vectors..."
            )
            # Apply Hellinger transform before training
            transformed_vectors = hellinger_transform(training_vectors)
            self.index.train(transformed_vectors)
            self.is_trained = True
            logger.info("FAISS training complete.")

    def add(self, vectors: np.ndarray) -> None:
        """
        Add Hellinger-transformed vectors to the FAISS index.
        Also assigns sequential IDs starting from 1.
        """
        if self.index is None:
            raise RuntimeError("FAISS index is not initialized.")

        if not self.index.is_trained and not isinstance(
            self.index, faiss.IndexHNSWFlat
        ):
            raise RuntimeError("FAISS index must be trained before adding vectors.")

        # Apply Hellinger transform
        transformed_vectors = hellinger_transform(vectors)

        # Get current count and assign sequential IDs
        start_id = self.index.ntotal + 1
        ids = np.arange(start_id, start_id + len(vectors), dtype=np.int64)

        # Add to the index with explicit IDs
        self.index.add_with_ids(transformed_vectors, ids)
        logger.debug(
            f"Added {len(vectors)} vectors with IDs {start_id}-{start_id+len(vectors)-1}"
        )

    def search(self, query_vector: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search the FAISS index for the k nearest neighbors.

        Args:
            query_vector: The L1-normalized histogram query vector.
            k: The number of nearest neighbors to retrieve.

        Returns:
            A tuple (distances, indices) of the k nearest neighbors.
        """
        if self.index is None:
            raise RuntimeError("FAISS index is not initialized.")

        # CRITICAL CHECK: The source of the user's previous 'k <= 0' error
        if self.index.ntotal == 0:
            raise ValueError(
                f"FAISS index is empty (0 vectors). Cannot perform search."
            )
        if k <= 0 or k > self.index.ntotal:
            # This is the actual error from FAISS
            raise ValueError(
                f"k must be positive and <= {self.index.ntotal}, got {k}. "
                "Ensure k is configured correctly for the total index size."
            )

        # Apply Hellinger transform to the query
        transformed_query = hellinger_transform(query_vector.reshape(1, -1))

        # Perform the search
        distances, indices = self.index.search(transformed_query, k)

        return distances.flatten(), indices.flatten()

    def save(self, index_path: str) -> None:
        """
        Save the FAISS index to disk.
        """
        if self.index is None:
            raise RuntimeError("FAISS index is not initialized.")

        if not self.index.is_trained:
            logger.warning(
                "Saving an untrained index. This is only valid for HNSW/Flat."
            )

        faiss.write_index(self.index, index_path)
        logger.info(f"FAISS index saved to {index_path}")

    # --- CRITICAL ADDITION ---
    def get_total_vectors(self) -> int:
        """
        Returns the total number of vectors in the FAISS index.
        """
        if self.index:
            return self.index.ntotal
        return 0

    # -------------------------

    def __len__(self):
        """Returns the number of vectors in the index."""
        return self.get_total_vectors()


class MetadataStore:
    """
    Wrapper for DuckDB metadata store used in Chromatica.
    Handles thread-safe connections for API use.
    """

    def __init__(
        self, db_path: str, table_name: str = "histograms"
    ):  # Changed default name
        self.db_path = db_path
        self.table_name = table_name

        # Create table with temp connection
        temp_conn = duckdb.connect(database=self.db_path, read_only=False)
        self.create_table_if_not_exists(temp_conn)
        temp_conn.close()

        logger.info(f"MetadataStore initialized with DB: {db_path}")

    def _get_thread_connection(self):
        """Get a thread-local connection to DuckDB."""
        # Check if a connection is already stored in thread-local storage
        if not hasattr(threading.current_thread(), "duckdb_conn"):
            # If not, create a new connection
            conn = duckdb.connect(database=self.db_path, read_only=False)
            threading.current_thread().duckdb_conn = conn
        return threading.current_thread().duckdb_conn

    def create_table_if_not_exists(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Create the histograms table if it does not already exist."""
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                image_id VARCHAR PRIMARY KEY,
                histogram BLOB NOT NULL,
                file_path VARCHAR,
                file_size BIGINT
            )
            """
        )
        conn.commit()
        logger.info(f"Table '{self.table_name}' checked/created with updated schema.")

    def _serialize_histogram(self, histogram: np.ndarray) -> bytes:
        """
        Convert a numpy array (histogram) to a binary blob for DuckDB storage.
        Ensures consistent dtype and serialization.
        """
        # Ensure the histogram is float32 and verify shape
        if histogram.dtype != np.float32:
            histogram = histogram.astype(np.float32)
        if len(histogram) != TOTAL_BINS:
            raise ValueError(
                f"Invalid histogram size: {len(histogram)} != {TOTAL_BINS}"
            )
        return histogram.tobytes()

    def _deserialize_histogram(self, blob: bytes) -> np.ndarray:
        """
        Convert a binary blob from DuckDB back into a numpy array.
        Ensures proper shape and dtype restoration.
        """
        # Convert binary blob back to numpy array with correct shape and dtype
        histogram = np.frombuffer(blob, dtype=np.float32)
        if len(histogram) != TOTAL_BINS:
            raise ValueError(
                f"Invalid histogram size: {len(histogram)} != {TOTAL_BINS}"
            )
        return histogram

    def add_batch(self, batch_data: List[Dict[str, Any]]) -> int:
        """Add a batch of image metadata to the store."""
        if not batch_data:
            return 0

        try:
            insert_data = []
            for record in batch_data:
                histogram_blob = self._serialize_histogram(record["histogram"])

                # Use sequential integer IDs matching FAISS
                image_id = f"{int(record['image_id']):05d}"

                insert_data.append(
                    (
                        image_id,
                        histogram_blob,
                        record.get("file_path"),
                        record.get("file_size"),
                    )
                )

            # Log sample of IDs being inserted
            sample_insert = [data[0] for data in insert_data[:5]]
            logger.debug(f"Inserting with IDs: {sample_insert}")

            query = f"""
                INSERT OR REPLACE INTO {self.table_name} 
                (image_id, histogram, file_path, file_size)
                VALUES (?, ?, ?, ?)
            """

            conn = self._get_thread_connection()
            conn.executemany(query, insert_data)
            conn.commit()

            count = conn.execute(f"SELECT COUNT(*) FROM {self.table_name}").fetchone()[
                0
            ]
            logger.info(f"Total records after batch insert: {count}")

            return len(insert_data)

        except Exception as e:
            logger.error(f"Error adding batch to metadata store: {e}")
            return 0

    def check_existence(self, image_ids: List[str]) -> Set[str]:
        """
        Check which image_ids already exist in the database.
        """
        if not image_ids:
            return set()

        conn = self._get_thread_connection()

        # DuckDB uses UNNEST for checking against a list of values
        try:
            query = f"""
                SELECT image_id 
                FROM {self.table_name} 
                WHERE image_id IN (UNNEST(?))
            """
            result = conn.execute(query, (image_ids,)).fetchall()
            return {row[0] for row in result}
        except Exception as e:
            logger.error(f"Failed to check image existence: {e}")
            return set()

    def get_histogram(self, image_id: Union[str, float, int]) -> Optional[np.ndarray]:
        """Retrieve the raw histogram for a given image ID."""
        try:
            # Convert float/decimal IDs to integers, preserving original ID
            if isinstance(image_id, (float, np.float32, np.float64)):
                # Round to nearest integer since FAISS IDs start from 1
                query_id = f"{max(1, round(float(image_id))):05d}"
            else:
                query_id = f"{int(str(image_id)):05d}"

            logger.debug(
                f"Looking up histogram with ID: {query_id} (original: {image_id})"
            )

            query = f"""
                SELECT histogram 
                FROM {self.table_name}
                WHERE image_id = ?
                LIMIT 1
            """
            conn = self._get_thread_connection()
            result = conn.execute(query, [query_id]).fetchone()

            if result is None:
                logger.warning(
                    f"No histogram found for image_id: {image_id} (query_id: {query_id})"
                )
                return None

            return self._deserialize_histogram(result[0])

        except Exception as e:
            logger.error(f"Error retrieving histogram for {image_id}: {e}")
            return None

    def get_metadata_by_ids(self, image_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve metadata (including histogram) for a list of image IDs.
        """
        if not image_ids:
            return []

        conn = self._get_thread_connection()
        try:
            # Note: Select ALL columns
            query = f"""
                SELECT image_id, file_path, file_size, histogram 
                FROM {self.table_name} 
                WHERE image_id IN (UNNEST(?))
            """
            result = conn.execute(query, (image_ids,)).fetchall()

            metadata_list = []
            # Map column names to indices for clarity
            col_map = {"image_id": 0, "file_path": 1, "file_size": 2, "histogram": 3}

            for row in result:
                metadata_list.append(
                    {
                        "image_id": row[col_map["image_id"]],
                        "file_path": row[col_map["file_path"]],
                        "file_size": row[col_map["file_size"]],
                        "histogram": self._deserialize_histogram(
                            row[col_map["histogram"]]
                        ),
                    }
                )

            return metadata_list

        except Exception as e:
            logger.error(f"Failed to retrieve metadata by IDs: {e}")
            return []

    def get_all_image_metadata(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Retrieve simple metadata (no histogram) for all images in the store.
        """
        try:
            query = f"""
                SELECT image_id, file_path, file_size
                FROM {self.table_name}
                LIMIT ?
            """
            conn = self._get_thread_connection()
            result = conn.execute(query, (limit,)).fetchall()

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
        Close the database connection for the current thread.

        This method should be called when the MetadataStore is no longer needed
        to properly clean up database resources.
        """
        # Connection is thread-local, so we close the current thread's connection
        if hasattr(threading.current_thread(), "duckdb_conn"):
            threading.current_thread().duckdb_conn.close()
            del threading.current_thread().duckdb_conn
            logger.info("Database connection closed for current thread")

    def __enter__(self):
        """Context manager entry point."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point."""
        self.close()

    def get_image_count(self) -> int:
        """
        Get the total number of images in the metadata store.

        Returns:
            int: Total number of images in the database
        """
        try:
            query = f"""
                SELECT COUNT(*) 
                FROM {self.table_name}
            """
            conn = self._get_thread_connection()
            result = conn.execute(query).fetchone()

            count = result[0] if result else 0
            logger.debug(f"Total images in metadata store: {count}")
            return count

        except Exception as e:
            logger.error(f"Failed to get image count: {e}")
            return 0

    def check_stored_ids(self, limit: int = 10) -> None:
        """
        Debug method to check IDs stored in DuckDB.

        Args:
            limit: Number of IDs to sample (default 10)
        """
        try:
            query = f"""
                SELECT image_id 
                FROM {self.table_name} 
                LIMIT ?
            """
            conn = self._get_thread_connection()
            results = conn.execute(query, [limit]).fetchall()
            stored_ids = [row[0] for row in results]
            logger.info(f"Sample of stored IDs in DuckDB: {stored_ids}")

            # Also get count
            count_query = f"SELECT COUNT(*) FROM {self.table_name}"
            total = conn.execute(count_query).fetchone()[0]
            logger.info(f"Total records in DuckDB: {total}")

        except Exception as e:
            logger.error(f"Error checking stored IDs: {e}")

    def get_image_info(self, image_id: str) -> Optional[Dict[str, Any]]:
        """
        Get image information including file path and metadata.

        Args:
            image_id: The unique identifier for the image

        Returns:
            Dictionary with image information or None if not found
        """
        try:
            query = f"""
                SELECT image_id, file_path, file_size
                FROM {self.table_name}
                WHERE image_id = ?
                LIMIT 1
            """
            conn = self._get_thread_connection()
            result = conn.execute(query, [str(image_id)]).fetchone()

            if result is None:
                logger.warning(f"No image info found for image_id: {image_id}")
                return None

            return {
                "image_id": result[0],
                "file_path": result[1],
                "file_size": result[2],
            }

        except Exception as e:
            logger.error(f"Error retrieving image info for {image_id}: {e}")
            return None

    def get_image_path(self, image_id: Union[str, float, int]) -> Optional[str]:
        """
        Get the file path for a given image ID.

        Args:
            image_id: The unique identifier for the image

        Returns:
            str: The file path if found, None otherwise
        """
        try:
            # Format the image_id consistently
            if isinstance(image_id, (float, np.float32, np.float64)):
                query_id = f"{int(float(str(image_id))):05d}"
            else:
                query_id = f"{int(str(image_id)):05d}"

            query = f"""
                SELECT file_path
                FROM {self.table_name}
                WHERE image_id = ?
                LIMIT 1
            """
            conn = self._get_thread_connection()
            result = conn.execute(query, [query_id]).fetchone()

            if result is None:
                logger.warning(
                    f"No file path found for image_id: {image_id} (query_id: {query_id})"
                )
                return None

            return str(result[0])

        except Exception as e:
            logger.error(f"Error retrieving file path for {image_id}: {e}")
            return None
