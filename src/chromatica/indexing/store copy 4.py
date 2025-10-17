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
import json
import re

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

    (Docstring truncated for brevity)
    """
    # Ensure non-negative and handle potential division by zero if normalized
    return np.sqrt(np.maximum(vectors, 0))


# --- AnnIndex (FAISS HNSW Wrapper) (No changes needed based on prompt) ---
class AnnIndex:
    """
    Wrapper around the FAISS HNSW index for Approximate Nearest Neighbor search.
    """

    def __init__(self, index_path: Union[str, Path], dim: int = TOTAL_BINS):
        self.index_path = Path(index_path)
        self.dim = dim
        self.index: Optional[faiss.Index] = None
        self._lock = threading.Lock()
        self.load()

    def _ensure_index_exists(self):
        if self.index is None:
            # Fallback for empty index - create a dummy one
            self.index = faiss.IndexFlatL2(self.dim)
            logger.warning(
                "FAISS index was not loaded or created; initialized with IndexFlatL2."
            )

    def load(self):
        """Loads the FAISS index from disk or initializes an empty one."""
        with self._lock:
            if self.index_path.exists():
                logger.info(f"Loading FAISS index from {self.index_path}...")
                try:
                    self.index = faiss.read_index(str(self.index_path))
                    logger.info(
                        f"FAISS index loaded successfully. Size: {self.index.ntotal}, Dim: {self.index.d}"
                    )
                except Exception as e:
                    logger.error(f"Failed to load FAISS index: {e}")
                    self.index = None
            
            self._ensure_index_exists()

    def save(self):
        """Saves the FAISS index to disk."""
        with self._lock:
            if self.index is not None and self.index.ntotal > 0:
                try:
                    faiss.write_index(self.index, str(self.index_path))
                    logger.info(f"FAISS index saved to {self.index_path}.")
                except Exception as e:
                    logger.error(f"Failed to save FAISS index: {e}")
            elif self.index is not None:
                logger.warning("FAISS index is empty (0 vectors), skipping save.")
            else:
                logger.error("FAISS index object is None, cannot save.")

    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Performs k-nearest neighbor search."""
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Search called on an empty index. Returning empty results.")
            return np.array([]), np.array([])
        
        # Ensure nprobe is set if it's an Ivf index
        if isinstance(self.index, faiss.IndexIVF):
            try:
                # The prompt context suggests IVFPQ_NPROBE should be available
                self.index.nprobe = IVFPQ_NPROBE
            except NameError:
                # Default if config isn't available
                self.index.nprobe = 1 

        with self._lock:
            # D = distances, I = indices
            distances, indices = self.index.search(queries, k)
        return distances, indices

    def add_with_ids(self, vectors: np.ndarray, ids: np.ndarray):
        """Adds vectors with explicit IDs to the index."""
        with self._lock:
            if self.index is None:
                self._ensure_index_exists()

            # Ensure the index is capable of adding with IDs
            if not isinstance(self.index, faiss.IndexIDMap):
                logger.error("Index is not an IndexIDMap, cannot add with explicit IDs.")
                return

            if vectors.shape[1] != self.dim:
                raise ValueError(
                    f"Vector dimension mismatch: {vectors.shape[1]} != {self.dim}"
                )

            # Check for existing IDs and remove them first for a true update/replace
            existing_ids = set(self.get_ids().tolist())
            ids_to_remove = [id_ for id_ in ids if id_ in existing_ids]
            if ids_to_remove:
                logger.info(f"Removing {len(ids_to_remove)} existing IDs before adding.")
                id_map = faiss.IDSelectorBatch(ids_to_remove)
                self.index.remove_ids(id_map)
            
            # Add the new/updated vectors
            self.index.add_with_ids(vectors, ids)
            logger.info(
                f"Added {vectors.shape[0]} vectors to FAISS index. Total: {self.index.ntotal}"
            )
            
    def get_ids(self) -> np.ndarray:
        """Retrieves all image IDs currently in the index."""
        if self.index is None:
            return np.array([])
        
        # This assumes the index is an IndexIDMap. 
        # A proper implementation might need to handle IndexFlat types differently.
        if isinstance(self.index, faiss.IndexIDMap):
            return self.index.id_map.cpu().as_array()
        
        # Fallback for non-IDMap index (e.g., IndexFlatL2)
        logger.warning("Index is not IndexIDMap; returning sequential IDs.")
        return np.arange(self.index.ntotal)


# --- MetadataStore (DuckDB Wrapper) ---
class MetadataStore:
    """
    Wrapper around DuckDB for storing image metadata (histograms, paths, etc.).
    """

    def __init__(self, db_path: Path, table_name: str = "images"):
        """Initializes the MetadataStore (DuckDB connection)."""
        self.db_path = db_path
        self.table_name = table_name
        # Connect to DuckDB
        # We ensure the path exists as DuckDB will create the file if it doesn't
        self.connection = duckdb.connect(database=str(db_path), read_only=False)
        self.lock = threading.Lock()
        self._check_schema()  # CRITICAL: Check schema on init

    def _check_schema(self):
        """Checks if the 'metadata' column exists in the images table."""
        try:
            # Query the table schema using PRAGMA
            query = f"PRAGMA table_info('{self.table_name}')"
            columns = self.connection.execute(query).fetchall()
            column_names = {col[1] for col in columns}
            self.has_metadata_column = "metadata" in column_names
            
            if not self.has_metadata_column:
                logger.warning(
                    f"Table '{self.table_name}' is missing the 'metadata' column. "
                    "Using legacy query for compatibility. Please rebuild index for full functionality."
                )
            else:
                logger.info(f"Table '{self.table_name}' contains the 'metadata' column.")
        except Exception as e:
            logger.error(f"Failed to check database schema: {e}")
            self.has_metadata_column = False # Assume worst case if check fails

    def initialize_db(self):
        """Initializes the necessary table if it doesn't exist."""
        # Use LOCK for write operations if running in a threaded environment
        with self.lock:
            # The 'metadata' column is added here for NEW indices.
            query = f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    image_id VARCHAR PRIMARY KEY,
                    file_path VARCHAR NOT NULL,
                    file_size BIGINT,
                    histogram BLOB,
                    dominant_colors VARCHAR,
                    metadata VARCHAR
                )
            """
            self.connection.execute(query)
            logger.info(f"DuckDB table '{self.table_name}' ensured.")


    @lru_cache(maxsize=1024)
    def get_image_info(self, image_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves essential information (file_path, file_size, metadata) for a given image_id.

        This method is made schema-robust to handle indices built before the
        'metadata' column was introduced, preventing the reported Binder Error.
        """
        with self.lock:
            try:
                # Use conditional query based on schema check
                if self.has_metadata_column:
                    # Full query for new indices
                    query = f"""
                        SELECT image_id, file_path, file_size, metadata
                        FROM {self.table_name}
                        WHERE image_id = ?
                    """
                    result = self.connection.execute(query, (image_id,)).fetchone()
                    
                    if result:
                        image_info = {
                            "image_id": result[0],
                            "file_path": result[1],
                            "file_size": result[2],
                            "metadata": result[3] # result[3] is 'metadata'
                        }
                    else:
                        return None
                else:
                    # Legacy query for old indices (excludes 'metadata' column)
                    query = f"""
                        SELECT image_id, file_path, file_size
                        FROM {self.table_name}
                        WHERE image_id = ?
                    """
                    result = self.connection.execute(query, (image_id,)).fetchone()

                    if result:
                        image_info = {
                            "image_id": result[0],
                            "file_path": result[1],
                            "file_size": result[2],
                            "metadata": None # Manually add the missing key as None
                        }
                    else:
                        return None

                return image_info

            except Exception as e:
                logger.error(f"Error retrieving image info for {image_id} (Schema issue handled): {e}")
                return None


    def add_image(
        self,
        image_id: str,
        file_path: str,
        file_size: int,
        histogram: np.ndarray,
        dominant_colors: str,
        metadata: Optional[Union[str, Dict[str, Any]]] = None,
    ):
        """Adds or updates an image record, including its histogram."""
        # Convert histogram to bytes
        histogram_bytes = histogram.tobytes()
        
        # Convert metadata dict to JSON string if it's not already a string
        if isinstance(metadata, dict):
            metadata_str = json.dumps(metadata)
        elif metadata is None:
            metadata_str = None
        else:
            metadata_str = metadata

        with self.lock:
            # Use 'REPLACE' to handle both insert and update
            # We assume the table schema includes 'metadata' for new inserts.
            query = f"""
                INSERT OR REPLACE INTO {self.table_name} 
                (image_id, file_path, file_size, histogram, dominant_colors, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """
            self.connection.execute(
                query,
                (
                    image_id,
                    file_path,
                    file_size,
                    histogram_bytes,
                    dominant_colors,
                    metadata_str,
                ),
            )
            # Clear cache for the added/updated image
            self.get_image_info.cache_clear()

    def _connect(self):
        """Initializes or re-establishes the connection to DuckDB."""
        with self._lock:
            if self.connection:
                self.connection.close()
            # connect(read_only=False) is the default
            self.connection = duckdb.connect(database=str(self.db_path))
            self.connection.sql(f"SET search_path = '{self.table_name}'")

    def create_table(self, overwrite: bool = False, is_url_index: bool = False):
        """Creates the metadata table with the correct schema."""
        with self._lock:
            if overwrite or not self.table_exists():
                logger.info(f"Creating table '{self.table_name}' in {self.db_path}...")
                # The path column uses a dynamic name based on index type
                path_col_name = "image_url" if is_url_index else "file_path"

                # The standard schema
                schema = (
                    f"image_id VARCHAR PRIMARY KEY, "
                    f"{path_col_name} VARCHAR, "
                    "file_size BIGINT, "
                    "histogram BLOB, "
                    "metadata VARCHAR"  # Storing JSON metadata as string
                )

                # Check for existing table and drop if overwrite is true
                if self.table_exists():
                    logger.warning(f"Table '{self.table_name}' exists.")
                    if overwrite:
                        self.connection.execute(f"DROP TABLE {self.table_name}")
                        logger.info(f"Dropped existing table '{self.table_name}'.")
                    else:
                        logger.info("Skipping table creation (overwrite=False).")
                        return

                self.connection.execute(f"CREATE TABLE {self.table_name} ({schema})")
                logger.info(f"Table '{self.table_name}' created successfully.")
            else:
                logger.info(f"Table '{self.table_name}' already exists. Skipping creation.")

class MetadataStore:
    """
    Wrapper for DuckDB metadata store used in Chromatica.
    Handles thread-safe connections for API use.
    """

    def __init__(self, db_path: str, table_name: str = "histograms"):
        self.db_path = db_path
        self.table_name = table_name
        self._connections = {}  # Thread-local connections dictionary
        self._create_connection_if_not_exists()  # Ensure connection exists on initialization

        # Create table with initial connection
        self.create_table_if_not_exists(self._get_thread_connection())

        logger.info(f"MetadataStore initialized with DB: {db_path}")

    def _create_connection_if_not_exists(self):
        """Ensure a database connection exists for the current thread."""
        thread_id = threading.get_ident()
        if thread_id not in self._connections:
            self._connections[thread_id] = duckdb.connect(
                database=self.db_path, read_only=False
            )
        return self._connections[thread_id]

    def _get_thread_connection(self):
        """Get a thread-local connection to DuckDB."""
        return self._create_connection_if_not_exists()

    # For backward compatibility - property that returns the main connection
    @property
    def connection(self):
        """Legacy compatibility property for code that accessed .connection directly"""
        return self._get_thread_connection()

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
                # Check if histogram is already bytes or needs conversion
                if isinstance(record["histogram"], bytes):
                    histogram_blob = record["histogram"]
                else:
                    histogram_blob = self._serialize_histogram(record["histogram"])

                # Use sequential integer IDs matching FAISS
                image_id = (
                    f"{int(record['image_id']):05d}"
                    if record["image_id"].isdigit()
                    else record["image_id"]
                )

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

    def get_histogram(
        self, image_id: str, query_id: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """
        Retrieve the raw histogram for an image by ID.

        This method has been enhanced to support both numeric IDs and URL-based IDs
        from the URL-based indexing workflow.

        Args:
            image_id: Unique identifier for the image
            query_id: Optional query ID for logging

        Returns:
            The raw histogram as a numpy array, or None if not found
        """
        try:
            conn = self._get_thread_connection()

            # Try to get the histogram directly
            query = f"""
                SELECT histogram FROM {self.table_name}
                WHERE image_id = ?
            """

            result = conn.execute(query, [image_id]).fetchone()

            if result is None:
                logger.warning(
                    f"No histogram found for image_id: {image_id} (query_id: {query_id})"
                )
                return None

            histogram_blob = result[0]
            return self._deserialize_histogram(histogram_blob)

        except Exception as e:
            logger.warning(
                f"Error getting histogram for image_id: {image_id} (query_id: {query_id}): {e}"
            )
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
        Get image metadata for the specified image ID.

        Enhanced to support both traditional file-based and URL-based indexes.

        Args:
            image_id: Unique identifier for the image

        Returns:
            Dictionary with image metadata or None if not found
        """
        try:
            conn = self._get_thread_connection()

            # Query for both standard metadata and JSON metadata field
            query = f"""
                SELECT image_id, file_path, file_size, metadata 
                FROM {self.table_name}
                WHERE image_id = ?
            """

            result = conn.execute(query, [image_id]).fetchone()

            if result is None:
                logger.warning(f"No image info found for image_id: {image_id}")
                return None

            # Basic metadata
            info = {
                "image_id": result[0],
                "file_path": result[1],  # This could be a URL in URL-based indexes
                "file_size": result[2] if result[2] is not None else 0,
            }

            # Try to parse additional JSON metadata if available
            if len(result) > 3 and result[3] is not None:
                try:
                    json_metadata = json.loads(result[3])
                    if isinstance(json_metadata, dict):
                        # Add JSON metadata fields, but don't overwrite existing fields
                        for k, v in json_metadata.items():
                            if k not in info:
                                info[k] = v
                except (json.JSONDecodeError, TypeError) as e:
                    logger.debug(
                        f"Failed to parse metadata JSON for image_id {image_id}: {e}"
                    )

            return info

        except Exception as e:
            logger.warning(f"Error retrieving image info for {image_id}: {e}")
            return None

    def get_image_path(self, image_id: str) -> Optional[str]:
        """
        Get image file path from the metadata store.

        For URL-based indices, this returns the URL stored in file_path.

        Args:
            image_id: Image identifier (could be numeric ID or URL)

        Returns:
            File path or URL string, or None if not found
        """
        try:
            # First try direct lookup by image_id
            query = f"""
                SELECT file_path FROM {self.table_name}
                WHERE image_id = ?
                LIMIT 1
            """
            result = self.connection.execute(query, [image_id]).fetchone()

            # If not found, try looking up by metadata.original_id
            if result is None:
                query = f"""
                    SELECT file_path FROM {self.table_name}
                    WHERE json_extract(metadata, '$.original_id') = ?
                    LIMIT 1
                """
                result = self.connection.execute(query, [image_id]).fetchone()

            # If still not found and image_id looks like a filename pattern (e.g., '00845'),
            # try looking up by filename pattern in the file_path or metadata
            if result is None and re.match(r"^\d+$", image_id):
                # Try to match against filenames in the file_path column
                query = f"""
                    SELECT file_path FROM {self.table_name}
                    WHERE file_path LIKE ? OR json_extract(metadata, '$.local_filename') LIKE ?
                    LIMIT 1
                """
                pattern = f"%{image_id}%"
                result = self.connection.execute(query, [pattern, pattern]).fetchone()

            if result is None or not result[0]:
                return None

            return result[0]
        except Exception as e:
            logger.error(f"Error getting image path for image_id: {image_id}: {e}")
            return None

    def get_image_url(self, image_id: str) -> Optional[str]:
        """
        Get image URL for the specified image ID.

        This method has been enhanced to:
        1. Return file_path directly if it looks like a URL (URL-based indexes)
        2. Fall back to looking for image_url in the metadata
        3. Return the file_path as a fallback

        Args:
            image_id: Unique identifier for the image

        Returns:
            URL to the image or None if not found
        """
        try:
            info = self.get_image_info(image_id)
            if not info:
                return None

            # Case 1: If file_path looks like a URL, use it directly
            file_path = info.get("file_path")
            if file_path and (
                file_path.startswith("http://") or file_path.startswith("https://")
            ):
                return file_path

            # Case 2: Check for image_url in metadata
            if "image_url" in info:
                return info["image_url"]

            # Case 3: Fall back to file_path (could be a local file)
            return file_path

        except Exception as e:
            logger.warning(f"Error getting image URL for {image_id}: {e}")
            return None


class DuckDBStore:
    def __init__(
        self, db_path: Path, table_name: str = "image_metadata", append: bool = False
    ):
        """
        Initialize DuckDB store for image metadata and histograms.

        Args:
            db_path: Path to the DuckDB database file
            table_name: Name of the table to create/use
            append: If True, append to existing database; if False, create new
        """
        self.db_path = Path(db_path)
        self.table_name = table_name
        self.append = append
        self._connections = {}  # Thread-local connections

        if not append and self.db_path.exists():
            logger.info(f"Removing existing database at {self.db_path}")
            self.db_path.unlink()

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_table_if_not_exists()
        logger.info(f"DuckDBStore initialized at {self.db_path}")

    def _create_table_if_not_exists(self):
        """Initializes the DuckDB table with the correct schema."""
        logger.info(f"Creating table '{self.table_name}'...")
        conn = self._get_thread_connection()
        # MODIFIED SCHEMA: Replaced file_path with image_url and local_filename
        query = f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                image_id VARCHAR,
                histogram BLOB,
                image_url VARCHAR,  -- NEW: Stores the remote image URL
                local_filename VARCHAR, -- NEW: Stores the local filename for feature re-computation/reference
                metadata_json VARCHAR,
                PRIMARY KEY (image_id)
            )
        """
        conn.execute(query)

    def insert_many(self, data: List[Tuple[str, bytes, str, str, str]]):
        """
        Bulk insert feature vectors and metadata into the DuckDB table.
        Args:
            data: List of tuples (image_id (URL), histogram, image_url, local_filename, metadata_json)
        """
        conn = self._get_thread_connection()
        # MODIFIED INSERT: 5 fields instead of 4
        query = f"""
            INSERT INTO {self.table_name}
            VALUES (?, ?, ?, ?, ?)
        """
        conn.executemany(query, data)
        conn.commit()

    # MODIFIED METHOD: Renamed and updated the query to fetch the URL
    def get_image_url_by_id(self, image_id: str) -> Optional[str]:
        """
        Get image URL for a given image ID.

        This method is specifically designed for URL-based indices where
        the image URL is stored in file_path or metadata.image_url.

        Args:
            image_id: Image identifier

        Returns:
            URL string or None if not found
        """
        try:
            # First try getting from file_path (direct URL storage)
            url = self.get_image_path(image_id)
            if url and (url.startswith("http://") or url.startswith("https://")):
                return url

            # If not a URL or None, try getting from metadata
            image_info = self.get_image_info(image_id)
            if image_info and "metadata" in image_info and image_info["metadata"]:
                metadata = image_info["metadata"]
                if isinstance(metadata, dict) and "image_url" in metadata:
                    return metadata["image_url"]

            # If still not found, return file_path as fallback
            return url
        except Exception as e:
            logger.error(f"Error getting image URL for image_id: {image_id}: {e}")
            return None

    def get_all_image_ids(self) -> Set[str]:
        """Retrieves all image IDs (URLs) currently in the database."""
        conn = self._get_thread_connection()
        result = conn.execute(f"SELECT image_id FROM {self.table_name}").fetchall()
        return {r[0] for r in result}

    def inspect_database_content(self, limit: int = 5) -> Dict[str, Any]:
        """
        Inspect the database content for debugging purposes.

        Args:
            limit: Number of records to inspect

        Returns:
            Dictionary with database inspection information
        """
        try:
            result = {}

            # Get column names
            columns_query = f"PRAGMA table_info({self.table_name})"
            columns = self.connection.execute(columns_query).fetchall()
            result["columns"] = [col[1] for col in columns]

            # Get sample data
            sample_query = f"SELECT * FROM {self.table_name} LIMIT {limit}"
            samples = self.connection.execute(sample_query).fetchall()

            # Format sample data for display
            sample_data = []
            for row in samples:
                sample_row = {}
                for i, col_name in enumerate([col[1] for col in columns]):
                    if col_name == "histogram":
                        sample_row[col_name] = f"<{len(row[i])} bytes blob>"
                    elif col_name == "metadata" and isinstance(row[i], str):
                        try:
                            # Try to parse the metadata as JSON for better display
                            sample_row[col_name] = json.loads(row[i])
                        except:
                            sample_row[col_name] = row[i]
                    else:
                        sample_row[col_name] = row[i]
                sample_data.append(sample_row)

            result["sample_data"] = sample_data
            result["total_records"] = self.connection.execute(
                f"SELECT COUNT(*) FROM {self.table_name}"
            ).fetchone()[0]

            return result
        except Exception as e:
            logger.error(f"Error inspecting database content: {e}")
            return {"error": str(e)}
