# build_index_urls.py
# Based on build_index.py, modified to index image URLs instead of local file paths.

import argparse
import logging
import os
import sys
import time
import json
import numpy as np
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Union
from tqdm import tqdm

# Add src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import core components with correct class names
from chromatica.indexing.store import (
    AnnIndex,  # This is the FAISS store class
    MetadataStore,  # Use MetadataStore instead of DuckDBStore
    hellinger_transform,
)
from chromatica.indexing.pipeline import process_image

# --- Setup Logging ---
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# --- Unicode-Safe Image Processing ---
def process_unicode_image(image_path: Path) -> Optional[np.ndarray]:
    """
    Process an image with potential Unicode characters in the filename.

    This function handles Unicode filenames by temporarily copying the image
    to a safe ASCII filename for OpenCV processing, then cleaning up.

    Args:
        image_path: Path to the image file (may contain Unicode characters)

    Returns:
        Normalized color histogram as numpy array, or None if processing failed
    """
    import cv2

    # Check if filename contains Unicode characters
    try:
        # Try to encode the path as ASCII
        str(image_path).encode("ascii")
        # If successful, use the original path directly
        return process_image(str(image_path))
    except UnicodeEncodeError:
        # Filename contains Unicode characters, use temporary copy approach
        logger.debug(f"Using temporary copy for Unicode filename: {image_path.name}")

        temp_dir = None
        temp_file_path = None

        try:
            # Create a temporary directory
            temp_dir = tempfile.mkdtemp(prefix="chromatica_unicode_")

            # Create a safe ASCII filename
            import hashlib
            import time

            # Generate a unique safe filename based on hash and timestamp
            file_hash = hashlib.md5(str(image_path).encode("utf-8")).hexdigest()[:8]
            timestamp = str(int(time.time() * 1000))[
                -6:
            ]  # Last 6 digits of millisecond timestamp
            safe_extension = image_path.suffix.lower()

            safe_filename = f"temp_{file_hash}_{timestamp}{safe_extension}"
            temp_file_path = Path(temp_dir) / safe_filename

            # Copy the file to the temporary location
            shutil.copy2(image_path, temp_file_path)
            logger.debug(
                f"Copied {image_path.name} to temporary file: {temp_file_path.name}"
            )

            # Process the temporary file
            histogram = process_image(str(temp_file_path))

            if histogram is not None:
                logger.debug(
                    f"Successfully processed Unicode filename: {image_path.name}"
                )

            return histogram

        except Exception as e:
            logger.error(f"Failed to process Unicode filename {image_path.name}: {e}")
            return None

        finally:
            # Clean up temporary files
            try:
                if temp_file_path and temp_file_path.exists():
                    temp_file_path.unlink()
                if temp_dir and Path(temp_dir).exists():
                    Path(temp_dir).rmdir()
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up temporary files: {cleanup_error}")


def validate_filename_encoding(filename: str) -> bool:
    """
    Check if a filename contains characters that might cause issues with OpenCV.

    Args:
        filename: The filename to check

    Returns:
        True if filename should work with OpenCV, False if it might cause issues
    """
    try:
        # Try to encode the filename as ASCII
        filename.encode("ascii")
        return True
    except UnicodeEncodeError:
        # Contains non-ASCII characters
        return False


# --- Custom URL-based Metadata Store Wrapper ---
class URLMetadataStore:
    """
    Wrapper around MetadataStore to handle URL-based metadata storage.

    This class adapts the existing MetadataStore to work with image URLs
    as primary identifiers while maintaining backward compatibility.
    """

    def __init__(self, db_path: Path, append: bool = False):
        """
        Initialize the URL-based metadata store.

        Args:
            db_path: Path to the DuckDB database file
            append: Whether to append to existing database
        """
        self.db_path = Path(db_path)
        self.append = append
        self.table_name = (
            "histograms"  # IMPORTANT: Use the same table name as MetadataStore
        )

        # Remove existing database file if not appending
        if not append and self.db_path.exists():
            logger.info(f"Removing existing database: {self.db_path}")
            self.db_path.unlink()

        # Create parent directory if it doesn't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize the underlying MetadataStore with the same table name
        self.store = MetadataStore(str(self.db_path), table_name=self.table_name)
        logger.info(
            f"URLMetadataStore initialized with DB: {db_path}, table: {self.table_name}"
        )

    def insert_many(self, data: List[tuple]) -> int:
        """
        Insert many records into the metadata store.

        Args:
            data: List of tuples containing (image_id, histogram_blob, url, local_filename, metadata_json)

        Returns:
            Number of successfully inserted records
        """
        try:
            # Convert the data format to match what MetadataStore expects
            batch_data = []

            for item in data:
                if len(item) == 5:  # (id, histogram, url, filename, metadata)
                    image_id, histogram_blob, url, local_filename, metadata_json = item

                    # Create combined metadata with URL information
                    try:
                        metadata_dict = (
                            json.loads(metadata_json)
                            if isinstance(metadata_json, str)
                            else {}
                        )
                    except (json.JSONDecodeError, TypeError):
                        metadata_dict = {}

                    # Add URL and local filename to metadata
                    metadata_dict["image_url"] = url
                    metadata_dict["local_filename"] = local_filename
                    file_size = metadata_dict.get("file_size", 0)

                    # Process the histogram blob
                    # It could be either a numpy array or bytes
                    if isinstance(histogram_blob, np.ndarray):
                        # If it's a numpy array, convert to bytes
                        histogram = histogram_blob.astype(np.float32).tobytes()
                    elif isinstance(histogram_blob, bytes):
                        # If it's already bytes, use as is
                        histogram = histogram_blob
                    else:
                        # Skip this entry if histogram is invalid
                        logger.warning(
                            f"Skipping entry with invalid histogram type: {type(histogram_blob)}"
                        )
                        continue

                    # Format according to what MetadataStore.add_batch expects
                    batch_entry = {
                        "image_id": str(image_id),
                        "histogram": histogram,  # This should be bytes, not a numpy array
                        "file_path": url,
                        "file_size": file_size,
                        "metadata": json.dumps(
                            metadata_dict
                        ),  # Serialize metadata to JSON string
                    }
                    batch_data.append(batch_entry)

            # Use the store's add_batch method to insert the data
            if batch_data:
                inserted = self.store.add_batch(batch_data)
                logger.info(
                    f"Successfully inserted {inserted} records into metadata store"
                )
                return inserted
            return 0

        except Exception as e:
            logger.error(f"Error adding batch to metadata store: {e}")
            return 0

    def get_all_image_ids(self) -> Set[str]:
        """
        Get all image URLs from the database.

        Returns:
            Set of all image URLs that have been indexed
        """
        try:
            conn = self.store._get_thread_connection()
            result = conn.execute(f"SELECT image_id FROM {self.table_name}").fetchall()
            return {row[0] for row in result}
        except Exception as e:
            logger.warning(f"Failed to get existing URLs: {e}")
            return set()

    def close(self) -> None:
        """Close the database connection."""
        try:
            self.store.close()
        except Exception as e:
            logger.error(f"Error closing metadata store: {e}")
            raise


# --- Helper Class: Processor (Glue layer for the pipeline) ---
class Processor:
    """
    Orchestrates image processing, feature extraction, and indexing.

    This processor handles the two-stage indexing pipeline:
    1. Feature extraction from local images using color histograms
    2. Storage in both FAISS (for ANN search) and DuckDB (for metadata)
    """

    def __init__(self, index_path: Path, db_path: Path, append: bool = False):
        """
        Initialize the processor with FAISS and DuckDB stores.

        Args:
            index_path: Path where FAISS index will be stored
            db_path: Path where DuckDB metadata will be stored
            append: Whether to append to existing index or create new one
        """
        self.append = append
        self.indexed_ids = set()

        # Store paths for later use during saving
        self.index_path = index_path  # <-- Fix: Store the index_path
        self.db_path = db_path  # <-- Fix: Store the db_path

        # Initialize custom URL metadata store
        self.db_store = URLMetadataStore(db_path, append=append)

        # Handle FAISS store initialization based on append mode
        if append and index_path.exists():
            # Load existing FAISS index
            logger.info(f"Loading existing FAISS index from: {index_path}")
            try:
                self.faiss_store = AnnIndex(index_path=str(index_path))
            except (AttributeError, Exception) as e:
                logger.error(f"Failed to load existing FAISS index: {e}")
                logger.info("Creating new FAISS index instead...")
                if index_path.exists():
                    index_path.unlink()
                self.faiss_store = self._create_faiss_index(index_path)

            # Get existing image IDs from DuckDB for deduplication
            try:
                self.indexed_ids = self.db_store.get_all_image_ids()
                logger.info(
                    f"Initialized in append mode. Found {len(self.indexed_ids)} already indexed URLs."
                )
            except Exception as e:
                logger.warning(f"Could not get existing IDs: {e}. Starting fresh.")
                self.indexed_ids = set()
        else:
            # Create new FAISS index (will overwrite existing if present)
            if index_path.exists() and not append:
                logger.info(f"Removing existing FAISS index: {index_path}")
                index_path.unlink()

            # Create new index
            logger.info("Creating new FAISS index...")
            self.faiss_store = self._create_faiss_index(index_path)
            logger.info("Initialized in fresh mode. Starting new index.")

    def _create_faiss_index(self, index_path: Path) -> AnnIndex:
        """
        Create a new FAISS index with proper error handling and fallbacks.

        Args:
            index_path: Path where the index will be stored

        Returns:
            Initialized AnnIndex instance

        Raises:
            RuntimeError: If index creation fails
        """
        try:
            # Try the standard constructor first
            return AnnIndex(index_path)
        except Exception as e:
            logger.warning(f"Standard AnnIndex constructor failed: {e}")

            # Try with explicit create_new parameter
            try:
                return AnnIndex(index_path, create_new=True)
            except Exception as e2:
                logger.warning(f"AnnIndex with create_new failed: {e2}")

                # Try manual creation approach
                try:
                    import faiss
                    from chromatica.utils.config import TOTAL_BINS, IVFPQ_M

                    # Create the index manually with proper parameters
                    dimension = TOTAL_BINS  # Should be 1152
                    M = IVFPQ_M  # Should be 32

                    # Create HNSW index with proper parameters
                    index = faiss.IndexHNSWFlat(dimension, M, faiss.METRIC_L2)

                    # Create AnnIndex with the pre-created index
                    ann_index = AnnIndex.__new__(AnnIndex)
                    ann_index.index_path = index_path
                    ann_index.index = index
                    ann_index.dimension = dimension
                    ann_index.M = M

                    logger.info(
                        f"Successfully created FAISS index manually with dimension={dimension}, M={M}"
                    )
                    return ann_index

                except Exception as e3:
                    logger.error(f"Manual FAISS index creation also failed: {e3}")
                    raise RuntimeError(f"Could not create FAISS index: {e}") from e3

    def get_indexed_files(self) -> Set[str]:
        """
        Returns the set of already indexed image IDs (URLs).

        Returns:
            Set of image URLs that have already been indexed
        """
        return self.indexed_ids

    def process_image_path(self, path: Path) -> Optional[np.ndarray]:
        """
        Reads local image, processes it, and returns the normalized histogram.

        Uses Unicode-safe processing for filenames with special characters.

        Args:
            path: Path to the local image file

        Returns:
            Normalized color histogram as numpy array, or None if processing failed
        """
        try:
            # Use Unicode-safe processing
            histogram = process_unicode_image(path)
            return histogram
        except Exception as e:
            logger.warning(f"Failed to process image {path}: {e}")
            return None

    def add_batch_to_index(
        self,
        batch_ids: List[str],
        batch_histograms: List[np.ndarray],
        batch_metadata: List[Dict[str, Any]],
    ) -> None:
        """
        Adds a batch of vectors and metadata to both FAISS and DuckDB.

        Args:
            batch_ids: List of image IDs (primary identifiers)
            batch_histograms: List of normalized color histograms
            batch_metadata: List of metadata dictionaries for each image

        Raises:
            Exception: If batch processing fails
        """
        try:
            # Apply Hellinger transform for FAISS storage
            # This transform improves ANN search performance for histograms
            histograms_array = np.stack(batch_histograms)
            vectors_array = hellinger_transform(histograms_array)

            # Use simple add instead of add_with_ids for HNSW compatibility
            # HNSW indexes don't support add_with_ids, only plain add
            try:
                # First try adding the whole batch at once
                vectors_batch = vectors_array.astype(np.float32)
                self.faiss_store.index.add(vectors_batch)
                logger.debug(f"Added {len(vectors_array)} vectors to FAISS index as batch")
            except Exception as e:
                # If batch add fails, fall back to adding vectors individually
                logger.debug(f"Batch add failed: {e}. Adding vectors individually")
                for vector in vectors_array:
                    vector_2d = vector.reshape(1, -1).astype(np.float32)
                    self.faiss_store.index.add(vector_2d)
                logger.debug(f"Added {len(vectors_array)} vectors to FAISS index individually")

            # Prepare data for DuckDB insertion using our adapted wrapper
            db_entries = []
            for idx, (image_id, histogram, metadata) in enumerate(
                zip(batch_ids, batch_histograms, batch_metadata)
            ):
                # Make sure we have a metadata object
                if not isinstance(metadata, dict):
                    metadata = {}
                
                # Convert histogram to bytes if needed
                if isinstance(histogram, np.ndarray):
                    histogram_blob = histogram.astype(np.float32).tobytes()
                else:
                    histogram_blob = histogram  # Assume it's already bytes
                
                # Get URL and local filename from metadata
                url = metadata.get("url", "")
                local_filename = metadata.get("local_filename", "")
                
                # Get other metadata
                metadata_dict = metadata.get("metadata", {})
                
                # Format entry for the database
                db_entries.append(
                    (
                        str(image_id),  # Make sure image_id is a string
                        histogram_blob,  # Histogram as bytes
                        url,  # URL
                        local_filename,  # Local filename
                        json.dumps(metadata_dict)  # Other metadata as JSON string
                    )
                )

            # Add to DuckDB
            inserted = self.db_store.insert_many(db_entries)
            logger.debug(f"Added batch of {len(batch_ids)} entries to FAISS and DuckDB (inserted {inserted} records).")

        except Exception as e:
            logger.error(f"Error adding batch to index: {e}")
            raise

    def save_config(self, config_path: Path) -> None:
        """
        Save the indexing configuration to a JSON file.

        Args:
            config_path: Path to the configuration file

        Raises:
            Exception: If saving fails
        """
        try:
            config = {
                "index_path": str(self.index_path),
                "db_path": str(self.db_path),
                "index_type": "HNSW",  # Or whatever type is used
                "total_bins": TOTAL_BINS,
                "created_at": datetime.datetime.now().isoformat(),
                "total_vectors": self.faiss_store.get_total_vectors(),
            }

            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

            logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise

    def save_index(self) -> None:
        """
        Saves the FAISS index to disk.

        Raises:
            Exception: If saving fails
        """
        try:
            # Pass the index_path to the save method
            index_path_str = str(self.index_path)
            self.faiss_store.save(index_path_str)
            logger.info(f"FAISS index saved successfully to {index_path_str}")
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
            raise

    def save_metadata(self) -> None:
        """
        Close the database connection, ensuring data is saved.

        Raises:
            Exception: If closing fails
        """
        try:
            self.db_store.close()
            logger.info(f"DuckDB metadata closed and saved at {self.db_path}")
        except Exception as e:
            logger.error(f"Error closing metadata store: {e}")
            raise


def parse_json_data(json_file_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Loads and flattens the JSON data into a dictionary of final_filename -> metadata.

    Args:
        json_file_path: Path to the JSON file containing image metadata

    Returns:
        Dictionary mapping filenames to their metadata

    Raises:
        FileNotFoundError: If JSON file doesn't exist
        json.JSONDecodeError: If JSON file is malformed
    """
    logger.info(f"Loading JSON metadata from: {json_file_path}")
    if not json_file_path.exists():
        raise FileNotFoundError(f"JSON file not found at: {json_file_path}")

    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in {json_file_path}: {e}")
        raise

    url_map = {}
    skipped_no_download = 0
    skipped_missing_fields = 0
    unicode_filename_count = 0

    for album_id, entry in data.items():
        final_filename = entry.get("final_filename")
        image_url = entry.get("image_url_modified")
        download_success = entry.get("download_success", False)

        if final_filename and image_url and download_success:
            # Check for Unicode characters that might cause issues
            if not validate_filename_encoding(final_filename):
                unicode_filename_count += 1
                logger.debug(f"Unicode filename detected: {final_filename}")

            # Key the map by the local filename
            url_map[final_filename] = {
                "url": image_url,
                "album_id": album_id,
                "metadata": entry,  # Store the entire entry for full metadata lookup later
            }
        elif not download_success:
            skipped_no_download += 1
            logger.debug(f"Skipping album ID {album_id}: download_success is false.")
        else:
            skipped_missing_fields += 1
            logger.debug(
                f"Skipping album ID {album_id}: missing final_filename or image_url_modified."
            )

    logger.info(f"Successfully mapped {len(url_map)} entries from the JSON.")
    logger.info(f"Skipped {skipped_no_download} entries due to download failure.")
    logger.info(f"Skipped {skipped_missing_fields} entries due to missing fields.")
    if unicode_filename_count > 0:
        logger.info(
            f"Found {unicode_filename_count} files with Unicode characters - using safe processing mode."
        )

    return url_map


def main():
    """
    Main execution function for building the FAISS index using image URLs.

    This script processes a JSON file containing image metadata and builds
    both a FAISS index for ANN search and a DuckDB store for metadata.
    The primary difference from build_index.py is that this script stores
    image URLs as the primary identifiers instead of local file paths.

    Features Unicode-safe processing for filenames with special characters.
    """
    parser = argparse.ArgumentParser(
        description="Offline indexing script adapted to store image URLs from a JSON file with Unicode filename support.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "json_file_path",
        type=str,
        help="Path to the JSON file containing the 'final_filename' and 'image_url_modified' fields (e.g., done.json).",
    )
    parser.add_argument(
        "image_directory",
        type=str,
        help="Path to the directory containing the downloaded image files (used for histogram calculation).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./index_urls",
        help="Directory to save the FAISS index and DuckDB metadata.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Number of images to process before adding to the index in one go.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to an existing index and metadata store instead of creating a new one.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging (DEBUG level)."
    )

    args = parser.parse_args()

    # Configure logging based on verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        # Also set debug level for the store modules
        logging.getLogger("chromatica.indexing.store").setLevel(logging.DEBUG)
        logging.getLogger("chromatica.indexing.pipeline").setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # --- Setup Paths and Validation ---
    json_path = Path(args.json_file_path)
    image_dir = Path(args.image_directory)
    output_dir = Path(args.output_dir)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    faiss_index_path = output_dir / "faiss.index"
    db_path = output_dir / "metadata.duckdb"

    # Log configuration
    logger.info("=" * 60)
    logger.info("CHROMATICA URL-BASED INDEX BUILDER (Unicode-Safe)")
    logger.info(f"JSON File: {json_path}")
    logger.info(f"Image Directory: {image_dir}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"FAISS Index Path: {faiss_index_path}")
    logger.info(f"DuckDB Path: {db_path}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Append Mode: {args.append}")
    logger.info("=" * 60)

    # Validate input paths
    if not json_path.exists():
        logger.error(f"JSON file not found: {json_path}")
        sys.exit(1)

    if not image_dir.exists():
        logger.error(f"Image directory not found: {image_dir}")
        sys.exit(1)

    try:
        # Initialize processor
        logger.info("Initializing processor...")
        processor = Processor(
            index_path=faiss_index_path, db_path=db_path, append=args.append
        )

        # Parse JSON data
        logger.info("Parsing JSON metadata...")
        url_map = parse_json_data(json_path)

        # Get already indexed URLs
        logger.info("Checking for already indexed URLs...")
        indexed_urls = processor.get_indexed_files()

        # Determine files to process
        files_to_process: List[Dict[str, Any]] = []
        missing_files = 0

        for filename, data in url_map.items():
            image_url = data["url"]
            if image_url in indexed_urls:
                logger.debug(f"Skipping {filename}: Already indexed by URL.")
                continue

            local_image_path = image_dir / filename
            if not local_image_path.exists():
                missing_files += 1
                logger.debug(
                    f"Skipping {filename}: Local file not found at {local_image_path}."
                )
                continue

            files_to_process.append(
                {"local_path": local_image_path, "url": image_url, "metadata": data}
            )

        # Log processing summary
        logger.info(f"Total images found in JSON: {len(url_map)}")
        logger.info(f"Already indexed: {len(indexed_urls)}")
        logger.info(f"Missing local files: {missing_files}")
        logger.info(f"New images to index: {len(files_to_process)}")

        if not files_to_process:
            logger.info("No new files to index. Exiting.")
            return

        # Main Batch Processing Loop
        histograms_to_add = []
        metadata_to_add = []
        total_successful = 0
        total_failed = 0
        unicode_failures = 0
        unicode_successes = 0
        start_time = time.time()

        logger.info("Starting batch processing with Unicode-safe handling...")
        for item in tqdm(files_to_process, desc="Indexing Images", unit="image"):
            local_path = item["local_path"]
            image_url = item["url"]

            try:
                # Check for Unicode characters in filename
                is_unicode_filename = not validate_filename_encoding(local_path.name)
                if is_unicode_filename:
                    logger.debug(f"Processing Unicode filename: {local_path.name}")

                # Process the image and get the normalized histogram
                histogram = processor.process_image_path(local_path)
                if histogram is None:
                    total_failed += 1
                    if is_unicode_filename:
                        unicode_failures += 1
                        logger.warning(
                            f"Unicode filename processing failed: {local_path.name}"
                        )
                    continue

                # Track Unicode processing success
                if is_unicode_filename:
                    unicode_successes += 1

                # Add to current batch
                histograms_to_add.append(histogram)

                # Collect all necessary metadata for DuckDB insertion
                metadata_to_add.append(
                    {
                        "id": image_url,  # The primary ID for FAISS and DuckDB
                        "url": image_url,
                        "local_filename": local_path.name,
                        "histogram": histogram,  # Pass the histogram for the DuckDB insert
                        "metadata": item["metadata"],  # The full JSON entry
                    }
                )

                # Process the batch when it reaches the specified size
                if len(histograms_to_add) >= args.batch_size:
                    logger.debug(
                        f"Processing batch of {len(histograms_to_add)} images..."
                    )
                    processor.add_batch_to_index(
                        batch_ids=[m["id"] for m in metadata_to_add],
                        batch_histograms=histograms_to_add,
                        batch_metadata=metadata_to_add,
                    )
                    total_successful += len(histograms_to_add)
                    histograms_to_add = []
                    metadata_to_add = []

            except Exception as e:
                total_failed += 1
                if not validate_filename_encoding(local_path.name):
                    unicode_failures += 1
                    logger.error(f"Unicode filename error for {local_path.name}: {e}")
                else:
                    logger.error(
                        f"An error occurred while processing {local_path}: {e}"
                    )

                if args.verbose:
                    import traceback

                    logger.debug(f"Full traceback:\n{traceback.format_exc()}")

        # Process the final batch if there are remaining items
        if histograms_to_add:
            logger.info(f"Processing final batch of {len(histograms_to_add)} images...")
            processor.add_batch_to_index(
                batch_ids=[m["id"] for m in metadata_to_add],
                batch_histograms=histograms_to_add,
                batch_metadata=metadata_to_add,
            )
            total_successful += len(histograms_to_add)

        # Calculate and log performance metrics
        total_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info("PROCESSING COMPLETE")
        logger.info(f"Total successful indexing operations: {total_successful}")
        logger.info(f"Total failed operations: {total_failed}")

        # Unicode-specific metrics
        if unicode_successes > 0 or unicode_failures > 0:
            logger.info(f"Unicode filename successes: {unicode_successes}")
            logger.info(f"Unicode filename failures: {unicode_failures}")
            unicode_success_rate = (
                unicode_successes / (unicode_successes + unicode_failures) * 100
            )
            logger.info(f"Unicode processing success rate: {unicode_success_rate:.1f}%")

        logger.info(f"Total processing time: {total_time:.2f} seconds")

        if total_successful > 0:
            throughput = total_successful / total_time
            logger.info(f"Processing throughput: {throughput:.1f} images/second")

        # Save the index and metadata
        logger.info("Saving FAISS index and DuckDB metadata...")
        processor.save_index()
        processor.save_metadata()

        logger.info(f"FAISS index saved to: {faiss_index_path}")
        logger.info(f"Metadata database saved to: {db_path}")

        logger.info("=" * 60)
        logger.info("INDEXING COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.warning("Indexing interrupted by user. Partial results may be saved.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Indexing process failed: {str(e)}")
        if args.verbose:
            import traceback

            logger.error(f"Full traceback:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
