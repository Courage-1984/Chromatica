# Example: python scripts/build_covers_index.py data/done.json data/images --output-dir index_urls --batch-size 1000 --verbose

"""
Offline indexing script for the Chromatica color search engine with JSON metadata support.

This script processes a JSON metadata file containing image information and a directory
of images, then populates both the FAISS HNSW index and DuckDB metadata store. It's
designed specifically for handling album cover metadata with image URLs and filenames.

The script follows the two-stage search architecture:
1. FAISS index stores Hellinger-transformed histograms for fast ANN search
2. DuckDB stores raw histograms and metadata for accurate reranking

Key Features:
- Processes JSON metadata files with 'final_filename' and 'image_url_modified' fields
- Downloads images from URLs if local files are not found
- Supports append mode for incremental indexing
- Maintains compatibility with existing Chromatica infrastructure

Usage:
    python scripts/build_covers_index.py <json_metadata_file> <image_directory> [--output-dir <output_dir>] [--batch-size <batch_size>] [--append]

Example:
    python scripts/build_covers_index.py data/done.json data/images --output-dir index_urls --batch-size 1000 --verbose

Features:
- JSON metadata parsing with final_filename and image_url_modified fields
- Image URL downloading with retry logic
- Batch processing for memory efficiency
- Comprehensive logging and progress tracking
- Error handling with graceful degradation
- Validation of processed histograms
- Performance monitoring and timing
- Automatic output directory creation
- Duplicate image detection in append mode
"""

import argparse
import logging
import os
import sys
import time
import json
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
import numpy as np
from urllib.parse import urlparse
import tempfile
from datetime import datetime, timedelta

# Add src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chromatica.indexing.pipeline import process_image, validate_processed_image
from chromatica.indexing.store import AnnIndex, MetadataStore
from chromatica.utils.config import TOTAL_BINS


class ProgressTracker:
    """
    Comprehensive progress tracking for the indexing process.
    """
    
    def __init__(self, total_entries: int, batch_size: int):
        self.total_entries = total_entries
        self.batch_size = batch_size
        self.start_time = time.time()
        self.processed_entries = 0
        self.successful_entries = 0
        self.failed_entries = 0
        self.skipped_entries = 0
        self.current_batch = 0
        self.total_batches = (total_entries + batch_size - 1) // batch_size
        
        # Batch tracking variables
        self.batch_processed = 0
        self.batch_successful = 0
        self.batch_failed = 0
        self.batch_skipped = 0
        self.batch_start_time = time.time()  # Initialize batch start time
        self.batch_entries = 0  # Initialize batch entries count
        
        # Timing statistics
        self.batch_times = []
        self.entry_times = []
        
        # Progress logging
        self.last_progress_log = 0
        self.progress_interval = 10  # Log progress every 10 seconds
        
    def start_batch(self, batch_num: int, batch_entries: int):
        """Start tracking a new batch."""
        self.current_batch = batch_num
        self.batch_start_time = time.time()
        self.batch_entries = batch_entries
        self.batch_processed = 0
        self.batch_successful = 0
        self.batch_failed = 0
        self.batch_skipped = 0
        
    def update_entry(self, success: bool, skipped: bool = False):
        """Update progress for a single entry."""
        self.processed_entries += 1
        self.batch_processed += 1
        
        if skipped:
            self.skipped_entries += 1
            self.batch_skipped += 1
        elif success:
            self.successful_entries += 1
            self.batch_successful += 1
        else:
            self.failed_entries += 1
            self.batch_failed += 1
            
        # Record timing
        entry_time = time.time() - self.batch_start_time
        self.entry_times.append(entry_time)
        
    def finish_batch(self):
        """Finish tracking the current batch."""
        batch_time = time.time() - self.batch_start_time
        self.batch_times.append(batch_time)
        
    def get_progress_stats(self) -> Dict[str, Any]:
        """Get comprehensive progress statistics."""
        elapsed_time = time.time() - self.start_time
        
        # Calculate rates
        if elapsed_time > 0:
            entries_per_second = self.processed_entries / elapsed_time
            successful_per_second = self.successful_entries / elapsed_time
        else:
            entries_per_second = 0
            successful_per_second = 0
            
        # Calculate ETA
        if self.successful_entries > 0 and successful_per_second > 0:
            remaining_entries = self.total_entries - self.processed_entries
            eta_seconds = remaining_entries / successful_per_second
            eta = datetime.now() + timedelta(seconds=eta_seconds)
        else:
            eta = None
            
        # Calculate percentages
        progress_percent = (self.processed_entries / self.total_entries) * 100 if self.total_entries > 0 else 0
        success_rate = (self.successful_entries / self.processed_entries) * 100 if self.processed_entries > 0 else 0
        
        # Average times
        avg_batch_time = np.mean(self.batch_times) if self.batch_times else 0
        avg_entry_time = np.mean(self.entry_times) if self.entry_times else 0
        
        return {
            'total_entries': self.total_entries,
            'processed_entries': self.processed_entries,
            'successful_entries': self.successful_entries,
            'failed_entries': self.failed_entries,
            'skipped_entries': self.skipped_entries,
            'current_batch': self.current_batch,
            'total_batches': self.total_batches,
            'progress_percent': progress_percent,
            'success_rate': success_rate,
            'elapsed_time': elapsed_time,
            'entries_per_second': entries_per_second,
            'successful_per_second': successful_per_second,
            'eta': eta,
            'avg_batch_time': avg_batch_time,
            'avg_entry_time': avg_entry_time
        }
        
    def should_log_progress(self) -> bool:
        """Check if it's time to log progress."""
        current_time = time.time()
        if current_time - self.last_progress_log >= self.progress_interval:
            self.last_progress_log = current_time
            return True
        return False
        
    def format_time(self, seconds: float) -> str:
        """Format time duration in a human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"


def safe_log_message(logger, level: str, message: str) -> None:
    """
    Safely log a message that may contain Unicode characters.
    
    Args:
        logger: Logger instance
        level: Log level ('debug', 'info', 'warning', 'error')
        message: Message to log (may contain Unicode)
    """
    try:
        # Try to log normally first
        getattr(logger, level)(message)
    except UnicodeEncodeError:
        # If Unicode encoding fails, replace problematic characters
        safe_message = message.encode('ascii', 'replace').decode('ascii')
        getattr(logger, level)(f"[Unicode] {safe_message}")
    except Exception as e:
        # Fallback for any other logging issues
        getattr(logger, level)(f"[Logging Error] {str(e)}")


def log_progress_summary(logger, progress_tracker: ProgressTracker, is_batch_complete: bool = False):
    """
    Log a comprehensive progress summary.
    
    Args:
        logger: Logger instance
        progress_tracker: ProgressTracker instance
        is_batch_complete: Whether this is a batch completion log
    """
    stats = progress_tracker.get_progress_stats()
    
    if is_batch_complete:
        # Detailed batch completion log
        safe_log_message(logger, 'info', 
            f"Batch {stats['current_batch']}/{stats['total_batches']} completed: "
            f"{stats['batch_successful']} successful, {stats['batch_failed']} failed, "
            f"{stats['batch_skipped']} skipped in {progress_tracker.format_time(stats['avg_batch_time'])}"
        )
    
    # Overall progress summary
    eta_str = stats['eta'].strftime("%H:%M:%S") if stats['eta'] else "Unknown"
    
    safe_log_message(logger, 'info',
        f"Overall Progress: {stats['processed_entries']}/{stats['total_entries']} "
        f"({stats['progress_percent']:.1f}%) | "
        f"Success: {stats['successful_entries']} ({stats['success_rate']:.1f}%) | "
        f"Failed: {stats['failed_entries']} | "
        f"Skipped: {stats['skipped_entries']} | "
        f"Rate: {stats['successful_per_second']:.1f} entries/sec | "
        f"ETA: {eta_str}"
    )
    
    # Performance metrics
    if stats['avg_entry_time'] > 0:
        safe_log_message(logger, 'debug',
            f"Performance: Avg batch time: {progress_tracker.format_time(stats['avg_batch_time'])}, "
            f"Avg entry time: {progress_tracker.format_time(stats['avg_entry_time'])}"
        )


def setup_logging(log_level: str = "INFO") -> None:
    """
    Set up logging configuration for the indexing process.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Set up console handler with UTF-8 encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    
    # Ensure console handler can handle Unicode
    if hasattr(console_handler.stream, 'reconfigure'):
        try:
            console_handler.stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass  # Fallback if reconfigure is not available

    # Set up file handler for detailed logging with UTF-8 encoding
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    file_handler = logging.FileHandler(
        log_dir / f"build_covers_index_{int(time.time())}.log",
        encoding='utf-8',
        errors='replace'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Set specific levels for verbose modules
    logging.getLogger("chromatica.core.histogram").setLevel(logging.INFO)
    logging.getLogger("chromatica.indexing.pipeline").setLevel(logging.INFO)

    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level: {log_level}")


def load_json_metadata(json_file: Path) -> Dict[str, Any]:
    """
    Load and parse JSON metadata file.

    Args:
        json_file: Path to the JSON metadata file

    Returns:
        Dictionary containing the parsed JSON data

    Raises:
        ValueError: If file doesn't exist or contains invalid JSON
    """
    if not json_file.exists():
        raise ValueError(f"JSON metadata file does not exist: {json_file}")

    if not json_file.is_file():
        raise ValueError(f"Path is not a file: {json_file}")

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        logger = logging.getLogger(__name__)
        logger.info(f"Loaded JSON metadata with {len(metadata)} entries")
        
        return metadata
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in metadata file {json_file}: {e}")
    except Exception as e:
        raise ValueError(f"Failed to load JSON metadata from {json_file}: {e}")


def download_image_from_url(url: str, output_path: Path, max_retries: int = 3) -> bool:
    """
    Download an image from a URL to the specified path.

    Args:
        url: URL of the image to download
        output_path: Local path where the image should be saved
        max_retries: Maximum number of retry attempts

    Returns:
        bool: True if download successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    for attempt in range(max_retries):
        try:
            logger.debug(f"Downloading image from {url} (attempt {attempt + 1}/{max_retries})")
            
            # Set headers to mimic a browser request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=30, stream=True)
            response.raise_for_status()
            
            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write the image data
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Successfully downloaded image to {output_path}")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Download attempt {attempt + 1} failed for {url}: {e}")
            if attempt == max_retries - 1:
                logger.error(f"Failed to download image from {url} after {max_retries} attempts")
                return False
            time.sleep(1)  # Wait before retry
        except Exception as e:
            logger.error(f"Unexpected error downloading {url}: {e}")
            return False
    
    return False


def find_image_file(filename: str, image_dir: Path) -> Optional[Path]:
    """
    Find an image file in the directory, trying different extensions.

    Args:
        filename: Base filename to search for
        image_dir: Directory to search in

    Returns:
        Path to the found image file, or None if not found
    """
    # Remove extension from filename if present
    base_name = Path(filename).stem
    
    # Common image extensions to try
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
    
    for ext in extensions:
        # Try with original extension first
        if filename.endswith(ext):
            full_path = image_dir / filename
            if full_path.exists():
                return full_path
        
        # Try with different extensions
        full_path = image_dir / f"{base_name}{ext}"
        if full_path.exists():
            return full_path
    
    return None


def process_metadata_entry(
    entry_id: str, 
    entry_data: Dict[str, Any], 
    image_dir: Path,
    download_missing: bool = True
) -> Optional[Tuple[str, Path, str]]:
    """
    Process a single metadata entry and return image information.

    Args:
        entry_id: ID of the metadata entry
        entry_data: Dictionary containing the metadata
        image_dir: Directory containing local images
        download_missing: Whether to download missing images from URLs

    Returns:
        Tuple of (image_id, image_path, image_url) or None if processing failed
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Extract required fields
        final_filename = entry_data.get('final_filename')
        image_url = entry_data.get('image_url_modified')
        
        if not final_filename:
            logger.warning(f"Entry {entry_id} missing 'final_filename' field")
            return None
        
        if not image_url:
            logger.warning(f"Entry {entry_id} missing 'image_url_modified' field")
            return None
        
        # Try to find the image file locally first
        image_path = find_image_file(final_filename, image_dir)
        
        if image_path is None and download_missing:
            # Try to download the image
            logger.info(f"Image {final_filename} not found locally, attempting download from {image_url}")
            
            # Create a temporary filename for download
            temp_filename = f"downloaded_{entry_id}_{final_filename}"
            temp_path = image_dir / temp_filename
            
            if download_image_from_url(image_url, temp_path):
                image_path = temp_path
            else:
                logger.error(f"Failed to download image for entry {entry_id}")
                return None
        
        if image_path is None:
            logger.error(f"Image {final_filename} not found and download failed for entry {entry_id}")
            return None
        
        # Use the entry ID as the image ID for indexing
        return (entry_id, image_path, image_url)
        
    except Exception as e:
        logger.error(f"Error processing metadata entry {entry_id}: {e}")
        return None


def process_image_batch(
    image_entries: List[Tuple[str, Path, str]],
    batch_size: int,
    ann_index: AnnIndex,
    metadata_store: MetadataStore,
    progress_tracker: ProgressTracker,
    is_append_mode: bool = False,
) -> Dict[str, Any]:
    """
    Process a batch of image entries and add them to the index and metadata store.

    Args:
        image_entries: List of (image_id, image_path, image_url) tuples
        batch_size: Number of images to process in this batch
        ann_index: FAISS index for storing transformed histograms
        metadata_store: DuckDB store for metadata and raw histograms
        is_append_mode: If True, check for and skip images already in the store

    Returns:
        Dictionary with batch processing statistics
    """
    logger = logging.getLogger(__name__)

    batch_start_time = time.time()
    successful_count = 0
    error_count = 0
    total_processing_time = 0

    files_to_process = image_entries[:batch_size]

    # 1. Check for existing images if in append mode
    existing_ids: Set[str] = set()
    if is_append_mode and files_to_process:
        image_ids_to_check = [entry[0] for entry in files_to_process]
        try:
            existing_ids = metadata_store.check_existence(image_ids_to_check)
            logger.info(
                f"Found {len(existing_ids)} images already indexed in this batch."
            )
        except AttributeError:
            logger.error(
                "MetadataStore is missing required 'check_existence' method for de-duplication. Indexing all images."
            )
        except Exception as e:
            logger.error(
                f"Error checking for existing images: {str(e)}. Indexing all images."
            )

    # Prepare batch data for metadata store
    metadata_batch = []

    # 2. Process images, skipping duplicates if in append mode
    for i, (image_id, image_path, image_url) in enumerate(files_to_process):
        # Skip if already exists in append mode
        if is_append_mode and image_id in existing_ids:
            logger.debug(f"Skipping {image_path.name}: already indexed.")
            progress_tracker.update_entry(success=False, skipped=True)
            continue

        image_start_time = time.time()

        try:
            # Log progress periodically
            if progress_tracker.should_log_progress():
                log_progress_summary(logger, progress_tracker)

            safe_log_message(
                logger, 'info',
                f"Processing image {i+1}/{len(files_to_process)}: {image_path.name} (ID: {image_id})"
            )

            # Process image and generate histogram
            histogram = process_image(str(image_path))

            # Validate histogram
            validate_processed_image(histogram, str(image_path))

            # Get file size
            file_size = image_path.stat().st_size

            # Add to metadata batch
            metadata_batch.append(
                {
                    "image_id": image_id,
                    "file_path": str(image_path),
                    "histogram": histogram,
                    "file_size": file_size,
                    "image_url": image_url,  # Store the original URL
                }
            )

            successful_count += 1
            processing_time = time.time() - image_start_time
            total_processing_time += processing_time
            
            # Update progress tracker
            progress_tracker.update_entry(success=True)

            safe_log_message(
                logger, 'debug',
                f"Successfully processed {image_path.name} in {processing_time:.3f}s"
            )

        except Exception as e:
            error_count += 1
            progress_tracker.update_entry(success=False)
            safe_log_message(logger, 'error', f"Failed to process {image_path.name}: {str(e)}")
            continue

    # Add histograms to FAISS index (Hellinger-transformed)
    if metadata_batch:
        try:
            # Extract histograms for FAISS
            histograms = np.array([record["histogram"] for record in metadata_batch])

            # Add to FAISS index and receive assigned IDs
            faiss_ids = ann_index.add(histograms)

            # Attach FAISS IDs to metadata records
            for rec, fid in zip(metadata_batch, faiss_ids.tolist()):
                rec["faiss_id"] = int(fid)

            # Add metadata to DuckDB (now with faiss_id)
            metadata_store.add_batch(metadata_batch)

            logger.info(f"Successfully indexed batch: {len(metadata_batch)} new images")

        except Exception as e:
            logger.error(f"Failed to index batch: {str(e)}")
            # Rollback successful processing
            successful_count -= len(metadata_batch)
            error_count += len(metadata_batch)

    batch_time = time.time() - batch_start_time
    avg_processing_time = (
        total_processing_time / successful_count if successful_count > 0 else 0
    )

    return {
        "successful_count": successful_count,
        "error_count": error_count,
        "batch_time": batch_time,
        "avg_processing_time": avg_processing_time,
    }


INDEX_CONFIG_FILENAME = "chromatica_index_config.json"


def prompt_index_type(total_images: int) -> str:
    """
    Prompt the user to select index type based on dataset size.
    Returns: "small" or "large"
    """
    print("\n[Chromatica] Indexing strategy selection:")
    print(f"  Detected {total_images} images in this batch.")
    print("  [S]mall: HNSW/Flat (recommended for <100,000 images)")
    print("  [L]arge: IVFPQ (recommended for >100,000 images, required for millions)")
    while True:
        user_input = input("Select index type ([S]mall/[L]arge): ").strip().upper()
        if user_input in ("S", "SMALL"):
            return "small"
        elif user_input in ("L", "LARGE"):
            return "large"
        else:
            print("Invalid input. Please type 'S' or 'L'.")


def save_index_config(output_dir: Path, config: dict) -> None:
    """
    Save index configuration to a JSON file in the output directory.
    """
    config_path = output_dir / INDEX_CONFIG_FILENAME
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def load_index_config(output_dir: Path) -> Optional[dict]:
    """
    Load index configuration from the output directory if it exists.
    """
    config_path = output_dir / INDEX_CONFIG_FILENAME
    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)
    return None


def prompt_ivfpq_params(num_training_vectors: int) -> dict:
    """
    Prompt for IVFPQ parameters, ensuring nlist <= num_training_vectors.
    """
    print(f"\n[Chromatica] IVFPQ parameter selection:")
    print(f"  Number of training vectors: {num_training_vectors}")
    max_nlist = max(2, 2 ** (num_training_vectors.bit_length() - 1))
    suggested_nlist = min(16384, max_nlist, num_training_vectors)
    print(f"  Recommended nlist: {suggested_nlist} (must be <= {num_training_vectors})")
    while True:
        try:
            nlist = int(
                input(f"Enter nlist (clusters) [default {suggested_nlist}]: ")
                or suggested_nlist
            )
            if nlist > num_training_vectors:
                print(
                    f"nlist must be <= number of training vectors ({num_training_vectors})"
                )
            elif nlist < 2:
                print("nlist must be at least 2")
            else:
                break
        except ValueError:
            print("Please enter a valid integer.")
    M = 32
    nbits = 8
    return {"type": "IVFPQ", "nlist": nlist, "M": M, "nbits": nbits}


def auto_ivfpq_params(num_training_vectors: int) -> dict:
    """
    Automatically select IVFPQ parameters based on available training vectors.
    Ensures nlist <= num_training_vectors and is a power of two.
    """
    # Use largest power of two <= num_training_vectors, but cap at 16384
    max_nlist = max(2, 2 ** (num_training_vectors.bit_length() - 1))
    nlist = min(16384, max_nlist, num_training_vectors)
    if nlist < 16384:
        print(
            f"[Chromatica] Warning: nlist set to {nlist} due to limited training vectors ({num_training_vectors})."
        )
    M = 32
    nbits = 8
    return {"type": "IVFPQ", "nlist": nlist, "M": M, "nbits": nbits}


def main():
    """Main function for the covers indexing script."""
    parser = argparse.ArgumentParser(
        description="Build offline index for Chromatica color search engine from JSON metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/build_covers_index.py data/done.json data/images
  python scripts/build_covers_index.py data/done.json data/images --output-dir index_urls --batch-size 1000 --verbose
  python scripts/build_covers_index.py data/done.json data/images --append
        """,
    )

    parser.add_argument(
        "json_metadata_file", type=str, help="JSON file containing image metadata"
    )

    parser.add_argument(
        "image_directory", type=str, help="Directory containing images to index"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./index",
        help="Output directory for index files (default: ./index)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of images to process in each batch (default: 100)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging (DEBUG level)",
    )

    parser.add_argument(
        "--start-index",
        type=int,
        help="Starting index in the metadata list (for chunked processing)",
    )

    parser.add_argument(
        "--end-index",
        type=int,
        help="Ending index in the metadata list (for chunked processing)",
    )

    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing index instead of creating new ones",
    )

    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip downloading missing images from URLs",
    )

    args = parser.parse_args()

    # Set up logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Starting Chromatica covers indexing process")
    logger.info("=" * 60)

    try:
        # Validate and prepare paths
        json_file = Path(args.json_metadata_file).resolve()
        image_dir = Path(args.image_directory).resolve()
        output_dir = Path(args.output_dir).resolve()

        logger.info(f"JSON metadata file: {json_file}")
        logger.info(f"Image directory: {image_dir}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Batch size: {args.batch_size}")

        # Validate paths
        if not image_dir.exists():
            raise ValueError(f"Image directory does not exist: {image_dir}")

        if not image_dir.is_dir():
            raise ValueError(f"Image path is not a directory: {image_dir}")

        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory created/verified: {output_dir}")

        # Load JSON metadata
        metadata = load_json_metadata(json_file)
        total_entries = len(metadata)

        logger.info(f"Total metadata entries found: {total_entries}")

        # Process metadata entries to get image information
        logger.info("Processing metadata entries...")
        image_entries = []
        failed_entries = 0

        for entry_id, entry_data in metadata.items():
            result = process_metadata_entry(
                entry_id, 
                entry_data, 
                image_dir, 
                download_missing=not args.no_download
            )
            if result:
                image_entries.append(result)
            else:
                failed_entries += 1

        logger.info(f"Successfully processed {len(image_entries)} entries, {failed_entries} failed")

        if not image_entries:
            raise ValueError("No valid image entries found to process")

        # Handle chunked processing
        start_idx = args.start_index if args.start_index is not None else 0
        end_idx = args.end_index if args.end_index is not None else len(image_entries)

        if start_idx < 0 or end_idx > len(image_entries) or start_idx >= end_idx:
            raise ValueError(
                f"Invalid start/end indices. Must be: 0 <= start ({start_idx}) < end ({end_idx}) <= total ({len(image_entries)})"
            )

        # Adjust image entries list for chunked processing
        image_entries = image_entries[start_idx:end_idx]
        chunk_total = len(image_entries)

        logger.info(
            f"Processing chunk: entries {start_idx} to {end_idx} (total: {chunk_total} entries)"
        )

        # Initialize progress tracker
        progress_tracker = ProgressTracker(chunk_total, args.batch_size)
        logger.info(f"Progress tracking initialized for {chunk_total} entries in {progress_tracker.total_batches} batches")

        # Initialize index and metadata store
        logger.info("Initializing FAISS index and metadata store...")

        # FAISS index file path
        faiss_index_path = output_dir / "chromatica_index.faiss"
        # DuckDB database file path
        db_path = output_dir / "chromatica_metadata.db"
        # Config file path
        config_path = output_dir / INDEX_CONFIG_FILENAME

        index_exists = faiss_index_path.exists() or db_path.exists()
        append_mode = args.append  # Respect the command line flag initially

        # Load or prompt for index config
        index_config = load_index_config(output_dir)
        if not index_exists or not index_config:
            # New index: prompt for type
            index_type = prompt_index_type(chunk_total)
            if index_type == "small":
                index_params = {"type": "HNSW", "M": 32}
            else:
                # For IVFPQ, use recommended params for large datasets
                index_params = {"type": "IVFPQ", "nlist": 16384, "M": 32, "nbits": 8}
            index_config = index_params
            save_index_config(output_dir, index_config)
            append_mode = False  # Force to False if we are creating a new index
        elif index_exists and not args.append:
            # Existing index: prompt for append/replace/quit
            print(f"\n[Chromatica] Existing index or metadata found in {output_dir}:")
            if faiss_index_path.exists():
                print(f"  - FAISS index: {faiss_index_path}")
            if db_path.exists():
                print(f"  - DuckDB metadata: {db_path}")
            print(
                "Would you like to (A)ppend to the existing index, (D)elete/replace it, or (Q)uit?"
            )
            while True:
                user_input = (
                    input("Type 'A' to append, 'D' to delete/replace, or 'Q' to quit: ")
                    .strip()
                    .upper()
                )
                if user_input == "A":
                    append_mode = True
                    print(
                        "[Chromatica] Append mode selected. Will add to existing index."
                    )
                    break
                elif user_input == "D":
                    print("[Chromatica] Deleting existing index and metadata...")
                    if faiss_index_path.exists():
                        faiss_index_path.unlink()
                    if db_path.exists():
                        db_path.unlink()
                    if config_path.exists():
                        config_path.unlink()
                    append_mode = False
                    # Prompt again for index type
                    index_type = prompt_index_type(chunk_total)
                    if index_type == "small":
                        index_params = {"type": "HNSW", "M": 32}
                    else:
                        index_params = {
                            "type": "IVFPQ",
                            "nlist": 16384,
                            "M": 32,
                            "nbits": 8,
                        }
                    index_config = index_params
                    save_index_config(output_dir, index_config)
                    break
                elif user_input == "Q":
                    print("[Chromatica] Aborting as requested by user.")
                    sys.exit(0)
                else:
                    print("Invalid input. Please type 'A', 'D', or 'Q'.")

        logger.info(f"Indexing Mode: {'APPEND' if append_mode else 'CREATE/REPLACE'}")

        # Initialize FAISS index according to config
        if index_config["type"] == "HNSW":
            ann_index = AnnIndex(
                dimension=TOTAL_BINS,
                use_simple_index=True,
                index_path=str(faiss_index_path) if append_mode else None,
                M=index_config.get("M", 32),
            )
        elif index_config["type"] == "IVFPQ":
            ann_index = AnnIndex(
                dimension=TOTAL_BINS,
                use_simple_index=False,
                index_path=str(faiss_index_path) if append_mode else None,
                nlist=index_config.get("nlist", 16384),
                M=index_config.get("M", 32),
                nbits=index_config.get("nbits", 8),
            )
        else:
            raise ValueError(f"Unknown index type: {index_config['type']}")

        metadata_store = MetadataStore(db_path=str(db_path))

        logger.info("Components initialized successfully")

        training_sample_size = 0
        training_histograms = []
        training_entries = []
        training_data = np.array([])

        # Train the FAISS index with a sample of images
        # Training is only needed if the index is NOT already trained and loaded (i.e. not in append mode)
        if hasattr(ann_index.index, "is_trained") and not ann_index.index.is_trained:
            logger.info("Training FAISS index with representative data...")
            # Use up to 1000 images for training from the current chunk
            training_sample_size = min(1000, chunk_total)
            training_entries = image_entries[:training_sample_size]

            # Process training images to get histograms
            for i, (image_id, image_path, image_url) in enumerate(training_entries):
                try:
                    safe_log_message(
                        logger, 'debug',
                        f"Processing training image {i+1}/{len(training_entries)}: {image_path.name}"
                    )
                    histogram = process_image(str(image_path))
                    validate_processed_image(histogram, str(image_path))
                    training_histograms.append(histogram)
                    
                    # Update progress tracker for training images
                    progress_tracker.update_entry(success=True)
                    
                except Exception as e:
                    safe_log_message(
                        logger, 'warning',
                        f"Failed to process training image {image_path.name}: {str(e)}"
                    )
                    progress_tracker.update_entry(success=False)
                    continue

            if not training_histograms:
                # If training fails, index cannot proceed for IndexIVFPQ
                if index_config["type"] == "IVFPQ":
                    raise RuntimeError(
                        "No valid training histograms could be generated. Indexing aborted."
                    )
                else:
                    logger.warning(
                        "No valid training histograms generated. Proceeding without training for HNSW."
                    )
            else:
                # Convert to numpy array and train the index
                training_data = np.array(training_histograms)

                # If using IVFPQ, auto-adjust and update nlist
                if index_config["type"] == "IVFPQ":
                    ivfpq_params = auto_ivfpq_params(len(training_data))
                    index_config.update(ivfpq_params)
                    save_index_config(output_dir, index_config)

                    # Re-initialize AnnIndex with final IVFPQ params
                    ann_index = AnnIndex(
                        dimension=TOTAL_BINS,
                        use_simple_index=False,
                        index_path=str(faiss_index_path) if append_mode else None,
                        nlist=index_config["nlist"],
                        M=index_config["M"],
                        nbits=index_config["nbits"],
                    )

                ann_index.train(training_data)

                logger.info(
                    f"FAISS index training completed with {len(training_histograms)} histograms"
                )

        # --- Index the training images immediately after training ---
        total_successful = 0
        total_errors = 0

        if hasattr(ann_index.index, "is_trained") and ann_index.index.is_trained:
            # If training occurred, index the data used for training
            if training_histograms:
                try:
                    # Add training histograms to FAISS index and receive IDs
                    training_faiss_ids = ann_index.add(training_data)

                    # Add training metadata to DuckDB
                    training_metadata = []
                    for i, (image_id, image_path, image_url) in enumerate(
                        training_entries[: len(training_histograms)]
                    ):
                        training_metadata.append(
                            {
                                "image_id": image_id,
                                "file_path": str(image_path),
                                "histogram": training_histograms[i],
                                "file_size": image_path.stat().st_size,
                                "image_url": image_url,
                                "faiss_id": int(training_faiss_ids[i]) if i < len(training_faiss_ids) else None,
                            }
                        )

                    # Note: DuckDB's add_batch is INSERT OR REPLACE, so it's idempotent.
                    metadata_store.add_batch(training_metadata)
                    total_successful += len(training_histograms)

                    logger.info(
                        f"Successfully indexed {len(training_histograms)} training images"
                    )
                except Exception as e:
                    logger.error(f"Failed to index training images: {str(e)}")
                    total_errors += len(training_histograms)

        # Process remaining images in batches
        # Correctly determine the list of remaining images after the training set
        remaining_entries = image_entries[training_sample_size:]
        remaining_count = len(remaining_entries)

        logger.info(f"Processing {remaining_count} remaining entries in batches...")

        start_time = time.time()

        for batch_start in range(0, remaining_count, args.batch_size):
            batch_end = min(batch_start + args.batch_size, remaining_count)
            batch_entries = remaining_entries[batch_start:batch_end]
            batch_num = batch_start//args.batch_size + 1

            # Start tracking this batch
            progress_tracker.start_batch(batch_num, len(batch_entries))

            logger.info(
                f"Processing batch {batch_num}/{progress_tracker.total_batches}: "
                f"entries {batch_start + 1}-{batch_end} of {remaining_count} remaining"
            )

            # Process batch with de-duplication check
            batch_stats = process_image_batch(
                batch_entries,
                len(batch_entries),
                ann_index,
                metadata_store,
                progress_tracker,
                is_append_mode=append_mode,
            )

            # Finish tracking this batch
            progress_tracker.finish_batch()

            total_successful += batch_stats["successful_count"]
            total_errors += batch_stats["error_count"]

            # Log detailed batch completion
            log_progress_summary(logger, progress_tracker, is_batch_complete=True)

        # Save index and close connections
        logger.info("Saving index and closing connections...")

        ann_index.save(str(faiss_index_path))
        metadata_store.close()

        # Final statistics
        total_time = time.time() - start_time
        final_total_vectors = ann_index.get_total_vectors()

        # Get final progress statistics
        final_stats = progress_tracker.get_progress_stats()

        logger.info("=" * 60)
        logger.info("Covers indexing process completed successfully!")
        logger.info("=" * 60)
        
        # Final progress summary
        log_progress_summary(logger, progress_tracker, is_batch_complete=False)
        
        logger.info("-" * 60)
        logger.info("FINAL STATISTICS:")
        logger.info(f"Total entries requested in chunk: {chunk_total}")
        logger.info(f"Successfully indexed (new): {total_successful}")
        logger.info(f"Errors: {total_errors}")
        logger.info(f"Success rate (new indexings): {final_stats['success_rate']:.1f}%")
        logger.info(f"Total processing time: {progress_tracker.format_time(final_stats['elapsed_time'])}")
        logger.info(f"Average processing rate: {final_stats['successful_per_second']:.1f} entries/sec")
        if total_successful > 0:
            logger.info(f"Average time per successful entry: {progress_tracker.format_time(final_stats['avg_entry_time'])}")

        logger.info("-" * 60)
        logger.info(f"Final Index count: {final_total_vectors} vectors")

        # --- NEW INDEX COUNT CHECK AND WARNING ---
        if final_total_vectors == 0:
            logger.error(
                "\n*** FATAL INDEXING ERROR: FAISS index contains 0 vectors. ***"
            )
            logger.error(
                "This is why your search is failing with 'k must be positive and <= 0'."
            )
            logger.error("ACTIONS REQUIRED:")
            logger.error(
                "1. Run the script with the '--verbose' flag to check the logs for 'Failed to process' messages."
            )
            logger.error(
                "2. Verify that the image paths in your image directory are correct and accessible by OpenCV."
            )
            logger.error(
                "3. Ensure all dependencies (OpenCV, skimage, numpy, FAISS, DuckDB) are correctly installed."
            )
            logger.error("4. Check the size of the images you are trying to index.")
            logger.error("5. Verify that the JSON metadata contains valid 'final_filename' and 'image_url_modified' fields.")

        logger.info(f"FAISS index saved to: {faiss_index_path}")
        logger.info(f"Metadata database saved to: {db_path}")

        # Performance summary
        if total_successful > 0:
            throughput = total_successful / total_time
            logger.info(f"Processing throughput: {throughput:.1f} entries/second")

        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.warning("Indexing process interrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Indexing process failed: {str(e)}")
        logger.exception("Full error details:")
        sys.exit(1)


if __name__ == "__main__":
    main()
