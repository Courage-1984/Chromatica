# Example: python scripts/build_index.py ./data/unsplash-lite

"""
Offline indexing script for the Chromatica color search engine.

This script processes a directory of images and populates both the FAISS HNSW index
and DuckDB metadata store. It serves as the main entry point for building the
offline index that powers the color search functionality.

The script follows the two-stage search architecture:
1. FAISS index stores Hellinger-transformed histograms for fast ANN search
2. DuckDB stores raw histograms and metadata for accurate reranking

Usage:
    python scripts/build_index.py <image_directory> [--output-dir <output_dir>] [--batch-size <batch_size>]

Example:
    python scripts/build_index.py ./datasets/test-dataset-5000 --output-dir ./index --batch-size 100

Features:
- Batch processing for memory efficiency
- Comprehensive logging and progress tracking
- Error handling with graceful degradation
- Validation of processed histograms
- Performance monitoring and timing
- Automatic output directory creation
- **ADDED: Duplicate image detection in append mode to prevent duplicate indexing.**
"""

import argparse
import logging
import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import numpy as np

# Add src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chromatica.indexing.pipeline import process_image, validate_processed_image
from chromatica.indexing.store import AnnIndex, MetadataStore
from chromatica.utils.config import TOTAL_BINS


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

    # Set up console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)

    # Set up file handler for detailed logging
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    file_handler = logging.FileHandler(log_dir / f"build_index_{int(time.time())}.log")
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


def get_image_files(directory: Path) -> List[Path]:
    """
    Get all image files from the specified directory.

    Args:
        directory: Path to the directory containing images

    Returns:
        List of image file paths

    Raises:
        ValueError: If directory doesn't exist or contains no images
    """
    if not directory.exists():
        raise ValueError(f"Directory does not exist: {directory}")

    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")

    # Supported image extensions
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

    image_files = []
    for ext in image_extensions:
        image_files.extend(directory.glob(f"*{ext}"))
        image_files.extend(directory.glob(f"*{ext.upper()}"))

    # Remove duplicates and sort for consistent processing
    image_files = sorted(list(set(image_files)))

    if not image_files:
        raise ValueError(f"No image files found in directory: {directory}")

    logger = logging.getLogger(__name__)
    logger.info(f"Found {len(image_files)} image files in {directory}")

    return image_files


def process_image_batch(
    image_files: List[Path],
    batch_size: int,
    ann_index: AnnIndex,
    metadata_store: MetadataStore,
    is_append_mode: bool = False,  # Added is_append_mode flag
) -> Dict[str, Any]:
    """
    Process a batch of images and add them to the index and metadata store.

    In append mode, this function uses the metadata_store.check_existence()
    method to skip images that are already indexed, preventing duplicates.

    Args:
        image_files: List of image file paths to process
        batch_size: Number of images to process in this batch
        ann_index: FAISS index for storing transformed histograms
        metadata_store: DuckDB store for metadata and raw histograms
        is_append_mode: If True, check for and skip images already in the store.

    Returns:
        Dictionary with batch processing statistics
    """
    logger = logging.getLogger(__name__)

    batch_start_time = time.time()
    successful_count = 0
    error_count = 0
    total_processing_time = 0

    files_to_process = image_files[:batch_size]

    # 1. Check for existing images if in append mode
    existing_ids: Set[str] = set()
    if is_append_mode and files_to_process:
        image_ids_to_check = [p.stem for p in files_to_process]
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
    for i, image_path in enumerate(files_to_process):
        image_id = image_path.stem

        # Skip if already exists in append mode
        if is_append_mode and image_id in existing_ids:
            logger.debug(f"Skipping {image_path.name}: already indexed.")
            continue

        image_start_time = time.time()

        try:
            logger.info(
                f"Processing image {i+1}/{len(files_to_process)}: {image_path.name}"
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
                }
            )

            successful_count += 1
            processing_time = time.time() - image_start_time
            total_processing_time += processing_time

            logger.debug(
                f"Successfully processed {image_path.name} in {processing_time:.3f}s"
            )

        except Exception as e:
            error_count += 1
            logger.error(f"Failed to process {image_path.name}: {str(e)}")
            continue

    # Add histograms to FAISS index (Hellinger-transformed)
    if metadata_batch:
        try:
            # Extract histograms for FAISS
            histograms = np.array([record["histogram"] for record in metadata_batch])

            # Add to FAISS index (Hellinger transform applied automatically)
            ann_index.add(histograms)

            # Add metadata to DuckDB
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
    """Main function for the offline indexing script."""
    parser = argparse.ArgumentParser(
        description="Build offline index for Chromatica color search engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/build_index.py ./datasets/test-dataset-20
  python scripts/build_index.py ./datasets/test-dataset-5000 --output-dir ./index --batch-size 200
  python scripts/build_index.py ./data/unsplash-lite --verbose
  python scripts/build_index.py ./huge-dataset --start-index 0 --end-index 1000000 --append
        """,
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
        help="Starting index in the image list (for chunked processing)",
    )

    parser.add_argument(
        "--end-index",
        type=int,
        help="Ending index in the image list (for chunked processing)",
    )

    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing index instead of creating new ones",
    )

    args = parser.parse_args()

    # Set up logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Starting Chromatica offline indexing process")
    logger.info("=" * 60)

    try:
        # Validate and prepare paths
        image_dir = Path(args.image_directory).resolve()
        output_dir = Path(args.output_dir).resolve()

        logger.info(f"Image directory: {image_dir}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Batch size: {args.batch_size}")

        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory created/verified: {output_dir}")

        # Get list of image files
        image_files = get_image_files(image_dir)
        total_images = len(image_files)

        logger.info(f"Total images found: {total_images}")

        # Handle chunked processing
        start_idx = args.start_index if args.start_index is not None else 0
        end_idx = args.end_index if args.end_index is not None else total_images

        if start_idx < 0 or end_idx > total_images or start_idx >= end_idx:
            raise ValueError(
                f"Invalid start/end indices. Must be: 0 <= start ({start_idx}) < end ({end_idx}) <= total ({total_images})"
            )

        # Adjust image files list for chunked processing
        image_files = image_files[start_idx:end_idx]
        chunk_total = len(image_files)

        logger.info(
            f"Processing chunk: images {start_idx} to {end_idx} (total: {chunk_total} images)"
        )

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
        training_files = []
        training_data = np.array([])

        # Train the FAISS index with a sample of images
        # Training is only needed if the index is NOT already trained and loaded (i.e. not in append mode)
        if hasattr(ann_index.index, "is_trained") and not ann_index.index.is_trained:
            logger.info("Training FAISS index with representative data...")
            # Use up to 1000 images for training from the current chunk
            training_sample_size = min(1000, chunk_total)
            training_files = image_files[:training_sample_size]

            # Process training images to get histograms
            for i, image_path in enumerate(training_files):
                try:
                    logger.debug(
                        f"Processing training image {i+1}/{len(training_files)}: {image_path.name}"
                    )
                    histogram = process_image(str(image_path))
                    validate_processed_image(histogram, str(image_path))
                    training_histograms.append(histogram)
                except Exception as e:
                    logger.warning(
                        f"Failed to process training image {image_path.name}: {str(e)}"
                    )
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
                    # Add training histograms to FAISS index
                    ann_index.add(training_data)

                    # Add training metadata to DuckDB
                    training_metadata = []
                    for i, image_path in enumerate(
                        training_files[: len(training_histograms)]
                    ):
                        training_metadata.append(
                            {
                                "image_id": image_path.stem,
                                "file_path": str(image_path),
                                "histogram": training_histograms[i],
                                "file_size": image_path.stat().st_size,
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
        remaining_images = image_files[training_sample_size:]
        remaining_count = len(remaining_images)

        logger.info(f"Processing {remaining_count} remaining images in batches...")

        start_time = time.time()

        for batch_start in range(0, remaining_count, args.batch_size):
            batch_end = min(batch_start + args.batch_size, remaining_count)
            batch_files = remaining_images[batch_start:batch_end]

            logger.info(
                f"Processing batch {batch_start//args.batch_size + 1}: "
                f"images {batch_start + 1}-{batch_end} of {remaining_count} remaining"
            )

            # Process batch with de-duplication check
            batch_stats = process_image_batch(
                batch_files,
                len(batch_files),
                ann_index,
                metadata_store,
                is_append_mode=append_mode,
            )

            total_successful += batch_stats["successful_count"]
            total_errors += batch_stats["error_count"]

            logger.info(
                f"Batch completed: {batch_stats['successful_count']} successful (newly indexed), "
                f"{batch_stats['error_count']} errors, "
                f"time: {batch_stats['batch_time']:.2f}s"
            )

        # Save index and close connections
        logger.info("Saving index and closing connections...")

        ann_index.save(str(faiss_index_path))
        metadata_store.close()

        # Final statistics
        total_time = time.time() - start_time
        final_total_vectors = ann_index.get_total_vectors()

        # Calculate success rate based on total images in chunk
        success_rate = (total_successful / chunk_total) * 100 if chunk_total > 0 else 0

        logger.info("=" * 60)
        logger.info("Indexing process completed successfully!")
        logger.info("=" * 60)
        logger.info(f"Total images requested in chunk: {chunk_total}")
        logger.info(f"Successfully indexed (new): {total_successful}")
        logger.info(f"Errors: {total_errors}")
        logger.info(f"Success rate (new indexings): {success_rate:.1f}%")
        logger.info(f"Total time (indexing phase): {total_time:.2f}s")
        if total_successful > 0:
            logger.info(
                f"Average time per *new* image: {total_time/total_successful:.3f}s"
            )

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

        logger.info(f"FAISS index saved to: {faiss_index_path}")
        logger.info(f"Metadata database saved to: {db_path}")

        # Performance summary
        if total_successful > 0:
            throughput = total_successful / total_time
            logger.info(f"Processing throughput: {throughput:.1f} images/second")

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

