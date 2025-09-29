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
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
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
) -> Dict[str, Any]:
    """
    Process a batch of images and add them to the index and metadata store.

    Args:
        image_files: List of image file paths to process
        batch_size: Number of images to process in this batch
        ann_index: FAISS index for storing transformed histograms
        metadata_store: DuckDB store for metadata and raw histograms

    Returns:
        Dictionary with batch processing statistics
    """
    logger = logging.getLogger(__name__)

    batch_start_time = time.time()
    successful_count = 0
    error_count = 0
    total_processing_time = 0

    # Prepare batch data for metadata store
    metadata_batch = []

    for i, image_path in enumerate(image_files[:batch_size]):
        image_start_time = time.time()

        try:
            logger.info(
                f"Processing image {i+1}/{len(image_files[:batch_size])}: {image_path.name}"
            )

            # Process image and generate histogram
            histogram = process_image(str(image_path))

            # Validate histogram
            validate_processed_image(histogram, str(image_path))

            # Generate unique image ID (using filename without extension)
            image_id = image_path.stem

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

            logger.info(f"Successfully indexed batch: {len(metadata_batch)} images")

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

        # Initialize components with append mode if specified
        use_simple_index = chunk_total < 2000 and not args.append
        if use_simple_index:
            logger.info(f"Using IndexFlatL2 for small dataset ({chunk_total} images)")
        else:
            logger.info(f"Using IndexIVFPQ for dataset")

        ann_index = AnnIndex(
            dimension=TOTAL_BINS,
            use_simple_index=use_simple_index,
            load_existing=args.append,
            index_path=str(faiss_index_path) if args.append else None,
        )
        metadata_store = MetadataStore(db_path=str(db_path), create_new=not args.append)

        logger.info("Components initialized successfully")

        # Train the FAISS index with a sample of images
        logger.info("Training FAISS index with representative data...")
        training_sample_size = min(
            1000, total_images
        )  # Use up to 1000 images for training
        training_files = image_files[:training_sample_size]

        # Process training images to get histograms
        training_histograms = []
        for i, image_path in enumerate(training_files):
            try:
                logger.info(
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
            raise RuntimeError("No valid training histograms could be generated")

        # Convert to numpy array and train the index
        training_data = np.array(training_histograms)
        ann_index.train(training_data)

        logger.info(
            f"FAISS index training completed with {len(training_histograms)} histograms"
        )

        # Process remaining images in batches (skip training images)
        start_time = time.time()
        total_successful = 0
        total_errors = 0

        # Add training images to the index and metadata store
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

                metadata_store.add_batch(training_metadata)
                total_successful += len(training_histograms)

                logger.info(
                    f"Successfully indexed {len(training_histograms)} training images"
                )
            except Exception as e:
                logger.error(f"Failed to index training images: {str(e)}")
                total_errors += len(training_histograms)

        # Process remaining images in batches
        remaining_images = image_files[training_sample_size:]
        remaining_count = len(remaining_images)

        logger.info(f"Processing {remaining_count} remaining images in batches...")

        for batch_start in range(0, remaining_count, args.batch_size):
            batch_end = min(batch_start + args.batch_size, remaining_count)
            batch_files = remaining_images[batch_start:batch_end]

            logger.info(
                f"Processing batch {batch_start//args.batch_size + 1}: "
                f"images {batch_start + 1}-{batch_end} of {remaining_count} remaining"
            )

            # Process batch
            batch_stats = process_image_batch(
                batch_files, len(batch_files), ann_index, metadata_store
            )

            total_successful += batch_stats["successful_count"]
            total_errors += batch_stats["error_count"]

            logger.info(
                f"Batch completed: {batch_stats['successful_count']} successful, "
                f"{batch_stats['error_count']} errors, "
                f"time: {batch_stats['batch_time']:.2f}s"
            )

        # Save index and close connections
        logger.info("Saving index and closing connections...")

        ann_index.save(str(faiss_index_path))
        metadata_store.close()

        # Final statistics
        total_time = time.time() - start_time
        success_rate = (
            (total_successful / total_images) * 100 if total_images > 0 else 0
        )

        logger.info("=" * 60)
        logger.info("Indexing process completed successfully!")
        logger.info("=" * 60)
        logger.info(f"Total images processed: {total_images}")
        logger.info(f"Successful: {total_successful}")
        logger.info(f"Errors: {total_errors}")
        logger.info(f"Success rate: {success_rate:.1f}%")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Average time per image: {total_time/total_images:.3f}s")
        logger.info(f"FAISS index saved to: {faiss_index_path}")
        logger.info(f"Metadata database saved to: {db_path}")
        logger.info(f"Index contains {ann_index.get_total_vectors()} vectors")

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
