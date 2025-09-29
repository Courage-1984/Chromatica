"""
Helper script to run build_index.py in chunks for very large datasets.
"""

import subprocess
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run build_index.py in chunks")
    parser.add_argument("image_directory", help="Directory containing images")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000000,
        help="Number of images per chunk (default: 1,000,000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for processing (default: 1,000)",
    )
    parser.add_argument(
        "--start-chunk",
        type=int,
        default=0,
        help="Chunk number to start from (default: 0)",
    )
    args = parser.parse_args()

    while True:
        start_idx = args.start_chunk * args.chunk_size
        end_idx = start_idx + args.chunk_size

        cmd = [
            sys.executable,
            "scripts/build_index.py",
            args.image_directory,
            "--start-index",
            str(start_idx),
            "--end-index",
            str(end_idx),
            "--batch-size",
            str(args.batch_size),
        ]

        # Append flag for all chunks after the first
        if args.start_chunk > 0:
            cmd.append("--append")

        print(f"\nProcessing chunk {args.start_chunk}: images {start_idx} to {end_idx}")
        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(
                f"\nError processing chunk {args.start_chunk}. Please retry from this chunk."
            )
            sys.exit(1)

        args.start_chunk += 1
        choice = input("\nContinue to next chunk? (y/n): ").lower()
        if choice != "y":
            break


if __name__ == "__main__":
    main()
