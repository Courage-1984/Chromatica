#!/usr/bin/env python3
"""
Chromatica Output Cleanup Tool

This tool provides comprehensive cleanup functionality for managing output files
generated during development and testing. It supports selective deletion of
specific output types or complete cleanup of all generated files.

Usage:
    python tools/cleanup_outputs.py [options]

Examples:
    # Interactive mode - choose what to clean
    python tools/cleanup_outputs.py

    # Clean all outputs with confirmation
    python tools/cleanup_outputs.py --all --confirm

    # Clean specific output types
    python tools/cleanup_outputs.py --histograms --reports --logs

    # Dry run to see what would be deleted
    python tools/cleanup_outputs.py --all --dry-run

    # Clean test datasets outputs only
    python tools/cleanup_outputs.py --datasets

Author: Chromatica Development Team
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chromatica.utils.config import validate_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/cleanup.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


class OutputCleanupTool:
    """Comprehensive tool for cleaning up Chromatica output files."""

    def __init__(self, project_root: Path = None):
        """Initialize the cleanup tool.

        Args:
            project_root: Path to the project root directory
        """
        self.project_root = project_root or Path(__file__).parent.parent
        # Validate configuration on initialization
        validate_config()

        # Define output directories and their descriptions
        self.output_dirs = {
            "histograms": {
                "path": "datasets/*/histograms",
                "description": "Generated histogram files (.npy)",
                "pattern": "**/histograms/**/*.npy",
            },
            "reports": {
                "path": "datasets/*/reports",
                "description": "Analysis reports and logs",
                "pattern": "**/reports/**/*",
            },
            "logs": {
                "path": "logs",
                "description": "Application and build logs",
                "pattern": "logs/**/*.log",
            },
            "test_index": {
                "path": "test_index",
                "description": "FAISS index and DuckDB metadata files",
                "pattern": "test_index/**/*",
            },
            "cache": {
                "path": "__pycache__",
                "description": "Python bytecode cache files",
                "pattern": "**/__pycache__/**/*",
            },
            "temp": {
                "path": "temp",
                "description": "Temporary files and directories",
                "pattern": "temp/**/*",
            },
        }

        # Define file patterns for different output types
        self.file_patterns = {
            "histograms": ["**/*.npy"],
            "reports": ["**/*.txt", "**/*.json", "**/*.csv", "**/*.html"],
            "logs": ["**/*.log"],
            "index_files": ["**/*.faiss", "**/*.db"],
            "cache": ["**/__pycache__/**/*.pyc", "**/__pycache__/**/*.pyo"],
            "temp": ["**/*.tmp", "**/*.temp", "**/.DS_Store", "**/Thumbs.db"],
        }

    def scan_outputs(self) -> Dict[str, List[Path]]:
        """Scan for all output files and directories.

        Returns:
            Dictionary mapping output types to lists of found files/directories
        """
        logger.info("Scanning for output files...")
        found_outputs = {}

        for output_type, config in self.output_dirs.items():
            found_files = []

            # Use glob patterns to find files
            if "pattern" in config:
                pattern = config["pattern"]
                for file_path in self.project_root.glob(pattern):
                    if file_path.is_file() or file_path.is_dir():
                        found_files.append(file_path)

            # Also check specific paths
            if "path" in config:
                path_pattern = config["path"]
                for path in self.project_root.glob(path_pattern):
                    if path.exists():
                        found_files.append(path)

            found_outputs[output_type] = found_files
            logger.info(f"Found {len(found_files)} {output_type} items")

        return found_outputs

    def get_file_sizes(self, file_paths: List[Path]) -> Tuple[int, str]:
        """Calculate total size of files and directories.

        Args:
            file_paths: List of file/directory paths

        Returns:
            Tuple of (total_size_bytes, formatted_size_string)
        """
        total_size = 0

        for path in file_paths:
            if path.is_file():
                total_size += path.stat().st_size
            elif path.is_dir():
                for file_path in path.rglob("*"):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size

        # Format size
        if total_size < 1024:
            size_str = f"{total_size} B"
        elif total_size < 1024**2:
            size_str = f"{total_size/1024:.1f} KB"
        elif total_size < 1024**3:
            size_str = f"{total_size/(1024**2):.1f} MB"
        else:
            size_str = f"{total_size/(1024**3):.1f} GB"

        return total_size, size_str

    def display_scan_results(
        self, found_outputs: Dict[str, List[Path]], selected_types: List[str] = None
    ) -> None:
        """Display scan results in a formatted table.

        Args:
            found_outputs: Dictionary of found output files by type
            selected_types: Optional list of specific types to display
        """
        print("\n" + "=" * 80)
        print("CHROMATICA OUTPUT CLEANUP TOOL")
        print("=" * 80)
        print(f"Project Root: {self.project_root}")
        print(
            f"Scan Time: {logging.Formatter().formatTime(logging.LogRecord('', 0, '', 0, '', (), None))}"
        )
        print("-" * 80)

        total_files = 0
        total_size = 0

        # Filter to selected types if specified
        output_items = found_outputs.items()
        if selected_types:
            output_items = [
                (t, files) for t, files in found_outputs.items() if t in selected_types
            ]

        for output_type, files in output_items:
            if files:
                size_bytes, size_str = self.get_file_sizes(files)
                total_size += size_bytes
                total_files += len(files)

                print(
                    f"{output_type.upper():<15} {len(files):>6} items  {size_str:>10}"
                )

                # Show first few items as examples
                for i, file_path in enumerate(files[:3]):
                    rel_path = file_path.relative_to(self.project_root)
                    print(f"  {'  └─' if i == len(files)-1 else '  ├─'} {rel_path}")

                if len(files) > 3:
                    print(f"  └─ ... and {len(files) - 3} more items")
            else:
                print(f"{output_type.upper():<15} {0:>6} items  {'0 B':>10}")

        print("-" * 80)
        # Format total size properly
        if total_size < 1024:
            total_size_str = f"{total_size} B"
        elif total_size < 1024**2:
            total_size_str = f"{total_size/1024:.1f} KB"
        elif total_size < 1024**3:
            total_size_str = f"{total_size/(1024**2):.1f} MB"
        else:
            total_size_str = f"{total_size/(1024**3):.1f} GB"

        print(f"{'TOTAL':<15} {total_files:>6} items  {total_size_str:>10}")
        print("=" * 80)

    def interactive_cleanup(self, found_outputs: Dict[str, List[Path]]) -> None:
        """Interactive cleanup mode with user selection.

        Args:
            found_outputs: Dictionary of found output files by type
        """
        print("\nInteractive Cleanup Mode")
        print("-" * 40)

        # Show available options
        available_types = [t for t, files in found_outputs.items() if files]

        if not available_types:
            print("No output files found to clean.")
            return

        print("Available output types to clean:")
        for i, output_type in enumerate(available_types, 1):
            count = len(found_outputs[output_type])
            size_bytes, size_str = self.get_file_sizes(found_outputs[output_type])
            print(f"  {i}. {output_type} ({count} items, {size_str})")

        print(f"  {len(available_types) + 1}. All outputs")
        print("  0. Cancel")

        # Get user selection
        while True:
            try:
                choice = input(
                    f"\nSelect option (0-{len(available_types) + 1}): "
                ).strip()
                choice_num = int(choice)

                if choice_num == 0:
                    print("Cleanup cancelled.")
                    return
                elif choice_num == len(available_types) + 1:
                    # Clean all
                    selected_types = available_types
                    break
                elif 1 <= choice_num <= len(available_types):
                    selected_types = [available_types[choice_num - 1]]
                    break
                else:
                    print(f"Invalid choice. Please enter 0-{len(available_types) + 1}")
            except ValueError:
                print("Invalid input. Please enter a number.")

        # Confirm deletion
        print(f"\nSelected: {', '.join(selected_types)}")
        total_items = sum(len(found_outputs[t]) for t in selected_types)
        total_size_bytes = sum(
            self.get_file_sizes(found_outputs[t])[0] for t in selected_types
        )
        _, total_size_str = self.get_file_sizes([Path() for _ in range(total_items)])

        print(f"Total items to delete: {total_items}")
        print(f"Total size: {total_size_str}")

        confirm = (
            input("\nAre you sure you want to delete these files? (yes/no): ")
            .strip()
            .lower()
        )
        if confirm in ["yes", "y"]:
            self.clean_outputs(selected_types, found_outputs)
        else:
            print("Cleanup cancelled.")

    def clean_outputs(
        self,
        output_types: List[str],
        found_outputs: Dict[str, List[Path]],
        dry_run: bool = False,
    ) -> None:
        """Clean specified output types.

        Args:
            output_types: List of output types to clean
            found_outputs: Dictionary of found output files by type
            dry_run: If True, only show what would be deleted
        """
        if dry_run:
            print("\nDRY RUN MODE - No files will actually be deleted")
            print("-" * 50)

        total_deleted = 0
        total_size_deleted = 0

        for output_type in output_types:
            if output_type not in found_outputs or not found_outputs[output_type]:
                continue

            files = found_outputs[output_type]
            print(f"\nCleaning {output_type}...")

            for file_path in files:
                try:
                    rel_path = file_path.relative_to(self.project_root)

                    if dry_run:
                        print(f"  [DRY RUN] Would delete: {rel_path}")
                    else:
                        if file_path.is_file():
                            file_size = file_path.stat().st_size
                            file_path.unlink()
                            total_size_deleted += file_size
                            print(f"  Deleted file: {rel_path}")
                        elif file_path.is_dir():
                            # Calculate directory size before deletion
                            dir_size = sum(
                                f.stat().st_size
                                for f in file_path.rglob("*")
                                if f.is_file()
                            )
                            shutil.rmtree(file_path)
                            total_size_deleted += dir_size
                            print(f"  Deleted directory: {rel_path}")

                    total_deleted += 1

                except Exception as e:
                    logger.error(f"Error deleting {file_path}: {e}")
                    print(f"  ERROR: Could not delete {rel_path}: {e}")

        # Summary
        if not dry_run:
            _, size_str = self.get_file_sizes([Path() for _ in range(total_deleted)])
            print(f"\nCleanup completed!")
            print(f"Deleted {total_deleted} items")
            print(f"Freed {size_str} of disk space")
        else:
            print(f"\nDry run completed - would delete {total_deleted} items")

    def create_cleanup_script(self, output_types: List[str]) -> None:
        """Create a standalone cleanup script for the selected output types.

        Args:
            output_types: List of output types to include in the script
        """
        script_content = f'''#!/usr/bin/env python3
"""
Auto-generated cleanup script for Chromatica outputs.
Generated by cleanup_outputs.py
"""

import os
import shutil
from pathlib import Path

def cleanup():
    """Clean up Chromatica output files."""
    project_root = Path(__file__).parent.parent
    
    # Output types to clean: {', '.join(output_types)}
    cleanup_paths = [
'''

        for output_type in output_types:
            if output_type in self.output_dirs:
                config = self.output_dirs[output_type]
                if "path" in config:
                    script_content += f"        '{config['path']}',\n"

        script_content += """    ]
    
    deleted_count = 0
    for pattern in cleanup_paths:
        for path in project_root.glob(pattern):
            if path.exists():
                try:
                    if path.is_file():
                        path.unlink()
                    elif path.is_dir():
                        shutil.rmtree(path)
                    print(f"Deleted: {path.relative_to(project_root)}")
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting {path}: {e}")
    
    print(f"Cleanup completed. Deleted {deleted_count} items.")

if __name__ == "__main__":
    cleanup()
"""

        script_path = self.project_root / "tools" / "quick_cleanup.py"
        with open(script_path, "w") as f:
            f.write(script_content)

        # Make executable
        os.chmod(script_path, 0o755)
        print(f"Created cleanup script: {script_path}")

    def run(self, args: argparse.Namespace) -> None:
        """Run the cleanup tool with the given arguments.

        Args:
            args: Parsed command line arguments
        """
        # Ensure logs directory exists
        logs_dir = self.project_root / "logs"
        logs_dir.mkdir(exist_ok=True)

        # Scan for outputs
        found_outputs = self.scan_outputs()

        # Determine what to clean
        if args.all:
            output_types = list(found_outputs.keys())
        elif args.output_types:
            output_types = args.output_types
        else:
            # Interactive mode
            self.display_scan_results(found_outputs)
            self.interactive_cleanup(found_outputs)
            return

        # Filter to only types that have files
        output_types = [t for t in output_types if found_outputs.get(t)]

        if not output_types:
            print("No output files found for the specified types.")
            return

        # Display results for selected types
        self.display_scan_results(found_outputs, selected_types=output_types)

        # Confirm deletion unless --confirm is specified
        if not args.confirm and not args.dry_run:
            total_items = sum(len(found_outputs[t]) for t in output_types)
            print(f"\nAbout to delete {total_items} items.")
            confirm = input("Continue? (yes/no): ").strip().lower()
            if confirm not in ["yes", "y"]:
                print("Cleanup cancelled.")
                return

        # Perform cleanup
        self.clean_outputs(output_types, found_outputs, dry_run=args.dry_run)

        # Create cleanup script if requested
        if args.create_script:
            self.create_cleanup_script(output_types)


def main():
    """Main entry point for the cleanup tool."""
    parser = argparse.ArgumentParser(
        description="Chromatica Output Cleanup Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Interactive mode
  %(prog)s --all --confirm          # Clean all outputs
  %(prog)s --histograms --reports   # Clean specific types
  %(prog)s --all --dry-run          # See what would be deleted
  %(prog)s --datasets --create-script # Create cleanup script
        """,
    )

    # Output type selection
    parser.add_argument("--all", action="store_true", help="Clean all output types")

    parser.add_argument(
        "--output-types",
        nargs="+",
        choices=["histograms", "reports", "logs", "test_index", "cache", "temp"],
        help="Specific output types to clean",
    )

    # Convenience shortcuts
    parser.add_argument(
        "--histograms", action="store_true", help="Clean histogram files"
    )

    parser.add_argument("--reports", action="store_true", help="Clean report files")

    parser.add_argument("--logs", action="store_true", help="Clean log files")

    parser.add_argument(
        "--datasets",
        action="store_true",
        help="Clean dataset outputs (histograms and reports)",
    )

    parser.add_argument("--index", action="store_true", help="Clean index files")

    # Behavior options
    parser.add_argument(
        "--confirm", action="store_true", help="Skip confirmation prompt"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )

    parser.add_argument(
        "--create-script",
        action="store_true",
        help="Create a standalone cleanup script",
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        help="Project root directory (default: auto-detect)",
    )

    args = parser.parse_args()

    # Handle convenience shortcuts
    if args.histograms or args.reports or args.logs or args.datasets or args.index:
        output_types = []
        if args.histograms:
            output_types.append("histograms")
        if args.reports:
            output_types.append("reports")
        if args.logs:
            output_types.append("logs")
        if args.datasets:
            output_types.extend(["histograms", "reports"])
        if args.index:
            output_types.append("test_index")

        args.output_types = output_types

    # Initialize and run tool
    tool = OutputCleanupTool(project_root=args.project_root)

    try:
        tool.run(args)
    except KeyboardInterrupt:
        print("\nCleanup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Cleanup tool error: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
