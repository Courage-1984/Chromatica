# Chromatica Output Cleanup Tool

## Overview

The Chromatica Output Cleanup Tool (`tools/cleanup_outputs.py`) is a comprehensive utility for managing and cleaning up output files generated during development, testing, and production operations. This tool helps maintain a clean development environment by providing selective or complete removal of generated files.

## Features

### Core Functionality

- **Selective Cleanup**: Choose specific output types to clean
- **Interactive Mode**: User-friendly interface for guided cleanup
- **Batch Operations**: Clean multiple output types simultaneously
- **Safety Features**: Confirmation prompts and dry-run mode
- **Size Reporting**: Shows disk space usage and freed space
- **Script Generation**: Create standalone cleanup scripts

### Supported Output Types

- **Histograms**: Generated histogram files (`.npy`) from dataset processing
- **Reports**: Analysis reports, logs, and documentation files
- **Logs**: Application and build log files
- **Test Index**: FAISS index and DuckDB metadata files
- **Cache**: Python bytecode cache files (`__pycache__`)
- **Temp**: Temporary files and system artifacts

## Usage

### Basic Usage

```bash
# Interactive mode - guided cleanup selection
python tools/cleanup_outputs.py

# Clean all outputs with confirmation
python tools/cleanup_outputs.py --all --confirm

# Clean specific output types
python tools/cleanup_outputs.py --histograms --reports --logs

# Dry run to preview what would be deleted
python tools/cleanup_outputs.py --all --dry-run
```

### Command Line Options

#### Output Type Selection

- `--all`: Clean all output types
- `--output-types TYPE [TYPE ...]`: Specify exact output types to clean
- `--histograms`: Clean histogram files only
- `--reports`: Clean report files only
- `--logs`: Clean log files only
- `--datasets`: Clean dataset outputs (histograms and reports)
- `--index`: Clean index files only

#### Behavior Options

- `--confirm`: Skip confirmation prompt (use with caution)
- `--dry-run`: Show what would be deleted without actually deleting
- `--create-script`: Generate a standalone cleanup script
- `--project-root PATH`: Specify project root directory

### Examples

#### Interactive Cleanup

```bash
python tools/cleanup_outputs.py
```

This launches an interactive mode where you can:

1. View all available output types and their sizes
2. Select specific types to clean
3. Confirm the deletion operation

#### Batch Cleanup

```bash
# Clean all outputs
python tools/cleanup_outputs.py --all --confirm

# Clean only dataset-related outputs
python tools/cleanup_outputs.py --datasets

# Clean specific types
python tools/cleanup_outputs.py --histograms --reports --logs
```

#### Safe Preview

```bash
# See what would be deleted without actually deleting
python tools/cleanup_outputs.py --all --dry-run

# Preview specific cleanup
python tools/cleanup_outputs.py --datasets --dry-run
```

#### Script Generation

```bash
# Create a standalone cleanup script for dataset outputs
python tools/cleanup_outputs.py --datasets --create-script
```

## Output Types and Locations

### Histograms

- **Location**: `datasets/*/histograms/`
- **Files**: `*.npy` files containing color histograms
- **Size**: Typically 4-10 KB per file
- **Purpose**: Generated during dataset processing for color analysis

### Reports

- **Location**: `datasets/*/reports/`
- **Files**: Analysis reports, JSON summaries, CSV data
- **Size**: Varies (1-100 KB per file)
- **Purpose**: Generated analysis and validation reports

### Logs

- **Location**: `logs/`
- **Files**: `*.log` files from application runs
- **Size**: Can be large (1-100 MB per file)
- **Purpose**: Application logging and debugging information

### Test Index

- **Location**: `test_index/`
- **Files**: `*.faiss` (FAISS index) and `*.db` (DuckDB metadata)
- **Size**: Large (10-1000 MB depending on dataset size)
- **Purpose**: Search index and metadata storage

### Cache Files

- **Location**: `**/__pycache__/`
- **Files**: `*.pyc`, `*.pyo` bytecode files
- **Size**: Small (1-10 KB per file)
- **Purpose**: Python bytecode cache for faster imports

### Temporary Files

- **Location**: Various locations
- **Files**: `*.tmp`, `*.temp`, `.DS_Store`, `Thumbs.db`
- **Size**: Small (1-100 KB per file)
- **Purpose**: System and application temporary files

## Safety Features

### Confirmation Prompts

- Interactive mode requires explicit confirmation
- Shows total items and disk space to be freed
- Clear "yes/no" prompts for destructive operations

### Dry Run Mode

- Preview what would be deleted without making changes
- Shows file paths and sizes
- Safe way to understand cleanup scope

### Error Handling

- Graceful handling of permission errors
- Continues cleanup even if individual files fail
- Detailed error logging and reporting

### Logging

- All operations logged to `logs/cleanup.log`
- Timestamps and detailed operation records
- Error tracking and debugging information

## Integration with Development Workflow

### Pre-Development Cleanup

```bash
# Clean all outputs before starting fresh development
python tools/cleanup_outputs.py --all --confirm
```

### Post-Testing Cleanup

```bash
# Clean test outputs after validation
python tools/cleanup_outputs.py --datasets --logs
```

### Maintenance Cleanup

```bash
# Regular maintenance - clean cache and temp files
python tools/cleanup_outputs.py --cache --temp
```

### CI/CD Integration

```bash
# Automated cleanup in build scripts
python tools/cleanup_outputs.py --all --confirm --project-root /build/path
```

## Performance Considerations

### Large Datasets

- For large datasets (5000+ images), cleanup can take several minutes
- Index files can be very large (100+ MB) and take time to delete
- Consider using `--dry-run` first to estimate cleanup time

### Disk Space

- Histogram files: ~5-10 KB each
- Index files: 10-1000 MB depending on dataset size
- Log files: Can accumulate to 100+ MB over time
- Cache files: Small but numerous

### Network Storage

- If project is on network storage, cleanup may be slower
- Consider local cleanup for large operations
- Use `--create-script` for remote execution

## Troubleshooting

### Permission Errors

```bash
# If you get permission errors, try:
python tools/cleanup_outputs.py --dry-run  # Check what's causing issues
sudo python tools/cleanup_outputs.py --all --confirm  # Use elevated privileges
```

### Large File Cleanup

```bash
# For very large index files, consider manual deletion:
rm -rf test_index/
python tools/cleanup_outputs.py --histograms --reports --logs
```

### Partial Cleanup

```bash
# If cleanup is interrupted, you can resume:
python tools/cleanup_outputs.py --all --dry-run  # See what remains
python tools/cleanup_outputs.py --all --confirm  # Complete cleanup
```

## Best Practices

### Regular Maintenance

- Run cleanup weekly during active development
- Clean logs regularly to prevent disk space issues
- Remove old test outputs before running new tests

### Before Major Operations

- Clean all outputs before building new indexes
- Clear cache before performance testing
- Remove old reports before generating new ones

### Safety First

- Always use `--dry-run` for large operations
- Keep backups of important index files
- Use interactive mode when unsure

### Automation

- Create cleanup scripts for common operations
- Integrate into build and test scripts
- Use `--confirm` flag in automated environments

## Script Generation

The tool can generate standalone cleanup scripts for specific operations:

```bash
# Generate script for dataset cleanup
python tools/cleanup_outputs.py --datasets --create-script

# This creates tools/quick_cleanup.py
python tools/quick_cleanup.py  # Run the generated script
```

Generated scripts are:

- Self-contained and portable
- Include error handling
- Provide progress feedback
- Safe for automated execution

## Integration with Other Tools

### Testing Tools

- Clean outputs before running test suites
- Remove old test results for clean comparisons
- Clear cache for consistent test environments

### Build Scripts

- Clean outputs before building indexes
- Remove old artifacts before deployment
- Clear logs before production runs

### Development Tools

- Clean outputs before code reviews
- Remove temporary files before commits
- Clear cache for clean development environment

## Future Enhancements

### Planned Features

- **Selective File Patterns**: More granular file selection
- **Backup Options**: Create backups before cleanup
- **Scheduling**: Automated cleanup scheduling
- **Metrics**: Detailed cleanup statistics and reporting
- **Integration**: Better integration with IDE and development tools

### Configuration

- **Custom Patterns**: User-defined file patterns
- **Exclusion Lists**: Files to never delete
- **Size Limits**: Automatic cleanup based on size thresholds
- **Age Limits**: Cleanup based on file age

## Conclusion

The Chromatica Output Cleanup Tool provides a comprehensive solution for managing output files throughout the development lifecycle. With its safety features, flexible options, and integration capabilities, it helps maintain a clean and efficient development environment while preventing accidental data loss.

For questions or issues, refer to the troubleshooting guide or check the cleanup logs in `logs/cleanup.log`.
