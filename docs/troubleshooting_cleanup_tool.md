# Chromatica Output Cleanup Tool - Troubleshooting Guide

## Overview

This guide provides comprehensive troubleshooting information for the Chromatica Output Cleanup Tool (`tools/cleanup_outputs.py`). It covers common issues, error scenarios, and resolution steps to help users effectively use the cleanup tool.

## Common Issues and Solutions

### 1. Interactive Mode Hanging or Not Responding

#### Symptoms

- Tool starts but doesn't show the interactive menu
- Tool appears to freeze after displaying scan results
- No response to keyboard input

#### Causes

- Running in non-interactive environment (CI/CD, automated scripts)
- Terminal doesn't support interactive input
- Input buffer issues

#### Solutions

**Option 1: Use Command Line Mode**

```bash
# Instead of interactive mode, use specific options
python tools/cleanup_outputs.py --logs --dry-run
python tools/cleanup_outputs.py --datasets --confirm
```

**Option 2: Test Interactive Mode**

```bash
# Test with simulated input
echo "0" | python tools/cleanup_outputs.py
```

**Option 3: Check Terminal Support**

```bash
# Ensure you're in an interactive terminal
python -c "import sys; print('Interactive:', sys.stdin.isatty())"
```

### 2. Permission Errors During Cleanup

#### Symptoms

- Error messages like "Permission denied" or "Access denied"
- Some files deleted but others fail
- Tool stops with permission error

#### Causes

- Files are in use by another process
- Insufficient file system permissions
- Files are read-only or system-protected

#### Solutions

**Option 1: Use Elevated Privileges**

```bash
# On Windows (PowerShell as Administrator)
python tools/cleanup_outputs.py --logs --confirm

# On Linux/Mac
sudo python tools/cleanup_outputs.py --logs --confirm
```

**Option 2: Check File Usage**

```bash
# On Windows - check if files are in use
# Close any applications that might be using the files
# Restart your development environment

# On Linux/Mac
lsof | grep "path/to/file"
```

**Option 3: Manual Cleanup**

```bash
# Use dry-run to see what would be deleted
python tools/cleanup_outputs.py --logs --dry-run

# Manually delete files that can be deleted
# The tool will continue with remaining files
```

### 3. Large File Cleanup Performance Issues

#### Symptoms

- Tool takes a very long time to scan or delete files
- Memory usage increases significantly
- Tool appears to hang on large operations

#### Causes

- Very large index files (100+ MB)
- Large number of cache files (7000+)
- Network storage with slow I/O

#### Solutions

**Option 1: Selective Cleanup**

```bash
# Clean smaller file types first
python tools/cleanup_outputs.py --logs --reports --confirm

# Then clean larger files separately
python tools/cleanup_outputs.py --test_index --confirm
```

**Option 2: Use Dry Run First**

```bash
# Preview the operation to estimate time
python tools/cleanup_outputs.py --all --dry-run

# Clean specific types based on size
python tools/cleanup_outputs.py --cache --confirm
```

**Option 3: Manual Large File Cleanup**

```bash
# For very large index files, consider manual deletion
rm -rf test_index/
python tools/cleanup_outputs.py --histograms --reports --logs --confirm
```

### 4. Configuration Validation Errors

#### Symptoms

- Error: "Configuration validation failed"
- Import errors related to config module
- Tool fails to start

#### Causes

- Missing or corrupted configuration files
- Python path issues
- Virtual environment not activated

#### Solutions

**Option 1: Check Virtual Environment**

```bash
# Ensure virtual environment is activated
venv311\Scripts\activate  # Windows
source venv311/bin/activate  # Linux/Mac

# Verify Python path
python -c "import sys; print(sys.path)"
```

**Option 2: Validate Configuration**

```bash
# Test configuration directly
python -c "from src.chromatica.utils.config import validate_config; validate_config()"
```

**Option 3: Check Project Structure**

```bash
# Ensure you're in the project root
pwd  # Should show Chromatica directory
ls src/chromatica/utils/config.py  # Should exist
```

### 5. File Not Found Errors

#### Symptoms

- "File not found" errors during cleanup
- Tool reports files that don't exist
- Inconsistent file discovery

#### Causes

- Files were deleted by another process
- Path resolution issues
- Symbolic link problems

#### Solutions

**Option 1: Refresh File Discovery**

```bash
# Run a fresh scan
python tools/cleanup_outputs.py --dry-run

# Check if files still exist
ls -la datasets/*/histograms/
ls -la logs/
```

**Option 2: Check Path Resolution**

```bash
# Verify project root detection
python -c "from pathlib import Path; print(Path.cwd())"

# Check if running from correct directory
python tools/cleanup_outputs.py --project-root .
```

### 6. Logging Issues

#### Symptoms

- No log file created
- Log file is empty
- Logging errors in console

#### Causes

- Logs directory doesn't exist
- Permission issues with log file
- Logging configuration problems

#### Solutions

**Option 1: Create Logs Directory**

```bash
# Ensure logs directory exists
mkdir -p logs
chmod 755 logs
```

**Option 2: Check Log File Permissions**

```bash
# Check if log file can be written
touch logs/cleanup.log
ls -la logs/cleanup.log
```

**Option 3: Manual Logging Test**

```bash
# Test logging functionality
python -c "
import logging
logging.basicConfig(filename='logs/test.log', level=logging.INFO)
logging.info('Test log entry')
"
```

## Advanced Troubleshooting

### Debug Mode

Enable detailed debugging information:

```bash
# Set debug environment variable
export PYTHONPATH=src:$PYTHONPATH
python -u tools/cleanup_outputs.py --logs --dry-run
```

### Memory Issues

For systems with limited memory:

```bash
# Clean smaller batches
python tools/cleanup_outputs.py --logs --confirm
python tools/cleanup_outputs.py --reports --confirm
python tools/cleanup_outputs.py --histograms --confirm
```

### Network Storage Issues

For projects on network storage:

```bash
# Use local cleanup for large operations
# Copy project to local storage temporarily
# Run cleanup locally
# Sync back if needed
```

## Error Code Reference

### Common Error Messages

| Error Message                     | Cause                            | Solution                                        |
| --------------------------------- | -------------------------------- | ----------------------------------------------- |
| "Configuration validation failed" | Config module issues             | Check virtual environment and project structure |
| "Permission denied"               | File system permissions          | Use elevated privileges or check file usage     |
| "File not found"                  | Path resolution issues           | Verify project root and file existence          |
| "Interactive mode not supported"  | Non-interactive environment      | Use command line options instead                |
| "No output files found"           | Clean project or wrong directory | Verify you're in the correct project directory  |

### Exit Codes

- **0**: Success
- **1**: General error (check logs)
- **2**: Configuration error
- **3**: Permission error
- **4**: File system error

## Prevention and Best Practices

### Regular Maintenance

1. **Regular Cleanup**: Run cleanup regularly to prevent accumulation
2. **Monitor Disk Space**: Check disk usage before large operations
3. **Backup Important Files**: Keep backups of important index files
4. **Test with Dry Run**: Always use `--dry-run` for large operations

### Safe Usage Patterns

```bash
# Safe cleanup workflow
python tools/cleanup_outputs.py --all --dry-run  # Preview
python tools/cleanup_outputs.py --logs --confirm  # Start small
python tools/cleanup_outputs.py --cache --confirm  # Clean cache
python tools/cleanup_outputs.py --datasets --confirm  # Clean datasets
```

### Environment Setup

```bash
# Proper environment setup
venv311\Scripts\activate  # Activate virtual environment
cd /path/to/Chromatica  # Ensure correct directory
python tools/cleanup_outputs.py --help  # Test tool availability
```

## Getting Help

### Log Analysis

Check the cleanup log for detailed information:

```bash
# View recent cleanup operations
tail -f logs/cleanup.log

# Search for specific errors
grep -i "error" logs/cleanup.log
grep -i "permission" logs/cleanup.log
```

### Diagnostic Commands

```bash
# System information
python --version
pip list | grep -E "(faiss|duckdb|numpy)"

# Project structure
find . -name "*.py" -path "./src/*" | head -10
ls -la datasets/
ls -la logs/
```

### Support Resources

1. **Documentation**: `docs/tools_cleanup_outputs.md`
2. **Progress Report**: `docs/progress.md`
3. **Critical Instructions**: `docs/.cursor/critical_instructions.md`
4. **Tool README**: `tools/README.md`

## Recovery Procedures

### Complete Reset

If cleanup tool is completely broken:

```bash
# Reset to clean state
git checkout HEAD -- tools/cleanup_outputs.py
python tools/cleanup_outputs.py --help
```

### Manual Cleanup

If automated cleanup fails:

```bash
# Manual cleanup commands
rm -rf datasets/*/histograms/
rm -rf datasets/*/reports/
rm -rf logs/*.log
rm -rf test_index/
find . -name "__pycache__" -type d -exec rm -rf {} +
```

### Rebuild Project

If project state is corrupted:

```bash
# Clean everything and rebuild
python tools/cleanup_outputs.py --all --confirm
python scripts/build_index.py
```

## Conclusion

The Chromatica Output Cleanup Tool is designed to be robust and user-friendly, but occasional issues may arise. This troubleshooting guide provides comprehensive solutions for common problems and best practices for safe usage.

For additional help or to report new issues, refer to the project documentation or create an issue in the project repository.
