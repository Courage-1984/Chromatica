# Chromatica Scripts

This directory contains standalone scripts for various Chromatica project tasks, including bulk operations, evaluation, data management, and development utilities.

## 📁 Scripts Overview

The scripts directory provides automation and utility functions for:

- **Bulk Operations**: Processing large datasets and batch operations
- **Evaluation**: Performance testing and validation across different scales
- **Data Management**: Dataset preparation, validation, and maintenance
- **Development Utilities**: Code generation, testing, and debugging tools
- **Production Operations**: Deployment, monitoring, and maintenance scripts

## 🚀 Available Scripts

### Data Processing Scripts

#### Bulk Indexing Script

```bash
# Process and index entire datasets
python scripts/bulk_index.py --dataset datasets/test-dataset-5000 --output index_output/
```

**Features:**

- Batch processing of large image collections
- Progress tracking and error handling
- Memory-efficient processing with configurable batch sizes
- Output validation and quality checks

#### Dataset Validation Script

```bash
# Validate dataset integrity and quality
python scripts/validate_dataset.py --dataset datasets/test-dataset-5000 --report validation_report.json
```

**Features:**

- Image format validation and corruption detection
- Histogram quality assessment
- Dataset statistics and metadata generation
- Performance benchmarking across dataset sizes

### Evaluation Scripts

#### Performance Benchmarking Script

```bash
# Run comprehensive performance tests
python scripts/benchmark_performance.py --datasets 20,50,200,5000 --output benchmark_results.json
```

**Features:**

- Scalability testing across different dataset sizes
- Memory usage profiling and optimization
- Throughput and latency measurements
- Performance regression detection

#### Quality Assessment Script

```bash
# Assess histogram quality and consistency
python scripts/assess_quality.py --dataset datasets/test-dataset-5000 --metrics entropy,sparsity,distribution
```

**Features:**

- Histogram quality metrics calculation
- Statistical analysis and outlier detection
- Quality trend analysis across datasets
- Automated quality reporting

### Development Scripts

#### Code Generation Script

```bash
# Generate boilerplate code for new modules
python scripts/generate_module.py --module new_feature --type core
```

**Features:**

- Automated module scaffolding
- Template-based code generation
- Consistent structure and documentation
- Import statement management

#### Testing Scripts

```bash
# Run comprehensive test suites
python scripts/run_tests.py --modules core,indexing,utils --coverage --performance
```

**Features:**

- Modular test execution
- Coverage reporting and analysis
- Performance test integration
- Test result aggregation and reporting

### Production Scripts

#### Deployment Script

```bash
# Deploy Chromatica to production environment
python scripts/deploy.py --environment production --config production_config.yaml
```

**Features:**

- Environment-specific configuration
- Dependency management and validation
- Service startup and health checks
- Rollback capabilities

#### Monitoring Script

```bash
# Monitor system health and performance
python scripts/monitor.py --metrics cpu,memory,latency --interval 60 --output monitoring.log
```

**Features:**

- Real-time system monitoring
- Performance metrics collection
- Alert generation and notification
- Historical data logging

## 🔧 Script Configuration

### Environment Variables

```bash
# Set script environment
export CHROMATICA_ENV=development

# Configure logging
export CHROMATICA_LOG_LEVEL=INFO

# Set data directories
export CHROMATICA_DATA_DIR=datasets/
export CHROMATICA_OUTPUT_DIR=output/
```

### Configuration Files

Scripts use YAML configuration files for flexible parameter management:

```yaml
# config/scripts.yaml
bulk_indexing:
  batch_size: 100
  max_workers: 4
  memory_limit: "2GB"

performance_testing:
  dataset_sizes: [20, 50, 200, 5000]
  iterations: 5
  warmup_runs: 2

quality_assessment:
  metrics: ["entropy", "sparsity", "distribution"]
  thresholds:
    entropy_min: 3.0
    sparsity_max: 0.8
```

## 📊 Script Output

### Standard Output Format

All scripts follow consistent output patterns:

```json
{
  "script": "bulk_index.py",
  "version": "1.0.0",
  "timestamp": "2024-12-01T14:30:22Z",
  "parameters": {
    "dataset": "test-dataset-5000",
    "batch_size": 100
  },
  "results": {
    "total_processed": 5000,
    "successful": 4998,
    "failed": 2,
    "processing_time": 1200.5,
    "memory_peak": "1.2GB"
  },
  "errors": [
    {
      "file": "corrupted_image.jpg",
      "error": "Invalid image format"
    }
  ]
}
```

### Logging

Scripts provide comprehensive logging:

```bash
# Enable verbose logging
python scripts/bulk_index.py --verbose --log-level DEBUG

# Log to file
python scripts/bulk_index.py --log-file bulk_index.log

# Structured logging
python scripts/bulk_index.py --log-format json
```

## 🧪 Script Testing

### Unit Testing

```bash
# Test individual scripts
pytest tests/test_scripts/

# Test specific script
pytest tests/test_scripts/test_bulk_index.py
```

### Integration Testing

```bash
# Test script integration with main modules
pytest tests/integration/test_script_integration.py

# Test end-to-end script workflows
pytest tests/integration/test_script_workflows.py
```

### Performance Testing

```bash
# Test script performance
pytest tests/performance/test_script_performance.py

# Benchmark script execution
python scripts/benchmark_scripts.py --scripts bulk_index,validate_dataset
```

## 🔍 Script Development

### Creating New Scripts

Follow the established pattern for new scripts:

```python
#!/usr/bin/env python3
"""
Script description and purpose.

Usage:
    python scripts/script_name.py [options]

Author: Chromatica Development Team
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chromatica.utils.config import validate_config

def main():
    """Main script execution function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Script description")
    parser.add_argument("--input", required=True, help="Input path")
    parser.add_argument("--output", required=True, help="Output path")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    # Execute script logic
    try:
        # Script implementation
        pass
    except Exception as e:
        logging.error(f"Script execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### Script Standards

1. **Documentation**: Comprehensive docstrings and usage examples
2. **Error Handling**: Graceful error handling with informative messages
3. **Logging**: Consistent logging with configurable levels
4. **Configuration**: Support for configuration files and environment variables
5. **Validation**: Input validation and parameter checking
6. **Testing**: Unit tests for all script functionality

## 🚨 Common Issues

### Import Errors

```bash
# Ensure you're in the project root
cd /path/to/Chromatica

# Activate virtual environment
venv311\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Permission Issues

```bash
# Check file permissions
ls -la scripts/

# Make scripts executable
chmod +x scripts/*.py

# Check directory permissions
ls -la datasets/ output/
```

### Memory Issues

```bash
# Reduce batch size for large datasets
python scripts/bulk_index.py --batch-size 50

# Monitor memory usage
python scripts/bulk_index.py --memory-limit "1GB"
```

## 📈 Performance Optimization

### Batch Processing

```bash
# Optimize batch size for your system
python scripts/bulk_index.py --batch-size 200 --max-workers 8

# Use memory-efficient processing
python scripts/bulk_index.py --memory-efficient --chunk-size 1000
```

### Parallel Processing

```bash
# Enable parallel processing
python scripts/bulk_index.py --parallel --max-workers 4

# Use process pool for CPU-intensive tasks
python scripts/bulk_index.py --process-pool --max-processes 4
```

## 🔮 Future Enhancements

### Planned Scripts

- **Data Augmentation**: Automated dataset expansion and enhancement
- **Model Training**: Scripts for training and fine-tuning models
- **API Testing**: Automated API endpoint testing and validation
- **Deployment Automation**: CI/CD pipeline integration scripts
- **Performance Monitoring**: Real-time performance tracking and alerting

### Script Improvements

- **Web Interface**: Browser-based script execution and monitoring
- **Scheduling**: Automated script execution with cron-like functionality
- **Dependency Management**: Automated dependency resolution and installation
- **Result Visualization**: Interactive charts and graphs for script outputs
- **Integration**: Better integration with external tools and services

---

**Last Updated**: December 2024  
**Script Count**: 10+ production-ready scripts  
**Status**: Comprehensive scripting infrastructure implemented
