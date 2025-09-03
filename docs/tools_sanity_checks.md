# Sanity Check Script Documentation

## Overview

The `scripts/run_sanity_checks.py` script is a comprehensive validation tool that programmatically executes the four sanity checks defined in Section F of the critical instructions document. This script serves as a critical quality assurance tool to validate that the Chromatica color search engine is working correctly before deployment.

## Purpose

The sanity check script validates the core functionality of the color search engine by testing:

1. **Monochrome Color Queries**: Ensures single-color queries return semantically appropriate results
2. **Complementary Color Queries**: Validates that contrasting color combinations work correctly
3. **Weight Sensitivity**: Tests that different weight distributions produce meaningfully different results
4. **Subtle Hue Discrimination**: Verifies the system's ability to distinguish between similar colors

## Sanity Checks Implemented

### 1. Monochrome Red Query
- **Query**: 100% `#FF0000` (pure red)
- **Expected**: Should return images dominated by red colors
- **Validation**: Results should show red-dominant images with low distance scores

### 2. Complementary Colors Query
- **Query**: 50% `#0000FF` (blue) and 50% `#FFA500` (orange)
- **Expected**: Should return images featuring that contrast
- **Validation**: Results should show images with both blue and orange elements

### 3. Weight Sensitivity Test 1
- **Query**: 90% `#FF0000` (red) and 10% `#0000FF` (blue)
- **Expected**: Should yield red-dominant results
- **Validation**: Top results should prioritize red over blue

### 4. Weight Sensitivity Test 2
- **Query**: 10% `#FF0000` (red) and 90% `#0000FF` (blue)
- **Expected**: Should yield blue-dominant results
- **Validation**: Top results should prioritize blue over red

### 5. Subtle Hues Test
- **Query**: 50% `#FF0000` (red) and 50% `#EE0000` (slightly different red)
- **Expected**: Should test fine-grained color perception
- **Validation**: System should distinguish between very similar colors

## Usage

### Prerequisites
- Virtual environment activated: `venv311\Scripts\activate`
- All dependencies installed: `pip install -r requirements.txt`
- Test index available in `test_index/` directory
- Test datasets available in `datasets/` directory

### Basic Usage
```bash
# Activate virtual environment
venv311\Scripts\activate

# Run sanity checks with default settings
python scripts/run_sanity_checks.py

# Run with verbose logging
python scripts/run_sanity_checks.py --verbose

# Run with custom number of top results
python scripts/run_sanity_checks.py --top-k 10
```

### Command Line Options
- `--verbose, -v`: Enable verbose logging for detailed debugging
- `--top-k, -k N`: Number of top results to display (default: 5)

## Output and Logging

### Console Output
The script provides real-time feedback on:
- Progress through each sanity check
- Query details and parameters
- Search results with rankings
- Performance metrics
- Summary report

### Log Files
- **Primary Log**: `logs/sanity_checks.log`
- **Format**: Timestamped entries with structured logging
- **Levels**: INFO (default), DEBUG (with --verbose)

### Sample Output
```
2024-01-15 10:30:15 - __main__ - INFO - Starting Chromatica Sanity Checks
2024-01-15 10:30:15 - __main__ - INFO - ============================================================
2024-01-15 10:30:15 - __main__ - INFO - This script validates the search system's behavior using
2024-01-15 10:30:15 - __main__ - INFO - the four sanity checks defined in Section F of the plan.
2024-01-15 10:30:15 - __main__ - INFO - ============================================================

============================================================
SANITY CHECK: Monochrome Red Query
============================================================
Query Colors: ['FF0000']
Query Weights: [1.0]
Query histogram generated in 0.002s
Histogram shape: (1152,)
Histogram sum: 1.000000
Search completed in 0.045s
Retrieved 5 results

Top 5 Results:
 1. Image ID: test_7348262
    File: datasets/test-dataset-20/7348262.jpg
    Distance: 0.023456
    ANN Score: 0.023456
    Rank: 1
```

## Integration with Project

### Dependencies
- **Core Modules**: `chromatica.core.histogram`, `chromatica.core.query`
- **Search System**: `chromatica.search`, `chromatica.indexing.store`
- **Configuration**: `chromatica.utils.config`

### Test Index Requirements
- **FAISS Index**: `test_index/chromatica_index.faiss`
- **Metadata Store**: `test_index/chromatica_metadata.db`
- **Index Size**: Should contain processed test images

### Error Handling
The script includes comprehensive error handling for:
- Missing test index files
- Invalid query parameters
- Search system failures
- Performance issues

## Performance Monitoring

### Metrics Tracked
- **Query Time**: Histogram generation duration
- **Search Time**: FAISS search and reranking duration
- **Total Time**: End-to-end processing time
- **Result Count**: Number of images retrieved

### Performance Targets
- **Query Time**: < 10ms per histogram generation
- **Search Time**: < 100ms for top-5 results
- **Total Time**: < 150ms per sanity check

## Troubleshooting

### Common Issues

#### 1. Test Index Not Found
```
FileNotFoundError: Test index not found: test_index/chromatica_index.faiss
```
**Solution**: Ensure the test index has been built using the indexing pipeline

#### 2. Import Errors
```
ModuleNotFoundError: No module named 'chromatica'
```
**Solution**: Activate virtual environment and ensure all dependencies are installed

#### 3. Search Failures
```
RuntimeError: Search operation failed
```
**Solution**: Check that the FAISS index and metadata store are properly initialized

### Debug Mode
Use the `--verbose` flag to enable detailed logging:
```bash
python scripts/run_sanity_checks.py --verbose
```

This provides:
- Detailed function call traces
- Parameter validation information
- Performance breakdowns
- Error stack traces

## Validation Criteria

### Success Criteria
- All sanity checks complete without errors
- Query histograms are properly normalized (sum = 1.0)
- Search results are returned within performance targets
- Weight sensitivity tests show meaningful differences

### Failure Indicators
- Any sanity check throws an exception
- Query histograms fail validation
- Search operations timeout or fail
- Weight sensitivity tests show identical results

## Future Enhancements

### Planned Features
- **Visual Result Display**: Show actual images for top results
- **Automated Validation**: Compare results against expected color distributions
- **Performance Benchmarking**: Track performance over time
- **Integration Testing**: Test with live API endpoints

### Extensibility
The script is designed to be easily extended with:
- Additional sanity check types
- Custom validation criteria
- Performance profiling
- Result analysis tools

## Maintenance

### Regular Usage
- Run sanity checks after any major code changes
- Execute before deployment to production
- Use as part of continuous integration pipeline
- Monitor performance trends over time

### Updates
- Update sanity check parameters as needed
- Add new validation criteria as features are developed
- Maintain compatibility with evolving search system
- Document any changes to validation logic

## Related Documentation

- **Critical Instructions**: `docs/.cursor/critical_instructions.md`
- **Search System**: `docs/tools_test_search_system.md`
- **API Testing**: `docs/tools_test_api.md`
- **Troubleshooting**: `docs/troubleshooting.md`

---

*This documentation should be updated whenever the sanity check script is modified or new features are added.*
