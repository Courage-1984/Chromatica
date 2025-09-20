# Evaluation Harness Documentation

## Overview

The Chromatica evaluation harness (`scripts/evaluate.py`) provides comprehensive testing and benchmarking capabilities for the color search engine. It measures both performance metrics (latency) and quality metrics (precision, recall) to ensure the system meets production requirements.

## Features

### Performance Metrics

- **Latency Measurement**: Precise timing of search operations
- **P95/P99 Latency**: Statistical analysis of response times
- **Component Timing**: Breakdown of ANN search vs. reranking time
- **Memory Usage**: Monitoring during evaluation

### Quality Metrics

- **Precision@K**: Accuracy of top-K results
- **Recall@K**: Coverage of relevant results
- **Ground Truth Support**: Comparison against labeled data
- **Query-Specific Analysis**: Individual query performance

### Comprehensive Logging

- **Structured Logging**: Detailed operation logs
- **Result Export**: JSON format for further analysis
- **Progress Tracking**: Real-time evaluation status
- **Error Handling**: Robust error reporting and recovery

## Usage

### Basic Evaluation

```bash
# Activate virtual environment
venv311\Scripts\activate

# Run evaluation with default test queries
python scripts/evaluate.py

# Run with custom query file
python scripts/evaluate.py --queries datasets/test-queries.json
```

### Advanced Evaluation

```bash
# Run with ground truth for quality metrics
python scripts/evaluate.py \
  --queries datasets/test-queries.json \
  --ground-truth datasets/ground-truth.json

# Custom parameters
python scripts/evaluate.py \
  --k 20 \
  --log-level DEBUG \
  --index-path index/custom_index.faiss

# Create sample test queries
python scripts/evaluate.py --create-sample-queries
```

### Command Line Options

| Option                    | Description                              | Default                        |
| ------------------------- | ---------------------------------------- | ------------------------------ |
| `--queries`               | Path to test queries JSON file           | `datasets/test-queries.json`   |
| `--ground-truth`          | Path to ground truth JSON file           | None (optional)                |
| `--index-path`            | Path to FAISS index file                 | `index/chromatica_index.faiss` |
| `--metadata-path`         | Path to DuckDB metadata file             | `index/chromatica_metadata.db` |
| `--k`                     | Number of results per query              | 10                             |
| `--create-sample-queries` | Generate sample queries and exit         | False                          |
| `--log-level`             | Logging level (DEBUG/INFO/WARNING/ERROR) | INFO                           |

## Test Data Format

### Query Format

Test queries are stored in JSON format with the following structure:

```json
[
  {
    "query_id": "q001",
    "colors": ["FF0000", "00FF00", "0000FF"],
    "weights": [0.4, 0.4, 0.2],
    "description": "Primary RGB colors with balanced weights"
  },
  ...
]
```

**Required Fields:**

- `query_id`: Unique identifier for the query
- `colors`: List of hex color codes (e.g., "FF0000" for red)

**Optional Fields:**

- `weights`: List of weights for each color (defaults to equal weights)
- `description`: Human-readable description of the query

### Ground Truth Format

Ground truth data maps query IDs to relevant image IDs:

```json
{
  "q001": ["img_000001", "img_000015", "img_000023"],
  "q002": ["img_000002", "img_000016", "img_000024"],
  ...
}
```

## Performance Targets

### Latency Targets

- **P95 Latency**: < 450ms (primary target)
- **P99 Latency**: < 1000ms
- **Mean Latency**: < 300ms

### Quality Targets

- **Precision@10**: > 0.8 (with ground truth)
- **Recall@10**: > 0.6 (with ground truth)

### Component Breakdown

- **ANN Search**: < 150ms (P95)
- **Reranking**: < 300ms (P95)
- **Total Overhead**: < 50ms

## Output Format

### Console Output

```
============================================================
🎯 CHROMATICA EVALUATION RESULTS
============================================================

📊 OVERVIEW
Total queries evaluated: 20

⏱️  LATENCY METRICS
Mean latency:     245.32 ms
Median latency:   238.15 ms
P95 latency:      412.50 ms ✅ PASS
P99 latency:      567.20 ms
Min latency:      156.30 ms
Max latency:      589.40 ms

🎯 QUALITY METRICS
Mean Precision@10: 0.850
Mean Recall@10:    0.720
Queries with GT:   20

🎯 PERFORMANCE TARGETS
P95 latency target: 450 ms
P95 latency actual: 412.50 ms ✅ PASS
Precision@10 target: 0.8
Precision@10 actual: 0.850 ✅ PASS
============================================================
```

### Log Files

**`logs/evaluation.log`**: Detailed operation logs

```
2024-01-15 10:30:15 - __main__ - INFO - Starting evaluation of 20 queries...
2024-01-15 10:30:15 - __main__ - INFO - Processing query 1/20: q001
2024-01-15 10:30:15 - __main__ - INFO - Completed 10/20 queries
```

**`logs/evaluation_results.json`**: Raw evaluation data

```json
{
  "total_queries": 20,
  "latency_stats": {
    "mean_ms": 245.32,
    "p95_ms": 412.50,
    ...
  },
  "quality_stats": {
    "mean_precision_at_10": 0.850,
    "mean_recall_at_10": 0.720,
    ...
  },
  "query_results": [...]
}
```

## Implementation Details

### Search Pipeline

1. **Query Processing**: Convert hex colors to Lab histograms
2. **ANN Search**: FAISS IndexIVFPQ retrieval of candidates
3. **Reranking**: Sinkhorn-EMD distance calculation
4. **Result Formatting**: Return ranked image IDs

### Timing Measurement

```python
# Component-level timing
ann_start = time.perf_counter()
distances, indices = ann_index.search(query_histogram, RERANK_K)
ann_time = (time.perf_counter() - ann_start) * 1000

# Total latency
total_time = (time.perf_counter() - start_time) * 1000
```

### Quality Metrics

```python
def calculate_precision_at_k(retrieved, relevant, k):
    top_k = retrieved[:k]
    relevant_count = sum(1 for img_id in top_k if img_id in relevant)
    return relevant_count / min(k, len(top_k))
```

## Creating Test Data

### Sample Query Generation

```bash
# Generate 50 sample queries
python scripts/evaluate.py --create-sample-queries --queries datasets/my-queries.json
```

The script generates diverse color palettes:

- Primary colors (RGB, CMY)
- Pastel combinations
- Dark themes
- Monochromatic gradients
- Single color queries

### Ground Truth Creation

For quality evaluation, create ground truth labels:

1. **Manual Labeling**: Expert annotation of relevant images
2. **Semi-Automatic**: Use similarity thresholds
3. **Crowdsourcing**: Multiple annotators for consensus

Example ground truth creation:

```python
# For each query, identify relevant images
ground_truth = {
    "q001": ["img_000001", "img_000015", "img_000023"],
    "q002": ["img_000002", "img_000016", "img_000024"],
    # ...
}
```

## Benchmarking Workflows

### Development Testing

```bash
# Quick evaluation during development
python scripts/evaluate.py --queries datasets/quick-test.json --k 5
```

### Performance Regression Testing

```bash
# Compare against baseline
python scripts/evaluate.py --queries datasets/regression-tests.json
# Compare results with previous runs
```

### Production Validation

```bash
# Full evaluation with all metrics
python scripts/evaluate.py \
  --queries datasets/production-queries.json \
  --ground-truth datasets/production-ground-truth.json \
  --k 20
```

## Troubleshooting

### Common Issues

**Index Not Found**

```
FileNotFoundError: IndexIVFPQ file not found: index/chromatica_index.faiss
```

Solution: Build index first using `python scripts/build_index.py`

**Insufficient Training Data**

```
RuntimeError: IndexIVFPQ training failed: Number of training points (100) should be at least as large as number of clusters (256)
```

Solution: Use more training data or reduce `IVFPQ_NLIST` in config

**Query Format Error**

```
ValueError: Query 0 missing required field: colors
```

Solution: Ensure query JSON has required fields

### Debug Mode

```bash
# Enable detailed logging
python scripts/evaluate.py --log-level DEBUG

# Check individual query processing
python scripts/evaluate.py --queries datasets/single-query.json --log-level DEBUG
```

## Integration with CI/CD

### Automated Testing

```yaml
# GitHub Actions example
- name: Run Evaluation
  run: |
    venv311\Scripts\activate
    python scripts/evaluate.py --queries datasets/ci-tests.json
    # Check if P95 latency < 450ms
    python -c "
    import json
    with open('logs/evaluation_results.json') as f:
        results = json.load(f)
    assert results['latency_stats']['p95_ms'] < 450, 'P95 latency too high'
    "
```

### Performance Monitoring

```bash
# Regular performance checks
python scripts/evaluate.py --queries datasets/monitoring-queries.json
# Alert if performance degrades
```

## Future Enhancements

### Planned Features

1. **A/B Testing**: Compare different configurations
2. **Statistical Significance**: Confidence intervals for metrics
3. **Visualization**: Charts and graphs for results
4. **Automated Reporting**: Email/Slack notifications
5. **Load Testing**: Concurrent query simulation

### Custom Metrics

- **Diversity Metrics**: Result set variety
- **Coverage Metrics**: Query space coverage
- **User Satisfaction**: Click-through rates (with user data)

## Conclusion

The evaluation harness provides comprehensive testing capabilities for the Chromatica color search engine, ensuring both performance and quality requirements are met. It supports various testing scenarios from development to production validation, with detailed logging and result analysis capabilities.

Regular evaluation helps maintain system performance and catch regressions early, supporting the continuous improvement of the color search engine.
