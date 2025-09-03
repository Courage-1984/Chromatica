# Chromatica API

This module provides the FastAPI application for the Chromatica color search engine, exposing the search functionality via REST endpoints.

## Quick Start

### 1. Start the API Server

```bash
# Activate virtual environment
venv311\Scripts\activate

# Start the server
uvicorn src.chromatica.api.main:app --reload
```

The API will be available at `http://localhost:8000`

### 2. API Documentation

Once the server is running, you can access:

- **Interactive API docs**: `http://localhost:8000/docs` (Swagger UI)
- **Alternative docs**: `http://localhost:8000/redoc` (ReDoc)

## API Endpoints

### Root Endpoint

**GET /** - API information and status

```bash
curl http://localhost:8000/
```

**Response:**

```json
{
  "message": "Chromatica Color Search Engine API",
  "version": "1.0.0",
  "status": "running",
  "endpoints": {
    "search": "/search",
    "docs": "/docs",
    "health": "/health"
  }
}
```

### Health Check

**GET /health** - System health and status

```bash
curl http://localhost:8000/health
```

**Response:**

```json
{
  "status": "healthy",
  "message": "Search system is operational",
  "data": {
    "images_in_store": 5000,
    "vectors_in_index": 5000
  },
  "timestamp": 1703123456.789
}
```

### Search Endpoint

**GET /search** - Search for images by color similarity

#### Query Parameters

- **`colors`** (required): Comma-separated hex color codes (without `#`)
- **`weights`** (required): Comma-separated float weights corresponding to colors
- **`k`** (optional): Number of results to return (default: 50, max: 200)
- **`fuzz`** (optional): Query fuzziness multiplier (default: 1.0)

#### Example Requests

**Basic search with two colors:**

```bash
curl "http://localhost:8000/search?colors=ea6a81,f6d727&weights=0.49,0.51&k=20"
```

**Single color search:**

```bash
curl "http://localhost:8000/search?colors=FF0000&weights=1.0&k=10"
```

**Multi-color search with varied weights:**

```bash
curl "http://localhost:8000/search?colors=00FF00,0000FF,FFFF00&weights=0.4,0.3,0.3&k=15"
```

#### Response Format

The response follows the exact structure specified in Section H of the critical instructions:

```json
{
  "query_id": "a1b2c3d4-e5f6-7890-g1h2-i3j4k5l6m7n8",
  "query": {
    "colors": ["#ea6a81", "#f6d727"],
    "weights": [0.49, 0.51]
  },
  "results_count": 20,
  "results": [
    {
      "image_id": "img_abc123",
      "distance": 0.087,
      "dominant_colors": ["#e96d80", "#f5d52b", "#ffffff"]
    },
    {
      "image_id": "img_def456",
      "distance": 0.091,
      "dominant_colors": ["#d05f71", "#f9e045"]
    }
  ],
  "metadata": {
    "ann_time_ms": 110,
    "rerank_time_ms": 285,
    "total_time_ms": 395
  }
}
```

## Testing

Use the provided test script to verify the API functionality:

```bash
# Test all endpoints
python tools/test_api.py

# Test specific endpoints
python tools/test_api.py --health
python tools/test_api.py --search
python tools/test_api.py --invalid
```

## Architecture

The API implements the two-stage search architecture:

1. **ANN Search Stage**: Uses FAISS HNSW index for fast candidate retrieval
2. **Reranking Stage**: Uses Sinkhorn-EMD for high-fidelity distance calculation

### Startup Process

1. Load FAISS index from `test_index/chromatica_index.faiss`
2. Load DuckDB metadata store from `test_index/chromatica_metadata.db`
3. Validate system components
4. Start accepting requests

### Error Handling

- **400 Bad Request**: Invalid query parameters (colors, weights, etc.)
- **503 Service Unavailable**: Search system not initialized
- **500 Internal Server Error**: Unexpected errors during search

## Configuration

The API uses the configuration constants defined in `src/chromatica/utils/config.py`:

- **TOTAL_BINS**: 1152 (8×12×12 Lab color space bins)
- **RERANK_K**: 200 (default candidates for reranking)
- **HNSW_M**: 32 (FAISS HNSW graph neighbors)

## Performance

Expected performance characteristics:

- **ANN stage**: ~1-5ms for 200 candidates
- **Reranking stage**: ~100-500ms for 200 candidates
- **Total search time**: ~150-600ms for typical queries

## Development

### Adding New Endpoints

1. Define Pydantic models for request/response validation
2. Create the endpoint function with proper error handling
3. Add comprehensive logging
4. Update this README with endpoint documentation

### Testing New Features

1. Add test cases to `tools/test_api.py`
2. Test with various input combinations
3. Verify error handling for edge cases
4. Check performance impact

## Troubleshooting

### Common Issues

**"Search components not initialized"**

- Ensure the FAISS index and DuckDB store exist
- Check file paths in the startup code
- Verify the indexing pipeline has been run

**"Could not connect to API server"**

- Make sure uvicorn is running
- Check if port 8000 is available
- Verify virtual environment is activated

**"Invalid parameters"**

- Check hex color format (3 or 6 characters, no #)
- Ensure weights are positive numbers
- Verify colors and weights arrays have matching lengths

### Logs

The API provides comprehensive logging at INFO level. Check the console output for:

- Startup and initialization messages
- Search query processing details
- Performance timing information
- Error details and stack traces
