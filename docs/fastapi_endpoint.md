# FastAPI Endpoint

## Chromatica Color Search Engine

---

## Overview

The FastAPI Endpoint provides a RESTful API interface for the Chromatica color search engine. This component exposes search functionality through HTTP endpoints, enabling integration with web applications, mobile apps, and other services.

### Key Features

- **RESTful Design**: Standard HTTP methods and status codes
- **Multiple Input Formats**: Support for image uploads and URL-based queries
- **Comprehensive Error Handling**: Detailed error messages and status codes
- **Performance Monitoring**: Request timing and response metrics
- **Async Support**: Non-blocking request processing
- **OpenAPI Documentation**: Auto-generated API documentation

---

## API Structure

### Base Configuration

```python
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Chromatica Color Search Engine",
    description="High-performance color-based image search API",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components
index: Optional[AnnIndex] = None
store: Optional[MetadataStore] = None
reranker: Optional[SinkhornReranker] = None
processor: Optional[QueryProcessor] = None
```

### Health Check Endpoint

```python
@app.get("/health")
async def health_check():
    """Check API health and component status."""

    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "faiss_index": index is not None,
            "metadata_store": store is not None,
            "reranker": reranker is not None,
            "query_processor": processor is not None
        }
    }

    # Check if all components are available
    if not all(health_status["components"].values()):
        health_status["status"] = "degraded"
        return JSONResponse(
            status_code=503,
            content=health_status
        )

    return health_status
```

---

## Search Endpoints

### Image Upload Search

```python
@app.post("/search/upload")
async def search_by_upload(
    file: UploadFile = File(...),
    k: int = 50,
    max_rerank: int = 200
):
    """Search by uploaded image file."""

    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPEG, PNG, etc.)"
        )

    # Validate file size (max 10MB)
    if file.size > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File size must be less than 10MB"
        )

    try:
        # Save uploaded file temporarily
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Process image
        query_histogram = processor.process_image_file(temp_path)

        # Perform search
        results = find_similar(
            query_histogram=query_histogram,
            index=index,
            store=store,
            k=k,
            max_rerank=max_rerank
        )

        # Clean up temporary file
        Path(temp_path).unlink()

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                'image_id': result.image_id,
                'file_path': result.file_path,
                'distance': result.distance,
                'rank': result.rank
            })

        return {
            'status': 'success',
            'query_type': 'image_upload',
            'filename': file.filename,
            'results': formatted_results,
            'total_results': len(formatted_results),
            'query_parameters': {
                'k': k,
                'max_rerank': max_rerank
            }
        }

    except Exception as e:
        # Clean up on error
        if Path(temp_path).exists():
            Path(temp_path).unlink()

        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )
```

### URL-based Search

```python
@app.post("/search/url")
async def search_by_url(
    url: str,
    k: int = 50,
    max_rerank: int = 200
):
    """Search by image URL."""

    # Validate URL
    if not url.startswith(('http://', 'https://')):
        raise HTTPException(
            status_code=400,
            detail="Invalid URL format"
        )

    try:
        # Process image from URL
        query_histogram = processor.process_image_url(url)

        # Perform search
        results = find_similar(
            query_histogram=query_histogram,
            index=index,
            store=store,
            k=k,
            max_rerank=max_rerank
        )

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                'image_id': result.image_id,
                'file_path': result.file_path,
                'distance': result.distance,
                'rank': result.rank
            })

        return {
            'status': 'success',
            'query_type': 'image_url',
            'url': url,
            'results': formatted_results,
            'total_results': len(formatted_results),
            'query_parameters': {
                'k': k,
                'max_rerank': max_rerank
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )
```

### Histogram-based Search

```python
@app.post("/search/histogram")
async def search_by_histogram(
    histogram: List[float],
    k: int = 50,
    max_rerank: int = 200
):
    """Search by pre-computed histogram."""

    # Validate histogram
    if len(histogram) != 1152:
        raise HTTPException(
            status_code=400,
            detail="Histogram must have exactly 1152 dimensions"
        )

    try:
        # Convert to numpy array
        query_histogram = np.array(histogram, dtype=np.float32)

        # Validate normalization
        if not np.allclose(query_histogram.sum(), 1.0, atol=1e-6):
            raise HTTPException(
                status_code=400,
                detail="Histogram must be normalized (sum to 1.0)"
            )

        # Perform search
        results = find_similar(
            query_histogram=query_histogram,
            index=index,
            store=store,
            k=k,
            max_rerank=max_rerank
        )

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                'image_id': result.image_id,
                'file_path': result.file_path,
                'distance': result.distance,
                'rank': result.rank
            })

        return {
            'status': 'success',
            'query_type': 'histogram',
            'results': formatted_results,
            'total_results': len(formatted_results),
            'query_parameters': {
                'k': k,
                'max_rerank': max_rerank
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )
```

---

## Utility Endpoints

### Index Information

```python
@app.get("/index/info")
async def get_index_info():
    """Get information about the search index."""

    if index is None or store is None:
        raise HTTPException(
            status_code=503,
            detail="Index not available"
        )

    try:
        # Get index statistics
        index_stats = {
            'total_vectors': index.ntotal,
            'dimension': index.d,
            'index_type': 'HNSW',
            'hnsw_m': index.hnsw.m,
            'hnsw_ef_construction': index.hnsw.efConstruction
        }

        # Get store statistics
        store_stats = store.get_statistics()

        return {
            'status': 'success',
            'index': index_stats,
            'store': store_stats,
            'timestamp': datetime.utcnow().isoformat()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get index info: {str(e)}"
        )
```

### Performance Metrics

```python
@app.get("/metrics/performance")
async def get_performance_metrics():
    """Get performance metrics for the search system."""

    if reranker is None:
        raise HTTPException(
            status_code=503,
            detail="Reranker not available"
        )

    try:
        # Get reranker performance stats
        reranker_stats = reranker.get_performance_stats()

        # Get system metrics
        import psutil
        system_stats = {
            'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent
        }

        return {
            'status': 'success',
            'reranker': reranker_stats,
            'system': system_stats,
            'timestamp': datetime.utcnow().isoformat()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get performance metrics: {str(e)}"
        )
```

---

## Error Handling

### Custom Exception Handlers

```python
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """Handle validation errors."""

    return JSONResponse(
        status_code=422,
        content={
            'status': 'error',
            'error_type': 'validation_error',
            'detail': 'Invalid request parameters',
            'errors': exc.errors()
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""

    return JSONResponse(
        status_code=exc.status_code,
        content={
            'status': 'error',
            'error_type': 'http_error',
            'detail': exc.detail,
            'status_code': exc.status_code
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""

    return JSONResponse(
        status_code=500,
        content={
            'status': 'error',
            'error_type': 'internal_error',
            'detail': 'Internal server error',
            'message': str(exc)
        }
    )
```

### Input Validation

```python
from pydantic import BaseModel, validator
from typing import List

class SearchRequest(BaseModel):
    """Base search request model."""

    k: int = 50
    max_rerank: int = 200

    @validator('k')
    def validate_k(cls, v):
        if v < 1 or v > 1000:
            raise ValueError('k must be between 1 and 1000')
        return v

    @validator('max_rerank')
    def validate_max_rerank(cls, v):
        if v < 1 or v > 1000:
            raise ValueError('max_rerank must be between 1 and 1000')
        return v

class HistogramSearchRequest(SearchRequest):
    """Histogram search request model."""

    histogram: List[float]

    @validator('histogram')
    def validate_histogram(cls, v):
        if len(v) != 1152:
            raise ValueError('Histogram must have exactly 1152 dimensions')
        return v
```

---

## Middleware and Configuration

### Request Logging

```python
import time
import logging
from fastapi import Request

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing information."""

    start_time = time.time()

    # Process request
    response = await call_next(request)

    # Calculate timing
    process_time = time.time() - start_time

    # Log request
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )

    # Add timing header
    response.headers["X-Process-Time"] = str(process_time)

    return response
```

### Rate Limiting

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Add rate limiting to app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Apply rate limiting to search endpoints
@app.post("/search/upload")
@limiter.limit("10/minute")
async def search_by_upload_limited(request: Request, file: UploadFile = File(...)):
    """Rate-limited image upload search."""
    return await search_by_upload(file)

@app.post("/search/url")
@limiter.limit("20/minute")
async def search_by_url_limited(request: Request, url: str):
    """Rate-limited URL search."""
    return await search_by_url(url)
```

---

## Testing and Development

### Running the Server

```python
if __name__ == "__main__":
    # Load components
    from chromatica.indexing.store import AnnIndex, MetadataStore
    from chromatica.core.rerank import SinkhornReranker
    from chromatica.core.query import QueryProcessor

    # Initialize components
    index = AnnIndex.load("test_index/chromatica_index.faiss")
    store = MetadataStore("test_index/chromatica_metadata.db")
    reranker = SinkhornReranker(epsilon=0.1)
    processor = QueryProcessor(max_dimension=256)

    # Run server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
```

### API Testing

```python
def test_search_endpoints():
    """Test search endpoints with sample data."""

    import requests
    import numpy as np

    base_url = "http://localhost:8000"

    # Test health check
    response = requests.get(f"{base_url}/health")
    assert response.status_code == 200

    # Test histogram search
    test_histogram = np.random.random(1152)
    test_histogram = test_histogram / test_histogram.sum()

    response = requests.post(
        f"{base_url}/search/histogram",
        json={
            "histogram": test_histogram.tolist(),
            "k": 10,
            "max_rerank": 100
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["results"]) <= 10

    print("API tests passed")
```

---

## Conclusion

The FastAPI Endpoint provides a robust, scalable REST API for the Chromatica color search engine:

- **Standards Compliance**: RESTful design with proper HTTP status codes
- **Flexibility**: Multiple input methods (upload, URL, histogram)
- **Reliability**: Comprehensive error handling and validation
- **Performance**: Async processing and rate limiting
- **Monitoring**: Health checks and performance metrics

The API successfully implements the endpoint specification from the critical instructions document, enabling easy integration with various client applications.

For more information, see:

- [Two-Stage Search Logic](two_stage_search_logic.md)
- [Query Processor](query_processor.md)
- [Sinkhorn Reranking Logic](sinkhorn_reranking_logic.md)
- [FAISS and DuckDB Wrappers](faiss_duckdb_wrappers.md)
