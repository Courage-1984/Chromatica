# Query Processor

## Chromatica Color Search Engine

---

## Overview

The Query Processor handles the conversion of user input (image files or URLs) into normalized color histograms suitable for search operations. This component serves as the entry point for all search queries, ensuring consistent preprocessing and validation.

### Key Features

- **Multi-format Support**: Handles image files (JPEG, PNG) and URLs
- **Automatic Preprocessing**: Resizes images and converts to CIE Lab color space
- **Input Validation**: Comprehensive error checking and format verification
- **Histogram Generation**: Creates 1152-dimensional Lab histograms
- **Error Handling**: Graceful fallbacks and informative error messages

---

## Core Components

### QueryProcessor Class

```python
from chromatica.core.query import QueryProcessor
from pathlib import Path

# Initialize processor
processor = QueryProcessor(max_dimension=256)

# Process image file
image_path = Path("path/to/query_image.jpg")
query_histogram = processor.process_image_file(image_path)

# Process image URL
url = "https://example.com/image.jpg"
query_histogram = processor.process_image_url(url)

print(f"Generated histogram: {query_histogram.shape}")
```

### Image Loading and Validation

```python
def load_and_validate_image(
    image_path: Union[str, Path],
    max_dimension: int = 256
) -> np.ndarray:
    """Load and validate image for processing."""

    # Check file exists
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Validate dimensions
    height, width = image.shape[:2]
    if height == 0 or width == 0:
        raise ValueError(f"Invalid image dimensions: {width}x{height}")

    # Resize if necessary
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height))

    return image
```

---

## Usage Examples

### Basic Image Processing

```python
from chromatica.core.query import QueryProcessor
import numpy as np

# Initialize processor
processor = QueryProcessor(max_dimension=256)

# Process local image
try:
    histogram = processor.process_image_file("query.jpg")
    print(f"Successfully processed image: {histogram.shape}")
    print(f"Histogram sum: {histogram.sum():.6f}")

except Exception as e:
    print(f"Processing failed: {e}")
```

### Batch Processing

```python
def process_multiple_queries(
    image_paths: List[Path],
    processor: QueryProcessor
) -> Dict[str, np.ndarray]:
    """Process multiple query images."""

    results = {}

    for image_path in image_paths:
        try:
            histogram = processor.process_image_file(image_path)
            results[image_path.name] = histogram

        except Exception as e:
            print(f"Failed to process {image_path.name}: {e}")
            continue

    print(f"Successfully processed {len(results)}/{len(image_paths)} images")
    return results

# Usage
image_paths = [
    Path("query1.jpg"),
    Path("query2.png"),
    Path("query3.jpg")
]

processor = QueryProcessor(max_dimension=256)
histograms = process_multiple_queries(image_paths, processor)
```

### URL Processing

```python
def process_image_urls(
    urls: List[str],
    processor: QueryProcessor
) -> Dict[str, np.ndarray]:
    """Process images from URLs."""

    results = {}

    for url in urls:
        try:
            histogram = processor.process_image_url(url)
            results[url] = histogram

        except Exception as e:
            print(f"Failed to process {url}: {e}")
            continue

    return results

# Usage
urls = [
    "https://example.com/image1.jpg",
    "https://example.com/image2.png"
]

processor = QueryProcessor(max_dimension=256)
histograms = process_image_urls(urls, processor)
```

---

## Error Handling

### Custom Exceptions

```python
class QueryProcessingError(Exception):
    """Base exception for query processing errors."""
    pass

class ImageLoadError(QueryProcessingError):
    """Raised when image loading fails."""
    pass

class ImageValidationError(QueryProcessingError):
    """Raised when image validation fails."""
    pass

class HistogramGenerationError(QueryProcessingError):
    """Raised when histogram generation fails."""
    pass
```

### Recovery Strategies

```python
def robust_query_processing(
    image_path: Path,
    processor: QueryProcessor,
    fallback_dimensions: List[int] = [128, 64]
) -> Optional[np.ndarray]:
    """Robust query processing with fallbacks."""

    # Try original settings
    try:
        return processor.process_image_file(image_path)
    except Exception as e:
        print(f"Original processing failed: {e}")

    # Try with smaller dimensions
    for dim in fallback_dimensions:
        try:
            processor.max_dimension = dim
            return processor.process_image_file(image_path)
        except Exception as e:
            print(f"Processing with {dim}x{dim} failed: {e}")

    print("All processing attempts failed")
    return None
```

---

## Performance Optimization

### Memory Management

```python
def optimize_memory_usage(
    image_paths: List[Path],
    batch_size: int = 10
) -> List[np.ndarray]:
    """Process images in batches to manage memory."""

    processor = QueryProcessor(max_dimension=256)
    all_histograms = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]

        # Process batch
        batch_histograms = []
        for path in batch_paths:
            try:
                hist = processor.process_image_file(path)
                batch_histograms.append(hist)
            except Exception as e:
                print(f"Failed to process {path}: {e}")
                continue

        all_histograms.extend(batch_histograms)

        # Clear batch from memory
        del batch_histograms

        print(f"Processed batch {i//batch_size + 1}")

    return all_histograms
```

### Caching

```python
from functools import lru_cache

class CachedQueryProcessor(QueryProcessor):
    """Query processor with histogram caching."""

    def __init__(self, max_dimension: int = 256, cache_size: int = 100):
        super().__init__(max_dimension)
        self.cache_size = cache_size

    @lru_cache(maxsize=100)
    def process_image_file_cached(self, image_path: str) -> np.ndarray:
        """Process image file with caching."""
        return self.process_image_file(Path(image_path))

    def get_cache_info(self) -> Dict[str, int]:
        """Get cache statistics."""
        cache_info = self.process_image_file_cached.cache_info()
        return {
            'hits': cache_info.hits,
            'misses': cache_info.misses,
            'maxsize': cache_info.maxsize,
            'currsize': cache_info.currsize
        }

# Usage
cached_processor = CachedQueryProcessor(max_dimension=256, cache_size=100)

# First call - cache miss
hist1 = cached_processor.process_image_file_cached("image1.jpg")

# Second call - cache hit
hist2 = cached_processor.process_image_file_cached("image1.jpg")

print(f"Cache info: {cached_processor.get_cache_info()}")
```

---

## Testing and Validation

### Unit Tests

```python
def test_query_processor_basic():
    """Test basic QueryProcessor functionality."""

    processor = QueryProcessor(max_dimension=256)

    # Test with valid image
    try:
        histogram = processor.process_image_file("test_image.jpg")
        assert histogram.shape == (1152,)
        assert np.allclose(histogram.sum(), 1.0, atol=1e-6)
        print("Basic processing test passed")

    except Exception as e:
        print(f"Basic test failed: {e}")

def test_error_handling():
    """Test error handling."""

    processor = QueryProcessor(max_dimension=256)

    # Test with non-existent file
    try:
        processor.process_image_file("nonexistent.jpg")
        assert False, "Should have raised exception"
    except FileNotFoundError:
        print("File not found handling works")

    # Test with invalid file
    try:
        processor.process_image_file("invalid.txt")
        assert False, "Should have raised exception"
    except Exception:
        print("Invalid file handling works")
```

### Performance Testing

```python
def benchmark_query_processing():
    """Benchmark query processing performance."""

    import time

    processor = QueryProcessor(max_dimension=256)
    image_paths = [f"test_image_{i}.jpg" for i in range(10)]

    # Warm up
    for _ in range(3):
        try:
            processor.process_image_file(image_paths[0])
        except:
            pass

    # Benchmark
    start_time = time.time()

    for path in image_paths:
        try:
            processor.process_image_file(path)
        except:
            continue

    end_time = time.time()
    total_time = end_time - start_time

    print(f"Processed {len(image_paths)} images in {total_time:.3f}s")
    print(f"Average time per image: {total_time/len(image_paths)*1000:.2f}ms")
```

---

## Integration

### With Search System

```python
def perform_search_with_query(
    query_image_path: Path,
    index: AnnIndex,
    store: MetadataStore,
    reranker: SinkhornReranker
) -> List[SearchResult]:
    """Perform complete search using query processor."""

    # Process query
    processor = QueryProcessor(max_dimension=256)
    query_histogram = processor.process_image_file(query_image_path)

    # Perform search
    from chromatica.search import find_similar
    results = find_similar(
        query_histogram=query_histogram,
        index=index,
        store=store,
        k=50,
        max_rerank=200
    )

    return results
```

### With API Endpoints

```python
from fastapi import FastAPI, UploadFile, File
from chromatica.core.query import QueryProcessor

app = FastAPI()
processor = QueryProcessor(max_dimension=256)

@app.post("/search/upload")
async def search_by_upload(file: UploadFile = File(...)):
    """Search by uploaded image file."""

    try:
        # Save uploaded file temporarily
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # Process image
        query_histogram = processor.process_image_file(temp_path)

        # Perform search
        results = perform_search_with_query(
            Path(temp_path), index, store, reranker
        )

        # Clean up
        Path(temp_path).unlink()

        return {"results": results}

    except Exception as e:
        return {"error": str(e)}
```

---

## Conclusion

The Query Processor provides a robust, efficient interface for converting user input into searchable color histograms. Key benefits include:

- **Reliability**: Comprehensive error handling and validation
- **Performance**: Optimized image processing and optional caching
- **Flexibility**: Support for multiple input formats and configurations
- **Integration**: Seamless connection to the search pipeline

The component successfully implements the image processing requirements specified in the critical instructions document, ensuring consistent histogram generation for all search queries.

For more information, see:

- [Image Processing Pipeline](image_processing_pipeline.md)
- [Two-Stage Search Logic](two_stage_search_logic.md)
- [FastAPI Endpoint](fastapi_endpoint.md)
