# 🌐 Chromatica API Reference

This document provides a comprehensive reference for all Chromatica API endpoints, including request/response formats, parameters, and usage examples.

## 📋 Table of Contents

1. [API Overview](#api-overview)
2. [Authentication](#authentication)
3. [Base URL](#base-url)
4. [Core Endpoints](#core-endpoints)
5. [Visualization Endpoints](#visualization-endpoints)
6. [System Endpoints](#system-endpoints)
7. [Error Handling](#error-handling)
8. [Rate Limiting](#rate-limiting)
9. [Response Formats](#response-formats)
10. [Usage Examples](#usage-examples)

## 🎯 API Overview

The Chromatica API provides a RESTful interface for color-based image search with the following features:

- **Color-based search** using hex color codes and weights
- **Real-time visualization** of queries and results
- **High-performance search** with sub-second response times
- **Comprehensive metadata** including timing and distance scores
- **Interactive documentation** available at `/docs`

### **API Version**

- **Current Version**: 1.0.0
- **Status**: Production Ready
- **Last Updated**: 2024

## 🔐 Authentication

Currently, the Chromatica API does not require authentication. All endpoints are publicly accessible.

**Note**: In production environments, consider implementing API keys or OAuth2 for security.

## 🌍 Base URL

### **Development**

```
http://localhost:8000
```

### **Production**

```
https://your-domain.com
```

### **API Documentation**

```
http://localhost:8000/docs          # Interactive Swagger UI
http://localhost:8000/openapi.json  # OpenAPI specification
```

## 🔍 Core Endpoints

### **1. Search Images**

Search for images based on color queries with customizable weights.

#### **Endpoint**

```
GET /search
```

#### **Parameters**

| Parameter | Type    | Required | Description                                         | Example         |
| --------- | ------- | -------- | --------------------------------------------------- | --------------- |
| `colors`  | string  | Yes      | Comma-separated hex color codes (without #)         | `FF0000,00FF00` |
| `weights` | string  | Yes      | Comma-separated weights (0.0 to 1.0)                | `0.7,0.3`       |
| `k`       | integer | No       | Number of results to return (default: 10, max: 100) | `5`             |

#### **Request Examples**

```bash
# Single color search
curl "http://localhost:8000/search?colors=FF0000&weights=1.0&k=5"

# Multi-color weighted search
curl "http://localhost:8000/search?colors=FF0000,00FF00,0000FF&weights=0.5,0.3,0.2&k=10"

# Search with default result count
curl "http://localhost:8000/search?colors=FF0000&weights=1.0"
```

#### **Response Format**

```json
{
  "query_id": "ee0c97fa-a1f7-47e6-b817-4cc6627440f7",
  "query": {
    "colors": ["FF0000", "00FF00"],
    "weights": [0.7, 0.3]
  },
  "results_count": 5,
  "results": [
    {
      "image_id": "7349631",
      "distance": 0.0,
      "dominant_colors": ["#placeholder"],
      "file_path": "datasets/test-dataset-20/7349631.png"
    },
    {
      "image_id": "a2",
      "distance": 0.1,
      "dominant_colors": ["#placeholder"],
      "file_path": "datasets/test-dataset-20/a2.png"
    }
  ],
  "metadata": {
    "ann_time_ms": 70,
    "rerank_time_ms": 165,
    "total_time_ms": 236,
    "index_size": 20
  }
}
```

#### **Response Fields**

| Field                       | Type    | Description                                        |
| --------------------------- | ------- | -------------------------------------------------- |
| `query_id`                  | string  | Unique identifier for the search query             |
| `query`                     | object  | The original search parameters                     |
| `query.colors`              | array   | List of hex color codes used                       |
| `query.weights`             | array   | List of corresponding weights                      |
| `results_count`             | integer | Number of results returned                         |
| `results`                   | array   | List of search results                             |
| `results[].image_id`        | string  | Unique identifier for the image                    |
| `results[].distance`        | float   | Similarity distance (lower = more similar)         |
| `results[].dominant_colors` | array   | Placeholder for future dominant color extraction   |
| `results[].file_path`       | string  | Path to the image file                             |
| `metadata.ann_time_ms`      | integer | Time taken for approximate nearest neighbor search |
| `metadata.rerank_time_ms`   | integer | Time taken for Sinkhorn-EMD reranking              |
| `metadata.total_time_ms`    | integer | Total search time                                  |
| `metadata.index_size`       | integer | Total number of images in the search index         |

## 🎨 Visualization Endpoints

### **1. Query Visualization**

Generate a visual representation of the color query showing colors and weights.

#### **Endpoint**

```
GET /visualize/query
```

#### **Parameters**

| Parameter | Type   | Required | Description                                 | Example         |
| --------- | ------ | -------- | ------------------------------------------- | --------------- |
| `colors`  | string | Yes      | Comma-separated hex color codes (without #) | `FF0000,00FF00` |
| `weights` | string | Yes      | Comma-separated weights (0.0 to 1.0)        | `0.7,0.3`       |

#### **Request Examples**

```bash
# Single color visualization
curl "http://localhost:8000/visualize/query?colors=FF0000&weights=1.0" \
  --output query_viz.png

# Multi-color visualization
curl "http://localhost:8000/visualize/query?colors=FF0000,00FF00,0000FF&weights=0.5,0.3,0.2" \
  --output query_viz.png
```

#### **Response**

- **Content-Type**: `image/png`
- **Body**: PNG image showing the query visualization
- **Format**: 800x600 pixel image with:
  - Weighted color distribution bar
  - Color palette wheel
  - Color information table
  - Weight distribution pie chart

### **2. Results Collage**

Generate a visual collage of search results with distance annotations.

#### **Endpoint**

```
GET /visualize/results
```

#### **Parameters**

| Parameter | Type    | Required | Description                                 | Example         |
| --------- | ------- | -------- | ------------------------------------------- | --------------- |
| `colors`  | string  | Yes      | Comma-separated hex color codes (without #) | `FF0000,00FF00` |
| `weights` | string  | Yes      | Comma-separated weights (0.0 to 1.0)        | `0.7,0.3`       |
| `k`       | integer | No       | Number of results to include (default: 10)  | `5`             |

#### **Request Examples**

```bash
# Generate collage with 5 results
curl "http://localhost:8000/visualize/results?colors=FF0000,00FF00&weights=0.7,0.3&k=5" \
  --output results_collage.png

# Generate collage with default result count
curl "http://localhost:8000/visualize/results?colors=FF0000&weights=1.0" \
  --output results_collage.png
```

#### **Response**

- **Content-Type**: `image/png`
- **Body**: PNG image showing the results collage
- **Format**: Configurable size (default: 1200x800) with:
  - Grid layout of result images
  - Distance annotations on each image
  - Consistent sizing and spacing

## ⚙️ System Endpoints

### **1. API Information**

Get information about the current system status and available endpoints.

#### **Endpoint**

```
GET /api/info
```

#### **Parameters**

None

#### **Request Example**

```bash
curl "http://localhost:8000/api/info"
```

#### **Response Format**

```json
{
  "status": "ready",
  "message": "Search engine is ready",
  "index_size": 20,
  "endpoints": ["/search", "/visualize/query", "/visualize/results"],
  "version": "1.0.0",
  "uptime_seconds": 3600
}
```

#### **Response Fields**

| Field            | Type    | Description                                      |
| ---------------- | ------- | ------------------------------------------------ |
| `status`         | string  | System status (`ready`, `initializing`, `error`) |
| `message`        | string  | Human-readable status message                    |
| `index_size`     | integer | Number of images in the search index             |
| `endpoints`      | array   | List of available API endpoints                  |
| `version`        | string  | API version number                               |
| `uptime_seconds` | integer | Server uptime in seconds                         |

### **2. Root Endpoint**

Access the interactive web interface or get basic API information.

#### **Endpoint**

```
GET /
```

#### **Parameters**

None

#### **Request Example**

```bash
curl "http://localhost:8000/"
```

#### **Response**

- **Content-Type**: `text/html` (default) or `application/json`
- **Body**: HTML web interface or JSON API info
- **Behavior**:
  - Returns HTML interface for browser requests
  - Returns JSON info for API requests (with `Accept: application/json` header)

## ❌ Error Handling

### **HTTP Status Codes**

| Status Code | Description           | Common Causes                           |
| ----------- | --------------------- | --------------------------------------- |
| `200`       | Success               | Request completed successfully          |
| `400`       | Bad Request           | Invalid parameters or malformed request |
| `404`       | Not Found             | Endpoint or resource not found          |
| `422`       | Validation Error      | Parameter validation failed             |
| `500`       | Internal Server Error | Server-side processing error            |
| `503`       | Service Unavailable   | Search components not initialized       |

### **Error Response Format**

```json
{
  "detail": [
    {
      "loc": ["query", "colors"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### **Common Error Scenarios**

#### **Missing Required Parameters**

```json
{
  "detail": [
    {
      "loc": ["query", "weights"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

#### **Invalid Color Format**

```json
{
  "detail": [
    {
      "loc": ["query", "colors"],
      "msg": "invalid color format",
      "type": "value_error"
    }
  ]
}
```

#### **System Not Ready**

```json
{
  "detail": "Search components not initialized. Please ensure the index has been built."
}
```

## 🚦 Rate Limiting

Currently, the Chromatica API does not implement rate limiting. However, consider the following guidelines:

- **Search requests**: Maximum 100 requests per minute per client
- **Visualization requests**: Maximum 50 requests per minute per client
- **System requests**: Maximum 200 requests per minute per client

**Note**: These are recommendations for production use. Implement proper rate limiting based on your infrastructure requirements.

## 📊 Response Formats

### **Standard Response Structure**

All successful API responses follow this structure:

```json
{
  "data": {...},           // Main response data
  "metadata": {...},       // Timing and system information
  "timestamp": "2024-01-01T00:00:00Z",  // ISO 8601 timestamp
  "request_id": "uuid"     // Unique request identifier
}
```

### **Error Response Structure**

All error responses follow this structure:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": {...}       // Additional error details
  },
  "timestamp": "2024-01-01T00:00:00Z",
  "request_id": "uuid"
}
```

## 💡 Usage Examples

### **Complete Search Workflow**

```bash
# 1. Check system status
curl "http://localhost:8000/api/info"

# 2. Perform a search
curl "http://localhost:8000/search?colors=FF0000,00FF00&weights=0.7,0.3&k=5"

# 3. Generate query visualization
curl "http://localhost:8000/visualize/query?colors=FF0000,00FF00&weights=0.7,0.3" \
  --output query.png

# 4. Generate results collage
curl "http://localhost:8000/visualize/results?colors=FF0000,00FF00&weights=0.7,0.3&k=5" \
  --output results.png
```

### **JavaScript/Node.js Example**

```javascript
const axios = require("axios");

async function searchImages(colors, weights, k = 10) {
  try {
    // Perform search
    const searchResponse = await axios.get("http://localhost:8000/search", {
      params: { colors: colors.join(","), weights: weights.join(","), k },
    });

    console.log("Search results:", searchResponse.data);

    // Generate visualizations
    const queryViz = await axios.get("http://localhost:8000/visualize/query", {
      params: { colors: colors.join(","), weights: weights.join(","), k },
      responseType: "arraybuffer",
    });

    // Save query visualization
    require("fs").writeFileSync("query.png", queryViz.data);

    return searchResponse.data;
  } catch (error) {
    console.error("Search failed:", error.response?.data || error.message);
  }
}

// Usage
searchImages(["FF0000", "00FF00"], [0.7, 0.3], 5);
```

### **Python Example**

```python
import requests
import json

def search_images(colors, weights, k=10):
    """Search for images using the Chromatica API."""

    # API base URL
    base_url = "http://localhost:8000"

    try:
        # Perform search
        search_params = {
            'colors': ','.join(colors),
            'weights': ','.join(map(str, weights)),
            'k': k
        }

        search_response = requests.get(f"{base_url}/search", params=search_params)
        search_response.raise_for_status()

        search_data = search_response.json()
        print(f"Found {search_data['results_count']} results")

        # Generate query visualization
        viz_params = {
            'colors': ','.join(colors),
            'weights': ','.join(map(str, weights))
        }

        viz_response = requests.get(f"{base_url}/visualize/query", params=viz_params)
        viz_response.raise_for_status()

        # Save visualization
        with open('query_visualization.png', 'wb') as f:
            f.write(viz_response.content)

        print("Query visualization saved as 'query_visualization.png'")
        return search_data

    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None

# Usage
if __name__ == "__main__":
    colors = ['FF0000', '00FF00', '0000FF']
    weights = [0.5, 0.3, 0.2]

    results = search_images(colors, weights, k=5)
    if results:
        print(json.dumps(results, indent=2))
```

### **cURL Examples for Testing**

```bash
# Test single color search
curl "http://localhost:8000/search?colors=FF0000&weights=1.0&k=3"

# Test multi-color search
curl "http://localhost:8000/search?colors=FF0000,00FF00,0000FF&weights=0.4,0.4,0.2&k=5"

# Test visualization endpoints
curl "http://localhost:8000/visualize/query?colors=FF0000,00FF00&weights=0.6,0.4" \
  --output test_query.png

curl "http://localhost:8000/visualize/results?colors=FF0000,00FF00&weights=0.6,0.4&k=3" \
  --output test_results.png

# Test system endpoints
curl "http://localhost:8000/api/info"
curl "http://localhost:8000/" -H "Accept: application/json"
```

## 🔧 Advanced Usage

### **Batch Processing**

For processing multiple queries efficiently:

```bash
# Create a batch script
cat > batch_search.sh << 'EOF'
#!/bin/bash
colors=("FF0000" "00FF00" "0000FF" "FFFF00" "FF00FF")
weights=("1.0" "1.0" "1.0" "1.0" "1.0")

for i in "${!colors[@]}"; do
    echo "Searching for color: ${colors[$i]}"
    curl -s "http://localhost:8000/search?colors=${colors[$i]}&weights=${weights[$i]}&k=5" \
      | jq '.results_count'
done
EOF

chmod +x batch_search.sh
./batch_search.sh
```

### **Performance Monitoring**

Monitor API performance:

```bash
# Test response times
time curl -s "http://localhost:8000/search?colors=FF0000&weights=1.0&k=5" > /dev/null

# Monitor with detailed timing
curl -w "@curl-format.txt" "http://localhost:8000/search?colors=FF0000&weights=1.0&k=5"

# Create curl-format.txt
cat > curl-format.txt << 'EOF'
     time_namelookup:  %{time_namelookup}\n
        time_connect:  %{time_connect}\n
     time_appconnect:  %{time_appconnect}\n
    time_pretransfer:  %{time_pretransfer}\n
       time_redirect:  %{time_redirect}\n
  time_starttransfer:  %{time_starttransfer}\n
                     ----------\n
          time_total:  %{time_total}\n
EOF
```

---

## 📚 Additional Resources

- **Interactive Documentation**: `http://localhost:8000/docs`
- **OpenAPI Specification**: `http://localhost:8000/openapi.json`
- **Project Documentation**: See the `docs/` directory
- **Source Code**: See the `src/` directory
- **Testing Tools**: See the `tools/` directory

---

_For implementation details and advanced usage, refer to the project source code and additional documentation._
