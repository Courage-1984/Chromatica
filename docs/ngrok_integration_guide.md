# Chromatica ngrok Integration Guide

## Overview

This guide explains how to use ngrok to expose your local Chromatica API server for remote access, enabling you to test parallel processing capabilities from anywhere on the internet.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Manual Setup](#manual-setup)
4. [Testing Remote Access](#testing-remote-access)
5. [Parallel Processing with ngrok](#parallel-processing-with-ngrok)
6. [Troubleshooting](#troubleshooting)
7. [Security Considerations](#security-considerations)

## Prerequisites

### 1. Install ngrok

Download and install ngrok from [https://ngrok.com/](https://ngrok.com/):

```bash
# Windows (using Chocolatey)
choco install ngrok

# macOS (using Homebrew)
brew install ngrok

# Or download directly from https://ngrok.com/download
```

### 2. Sign up for ngrok (Optional but Recommended)

- Create a free account at [https://ngrok.com/](https://ngrok.com/)
- Get your authtoken from the dashboard
- Configure ngrok: `ngrok config add-authtoken YOUR_AUTHTOKEN`

## Quick Start

### Option 1: Use the Integrated Script

The easiest way to start Chromatica with ngrok:

```bash
# Activate virtual environment
venv311\Scripts\activate

# Start Chromatica with ngrok
python tools/start_with_ngrok.py

# Or with custom subdomain
python tools/start_with_ngrok.py --subdomain chromatica-demo

# Test endpoints automatically
python tools/start_with_ngrok.py --test
```

This script will:
- Start the Chromatica API server
- Start ngrok tunnel
- Display the public URL
- Optionally test the endpoints

### Option 2: Manual Setup

```bash
# Terminal 1: Start Chromatica API
venv311\Scripts\activate
uvicorn src.chromatica.api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Start ngrok
ngrok http 8000
```

## Manual Setup

### Step 1: Start Chromatica API

```bash
# Activate virtual environment
venv311\Scripts\activate

# Start API server (important: use 0.0.0.0, not 127.0.0.1)
uvicorn src.chromatica.api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Important**: Use `--host 0.0.0.0` instead of `--host 127.0.0.1` to allow external connections.

### Step 2: Start ngrok

```bash
# Basic ngrok tunnel
ngrok http 8000

# With custom subdomain (requires paid plan)
ngrok http 8000 --subdomain chromatica-demo

# With custom region
ngrok http 8000 --region us
```

### Step 3: Get Your Public URL

ngrok will display something like:

```
Session Status                online
Account                       your-email@example.com
Version                       3.x.x
Region                        United States (us)
Latency                       -
Web Interface                 http://127.0.0.1:4040
Forwarding                    https://abc123.ngrok.io -> http://localhost:8000
```

Your public URL is: `https://abc123.ngrok.io`

## Testing Remote Access

### 1. Basic Health Check

```bash
# Test health endpoint
curl "https://abc123.ngrok.io/health"

# Expected response:
{
  "status": "healthy",
  "message": "Chromatica Color Search Engine API",
  "version": "1.0.0",
  "components": {
    "faiss_index": "loaded",
    "metadata_store": "loaded"
  }
}
```

### 2. Test Parallel Processing

```bash
# Test parallel search endpoint
curl -X POST "https://abc123.ngrok.io/search/parallel" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      {"colors": "FF0000", "weights": "1.0", "k": 5},
      {"colors": "00FF00", "weights": "1.0", "k": 5},
      {"colors": "0000FF", "weights": "1.0", "k": 5}
    ],
    "max_concurrent": 3
  }'
```

### 3. Test Performance Monitoring

```bash
# Get performance statistics
curl "https://abc123.ngrok.io/performance/stats"
```

## Parallel Processing with ngrok

### Using Environment Variables

Set the ngrok URL as an environment variable:

```bash
# Windows
set CHROMATICA_API_URL=https://abc123.ngrok.io

# Unix/macOS
export CHROMATICA_API_URL=https://abc123.ngrok.io

# Test parallel capabilities
python tools/test_parallel_api.py
python tools/demo_parallel_search.py --compare
```

### Python Examples

```python
import requests
import asyncio
import aiohttp

# Set the ngrok URL
BASE_URL = "https://abc123.ngrok.io"

# Test parallel search
def test_parallel_search():
    request_data = {
        "queries": [
            {"colors": "FF0000", "weights": "1.0", "k": 10},
            {"colors": "00FF00", "weights": "1.0", "k": 10},
            {"colors": "0000FF", "weights": "1.0", "k": 10}
        ],
        "max_concurrent": 5
    }
    
    response = requests.post(f"{BASE_URL}/search/parallel", json=request_data)
    return response.json()

# Test async requests
async def test_async_requests():
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(5):
            task = session.get(f"{BASE_URL}/search", params={
                "colors": f"FF{i:02x}00",
                "weights": "1.0",
                "k": 5
            })
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results

# Run tests
print("Parallel search result:", test_parallel_search())
print("Async requests:", asyncio.run(test_async_requests()))
```

### Load Testing

```python
import requests
from concurrent.futures import ThreadPoolExecutor
import time

def load_test_ngrok(base_url, num_requests=20):
    """Load test the ngrok-exposed API."""
    
    def make_request(i):
        try:
            start = time.time()
            response = requests.get(f"{base_url}/search", params={
                "colors": f"FF{i:02x}00",
                "weights": "1.0",
                "k": 5
            })
            return {
                "success": response.status_code == 200,
                "time": time.time() - start,
                "request_id": i
            }
        except Exception as e:
            return {"success": False, "error": str(e), "request_id": i}
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request, i) for i in range(num_requests)]
        results = [future.result() for future in futures]
    
    total_time = time.time() - start_time
    successful = sum(1 for r in results if r["success"])
    
    print(f"Load test results:")
    print(f"  Total requests: {num_requests}")
    print(f"  Successful: {successful}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Requests/second: {num_requests/total_time:.2f}")
    
    return results

# Run load test
base_url = "https://abc123.ngrok.io"
load_test_ngrok(base_url, num_requests=50)
```

## Troubleshooting

### Common Issues

#### 1. Connection Refused

**Problem**: `Connection refused` when accessing ngrok URL

**Solutions**:
- Ensure Chromatica API is running on `0.0.0.0:8000` (not `127.0.0.1:8000`)
- Check if ngrok is running: `ngrok http 8000`
- Verify the ngrok URL is correct

#### 2. CORS Errors

**Problem**: CORS errors in browser when accessing from web applications

**Solutions**:
- The API includes CORS middleware, but ensure you're using the correct headers
- For web applications, use the ngrok HTTPS URL

#### 3. Slow Performance

**Problem**: Slow response times through ngrok

**Solutions**:
- ngrok free tier has bandwidth limits
- Consider using a paid ngrok plan for better performance
- Test with smaller batch sizes
- Use `fast_mode=True` for approximate results

#### 4. ngrok Tunnel Not Starting

**Problem**: ngrok fails to start or shows errors

**Solutions**:
- Check if port 8000 is already in use
- Ensure ngrok is properly installed and configured
- Try a different port: `ngrok http 8001`

### Debugging Commands

```bash
# Check if API is running locally
curl "http://localhost:8000/health"

# Check ngrok status
curl "http://localhost:4040/api/tunnels"

# Test with verbose output
ngrok http 8000 --log stdout

# Check ngrok logs
ngrok http 8000 --log-level debug
```

### Performance Monitoring

```bash
# Monitor API performance through ngrok
python -c "
import requests
import time
base_url = 'https://abc123.ngrok.io'
while True:
    try:
        start = time.time()
        response = requests.get(f'{base_url}/performance/stats')
        latency = time.time() - start
        if response.status_code == 200:
            stats = response.json()
            print(f'Latency: {latency:.3f}s, Total searches: {stats[\"total_searches\"]}')
        time.sleep(5)
    except KeyboardInterrupt:
        break
"
```

## Security Considerations

### 1. Public Access

**Warning**: ngrok exposes your API to the entire internet. Consider:

- **Temporary Use**: Only use ngrok for testing and demonstrations
- **Authentication**: Add API key authentication for production use
- **Rate Limiting**: Implement rate limiting to prevent abuse
- **Monitoring**: Monitor access logs for suspicious activity

### 2. Data Privacy

- **Sensitive Data**: Don't expose APIs with sensitive data through ngrok
- **Logs**: ngrok may log request data
- **HTTPS**: Always use HTTPS URLs (ngrok provides this by default)

### 3. Production Deployment

For production use, consider:

- **Cloud Deployment**: Deploy to AWS, GCP, or Azure
- **Domain**: Use a proper domain name
- **SSL**: Use proper SSL certificates
- **Load Balancing**: Implement load balancing for high availability

## Advanced Configuration

### Custom Subdomain

```bash
# Requires paid ngrok plan
ngrok http 8000 --subdomain chromatica-demo
# Accessible at: https://chromatica-demo.ngrok.io
```

### Custom Domain

```bash
# Requires paid ngrok plan
ngrok http 8000 --hostname your-domain.com
# Accessible at: https://your-domain.com
```

### Authentication

```bash
# Add basic authentication
ngrok http 8000 --basic-auth "username:password"
# Accessible at: https://username:password@abc123.ngrok.io
```

### Webhook Testing

```bash
# Expose for webhook testing
ngrok http 8000 --host-header="localhost:8000"
```

## Integration with CI/CD

### GitHub Actions

```yaml
name: Test with ngrok
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Start API
        run: |
          uvicorn src.chromatica.api.main:app --host 0.0.0.0 --port 8000 &
          sleep 10
      
      - name: Install ngrok
        run: |
          wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
          tar -xzf ngrok-v3-stable-linux-amd64.tgz
          sudo mv ngrok /usr/local/bin/
      
      - name: Start ngrok
        run: |
          ngrok http 8000 --log=stdout > ngrok.log &
          sleep 5
          curl -s http://localhost:4040/api/tunnels | jq -r '.tunnels[0].public_url' > ngrok_url.txt
      
      - name: Test API
        run: |
          export CHROMATICA_API_URL=$(cat ngrok_url.txt)
          python tools/test_parallel_api.py
```

## Conclusion

ngrok provides an excellent way to expose your local Chromatica API for remote testing and demonstrations. The parallel processing capabilities work seamlessly through ngrok, allowing you to test high-throughput scenarios from anywhere on the internet.

Remember to use ngrok responsibly and consider security implications when exposing your API publicly.
