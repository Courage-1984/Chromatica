# Production Deployment Guide for Chromatica

## Concurrency and Scalability Assessment

### Current Architecture

Chromatica is designed with concurrency in mind:

1. **FastAPI Async Framework**: Uses async/await for non-blocking I/O
2. **Thread Pool Executor**: Handles CPU-intensive operations with `ThreadPoolExecutor(max_workers=os.cpu_count() * 2)`
3. **Thread-Safe Components**:
   - DuckDB uses thread-local connections (`_get_thread_connection()`) - safe for concurrent reads
   - FAISS HNSW index supports concurrent read operations (search-only, no writes during runtime)
   - Performance stats use `threading.Lock()` for thread-safe updates
4. **Concurrent Search Tracking**: System tracks concurrent searches with proper locking

### Limitations for Production

#### 1. **Thread Pool Bottleneck**
- Current thread pool: `os.cpu_count() * 2` workers
- On a typical 4-core machine: 8 concurrent searches max
- **Impact**: If more than 8 people search simultaneously, additional requests will queue

#### 2. **GIL Limitations**
- Python's Global Interpreter Lock (GIL) can limit true parallelism
- CPU-intensive operations (reranking) may not fully utilize all cores
- **Impact**: Throughput limited even with thread pool

#### 3. **Memory Constraints**
- FAISS index and DuckDB data are loaded into memory
- Multiple concurrent searches share the same memory footprint
- **Impact**: Memory usage is constant regardless of concurrent users

#### 4. **No Request Rate Limiting**
- Currently no protection against request flooding
- **Impact**: Malicious or excessive requests could overwhelm the server

### Recommendations for Internet Exposure

#### ✅ **Can Handle Moderate Traffic**
For a small group of friends (5-15 users), the current setup should work fine with these considerations:

1. **Thread pool is sufficient** for casual concurrent usage
2. **Read-only operations** are thread-safe
3. **Async architecture** handles I/O efficiently

#### ⚠️ **Improvements Needed for Better Production**

1. **Use Production ASGI Server**:
   ```bash
   # Instead of uvicorn default:
   uvicorn src.chromatica.api.main:app --host 0.0.0.0 --port 8000 --workers 1
   
   # Better for production:
   uvicorn src.chromatica.api.main:app --host 0.0.0.0 --port 8000 --workers 2 --timeout-keep-alive 30
   ```
   - `--workers 2`: Multiple worker processes (bypasses GIL limitations)
   - Each worker has its own thread pool

2. **Add Request Rate Limiting**:
   ```python
   # In main.py, add:
   from slowapi import Limiter, _rate_limit_exceeded_handler
   from slowapi.util import get_remote_address
   
   limiter = Limiter(key_func=get_remote_address)
   app.state.limiter = limiter
   
   # On search endpoint:
   @app.get("/search")
   @limiter.limit("10/minute")  # 10 requests per minute per IP
   async def search_images(...):
       ...
   ```

3. **Add Reverse Proxy (Nginx)**:
   - Provides additional connection handling
   - SSL/TLS termination
   - Rate limiting at network level
   - Better security

4. **Monitoring and Logging**:
   - Monitor concurrent search count
   - Track response times
   - Set up alerts for high load

### Expected Performance

#### Small Group (5-10 concurrent users):
- ✅ **Will work well**
- Typical search: 200-500ms (normal mode), 50-100ms (fast mode)
- Thread pool handles queuing gracefully

#### Medium Group (10-20 concurrent users):
- ⚠️ **May experience queuing**
- Some requests may wait 1-2 seconds before processing starts
- Consider adding workers or increasing thread pool

#### Large Group (20+ concurrent users):
- ❌ **Needs scaling**
- Requires multiple workers or horizontal scaling
- Consider load balancing

### Security Considerations

⚠️ **IMPORTANT**: Exposing `http://127.0.0.1:8000` to the internet has security implications:

1. **No Authentication**: Anyone with the URL can use your API
2. **No Rate Limiting**: Vulnerable to abuse/DoS
3. **No HTTPS**: Data transmitted in plaintext
4. **Resource Exhaustion**: No protection against malicious requests

#### Recommended Security Steps:

1. **Use HTTPS**: Set up SSL/TLS certificate
2. **Add Authentication**: API keys or user authentication
3. **Rate Limiting**: Prevent abuse
4. **Firewall Rules**: Restrict access if possible
5. **Monitor Access**: Log all requests

### Quick Production Setup

```bash
# 1. Install production dependencies
pip install slowapi  # For rate limiting

# 2. Run with multiple workers
uvicorn src.chromatica.api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 2 \
    --timeout-keep-alive 30 \
    --log-level info

# 3. Behind Nginx (recommended)
# See nginx.conf.example in docs/
```

### Testing Concurrency

You can test concurrent load using:
```bash
# Install Apache Bench
apt-get install apache2-utils  # Ubuntu/Debian
brew install httpd  # macOS

# Test 20 concurrent requests
ab -n 100 -c 20 "http://127.0.0.1:8000/search?colors=FF0000&weights=1.0&k=10&fast_mode=true"
```

## Conclusion

**For casual use with friends (5-15 users)**: ✅ Current setup is adequate

**For public internet exposure**: ⚠️ Add rate limiting, HTTPS, and monitoring

**For high-traffic scenarios**: Consider horizontal scaling with multiple servers behind a load balancer

