#!/usr/bin/env python3
"""
Start Chromatica API with ngrok integration.

This script starts the Chromatica API server and optionally exposes it via ngrok
for remote access. It handles the integration between the local server and ngrok.

Usage:
    # Start API with ngrok
    python tools/start_with_ngrok.py

    # Start API only (no ngrok)
    python tools/start_with_ngrok.py --no-ngrok

    # Custom ngrok subdomain
    python tools/start_with_ngrok.py --subdomain chromatica-demo

    # Custom port
    python tools/start_with_ngrok.py --port 8001
"""

import subprocess
import sys
import time
import requests
import json
import argparse
import os
from pathlib import Path
import threading
import signal


def check_ngrok_installed():
    """Check if ngrok is installed and available."""
    try:
        result = subprocess.run(['ngrok', 'version'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def start_chromatica_api(port=8000, host="0.0.0.0"):
    """Start the Chromatica API server."""
    print(f"🚀 Starting Chromatica API on {host}:{port}")
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    
    # Python executable path
    python_path = (
        str(project_root / "venv311" / "Scripts" / "python.exe")
        if os.name == "nt"
        else str(project_root / "venv311" / "bin" / "python")
    )
    
    # Start the API server
    cmd = [
        python_path, "-m", "uvicorn",
        "src.chromatica.api.main:app",
        "--host", host,
        "--port", str(port),
        "--reload"
    ]
    
    return subprocess.Popen(cmd, cwd=str(project_root))


def start_ngrok(port=8000, subdomain=None):
    """Start ngrok tunnel."""
    print(f"🌐 Starting ngrok tunnel for port {port}")
    
    cmd = ["ngrok", "http", str(port)]
    if subdomain:
        cmd.extend(["--subdomain", subdomain])
    
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def get_ngrok_url():
    """Get the public ngrok URL."""
    try:
        response = requests.get("http://localhost:4040/api/tunnels")
        data = response.json()
        
        for tunnel in data.get("tunnels", []):
            if tunnel.get("proto") == "https":
                return tunnel.get("public_url")
        
        # Fallback to HTTP if HTTPS not available
        for tunnel in data.get("tunnels", []):
            if tunnel.get("proto") == "http":
                return tunnel.get("public_url")
        
        return None
    except Exception as e:
        print(f"❌ Failed to get ngrok URL: {e}")
        return None


def wait_for_api_ready(port=8000, timeout=30):
    """Wait for the API to be ready."""
    print("⏳ Waiting for API to be ready...")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=5)
            if response.status_code == 200:
                print("✅ API is ready!")
                return True
        except requests.exceptions.ConnectionError:
            pass
        
        time.sleep(1)
    
    print("❌ API failed to start within timeout")
    return False


def test_parallel_endpoints(base_url):
    """Test the parallel processing endpoints."""
    print(f"🧪 Testing parallel endpoints at {base_url}")
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Health endpoint working")
        else:
            print("❌ Health endpoint failed")
            return False
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")
        return False
    
    # Test performance stats
    try:
        response = requests.get(f"{base_url}/performance/stats")
        if response.status_code == 200:
            stats = response.json()
            print(f"✅ Performance stats: {stats['total_searches']} searches")
        else:
            print("❌ Performance stats failed")
    except Exception as e:
        print(f"❌ Performance stats error: {e}")
    
    # Test parallel search
    try:
        test_data = {
            "queries": [
                {"colors": "FF0000", "weights": "1.0", "k": 5},
                {"colors": "00FF00", "weights": "1.0", "k": 5}
            ],
            "max_concurrent": 2
        }
        
        response = requests.post(f"{base_url}/search/parallel", json=test_data)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Parallel search: {result['successful_queries']}/{result['total_queries']} successful")
        else:
            print("❌ Parallel search failed")
    except Exception as e:
        print(f"❌ Parallel search error: {e}")
    
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Start Chromatica API with ngrok")
    parser.add_argument("--no-ngrok", action="store_true", help="Don't start ngrok")
    parser.add_argument("--port", type=int, default=8000, help="API port (default: 8000)")
    parser.add_argument("--subdomain", help="ngrok subdomain")
    parser.add_argument("--test", action="store_true", help="Test endpoints after starting")
    
    args = parser.parse_args()
    
    print("🚀 Chromatica API with ngrok Integration")
    print("=" * 50)
    
    # Check if ngrok is installed
    if not args.no_ngrok and not check_ngrok_installed():
        print("❌ ngrok is not installed or not in PATH")
        print("Please install ngrok from https://ngrok.com/")
        sys.exit(1)
    
    # Start the API server
    api_process = start_chromatica_api(args.port)
    
    try:
        # Wait for API to be ready
        if not wait_for_api_ready(args.port):
            print("❌ Failed to start API server")
            sys.exit(1)
        
        ngrok_process = None
        ngrok_url = None
        
        if not args.no_ngrok:
            # Start ngrok
            ngrok_process = start_ngrok(args.port, args.subdomain)
            
            # Wait for ngrok to start
            print("⏳ Waiting for ngrok to start...")
            time.sleep(3)
            
            # Get ngrok URL
            ngrok_url = get_ngrok_url()
            if ngrok_url:
                print(f"🌐 ngrok URL: {ngrok_url}")
                print(f"🔗 API accessible at: {ngrok_url}")
                print(f"📊 ngrok dashboard: http://localhost:4040")
            else:
                print("❌ Failed to get ngrok URL")
                ngrok_process = None
        
        # Test endpoints if requested
        if args.test:
            base_url = ngrok_url if ngrok_url else f"http://localhost:{args.port}"
            test_parallel_endpoints(base_url)
        
        # Print usage information
        print("\n" + "=" * 50)
        print("📋 Usage Information:")
        print(f"   Local API: http://localhost:{args.port}")
        if ngrok_url:
            print(f"   Public API: {ngrok_url}")
            print(f"   ngrok Dashboard: http://localhost:4040")
        
        print("\n🔧 Parallel Processing Examples:")
        if ngrok_url:
            base_url = ngrok_url
        else:
            base_url = f"http://localhost:{args.port}"
        
        print(f"   # Test parallel search")
        print(f"   curl -X POST \"{base_url}/search/parallel\" \\")
        print(f"     -H \"Content-Type: application/json\" \\")
        print(f"     -d '{{\"queries\":[{{\"colors\":\"FF0000\",\"weights\":\"1.0\",\"k\":10}}],\"max_concurrent\":5}}'")
        
        print(f"\n   # Get performance stats")
        print(f"   curl \"{base_url}/performance/stats\"")
        
        print(f"\n   # Test with Python")
        print(f"   python tools/test_parallel_api.py --base-url {base_url}")
        
        print("\n⏹️  Press Ctrl+C to stop all services")
        
        # Keep running until interrupted
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
    
    finally:
        # Cleanup
        if api_process:
            api_process.terminate()
            api_process.wait()
            print("✅ API server stopped")
        
        if ngrok_process:
            ngrok_process.terminate()
            ngrok_process.wait()
            print("✅ ngrok tunnel stopped")


if __name__ == "__main__":
    main()
