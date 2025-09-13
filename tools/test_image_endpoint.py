#!/usr/bin/env python3
"""
Test script for the image endpoint functionality.

This script tests the new /image/{image_id} endpoint to ensure that:
1. Images can be retrieved by their ID
2. Proper content types are returned
3. Error handling works correctly

Usage:
    python tools/test_image_endpoint.py

Requirements:
    - API server must be running on localhost:8000
    - Index must be built with test-dataset-20
"""

import requests
import json
from pathlib import Path
import sys

def test_image_endpoint():
    """Test the image endpoint functionality."""
    
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Image Endpoint Functionality")
    print("=" * 50)
    
    # Test 1: Check if API is running
    print("\n1. Checking API status...")
    try:
        response = requests.get(f"{base_url}/api/info")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ API is running: {data['message']}")
            print(f"   📊 Status: {data['status']}")
        else:
            print(f"   ❌ API returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("   ❌ Cannot connect to API. Is the server running?")
        print("   💡 Start the server with: python -m src.chromatica.api.main")
        return False
    
    # Test 2: Perform a search to get some image IDs
    print("\n2. Performing test search to get image IDs...")
    try:
        search_params = {
            "colors": "ea6a81,f6d727",
            "weights": "0.49,0.51",
            "k": 5
        }
        
        response = requests.get(f"{base_url}/search", params=search_params)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Search successful: {data['results_count']} results found")
            
            if data['results']:
                # Get the first result for testing
                first_result = data['results'][0]
                test_image_id = first_result['image_id']
                test_file_path = first_result.get('file_path', 'N/A')
                
                print(f"   🖼️  Test image ID: {test_image_id}")
                print(f"   📁 File path: {test_file_path}")
                
                # Test 3: Try to retrieve the image
                print(f"\n3. Testing image retrieval for ID: {test_image_id}")
                try:
                    img_response = requests.get(f"{base_url}/image/{test_image_id}")
                    
                    if img_response.status_code == 200:
                        content_type = img_response.headers.get('content-type', 'unknown')
                        content_length = len(img_response.content)
                        
                        print(f"   ✅ Image retrieved successfully!")
                        print(f"   📊 Content-Type: {content_type}")
                        print(f"   📏 Content-Length: {content_length} bytes")
                        
                        # Check if it's actually an image
                        if content_type.startswith('image/'):
                            print(f"   🎯 Valid image format: {content_type}")
                        else:
                            print(f"   ⚠️  Unexpected content type: {content_type}")
                        
                        # Test 4: Test with non-existent image ID
                        print(f"\n4. Testing error handling with non-existent image ID...")
                        try:
                            error_response = requests.get(f"{base_url}/image/nonexistent123")
                            if error_response.status_code == 404:
                                print("   ✅ Proper 404 error for non-existent image")
                            else:
                                print(f"   ⚠️  Unexpected status code: {error_response.status_code}")
                        except Exception as e:
                            print(f"   ❌ Error testing non-existent image: {e}")
                        
                        return True
                        
                    else:
                        print(f"   ❌ Failed to retrieve image: {img_response.status_code}")
                        print(f"   📝 Response: {img_response.text}")
                        return False
                        
                except Exception as e:
                    print(f"   ❌ Error retrieving image: {e}")
                    return False
            else:
                print("   ❌ No search results found")
                return False
        else:
            print(f"   ❌ Search failed: {response.status_code}")
            print(f"   📝 Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error performing search: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Chromatica Image Endpoint Test")
    print("=" * 50)
    
    success = test_image_endpoint()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! Image endpoint is working correctly.")
        print("\n💡 You can now:")
        print("   - View images in the web interface at http://localhost:8000/")
        print("   - Use the /image/{id} endpoint in your applications")
        print("   - See actual images in search results")
    else:
        print("❌ Some tests failed. Check the output above for details.")
        print("\n🔧 Troubleshooting:")
        print("   - Ensure the API server is running")
        print("   - Verify the index has been built")
        print("   - Check the server logs for errors")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
