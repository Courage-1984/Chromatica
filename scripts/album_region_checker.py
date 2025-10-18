import requests
import json
import argparse
from typing import List, Dict, Any
import sys # Import sys for flushing output

# --- Configuration ---
STOREFRONTS_URL = "https://amp-api.music.apple.com/v1/storefronts"
CATALOG_BASE_URL = "https://amp-api.music.apple.com/v1/catalog"

def create_headers(developer_token: str) -> Dict[str, str]:
    """Creates the standard headers required for Apple Music API requests."""
    return {
        "Authorization": f"Bearer {developer_token}",
        "User-Agent": "AlbumRegionChecker/1.1",
        "Origin": "https://music.apple.com",
        "Referer": "https://music.apple.com"
    }

def fetch_storefront_ids(headers: Dict[str, str]) -> List[str]:
    """
    Fetches the list of all available Apple Music Storefront IDs.

    Args:
        headers: Dictionary containing the Authorization header.

    Returns:
        A list of two-letter storefront codes (e.g., ['us', 'gb', 'jp']).
    """
    # Force print to confirm execution
    print("-> SCRIPT STARTED: Entering fetch_storefront_ids...")
    sys.stdout.flush() 

    print("-> Step 1: Fetching all available Apple Music storefront IDs...")
    try:
        response = requests.get(STOREFRONTS_URL, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            storefront_ids = [storefront['id'] for storefront in data.get('data', [])]
            print(f"-> Found {len(storefront_ids)} storefronts.")
            return storefront_ids
        else:
            print(f"\n--- FATAL ERROR: Failed to fetch storefronts ---")
            print(f"HTTP Status Code: {response.status_code}")
            print("Server Response Body:")
            print(response.text)
            print("-------------------------------------------------")
            if response.status_code == 401:
                print("HINT: 401 Unauthorized usually means the Developer Token is invalid, expired, or incorrect.")
            
    except requests.exceptions.RequestException as e:
        # Added flush() here to ensure error messages are shown
        print(f"\n--- FATAL ERROR: An error occurred during the API request ---")
        print(f"Connection/Network Error: {e}")
        print("---------------------------------------------------------------")
    except json.JSONDecodeError:
        print("\n--- FATAL ERROR: Could not decode JSON response from storefronts endpoint. ---")
    
    # Force flush output after error printing
    sys.stdout.flush() 
    return []

def check_album_availability(developer_token: str, album_id: str):
    """
    Checks the availability of a specific album ID across all storefronts.

    Args:
        developer_token: The Apple Music Developer Token.
        album_id: The unique identifier for the album.
    """
    headers = create_headers(developer_token)
    storefront_ids = fetch_storefront_ids(headers)

    if not storefront_ids:
        print("\nAborting check due to failure in retrieving storefront list.")
        return

    print(f"\n-> Step 2: Checking Album ID '{album_id}' across {len(storefront_ids)} regions...")

    available_regions = []
    
    # We create a new session to reuse connections, improving performance
    with requests.Session() as session:
        session.headers.update(headers)

        total_storefronts = len(storefront_ids)
        for i, storefront_id in enumerate(storefront_ids):
            # Construct the catalog URL for the specific storefront and album
            catalog_url = f"{CATALOG_BASE_URL}/{storefront_id}/albums/{album_id}"
            
            # Simple progress display (using carriage return to update the line)
            print(f"  [{i+1}/{total_storefronts}] Checking region: {storefront_id.upper()}", end='\r', flush=True)

            try:
                # Making the request to check for the album's existence
                response = session.get(catalog_url, timeout=5)

                if response.status_code == 200:
                    available_regions.append(storefront_id)
                    # Print success message over the progress line, then pad with spaces
                    print(f"  [{i+1}/{total_storefronts}] AVAILABLE in: {storefront_id.upper()} (200 OK)       ")
                
                # 404 is the expected response if the album is not in that region's catalog
                elif response.status_code == 404:
                    pass # Album is simply not available, move on silently
                
                # Log any other non-200/non-404 status codes
                elif response.status_code == 429:
                    print(f"\n\n--- RATE LIMIT HIT ({response.status_code}) ---")
                    print("Apple is temporarily blocking requests. Try again later.")
                    break # Stop checking if rate limit is hit
                else:
                    # Print unexpected API errors
                    print(f"\n  [{i+1}/{total_storefronts}] UNEXPECTED STATUS ({response.status_code}) for {storefront_id.upper()}. Response: {response.text[:50]}...")

            except requests.exceptions.RequestException as e:
                print(f"\n  [{i+1}/{total_storefronts}] Connection Error for {storefront_id.upper()}: {e}")
                # We do not break here, but continue to the next region.

    # Final summary of results
    print("\n\n--- Results Summary ---")
    if available_regions:
        print(f"Album ID '{album_id}' is available in {len(available_regions)} regions:")
        # Sort regions alphabetically for easier readability
        print(json.dumps(sorted(available_regions), indent=2))
    else:
        print(f"Album ID '{album_id}' was NOT found in any checked region (or an API error occurred).")
    
    print("\n-----------------------")
    sys.stdout.flush() 


if __name__ == "__main__":
    # Add immediate debug output
    print("=== SCRIPT STARTING ===")
    sys.stdout.flush()
    
    parser = argparse.ArgumentParser(
        description="Check Apple Music Album Availability Across All Storefronts."
    )
    parser.add_argument(
        "token", 
        type=str, 
        help="Your Apple Music Developer Token."
    )
    parser.add_argument(
        "album_id", 
        type=str, 
        help="The Apple Music Catalog Album ID (e.g., 1651024421)."
    )
    
    print("=== PARSING ARGUMENTS ===")
    sys.stdout.flush()
    
    try:
        args = parser.parse_args()
        print(f"=== ARGUMENTS PARSED: Token length={len(args.token)}, Album ID={args.album_id} ===")
        sys.stdout.flush()
        
        print("=== STARTING ALBUM AVAILABILITY CHECK ===")
        sys.stdout.flush()
        
        check_album_availability(args.token, args.album_id)
        
    except SystemExit:
        # This happens when argparse fails (e.g., missing arguments)
        print("=== ARGUMENT PARSING FAILED ===")
        sys.stdout.flush()
        raise
    except Exception as e:
        print("\n\n--- CRITICAL PYTHON CRASH ---")
        print("An unexpected error occurred outside the API logic. This might be a missing dependency.")
        print(f"Error: {e}")
        print("-----------------------------")
        print("\nHINT: Ensure 'requests' library is installed: `pip install requests`")
        sys.stdout.flush()
