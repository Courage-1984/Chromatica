import requests
import re
import json
import sys
import os
import argparse
import time
import hashlib
import subprocess
from typing import Optional, Dict, Any, List, Tuple
from urllib.parse import urlparse

# --- Configuration ---
REQUEST_TIMEOUT = 30  # 30 seconds timeout for all requests
MAX_TOTAL_RETRIES = 5  # Max attempts for API fetch or download sequence
DOWNLOAD_SUBDIR = "image_downloads"

# Downloader options based on aria2c_wget_curl_downloader.py
ARIA2C_COMMON_OPTIONS = [
    "--max-tries=1",
    "--max-concurrent-downloads=1",
    "--min-split-size=10M",
    "--split=1",
    "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "--check-certificate=false",
    "--timeout=30",
]
WGET_COMMON_OPTIONS = [
    "--tries=1",
    "--wait=5",
    "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "--no-check-certificate",
    "-q",
]
CURL_COMMON_OPTIONS = [
    "-L",
    "-f",
    "--max-redirs",
    "10",
    "--connect-timeout",
    "30",
    "--retry",
    "0",
    "--user-agent",
    "Mozilla/50 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "--silent",
    "--show-error",
]
# --- Utility Functions ---


def read_json_file(filepath: str) -> Dict[str, Any]:
    """Reads JSON data from a file, returning an empty dict if not found."""
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not read or decode JSON from '{filepath}'. Error: {e}")
        return {}


def write_json_file(filepath: str, data: Dict[str, Any]):
    """Writes data to a JSON file."""
    try:
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)
    except IOError as e:
        print(f"Error writing to file '{filepath}': {e}", file=sys.stderr)
        sys.exit(1)


def get_token() -> str:
    """Extract developer token from Apple Music's public site."""
    print("Fetching Apple Music token...")
    main_page_url = "https://beta.music.apple.com"

    # Step 1: Fetch the main page to find the JS file
    try:
        main_page_response = requests.get(main_page_url, timeout=REQUEST_TIMEOUT)
        main_page_response.raise_for_status()
    except requests.RequestException as e:
        raise Exception(f"Failed to fetch main page: {e}")

    main_page_body = main_page_response.text
    js_file_regex = r"/assets/index-legacy-[^/]+\.js"
    index_js_uri = re.search(js_file_regex, main_page_body)

    if not index_js_uri:
        raise Exception("Index JS file not found")

    # Step 2: Fetch the JS file to extract the token
    js_file_url = main_page_url + index_js_uri.group(0)
    try:
        js_file_response = requests.get(js_file_url, timeout=REQUEST_TIMEOUT)
        js_file_response.raise_for_status()
    except requests.RequestException as e:
        raise Exception(f"Failed to fetch JS file: {e}")

    js_file_body = js_file_response.text
    token_regex = r'eyJh[^"]+'
    token = re.search(token_regex, js_file_body)

    if not token:
        raise Exception("Token not found in JS file")

    print("Successfully obtained Apple Music token")
    return token.group(0)


def fetch_album_apple_music(
    album_id: str, token: str, storefront: str = "us"
) -> Optional[Dict[str, Any]]:
    """Fetch album data from Apple Music API (amp-api)."""
    url = f"https://amp-api.music.apple.com/v1/catalog/{storefront}/albums/{album_id}"

    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Origin": "https://music.apple.com",
    }
    params = {"include": "albums", "extend": "extendedAssetUrls", "l": ""}

    try:
        print(f"  Trying Apple Music API with storefront: {storefront}")
        response = requests.get(
            url, headers=headers, params=params, timeout=REQUEST_TIMEOUT
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        print(f"  Successfully fetched data from {storefront}")
        return response.json()
    except requests.RequestException as e:
        # Note: If this fails, the token might be invalid/expired, but we'll try iTunes first before giving up.
        print(f"  Error fetching from Apple Music API ({storefront}): {e}")
        return None


def fetch_album_itunes(album_id: str) -> Optional[Dict[str, Any]]:
    """Fetch album data from iTunes API (fallback)."""
    url = f"https://itunes.apple.com/lookup"
    params = {"id": album_id, "entity": "album"}

    try:
        print(f"  Trying iTunes API request for album ID: {album_id}")
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        if data.get("resultCount", 0) > 0:
            print(f"  Successfully fetched iTunes data")
            return data
        print(f"  No results found in iTunes API")
        return None
    except requests.RequestException as e:
        print(f"  Error fetching from iTunes API: {e}")
        return None


def fetch_album_data_with_token(album_id: str, token: str) -> Optional[Dict[str, Any]]:
    """Fetch album data using a provided Apple Music token, fallback to iTunes if needed."""
    storefronts = ["us", "gb"]  # Primary storefronts
    for storefront in storefronts:
        result = fetch_album_apple_music(album_id, token, storefront)
        if result:
            return result
    print("No results from Apple Music API, trying iTunes API...")
    return fetch_album_itunes(album_id)


def get_album_name(api_data: Dict[str, Any]) -> str:
    """Extract and sanitize album name."""
    name = ""
    is_itunes = "results" in api_data  # Check if it's an iTunes API response

    if is_itunes and api_data.get("results"):
        # iTunes API structure
        album_info = api_data["results"][0]
        name = album_info.get("collectionName", "")
    elif api_data.get("data"):
        # Apple Music API structure
        album_info = api_data["data"][0].get("attributes", {})
        name = album_info.get("name", "")

    if not name:
        return f"AlbumID_{hashlib.sha1(api_data.__str__().encode()).hexdigest()[:8]}"

    # Sanitization
    # Keep alphanumeric, spaces, hyphens, and underscores. Replace everything else with an underscore.
    safe_name = re.sub(r"[^\w\s-]", "_", name).strip()
    safe_name = re.sub(
        r"[-\s]+", "_", safe_name
    )  # Convert all spaces/hyphens to single underscore
    # Truncate to a reasonable length to avoid filesystem issues
    if len(safe_name) > 100:
        safe_name = safe_name[:100]

    return safe_name


# --- Download Functions ---


def sanitize_and_rename_url(original_url: str) -> str:
    """
    Modifies the Apple Music URL to ensure it is always suffixed with
    /1000x1000bb and uses the original extension (.png or .jpg),
    while preserving the necessary image identifier path segments.
    """

    target_size_prefix = "/1000x1000bb"

    # 1. Find the position of the last forward slash.
    last_slash_index = original_url.rfind("/")

    if last_slash_index == -1:
        return original_url

    # base_url is the part before the size suffix
    base_url = original_url[:last_slash_index]
    original_suffix = original_url[last_slash_index:]

    # 2. Extract the file extension from the original suffix (e.g., .png or .jpg).
    ext_match = re.search(r"\.(png|jpe?g)$", original_suffix, re.IGNORECASE)

    if ext_match:
        # Construct the new, correct URL: [Base Path/Image ID] + /1000x1000bb + [Original Extension]
        new_ext = ext_match.group(0).lower()
        return f"{base_url}{target_size_prefix}{new_ext}"

    return original_url


def run_downloader_command(
    command: List[str], downloader_name: str, url: str, output_path: str
) -> bool:
    """Executes a command line downloader (aria2c, wget, or curl)."""
    # Note: output_path is only used for logging here, actual output control is in command construction
    print(f"  Attempting download with {downloader_name}...")
    try:
        # aria2c needs --dir and --out arguments separately
        if downloader_name == "aria2c":
            download_dir = os.path.dirname(output_path)
            filename = os.path.basename(output_path)
            final_command = command + ["--dir", download_dir, "--out", filename, url]
        # wget and curl need the full output path
        elif downloader_name == "wget":
            download_dir = os.path.dirname(output_path)
            final_command = command + [
                "--directory-prefix=" + download_dir,
                "--output-document=" + output_path,
                url,
            ]
        elif downloader_name == "curl":
            final_command = command + ["--output", output_path, url]
        else:  # Should not happen
            return False

        subprocess.run(
            final_command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print(f"  SUCCESS ({downloader_name}): Downloaded to '{output_path}'.")
        return True

    except subprocess.CalledProcessError as e:
        stderr_output = e.stderr[-300:] if e.stderr else "No error output."
        print(
            f"  FAILURE ({downloader_name}): Exit code {e.returncode}. Error output snippet:\n{stderr_output}"
        )
        return False
    except FileNotFoundError:
        print(
            f"  Error: '{downloader_name}' command not found. Skipping.",
            file=sys.stderr,
        )
        return False


def download_image_with_fallbacks(modified_url: str, output_path: str) -> bool:
    """
    Attempts to download an image using requests, then aria2c, then wget, then curl.
    Returns True on success, False on failure.
    """
    download_dir = os.path.dirname(output_path)
    os.makedirs(download_dir, exist_ok=True)  # Ensure output directory exists

    # 1. Attempt with 'requests' (built-in Python method)
    print("  Attempting download with 'requests' (Python built-in)...")
    try:
        response = requests.get(modified_url, timeout=REQUEST_TIMEOUT, stream=True)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  SUCCESS (requests): Downloaded to '{output_path}'.")
        return True
    except requests.RequestException as e:
        print(f"  FAILURE (requests): {e}")

    # 2. Fallback to aria2c
    aria2c_command = ["aria2c", *ARIA2C_COMMON_OPTIONS]
    if run_downloader_command(aria2c_command, "aria2c", modified_url, output_path):
        return True

    # 3. Fallback to wget
    wget_command = ["wget", *WGET_COMMON_OPTIONS]
    if run_downloader_command(wget_command, "wget", modified_url, output_path):
        return True

    # 4. Fallback to curl
    curl_command = ["curl", *CURL_COMMON_OPTIONS]
    if run_downloader_command(curl_command, "curl", modified_url, output_path):
        return True

    return False


# --- Main Processing Functions ---


def prepare_busy_file(source_file: str, busy_file: str, amount: int):
    """Initial load of the busy file from the source file."""
    source_data = read_json_file(source_file)
    if not source_data:
        print("Source file is empty or invalid. Exiting.")
        sys.exit(0)

    # Convert to list of (URL, AlbumID) tuples to maintain order
    source_items = list(source_data.items())

    amount_to_move = min(amount, len(source_items))
    if amount_to_move == 0:
        print("Amount to download is 0 or no more items in source. Exiting.")
        return

    items_to_process = source_items[:amount_to_move]
    remaining_items = source_items[amount_to_move:]

    # Write subset to busy.json (as key=url, value={"album_id": "...", "retries": 0})
    busy_data = {
        url: {"album_id": album_id, "retries": 0} for url, album_id in items_to_process
    }
    write_json_file(busy_file, busy_data)
    print(f"Moved {amount_to_move} item(s) to '{os.path.basename(busy_file)}'.")

    # Rewrite the source file without the moved items
    remaining_data = dict(remaining_items)
    write_json_file(source_file, remaining_data)
    print(
        f"Updated '{os.path.basename(source_file)}' with {len(remaining_items)} item(s) remaining."
    )


# MODIFIED FUNCTION TO INCLUDE PROGRESS LOGGING
def process_busy_file(
    busy_file: str, done_file: str, download_dir: str, token: str, total_items: int
) -> int:
    """
    Main loop to process one item from busy.json, updating progress.
    Returns the number of URLs remaining in busy.json.
    """
    busy_data = read_json_file(busy_file)
    done_data = read_json_file(done_file)  # Always reads existing data for appending

    if not busy_data:
        # This case is handled by the main loop check, but included for completeness
        return 0

    # Get the first item (key=url, value={"album_id": "...", "retries": 0})
    current_url, item_details = list(busy_data.items())[0]
    album_id = item_details["album_id"]
    current_retries = item_details["retries"]

    # Calculate current index for logging
    items_completed_so_far = total_items - len(busy_data)

    print("\n" + "=" * 80)
    print(
        f"Processing item {items_completed_so_far + 1}/{total_items}: Album ID: {album_id} (Attempt {current_retries + 1}/{MAX_TOTAL_RETRIES})"
    )
    print(f"Original URL: {current_url}")
    print("=" * 80)

    # 1. API Fetch
    api_data = fetch_album_data_with_token(album_id, token)

    if not api_data:
        # If API fetch fails, we can't proceed with naming/metadata. Treat as failure.
        print(f"  API data fetch failed for Album ID {album_id}. Skipping download.")
        api_data = {"error": "Failed to fetch album data from all APIs."}
        album_name = f"Failed_{album_id}"

        download_successful = False
    else:
        # 2. URL Modification and Naming
        modified_url = sanitize_and_rename_url(current_url)
        album_name = get_album_name(api_data)

        # 3. Download
        # Use the extension from the modified URL, which now correctly preserves the original
        final_filename = f"{album_name}{os.path.splitext(modified_url)[1]}"
        output_path = os.path.join(download_dir, final_filename)

        print(f"  Modified Image URL: {modified_url}")
        print(f"  Sanitized Album Name: {album_name}")
        print(f"  Final Download Path: {output_path}")

        download_successful = download_image_with_fallbacks(modified_url, output_path)

    # --- Logic to update busy/done files ---

    # Store data for done.json
    done_entry = {
        "original_url": current_url,
        "album_id": album_id,
        "api_data": api_data,
        "image_url_modified": modified_url if "modified_url" in locals() else None,
        "download_success": download_successful,
        "final_filename": final_filename if download_successful else None,
        "retries_count": current_retries,
    }

    if download_successful:
        # On success: Add to done.json and remove from busy.json
        print(f"\nSUCCESS: Processing for Album ID {album_id} complete.")
        done_data[album_id] = done_entry
        busy_data.pop(current_url)

    elif api_data and "error" in api_data:
        # API failed: Remove from busy.json immediately to avoid infinite token loops
        print(
            f"\nFATAL API FAILURE: Removing Album ID {album_id} from '{os.path.basename(busy_file)}'."
        )
        done_entry["final_filename"] = f"API_FETCH_FAILED. Album ID {album_id}."
        done_data[album_id] = done_entry
        busy_data.pop(current_url)
    else:
        # Download failed: Check retry count
        current_retries += 1
        if current_retries < MAX_TOTAL_RETRIES:
            # Increment retry count and keep at the top of busy.json
            print(
                f"\nRETRY: Incrementing retry count to {current_retries}. Will re-attempt."
            )
            item_details["retries"] = current_retries  # Update
            busy_data[current_url] = item_details
        else:
            # Max retries reached: Remove the URL to move on
            print(
                f"\nFAILED PERMANENTLY: Max retries ({MAX_TOTAL_RETRIES}) reached. Moving entry to '{os.path.basename(done_file)}'."
            )
            done_entry["download_success"] = False
            done_entry["final_filename"] = (
                f"PERMANENT_FAILURE. Album ID {album_id}. URL {current_url}."
            )
            done_data[album_id] = done_entry
            busy_data.pop(current_url)

    write_json_file(busy_file, busy_data)
    write_json_file(done_file, done_data)

    return len(busy_data)


# MODIFIED FUNCTION FOR PROGRESS TRACKING
def main():
    parser = argparse.ArgumentParser(
        description="A script to sequentially fetch album data, modify image URLs, and download images with multiple fallbacks, managing state via JSON files."
    )
    parser.add_argument(
        "source_json",
        type=str,
        help="Path to the main input .json file containing image URL: Album ID pairs.",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="The output folder where images will be downloaded and busy/done JSON files will be stored.",
    )
    parser.add_argument(
        "--amount",
        type=int,
        default=1,
        help="The number of objects to transfer from the source JSON to busy.json and attempt to process (default: 1).",
    )

    args = parser.parse_args()

    # Define file paths
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, DOWNLOAD_SUBDIR), exist_ok=True)
    busy_file = os.path.join(args.output_dir, "busy.json")
    done_file = os.path.join(args.output_dir, "done.json")
    download_dir = os.path.join(args.output_dir, DOWNLOAD_SUBDIR)

    # 1. Initial Setup: Read token and prepare busy.json
    try:
        token = get_token()
    except Exception as e:
        print(f"FATAL: Failed to get Apple Music token: {e}")
        token = ""

    # Check/Prepare busy.json
    busy_data = read_json_file(busy_file)
    initial_busy_count = len(busy_data)

    if not busy_data:
        print(
            f"'{os.path.basename(busy_file)}' not found or is empty. Preparing it now."
        )
        prepare_busy_file(args.source_json, busy_file, args.amount)
        # Re-read busy_data after preparation
        busy_data = read_json_file(busy_file)
        initial_busy_count = len(busy_data)

    # Calculate total items initially in busy.json
    TOTAL_ITEMS_TO_PROCESS = initial_busy_count

    if TOTAL_ITEMS_TO_PROCESS == 0:
        print("No items to process. Exiting.")
        return

    # Items completed tracks how many were successfully removed from busy.json
    items_completed_at_start = len(read_json_file(done_file))

    print(
        f"\n[PROGRESS START]: {TOTAL_ITEMS_TO_PROCESS} items in busy.json to process in this run."
    )
    print(
        f"[NOTE]: {items_completed_at_start} album IDs already found in '{os.path.basename(done_file)}'."
    )

    # 2. Sequential Processing Loop
    urls_remaining = len(busy_data)

    while urls_remaining > 0:
        # Pass the total item count to the processing function
        urls_remaining = process_busy_file(
            busy_file, done_file, download_dir, token, TOTAL_ITEMS_TO_PROCESS
        )

        # Calculate current progress based on what was removed from busy.json
        items_removed_from_busy = TOTAL_ITEMS_TO_PROCESS - urls_remaining

        print(
            f"\n[PROGRESS UPDATE]: Completed {items_removed_from_busy}/{TOTAL_ITEMS_TO_PROCESS}. Remaining: {urls_remaining}"
        )

        if urls_remaining > 0:
            # Wait a bit between downloads to be polite
            time.sleep(2)

    print("\n" + "=" * 80)
    print("All items in busy.json have been processed.")
    print(f"Results have been appended to '{done_file}'.")
    print("=" * 80)


if __name__ == "__main__":
    main()

