Here's how to resume your indexing:
Step 1: Check Current State
First, let's check how many images are currently indexed in your metadata database:

# Activate virtual environment

venv311\Scripts\activate

# Check the current state of your index

python -c "
import sys
sys.path.insert(0, 'src')
from chromatica.indexing.store import MetadataStore
import duckdb

# Check metadata database

db_path = 'C:/Users/anon/github/Chromatica/85k/chromatica_metadata.db'
store = MetadataStore(db_path=db_path)

# Get count of indexed images

result = store.connection.execute('SELECT COUNT(\*) FROM image_metadata').fetchone()
print(f'Currently indexed images: {result[0]}')

# Get the last few indexed images to see where we left off

result = store.connection.execute('SELECT image_id FROM image_metadata ORDER BY rowid DESC LIMIT 5').fetchall()
print('Last 5 indexed images:')
for row in result:
print(f' {row[0]}')

store.close()
"

Step 2: Resume Indexing
Based on your log, you were processing around image 60,294 out of 85,843 total images. Here's a resume script:

# Set environment variable for current session

$env:CHROMATICA_INDEX_DIR = "C:\Users\anon\github\Chromatica\85k"

# Resume indexing from where you left off

python scripts/build_index.py "C:\Users\anon\ai\train_1_2_combined_both\hq" --output-dir "C:\Users\anon\github\Chromatica\85k" --batch-size 100 --verbose

Quick Resume Commands
Here are the commands to resume your indexing:

# Activate virtual environment

venv311\Scripts\activate

# Set environment variable

$env:CHROMATICA_INDEX_DIR = "C:\Users\anon\github\Chromatica\85k"

# Resume indexing

python scripts/build_index.py "C:\Users\anon\ai\train_1_2_combined_both\hq" --output-dir "C:\Users\anon\github\Chromatica\85k" --batch-size 100 --verbose

http://127.0.0.1:8000/

python -m src.chromatica.api.main

uvicorn src.chromatica.api.main:app --host 0.0.0.0 --port 8000

venv311\Scripts\activate && taskkill /F /IM python.exe

venv311\Scripts\activate && taskkill /F /IM python.exe


I see the error - port 8000 is already in use. Let's first kill any existing Python processes and then try again:

venv311\Scripts\activate && taskkill /F /IM python.exe


env FIRECRAWL_API_KEY=fc-YOUR_API_KEY npx -y firecrawl-mcp


env FIRECRAWL_API_KEY=fc-fc-17511dcdf1114e2eb5adb76eb6b12fc2 npx -y firecrawl-mcp

cmd /c "set FIRECRAWL_API_KEY=your-api-key && npx -y firecrawl-mcp"

cmd /c "set FIRECRAWL_API_KEY=17511dcdf1114e2eb5adb76eb6b12fc2 && npx -y firecrawl-mcp"





& "C:\Program Files\PowerShell\7\pwsh.exe" -Command "C:/Users/anon/AppData/Roaming/npm/firecrawl-mcp.cmd http://localhost:8000 -o crawl_results.json --max-depth 3 -d --env FIRECRAWL_API_KEY=fc-17511dcdf1114e2eb5adb76eb6b12fc2"


$env:FIRECRAWL_API_KEY = "fc-17511dcdf1114e2eb5adb76eb6b12fc2"

& "C:/Users/anon/AppData/Roaming/npm/firecrawl-mcp.cmd" http://localhost:8000 -o crawl_results.json --max-depth 3 -d








Now you can process your 60 million images in chunks! Here's how to use it:

First chunk (first million images):

python scripts/build_index.py ./your-image-dir --output-dir C:/Users/anon/github/Chromatica/covers_index --start-index 0 --end-index 1000000 --batch-size 1000

Next chunk (append to existing index):

python scripts/build_index.py ./your-image-dir --output-dir C:/Users/anon/github/Chromatica/covers_index --start-index 1000000 --end-index 2000000 --batch-size 1000 --append


IN MY CASE:

use different input folders:

python scripts/build_index.py C:/Users/anon/github/cover-dl/covers1_10000 --output-dir C:/Users/anon/github/Chromatica/covers_index --batch-size 1000

python scripts/build_index.py C:/Users/anon/github/cover-dl/covers2_10000 --output-dir C:/Users/anon/github/Chromatica/covers_index --batch-size 1000 --append




### ngrok

ngrok.exe authtoken YOUR_AUTH_TOKEN_HERE

python -m src.chromatica.api.main

ngrok.exe http 8000




uvx --from git+https://github.com/oraios/serena serena project index


venv311\Scripts\activate

10,003 items


python scripts/build_index.py C:/Users/anon/github/cover-dl/covers1_10000 --output-dir C:/Users/anon/github/Chromatica/_covers_index --batch-size 1000



python scripts/build_index.py C:/Users/anon/github/cover-dl/covers2_10000 --output-dir C:/Users/anon/github/Chromatica/covers_index --batch-size 1000 --append




venv311\Scripts\activate

$env:CHROMATICA_INDEX_DIR="C:\Users\anon\github\Chromatica\85k"
echo $env:CHROMATICA_INDEX_DIR

python -m src.chromatica.api.main



venv311\Scripts\activate

taskkill /F /IM python.exe

$env:CHROMATICA_INDEX_DIR="C:\Users\anon\github\Chromatica\85k"

uvicorn src.chromatica.api.main:app --host 0.0.0.0 --port 8000 --reload --reload-dirs src --reload-exclude logs --reload-exclude venv311 --reload-exclude datasets --reload-exclude "**/pycache"




$env:CHROMATICA_INDEX_DIR = "$PWD\85k"


$env:CHROMATICA_INDEX_DIR = "$PWD\85k"
$env:CHROMATICA_INDEX_FILE = "chromatica_index.faiss"
$env:CHROMATICA_DB_FILE    = "chromatica_metadata.db"

python -m src.chromatica.api.main


venv311\Scripts\activate

Get-ChildItem -Path $env:CHROMATICA_INDEX_DIR | Where-Object {$_.Name -match '\.faiss|\.db|\.duckdb'}
Test-Path (Join-Path $env:CHROMATICA_INDEX_DIR $env:CHROMATICA_INDEX_FILE)
Test-Path (Join-Path $env:CHROMATICA_INDEX_DIR $env:CHROMATICA_DB_FILE)


$env:CHROMATICA_INDEX_DIR = (Resolve-Path .\85k).Path
$env:CHROMATICA_INDEX_FILE = "chromatica_index.faiss"   # adjust if different
$env:CHROMATICA_DB_FILE    = "chromatica_metadata.db"   # adjust if different


venv311\Scripts\activate

pytest tests/

$env:PYTHONPATH="$pwd;$env:PYTHONPATH"; pytest tests/

venv311\Scripts\activate

python scripts/build_index.py C:/Users/anon/github/Chromatica/_test_data --output-dir C:/Users/anon/github/Chromatica/_test_data_db --batch-size 1000

python scripts/build_index.py C:/Users/anon/github/cover-dl/covers2_10000 --output-dir C:/Users/anon/github/Chromatica/covers_index --batch-size 1000 --append

$env:CHROMATICA_INDEX_DIR = (Resolve-Path .\index_urls).Path

cls

python scripts/build_index.py C:/Users/anon/github/Chromatica/_test_data --output-dir C:/Users/anon/github/Chromatica/_test_data_db --batch-size 1000 --verbose

# Basic usage
python scripts/build_covers_index.py data/done.json data/images

# With custom output directory and batch size
python scripts/build_covers_index.py data/done.json data/images --output-dir index_urls --batch-size 1000 --verbose

# Append mode for incremental indexing
python scripts/build_covers_index.py data/done.json data/images --append

# Skip downloading missing images
python scripts/build_covers_index.py data/done.json data/images --no-download


python scripts/build_covers_index.py data/done.json data/images --output-dir index_urls --batch-size 1000 --verbose


python scripts/build_covers_index.py image_downloads_001_10k/done.json image_downloads_001_10k/images --output-dir _covers_indexed_urls --batch-size 1000 --verbose

--enhance-metadata --update-json

python scripts/build_covers_index.py image_downloads_001_10k/done.json image_downloads_001_10k/images --output-dir _covers_indexed_urls --batch-size 1000 --verbose --enhance-metadata --update-json


python scripts/build_covers_index.py data/done.json data/images --output-dir _covers_indexed_urls --batch-size 1000 --verbose --enhance-metadata --update-json


venv311\Scripts\activate

$env:CHROMATICA_INDEX_DIR = (Resolve-Path .\_covers_indexed_urls).Path

python -m src.chromatica.api.main


https://itunes.apple.com/lookup?id=<ALBUM_ID_HERE>&entity=album

https://amp-api.music.apple.com/v1/catalog/jp/albums/(id)

https://amp-api.music.apple.com/v1/catalog/us/albums/(id)

curl -X GET 'https://amp-api.music.apple.com/v1/catalog/<region>/albums/<id>' \
  -H 'Authorization: Bearer <your_auth_token>' \
  -H 'Origin: https://music.apple.com' \
  -H 'Referer: https://music.apple.com'

curl -X GET 'https://amp-api.music.apple.com/v1/catalog/us/albums/1667184596' -H 'Authorization: Bearer eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6IldlYlBsYXlLaWQifQ.eyJpc3MiOiJBTVBXZWJQbGF5IiwiaWF0IjoxNzU5OTcwNjMxLCJleHAiOjE3NjcyMjgyMzEsInJvb3RfaHR0cHNfb3JpZ2luIjpbImFwcGxlLmNvbSJdfQ.2olNgPLuL51wQBjlYwZWslVBxqV65I921NlgdXHazA9DL_-zksa42Lr4aiGC0TV3SAe4vs9FSRtdKe9gTCCiwQ' -H 'Origin: https://music.apple.com' -H 'Referer: https://music.apple.com'


1651024421

325333357

eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6IldlYlBsYXlLaWQifQ.eyJpc3MiOiJBTVBXZWJQbGF5IiwiaWF0IjoxNzU5OTcwNjMxLCJleHAiOjE3NjcyMjgyMzEsInJvb3RfaHR0cHNfb3JpZ2luIjpbImFwcGxlLmNvbSJdfQ.2olNgPLuL51wQBjlYwZWslVBxqV65I921NlgdXHazA9DL_-zksa42Lr4aiGC0TV3SAe4vs9FSRtdKe9gTCCiwQ




curl -X GET 'https://api.music.apple.com/v1/storefronts' \
  -H 'Authorization: Bearer <your_developer_token>' \
  -H 'Origin: https://music.apple.com'


  curl -X GET 'https://api.music.apple.com/v1/storefronts' -H 'Authorization: Bearer eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6IldlYlBsYXlLaWQifQ.eyJpc3MiOiJBTVBXZWJQbGF5IiwiaWF0IjoxNzU5OTcwNjMxLCJleHAiOjE3NjcyMjgyMzEsInJvb3RfaHR0cHNfb3JpZ2luIjpbImFwcGxlLmNvbSJdfQ.2olNgPLuL51wQBjlYwZWslVBxqV65I921NlgdXHazA9DL_-zksa42Lr4aiGC0TV3SAe4vs9FSRtdKe9gTCCiwQ' -H 'Origin: https://music.apple.com'


python album_region_checker.py <YOUR_DEV_TOKEN> <ID>

python ./scripts/album_region_checker.py <YOUR_DEV_TOKEN> <ID>

325333357


curl -X GET 'https://amp-api.music.apple.com/v1/catalog/us/albums/1667184596' -H 'Authorization: Bearer eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6IldlYlBsYXlLaWQifQ.eyJpc3MiOiJBTVBXZWJQbGF5IiwiaWF0IjoxNzU5OTcwNjMxLCJleHAiOjE3NjcyMjgyMzEsInJvb3RfaHR0cHNfb3JpZ2luIjpbImFwcGxlLmNvbSJdfQ.2olNgPLuL51wQBjlYwZWslVBxqV65I921NlgdXHazA9DL_-zksa42Lr4aiGC0TV3SAe4vs9FSRtdKe9gTCCiwQ' -H 'Origin: https://music.apple.com' -H 'Referer: https://music.apple.com'


$env:CHROMATICA_INDEX_DIR = "$PWD\85k"


python scripts/build_covers_index.py _small_covers_db_1_10k/done.json _small_covers_db_1_10k/images --output-dir _small_covers_db --batch-size 1000 --verbose --enhance-metadata --update-json


python scripts/build_covers_index.py _covers01_100k/done.json _covers01_100k/images --output-dir _covers01_100k_db --batch-size 1000 --verbose --enhance-metadata --update-json



$env:CHROMATICA_INDEX_DIR = "$PWD\_covers01_100k_db"
python -m src.chromatica.api.main






venv311\Scripts\activate

$env:CHROMATICA_INDEX_DIR = "C:\path\to\index"  # folder with chromatica_index.faiss and chromatica_metadata.db

$env:CHROMATICA_INDEX_DIR = "$PWD\_covers01_100k_db"

```
- Single color:
  ```bash
  python tools/test_search_quality.py --colors "#ff00ff" --weights "1.0" --k 9
  ```
- Two colors:
  ```bash
  python tools/test_search_quality.py --colors "#ff00ff,#00ff00" --weights "0.65,0.35" --k 12
  ```
- Save JSON:
  ```bash
  python tools/test_search_quality.py --colors "#d6cbbe,#e1fde4" --weights "0.5,0.5" --k 12 --out reports\search_quality.json
  python tools/test_search_quality.py --colors "#d6cbbe,#e1fde4" --weights "0.4,0.6" --k 12 --out reports\search_quality.json
  ```

What it does
- Builds the query histogram via `src.chromatica.core.query.create_query_histogram`.
- Runs `find_similar` twice: Normal (Sinkhorn-EMD) and Fast (L2).
- For each result, loads the image file and computes top dominant colors using k-means in Lab space, then reports them as hex with percentages.
- Prints concise side-by-side summaries and optionally saves a JSON report.

File added
- `tools/test_search_quality.py` with OpenCV + scikit-image based color extraction and clean console output.

If you want, I can extend it to compare result overlap between modes, output CSV, or visualize color bars per result.


Bias control notes (config in `src/chromatica/utils/config.py`)
- CHROMA_CUTOFF: suppress near-neutral bins during search (default 10)
- CHROMA_SIGMA: strength of chroma weighting (higher = stronger growth)
- QUERY_SHARPEN_EXPONENT: concentrates query mass around peaks (>1)
- RERANK_ALPHA_L1: extra L1 penalty in rerank to preserve query balance


python tools/test_search_quality.py --colors "#d6cbbe,#e1fde4" --weights "0.4,0.6" --k 12 --top-colors 7 --out reports\search_quality_v3.json

python tools/test_search_quality.py --colors "#d6cbbe,#e1fde4" --weights "0.4,0.6" --k 12 --top-colors 7 --out reports\search_quality_v4.json

python tools/test_search_quality.py --colors "#03e9fc,#ff00ff" --weights "0.4,0.6" --k 12 --top-colors 7 --out reports\search_quality_v5.json



python scripts/build_covers_index.py _covers02_100k/done.json _covers02_100k/images --output-dir _covers01_100k_db --batch-size 1000 --verbose --enhance-metadata --update-json --append





venv311\Scripts\activate

python scripts/build_covers_index.py _covers02_100k/done.json _covers02_100k/images --output-dir _covers01_100k_db --batch-size 1000 --verbose --enhance-metadata --update-json --append

$env:CHROMATICA_INDEX_DIR = "$PWD\_covers01_100k_db"

python -m src.chromatica.api.main


http://127.0.0.1:8000/get_stats


# Using curl
curl -X POST http://127.0.0.1:8000/restart

# Using Python requests
import requests
response = requests.post("http://127.0.0.1:8000/restart")

http://127.0.0.1:8000/docs

