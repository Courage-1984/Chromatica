# commands

# Create venv with Python 3.11

C:/Users/anon/AppData/Local/Programs/Python/Python311/python.exe -m venv venv311

# Activate it

# On Windows:

venv311/Scripts/activate

# On Unix/MacOS:

source venv311/Scripts/activate

[notice] A new release of pip is available: 24.0 -> 25.2
[notice] To update, run: python.exe -m pip install --upgrade pip

### ngrok

ngrok.exe authtoken YOUR_AUTH_TOKEN_HERE

python -m src.chromatica.api.main

ngrok.exe http 8000

ngrok will show something like: https://abc123.ngrok.io

python tools/cleanup_outputs.py
python tools/cleanup_outputs.py --datasets
python tools/cleanup_outputs.py --all
python tools/cleanup_outputs.py --all --confirm

cd src\chromatica\api
python main.py

# Basic usage

python scripts/build_index.py datasets/test-dataset-200

python scripts/build_index.py C:\Users\anon\ai\train\_1_2_combined_both\hq

# Advanced usage with custom settings

python scripts/build_index.py datasets/test-dataset-5000 --output-dir ./production_index --batch-size 200 --verbose

# Help and options

python scripts/build_index.py --help


