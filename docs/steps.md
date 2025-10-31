# Steps for Chromatica

# Optimized Prompts for Building "Chromatica"

Here is a list of 21+ optimized prompts to guide you through the development process. They are designed to be used sequentially and leverage the specified tools and rules.

## 🚨 IMPORTANT: Virtual Environment Setup

**Before running ANY commands, tests, or scripts in this project, you MUST activate the virtual environment:**

```bash
# On Windows (PowerShell/CMD):
venv311/Scripts/activate

# On Unix/MacOS:
source venv311/Scripts/activate
```

**The virtual environment is located at `venv311` and was created with Python 3.11.**

### Phase 1: Project Setup and Core Logic (Week 1)

**Prompt 1: Initialize Project Structure**

> **Goal**: Create the complete directory structure for our "Chromatica" project.
> **Tool**: `filesystem` > **Instructions**: Based on the `file-layout` section in our new `.cursorrules` file, create the entire directory tree. This includes `src/chromatica/core`, `src/chromatica/indexing`, `src/chromatica/api`, `src/chromatica/utils`, `scripts`, `data`, `notebooks`, `tests`, and a `docs` folder. Also, create an empty `requirements.txt` and a `docs/troubleshooting.md` file.

**Prompt 2: Create Environment and Install Dependencies**

> **Goal**: Populate the `requirements.txt` file with all necessary libraries.
> **Tool**: `brave-search`, `filesystem` (edit_file)
> **Instructions**: Use `brave-search` to find the latest stable versions of the required libraries. Edit `requirements.txt` to include: `opencv-python`, `scikit-image`, `numpy`, `faiss-cpu`, `POT-Python-Optimal-Transport`, `duckdb`, `fastapi`, `uvicorn`, and `pytest`.
>
> **Note**: After updating requirements.txt, activate the virtual environment (`venv311/Scripts/activate`) and run `pip install -r requirements.txt` to install all dependencies.

**Prompt 3: Implement Core Configuration**

> **Goal**: Create a configuration module for all key parameters.
> **Tool**: `filesystem` (write_file)
> **Instructions**: Create `src/chromatica/utils/config.py`. Define all global constants from `critical_instructions.md` (Sections D & E), including: `L_BINS`, `A_BINS`, `B_BINS`, `TOTAL_BINS`, `LAB_RANGES`, and `RERANK_K`. Add a module-level docstring explaining its purpose.

**Prompt 4: Implement Histogram Generation**

> **Goal**: Create the core function for converting an image into a color histogram.
> **Tool**: `filesystem` (write_file)
> **Instructions**: Create `src/chromatica/core/histogram.py`. Implement the `build_histogram` function as described in Section E of `critical_instructions.md`.
>
> 1.  The function should accept a NumPy array of Lab pixels.
> 2.  It must use the constants from `src.chromatica.utils.config`.
> 3.  Implement the vectorized soft-assignment logic.
> 4.  The final histogram must be flattened and L1-normalized.
> 5.  **Add a detailed Google-style docstring** explaining the parameters, return value, and the soft-assignment logic. Add inline comments for the mathematical steps.

### Phase 2: Indexing Pipeline (Week 2)

**Prompt 5: Create Image Processing Pipeline**

> **Goal**: Create a utility function that orchestrates the full preprocessing of a single image.
> **Tool**: `filesystem` (write_file)
> **Instructions**: Create `src/chromatica/indexing/pipeline.py`. Write `process_image(image_path: str) -> np.ndarray`. This function must:
>
> 1.  Load an image using `opencv-python`.
> 2.  Downscale it (max side 256px).
> 3.  Convert from BGR to RGB, then to CIE Lab using `scikit-image`.
> 4.  Pass the pixel data to the `build_histogram` function.
> 5.  Return the normalized histogram.
> 6.  Add a proper docstring and type hints.

**Prompt 6: Set up FAISS and DuckDB Wrappers**

> **Goal**: Create classes to manage the FAISS index and DuckDB database.
> **Tool**: `filesystem` (write_file)
> **Instructions**: Create `src/chromatica/indexing/store.py`.
>
> 1.  Define a class `AnnIndex` wrapping `faiss.IndexHNSWFlat`. The `add` method MUST apply the Hellinger transform.
> 2.  Define a class `MetadataStore` managing a DuckDB connection with methods `setup_table()`, `add_batch()`, and `get_histograms_by_ids()`.
> 3.  **Add class and method docstrings** explaining the responsibilities of each component.

**Prompt 7: Create Offline Indexing Script**

> **Goal**: Write the main script to process a directory of images and populate the index/database.
> **Tool**: `filesystem` (write_file)
> **Instructions**: Create `scripts/build_index.py`. This script should:
>
> 1.  Take a directory path as a command-line argument.
> 2.  **Import and use the `logging` module** to log progress (e.g., "Processing image X of Y...") and any errors.
> 3.  Initialize `AnnIndex` and `MetadataStore`.
> 4.  Iterate through images, using `process_image` to generate histograms.
> 5.  Add histograms to FAISS (Hellinger-transformed) and DuckDB (raw).
> 6.  Save the index and close the connection.
> 7.  **Add a comment at the top with an example command**: `# Example: python scripts/build_index.py ./data/unsplash-lite`
>
> **Note**: Remember to activate the virtual environment first: `venv311/Scripts/activate`

### Phase 3: Search and API (Weeks 3-5)

**Prompt 8: Implement Sinkhorn Reranking Logic**

> **Goal**: Implement the high-fidelity reranking functions.
> **Tool**: `filesystem` (write_file)
> **Instructions**: Create `src/chromatica/core/rerank.py`.
>
> 1.  Implement `build_cost_matrix` as specified in `critical_instructions.md`.
> 2.  Implement `rerank_candidates` to compute Sinkhorn distance using `ot.sinkhorn2`.
> 3.  Add comprehensive docstrings and comments, especially for the cost matrix generation, as this is a complex part of the algorithm.

**Prompt 9: Create Query Processor**

> **Goal**: Write a function to convert API query parameters into a query histogram.
> **Tool**: `filesystem` (write_file)
> **Instructions**: Create `src/chromatica/core/query.py`. Write `create_query_histogram(colors: list[str], weights: list[float])`. This function will:
>
> 1.  Convert hex color codes to Lab values.
> 2.  Create a "softened" query histogram by distributing weights to the nearest bins.
> 3.  Ensure the result is L1-normalized.
> 4.  Document the function thoroughly.

**Prompt 10: Implement the Full Two-Stage Search Logic**

> **Goal**: Combine all pieces into a single search function.
> **Tool**: `filesystem` (write_file)
> **Instructions**: Create `src/chromatica/search.py`. Write `find_similar(query_histogram: np.ndarray, index: AnnIndex, store: MetadataStore) -> list`. This function must:
>
> 1.  Use `AnnIndex.search` to retrieve top-K candidates.
> 2.  Fetch raw histograms for these candidates from `MetadataStore`.
> 3.  Use `rerank_candidates` to re-sort them.
> 4.  Return the final, reranked list of image IDs and scores.
> 5.  Add logging to record the latency of the ANN search and the reranking stage separately.

**Prompt 11: Build the FastAPI Endpoint**

> **Goal**: Expose the search functionality via a REST API.
> **Tool**: `filesystem` (write_file)
> **Instructions**: Create `src/chromatica/api/main.py`.
>
> 1.  Initialize a FastAPI app and configure basic logging.
> 2.  Load the FAISS index and DuckDB store on startup.
> 3.  Create the `GET /search` endpoint.
> 4.  Parse `colors` and `weights` query parameters.
> 5.  Use `create_query_histogram` and `find_similar` to get results.
> 6.  Format the response into the exact JSON structure from Section H, including timings captured by the logger.
>
> **Note**: To test the API, activate the virtual environment (`venv311/Scripts/activate`) and run `uvicorn src.chromatica.api.main:app --reload`

### Phase 4: Evaluation and Refinement (Weeks 6-8)

**Prompt 12: Create Sanity Check Script**

> **Goal**: Write a script to run the sanity checks defined in the plan.
> **Tool**: `filesystem` (write_file)
> **Instructions**: Create `scripts/run_sanity_checks.py`. This script should programmatically execute the four sanity checks from Section F. For each check, print the query and the top 5 results for manual verification. Log the output clearly.
>
> **Note**: To run sanity checks, activate the virtual environment (`venv311/Scripts/activate`) and execute `python scripts/run_sanity_checks.py`

**Prompt 13: Address Background Dominance Risk**

> **Goal**: Draft an implementation for "saliency weighting."
> **Tool**: `brave-search`, `filesystem` (write_file)
> **Instructions**: Use `brave-search` to find documentation for `cv2.saliency.StaticSaliencySpectralResidual_create()`. Following the mitigation plan in Section I, Risk 3, create a new function in `src/chromatica/core/histogram.py` called `build_saliency_weighted_histogram`. It should:
>
> 1.  Accept an RGB image.
> 2.  Generate a saliency map.
> 3.  Weight the contribution of each pixel by its saliency value during histogram generation.
> 4.  Add extensive comments explaining how this approach mitigates background dominance.

**Prompt 14: Refactor for Memory Scaling**

> **Goal**: Propose code changes to address Risk 2 (Memory Footprint).
> **Tool**: `gitmvp` (read_repository), `filesystem` (edit_file), `gitmcp-docs` > **Instructions**: Use `gitmcp-docs` to find the FAISS documentation for `IndexIVFPQ`. Read `src/chromatica/indexing/store.py` using `gitmvp`. Now, edit the `AnnIndex` class to switch from `faiss.IndexHNSWFlat` to `faiss.IndexIVFPQ`. Add comments explaining what Product Quantization is, the new training step required, and how this reduces memory.

**Prompt 15: Add an Evaluation Harness**

> **Goal**: Create a script to run a batch of queries and calculate performance metrics.
> **Tool**: `filesystem` (write_file)
> **Instructions**: Create `scripts/evaluate.py`. The script should:
>
> 1.  Load test queries from a JSON file.
> 2.  Execute each search, recording latency.
> 3.  Use the `logging` module to output results.
> 4.  Calculate and print P95 latency.
> 5.  Provide a placeholder structure for loading ground truth labels and calculating Precision@10.
>
> **Note**: To run evaluation, activate the virtual environment (`venv311/Scripts/activate`) and execute `python scripts/evaluate.py`

### Phase 5: Testing and Validation (Weeks 7-8)

**Prompt 16: Unit Test the Core Histogram Logic**

> **Goal**: Create unit tests for the histogram and query logic.
> **Tool**: `filesystem` (write_file)
> **Instructions**: Create `tests/test_core_logic.py`. Using `pytest`, write tests for `src/chromatica/core/`:
>
> 1.  **Test `build_histogram`**: Use a synthetic image to assert correct dimensions, normalization, and bin distribution.
> 2.  **Test `create_query_histogram`**: Use a simple two-color query and assert correct normalization and weight concentration.
> 3.  Add docstrings to test functions explaining what they are testing.
>
> **Note**: To run tests, activate the virtual environment (`venv311/Scripts/activate`) and execute `pytest tests/`

**Prompt 17: Write Integration Test for the Search Pipeline**

> **Goal**: Test the end-to-end two-stage search process.
> **Tool**: `filesystem` (write_file)
> **Instructions**: Create `tests/test_search_pipeline.py`. This test should:
>
> 1.  Create a small, in-memory mock FAISS index and DuckDB database with 5-10 sample histograms.
> 2.  Include obvious candidates for a test query (e.g., a 99% red image).
> 3.  Execute `find_similar` against the mock data.
> 4.  Assert that the final ranked list is correct, validating that both stages are working together.

**Prompt 18: Implement API Endpoint Tests**

> **Goal**: Write tests for the FastAPI endpoint.
> **Tool**: `filesystem` (write_file)
> **Instructions**: Create `tests/test_api.py`. Use FastAPI's `TestClient`:
>
> 1.  **Test for a valid request**: Assert a 200 status and correct JSON response structure.
> 2.  **Test for invalid parameters**: Assert that malformed requests return 4xx error codes.
> 3.  **Test `k` parameter**: Send a request with `&k=5` and assert the response contains exactly 5 results.

### Phase 6: Production Readiness and Documentation (Week 8)

**Prompt 19: Enhance API Documentation**

> **Goal**: Improve the auto-generated API documentation.
> **Tool**: `filesystem` (edit_file)
> **Instructions**: Edit `src/chromatica/api/main.py`. Using FastAPI's `Query` dependency and docstrings:
>
> 1.  Add a descriptive title and description to the `FastAPI()` app instance.
> 2.  In the `@app.get("/search")` decorator, add `summary` and `description`.
> 3.  Use `Query` to add descriptions and examples for each query parameter in the function signature to enrich the `/docs` page.

**Prompt 20: Containerize the Application**

> **Goal**: Create a `Dockerfile` to package the Chromatica application.
> **Tool**: `filesystem` (write_file)
> **Instructions**: Create a `Dockerfile` in the project root. It should:
>
> 1.  Start from a `python:3.10-slim` base image.
> 2.  Copy `requirements.txt` and install dependencies.
> 3.  Copy the `src` directory.
> 4.  Copy the pre-built FAISS index and DuckDB database.
> 5.  Expose the Uvicorn port (8000).
> 6.  Set the `CMD` to run the application using `uvicorn`.

**Prompt 21: Implement a Batch Update Script**

> **Goal**: Create a script for adding new images to the index without rebuilding.
> **Tool**: `filesystem` (write_file)
> **Instructions**: Create `scripts/update_index.py`. This script must:
>
> 1.  Accept a directory path for new images.
> 2.  Load the _existing_ FAISS index and DuckDB database.
> 3.  **Use logging** to report how many images are being added.
> 4.  Process only the new images into histograms.
> 5.  Append the new data to the existing stores.
> 6.  Save the updated index back to disk. Add an example run command in the comments.
>
> **Note**: To run the update script, activate the virtual environment (`venv311/Scripts/activate`) and execute `python scripts/update_index.py ./path/to/new/images`
