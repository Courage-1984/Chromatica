# Steps for Chromatica

# Optimized Prompts for Building "Chromatica"

Here is a list of 15+ optimized prompts to guide you through the development process. They are designed to be used sequentially and leverage the specified tools and rules.

### Phase 1: Project Setup and Core Logic (Week 1)

**Prompt 1: Initialize Project Structure**

> **Goal**: Create the complete directory structure for our "Chromatica" project.
> **Tool**: `filesystem`
> **Instructions**: Based on the `file-layout` section in our `.cursorrules` file, create the entire directory tree. This includes `src/chromatica/core`, `src/chromatica/indexing`, `src/chromatica/api`, `src/chromatica/utils`, `scripts`, `data`, `notebooks`, and `tests`. Also, create an empty `requirements.txt` file at the root.

**Prompt 2: Create Environment and Install Dependencies**

> **Goal**: Populate the `requirements.txt` file with all necessary libraries.
> **Tool**: `filesystem` (edit\_file)
> **Instructions**: Edit the `requirements.txt` file to include all the Python libraries mentioned in Section E and the `.cursorrules` file. Include: `opencv-python`, `scikit-image`, `numpy`, `faiss-cpu`, `POT-Python-Optimal-Transport`, `duckdb`, `fastapi`, and `uvicorn`. Ensure you specify versions that are known to be stable together.

**Prompt 3: Implement Core Configuration**

> **Goal**: Create a configuration module for all key parameters.
> **Tool**: `filesystem` (write\_file)
> **Instructions**: Create a new file at `src/chromatica/utils/config.py`. Following `critical_instructions.md` (Sections D & E), define all global constants in this file. This must include: `L_BINS`, `A_BINS`, `B_BINS`, `TOTAL_BINS`, `LAB_RANGES`, and `RERANK_K`. This ensures we have a single source of truth for parameters.

**Prompt 4: Implement Histogram Generation**

> **Goal**: Create the core function for converting an image into a color histogram.
> **Tool**: `filesystem` (write\_file)
> **Instructions**: Create a new file at `src/chromatica/core/histogram.py`. Implement the `build_histogram` function as described in Section E of `critical_instructions.md`.
>
> 1.  The function should accept a NumPy array of Lab pixels.
> 2.  It must use the constants from `src.chromatica.utils.config`.
> 3.  Implement the vectorized soft-assignment logic to generate the 1,152-dimension histogram.
> 4.  The final histogram must be flattened and L1-normalized (sum to 1).
> 5.  Adhere strictly to the `.cursorrules` and the reference snippet. Add detailed comments explaining the soft-assignment math.

### Phase 2: Indexing Pipeline (Week 2)

**Prompt 5: Create Image Processing Pipeline**

> **Goal**: Create a utility function that orchestrates the full preprocessing of a single image.
> **Tool**: `filesystem` (write\_file)
> **Instructions**: Create a file at `src/chromatica/indexing/pipeline.py`. Write a function `process_image(image_path: str) -> np.ndarray`. This function must:
>
> 1.  Load an image using `opencv-python`.
> 2.  Downscale it so its max side is 256px.
> 3.  Convert the color from BGR (OpenCV default) to RGB.
> 4.  Convert the RGB image to CIE Lab using `scikit-image`.
> 5.  Reshape the pixel data and pass it to the `build_histogram` function from `src.chromatica.core.histogram`.
> 6.  Return the normalized histogram.

**Prompt 6: Set up FAISS and DuckDB Wrappers**

> **Goal**: Create classes to manage the FAISS index and DuckDB database.
> **Tool**: `filesystem` (write\_file)
> **Instructions**: Create a file at `src/chromatica/indexing/store.py`.
>
> 1.  Define a class `AnnIndex` as shown in Section E, which wraps the `faiss.IndexHNSWFlat`. The `add` method MUST apply the Hellinger transform (`np.sqrt`) before adding vectors.
> 2.  Define a class `MetadataStore` that manages a DuckDB connection. It should have methods like `setup_table()`, `add_batch(image_ids, histograms)`, and `get_histograms_by_ids(ids)`. This will store the raw, non-transformed histograms.

**Prompt 7: Create Offline Indexing Script**

> **Goal**: Write the main script to process a directory of images and populate the index/database.
> **Tool**: `filesystem` (write\_file)
> **Instructions**: Create a file at `scripts/build_index.py`. This script should:
>
> 1.  Take a directory path as a command-line argument.
> 2.  Initialize the `AnnIndex` and `MetadataStore` from the `store.py` module.
> 3.  Iterate through all images in the directory.
> 4.  For each image, use the `process_image` pipeline to generate its histogram.
> 5.  Add the Hellinger-transformed histogram to the FAISS index and the raw histogram (along with its ID/path) to DuckDB.
> 6.  Save the FAISS index and close the DuckDB connection upon completion.

### Phase 3: Search and API (Weeks 3-5)

**Prompt 8: Implement Sinkhorn Reranking Logic**

> **Goal**: Implement the high-fidelity reranking functions.
> **Tool**: `filesystem` (write\_file)
> **Instructions**: Create `src/chromatica/core/rerank.py`.
>
> 1.  Implement the `build_cost_matrix` function exactly as specified in Section E of `critical_instructions.md`. It must be pre-computed.
> 2.  Implement the `rerank_candidates` function. It takes a query histogram and a list of candidate histograms, computes the Sinkhorn distance for each using `ot.sinkhorn2`, and returns a sorted list of `(id, score)`.

**Prompt 9: Create Query Processor**

> **Goal**: Write a function to convert API query parameters into a query histogram.
> **Tool**: `filesystem` (write\_file)
> **Instructions**: Create `src/chromatica/core/query.py`. Write a function `create_query_histogram(colors: list[str], weights: list[float])`. This function will:
>
> 1.  Convert hex color codes to Lab values.
> 2.  Create a "softened" query histogram based on these Lab values and weights. This involves distributing the weight of each query color to the nearest bins in the 8x12x12 grid, similar to the image histogram creation. This is the "softened" query Lab histogram mentioned in Section C.
> 3.  Ensure the resulting histogram is L1-normalized.

**Prompt 10: Implement the Full Two-Stage Search Logic**

> **Goal**: Combine all pieces into a single search function.
> **Tool**: `filesystem` (write\_file)
> **Instructions**: Create `src/chromatica/search.py`. Write a function `find_similar(query_histogram: np.ndarray, index: AnnIndex, store: MetadataStore) -> list`. This function must perform the two-stage search:
>
> 1.  Use the `AnnIndex.search` method to retrieve the top-K candidates (use `RERANK_K` from config).
> 2.  Fetch the raw histograms for these candidate IDs from the `MetadataStore`.
> 3.  Use the `rerank_candidates` function to re-sort them by Sinkhorn distance.
> 4.  Return the final, reranked list of image IDs and their scores.

**Prompt 11: Build the FastAPI Endpoint**

> **Goal**: Expose the search functionality via a REST API.
> **Tool**: `filesystem` (write\_file)
> **Instructions**: Create `src/chromatica/api/main.py`.
>
> 1.  Initialize a FastAPI app.
> 2.  Load the FAISS index and DuckDB store on startup.
> 3.  Create the `GET /search` endpoint as specified in Section H.
> 4.  The endpoint should parse `colors` and `weights` query parameters.
> 5.  Use `create_query_histogram` to generate the query vector.
> 6.  Call the `find_similar` function to get results.
> 7.  Format the response into the exact JSON structure defined in Section H, including metadata like timings.

### Phase 4: Evaluation and Refinement (Weeks 6-8)

**Prompt 12: Create Sanity Check Script**

> **Goal**: Write a script to run the sanity checks defined in the plan.
> **Tool**: `filesystem` (write\_file)
> **Instructions**: Create `scripts/run_sanity_checks.py`. This script should programmatically execute the four sanity checks outlined in Section F (Monochrome, Complementary, Weight Sensitivity, Subtle Hues). For each check, define the query, call the search API (or function directly), and print the top 5 results so we can manually verify if they make sense.

**Prompt 13: Address Background Dominance Risk**

> **Goal**: Draft an implementation for the "saliency weighting" mitigation.
> **Tool**: `filesystem` (write\_file)
> **Instructions**: Following the mitigation plan in Section I, Risk 3, create a new function in `src/chromatica/core/histogram.py` called `build_saliency_weighted_histogram`. It should:
>
> 1.  Accept an RGB image as input.
> 2.  Use a simple saliency algorithm (e.g., spectral residual via `cv2.saliency.StaticSaliencySpectralResidual_create()`).
> 3.  Generate a saliency map where brighter pixels are more important.
> 4.  During histogram generation, weight the contribution of each pixel by its saliency value.
> 5.  This will be an alternative to `build_histogram` that we can enable with a flag.

**Prompt 14: Refactor for Memory Scaling**

> **Goal**: Propose code changes to address Risk 2 (Memory Footprint).
> **Tool**: `gitmvp` (read\_repository) and `filesystem` (edit\_file)
> **Instructions**: Read the current implementation in `src/chromatica/indexing/store.py`. Now, edit the `AnnIndex` class. Propose changes to switch the index from `faiss.IndexHNSWFlat` to `faiss.IndexIVFPQ`. Add comments explaining what Product Quantization is, what new training steps are required, and how this reduces memory usage at the cost of some accuracy, as stated in the mitigation plan.

**Prompt 15: Add an Evaluation Harness**

> **Goal**: Create a script to run a batch of queries and calculate performance metrics.
> **Tool**: `filesystem` (write\_file)
> **Instructions**: Create `scripts/evaluate.py`. The script should:
>
> 1.  Load a predefined set of test queries (e.g., from a JSON file).
> 2.  For each query, execute the search and record the latency.
> 3.  After running all queries, calculate and print the P95 latency.
> 4.  Provide a placeholder structure for loading ground truth labels and calculating Precision@10, as described in Section F. This prepares us for the human labeling task.

### Phase 5: Testing and Validation (Weeks 7-8)

**Prompt 16: Unit Test the Core Histogram Logic**
> **Goal**: Create unit tests to validate the correctness of the histogram generation and query processing logic.
> **Tool**: `filesystem` (write_file)
> **Instructions**: Create a new test file at `tests/test_core_logic.py`. Using the `pytest` framework, write tests for the functions in `src/chromatica/core/`.
> 1.  **Test `build_histogram`**: Create a small, synthetic image (e.g., a 4x4 pixel NumPy array) with known Lab values. Assert that the resulting histogram has the correct dimensions (1,152), is properly normalized (sums to 1), and that the pixel counts are distributed to the expected bins.
> 2.  **Test `create_query_histogram`**: Provide a simple two-color query (e.g., 50% red, 50% blue). Assert that the generated histogram is normalized and that the weights are concentrated in the correct regions of the color space.
> 3.  Follow all rules in `.cursorrules`.

**Prompt 17: Write Integration Test for the Search Pipeline**
> **Goal**: Test the end-to-end two-stage search process to ensure all components work together correctly.
> **Tool**: `filesystem` (write_file)
> **Instructions**: Create a file `tests/test_search_pipeline.py`. This test should:
> 1.  Create a small, in-memory "mock" FAISS index and DuckDB database with 5-10 sample image histograms.
> 2.  The samples should include obvious candidates for a test query (e.g., an image that is 99% red).
> 3.  Execute the `find_similar` function from `src/chromatica/search.py` against this mock data.
> 4.  Assert that the final ranked list is correct. For example, a query for pure red should return the "99% red" image as the top result. This validates that both the ANN search and the Sinkhorn reranking are functioning as expected.

**Prompt 18: Implement API Endpoint Tests**
> **Goal**: Write tests for the FastAPI endpoint to validate request handling, response formats, and error cases.
> **Tool**: `filesystem` (write_file)
> **Instructions**: Create `tests/test_api.py`. Use FastAPI's `TestClient` to write tests for the `/search` endpoint.
> 1.  **Test for a valid request**: Send a well-formed request like the one in Section H of `critical_instructions.md`. Assert that the HTTP status code is 200 and that the JSON response body matches the specified structure.
> 2.  **Test for invalid parameters**: Send requests with missing `colors` or `weights`, or with weights that don't sum to 1. Assert that the API returns appropriate 4xx error codes and informative error messages.
> 3.  **Test `k` parameter**: Send a request with `&k=5` and assert that the `results` array in the response contains exactly 5 items.

### Phase 6: Production Readiness and Documentation (Week 8)

**Prompt 19: Enhance API Documentation**
> **Goal**: Improve the auto-generated API documentation with detailed descriptions and examples.
> **Tool**: `filesystem` (edit_file)
> **Instructions**: Edit the file `src/chromatica/api/main.py`. Using FastAPI's documentation features:
> 1.  Add a descriptive title and description to the `FastAPI()` app instance itself.
> 2.  In the `@app.get("/search")` decorator, add `summary` and `description` parameters to explain what the endpoint does.
> 3.  Use FastAPI's `Query` dependency to add descriptions, examples, and validation rules for each query parameter (`colors`, `weights`, `k`, `fuzz`) right in the endpoint function's signature. This will automatically enrich the `/docs` page.

**Prompt 20: Containerize the Application**
> **Goal**: Create a `Dockerfile` to package the Chromatica application for easy deployment.
> **Tool**: `filesystem` (write_file)
> **Instructions**: Create a `Dockerfile` in the project root. The Dockerfile should:
> 1.  Start from a slim Python base image (e.g., `python:3.10-slim`).
> 2.  Set up a working directory and copy the `requirements.txt` file.
> 3.  Install all Python dependencies.
> 4.  Copy the entire `src` directory into the container.
> 5.  Copy the pre-built FAISS index file and the DuckDB database file into the container.
> 6.  Expose the port used by Uvicorn (e.g., 8000).
> 7.  Set the `CMD` to run the application using `uvicorn`, pointing to the FastAPI app instance in `src.chromatica.api.main`.

**Prompt 21: Implement a Batch Update Script**
> **Goal**: Create a script for adding new images to the index without rebuilding everything from scratch.
> **Tool**: `filesystem` (write_file)
> **Instructions**: Create a new script `scripts/update_index.py`. This script will be a variation of `build_index.py` and must:
> 1.  Accept a directory path for new images.
> 2.  Load the *existing* FAISS index and open the *existing* DuckDB database in read-write mode.
> 3.  Process only the new images into histograms.
> 4.  Use the `AnnIndex.add` and `MetadataStore.add_batch` methods to append the new data to the existing stores.
> 5.  Save the updated index back to disk. This is a key feature for making the system maintainable, as mentioned in the "Production Readiness" milestone.

