# Build a Color Search Engine (à la TinEye Multicolr)

## Role

Act as a senior ML/IR engineer and architect. Think deeply, research rigorously, and deliver a comprehensive, step-by-step plan plus reference code snippets.

## Context

* Reference demo: [https://labs.tineye.com/multicolr/](https://labs.tineye.com/multicolr/)
* Reference article: [https://medium.com/synthesio-engineering/a-journey-towards-creating-a-color-search-engine-194f1c388680](https://medium.com/synthesio-engineering/a-journey-towards-creating-a-color-search-engine-194f1c388680)
* Definition: A “color search engine” takes one or more query colors (with optional weights) and returns images whose dominant palette closely matches the query palette.

## Goal

Return images with the **closest color palette** to a **query palette with weights** (e.g., color count / total pixels). Example of colors+weights: [https://labs.tineye.com/multicolr/#colors=ea6a81,f6d727;weights=49,51](https://labs.tineye.com/multicolr/#colors=ea6a81,f6d727;weights=49,51)

## Tech constraints (use these unless strongly justified otherwise)

* Language: **Python**
* Libraries: **OpenCV** (image I/O), **scikit-image** (color conversion), **NumPy** (math)
* (Optional if beneficial): **faiss-cpu** for ANN, **scikit-learn** for clustering, **DuckDB**/**SQLite** for metadata.

## Known pitfalls from prior attempts (address these explicitly)

1. NN from article produced poor results.
2. KMeans + Euclidean on \[colors + weights] struggled to combine distances and was expensive per query; didn’t fit a standard vector DB easily.
3. HSV quantization (≈64 colors) → histogram vector; Gaussian blur for query fuzziness; issues: speed, quality, and **hue wraparound** (0° ≡ 360°) not handled.

## Requirements (what to design and justify)

1. **Color space & quantization**

   * Choose and justify: **CIE Lab** or **JzAzBz** (perceptual uniformity) over HSV/RGB.
   * Binning strategy (e.g., 10×10×10 Lab grid, \~1–2K bins; or learned palette).
   * Handle **circular dimensions** if HSV/HCL is used (e.g., hue wraparound).
2. **Representation**

   * Build **sparse normalized histograms** (palette weights) per image.
   * Consider **soft assignment** (e.g., Gaussian spread to neighboring bins) to improve robustness.
3. **Distance / similarity**

   * Propose and compare:

     * **EMD / Wasserstein** on color histograms (with ground distance in Lab).
     * **Quadratic-form distance** using a bin-similarity matrix (Mahalanobis-like).
     * **Histogram intersection / χ² / Hellinger (Bhattacharyya)** with kernel mappings to enable fast ANN.
   * Ensure **hue wraparound** if using hue-based coordinates.
   * Provide a **two-stage retrieval**: fast ANN (proxy metric) → exact rerank on top-K with higher-fidelity metric (e.g., Sinkhorn-EMD).
4. **Indexing & scaling**

   * ANN options: **FAISS** (IVF/PQ or HNSW) on embedded vectors.
   * Show how to map non-Euclidean distances to ANN-friendly space (e.g., **Hellinger transform** `sqrt(hist)` for Bhattacharyya, **χ² feature map**, or **Sliced Wasserstein** embeddings).
   * Storage of vectors + metadata; precomputation pipeline; batch updates.
5. **Query handling**

   * Input: hex colors + weights (sum to 1).
   * Convert query to same space; apply **softening/fuzziness** (Gaussian in Lab; circular Gaussian for hue).
   * Build query histogram and run the two-stage search.
6. **Quality controls**

   * Options: skin-tone bias handling, transparency/alpha pixels, background masking, thumbnail vs full-res extraction.
   * Palette extraction for indexing: compare **median cut / k-means / GMM** vs fixed-grid histograms; pick one and justify.
7. **Evaluation**

   * Metrics: Precision\@K, nDCG, mAP with human-labeled pairs; **ablation** for color space, bins, and distance.
   * Sanity tests: monochrome queries, complementary colors, multi-color with weights.
8. **Performance targets**

   * Index size: N images (specify N), memory budget.
   * Query latency: P50/P95 targets (e.g., <150ms ANN, <500ms with rerank on K=200).
   * Throughput goals and batch indexing time.
9. **API & UX**

   * REST endpoints: `/index`, `/search?colors=...&weights=...&k=...&fuzz=...`.
   * Deterministic seed for reproducibility.
   * Return: image IDs, distances, top swatches detected per result.
10. **Deliverables**

    * Architecture diagram and data flow.
    * Step-by-step build plan with milestones/week-by-week timeline.
    * Minimal working **Python** reference (OpenCV/skimage/NumPy) for:

      * image → Lab histogram (soft assignment)
      * query colors → histogram
      * fast ANN search (if using FAISS, show setup)
      * rerank with higher-fidelity distance (e.g., Sinkhorn via POT if allowed, else your approximation)
    * Benchmarks on a small public dataset (e.g., COCO/Unsplash-lite) with reported metrics.

## Hard requirements to address directly

* Combine **color distance** and **weight distance** in a principled way (not ad hoc).
* Handle **circular hue** if using hue-based spaces.
* Provide a **vector-DB-compatible** embedding or kernel map to enable ANN, with a clear reranking stage.
* Explain how to tune bin size/softening and choose K for reranking.

## Output format

1. **Executive summary** (1–2 paragraphs).
2. **Deep research notes** (citations to key papers/implementations).
3. **System design** (choices + rationale, trade-offs).
4. **Algorithm spec** (math for distance/embedding; pseudocode).
5. **Implementation plan** (checklist + code snippets).
6. **Evaluation plan** (protocols, metrics, datasets).
7. **Risks & mitigations**.
8. **Next steps** (immediate tasks).

## Nice-to-haves (optional)

* Learning a low-dimensional **color embedding** (e.g., train a small autoencoder in Lab) but keep primary solution classical.
* Simple UI to visualize palette matches and per-bin contributions.
* Caching and query expansion (e.g., broaden around nearest color bins).

---
