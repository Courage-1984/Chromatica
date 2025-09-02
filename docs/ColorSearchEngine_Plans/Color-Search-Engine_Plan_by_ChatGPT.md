# Color-Search-Engine_Plan_by_ChatGPT

Executive Summary

We propose a two-stage color‐based image search system that represents each image by a perceptually uniform color histogram and retrieves matches via a fast ANN index followed by an exact re-ranking. In the indexing phase, we convert each image to a perceptual color space (e.g. CIE Lab* or JzAzBz), quantize the color volume into a fixed 3D grid (e.g. 10×10×10 bins), and build a sparse normalized histogram (palette weights) for the image. We optionally apply a small Gaussian blur in color space (“soft assignment”) to spread each pixel’s weight to neighboring bins, improving robustness to quantization. The resulting histogram vector (e.g. 1–2K dimensions) is then embedded for fast ANN search: for example, we take the square-root of each bin (the Hellinger transform), so that Euclidean distance corresponds to Bhattacharyya similarity
docs.opencv.org
. We index these vectors in FAISS (HNSW or IVF-PQ) or via DuckDB’s VSS extension
duckdb.org
for large-scale nearest-neighbor retrieval.

At query time, the user supplies one or more query colors (hex values) with weights. We convert each color to the same space, assign it to the nearest histogram bins (applying a Gaussian spread if a fuzziness parameter is given), and normalize to form a query histogram. We then perform a fast ANN lookup (L2 or cosine distance on the transformed vector) to retrieve, say, the top‑K candidates. Finally, we re-rank those top results using a higher-fidelity color-distance metric such as the Earth Mover’s Distance (Wasserstein-1) in Lab space, computed by the Sinkhorn algorithm (via the POT library). This two-stage approach combines speed with accuracy: the ANN (e.g. Hellinger/Eucleadian) provides sub-150 ms lookup, and the exact EMD rerank on ~200 images adds minimal overhead for precise results.

Throughout, we focus on perceptual color metrics. CIELab (intended to be uniform
en.wikipedia.org
) or its HDR successor JzAzBz
ajalt.github.io
ensure that Euclidean distances roughly match human color differences. Our pipeline also handles wrap-around hues (if using a cylindrical space) and allows configurable “fuzziness” by Gaussian blurring in Lab. We will evaluate the system on ~100K sample images (e.g. from Unsplash-25K or COCO), measuring precision@K, mAP, and user studies for monochrome/complementary/multi-color queries, and tune bin size, softening, and K for rerank.
Deep Research Notes

    Color spaces (perceptual uniformity): We use a device-independent perceptual space. CIE Lab* was designed to be (approximately) perceptually uniform
    en.wikipedia.org
    , so that Euclidean distance correlates with perceived color difference. A newer space, JzAzBz (Safdar et al. 2017), is even more uniform for HDR colors
    ajalt.github.io
    . In practice Lab is easy to compute (OpenCV) and widely supported, but JzAzBz could further improve fidelity at extreme brightness (if implemented). Key point: distances in these spaces reflect human color percepts, unlike RGB or raw HSV.

    Color binning / histogram representation: We quantize the 3D color space into bins (e.g. 10×10×10 Lab bins = 1,000 bins). Alternatives include adaptive palettes (median-cut, k-means, Gaussian mixtures), but fixed-grid histograms are simple and produce fixed-length vectors. Each image’s histogram is normalized to sum=1, storing the relative weight of each color bin. We also consider soft assignment: after counting pixels into bins, we convolve the 3D histogram with a small Gaussian kernel so each bin’s weight spreads to neighbors (in Lab space). This reduces sensitivity to quantization boundaries and model color fuzziness. This idea is similar to kernel density estimation on the discrete grid.

    Distance metrics: We surveyed several ways to compare color distributions:

        Wasserstein/EMD: The Earth Mover’s Distance treats each histogram as a distribution in color space and computes the minimal “work” to transform one into the other. It inherently combines color differences (ground distance between bin centroids) and weight differences
        cran.r-project.org
        cs.cmu.edu
        . EMD handles partial matches gracefully (e.g. a large gray bin vs a large black bin will incur moderate cost, reflecting the similarity of black/gray), whereas binwise distances (like χ²) would treat them as completely different
        cran.r-project.org
        . Indeed, Rubner et al. and others have shown EMD to be a flexible, powerful metric for color retrieval
        cs.cmu.edu
        cran.r-project.org
        . We will use the Sinkhorn approximation (entropic regularization) via Python Optimal Transport (POT) for speed in reranking.

        Quadratic-form (bin similarity): A generalized Mahalanobis distance on histograms uses a bin-similarity matrix. If we define a similarity A<sub>ij</sub> = exp(-d_lab(c<sub>i</sub>, c<sub>j</sub>)/σ), then distance = (h₁-h₂)<sup>T</sup> A (h₁-h₂). This captures cross-bin correlations, but is expensive (matrix multiply) and less common in ANN. We note this option but will prioritize EMD.

        Histogram intersection / Bhattacharyya (Hellinger) / χ²: These are simple, fast measures. Intersection = ∑min(h₁,h₂) prioritizes common mass. Bhattacharyya distance = sqrt(1 - ∑√(h₁ h₂)) (Hellinger metric) measures “overlap” of distributions
        docs.opencv.org
        . χ² distance = ∑(h₁-h₂)²/(h₁+h₂) treats each bin independently. We embed histograms via the Hellinger transform (xᵢ←√hᵢ) so that L₂ distance ≈ Bhattacharyya (Hellinger) distance. This allows fast ANN search (Euclidean) on the transformed vectors, as suggested by prior work on histogram embeddings.

    Hue wrap-around: If we used HSV or HCL, we’d need to handle circular hue (0° ≡ 360°). In Lab we avoid this, since hue is implicit in a*,b*. If using a cylindrical Lab (LCh) we would explicitly wrap the angle. For optional query “fuzziness” in hue, we could blur angularly.

    ANN indexing: For large scale (100K–1M images), we use FAISS (CPU) or DuckDB with its Vector Similarity Search (HNSW) extension
    duckdb.org
    . We store each histogram (after sqrt) as a vector. For FAISS, we could use IndexHNSWFlat for fast (no training) search, or IVF+PQ to compress. HNSW gives very fast queries (~ms) and high recall
    duckdb.org
    duckdb.org
    . For example, in DuckDB one can CREATE INDEX img_idx ON images USING HNSW (histogram_vec) WITH(metric='l2sq')
    duckdb.org
    . We will tune quantizer/index (HNSW neighbors or IVF clusters) to meet latency/accuracy targets.

    Two-stage retrieval: ANNs (L2 on sqrt-hist) give a rough nearest-neighbors in color space. We then rerank top-K (say K=200) with our “true” metric (Sinkhorn-EMD in Lab). This ensures distance measurements respect color geometry and weights exactly. This approach (fast prefilter + slow rerank) is standard in high-dimensional search.

    Handling query weights and fuzziness: A query is “colors+weights”. We convert each hex to Lab and add weight to the corresponding histogram bins. We then apply Gaussian smoothing (in Lab space) to the query bins; the user’s fuzz parameter can control the kernel width. Finally we normalize to sum=1. For example, a query palette red:49%, yellow:51% yields a two-peak distribution. Without blur, only bins nearest those colors get weight; with blur, nearby colors also contribute, making matching more forgiving.

    Quality controls: We will include options to handle known pitfalls: e.g. skin-tone bias (many images have fleshtones). If a query color is similar to skin hues, the system may flood images with faces. Mitigation: provide an option to down-weight skin-region pixels (using a simple skin-color mask) during indexing, or annotate images with detected faces and ignore large face areas in histogram. Transparency/alpha: treat fully-transparent pixels as background (ignore) so they do not distort the palette. Background masking: if an image has a large uniform background, we could optionally segment the subject (e.g. via saliency) and index only foreground colors. For a first version, we assume full images. Thumbnails vs full-res: we index on a fixed-size (e.g. 256px) thumbnail for speed. This captures palette well with minimal cost.

    Palette extraction trade-offs: Instead of fixed bins, one could extract a small palette per image (via median-cut or k-means), storing e.g. 5 cluster colors+weights. This reduces vector dimension (5×Lab vectors) but complicates indexing (variable-length signatures). We prefer fixed histograms for simplicity and direct vector embedding.

    Related work: TinEye’s MultiColorSearch presumably uses similar concepts (color histograms and EMD). Synthesio’s blog “Journey to a color search engine” describes k-means on query colors and custom NN search, which we improve by direct histogram methods. Prior CBIR literature (Rubner et al. 1998) established EMD for images; recent libraries (e.g. POT) allow fast implementation. The DuckDB VSS blog
    duckdb.org
    duckdb.org
    confirms that HNSW indexing is effective for vector search, which we leverage.

System Design

    Color Space & Quantization: We choose CIE L*a*b* by default (with D65 white). Lab was explicitly intended as (approximately) perceptually uniform
    en.wikipedia.org
    , meaning ΔE approximates human-judged color difference. For extreme or HDR images we note JzAzBz as an option
    ajalt.github.io
    . Using Lab avoids hue wrap issues. We quantize L∈[0,100] and a*,b*∈[-128,127] into a fixed grid. A baseline is 10×10×10 bins (~1,000 bins). This balances expressiveness vs. vector size. If needed, 12×12×12 (~1,728 bins) or 20×20×20 (~8,000 bins) can capture more nuance but at memory cost. We will empirically tune bin count.

    Histogram Representation: Each image is converted to Lab (via OpenCV/cv2.cvtColor) and optionally downsampled (e.g. 256px). We build a 3D histogram: use cv2.calcHist or np.histogramdd to count Lab pixel values into bins. Then soften: we convolve the 3D histogram with a small Gaussian kernel (σ tuned, e.g. 1–2 bins). This “spreads” each color’s weight to nearby bins, making the descriptor robust to small shifts. Finally normalize so the histogram sums to 1 (a probability distribution). Sparse storage: most bins will be zero if palette is limited; we store as a dense float vector for indexing (sparse FAISS is possible but simpler to keep fixed-size arrays).

    Distance Metric and Embedding: - Fast metric (ANN): We use the square-root of each histogram bin (Hellinger/Bhattacharyya trick) so that Euclidean distance ≈ Hellinger distance
    docs.opencv.org
    . Thus we index the vector v = √h. The ANN engine (FAISS or DuckDB HNSW) operates on v with L₂ (or inner product) metrics. This approximates a “binwise overlap” similarity.

        Exact metric (rerank): For top-K candidates we compute a high-fidelity color distance. We choose Wasserstein-1 (EMD) in Lab space. Concretely, treat each histogram h as a “signature” of weighted Lab points (the bin centers). The cost matrix D<sub>ij</sub> is the Euclidean distance between Lab centroids of bin i and j. Then EMD(h_query, h_img) = min<sub>flow</sub> ∑F<sub>ij</sub>D<sub>ij</sub>, which we compute via Sinkhorn regularization (POT library). This handles color and weight jointly
        cran.r-project.org
        cran.r-project.org
        .

        (Optional) As a quicker filter, χ² or Intersection could be used, but they ignore cross-bin color similarity. We will mainly rely on ANN/Hellinger + EMD for quality.

    Indexing & Scaling: We use FAISS (CPU) for ANN. For up to ~100K images, an IndexFlat (exact) or IndexHNSWFlat provides fast L2 search. For 1M+ images, we propose HNSW (32 neighbors) or IVF-PQ. E.g. faiss.IndexHNSWFlat(d,32), or IndexIVFPQ(d,nlist, m, 8). We store each image’s √histogram as a float32 vector of length N_bins. FAISS holds these vectors in RAM; we may add PQ compression to save memory (e.g. 8-byte per vector via PQ). For metadata (image IDs, swatches), we use DuckDB: e.g., a table images(id, vect ARRAY[float], swatches JSON, ...). DuckDB’s new VSS extension even allows creating an HNSW index on a float[] column
    duckdb.org
    , enabling purely-SQL ANN search. We will likely prototype in Python/FAISS and use DuckDB for metadata queries (e.g. join results with image info).

    Query Handling: The API accepts colors=hex1,hex2,...&weights=w1,w2,.... We parse and validate (normalize weights to sum to 1). Convert each hex to Lab. Build a query histogram: for each color, find the corresponding 3D bin index; add its weight to that bin. (If using continuous, one could distribute to two adjacent bins, but we simply pick nearest for simplicity.) Apply a Gaussian blur (in Lab) around each nonzero bin, controlled by a user fuzz parameter (optional). Normalize query hist to sum=1. Transform via sqrt. Query the ANN index (search(query_vec, K_fast)) to get candidate IDs. Fetch their histograms/ID from storage and compute EMD to the query (e.g. via ot.emd2(query_hist, img_hist, D)), then sort by EMD. Return top results, including image IDs, distances, and each image’s top-k palette (to show matched swatches).

    Quality Controls: - Skin-tone adjustment: We detect skin regions (e.g. simple color threshold or off-the-shelf face detector) during indexing; either ignore them in the histogram or tag images as containing skin. This avoids biasing retrieval when query colors match skin. - Transparency: For images with alpha, we ignore fully transparent pixels (or treat them as background white). - Background masking: Optionally segment the dominant subject (if time allows) and exclude large uniform backgrounds from histograms. - Thumbnail size: We index on a fixed thumbnail (e.g. 256×256) to speed up color extraction. This preserves overall palette with minimal loss.

    API & UX: Expose REST endpoints like:

        POST /index (with image file or URL) to add to database.

        GET /search?colors=RRGGBB,...&weights=...&k=...&fuzz=... returns top-k results.
        Results include image IDs, distances, and key swatches. We fix random seeds (in FAISS, binning) for reproducibility. Optionally cache recent queries.

References: CIELab uniformity
en.wikipedia.org
; JzAzBz perceptual space
ajalt.github.io
; EMD in image retrieval
cran.r-project.org
cs.cmu.edu
; DuckDB HNSW indexing
duckdb.org
.
Algorithm Specification

    Histogram construction (image → Lab histogram): Let I(x,y) be an image. Convert to Lab: L(x,y), a(x,y), b(x,y) = cv2.cvtColor(I, COLOR_BGR2LAB). Define bin edges for L,a,b each (e.g. equally spaced). Compute 3D histogram:

hist[i,j,k] = ∑_{pixels p} 1{ L(p)∈bin_i, a(p)∈bin_j, b(p)∈bin_k }.

Apply Gaussian smoothing: convolve hist = hist * G (a 3×3×3 Gaussian). Normalize: h = hist / sum(hist).

Query histogram (colors+weights → Lab histogram): For each query color c_n with weight w_n: convert c_n → Lab (Cv2 or skimage). Find the nearest bin index (i,j,k) in Lab space. Add w_n to h_query[i,j,k]. After all colors, (optionally) blur h_query, then normalize.

Distance embedding (Hellinger): Transform vector v = sqrt(h) elementwise. This makes L2 distance:
d(vq,vi)=∥vq−vi∥2≈∑(hq−hi)2=2(1−∑hqhi),d(vq​,vi​)=∥vq​−vi​∥2​≈∑(hq​
​−hi​
​)2
​=2(1−∑hq​hi​
​)

​,
which is (up to factor) the Bhattacharyya/Hellinger distance
docs.opencv.org
. We feed v to ANN.

Approximate Nearest Neighbor: Build a FAISS index on vectors v_i (one per image). E.g.

index = faiss.IndexHNSWFlat(D, M=32)  # D = dimension = #bins
index.add(np.array([v_i]))

Query: _, idxs = index.search(v_q.reshape(1,-1), K_fast).

Wasserstein re-ranking: Let h_q, h_i be the query/image histograms (normalized). Let c_1,...,c_N be the centers of the N bins in Lab space. Precompute pairwise ground distances D[k,l] = ||c_k - c_l||. Compute EMD (Wasserstein-1) between distributions:
EMD(hq,hi)=min⁡Fkl≥0∑k,lFkl Dkls.t.  ∑lFkl=hq,k,  ∑kFkl=hi,l.
EMD(hq​,hi​)=Fkl​≥0min​k,l∑​Fkl​Dkl​s.t.l∑​Fkl​=hq,k​,k∑​Fkl​=hi,l​.

In practice, use Sinkhorn: emd_dist = ot.emd2(h_q, h_i, D, reg=ε). Sort candidates by emd_dist.

Pseudocode (search):

    function search_color(colors, weights, K=50, fuzz=0):
        # Build query histogram
        h_q = zeros(bins^3)
        for (c,w) in zip(colors,weights):
            lab = hex_to_lab(c)
            i,j,k = find_bin(lab)
            h_q[i,j,k] += w
        if fuzz>0:
            h_q = gaussian_blur(h_q, sigma=fuzz)
        h_q /= sum(h_q)
        v_q = sqrt(h_q)            # Hellinger embed
        
        # ANN recall
        idxs = ANN_index.search(v_q, K_fast)
        candidates = []
        for idx in idxs:
            h_i = load_histogram(idx)
            dist = SinkhornEMD(h_q, h_i, ground_dist)
            candidates.append((idx, dist))
        sort by dist; return top-K with image IDs and dist.

Implementation Plan

    Milestones / Schedule (weeks):

        Week 1: Prototype color histogram extraction. Use OpenCV to convert images to Lab, build 3D histogram (10×10×10). Normalize and test on sample images. Verify palette/visualization.

        Week 2: Implement query processing. Parse hex+weights, build query hist, apply blur. Implement simple linear search using Hellinger or χ² to validate retrieval semantics.

        Week 3: Integrate FAISS. Vectorize all image histograms (Hellinger transform) and build an ANN index (HNSW). Test nearest-neighbor search on query vectors.

        Week 4: Integrate POT for Wasserstein. Precompute bin centers and cost matrix. For top candidates, compute Sinkhorn-EMD distances. Validate re-ranking improves results.

        Week 5: Optimize and scale. Replace IndexFlat with IVF-PQ if needed. Tune ANN parameters for speed. Store vectors/metadata in DuckDB; create HNSW index via DuckDB VSS extension for metadata joins.

        Week 6: Evaluation. Run retrieval on ~10K–100K images, measure P@K/mAP with sample queries. Perform ablations (bins=5 vs10 vs20, Lab vs RGB vs Jz, Hellinger vs χ²) and document.

    Key Code Snippets:

import cv2, numpy as np

def image_to_lab_hist(image, bins=(10,10,10), blur_sigma=0):
    # Convert BGR→Lab (OpenCV scales L:0-255, a:0-255, b:0-255)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hist = cv2.calcHist([lab], [0,1,2], None, list(bins), [0,256, 0,256, 0,256])
    if blur_sigma > 0:
        import scipy.ndimage
        hist = scipy.ndimage.gaussian_filter(hist, sigma=blur_sigma)
    hist = hist.flatten()
    if hist.sum() > 0:
        hist = hist / hist.sum()
    return hist  # 1D array length bins[0]*bins[1]*bins[2]

# Example: build and index histograms
images = [...]  # list of OpenCV images
hists = np.array([image_to_lab_hist(img) for img in images], dtype='float32')
# Hellinger transform
vecs = np.sqrt(hists)
import faiss
d = vecs.shape[1]
index = faiss.IndexHNSWFlat(d, 32)  # HNSW, 32 neighbors
index.hnsw.efConstruction = 200
index.add(vecs)

# Query
def query_colors(hex_colors, weights, K=50, fuzz=1.0):
    # Build query histogram
    hq = np.zeros_like(hists[0])
    for hexcol,w in zip(hex_colors, weights):
        rgb = tuple(int(hexcol[i:i+2],16) for i in (0,2,4))
        bgr = np.array([[[rgb[2], rgb[1], rgb[0]]]], dtype=np.uint8)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0,0]
        # find bin indices
        bin_idx = (int(lab[0]/(256/bins[0])), 
                   int(lab[1]/(256/bins[1])),
                   int(lab[2]/(256/bins[2])))
        idx_flat = bin_idx[0]*bins[1]*bins[2] + bin_idx[1]*bins[2] + bin_idx[2]
        hq[idx_flat] += w
    # Optional blur (on 3D grid)
    if fuzz > 0:
        hq = hq.reshape(bins)
        hq = scipy.ndimage.gaussian_filter(hq, sigma=fuzz).flatten()
    if hq.sum() > 0: hq /= hq.sum()
    vq = np.sqrt(hq.astype('float32'))
    # ANN search
    _, ids = index.search(vq.reshape(1,-1), K)
    return ids[0]

And rerank example with POT (Sinkhorn EMD):

    import ot
    # Precompute bin centers:
    L_bins = np.linspace(0,100,bins[0])
    a_bins = np.linspace(-128,127,bins[1])
    b_bins = np.linspace(-128,127,bins[2])
    centers = np.array([[L,a,b] for L in L_bins for a in a_bins for b in b_bins])
    ground_dist = ot.dist(centers, centers)  # cost matrix
    # For each candidate:
    for idx in candidate_ids:
        hi = hists[idx]  # image histogram
        emd = ot.emd2(hq, hi, ground_dist)  # exact EMD
        # store (idx, emd) and sort later

    Batch indexing / updates: We will process images in batches, converting and normalizing in vectorized form. The FAISS index (HNSW) supports incremental additions. DuckDB tables (images, histograms, swatches) can be updated in bulk or via SQL inserts.

Evaluation Plan

    Datasets: We will test on a subset of ~10–50K images. Possible sources: COCO, Unsplash-Lite, or a crawled image set. Each image’s palette is extracted and indexed. We will split for potential ground truth (though true “color-similarity” labels are subjective).

    Queries: - Monochrome: single-color queries (e.g. #FF0000 red). Expected to retrieve images with strong red.

        Dual colors: e.g. complementary (blue+orange) or thematic (forest green + brown), with weights.

        Multi-color: e.g. a sunset (#FFAA00 50%, #223377 50%).

        Edge cases: grayscale vs color difference, 0-weight skip.

        Fuzziness test: same query with fuzz=0 vs fuzz>0.

    We will qualitatively inspect results and refine metrics. Where possible, ask human raters to grade top-10 results for relevance to query palette.

    Metrics: - Precision@K: Fraction of top-K images that a user rates as color-matching.

        Normalized Discounted Cumulative Gain (NDCG): if we assign graded relevance (exact match vs partial).

        mAP: mean average precision over a set of queries, if binary relevance can be defined.

    We will also compute quantitative similarity of histograms (Bhattacharyya or 1−EMD) to gauge matching.

    Ablation studies: We will test variants: Lab vs RGB vs HSV color space; bin counts (e.g. 8³ vs 12³); with/without softening; Hellinger vs χ² vs no rerank. This shows which choices matter. For example, we expect Lab+EMD >> RGB+L2 in retrieval quality
    cran.r-project.org
    .

    Performance: Measure indexing time (histogram build, index build) and query latency (ANN+rerank). Targets: for N=100K, index build <1h, memory <8GB (with PQ compression), query (ANN) <150 ms, full search (ANN+K=200 rerank) <500 ms at P95. We will log timing and ensure scalability to ~1M.

Risks & Mitigations

    Color ambiguity: Color alone may not capture “semantic” relevance. A query for orange-yellow may return oranges OR giraffes (both have orange) and we cannot disambiguate without semantics. Mitigation: User must know it’s color-only. We can combine with optional image-tags filtering, but out of scope.

    Skin-tone bias: A red/pink query might mostly match people’s faces. We will mitigate by detecting and down-weighting skin regions (or providing a “exclude faces” mode).

    Circular hue: If we had chosen HSV/HSL, hue wrap (0=360) is tricky. We avoid it by using Lab.

    High dimension: A 10×10×10 hist → 1,000-D vector. 1M images → 1000M floats (4GB) before compression. Mitigation: Use PQ or float16 storage (DuckDB supports arrays of floats). Or reduce bins if memory is tight. HNSW overhead adds ~2×, so plan accordingly.

    Complexity of EMD: Exact EMD is O(n³) in naive LP. Using Sinkhorn (entropic regularization) reduces this to ~O(n²) or less. Still for K=200 and n≈1000 bins, this is feasible (<500 ms). If needed, we can reduce bins or prune small weights.

    Data freshness: If images are frequently added, we need dynamic index updates. HNSW handles adds but not deletes easily. Mitigation: Periodic re-indexing or marking deleted flags.

    Evaluation ground truth: Without labeled “color-similar” pairs, quantitative evaluation is hard. We will rely on user studies and intuitive tests. Lack of ground truth is inherent; we emphasize user-centered metrics.

Next Steps

    Prototype core pipeline: Implement image→Lab-histogram extraction, indexing in FAISS, and query logic (with Hellinger). Test retrieval qualitatively on a handful of images and queries.

    Integrate re-ranking: Add POT-based Sinkhorn EMD to rerank top-100 candidates. Compare results with/without EMD.

    Setup DuckDB metadata store: Store image IDs, file paths, vector (as float array), and swatches. Experiment with DuckDB’s VSS extension for ANN in SQL.

    Scale up: Ingest a larger image set (10K+) and benchmark indexing time, RAM. Adjust bin size, softness, and ANN parameters.

    Evaluation harness: Define a set of test queries and automatic script to compute precision@K and NDCG (with manual labeling if needed).

    Iterate on UI/UX: Design API (REST) endpoints and test from a simple client. Optionally start a minimal web demo to visualize query vs results.

The plan blends proven classical methods (color histograms, Hellinger/Bhattacharyya, EMD) with modern ANN indices to deliver a practical and scalable color search engine. Each design choice (Lab space, histogram vs clusters, metric) will be validated via ablation, ensuring we meet quality and performance goals.

Sources: We leverage color-vision research (CIELab uniformity
en.wikipedia.org
, JzAzBz uniformity
ajalt.github.io
), classic image retrieval (Rubner et al., EMD for histograms
cs.cmu.edu
cran.r-project.org
), and recent engineering writeups (DuckDB vector search
duckdb.org
). These inform our quantitative design of distance and indexing techniques.
Citations

OpenCV: Histogram Comparison
https://docs.opencv.org/3.4/d8/dc8/tutorial_histogram_comparison.html

Vector Similarity Search in DuckDB – DuckDB
https://duckdb.org/2024/05/03/vector-similarity-search-vss.html

CIELAB color space - Wikipedia
https://en.wikipedia.org/wiki/CIELAB_color_space

JzAzBz
https://ajalt.github.io/colormath/api/colormath/com.github.ajalt.colormath.model/-jz-az-bz/

Color Distance Metrics
https://cran.r-project.org/web/packages/colordistance/vignettes/color-metrics.html

journal.dvi
https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/rubner-jcviu-00.pdf

Color Distance Metrics
https://cran.r-project.org/web/packages/colordistance/vignettes/color-metrics.html

Vector Similarity Search in DuckDB – DuckDB
https://duckdb.org/2024/05/03/vector-similarity-search-vss.html

Vector Similarity Search in DuckDB – DuckDB
https://duckdb.org/2024/05/03/vector-similarity-search-vss.html

