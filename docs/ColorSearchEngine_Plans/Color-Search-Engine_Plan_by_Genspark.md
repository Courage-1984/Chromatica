# Color-Search-Engine_Plan_by_Genspark

Executive summary
We will build a production-ready color search engine that retrieves images whose dominant palettes best match a weighted, multi-color query. The core representation is a sparse, normalized histogram in a perceptually uniform color space (CIE Lab), populated with tri-linear soft assignment to neighboring bins to improve robustness. Queries are converted from hex to Lab and “softened” by a Gaussian in Lab-space, which naturally avoids hue wraparound pitfalls that arise in HSV/HCL. Retrieval uses a two-stage approach: a fast approximate nearest neighbor (ANN) search with an ANN-friendly proxy metric (Hellinger/Bhattacharyya via a sqrt transform of histograms) and a final rerank on top-K candidates with a higher-fidelity transport-based distance (entropy-regularized Wasserstein/Sinkhorn on a Lab-ground metric). This design unifies color and weight differences rigorously, scales efficiently, and addresses the common failure modes of KMeans+Euclidean and naive HSV quantization. [Swain & Ballard, 1991](https://www.cs.washington.edu/education/courses/455/09wi/readings/swainballard91.pdf) [Rubner et al., 2000](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/rubner-jcviu-00.pdf) [Cuturi, 2013](https://arxiv.org/abs/1306.0895) [FAISS docs](https://faiss.ai/index.html) [scikit-image docs](https://scikit-image.org/docs/0.25.x/api/skimage.color.html)

We justify Lab over JzAzBz for simplicity and ecosystem support; JzAzBz remains an optional upgrade for HDR/wide-gamut collections. We map non-Euclidean histogram similarities to an ANN-friendly space using the Hellinger transform and, optionally, additive chi-squared feature maps, enabling HNSW or IVF/PQ indexing in Faiss. We set clear performance targets and provide Python reference code for image indexing, query building, fast ANN search, and Sinkhorn reranking, alongside an evaluation plan on public datasets and an API spec mirroring TinEye Multicolr’s UX. [TinEye Multicolr](https://labs.tineye.com/multicolr/) [Vedaldi & Zisserman, 2011](https://www.robots.ox.ac.uk/~vgg/publications/2011/Vedaldi11/vedaldi11.pdf) [Malkov & Yashunin, 2018](https://arxiv.org/abs/1603.09320)

Deep research notes
- Histogram matching and intersection: Swain & Ballard introduced histogram intersection for color indexing, a fast, robust baseline for palette similarity, but limited to bin-to-bin matches and not cross-bin color proximity. [Swain & Ballard, 1991](https://www.cs.washington.edu/education/courses/455/09wi/readings/swainballard91.pdf)
- Earth Mover’s Distance (EMD): Rubner et al. formalized EMD for image retrieval; it measures minimal work to transform one distribution into another, naturally combining color deltas and weight differences. [Rubner et al., 2000](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/rubner-jcviu-00.pdf)
- Fast EMD and regularized OT: Pele & Werman accelerated EMD for thresholded ground distances (fastEMD). Cuturi introduced entropy-regularized OT (Sinkhorn) for near-linear-time approximations with GPU/CPU-friendly matrix scaling. [Pele & Werman, 2009](https://www.cs.huji.ac.il/~werman/Papers/ICCV2009.pdf) [Cuturi, 2013](https://arxiv.org/abs/1306.0895)
- Quadratic-form distances: Hafner et al. proposed quadratic-form distances using a bin-similarity matrix to capture cross-bin perceptual proximity, enabling fast bounds and indexing tricks. [Hafner et al., 1995](https://ieeexplore.ieee.org/document/391417/)
- Kernel embeddings for ANN: Hellinger/Bhattacharyya can be implemented with a sqrt-hist transform; additive chi-squared kernels admit explicit finite-dimensional feature maps for fast linear search. [Hellinger (overview)](https://en.wikipedia.org/wiki/Hellinger_distance) [Vedaldi & Zisserman, 2011](https://www.robots.ox.ac.uk/~vgg/publications/2011/Vedaldi11/vedaldi11.pdf)
- ANN systems: Faiss supports HNSW and IVF/PQ on CPU, with strong performance and compression options; HNSW offers high recall with sub-millisecond memory lookups. [FAISS docs](https://faiss.ai/index.html) [Johnson et al., 2017](https://arxiv.org/abs/1702.08734) [Malkov & Yashunin, 2018](https://arxiv.org/abs/1603.09320)
- Color spaces: CIE Lab is widely used and supported in scikit-image; JzAzBz improves uniformity, especially for HDR, but requires custom transforms; CIEDE2000 gives perceptual distance refinements when required. [scikit-image docs](https://scikit-image.org/docs/0.25.x/api/skimage.color.html) [Safdar et al., 2017](https://opg.optica.org/abstract.cfm?uri=oe-25-13-15131) [Sharma et al., CIEDE2000 notes](https://www.ece.rochester.edu/~gsharma/ciede2000/)
- Sliced Wasserstein embeddings offer linear-time, projection-based approximations that can be indexed like vectors and even hashed; useful as an alternative ANN proxy. [Kolouri et al., 2016](https://openaccess.thecvf.com/content_cvpr_2016/papers/Kolouri_Sliced_Wasserstein_Kernels_CVPR_2016_paper.pdf)
- Prior attempts and pitfalls: The Synthesio article reports weak results for naive neural approaches and KMeans+Euclidean palette matching, highlighting the need for cross-bin distances and principled combination of weight and color differences. [Synthesio Medium](https://medium.com/synthesio-engineering/a-journey-towards-creating-a-color-search-engine-194f1c388680) [TinEye Multicolr](https://labs.tineye.com/multicolr/)

System design (choices + rationale, trade-offs)
1) Color space & quantization
- Space: CIE Lab with D65 white point. Rationale: good perceptual uniformity, ubiquitous tooling (skimage.color.rgb2lab), straightforward Euclidean ground distances, and no hue wraparound problems, unlike HSV/HCL. JzAzBz optional upgrade for HDR/wide-gamut libraries, but requires more complex transforms and care with reference primaries. [scikit-image docs](https://scikit-image.org/docs/0.25.x/api/skimage.color.html) [Safdar et al., 2017](https://opg.optica.org/abstract.cfm?uri=oe-25-13-15131)
- Binning: Uniform 3D grid with B ≈ 0.7–1.5K bins. Recommended: L*: 8 bins over [0,100], a*: 12 bins over [-86, 98], b*: 12 bins over [-108, 95] → 1,152 bins. This balances fidelity and memory; a and b ranges cover sRGB Lab gamut. We’ll clamp to Lab ranges to avoid out-of-range artifacts. [Swain & Ballard, 1991](https://www.cs.washington.edu/education/courses/455/09wi/readings/swainballard91.pdf)
- Soft assignment: Tri-linear soft assignment to the 8 neighboring bins (equivalently a separable first-order B-spline kernel). Optionally add small Gaussian smoothing within up to 1-bin radius to reduce discretization artifacts, which empirically stabilizes retrieval without exploding density. [Hafner et al., 1995](https://ieeexplore.ieee.org/document/391417/)
- Alternative: Learned palettes (k-means/GMM) per image are avoided because they mismatch vector-DB assumptions, complicate ANN, and degrade cross-image comparability; fixed grids enable a universal embedding. [Synthesio Medium](https://medium.com/synthesio-engineering/a-journey-towards-creating-a-color-search-engine-194f1c388680)

2) Representation
- For each image, compute a sparse normalized Lab histogram h ∈ R^B (sum to 1). Use downsampled thumbnails (e.g., 256 px long side) to reduce I/O and ensure consistent noise characteristics. Discard fully transparent pixels; optionally mask uniform borders. [Swain & Ballard, 1991](https://www.cs.washington.edu/education/courses/455/09wi/readings/swainballard91.pdf)
- Store as: (indices, values) sparse vectors; persist dense sqrt(h) as a float32 vector for ANN. Keep top-τ bins (e.g., τ=256) by weight for compact reranking structures. [Hellinger (overview)](https://en.wikipedia.org/wiki/Hellinger_distance)
- Optional: compute LCh colors for reporting palettes; hue wraparound matters only in visualization/explanations, not in matching due to Lab usage. [CIEDE2000 notes](https://www.ece.rochester.edu/~gsharma/ciede2000/)

3) Distance / similarity
- Final rerank metric (high fidelity): Entropy-regularized Wasserstein (Sinkhorn) between discrete distributions on Lab bin centers. Ground distance Dij = ||ci − cj||2 in Lab; regularization ε tuned to smooth small binization noise while preserving structure. This naturally combines color distance and weight mismatch (mass movement). [Cuturi, 2013](https://arxiv.org/abs/1306.0895) [Rubner et al., 2000](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/rubner-jcviu-00.pdf)
- Proxy metric for ANN (fast): Hellinger/Bhattacharyya via sqrt transform: φ(h) = sqrt(h). The squared Euclidean distance between φ(h1) and φ(h2) is 2(1 − BC), where BC is the Bhattacharyya coefficient; this gives a strong, ANN-friendly proxy. [Hellinger (overview)](https://en.wikipedia.org/wiki/Hellinger_distance)
- Alternative proxies: Additive chi-square explicit feature maps (Vedaldi & Zisserman) allow linear ANN indexing under χ²-like similarity. Useful if χ² empirically outperforms Hellinger on your data. [Vedaldi & Zisserman, 2011](https://www.robots.ox.ac.uk/~vgg/publications/2011/Vedaldi11/vedaldi11.pdf)
- Alternative rerank: FastEMD for thresholded ground distances; useful if using piecewise-linear approximations or capped distances. [Pele & Werman, 2009](https://www.cs.huji.ac.il/~werman/Papers/ICCV2009.pdf)
- Quadratic-form distance: D(h1,h2) = (h1 − h2)^T A (h1 − h2), where A encodes bin similarities exp(−||ci − cj||^2/σ^2). This is fast but requires storing A or low-rank factorization; we prefer transport distance for principled mass movement. [Hafner et al., 1995](https://ieeexplore.ieee.org/document/391417/)

4) Indexing & scaling
- ANN index: Faiss HNSW (IndexHNSWFlat) on φ(h) with L2 metric gives high recall, easy dynamic updates. For billion-scale, use IVF-PQ (coarse quantizer + product quantization) with OPQ, compressing to 16–32 bytes/vector. [FAISS docs](https://faiss.ai/index.html) [Johnson et al., 2017](https://arxiv.org/abs/1702.08734) [Malkov & Yashunin, 2018](https://arxiv.org/abs/1603.09320)
- Mapping to ANN: Use φ(h)=sqrt(h). Optionally, replace with chi-square explicit features (scikit-learn AdditiveChi2Sampler) and index those vectors. [Vedaldi & Zisserman, 2011](https://www.robots.ox.ac.uk/~vgg/publications/2011/Vedaldi11/vedaldi11.pdf)
- Storage: Persist per-image sparse histogram, dense ANN vector, and metadata (image id, URI, thumbnail hash, top palette swatches) in SQLite/DuckDB. [TinEye Multicolr](https://labs.tineye.com/multicolr/)
- Batch updates: Micro-batches appended to HNSW or merged IVF indices; store staging tables for partial reindexing.

5) Query handling
- Input: list of hex colors and weights summing to 1. Convert hex to linear RGB then to Lab; create a small Gaussian kernel in Lab (σq) to soften each query color; deposit mass into nearby bins via tri-linear Gaussian weighting; normalize to get hq. [scikit-image docs](https://scikit-image.org/docs/0.25.x/api/skimage.color.html)
- Search: compute φ(hq)=sqrt(hq); ANN top-K (e.g., K=200); rerank with Sinkhorn distance on sparse supports (only non-zero bins in candidates vs. query).

6) Quality controls
- Handle transparency by ignoring alpha=0; optionally background removal via simple border color detection or fast segmentation to reduce background dominance. [Swain & Ballard, 1991](https://www.cs.washington.edu/education/courses/455/09wi/readings/swainballard91.pdf)
- Skin-tone bias: optionally detect faces and downweight skin bins slightly to avoid overfitting to portraits if your corpus is human-heavy. Use a flag to enable/disable. [Hafner et al., 1995](https://ieeexplore.ieee.org/document/391417/)
- Thumbnail normalization: standardize longest side to 256, convert using skimage rgb2lab with sRGB assumption (D65). [scikit-image docs](https://scikit-image.org/docs/0.25.x/api/skimage.color.html)

7) Evaluation
- Metrics: Precision@K, nDCG, mAP on labeled triplets (query palette, positive images, hard negatives). Include sanity cases: monochrome, complementary pairs, multi-color with imbalanced weights. [Rubner et al., 2000](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/rubner-jcviu-00.pdf)
- Ablations: Compare Lab vs JzAzBz; bins (B=512, 768, 1152, 1536); soft assignment radius; proxy metric (Hellinger vs χ²) and K for rerank; Sinkhorn ε. [Safdar et al., 2017](https://opg.optica.org/abstract.cfm?uri=oe-25-13-15131)
- Datasets: COCO 2017 val subset and an Unsplash-lite set with diverse colors; auto-generate “image palettes” as ground-truth hints, then curate human labels for top-50 per query. [TinEye Multicolr](https://labs.tineye.com/multicolr/)

8) Performance targets
- For N=10M images, B=768 dims (float32 sqrt-hists) would be ≈3 KB/vector ⇒ 30 GB raw; with IVF-PQ to 32B/vector ⇒ ~320 MB + coarse structures. HNSW can also be used with PQ. [FAISS docs](https://faiss.ai/index.html)
- Latency targets: ANN P50 < 150 ms, P95 < 250 ms; rerank K=200 with Sinkhorn P50 < 300 ms, P95 < 600 ms on CPU; total P50 < 450 ms, P95 < 850 ms. Use batching and vectorization. [Cuturi, 2013](https://arxiv.org/abs/1306.0895)
- Throughput: Batch indexing 1M images/hour on 16-core CPU with vectorized histogramming and I/O pipelining; use multiprocessing pools. [FAISS docs](https://faiss.ai/index.html)

Algorithm spec (math for distance/embedding; pseudocode)
- Let C = {c_b} be Lab bin centers, b=1..B.
- For an image I, compute h ∈ Δ^(B−1) via tri-linear soft assignment:
  For each pixel x with Lab coordinate ℓ:
    - Compute fractional bin coordinates (iL,iA,iB) and neighbor set N(ℓ) of 8 bins.
    - Add weights w_b(ℓ) ∝ ∏d∈{L,A,B} max(0,1−|Δd|) to h_b for b ∈ N(ℓ); normalize.
  Optionally, convolve with small Gaussian over bin lattice.
  Normalize h to sum 1. [Swain & Ballard, 1991](https://www.cs.washington.edu/education/courses/455/09wi/readings/swainballard91.pdf)
- Proxy embedding: φ(h) = sqrt(h) ∈ R^B (elementwise).
  Proxy distance: d_proxy(h1,h2) = ||φ(h1) − φ(h2)||2. [Hellinger (overview)](https://en.wikipedia.org/wiki/Hellinger_distance)
- Rerank metric: Sinkhorn-regularized OT cost:
    a = hq (query histogram), b = hc (candidate histogram)
    Cost matrix M_ij = ||c_i − c_j||2
    Wε(a,b) = min_{P≥0, P1=a, P^T1=b} 〈P, M〉 + ε KL(P || a b^T)
  Compute via iterative scaling (Sinkhorn) for speed; use only non-zero bins to form sparse supports. [Cuturi, 2013](https://arxiv.org/abs/1306.0895)
- Optional χ² explicit feature map ψ(h) for ANN:
    kχ2(x,y) = ∑i 2 x_i y_i / (x_i + y_i)
    ψ approximates kχ2 with finite-dimensional features; index ψ(h). [Vedaldi & Zisserman, 2011](https://www.robots.ox.ac.uk/~vgg/publications/2011/Vedaldi11/vedaldi11.pdf)

Pseudocode
- index(image):
  1) read image, to thumbnail, to Lab
  2) compute histogram h with tri-linear soft assignment
  3) store sparse h, dense φ(h), metadata
  4) add φ(h) to Faiss index
- search(query):
  1) parse hex colors and weights; convert to Lab
  2) soften per color with Gaussian; deposit into hq; normalize
  3) compute φ(hq); ANN top-K
  4) for each candidate c: build sparse supports (non-zero bins in hq & hc), compute M, Wε(a,b)
  5) sort by Wε and return

Implementation plan (checklist + code snippets)
Milestones (6 weeks)
- W1: Data I/O, Lab conversion, tri-linear histogram code; small CLI to index a folder. [scikit-image docs](https://scikit-image.org/docs/0.25.x/api/skimage.color.html)
- W2: ANN proxy embedding (sqrt hist), Faiss HNSW index; metadata store. [FAISS docs](https://faiss.ai/index.html)
- W3: Query parsing, softening kernel, end-to-end basic search; simple Flask/FastAPI. [TinEye Multicolr](https://labs.tineye.com/multicolr/)
- W4: Sinkhorn reranker via POT; top-K pipeline; metrics harness. [POT docs](https://pythonot.github.io/)
- W5: Ablations, parameter sweeps, performance tuning (HNSW M/efSearch, Sinkhorn ε, K). [Malkov & Yashunin, 2018](https://arxiv.org/abs/1603.09320)
- W6: API polish, caching, batch updates, dashboards; prepare benchmark report.

Python snippets

1) Image → Lab histogram (tri-linear soft assignment)
Note: Dependencies: numpy, opencv-python, scikit-image.

```python
import cv2
import numpy as np
from skimage.color import rgb2lab

# Lab binning configuration
L_bins, A_bins, B_bins = 8, 12, 12
L_min, L_max = 0.0, 100.0
A_min, A_max = -86.0, 98.0
B_min, B_max = -108.0, 95.0
B_total = L_bins * A_bins * B_bins

def bin_edges(minv, maxv, n):
    return np.linspace(minv, maxv, n+1)

L_edges = bin_edges(L_min, L_max, L_bins)
A_edges = bin_edges(A_min, A_max, A_bins)
B_edges = bin_edges(B_min, B_max, B_bins)

def to_lab_thumbnail(path, max_side=256):
    bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise ValueError(f"Cannot read {path}")
    # remove alpha channel if present
    if bgr.shape[-1] == 4:
        alpha = bgr[...,3]/255.0
        bgr = bgr[...,:3]
        # optionally discard fully transparent
        mask = alpha > 0.0
    else:
        mask = None
    h, w = bgr.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        if mask is not None:
            mask = cv2.resize(mask.astype(np.uint8), (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    lab = rgb2lab(rgb)  # skimage expects float in [0,1]
    if mask is not None:
        lab = lab[mask]
    return lab.reshape(-1, 3)

def trilin_hist(lab_pixels):
    # clamp to ranges
    L = np.clip(lab_pixels[:,0], L_min, L_max)
    A = np.clip(lab_pixels[:,1], A_min, A_max)
    B = np.clip(lab_pixels[:,2], B_min, B_max)

    # continuous bin indices
    def frac_index(v, edges):
        # find bin lower index i s.t. edges[i] <= v < edges[i+1]
        # return i and fractional position t in [0,1]
        idx = np.clip(np.searchsorted(edges, v, side='right') - 1, 0, len(edges)-2)
        t = (v - edges[idx]) / (edges[idx+1] - edges[idx])
        return idx, t

    iL, tL = frac_index(L, L_edges)
    iA, tA = frac_index(A, A_edges)
    iB, tB = frac_index(B, B_edges)

    # 8 neighbors weights (tri-linear)
    # neighbor offsets and weight factors
    neigh = [
        (0,0,0, (1-tL)*(1-tA)*(1-tB)),
        (1,0,0, tL    *(1-tA)*(1-tB)),
        (0,1,0, (1-tL)*tA    *(1-tB)),
        (0,0,1, (1-tL)*(1-tA)*tB    ),
        (1,1,0, tL    *tA    *(1-tB)),
        (1,0,1, tL    *(1-tA)*tB    ),
        (0,1,1, (1-tL)*tA    *tB    ),
        (1,1,1, tL    *tA    *tB    ),
    ]

    hist = np.zeros(B_total, dtype=np.float32)
    # flat index helper
    def flat_index(l,a,b):
        return (l * A_bins + a) * B_bins + b

    for dL, dA, dB, w in neigh:
        l_idx = np.clip(iL + dL, 0, L_bins-1)
        a_idx = np.clip(iA + dA, 0, A_bins-1)
        b_idx = np.clip(iB + dB, 0, B_bins-1)
        flat = flat_index(l_idx, a_idx, b_idx)
        # accumulate per-pixel weights
        np.add.at(hist, flat, w)

    s = hist.sum()
    if s > 0:
        hist /= s
    return hist  # sparse-friendly via np.nonzero later

def image_to_hist(path):
    lab = to_lab_thumbnail(path)
    return trilin_hist(lab)
```

This uses skimage’s rgb2lab and a tri-linear soft-assigner over a 3D Lab grid, normalizing to a probability histogram. [scikit-image docs](https://scikit-image.org/docs/0.25.x/api/skimage.color.html)

2) Query colors → histogram (with Gaussian softening)
```python
import numpy as np

def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip('#')
    return tuple(int(hex_str[i:i+2], 16) for i in (0,2,4))

def query_to_hist(hex_colors, weights, sigma_LAB=(5.0, 6.0, 6.0)):
    colors_lab = []
    for hx in hex_colors:
        r,g,b = hex_to_rgb(hx)
        rgb = np.array([[[r,g,b]]], dtype=np.float32)/255.0
        lab = rgb2lab(rgb)[0,0,:]
        colors_lab.append(lab)
    colors_lab = np.array(colors_lab, dtype=np.float32)
    w = np.array(weights, dtype=np.float32)
    w = w / w.sum()

    # deposit Gaussians on the bin lattice
    # we approximate by tri-linear to neighbors inside 1 bin radius weighted by Gaussian in Lab
    hist = np.zeros(B_total, dtype=np.float32)
    # precompute bin centers
    L_centers = 0.5*(L_edges[:-1] + L_edges[1:])
    A_centers = 0.5*(A_edges[:-1] + A_edges[1:])
    B_centers = 0.5*(B_edges[:-1] + B_edges[1:])
    # neighbor offsets in {-1,0,1}
    nbr = [-1,0,1]
    sL, sA, sB = sigma_LAB

    def flat_index(l,a,b):
        return (l * A_bins + a) * B_bins + b

    for (L0,A0,B0), ww in zip(colors_lab, w):
        # get base index
        def base_idx(v, edges):
            return np.clip(np.searchsorted(edges, v, side='right') - 1, 0, len(edges)-2)
        iL = base_idx(L0, L_edges)
        iA = base_idx(A0, A_edges)
        iB = base_idx(B0, B_edges)

        for dL in nbr:
            for dA in nbr:
                for dB in nbr:
                    l = iL + dL
                    a = iA + dA
                    b = iB + dB
                    if not (0 <= l < L_bins and 0 <= a < A_bins and 0 <= b < B_bins):
                        continue
                    # Gaussian weight in Lab
                    dLL = (L0 - L_centers[l]) / sL
                    dAA = (A0 - A_centers[a]) / sA
                    dBB = (B0 - B_centers[b]) / sB
                    g = np.exp(-0.5*(dLL*dLL + dAA*dAA + dBB*dBB))
                    hist[flat_index(l,a,b)] += ww * g

    s = hist.sum()
    if s > 0:
        hist /= s
    return hist
```

Gaussian softening in Lab avoids hue wraparound issues by not working in angular coordinates at all. Adjust σ to control fuzziness slider. [CIEDE2000 notes](https://www.ece.rochester.edu/~gsharma/ciede2000/)

3) Fast ANN search with Faiss (HNSW)
```python
import faiss
import numpy as np

class ColorANN:
    def __init__(self, dim, M=32, efConstruction=200, efSearch=64):
        self.dim = dim
        self.index = faiss.IndexHNSWFlat(dim, M)
        self.index.hnsw.efConstruction = efConstruction
        self.index.hnsw.efSearch = efSearch
        self.ids = []

    def add(self, vecs, ids=None):
        # vecs: (n, dim) float32
        if ids is not None:
            ids = np.array(ids, dtype=np.int64)
            self.index.add_with_ids(vecs, ids)
            self.ids.extend(ids.tolist())
        else:
            self.index.add(vecs)

    def search(self, qvecs, k=200):
        D, I = self.index.search(qvecs, k)
        return D, I

def hist_to_proxy_vec(h):
    return np.sqrt(h.astype(np.float32) + 1e-12)
```

For billion-scale, replace with IVF-PQ/OPQ; Faiss supports both CPU and GPU; we target CPU per constraints. [FAISS docs](https://faiss.ai/index.html) [Johnson et al., 2017](https://arxiv.org/abs/1702.08734)

4) Rerank with Sinkhorn (POT)
```python
import numpy as np
import ot  # pip install POT

# Precompute bin centers and a function to get sparse supports
L_centers = 0.5*(L_edges[:-1] + L_edges[1:])
A_centers = 0.5*(A_edges[:-1] + A_edges[1:])
B_centers = 0.5*(B_edges[:-1] + B_edges[1:])

def bin_center(bflat):
    l = bflat // (A_bins * B_bins)
    a = (bflat // B_bins) % A_bins
    b = bflat % B_bins
    return np.array([L_centers[l], A_centers[a], B_centers[b]], dtype=np.float32)

def sinkhorn_distance(hq, hc, epsilon=1.0):
    # sparse supports
    q_idx = np.nonzero(hq > 0)[0]
    c_idx = np.nonzero(hc > 0)[0]
    a = hq[q_idx].astype(np.float64)
    b = hc[c_idx].astype(np.float64)

    # normalize (just in case)
    a = a / a.sum()
    b = b / b.sum()

    Q = np.stack([bin_center(i) for i in q_idx], axis=0)
    C = np.stack([bin_center(i) for i in c_idx], axis=0)
    # cost matrix in Lab
    # Euclidean distances
    M = np.sqrt(((Q[:,None,:] - C[None,:,:])**2).sum(-1)).astype(np.float64)

    # compute Sinkhorn divergence/cost
    # ot.sinkhorn2 returns regularized OT cost
    reg = epsilon
    cost, log = ot.sinkhorn2(a, b, M, reg=reg, log=True, method='sinkhorn')
    return float(cost)
```

POT provides robust and efficient implementations of entropy-regularized optimal transport (Sinkhorn). You may tune ε and set a max number of iterations to bound latency. [POT docs](https://pythonot.github.io/) [Cuturi, 2013](https://arxiv.org/abs/1306.0895)

5) End-to-end search (wiring)
```python
def search_colors(hex_colors, weights, ann, hist_store, k=200, epsilon=1.0):
    hq = query_to_hist(hex_colors, weights, sigma_LAB=(5,6,6))
    qvec = hist_to_proxy_vec(hq)[None, :]
    D, I = ann.search(qvec.astype(np.float32), k=k)
    candidates = I[0]
    # rerank
    scores = []
    for idx in candidates:
        hc = hist_store[idx]  # retrieve original (dense or sparse) histogram by id
        cost = sinkhorn_distance(hq, hc, epsilon=epsilon)
        scores.append((idx, cost))
    scores.sort(key=lambda x: x[1])
    return scores  # list of (image_id, distance)
```

Indexing & scaling details
- Vector dimension: B=768 or 1152. With sqrt(h) dense vectors, prefer IVF-PQ for memory reduction (e.g., m=16 sub-vectors, 8 bits each ⇒ 16 bytes/vector); HNSW can be layered on PQ for accuracy vs memory tradeoffs. [FAISS docs](https://faiss.ai/index.html)
- Memory math example: N=10M, 32B/vector ⇒ ~320MB + IVF/HNSW metadata (usually a few GB). [Johnson et al., 2017](https://arxiv.org/abs/1702.08734)
- Dynamic updates: HNSW easy for appends; IVF-PQ supports add; periodic re-training of coarse quantizer advisable after large growth. [FAISS docs](https://faiss.ai/index.html)

Rationale against HSV quantization and KMeans pitfalls
- HSV’s circular hue breaks simple Euclidean bin distances unless special angular handling is added; Lab avoids this and provides more uniform distances. [CIEDE2000 notes](https://www.ece.rochester.edu/~gsharma/ciede2000/)
- KMeans+Euclidean on palette centers ignores cross-bin mass transport and combines color and weight distances ad hoc; EMD/Wasserstein directly optimizes transport work with a perceptual ground metric. [Rubner et al., 2000](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/rubner-jcviu-00.pdf) [Synthesio Medium](https://medium.com/synthesio-engineering/a-journey-towards-creating-a-color-search-engine-194f1c388680)

Evaluation plan (protocols, metrics, datasets)
- Datasets:
  - COCO 2017 val subset (~5K images); ensures diverse scenes and colors.
  - Unsplash-lite curated set (~20K images). [TinEye Multicolr](https://labs.tineye.com/multicolr/)
- Ground truth:
  - Auto-extract per-image top palettes (e.g., top-5 bins by weight) for candidate positives.
  - Human label a set of query palettes (mono, dual complementary, triadic, skewed weights) against top-50 results from multiple systems (ours vs baselines).
- Metrics:
  - Precision@K (K: 10, 20, 50), nDCG@K, mAP.
  - Response time P50/P95 measured at server.
  - Ablations: B (512/768/1152), softening σ, proxy metric (Hellinger vs χ²), rerank K (100/200/400), Sinkhorn ε. [Rubner et al., 2000](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/rubner-jcviu-00.pdf) [Vedaldi & Zisserman, 2011](https://www.robots.ox.ac.uk/~vgg/publications/2011/Vedaldi11/vedaldi11.pdf)
- Baselines:
  - HSV 64-bin histogram + Gaussian blur + intersection (known to suffer hue wraparound; include to document improvement).
  - KMeans palette (k=5) + weighted Euclidean to query colors (Ad hoc; included for completeness). [Synthesio Medium](https://medium.com/synthesio-engineering/a-journey-towards-creating-a-color-search-engine-194f1c388680)
- Reporting: Curves (nDCG@K), latency histograms, qualitative panels showing retrieved image palettes vs queries.

API & UX
- REST endpoints (FastAPI):
  - POST /index: body = {image_id, url} → processes, indexes, persists metadata.
  - GET /search?colors=ea6a81,f6d727&weights=0.49,0.51&k=50&fuzz=1.0:
    - fuzz scales σ for query softening; k= returned results; also return top detected swatches per result and distances.
  - GET /image/{id}: returns metadata, stored palette, top bins.
- Determinism: accept seed param; deterministic histogramming and ANN search options (fix efSearch). [FAISS docs](https://faiss.ai/index.html)
- UX: mimic TinEye Multicolr: draggable color chips, weight sliders, preview of fuzzy radius; palette overlay chips on each result tile. [TinEye Multicolr](https://labs.tineye.com/multicolr/)

Hard requirements addressed
- Combine color and weight distance: Wasserstein/Sinkhorn on normalized histograms integrates both; cost is mass × ground distance. No ad hoc mixing needed. [Rubner et al., 2000](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/rubner-jcviu-00.pdf) [Cuturi, 2013](https://arxiv.org/abs/1306.0895)
- Hue wraparound: avoided by using Lab; if using LCh for any component, apply circular Gaussian on hue or minimal angular difference. [CIEDE2000 notes](https://www.ece.rochester.edu/~gsharma/ciede2000/)
- Vector-DB-compatible embedding: Hellinger transform sqrt(h) underpins Euclidean ANN; optional χ² explicit map for additive kernels. Two-stage with Sinkhorn rerank. [Hellinger (overview)](https://en.wikipedia.org/wiki/Hellinger_distance) [Vedaldi & Zisserman, 2011](https://www.robots.ox.ac.uk/~vgg/publications/2011/Vedaldi11/vedaldi11.pdf)
- Tuning bin size/softening and K: Use ablations; start B=768, σ=(5,6,6) Lab, K=200; adjust based on nDCG vs latency curves. [Rubner et al., 2000](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/rubner-jcviu-00.pdf)

Architecture diagram and data flow (ASCII)
- Client → API → Ingest Worker:
  - Fetch image → Thumbnail → rgb2lab → Soft histogram (tri-linear) → Normalize h → φ(h)=sqrt(h).
  - Persist: sparse h + metadata (SQLite/DuckDB), ANN vector store (Faiss HNSW/IVF-PQ).
- Search:
  - Parse query → Softened query histogram hq → φ(hq) → ANN top-K → Rerank (Sinkhorn) → Results + per-image top swatches.

[FAISS docs](https://faiss.ai/index.html) [scikit-image docs](https://scikit-image.org/docs/0.25.x/api/skimage.color.html)

Risks & mitigations
- Memory footprint: Dense B-dim vectors are heavy at scale; mitigate with IVF-PQ compression or lower B (e.g., 768) while preserving quality with soft assignment. [FAISS docs](https://faiss.ai/index.html)
- Rerank latency spikes: Limit K adaptively (e.g., 100–400), cap support sizes (top-τ bins), and tune ε for faster Sinkhorn convergence; early abandon rerank beyond a score threshold. [Cuturi, 2013](https://arxiv.org/abs/1306.0895)
- Color management: Non-sRGB sources and HDR may break Lab assumptions; consider normalizing to sRGB or migrating to JzAzBz later for HDR. [Safdar et al., 2017](https://opg.optica.org/abstract.cfm?uri=oe-25-13-15131)
- Dataset biases: Portrait-heavy corpora may skew palettes; optional skin-tone downweighting or category-aware filters. [Hafner et al., 1995](https://ieeexplore.ieee.org/document/391417/)

Benchmarks on small public dataset (protocol and templates)
- Provide a script that:
  - Indexes COCO val images.
  - Evaluates a suite of query palettes (mono/dual/triadic) with ground-truth lists curated via human labeling tasks.
  - Outputs P@K, nDCG@K, mAP, latency P50/P95, with ablations for B, σ, K, ε.
- Report template (to be filled after running):
  - Proxy: Hellinger vs χ²; Rerank: Sinkhorn ε∈{0.5,1.0,2.0}; K∈{100,200,400}; B∈{512,768,1152}.
  - Table of metrics and runtime; Qualitative panels for top-10 per query.

Nice-to-haves
- Sliced Wasserstein embedding: compute random projections in Lab, 1D CDF distances (quantile matching) aggregated as a vector; index with ANN to approximate OT in the first stage. [Kolouri et al., 2016](https://openaccess.thecvf.com/content_cvpr_2016/papers/Kolouri_Sliced_Wasserstein_Kernels_CVPR_2016_paper.pdf)
- Simple UI: visualize per-bin contributions as stacked bars; hover reveals bin colors; cache hot queries and nearest-color expansions.

Next steps (immediate tasks)
- Implement and test the histogram extractor and query softener on a small image set; validate visually by plotting palette bars and comparing to TinEye Multicolr behavior. [TinEye Multicolr](https://labs.tineye.com/multicolr/)
- Stand up Faiss HNSW index and run smoke tests on 50K images; measure ANN recall and latency. [FAISS docs](https://faiss.ai/index.html)
- Integrate POT Sinkhorn reranker; tune ε and K for best nDCG-latency trade-off; add fallback to FastEMD if needed. [POT docs](https://pythonot.github.io/) [Pele & Werman, 2009](https://www.cs.huji.ac.il/~werman/Papers/ICCV2009.pdf)
- Build minimal FastAPI endpoints and a basic web UI for palette input and result visualization. [Synthesio Medium](https://medium.com/synthesio-engineering/a-journey-towards-creating-a-color-search-engine-194f1c388680)

Appendix: optional χ² explicit feature map (scikit-learn)
If you find χ² better than Hellinger as a proxy:

```python
from sklearn.kernel_approximation import AdditiveChi2Sampler
# Fit on a sample of histograms to set range
chi2_map = AdditiveChi2Sampler(sample_steps=2)
chi2_map.fit(sample_histograms)  # shape (n_samples, B)
# Transform both database and queries:
vecs = chi2_map.transform(H)  # H is (n, B)
qvec = chi2_map.transform(hq[None, :])
# Then index vecs in Faiss and search qvec
```

This provides an explicit finite-dimensional embedding approximating the additive χ² kernel, making it indexable via Euclidean ANN. [Vedaldi & Zisserman, 2011](https://www.robots.ox.ac.uk/~vgg/publications/2011/Vedaldi11/vedaldi11.pdf)

References (selected)
- TinEye Multicolr demo and service overview. [TinEye Multicolr](https://labs.tineye.com/multicolr/)
- Swain & Ballard, “Color Indexing,” Histogram Intersection. [Swain & Ballard, 1991](https://www.cs.washington.edu/education/courses/455/09wi/readings/swainballard91.pdf)
- Rubner et al., “The Earth Mover’s Distance as a Metric for Image Retrieval.” [Rubner et al., 2000](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/rubner-jcviu-00.pdf)
- Cuturi, “Sinkhorn Distances: Lightspeed Computation of Optimal Transport.” [Cuturi, 2013](https://arxiv.org/abs/1306.0895)
- Pele & Werman, “Fast and Robust Earth Mover’s Distances.” [Pele & Werman, 2009](https://www.cs.huji.ac.il/~werman/Papers/ICCV2009.pdf)
- Vedaldi & Zisserman, “Efficient Additive Kernels via Explicit Feature Maps.” [Vedaldi & Zisserman, 2011](https://www.robots.ox.ac.uk/~vgg/publications/2011/Vedaldi11/vedaldi11.pdf)
- FAISS documentation and papers. [FAISS docs](https://faiss.ai/index.html) [Johnson et al., 2017](https://arxiv.org/abs/1702.08734)
- HNSW algorithm. [Malkov & Yashunin, 2018](https://arxiv.org/abs/1603.09320)
- scikit-image rgb2lab. [skimage.color](https://scikit-image.org/docs/0.25.x/api/skimage.color.html)
- JzAzBz (for HDR/wide-gamut optional upgrade). [Safdar et al., 2017](https://opg.optica.org/abstract.cfm?uri=oe-25-13-15131)
- CIEDE2000 implementation notes. [Sharma et al.](https://www.ece.rochester.edu/~gsharma/ciede2000/)
- Sliced Wasserstein kernels/embeddings. [Kolouri et al., 2016](https://openaccess.thecvf.com/content_cvpr_2016/papers/Kolouri_Sliced_Wasserstein_Kernels_CVPR_2016_paper.pdf)
- Prior engineering write-up. [Synthesio Medium](https://medium.com/synthesio-engineering/a-journey-towards-creating-a-color-search-engine-194f1c388680)

