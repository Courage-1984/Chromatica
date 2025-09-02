SYSTEM / ROLE
You are a senior ML/IR engineer, technical architect, and editorial lead. Produce a single, authoritative, actionable consolidation of the five attached plans for building a "Color Search Engine" (TinEye Multicolr style). Be precise, pragmatic, and conservative with assumptions. Prioritize reproducibility and traceability.

INPUT FILES (attached)
- Color-Search-Engine_Plan_by_ChatGPT.md
- Color-Search-Engine_Plan_by_Google-Gemini.md
- Color-Search-Engine_Plan_by_Qwen.md
- Color-Search-Engine_Plan_by_Genspark.md
- Color-Search-Engine_Plan_by_Grok.md

TASK (what I want)
1. Read all attached files completely. Produce a **single consolidated plan** named:
   `Color-Search-Engine_Consolidated_Plan.md`.
2. Keep the final structure exactly as requested (below). Where plans disagree, resolve conflicts by (a) stating the trade-offs, (b) recommending a single option with justification, and (c) if unresolved, mark as `UNRESOLVED` and give a concrete experiment to resolve it.

MANDATORY OUTPUT STRUCTURE
(Produce these exact sections, in order.)

A. Executive summary — 1–2 paragraphs (high-level decision summary: chosen color space, representation, index strategy, evaluation summary).

B. Deep research notes — short citations / bullet list of the most relevant prior work or libraries used as justification (no more than 6 items).

C. System design — architecture diagram summary (textual), dataflow, storage choices, API endpoints.

D. Algorithm specification — math and pseudocode for:
   - color quantization & binning (explicit parameters)
   - histogram/soft-assignment pipeline
   - candidate embedding for ANN and reranker
   - final high-fidelity distance (e.g., Sinkhorn-EMD / Quadratic-form) with exact formulas

E. Implementation plan — checklist + minimal working Python reference snippets (OpenCV / scikit-image / numpy), plus FAISS example if used. Code snippets must be runnable (no placeholders) and short.

F. Evaluation plan — datasets, metrics (Precision@K, nDCG, mAP), ablation plan, sample testcases and sanity checks.

G. Performance & scaling — expected memory formulae, index sizes per N images, latency targets, and suggested K for rerank.

H. UX & API — REST endpoints examples with parameters and expected JSON return format.

I. Risks & mitigations — at least 6 items (technical and product).

J. Next steps — immediate tasks (first 6 actions), and a suggested 6–8 week milestone timeline (by-week bullets).

K. Source map & diff appendix:
   - For each major decision in the consolidated plan, list the **source files that recommended it** (e.g., ChatGPT: section X; Qwen: section Y). Use this inline format: `[decision]: recommended by {ChatGPT, Qwen, ...}`.
   - Add a short “Unique suggestions” table: lines that appear in only one of the five plans and whether we adopted them (Yes/No) and why.

SPECIAL INSTRUCTIONS (important)
- Preserve the tech constraints: use **Python**, OpenCV, scikit-image, NumPy. You may use FAISS, scikit-learn, DuckDB/SQLite only if well justified.
- Directly address these known pitfalls (one short paragraph each): poor NN results from the article, kmeans+euclidean problems with weights, HSV hue wraparound / circularity issues.
- When combining color + weight distance, produce a principled formula (not ad-hoc) and show pseudocode.
- Provide a two-stage search design: fast ANN-friendly embedding → exact rerank (show how to map the chosen exact metric to an ANN-friendly proxy).
- If you use any third-party algorithm or library for a metric (POT Sinkhorn, etc.), briefly note licensing and an alternative if licensing is an issue.
- For traceability, **annotate each decision** with provenance: which of the 5 files strongly recommended it (exact file names). If a decision is new or inferred (i.e., not in any file), mark it `INFERENCE`.
- Limit verbatim quoting from the files to <25 words per quote.
- Output must be in **Markdown**. Include a short machine-readable JSON manifest at the end with keys:
  `{"chosen_color_space": "...", "quant_bins": "...", "embedding_dim": "...", "ann_engine": "...", "rerank_metric": "...", "topK_rerank": 200, "estimated_index_size_per_M": "..."}`

FAILSAFES
- If any file is missing or cannot be parsed, state which file and proceed with the others.
- If the model's context is insufficient to read all files at once, process them in the order above and state clearly which parts came from which file chunk.

STYLE & TONE
- Clear, technical, and concise. Use numbered lists and code fences. Avoid marketing language.
- Executive summary must be copy/paste friendly for a product manager.

END
