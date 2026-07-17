# Document Retrieval — Study Notes

> A companion guide to `Document_Retrieval.ipynb`. It explains **what each part does**, **the logic behind the code**, and **why it matters**, so you can build intuition fast. Read top-to-bottom once; the notebook is designed so later sections reuse a few clean objects from earlier ones.

---

## The big picture

This notebook builds a **production-grade Retrieval-Augmented Generation (RAG)** document Q&A system end-to-end — not a toy. RAG grounds a language model in an external corpus: instead of relying only on what the model memorized during training, it **retrieves** relevant passages at query time and conditions its answer on them. This reduces hallucination and lets you update knowledge without retraining.

**The pipeline (two phases):**

```
INGESTION (offline, done once):
   raw files (PDF/DOCX/HTML/scans) → load → clean → CHUNK → EMBED → build VECTOR INDEX (FAISS) → persist

RETRIEVAL + GENERATION (online, per query):
   user question → EMBED query → ANN SEARCH (top-k) → (rerank) → LLM with context → grounded answer + citations
```

**Four recurring objects** the notebook builds early and reuses everywhere:
- `CORPUS` — one `Document` per paper (text + metadata).
- `EVALSET` — answerable questions, each with **gold evidence** paragraphs (this is what makes retrieval *measurable*).
- `EMB` — the embedder (an `Embedder` interface, SBERT by default).
- `CHUNKS` / `CHUNK_VECS` — the frozen winning chunk set and its vectors.

---

## Section 1 · Setup & configuration

**System binaries** (`tesseract`, `poppler`) are installed for the OCR demo only — safe to skip if you don't need OCR. **Python packages** are pinned to a coherent set (LangChain family, sentence-transformers, faiss-cpu, datasets, rank-bm25, etc.).

**Device detection:** picks `cuda` if a GPU is available, else `cpu`. Only affects embedding *speed*, not correctness. Seeds (`random`, `numpy`) are fixed to 42 for reproducibility.

### The `Config` dataclass — one control panel

All tunable knobs live in a single `@dataclass Config` object (`CFG`). Key fields:

| Field | Meaning |
|---|---|
| `dataset_name` / `split` | `allenai/qasper`, validation split (~281 papers) |
| `max_papers`, `max_eval_questions` | caps to keep it fast on a free tier |
| `sbert_model` | `all-MiniLM-L6-v2` — 384-dim, fast, solid local embedder |
| `normalize: True` | **L2-normalize vectors → inner product == cosine similarity** (key design choice) |
| `chunk_size` / `chunk_overlap` | default chunking (600/120 chars) |
| `top_k` / `eval_k` | chunks fed to LLM (5) / k used to score retrieval (10) |
| `chat_model` / `local_llm` | OpenAI model (if key) / local generative fallback |

> **Design principle:** knobs live in one object, not scattered through cells. Change behavior *there*.

**Optional OpenAI key:** the whole notebook runs **100% free & local** with no key. If `OPENAI_API_KEY` is set, the OpenAI embedding + LLM paths "light up" automatically (`HAS_OPENAI` flag). Nothing else changes.

---

## Section 2 · Document processing — PDF · DOCX · HTML · OCR

**Goal:** turn messy real-world file formats into clean text before anything can be searched.

**Logic:** synthesize one sample file of each format (so the notebook is self-contained), then read them back with the right loader.

### 2.1 LangChain loaders
`PyPDFLoader`, `Docx2txtLoader`, `BSHTMLLoader` each return `Document(page_content=..., metadata=...)`. Using battle-tested loaders (vs. hand-parsing) gives **uniform output** and handles edge cases (encrypted PDFs, weird encodings, nested HTML).

### 2.2 OCR basics
Many "PDFs" are just **photos of pages** — no text layer, so loaders return nothing. Fix = **Optical Character Recognition**:
```
PDF page → image (pdf2image/poppler) → Tesseract (pytesseract) → text
```
`ocr_pdf(path)` rasterizes each page at 200 DPI and OCRs it. Used **as a fallback only** when a normal loader yields little/no text.

### 2.3 Unified loader — `load_document(path)`
The single function the rest of an ingestion pipeline needs. Dispatches on file extension:
- `.pdf` → `PyPDFLoader`; **if extracted text < 32 chars → assume scanned → fall back to `ocr_pdf()`**.
- `.docx` / `.html` / images / `.txt` → the right loader.
- Stamps `source` + `format` metadata on every returned `Document`.

**Logic takeaway:** wrap format dispatch *once*, with an automatic OCR fallback, so callers never write `if extension == ...` chains.

---

## Section 3 · The dataset — `allenai/qasper`

**Why QASPER:** it has genuinely **long documents** (full NLP research papers — multi-section, multi-page, exactly where chunking matters), **real questions** (written by practitioners who saw only title+abstract), and crucially **gold evidence paragraphs** annotated per answer.

> That gold evidence turns *"did retrieval work?"* from a vibe into a **number**.

### Parsing (isolated in this one section)
- `paper_to_text(ex)` — flattens one paper (title + abstract + each section name + paragraphs) into a single clean string joined by blank lines.
- `paper_qas(ex)` — extracts **answerable** `(question, evidence, answer)` triples. A question is "answerable" if it has evidence, a free-form answer, extractive spans, or a yes/no. Evidence starting with `"FLOAT SELECTED"` (figure/table refs) is dropped; evidence is de-duplicated while keeping order.

### Building the two clean objects
```python
CORPUS  : list[Document]  # one per paper (skip near-empty parses < 500 chars)
EVALSET : list[dict]      # {paper_id, question, evidence:[...], answer, answerable}
```
`EVALSET` is shuffled (seed 42) and capped for speed. Everything downstream consumes only these two objects.

### 3.1 A look at the data
Plots document length (words/paper) and evidence paragraphs/question. **The point:** the median paper is *far* larger than an embedding model's context window (a few hundred words), so it **must** be chunked.

---

## Section 4 · Embeddings

An embedding maps text → a vector so that **semantic similarity becomes geometric proximity**. "How do I cite a source?" and "What is the citation format?" land close together even though they share few words.

| | Sentence-Transformers (SBERT) | OpenAI |
|---|---|---|
| runs | locally, free, offline | hosted API (paid) |
| default | `all-MiniLM-L6-v2` (384-d) | `text-embedding-3-small` (1536-d) |
| good for | most workloads, privacy, control | top quality, zero infra |

### The `Embedder` interface
Both providers hide behind one interface with `.embed(texts) -> float32 (n, dim)` and `.embed_query(text)`. Downstream code never cares which one is active.

- `SbertEmbedder` — wraps `SentenceTransformer`, encodes with batching and `normalize_embeddings=True`.
- `OpenAIEmbedder` — drop-in, batches at the API limit (256), manually L2-normalizes to **match** SBERT.

> **Crucial detail — normalization:** every vector is L2-normalized to unit length. With unit vectors, **inner product == cosine similarity**. This lets *every* FAISS index later use a single metric (`METRIC_INNER_PRODUCT`), keeping the whole notebook consistent.

`EMB = SbertEmbedder()` is the default embedder used throughout.

### 4.1 Sanity check
Embeds 4 probe sentences and prints the pairwise cosine matrix (`V @ V.T`, valid because vectors are normalized). Expectation: the two **paraphrases** score highest off-diagonal; the **unrelated** sentence ("the cat slept…") scores lowest with everything. Confirms the geometry is meaningful.

---

## Section 5 · Chunking strategies

A **chunk** is the atomic unit that gets embedded and retrieved. Chunking is the single most under-rated lever in RAG:
- **too small** → each chunk lacks context; the answer is scattered, no single chunk is convincing.
- **too large** → one vector must summarize many ideas → "blurry" embedding, irrelevant text dilutes the match, and you waste the LLM's context window.

Five interchangeable `text -> list[str]` strategies:

| strategy | splits on | keeps meaning? | cost |
|---|---|---|---|
| **fixed-size** | character count | no (cuts mid-word) | trivial |
| **token-based** | token count (tiktoken) | respects model budget | trivial |
| **recursive** | paragraphs→lines→sentences→words | mostly (natural boundaries) | trivial |
| **sliding window** | sentences, with overlap | yes, with redundancy for recall | cheap |
| **semantic** | embedding-similarity dips | yes (topic shifts) | needs embeddings (slow) |

**Code logic:**
- `fixed_splitter` — `CharacterTextSplitter` with empty separator → hard character windows.
- `token_splitter` — `TokenTextSplitter`, counts real tokens.
- `recursive_splitter` — `RecursiveCharacterTextSplitter` with `["\n\n","\n",". "," ",""]`; tries the biggest natural boundary first, falls back down. **The sensible default.**
- `sliding_splitter(window_sents, stride_sents)` — splits into sentences, emits overlapping windows (e.g. 5 sentences, step 3 → adjacent chunks share sentences → redundancy boosts recall).
- `semantic_splitter` — `SemanticChunker` embeds sentences and cuts where consecutive-sentence similarity **drops** (a topic shift). Chunk lengths vary because they follow meaning, not a fixed budget. Slowest.

### 5.1 See each strategy on one paragraph
Runs all five on the same 1400-char input. You *see* fixed-size slicing mid-word, recursive preferring sentence ends, sliding-window overlapping on purpose, and semantic chunks varying in length.

### 5.2 `chunk_corpus(splitter, strategy)`
Applies any splitter across the whole corpus and stamps each chunk with metadata: `paper_id`, `title`, `strategy`, and a unique `chunk_id` (`paper_id::j`). Drops slivers (< 20 chars). This metadata is what lets us later (a) trace a chunk back to its paper and (b) score retrieval.

---

## Section 6 · Chunking experiments — measure, don't guess

Turns the strategy table into **numbers**. For each config: chunk → embed → build an **exact** index → run all eval questions → check if a **gold evidence** passage appears in the top-k.

### How "relevant" is defined
A retrieved chunk counts as relevant to a question when:
1. it comes from the **right paper** (`paper_id` match), **AND**
2. it shares a substantial fraction of words with any gold evidence paragraph (**overlap coefficient ≥ 0.5**).

The overlap coefficient (`|a ∩ b| / min(|a|,|b|)`) tolerates the fact that chunk boundaries rarely align exactly with evidence boundaries.

### Two metrics
- **Hit@k** — fraction of questions with ≥1 relevant chunk in the top-k → *can it find the answer at all?*
- **MRR** (mean reciprocal rank) — average of `1/rank` of the first relevant chunk → *does it find it near the top?*

### The harness
- `embed_and_index(chunks, embedder)` — embeds chunk texts, builds an exact `IndexFlatIP`. Returns index, vectors, timing.
- `retrieve(...)` — embeds all questions, does one batched `index.search`, maps result ids back to chunks.
- Cell 39 sweeps 7 configs (fixed/recursive at two sizes, token, sliding, semantic), scoring each. Cell 40 plots Hit@k bars and a chunk-size-vs-MRR scatter.

### 6.1 Pick a winner and freeze it
Selects the best Hit@k (ties → MRR), then **freezes** its chunks and vectors as `CHUNKS` / `CHUNK_VECS` / `CHUNK_TEXTS` — reused for §7–§10 with **no re-embedding**. Other configs' vectors are freed to save RAM.

> Rule of thumb from the notebook: recursive at a moderate size usually wins; sliding-window when recall matters most; semantic when documents have sharp topic shifts (but it costs the most to build).

---

## Section 7 · Semantic search — the exact baseline

Before approximating anything, build the **exact** search that every approximate method is measured against. With unit-normalized vectors, `IndexFlatIP` computes inner product = cosine similarity against **every** chunk and returns the true top-k. It is **100% accurate by definition** and fast at this scale. Its only weakness: it scans every vector → cost grows **linearly** with corpus size (that's what §8 fixes).

### `SemanticSearcher`
A small wrapper making the query path one call: **embed query → search index → attach metadata**. `.exact(...)` builds a Flat index; `.search(query, k)` returns `(chunk, score)` pairs. The same class will wrap an approximate index later — **the interface never changes**.

`show_results(...)` pretty-prints hits with cosine score, title, and a text snippet.

---

## Section 8 · Approximate Nearest-Neighbour (ANN) search

Exact search is hopeless at tens of millions of vectors. **ANN** trades a *tiny* accuracy loss for huge speed/memory gains. Every ANN method is a point on a triangle you can't max out all at once:

```
        ACCURACY (recall)
              /\
             /  \
      SPEED /____\ MEMORY
```

- **IVF** — *be lazy about where you look* → faster search, same memory (clustering).
- **PQ** — *store vectors smaller* → huge memory savings, some accuracy lost (compression).
- **IVF+PQ** — *do both* → the billion-scale workhorse.
- **HNSW** — *follow a graph of shortcuts* → excellent speed **and** recall, at higher memory/build cost.

### How recall is measured
The exact `IndexFlatIP` gives the **true** top-k. For an approximate index:
```
recall@k = |approx_top-k ∩ exact_top-k| / k      (averaged over all queries)
```
Recall 1.0 = reproduced exact search perfectly.

**Benchmark harness (`bench`)** reports: `recall@k`, `ms/query` (best of N runs), `QPS`, `build_s`, and `size_MB` (serialized index). `NLIST` (IVF cluster count) is auto-sized from corpus size.

### 8.1 IVF — Inverted File (coarse quantization)
**Idea:** k-means clusters vectors into `nlist` cells (**training**). At query time, find the few cells whose centroids are nearest the query and scan **only those**.
- **Knob `nprobe`** = how many cells to scan. `nprobe=1` fastest/least accurate; `nprobe=nlist` degenerates to exact search.
- **Memory unchanged** (full vectors stored); only *speed* improves.
- Cell 49 builds once, sweeps `nprobe ∈ {1,4,8,16,32,NLIST}` to trace the speed↔recall curve.

### 8.2 PQ — Product Quantization (compress the vectors)
**Idea:** split each D-dim vector into `m` sub-vectors; run k-means *inside each sub-space* to learn a codebook of `2^nbits` (256 when nbits=8) centroids. Store each vector as `m` **bytes** (one centroid id per sub-vector) instead of D floats.
- **Knob `m`** — larger `m` → finer approximation → higher recall but more bytes. `m` must **divide D** (D=384 → legal 4,8,16,32,48).
- Compression vs float32 = `4·D / m` (e.g. m=8 → 192× smaller).
- PQ is **lossy** → recall lower than Flat/IVF. It's the **memory** play, not the speed play.
- Cell 51 sweeps `m` to show the memory↔recall trade.

### 8.3 IVF + PQ — the billion-scale workhorse
The two tricks are **orthogonal**, so combine them:
- IVF narrows search to a few cells → **speed**.
- PQ compresses what's stored in them → **memory**.

`IndexIVFPQ` assigns each vector to a cell, then PQ-encodes the **residual** (vector minus cell centroid) into `m` bytes. Sub-linear scanning *and* a tiny footprint — why this family backs most large production vector stores. Knobs: `nlist`/`nprobe` (from IVF) + `m`/`nbits` (from PQ), tuned independently. Cell 53 fixes `m=16`, sweeps `nprobe`; compare its `size_MB` against IVF-Flat to see the shrinkage.

### 8.4 HNSW — Hierarchical Navigable Small World (graph search)
Completely different: **no clustering, no training** — build a navigable multi-layer graph and walk it. Each vector is a node linked to its `M` nearest neighbours; sparse long-range links on top for big jumps, dense short-range links at the bottom for fine homing. Search enters at the top, greedily hops toward the query, descends layer by layer — reaching the neighbourhood in **logarithmic** hops.
- **Best recall-per-millisecond**, no training.
- **Trade-off:** memory (stores full vectors *plus* all graph edges) and slower build.
- **Knobs:** `M` (edges/node), `efConstruction` (build breadth → graph quality), `efSearch` (query breadth → the recall/speed dial). Cell 55 sweeps `efSearch`.

---

## Section 9 · Indexes head-to-head

Builds **one representative config per family** (Flat, IVF nprobe=16, PQ m=16, IVF+PQ, HNSW efS=64) and benches them uniformly on the same queries/k. Cell 58 plots two decisive pictures: **speed↔recall** (top-left = best) and **memory↔recall** (top-left = best).

### 9.1 Which index should I use?

| Situation | Pick | Why |
|---|---|---|
| Small corpus (≲50k) or you need exact/ground-truth | **Flat** | Zero approximation; brute force fast enough |
| High recall **and** low latency, memory not the bottleneck | **HNSW** | Best recall-per-ms; no training. Costs RAM + build time |
| Large corpus, latency matters, vectors still fit in RAM | **IVF (Flat)** | Big speed-up via cell pruning, recall stays high |
| Corpus too big as float32 (tens of millions+) | **PQ / IVF+PQ** | PQ compresses 100×+; add IVF to also prune |
| Frequent rebuilds / streaming inserts | **IVF / IVF+PQ** | Cheap to add to; HNSW rebuilds are pricier |

**Rules of thumb:** start with Flat (correct answers + a recall ceiling). Outgrow it on *latency* → HNSW (RAM-rich) or IVF (balanced). Outgrow it on *memory* → add PQ. **IVF+PQ** is the default for very large stores. Always re-tune the knob against *your* data — recall is dataset-dependent, so **measure, don't guess**.

### 9.2 Why this matters at scale
On a few-thousand-chunk corpus, Flat is already fast, so the ANN win looks small. The gap explodes as N grows because Flat is **O(N)** per query while IVF/HNSW are sub-linear. Cell 61 builds a **synthetic** ~200k-vector set (clearly labelled, flag-gated) to show ANN indexes serving many× more queries/sec than brute force.

---

## Section 10 · Production touches

Four upgrades that separate a demo from something behind an API.

### 10.1 Persistence
Embedding + index-building are the expensive **offline** steps — do them once, ship two artefacts:
1. the **FAISS index** (`faiss.write_index` → `index.faiss`).
2. a **sidecar** (`chunks.json`) holding each vector's text + metadata, aligned **1:1 with vector ids** (row order == add order).

At serve time: `faiss.read_index` + load the sidecar → answers in **milliseconds, no re-embedding**. Cell 64 reloads and confirms it answers identically.

### 10.2 Metadata filtering
Search *within a scope* (one paper, one tenant, a date window). Two patterns:
- **pre-filter** — restrict candidate ids before searching (e.g. FAISS `IDSelector` on IVF).
- **post-filter** (used here) — **over-fetch**, then drop anything failing the predicate. Index-agnostic, robust on every index type, cheap as long as you fetch a healthy `k`.

`filtered_search(query, where, k, overfetch)` fetches `overfetch` (20) candidates, keeps the first `k` whose metadata satisfies `where(meta)`. Demo restricts the same question to a single `paper_id`.

### 10.3 Hybrid search — dense + BM25 with RRF
Dense vectors capture **meaning** (paraphrases, synonyms) but can miss rare **exact tokens** (error codes, IDs, surnames). Sparse **BM25** keyword scoring is the opposite. **Hybrid** runs both and fuses rankings with **Reciprocal Rank Fusion**:
```
fused_score(doc) = Σ  1 / (k₀ + rank)     across both ranked lists   (k₀ = 60)
```
Anything ranked highly by *either* method floats up — **no score normalization needed**. Cell 68 compares Dense-only vs BM25-only vs Hybrid on the eval set; fusion usually matches or beats either alone.

### 10.4 Reranking with a cross-encoder
- **Bi-encoder** (our SBERT): embeds query and doc *separately* → fast, but never directly compares them.
- **Cross-encoder:** feeds `(query, chunk)` *together* through a transformer → one relevance score. Far more accurate, but too slow to run over the whole corpus.

**Production pattern = retrieve-then-rerank:** cheap bi-encoder fetches a shortlist (e.g. 20), the cross-encoder re-scores just those. Cell 70 measures the precision lift (`cross-encoder/ms-marco-MiniLM-L-6-v2`) on a capped subset — reranking reorders the shortlist for better MRR.

---

## Quick reference — the mental model

| Stage | Question it answers | Key lever |
|---|---|---|
| **Loading** | how do I get clean text from any format? | `load_document` + OCR fallback |
| **Chunking** | what's the atomic retrievable unit? | strategy + size/overlap |
| **Embedding** | how do I make meaning comparable? | model choice + **normalize → IP==cosine** |
| **Indexing** | how do I search fast at scale? | Flat → IVF → PQ → IVF+PQ → HNSW |
| **Retrieval quality** | is it finding the right passage? | Hit@k, MRR, recall@k |
| **Production** | how do I ship it? | persist, filter, hybrid, rerank |

**Three ideas to remember:**
1. **Normalize vectors** so inner product == cosine — one metric for every index.
2. **Measure retrieval with gold evidence** (Hit@k / MRR / recall@k) — don't guess chunking or index knobs.
3. **The ANN triangle** — accuracy, speed, memory: pick the index that matches your bottleneck, and re-tune on *your* data.
