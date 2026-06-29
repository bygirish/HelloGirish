# Section 7 — Vector Stores & ANN Indexing

> **Session:** IIT-KGP Module 2 · Session 4 · Instructor: Pawan Goyal
> **Theme:** *You cannot check every vector for every query. IVF and PQ are orthogonal solutions to two distinct scaling problems. HNSW is the low-latency default when RAM allows.*

---

## 7.1 What Is a Vector Database?

A vector database stores vectors of fixed dimensions (called embeddings) such that we can then query it to find all embeddings that are closest (most similar) to a given query vector using a distance metric.

**Formally:** A vector database stores `(id, vector, metadata)` tuples and answers:
> "Find the K vectors most similar to this query vector, optionally filtered by metadata."

```
Documents ──► Split into chunks ──► Embeddings ──► Store ──► Vector DB
                                                              ↑
Web Pages ─────────────────────────────────────────────────┘

Query ──► Embeddings ──► ANN Search in Vector DB ──► Top-K Context ──► LLM
```

Vector DBs are also used outside RAG: finding similar songs (Spotify), similar images (Google Images), similar products (Amazon) — anywhere "find similar" is the primary operation.

---

## 7.2 Why Efficient Vector Retrieval Is Critical

For RAG to work at scale, three things must be true:

1. **Fast semantic search at query time** — sub-100ms latency for user-facing applications
2. **Efficient indexing to scale to billions of text passages** — even a modest enterprise corpus can have millions of chunks
3. **Exact nearest neighbour (KNN) search is too slow** — you need approximate methods (ANN)

---

## 7.3 KNN: The Naive Approach and Its Problem

The most straightforward approach: compare the query vector against every stored vector, sort by distance, keep the top K.

```
Query vector: q (D dimensions)
Database: N vectors, each D dimensions

Compute distance(q, v_i) for i = 1 to N
Sort results
Return top K
```

**Computational complexity: O(N × D) per query**

Example: 1 million vectors × 768 dimensions × 4 bytes = 3 GB per query scan at 32-bit precision.

| N vectors | D dims | Time per query (CPU) |
|---|---|---|
| 10,000 | 768 | ~1 ms ✅ |
| 1,000,000 | 768 | ~100 ms ⚠️ |
| 100,000,000 | 768 | ~10,000 ms ❌ |

KNN is **exact** (recall = 1.0 by definition) — but completely impractical beyond ~50K vectors.

**FAISS `IndexFlatIP`** implements exact KNN and is the yardstick every ANN index is measured against:

```python
import faiss

index = faiss.IndexFlatIP(embed_dim)   # IndexFlatIP = Inner Product (cosine for normalised)
index.add(vecs)                         # that's it
scores, I = index.search(query_vec, k) # exact search
```

---

## 7.4 Approximate Nearest Neighbour (ANN) — The Solution

ANN algorithms find **approximate** nearest neighbours much faster than exact search, sacrificing a small amount of recall:

```
recall@k = |approx_results ∩ exact_results| / k
```

An ANN with recall@10 = 0.95 means 9.5 out of 10 results are the true nearest neighbours. This trade-off is almost always acceptable for RAG — you're not wrong if you miss 5% of top-10 results.

**The ANN Zoo:**

| Algorithm | Strategy | Best For |
|---|---|---|
| **IVF** | Cluster-then-search (be lazy about where you look) | Speed at large N, low RAM |
| **PQ** | Compress vectors (store vectors smaller) | Memory reduction at large N |
| **IVF+PQ** | Both — cluster then search over compressed vectors | Billion-vector scale |
| **HNSW** | Graph-based shortcut traversal | Best recall-per-ms; costs RAM |
| **Flat** | Exact KNN (no approximation) | Baseline; ≤50K vectors |

The trade-off triangle: **you can't maximise all three** simultaneously.

```
         ACCURACY (Flat)
            /\
           /  \
          /    \
    HNSW /      \ 
        /        \
       /  IVF+PQ  \
      /____________\
   SPEED           MEMORY
   (IVF)           (PQ)
```

---

## 7.5 IVF — Inverted File Index: Cluster-Then-Search

IVF solves the problem: **"too many vectors to check"**

### Phase 1: Indexing (Run Once, Offline)

```
Step 1: TRAIN
Run K-Means on the entire vector dataset
→ identify k representative centroids

Step 2: PARTITION
Assign every vector to its nearest centroid
(geometrically forms Voronoi cells)

Step 3: STORE
Organise vectors into "buckets" (Inverted Lists)
labelled by their centroid ID

Result: k buckets, each containing N/k vectors on average
```

```
Voronoi partition:
              ×             ×
  ·  ·  ·       · · ·         ·  ·
   ·    ·   ×      · ·    ×      ·
  ·  ·         ·    ·  ·      ·  ·
              ×             ×
(× = centroid, · = vector assigned to nearest centroid)
```

### Phase 2: Retrieval (Per Query)

```
Step 1: COARSE SEARCH
Compare query against only the k centroids
→ find the nprobe closest centroids

Step 2: FINE SEARCH
Scan only the vectors inside those nprobe buckets
→ ignore all other buckets

Result: O(k + nprobe × N/k) instead of O(N) — massive speedup
```

### The nprobe Parameter

`nprobe` controls the recall/speed trade-off:
- `nprobe = 1` → search only the 1 closest bucket — fastest, lowest recall
- `nprobe = 10` → search 10 closest buckets — 10× more work, much higher recall
- `nprobe = k` → search all buckets = exact KNN (defeats the purpose)

### IVF Work Savings Calculation

With N = 1,000,000 vectors, `nlist = √N = 1000` centroids, `nprobe = 1`:

```
Work = 1000 centroids + ~1000 vectors in the one bucket ≈ 2,000 comparisons
vs brute force: 1,000,000 comparisons

Speedup: ~500×  ✓

With nprobe = 10:
Work = 1000 centroids + 10 × 1000 = 11,000 comparisons
Still ~90× faster, with markedly higher recall  ✓
```

**Rule of thumb for nlist:** `nlist ≈ √N` works well in practice for most corpus sizes.

### FAISS Implementation

```python
import faiss

nlist = 100    # number of clusters
d = 768        # embedding dimension

# Build IVF index
quantizer = faiss.IndexFlatIP(d)          # centroid index (exact)
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

index.train(vecs)                          # learn centroids (K-Means)
index.add(vecs)                            # assign vectors to buckets

# Search
index.nprobe = 10                          # search 10 nearest clusters
scores, I = index.search(query_vec, k)
```

---

## 7.6 PQ — Product Quantisation: Store Vectors Smaller

PQ solves a different problem from IVF: **"vectors too big to store"**

IVF reduces the *number of vectors to search*. PQ reduces the *size of each vector*.

These are **orthogonal** problems and orthogonal solutions — they combine cleanly into IVF+PQ.

### Phase 1: Indexing (Compression)

```
Step 1: SPLIT (Sub-vectors)
Divide the high-dimensional original vector (e.g., 128D)
into m smaller sub-vectors (e.g., 8 chunks of 16D)

Step 2: TRAIN (Codebooks)
Run K-Means independently on each sub-space
to find K (e.g., 256) representative centroids
These centroids are the "Codebook"

Step 3: ENCODE
Replace each sub-vector with the ID (0-255)
of its nearest centroid (1 byte per sub-vector)

Result: Original 128D × 4 bytes = 512 bytes → 8 bytes (IDs)
```

### The 95%+ Memory Reduction — Worked Example

```
Original vector: 768 dimensions × 4 bytes (float32) = 3,072 bytes

PQ encoding: m = 8 sub-vectors, K = 256 centroids per sub-space
→ Each sub-vector stored as 1 byte (log2(256) = 8 bits)
→ PQ-compressed vector = 8 × 1 byte = 8 bytes

Compression ratio: 3,072 / 8 = 384×  (~99.7% reduction)
```

| Dimension | Original Size | PQ Compressed | Ratio |
|---|---|---|---|
| 768-dim | 3,072 bytes | 8 bytes (m=8) | 384× |
| 768-dim | 3,072 bytes | 16 bytes (m=16) | 192× |
| 3072-dim | 12,288 bytes | 8 bytes (m=8) | 1,536× |

At 100M vectors:
```
768-dim float32:  100M × 3072 bytes = 307 GB
PQ compressed:    100M × 8 bytes    = 0.8 GB  ← fits in RAM!
```

### Phase 2: Search — Asymmetric Distance Computation (ADC)

The database stores **centroid IDs**, not floats. Distance computation becomes a **table lookup + addition** — no float math per vector.

```
Once per query: Build the distance table
  Split query into m sub-vectors
  For each sub-space: compute distance from query sub-vector to all K centroids
  → m × K table (small: ~256 × 8 = 2048 values)

Per database vector: Look up + add
  A stored vector = [c2, c3, c4] (centroid IDs for each sub-space)
  Approx distance = table[sub1][c2] + table[sub2][c3] + table[sub3][c4]
  = just 3 table lookups + 2 additions → no float distance computation!
```

**Why "Asymmetric":** The query stays at full precision, while documents are compressed. The costly query-centroid math happens once; scanning a million vectors is then just lookups and additions — no per-vector float distance.

### FAISS Implementation

```python
import faiss

m = 8           # number of sub-vectors
nbits = 8       # bits per sub-vector (256 centroids)

# PQ index
index = faiss.IndexPQ(d, m, nbits)
index.train(vecs)
index.add(vecs)

# IVF+PQ (the production workhorse)
nlist = 100
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
index.train(vecs)
index.add(vecs)
index.nprobe = 10
```

---

## 7.7 HNSW — Hierarchical Navigable Small World

HNSW is the **default index in ChromaDB, Qdrant, and Weaviate** — and for good reason.

### How It Works

HNSW builds a **multi-layer graph** where:
- Bottom layer: dense graph of all vectors
- Upper layers: progressively sparser "express lanes"
- Each vector has a small number of neighbours (edges) in each layer

**Search:** Start at the top (sparse) layer, greedily traverse to the closest neighbour, drop down to the next layer, repeat. Like navigating a city on highways first, then surface roads.

### Trade-offs

| Property | HNSW | IVF |
|---|---|---|
| Recall at query time | Excellent (95–99%) | Good (varies with nprobe) |
| Query latency | Very low | Low |
| Build time | Moderate | Fast |
| **RAM requirement** | **High** — full graph in RAM | Low — only centroids in RAM |
| Memory footprint | Large | Manageable |

**HNSW is the best recall-per-millisecond option when RAM allows.** For corpora where the full index fits in RAM, HNSW is almost always the right choice.

---

## 7.8 When to Use Which Index

| Corpus Size | Index Choice | Reason |
|---|---|---|
| ≤ 50K chunks | `IndexFlatIP` (exact KNN) | Fast enough; perfect recall; no approximation error |
| 50K – 1M chunks | IVF or HNSW | IVF if memory is tight; HNSW if RAM is available |
| 1M – 10M chunks | HNSW or IVF+PQ | HNSW for quality; IVF+PQ for memory efficiency |
| > 10M chunks | IVF+PQ | Billion-vector workhorse; memory compression essential |

---

## 7.9 ChromaDB vs FAISS — The Two Primary Tools

### ChromaDB

- **What it is:** Python-native, open-source vector database — full-stack (storage + index + metadata + API)
- **Default index:** HNSW with metadata filtering
- **Persistent:** On disk; simple Python API
- **Metadata filtering:** Built in — filter by `{"source": "contracts", "date": {"$gt": "2024-01-01"}}`
- **Sweet spot:** Development, prototyping, small-to-medium production (≤ tens of millions of vectors)

```python
import chromadb

client = chromadb.Client()
collection = client.create_collection("my_docs")

# Add documents
collection.add(
    embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
    documents=["text of chunk 1", "text of chunk 2"],
    metadatas=[{"source": "contracts"}, {"source": "FAQs"}],
    ids=["id1", "id2"]
)

# Query
results = collection.query(
    query_embeddings=[[0.15, 0.25, ...]],
    n_results=5,
    where={"source": "contracts"}   # metadata filter
)
```

### FAISS (Facebook AI Similarity Search)

- **What it is:** A *library*, not a database — no metadata, no persistence layer, no networking
- **Strengths:** Reference implementation for ANN at scale; supports Flat, IVF, HNSW, PQ, OPQ, GPU indices; billion-vector deployments
- **Sweet spot:** Maximum control over the index; GPU acceleration; building a custom system at very large scale; understanding what's happening under the hood

```python
import faiss, numpy as np

d = 768
index = faiss.IndexFlatIP(d)           # exact search
# or
index = faiss.IndexHNSWFlat(d, 32)    # HNSW, M=32 neighbours
# or
index = faiss.IndexIVFPQ(...)          # IVF + PQ

index.add(vecs.astype(np.float32))
scores, I = index.search(query_vec, k=5)
```

### Choosing Between Them

| Criterion | ChromaDB | FAISS |
|---|---|---|
| Metadata filtering | Built-in ✅ | Manual ❌ |
| Persistence | Built-in ✅ | Manual ❌ |
| API simplicity | Simple ✅ | Complex |
| Index control | Limited | Full (IVF, PQ, HNSW, GPU) |
| Scale | ≤ tens of millions | Billions |
| When to use | Development, prototyping, standard RAG | Research, huge scale, GPU, custom systems |

### Other Notable Vector Stores

| Store | Type | Strengths |
|---|---|---|
| **Pinecone** | Managed SaaS | Fully managed, zero ops |
| **Qdrant** | Open-source (Rust) | Full-featured, hybrid search built-in |
| **Weaviate** | Open-source | GraphQL API, hybrid search built-in |
| **Milvus** | Open-source | Distributed, cloud-native |
| **pgvector** | Postgres extension | Already have Postgres? Add vectors |
| **Elasticsearch/OpenSearch** | Enterprise search | Vector + BM25 hybrid |
| **SQLite-VSS** | Lightweight | Embedded, no infrastructure |

**Choosing in production:** The choice is dominated by *operational* concerns (existing infrastructure, scale, latency SLAs) far more than by raw retrieval quality.

---

## 7.10 The Full RAG Pipeline with Vector DB

```
INGESTION (Offline · Run Once)
──────────────────────────────
raw files ──► load + clean ──► chunk ──► embed ──► FAISS / Chroma index
(pdf·docx·html·scan)  (OCR fallback)  (5 strategies)  (384-d)  (exact or ANN)
                                                           ↓
                                              persist: index + metadata

QUERY (Online · Every Request)
──────────────────────────────
question ──► embed ──► ANN top-k ──► rerank ──► LLM + context ──► answer
            (same model)  (ms search)  (+filter/hybrid)         (grounded+cited)
```

---

## 7.11 IVF vs PQ — The Key Distinction

This is one of the most important conceptual distinctions in vector search:

| | IVF | PQ |
|---|---|---|
| **Problem it solves** | Too many vectors to check | Vectors too big to store |
| **Mechanism** | Cluster → search only nearest clusters | Split → compress to centroid IDs |
| **Saves** | Search time (compute) | Memory (storage) |
| **Trade-off** | Can miss vectors in un-searched clusters | Approximate distances (quantisation error) |
| **Parameter** | `nprobe` (clusters searched) | `m` (sub-vectors), `nbits` (bits per sub-vector) |
| **Combine?** | Yes — IVF+PQ is the billion-scale workhorse | |

---

## 7.12 Semantic Search — Lexical vs Dense vs Hybrid

| Search Type | Mechanism | Strengths | Weaknesses |
|---|---|---|---|
| **Lexical (BM25)** | Match exact tokens | Fast, interpretable, rare terms, exact phrases | Vocabulary mismatch, no semantics |
| **Dense (ANN)** | Match embedding similarity | Paraphrase, synonymy, cross-lingual | Poor on exact identifiers, OOV |
| **Hybrid (Dense + BM25)** | Combine both via RRF | Best of both worlds | More complex infrastructure |

**Hybrid retrieval via Reciprocal Rank Fusion (RRF):**

```python
def rrf_score(results_list, k=60):
    scores = {}
    for results in results_list:          # one list per retriever
        for rank, doc_id in enumerate(results):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])

# Usage: merge BM25 and dense rankings
merged = rrf_score([bm25_results, dense_results])
```

RRF is scale-invariant: it only cares about rank position, not raw score magnitude. This makes fusion between BM25 (arbitrary score scale) and cosine similarity (−1 to 1) stable and clean.

---

## 7.13 Learning Thoughts

> **Thought 1:** IVF and PQ solve **orthogonal** problems and combine cleanly. IVF: "I have too many vectors to check all of them." PQ: "My vectors are too big to keep all of them in RAM." Understanding this distinction immediately clarifies which knob to turn when your system scales.

> **Thought 2:** HNSW's dominance in practice (default in ChromaDB, Qdrant, Weaviate) is because most RAG systems operate at a scale where the index fits in RAM, and in that regime, HNSW offers the best recall-per-millisecond. The moment RAM is the bottleneck, IVF+PQ becomes the answer.

> **Thought 3:** The nprobe parameter in IVF is a dial that trades recall for speed. There is no universally correct value — it depends on your latency budget and recall requirement. In production, you should profile at several nprobe values and pick the knee of the recall-latency curve.

> **Thought 4:** ChromaDB for understanding and prototyping; FAISS for understanding what's happening under the hood and for billion-scale. In real production, the vector store decision is dominated by operational concerns — does your team already run Postgres (pgvector)? Does your org already use Pinecone? These infrastructure concerns matter more than marginal recall differences.

> **Thought 5:** Asymmetric Distance Computation (ADC) in PQ is one of the most elegant algorithmic tricks in applied ML. The insight: the expensive part (query-centroid distance) happens once per query for all vectors. Per-vector scoring is then just table lookups + additions — O(1) arithmetic operations per vector regardless of dimensionality.

---

## 7.14 Important Interview Questions

**Conceptual**

1. **Why can't we use exact KNN search for large-scale RAG? What is the complexity?**
   - Exact KNN compares the query against every stored vector: O(N × D) per query. At 1M vectors × 768 dims, this is ~100ms per query on CPU — too slow for user-facing latency. We use ANN algorithms that trade a small amount of recall for massive speedup.

2. **What does IVF stand for and how does it work?**
   - Inverted File Index. Indexing: run K-Means on the corpus to find k centroids, assign each vector to its nearest centroid (Voronoi cell), store in buckets. Search: find the `nprobe` nearest centroids, then scan only those buckets. Reduces search from O(N) to O(k + nprobe × N/k).

3. **What is Product Quantisation and why does it achieve ~95%+ memory reduction?**
   - PQ splits high-dim vectors into m sub-vectors, runs K-Means on each sub-space to find K centroids, then encodes each sub-vector as a 1-byte centroid ID (for K=256). A 768-dim float32 vector (3072 bytes) becomes 8 bytes (m=8 sub-vectors × 1 byte each) — a 384× reduction.

4. **What is the difference between IVF and PQ? Are they compatible?**
   - IVF solves "too many vectors to check" by clustering + selective search. PQ solves "vectors too big to store" by compression. They are orthogonal and combine: IVF+PQ (FAISS `IndexIVFPQ`) is the standard billion-scale workhorse.

5. **What is HNSW and why is it the default in most vector databases?**
   - Hierarchical Navigable Small World — a multi-layer graph where upper layers are sparse "express lanes" and the bottom layer is dense. Search traverses hierarchically from coarse to fine. It offers the best recall-per-millisecond trade-off when the entire index fits in RAM — which is why ChromaDB, Qdrant, and Weaviate use it as their default.

6. **What is Asymmetric Distance Computation (ADC) in PQ search?**
   - The query is kept at full precision; database vectors are compressed (centroid IDs). The expensive query-centroid distances are computed once per query (m × K values). Per-vector scoring then becomes table lookups + additions — no per-vector float arithmetic. This asymmetry is what makes PQ search fast.

7. **What is ANN recall@K? How is it measured?**
   - `recall@K = |approximate_top_K ∩ exact_top_K| / K`. Measures what fraction of the true nearest neighbours appear in the ANN result. An ANN with recall@10 = 0.95 finds 9.5 of the true top-10 results. For RAG, recall@10 > 0.90 is typically acceptable.

**Applied / Design**

8. **You have 500M document chunks. RAM is limited to 32 GB. Which FAISS index do you choose?**
   - `IndexIVFPQ`. 500M × 3072 bytes (768-dim float32) = 1.5 TB — cannot fit in RAM. With PQ (m=16): 500M × 16 bytes = 8 GB — fits in 32 GB RAM with room for the index structure. Use nlist ≈ √500M ≈ 22,000 clusters with appropriate nprobe.

9. **When would you use ChromaDB vs FAISS in a production RAG system?**
   - ChromaDB: prototyping, standard enterprise RAG (< tens of millions of chunks), when you need built-in metadata filtering, persistence, and simple API. FAISS: billion-scale corpora, GPU acceleration, maximum index control, building a custom retrieval system.

10. **Explain the nprobe parameter in IVF. How do you tune it?**
    - `nprobe` = number of clusters (Voronoi cells) the search algorithm evaluates. Higher nprobe = higher recall but slower search. Tune by: build the index → run queries at nprobe = 1, 5, 10, 20, 50, 100 → measure recall@10 and latency → choose the nprobe value at the knee of the recall-latency curve.

11. **What is Reciprocal Rank Fusion and why is it used for hybrid retrieval?**
    - RRF combines rankings from multiple retrievers: `score(d) = Σ_r 1/(k + rank_r(d))`. Used because BM25 and dense similarity scores are on incompatible scales — RRF only uses rank positions, making fusion stable. k=60 is the standard parameter. It naturally down-weights documents ranked poorly by either retriever.

---

## 7.15 Section Summary

| Concept | One-line summary |
|---|---|
| Vector database | Stores (id, vector, metadata) tuples; primary operation is "find similar, not find equal" |
| KNN | Exact but O(N×D) — too slow beyond ~50K vectors |
| ANN | Approximate nearest neighbour — trades small recall loss for massive speedup |
| IVF | Cluster → search only nearest clusters; solves "too many vectors to check" |
| PQ | Compress vectors to centroid IDs; solves "vectors too big to store" |
| IVF+PQ | Combine both — the billion-scale production workhorse |
| ADC | Query-centroid distances computed once; per-vector scoring is just table lookups |
| HNSW | Graph-based; best recall-per-ms; costs RAM; default in Chroma/Qdrant/Weaviate |
| ChromaDB | Full-stack vector DB; built-in HNSW + metadata + persistence; for prototyping/SME RAG |
| FAISS | Library (not DB); maximum index control; GPU; for huge scale and custom systems |
| Hybrid retrieval | BM25 + dense via RRF = best of both worlds |

---

*Previous: [Section 6 — Chunking Strategies](S6_Chunking_Strategies.md)*
*Next: [Section 8 — Applied Practice & Quizzes](S8_Applied_Practice_Quizzes.md)*
