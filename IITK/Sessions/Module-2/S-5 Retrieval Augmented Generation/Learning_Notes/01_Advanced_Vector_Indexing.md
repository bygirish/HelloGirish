# Section 1 — Advanced Vector Indexing (ANN algorithms)

> Topics: IVF+PQ combined · HNSW · `efSearch` · why search is ~log(N) · FAISS · ChromaDB · the design decision tree.

---

## Big picture

This section answers one engineering question: **"How do I find the nearest vectors among millions/billions, fast, without running out of RAM?"**

Every method here is a point on the **recall ↔ latency ↔ memory** triangle. The skill is not memorising algorithms — it's knowing *which knob each one gives you* and *when to reach for it*.

```
        RECALL (accuracy)
              ▲
              │   FlatIP / FlatL2  (100% recall, slow, RAM-heavy)
              │   HNSW             (high recall, fast, RAM-heavy)
              │   IVFFlat          (good recall, fast, medium RAM)
              │   IVF+PQ           (some recall loss, fast, tiny RAM)
              ▼
     LATENCY ◀──────────────────────────────▶ MEMORY
```

---

## 1.1 IVF + PQ — combining the two

Recall from Section 0: **IVF** prunes *how many* vectors you score; **PQ** shrinks *how many bytes* each one costs. Combined, they give billion-scale search in limited RAM (this is FAISS's famous `IVF…,PQ…` index).

### Construction (indexing)
1. **Coarse (IVF):** run k-means (e.g. 1024 clusters) → coarse centroids.
2. **Residual:** assign each vector to its nearest centroid, then compute the **residual** = `vector − centroid` (the leftover after subtracting the cluster center).
3. **Fine (PQ):** split the *residual* into sub-vectors and train PQ codebooks **on the residuals**.
4. **Store:** save the PQ codes (byte IDs) into that centroid's inverted list.

> 🔑 **Why quantise residuals, not raw vectors?** Residuals are small and centered near zero (you already subtracted the bulk via the centroid). Small residuals → smaller quantisation error → better recall. This is the clever bit that makes IVF+PQ work.

### Search (retrieval)
1. **Coarse (IVF):** compare query to coarse centroids → pick `nprobe` closest buckets (e.g. top 10).
2. **Shift:** for each bucket, `query_shifted = query − centroid` (match the residual space the codes live in).
3. **Build ADC tables:** precompute **A**symmetric **D**istance **C**omputation lookup tables — query-to-centroid distances for each sub-space, computed *once*.
4. **Scan:** iterate PQ codes in the chosen buckets, summing values from the lookup table to approximate each distance.

### Why the approximation works (the three pillars)
| Pillar | What it does | Formula from deck |
|--------|--------------|-------------------|
| **PQ = compression** | split D-dim into m sub-vectors, 1 byte each | `768×4 B → m B` (m=96 ⇒ ~32× smaller) |
| **ADC = cheap distance** | keep query full-precision, DB quantised; precompute then sum lookups | `d(q,x)² = Σᵢ LUTᵢ[codeᵢ]` |
| **IVF = prune + recall** | k-means cells, scan only nearest; residual coding keeps error small | `scanned = (nprobe/nlist)·N` |

**One line:** *IVF skips ~90% of vectors by clustering; PQ makes the survivors ~32× smaller and each distance a table lookup — recall is bought back with `nprobe`.*

**ADC mental model:** Instead of measuring the distance to every compressed point one expensive subtraction at a time, you pre-measure "how far is my query from each of the 256 swatches in each sub-space," write it in a table, then scoring a point = just *adding up table entries by its byte codes*. Addition is cheap; that's the speed.

---

## 1.2 HNSW — Hierarchical Navigable Small World

A **graph-based** ANN index (the default in many vector DBs — Qdrant, Weaviate, Milvus, and FAISS). No clustering; instead, a navigable proximity graph.

### Key ideas
- **Multi-layer graph**, coarse (top) → fine (bottom):
  - **Top layers:** few nodes, long-range links (the "highways").
  - **Bottom layer (Level 0):** *all* nodes, short-range local links (the "local streets").
- **Small-world property:** most nodes are reachable in a small number of hops.
- **Greedy search across layers.**

### Search (for query q)
1. **Enter at the top layer** at a fixed entry point.
2. **Greedy hop** to the neighbour closest to q; repeat until no neighbour is closer.
3. **Drop down a layer** to that closest node; repeat.
4. At **Level 0**, return the closest node(s).

**Analogy (deck's own):** a road network — take highways to get *near* the destination fast, then drop to local streets for the exact address. Or: a **skip list lifted into high-dimensional space**.

### Complexity
| | |
|---|---|
| Search | **O(log N)** (~sub-linear, very fast in practice) |
| Insert | O(log N) |
| Space | O(M·N), where M = max neighbours per node |

### The two knobs
- **`M`** = edges per node, fixed at **build time**. Sets the graph's baseline quality (recall ceiling) and memory. Higher M → better recall, more RAM.
- **`efSearch`** = candidate list width at **query time** — the recall–speed dial (HNSW's equivalent of IVF's `nprobe`).

---

## 1.3 `efSearch` — the candidate list that controls recall

`efSearch` = the **width of the search at Layer 0** — how many close candidates HNSW keeps "in play" before it answers.

- Upper layers just **navigate** (beam = 1).
- At Layer 0, the search keeps the `ef` closest candidates, exploring neighbours until none beats the **worst one currently kept**.
- Insertion rule: a new node `d=0.13` is inserted into the candidate list if it's closer than the current worst (`0.22`), which then gets dropped.

Trade-off:
- **Low ef** → fast, may miss the true neighbour.
- **High ef** → wider search, higher recall, slower.
- Must hold **`ef ≥ k`** (you return top-k from the ef candidates).
- It's a **query-time dial** — tune per query, **no index rebuild needed**. (Contrast `M`, which needs a rebuild.)

> **Same dial, new hat:** `efSearch` is to HNSW what `nprobe` is to IVF — both are the runtime recall/speed knob.

---

## 1.4 Why HNSW search is ~log(N), not N

| Concept | Insight | Formula |
|---------|---------|---------|
| **Skip-list intuition** | Each higher layer keeps a fraction of nodes, so you leap exponentially far per hop. | `P(node in layer L) = m^(−L)` |
| **Why logarithmic** | Layers thin geometrically; the top holds O(1) nodes and each descent roughly halves the remaining distance. | `E[hops] = O(log N)` |
| **The two knobs** | M = edges/node (build-time, sets recall + memory); efSearch = query-time recall dial. | `memory = O(N·M)` |

**One line:** *greedy descent through thinning layers turns a linear scan into a logarithmic walk — recall traded against ef, baseline fixed by M.*

---

## 1.5 FAISS — the library

> **F**acebook **AI** **S**imilarity **S**earch (Johnson, Douze, Jégou — "Billion-scale similarity search with GPUs").

FAISS is a **library** for efficient similarity search and clustering of dense vectors — scales to vector sets that don't even fit in RAM.

Two main starter approaches from the deck:
1. **`IndexFlatL2`** — exact L2 matching; brute force but faster than a hand-rolled loop.
2. **`IndexIVF`** — inverted file; clusters features, searches only relevant clusters.

(Plus `IndexFlatIP` for inner-product/cosine, and `IVF…,PQ…` from 1.1.)

**Crucial framing:** FAISS is a *library*, **not a database**. It finds nearest vectors and returns **IDs** — no text, no metadata, no persistence out of the box. You must keep your own `id → text/metadata` map and save/load the index yourself. Losing that map is the classic **"FAISS gave me numbers but I lost my documents"** bug.

---

## 1.6 ChromaDB — a database wrapped around vector search

ChromaDB is the **database** that wraps an ANN library with everything a RAG app actually needs.

### The raw-search problem
A bare similarity search returns *numbers*: "vector #4471 is closest." But the LLM needs the **text**, and you need the **source** to filter & cite. With FAISS you track the ID→text map yourself — the "numbers but no documents" bug.

### Chroma stores all three together
| Stored | For |
|--------|-----|
| **Vector** | the similarity math |
| **Text** | the actual chunk the LLM reads |
| **Metadata** | source, page, date — to filter & cite |

### The feature raw search can't do: **metadata filtering**
```python
collection.query(query_embeddings=q, n_results=5, where={"source": "policy.docx"})
```
Pre-filter *before* the search → higher precision **and** a smaller, faster candidate set. (This `where` filter is also how access control gets enforced — see Section 6.)

### The trade-off
Under the hood Chroma uses HNSW (one index, CPU, single node). Convenience now, a **ceiling later** — graduate to FAISS for billions or fine index control.

**One line:** *take an ANN library like FAISS, wrap it in a real database — text, metadata, persistence, filtering included.*

---

## 1.6 Design decisions of a VectorDB

Two questions stack:

### Question 1 — database or library (or managed)?
| You are… | Choose | Because |
|----------|--------|---------|
| Prototype / course / app under ~a few million vectors, want metadata + persistence with minimal code | **ChromaDB** | stores vectors+text+metadata together, persists to disk, filters by metadata — a working semantic search service in ~15 lines. Runs FAISS-class indexes underneath anyway. |
| Need raw speed / huge scale / fine index control / embedding inside another system | **FAISS (library)** | fastest, most tunable, scales to billions, full control over index + compression. Cost: not a DB — no persistence, metadata, or text storage; you manage the `id→text` map. |
| (Implied) want it all managed | Managed (Pinecone, Weaviate, Milvus, Qdrant) | someone else runs ops/scaling. |

### Question 2 — which FAISS index? (corpus size is the main driver)
| Branch | Situation | Choose | Knobs / notes |
|--------|-----------|--------|---------------|
| **3D** | Up to ~1M vectors, **or** you require 100% recall | **`IndexFlatIP`** (exact brute force) | no training, no tuning. *Don't trade away accuracy to solve a speed problem you don't have yet.* |
| **3E** | Millions of vectors, tolerate small recall loss, fits in RAM | **`IVFFlat`** | `nlist ≈ √N`, `nprobe` = recall dial. Needs training (≥ ~30–40× nlist points). `nprobe=1` fastest; `nprobe=nlist` = exact. |
| (from quiz) | ~50M chunks, **strict RAM limits**, tolerate slight recall drop | **`IVF + PQ`** | IVF prunes scope, PQ compresses footprint → fits massive data in RAM. |

> The deck's quiz answer: *50M chunks + strict RAM + tolerable recall loss → **IVF+PQ** (option B).* HNSW would be fast but RAM-heavy; FlatIP is exact but won't fit; cross-encoder isn't an index at all.

### Decision cheat-sheet
```
Need 100% recall OR ≤1M vectors? ───────────────▶ FlatIP   (exact)
Millions, fits in RAM, small recall loss OK? ───▶ IVFFlat  (nprobe dial)
Tens of millions + tight RAM? ──────────────────▶ IVF+PQ   (compress)
Need lowest latency at high recall, RAM is fine?▶ HNSW     (efSearch dial)
Want text+metadata+persistence, minimal code? ──▶ ChromaDB (HNSW inside)
```

---

## 🎯 Interview questions

**Q1. IVF vs PQ — what does each solve, and why combine them?**
IVF = "too many vectors" (cluster, scan fewer). PQ = "vectors too big" (compress to byte codes). Orthogonal, so combine: IVF prunes count, PQ shrinks size → billion-scale in limited RAM.

**Q2. In IVF+PQ, why are residuals quantised instead of the original vectors?**
After subtracting the cluster centroid, residuals are small and centered near zero, so quantisation error is much smaller → better recall for the same code size.

**Q3. What is ADC and why is it fast?**
Asymmetric Distance Computation: keep the query full-precision, the DB quantised. Precompute query-to-centroid distance lookup tables per sub-space once, then approximate any point's distance by *summing table entries* indexed by its byte codes. Distance scoring becomes additions, not full vector math.

**Q4. Explain HNSW to a non-expert.**
A layered "road network" of vectors: sparse highways on top for big jumps, dense local streets at the bottom. You greedily hop toward the query on the highways, drop down layer by layer, and finish on local streets — turning a linear scan into a ~log(N) walk.

**Q5. `nprobe` vs `efSearch` vs `M` — compare.**
`nprobe` (IVF) and `efSearch` (HNSW) are both **query-time recall/speed dials**, tunable without rebuild. `M` (HNSW) is a **build-time** parameter setting edges-per-node → recall ceiling + memory; changing it requires a rebuild.

**Q6. Why is HNSW search logarithmic?**
Layers thin geometrically (`P(node in layer L)=m^(−L)`); the top has O(1) nodes and each descent roughly halves the remaining distance, giving `E[hops]=O(log N)`.

**Q7. FAISS vs ChromaDB — when each?**
FAISS = library: fastest, most tunable, scales to billions, but no text/metadata/persistence (you keep the id→text map). ChromaDB = database wrapping ANN: stores vector+text+metadata, persists, filters by metadata, ~15 lines to a working service; ceiling at very large scale.

**Q8. 50M chunks, strict RAM, slight recall loss acceptable — which index?**
IVF+PQ. IVF reduces search scope via clustering; PQ heavily compresses the memory footprint, directly addressing the RAM constraint. (FlatIP won't fit; HNSW is RAM-heavy.)

**Q9. Why is choosing FlatIP for 500k vectors often the *right* call?**
At that scale brute force is fast enough and gives 100% recall with zero training/tuning. Introducing IVF/PQ/HNSW adds approximation error and complexity to solve a speed problem you don't yet have.

---

## 🧠 Learning thoughts

- **There is no "best" index — only best *for a corpus size + constraint*.** Interviewers test whether you reach for approximation prematurely. Default to exact until scale forces your hand.
- The recurring pattern: **a build-time structural choice + a query-time recall dial.** IVF: (nlist build) + (nprobe query). HNSW: (M build) + (efSearch query). Spotting this pattern lets you reason about any new index.
- **Library vs database is an architecture decision, not a performance one.** FAISS gives raw power but pushes the "where's my text?" burden onto you. The id→text map bug is real and common.
- **Metadata filtering is secretly a security + precision feature**, not just convenience — it pre-shrinks the candidate set *and* (Section 6) enforces access control at the DB layer.
- Residual quantisation is the single most "aha" idea here: *subtract the predictable part, only compress the leftover.* That principle recurs across ML (e.g. residual connections, delta encoding).

## ✅ Self-check

1. For each of {FlatIP, IVFFlat, IVF+PQ, HNSW}, name the corpus situation it's the right choice for.
2. Which knobs are build-time vs query-time across IVF and HNSW?
3. Explain why ADC turns distance computation into addition.
4. Your teammate stores vectors in FAISS and is confused why search returns `[4471, 882, ...]` and "no documents." What's missing and how does ChromaDB avoid it?
