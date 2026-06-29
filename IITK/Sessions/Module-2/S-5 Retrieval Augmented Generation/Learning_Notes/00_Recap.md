# Section 0 — Recap: The RAG Pipeline, IVF & PQ

> Grounding material carried over from the previous lecture. Don't skip — Section 1 builds directly on IVF and PQ.

---

## Big picture

**Retrieval Augmented Generation (RAG)** = give an LLM the right *external* knowledge at answer-time, instead of relying only on what's baked into its weights.

Why it matters:
- LLMs have a **knowledge cutoff** and **hallucinate** on facts they never saw.
- Retraining/fine-tuning to add knowledge is slow and expensive.
- RAG injects fresh, private, or domain-specific text into the prompt **at query time** — cheap, auditable, and updatable by just editing documents.

The slogan: **"Don't memorise the library, learn to look things up."**

---

## 0.1 The RAG pipeline — 4 stages

The deck frames RAG as four stages. Internalise these labels; the rest of the lecture hangs off them.

```
PRE-RETRIEVAL          RETRIEVAL              POST-RETRIEVAL        AUGMENTATION + GENERATION
─────────────          ─────────              ──────────────        ─────────────────────────
Docs (JSON)            User query             Query + candidate     Query + reranked docs
   │                      │                    docs                     │
Section-based          Embed query              │                  Prompt template
 chunking                 │                   Reranker                  │
   │                   ANN search                │                   LLM
Embedding             in Vector DB           Reranked docs             │
 generation              │                                          Answer
   │                  Query-relevant docs
Store in              + similarity scores
Vector DB
```

| Stage | What happens | Key decisions |
|-------|--------------|---------------|
| **Pre-Retrieval** | Parse documents → chunk → embed → store in vector DB. **Offline / one-time.** | chunk size, chunking strategy, embedding model |
| **Retrieval** | Embed the user query → Approximate Nearest Neighbour (ANN) search → get top-k candidate chunks with similarity scores. | index type, k, similarity metric |
| **Post-Retrieval** | **Rerank** candidates with a more accurate (slower) model so the best chunks rise to the top. | reranker model, how many to keep |
| **Augmentation + Generation** | Stuff reranked chunks + query into a **prompt template** → LLM produces the grounded answer. | prompt design, context window budget |

**Mental model:** Pre-Retrieval is the *librarian shelving books*; Retrieval is *pulling a shortlist off the shelf fast*; Post-Retrieval is *a careful read to reorder the shortlist*; Generation is *writing the answer citing those pages*.

---

## 0.2 IVF — Inverted File index

**Problem it solves:** A brute-force nearest-neighbour search compares the query to **every** vector — O(N). At millions/billions of vectors that's too slow.

**Idea:** Cluster the vectors first (k-means → `nlist` centroids). At query time, don't scan everything — scan only the **nearest cluster(s)**.

Two-phase search:
1. **Coarse search** — compare the query *only* against the `k` centroids → find the closest cluster(s).
2. **Fine search** — scan *only* the vectors inside that bucket; ignore the rest.

Key knob:
- **`nprobe`** = how many clusters (centroids) the search actually inspects. Default `nprobe = 1`.
  - `nprobe ↑` → higher recall, slower (you check more buckets).
  - `nprobe = nlist` → degenerates back to exact brute force.

**Outcome:** search scope drops from **O(N) → ~O(N/k)**.

> ⚠️ The trap: if the true nearest neighbour sits in a *neighbouring* bucket you didn't probe, you miss it. That's the recall cost of clustering — `nprobe` buys it back.

---

## 0.3 PQ — Product Quantisation

**Problem it solves:** Vectors are **big**. A 768-dim float32 vector = 768 × 4 = 3072 bytes. Millions of those blow past RAM.

**Idea (compression):** Replace each vector with a tiny **byte code** that approximates it.

Indexing / compression phase:
1. **Split** — divide the D-dim vector into `m` sub-vectors (e.g. 128-D → 8 chunks of 16-D).
2. **Train codebooks** — run k-means independently on each sub-space to find `K` (e.g. 256) representative centroids = the "codebook" for that sub-space.
3. **Encode** — replace each sub-vector with the **ID (0–255)** of its nearest centroid.
4. **Result** — high-precision floats become small byte codes; memory drops **~95%+** (e.g. 4 KB → 64 bytes).

**Mental model:** Instead of storing the exact colour of every pixel, store "which of these 256 swatches is closest." You lose a little precision, save enormous space.

> Why 256? Because an ID 0–255 fits in exactly **one byte**. That's the whole point.

---

## How IVF and PQ relate (the bridge to Section 1)

They solve **two different problems**:

| Technique | Problem | Mechanism |
|-----------|---------|-----------|
| **IVF** | "Too many vectors to check" | clustering → prune how *many* you scan |
| **PQ** | "Vectors are too big to store" | quantisation → shrink how many *bytes* each costs |

Section 1 opens by **combining them (IVF + PQ)** — the workhorse index for billion-scale, RAM-constrained search.

---

## 🎯 Interview questions

**Q1. What is RAG and why use it over fine-tuning?**
RAG retrieves relevant external text and feeds it to the LLM at inference. Versus fine-tuning: it's cheaper, updates instantly (just change docs), gives citations/auditability, and avoids catastrophic forgetting. Fine-tuning teaches *behaviour/style*; RAG supplies *knowledge*.

**Q2. Walk me through the four stages of a RAG pipeline.**
Pre-retrieval (chunk + embed + store, offline) → Retrieval (embed query, ANN search for top-k) → Post-retrieval (rerank) → Augmentation+Generation (prompt template + LLM). Name what's tuned at each stage.

**Q3. Why is exact nearest-neighbour search impractical at scale, and how does IVF help?**
Exact search is O(N) per query. IVF clusters vectors and searches only the nearest `nprobe` clusters, cutting cost to ~O(N/k) — trading a small recall loss for big speedups.

**Q4. What does `nprobe` control and what's the trade-off?**
Number of clusters inspected per query. Higher `nprobe` = higher recall, more latency. `nprobe=1` is fastest/lowest recall; `nprobe=nlist` = exact search.

**Q5. How does Product Quantisation reduce memory, and what's the cost?**
It splits vectors into sub-vectors and replaces each with a 1-byte centroid ID from a trained codebook, cutting memory ~95%+. Cost: distances become **approximate** (quantisation error), so recall drops slightly.

**Q6. Why is the PQ codebook size typically 256?**
So each sub-vector's centroid ID fits in a single byte (0–255), maximising compression while keeping codes byte-aligned.

---

## 🧠 Learning thoughts

- **Every ANN technique is a deal with the same three devils: recall, latency, memory.** You cannot maximise all three. Learn each method by *which one it sacrifices*.
- IVF and PQ are **orthogonal** — one prunes count, one shrinks size. That's *why* they compose so well (Section 1).
- "Approximate" in ANN is a feature, not a bug. 99% recall at 50× speed beats 100% recall you can't afford.
- Pre-retrieval quality caps everything downstream. Garbage chunks → garbage answers, no matter how good your LLM is. (The deck later says **80% of bad RAG answers start with poor document processing.**)

## ✅ Self-check

1. Draw the 4 RAG stages from memory and label what's tuned in each.
2. A colleague sets `nprobe = nlist` "to be safe." What did they just throw away?
3. You compress 768-D float32 vectors with PQ (m=8, 256 centroids). What's the new size per vector, roughly? *(Answer: 8 bytes — one byte per sub-vector.)*
