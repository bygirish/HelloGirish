# Section 2 — The Retrieval Pipeline & Reranking

> Topics: semantic search pipeline + minimal code · the RAG design decision tree · cross-encoder · bi-encoder vs cross-encoder · ColBERT (late interaction) · the two-stage pattern · three silent failures.

---

## Big picture

Section 1 was *how the index finds vectors*. Section 2 is *how you turn a query into the best chunks for the LLM* — and the central trade-off of all retrieval:

> **Accuracy and speed pull in opposite directions. The industry answer is two stages: retrieve cheaply at scale, then rerank the survivors expensively.**

---

## 2.1 The semantic search pipeline

The ANN search pipeline — from query to top-k in 4 steps:

```
1. Embed the query     2. ANN search        3. Score & rank        4. Return top-k
   q(text) → q ∈ ℝᵈ       Index I →             score(q,c) →           top-k results
   (SAME model as docs)   candidates C          rank(C)                → to the LLM
```

1. **Embed the query** — with the **same model** used for the documents → same vector space (non-negotiable; mismatched models = garbage).
2. **ANN search** — query the index (HNSW/IVF) for nearest vectors → candidate set C.
3. **Score & rank** — cosine (default for text), dot product, or L2.
4. **Return top-k** — hand the best chunks to the LLM as context.

**The recall–speed dial:** ANN indexes expose tuning knobs (e.g. HNSW's `ef`). Push for recall → queries slow; push for speed → risk missing a true neighbour. Most text models are trained for **cosine** — stick with it unless you have a reason not to. In production, pair dense retrieval with **BM25 (hybrid)** for the best of both.

### Minimal end-to-end pipeline (from the deck)
```python
from sentence_transformers import SentenceTransformer
import faiss, numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')   # 384-dim, local

# 1) embed your chunks (already parsed + chunked)
emb = model.encode(chunks, normalize_embeddings=True)

# 2) build an index (cosine == inner product on L2-normed vectors)
index = faiss.IndexFlatIP(emb.shape[1])
index.add(np.asarray(emb, dtype='float32'))

# 3) embed the query with the SAME model, then search
q = model.encode(['how do I reduce API latency?'], normalize_embeddings=True)
scores, ids = index.search(np.asarray(q, 'float32'), k=5)
top_chunks = [chunks[i] for i in ids[0]]          # -> feed to the LLM
```
> 📌 Note the two subtleties that prevent the most common bugs: **same model** for chunks and query, and **`normalize_embeddings=True`** so inner product equals cosine.

---

## 2.2 Design choices in RAG systems (the decision tree)

The deck's master slide — five sequential decisions. Memorise this as your RAG design checklist:

| Stage | Question | Key choices |
|-------|----------|-------------|
| **1. Document processing** | What's the source format? | Layout-aware parsing (if structure carries meaning: headings, tables) vs plain text extraction. Then **clean & normalize** (strip headers/footers, de-dup, fix encoding). |
| **2. Chunking strategy** | What kind of retrieval? | Structure-aware (by section/paragraph/table) vs fixed/sliding window (overlap 10–20%). **Small chunks (200–400 tok) → higher recall; large (800–1500) → more context.** |
| **3. Embeddings** | Which model? | API (OpenAI, Cohere, Voyage) vs local (BGE, E5, Instructor, Qwen). Check model card for max tokens, dims, required `query:`/`passage:` prefixes; normalize. |
| **4. Vector store** | What scale? | Small (<100k): simple (FAISS-flat, Chroma, SQLite-VSS). Large (>10M) or needs metadata filtering: Pinecone, Weaviate, Milvus, Qdrant. Consider index type (HNSW/IVF/PQ), replication, hybrid. |
| **5. Retrieval & reranking** | How will you retrieve? | Speed → dense bi-encoder. Quality → cross-encoder rerank. **Recommended hybrid:** retrieve top 100–200 with bi-encoder → rerank with cross-encoder → use top 5–20 for generation. |

**Key principle (deck):** *Good RAG is making the right design choices at every step — clean data, meaningful chunks, appropriate embeddings, scalable storage, effective retrieval.* And: **80% of bad RAG answers start with poor document processing.**

---

## 2.3 Cross-Encoder (dense retrieval, costlier but accurate)

```
z = BERT(q; [SEP]; d)[CLS]
score(q,d) = softmax(U(z))
```

A **single encoder jointly encodes query + document**, then a linear layer over the `[CLS]` token produces a relevance score. Because every query token can attend to every document token, it's **very accurate** — but **too compute-expensive to use except in rescoring** (you can't precompute anything; every (query, doc) pair is a fresh forward pass).

---

## 2.4 Bi-encoder vs Cross-encoder — the cost of accuracy

Same goal (score (query, doc) relevance), but **where you spend attention** decides speed.

| | **Bi-encoder** | **Cross-encoder** |
|---|----------------|-------------------|
| How | Encode query and doc **separately**, score by dot product. | Feed `[q; SEP; d]` through **one** transformer; every query token attends to every doc token. |
| Formula | `score = E(q) · E(d)` | `score = sig(W · BERT([q;d]))` |
| Docs encoded | **once, offline**, then indexed | **never precomputed** — re-encode per pair |
| Speed | fast, approximate | slow, precise |
| Use | **first-stage retrieval** over millions | **reranking** a small candidate set |

**Why two stages:** cross-encoding all N docs = N forward passes → infeasible. So **retrieve K cheaply with the bi-encoder, rerank only those K** with the cross-encoder. `N passes → K passes (K ≪ N)`.

**One line:** *bi-encoders push work offline (cheap, approximate); cross-encoders pay per pair (precise, costly) — so you retrieve cheap and rerank dear.*

---

## 2.5 ColBERT — late interaction (the middle ground)

The spectrum of query–document interaction:
- **(a) Representation similarity** (DSSM, SNRM) — one vector each, dot product. *Bi-encoder.*
- **(b) Query-document interaction** (DRMM, KNRM) — interaction matrix.
- **(c) All-to-all interaction** (BERT) — full joint attention. *Cross-encoder.*
- **(d) Late interaction** (**ColBERT**) — the proposed middle ground.

### How ColBERT works
- **Token-level embeddings:** store **one vector per token**, not per document. `doc → {e₁, e₂, ... eₙ}`. These are precomputed and indexed offline (like a bi-encoder — no re-encoding at query time).
- **MaxSim operator:** for each query token, take its **best match** among all doc tokens, then **sum those maxima**:
  `score = Σᵢ maxⱼ (eᵢ · eⱼ)`
- Interaction happens **late** — *after* encoding, at query time, via the cheap MaxSim.

### The trade-off
Accuracy **near a cross-encoder** at a fraction of the cost — but storing **every token vector** inflates the index dramatically: `storage ~ tokens, not docs`.

**One line:** *ColBERT defers interaction to query time via MaxSim — near cross-encoder quality with bi-encoder-style precompute, paid for in index size.*

> Use ColBERT when you need cross-encoder-grade accuracy but **cannot afford query-time joint attention** and **can afford the storage** (this is exactly the deck's quiz scenario → answer D).

### The three approaches at a glance
| | Encode docs offline? | Query-time cost | Accuracy | Storage |
|---|---|---|---|---|
| Bi-encoder | ✅ one vector/doc | dot product (cheapest) | good | small |
| ColBERT | ✅ one vector/token | MaxSim (cheap) | near cross-encoder | **large** |
| Cross-encoder | ❌ nothing | full attention/pair (expensive) | best | n/a |

---

## 2.6 The two-stage pattern is industry standard

```
1. RETRIEVE                    2. RERANK                       3. GENERATE
Bi-encoder + ANN index         Cross-encoder rescores each     Top 5–20 reranked
pulls top 100–200 from         (query, candidate) pair         chunks → LLM as context
millions — fast, approximate   jointly — slow but precise
```

Seen in: **Google / Bing two-stage ranking**; **FAISS powering similarity search across Meta products**.

**Best practice (deck): *Retrieve broadly, then rerank narrowly for high-quality results.***

---

## 2.7 Three failures that silently wreck a RAG system

These don't crash — **no error, no exception, just worse answers.** The most dangerous kind of bug.

| # | Failure | Cause | Fix |
|---|---------|-------|-----|
| 1 | **Empty chunks to prod** | A scanned PDF has no text layer — `pypdf` returns empty strings silently. | Always **probe for a text layer**; route to **OCR** if absent. |
| 2 | **Half the recall, no warning** | An **asymmetric** embedding model used *without* its `query:` / `passage:` prefixes. | **Read the model card** and apply required prefixes. |
| 3 | **Euclidean on a cosine model** | Using the DB's default **L2** metric on a model trained for **cosine**. | **Match the metric to training** (cosine for cosine models). |

> These are the bugs your users hit in production. Each degrades quality invisibly — which is why **evaluation and observability** (Section 6) matter so much.

---

## 🎯 Interview questions

**Q1. Why must the query and documents use the same embedding model?**
They must live in the same vector space for distances to be meaningful. Different models produce incompatible geometries → similarity scores become noise.

**Q2. Bi-encoder vs cross-encoder — architecture and use-case?**
Bi-encoder encodes query and doc separately (precomputable, fast, approximate) → first-stage retrieval. Cross-encoder jointly encodes the pair with full attention (accurate, no precompute, expensive) → reranking a small candidate set.

**Q3. Why can't we just use a cross-encoder for everything?**
It can't precompute doc representations; scoring N docs = N forward passes. Infeasible at scale. Hence retrieve K cheaply, rerank only K.

**Q4. Explain ColBERT and where it fits.**
Late interaction: store one vector per token (precomputed), then at query time score via MaxSim (sum of per-query-token best matches). Near cross-encoder accuracy without query-time joint attention; cost is large index size (storage scales with tokens). Use when you need accuracy + low latency and can pay in storage.

**Q5. Enterprise needs near-cross-encoder accuracy, can't afford query-time joint attention, willing to spend storage. What architecture?**
ColBERT (late interaction) — precomputes token embeddings, defers interaction to a lightweight MaxSim, trading storage for accuracy. (Deck quiz answer D.)

**Q6. Describe the two-stage retrieve-rerank pattern and typical numbers.**
Stage 1: bi-encoder + ANN pulls ~100–200 candidates from millions (fast). Stage 2: cross-encoder rescores those pairs (precise). Then top 5–20 go to the LLM. "Retrieve broadly, rerank narrowly."

**Q7. Name three silent RAG failures and their fixes.**
(1) Scanned PDF → empty chunks; probe for text layer, OCR. (2) Asymmetric model without query/passage prefixes → recall halves; read the model card. (3) L2 metric on a cosine-trained model → underperforms; match metric to training.

**Q8. What is hybrid retrieval and why use it?**
Combine dense (semantic, embedding-based) with sparse (BM25, keyword/exact-match). Dense captures meaning; sparse nails exact terms, codes, rare jargon. Together they cover each other's blind spots.

**Q9. How do chunk size choices affect retrieval?**
Smaller chunks → higher recall (more granular matches) but less context per hit; larger chunks → richer context but lower recall and more noise. Tune to the query type; use overlap (10–20%) to avoid splitting answers.

---

## 🧠 Learning thoughts

- **The entire section is one trade-off curve**: bi-encoder (fast/approx) ── ColBERT (middle) ── cross-encoder (slow/precise). Place any new retrieval method on this curve and you understand it.
- **"Retrieve cheap, rerank dear" is the most important production pattern in RAG.** It's how you get cross-encoder accuracy at bi-encoder scale. Expect it in every system-design interview.
- ColBERT's lesson generalises: **when can you move interaction earlier (precompute) vs later (query-time)?** Late interaction is a sweet spot you can apply elsewhere.
- **Silent failures are the real enemy in production.** A crash gets fixed in minutes; a 20% recall drop with no error can live for months. This is the emotional case for evaluation/observability.
- Embeddings are a *contract*: same model, right prefixes, right metric, normalized. Break any clause and quality degrades quietly.

## ✅ Self-check

1. Place bi-encoder, ColBERT, cross-encoder on the speed↔accuracy line and state what each precomputes.
2. Write the MaxSim formula and explain it in one sentence.
3. Give the two-stage pattern with realistic candidate counts at each stage.
4. You inherit a RAG system with mysteriously poor recall. List the three silent failures you'd check first.
