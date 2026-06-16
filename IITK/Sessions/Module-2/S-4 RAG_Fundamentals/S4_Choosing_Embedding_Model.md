# Section 4 — Choosing an Embedding Model

> **Session:** IIT-KGP Module 2 · Session 4 · Instructor: Pawan Goyal
> **Theme:** *Every embedding choice is condition-driven. The leaderboard lies. Your data decides the winner.*

---

## 4.1 The Fundamental Principle

> "Don't pick by the leaderboard — pick by **measuring candidates on your own data**. MTEB rankings are volatile and a high average score hides which model is strong on *your* task and *your* domain."

The **Massive Text Embedding Benchmark (MTEB)** ranks hundreds of embedding models across dozens of tasks. It is a useful shortlist tool but a terrible selection tool, because:

1. It averages across tasks that may be irrelevant to you (e.g., clustering, classification, STS)
2. Your domain vocabulary may be underrepresented in MTEB evaluation sets
3. A model that ranks #3 on MTEB may rank #1 on your specific enterprise data

**The discipline:** Always run a **bake-off** — evaluate shortlisted models on your own queries and documents using Recall@K or MRR before committing.

---

## 4.2 Model Families

### OpenAI Embeddings: text-embedding-3-small / text-embedding-3-large

| Property | text-embedding-3-small | text-embedding-3-large |
|---|---|---|
| Dimensions | 1,536 (resizable) | 3,072 (resizable) |
| Normalisation | L2-normalised → dot product = cosine | Same |
| Pricing | Per-token API | Per-token API (higher) |
| Hosting | Hosted API only | Hosted API only |
| Matryoshka | Yes — can truncate dimensions | Yes |
| Best for | Fast start, no infra, top general quality | Highest quality when it earns the cost |

**Key capability: Matryoshka Representation Learning (MRL)**

`text-embedding-3-*` models are Matryoshka-trained — you can truncate the vector to any dimension ≤ native with graceful accuracy degradation:

```python
from openai import OpenAI
client = OpenAI()

response = client.embeddings.create(
    input="Your text string goes here",
    model="text-embedding-3-small",
    dimensions=512  # truncate from 1536 to 512 — graceful degradation
)
print(response.data[0].embedding)
```

**Trade-off:** Convenience and quality vs. cost, vendor lock-in, data egress, and rate limits.

### Sentence-Transformers (SBERT) Family

| Property | SBERT Family |
|---|---|
| Hosting | Self-hosted (CPU or GPU) |
| Data privacy | Stays on your infrastructure |
| Cost model | Compute you own (free library) |
| Typical dims | 384–768 (MiniLM, MPNet, BGE) |
| Customisation | Fine-tune on your domain data |
| Best for | Privacy, scale, custom domains |

Hundreds of pretrained models on HuggingFace:
- `all-MiniLM-L6-v2` — 22M params, 384 dims, CPU-friendly, for prototyping
- `all-mpnet-base-v2` — 110M params, 768 dims, better quality
- `BAAI/bge-large-en-v1.5` — top open-weight English model
- `intfloat/multilingual-e5-large` — multilingual, 560M params
- `BAAI/bge-m3` — multilingual, 100+ languages, 8192 token context

**Trade-off:** Full control vs. you own the operational burden (GPU memory, throughput, model updates).

---

## 4.3 Matryoshka Representation Learning (MRL) — Deep Dive

MRL is a training technique where the model is trained to make **nested prefixes of the vector independently useful**.

### How It Works

The same model, same forward pass. The trick is the **loss function**: it grades nested prefixes of the vector simultaneously.

```
ONE EMBEDDING VECTOR: [dim_0 ─────────────────────── dim_3071]

L(256) = loss computed on first 256 dims only
L(512) = loss computed on first 512 dims only
L(768) = loss computed on first 768 dims only
L(3072) = loss computed on all 3072 dims

L_total = L(256) + L(512) + L(768) + L(3072)
```

**Why early dimensions pack the most information:**
- Dims 0–255 are scored by *every* loss term
- The model feels the most pressure to pack broadly useful information in the first few dims
- Later dims only help the longer-prefix losses

**The analogy:** Describe an animal but you'll be cut off after an unknown number of words. You front-load: "small furry mammal…" first, fine details last. MRL trains the model with this front-loading pressure baked in.

**The key insight:** The coarse-to-fine ordering is *emergent* — a side effect of grading every prefix. A non-MRL model never felt this pressure, so slicing its first 512 dims shreds the embedding.

**Practical implication for RAG:**
```python
import numpy as np

def truncate(vecs, dim):
    v = vecs[:, :dim]
    return v / np.linalg.norm(v, axis=1, keepdims=True)  # renormalise!

# At 100M vectors, 3072-dim float32 = ~1.2 TB
# Truncating to 512 dims cuts storage 6× for a small recall hit
```

**Critical:** Always re-normalise after truncating — cosine similarity requires unit-length vectors.

---

## 4.4 Decision Tree: Picking an Embedding Model

The shortlist is driven by the properties of your dataset and deployment constraints. Here are the 7 branches:

### Branch A — General English prose, zero infrastructure

**Shortlist:** OpenAI `text-embedding-3-small/large`, Voyage `voyage-3-large`, Cohere `embed-v4`, Google `gemini-embedding`

**Why:** You send text, get vectors — no GPU, no model hosting, no version pinning. For most business RAG this is the right first move; the per-token cost is trivial next to an engineer's time.

**Default:** `text-embedding-3-small` (1536-dim) is the cheap safe default; go `-large` (3072-dim) only if the bake-off shows it earns the cost.

---

### Branch B — Data is sensitive / must stay on-prem

**Shortlist:** Open-weight local — `BGE-M3`, `Qwen3-Embedding` (0.6B / 4B / 8B), `NV-Embed-v2`, `multilingual-e5-large`

**Why:** Healthcare, legal, finance, or internal corporate data often *legally cannot* be sent to a third-party API. Open weights run inside your VPC; the only cost is the GPU you already have.

**The size dial:** Qwen3-Embedding ships at 0.6B / 4B / 8B. Bigger = better recall but more VRAM, slower, costlier per embed. Start at the smallest that passes your eval.

---

### Branch C — Multilingual / Indic / Low-resource languages

**Shortlist:** `BGE-M3` (100+ languages), `multilingual-e5-large`, `Qwen3-Embedding`, NVIDIA `Llama-Embed-Nemtron-8B`, `gemini-embedding`

**The critical caveat:** "Supports 100+ languages" ≠ "good at *your* language." Coverage for high-resource languages (Hindi, Arabic) is far better than for low-resource ones (Sanskrit, many African languages).

**Why multilingual matters:** A model trained mostly on English will map two sentences that mean the same thing in Hindi and English to *distant* vectors — cross-lingual retrieval silently fails.

**Rule:** Always verify on your language with a bake-off. If retrieval is weak, fine-tuning a multilingual base on in-domain pairs often beats swapping models.

---

### Branch D — Code, logs, or identifier-heavy technical text

**Shortlist:** Code-specialised embedders (Voyage code, Jina code models) OR a general model + hybrid search (dense + BM25)

**Why:** Code and logs are full of exact tokens — function names, error codes, SKUs — that dense semantic embeddings handle poorly. Two fixes:
1. A model pretrained on code
2. Or lean on BM25 in a hybrid setup to catch the exact tokens

**Decision rule:** If your queries are mostly natural-language questions *about* code, a general embedder + hybrid is usually enough. If you're doing code-to-code or symbol search, use a code-specialised model.

---

### Branch E — Long documents (legal contracts, papers, transcripts)

**Shortlist:** `BGE-M3` (8192 tokens), Jina long-context models, Nomic

**Check first:** The model's `max_seq_length`. Many embedders cap at 512 tokens and **silently truncate** longer input — you embed only the first paragraph and lose the rest.

**Why it's usually a chunking decision anyway:** Even with an 8k-token model, embedding a whole contract into one vector dilutes it. Long-context models buy you *flexibility in chunk size*, not the right to skip chunking.

---

### Branch F — Huge corpus, storage/latency constrained

**Shortlist:** MRL models — `text-embedding-3-*`, Nomic, Gemini — so you can *truncate* dimensions

**The math:**
```
At 100M vectors × 3072 dims × 4 bytes = ~1.2 TB
Truncating to 512 dims (÷6) = ~200 GB — fits in RAM

Only works if model is MRL-trained. Non-MRL truncation shreds the embedding.
```

---

### Branch G — Tiny corpus (< ~500 pages / 200K tokens)

**Consider not doing RAG at all.** If the whole knowledge base fits in the model's context window, just put it in the prompt — no embedder, no index, no retrieval failure modes.

> RAG is the solution for corpora too big for context; below that threshold, it's complexity you don't need.

---

## 4.5 The #1 Silent Production RAG Bug

> *Recall is mediocre. The team swaps to a fancier embedding model. Recall barely improves.*

**The real cause:** A missing query/passage prefix on an asymmetric model.

Models like **E5, BGE, GTE, Qwen3, Instructor, Jina** are trained with explicit prefixes to distinguish query-side from document-side encoding:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/multilingual-e5-large")

# WRONG — same encoding for both. Recall drops, no warning.
q_emb = model.encode(query)
d_emb = model.encode(documents)

# RIGHT — E5/BGE-style models need these exact prefixes:
q_emb = model.encode([f"query: {query}"],        normalize_embeddings=True)
d_emb = model.encode([f"passage: {d}" for d in documents], normalize_embeddings=True)
```

**Why this matters:** Without the prefix, the model treats a question like a document and collapses the asymmetry it was trained to use. Both query and passage vectors are projected into the same undifferentiated space, producing systematically misaligned representations.

**The fix is free:** Read the model card. Add the exact required prefixes. This recovers more recall than most model swaps would have gained — at zero cost.

---

## 4.6 Two Decisions That Make or Break RAG Search

### Decision 1: Distance Metric — Match the Model's Training

**The metric is not a free choice** — it must match how the model was trained.

| Model / Embedding Type | Use This Metric |
|---|---|
| Normalised embeddings (OpenAI, Cohere, most SBERT variants) | Cosine Similarity (or Inner Product) ✅ |
| Unnormalised embeddings (some custom or legacy models) | Euclidean Distance ⚠️ (rarely best for text) |

**The bug:** Using Euclidean distance on a cosine-trained model "because the DB defaulted to it" — silently underperforms, no error message.

**Rule of thumb:** If your embeddings are normalised to unit length, use cosine (or inner product — they're equivalent for unit vectors). Only reach for Euclidean with a strong reason.

### Decision 2: Dense-Only vs Hybrid — Driven by Identifier Density

Ask: how much of your corpus's value lives in exact tokens that dense search can't feel?

| Your Corpus | Recommendation |
|---|---|
| Mostly prose, concepts, natural language (Low Identifier Density) | Dense-only is often fine |
| Codes, IDs, names, jargon, acronyms, exact terms (High Identifier Density) | Hybrid (Dense + BM25) is better |

**Dense-only good for:** Natural-language FAQ, explanatory content, conceptual understanding queries

**Hybrid good for:** Product codes/part numbers, names/acronyms/legal terms, medical jargon, anything where exact token matches carry meaning

> Don't add BM25 complexity to a corpus that has no exact-match queries; don't omit it from one that does.

---

## 4.7 The Unifying Principle

Every embedding choice is decided by a condition — not by leaderboard position:

| Choice | The Condition That Decides It |
|---|---|
| Hosted vs local embedder | Can the data leave your network? Corpus size vs API cost |
| Which multilingual model | Your specific language's coverage — measured, not advertised |
| Code-specialised vs general+hybrid | Are queries about code, or code-to-code? |
| Long-context model vs small chunks | `max_seq_length` vs your chunk strategy |
| Embedding dimensions (MRL) | Storage/latency budget at your vector count |
| Query/passage prefixes | Is the model symmetric or asymmetric? (Read the model card) |
| Distance metric | What the model was trained for |
| Dense vs hybrid | Identifier density in the corpus |
| RAG vs just prompt | Does the corpus fit in context? |

---

## 4.8 Practical Quick-Reference: 2025 Rule of Thumb

| Need | Default Choice |
|---|---|
| Fast prototype, no infra concerns | OpenAI `text-embedding-3-small` |
| Production at scale, cost-sensitive | Self-hosted `bge-large-en-v1.5` or `text-embedding-3-small` |
| On-prem / sensitive data | SBERT family on your GPU |
| Multilingual | `paraphrase-multilingual-MiniLM-L12-v2`, BGE-M3, OpenAI v3 |
| Highest English retrieval quality | `bge-large`, `gte-large`, `text-embedding-3-large`, NV-Embed |
| Code search | Voyage code / Jina code + BM25 hybrid |
| Huge corpus, storage constrained | MRL model + dimension truncation |
| Tiny corpus (<200K tokens) | Just put it in the prompt — no RAG |

---

## 4.9 Learning Thoughts

> **Thought 1:** The asymmetric prefix bug is the single most common production RAG failure that is invisible without knowing to look for it. Always read the model card before assuming the default encode() call is correct. The models that need prefixes (E5, BGE, GTE, Instructor) are also some of the best models — so this matters a lot.

> **Thought 2:** Matryoshka models fundamentally change the economics of large-scale vector search. Being able to cut dimensions from 3072 → 512 (a 6× storage reduction) for a small recall hit changes what's feasible at 100M+ vectors. This is not a minor feature — it's a design-time decision that should influence which model you choose.

> **Thought 3:** "The leaderboard lies" is not hyperbole. MTEB averages over tasks that may not represent your use case. A model that tops the chart on STS (Semantic Textual Similarity) paraphrase tasks may underperform on asymmetric query-document retrieval for your specific domain. Always bake-off on your data.

> **Thought 4:** The hosted vs self-hosted decision is often a legal/compliance decision, not a technical one. If your data is healthcare, legal, or financial, you often *cannot* send it to a third-party API regardless of model quality. Know this constraint upfront — it eliminates half the decision tree immediately.

> **Thought 5:** Branch G is the most overlooked wisdom: consider NOT doing RAG. Engineering teams build RAG systems when the corpus fits comfortably in context because it seems more "production-ready." Often they're just adding failure modes for no benefit.

---

## 4.10 Important Interview Questions

**Conceptual**

1. **Why shouldn't you pick an embedding model from the MTEB leaderboard alone?**
   - MTEB averages over tasks (clustering, classification, STS, retrieval) that may not represent your use case. High average hides domain-specific weaknesses. Always bake-off on your actual queries and documents.

2. **What is Matryoshka Representation Learning? How does it work?**
   - MRL trains a model by grading nested prefixes of the embedding vector simultaneously. The loss is `L(256) + L(512) + L(768) + L(3072)`. Early dimensions receive the most gradient pressure and pack the most coarsely important information. This allows truncating the vector at query time with graceful degradation — critical for large-scale storage.

3. **What is the query/passage prefix bug and why does it cause silent failures?**
   - Asymmetric models (E5, BGE, GTE) are trained with `"query: "` prepended to queries and `"passage: "` prepended to documents. Without the prefix, both are encoded in the same undifferentiated space — the asymmetry the model learned is lost. Recall drops with no error message.

4. **Should you use cosine similarity or Euclidean distance for text embeddings?**
   - Match the metric to how the model was trained. Most modern text embedding models are L2-normalised → use cosine (or inner product, which is equivalent for unit vectors). Euclidean is rarely appropriate for text and quietly underperforms on cosine-trained models.

5. **When does hybrid retrieval (BM25 + dense) outperform dense-only?**
   - When the corpus has high identifier density — product codes, error codes, SKUs, acronyms, exact legal terms. Dense embeddings handle semantic similarity well but miss exact-token matches. BM25 catches exact tokens; dense catches meaning; together they're robust.

**Applied / Design**

6. **Your RAG system processes legal contracts and must stay on-premise. Which embedding model family do you choose and why?**
   - Open-weight self-hosted model (e.g., BGE-M3 or Qwen3-Embedding) because legal data legally cannot leave your infrastructure. Cannot use hosted APIs (OpenAI, Cohere). Choose based on your language requirements and GPU availability.

7. **You have 500M document chunks to index. Embedding dimension is 3072. What problem does this create and how do you solve it?**
   - Storage: 500M × 3072 × 4 bytes = ~6 TB — infeasible for most vector stores. Solution: use an MRL-trained model (text-embedding-3-large, Nomic, Gemini) and truncate to 512 dims. 500M × 512 × 4 bytes = ~1 TB. Run a bake-off to confirm the recall loss is acceptable.

8. **A team embeds queries and documents using the same `model.encode()` call with an E5 model. Recall is poor. What's wrong and how do you fix it?**
   - Missing asymmetric prefixes. E5 requires `"query: {text}"` for queries and `"passage: {text}"` for documents. Without these, both are encoded as if they were passages — the asymmetric encoding the model was trained to use is collapsed. Fix: add the required prefixes as specified in the model card.

9. **At what corpus size should you "consider not doing RAG"?**
   - When the entire knowledge base fits in the model's context window (~200K tokens / ~500 pages for most modern LLMs). Below that threshold, just stuffing the corpus in the prompt is simpler, has no retrieval failures, and produces better answers. RAG is warranted when the corpus is too large for the context window.

---

## 4.11 Section Summary

| Concept | One-line summary |
|---|---|
| Leaderboard misleads | MTEB averages over irrelevant tasks; bake-off on your own data |
| OpenAI text-embedding-3 | Hosted, MRL-trained, easy but costs per-token and sends data to API |
| SBERT family | Open-weight, self-hosted, fine-tuneable, best for privacy/scale |
| MRL | Train on nested prefixes → early dims most informative → can safely truncate |
| 7 decision branches | General prose, on-prem, multilingual, code, long-doc, huge-corpus, tiny-corpus |
| Silent #1 bug | Missing query/passage prefix on asymmetric model — fix is free |
| Distance metric | Must match training — cosine for normalised models, not Euclidean |
| Dense vs hybrid | Driven by identifier density in your corpus |
| Branch G rule | Corpus fits in context? → skip RAG entirely |

---

*Previous: [Section 3 — Dense Embeddings](S3_Dense_Embeddings.md)*
*Next: [Section 5 — Document Processing](S5_Document_Processing.md)*
