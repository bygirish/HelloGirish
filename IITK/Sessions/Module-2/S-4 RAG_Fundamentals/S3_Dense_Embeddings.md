# Section 3 — Dense Embeddings

> **Session:** IIT-KGP Module 2 · Session 4 · Instructor: Pawan Goyal
> **Theme:** *How dense vector representations work, the journey from Word2Vec to SBERT, and the retrieval architectures built on them.*

---

## 3.1 The Core Idea: Dense Representations

The failure of sparse retrieval (Section 2) boils down to one problem: **words are treated as independent atomic symbols**. The fix is to replace the sparse index with a short, dense vector where **every dimension carries learned meaning**.

### Sparse vs Dense — Side by Side

| Property | Sparse (One-Hot / TF-IDF) | Dense (Word Embedding) |
|---|---|---|
| Vector size | 50,000–500,000 dims | 128–3,072 dims |
| Zero entries | 99%+ zeros | All values active |
| Similarity computation | Dot product of mostly-zeros | Dense dot product (fast) |
| Semantic encoding | None | Geometry encodes meaning |
| Memory | Huge | Compact |

### The Geometry Encodes Meaning

The famous Word2Vec analogies demonstrate this:

```python
vec("king") - vec("man") + vec("woman")  ≈  vec("queen")
vec("Paris") - vec("France") + vec("Germany")  ≈  vec("Berlin")
cosine( vec("cat"), vec("kitten") )  ≈  0.92   ✓
```

These relationships **emerge from training** — the model was never told that king/queen or cat/kitten are related. It learned this purely from co-occurrence patterns in text.

---

## 3.2 What Can Be Embedded?

Embeddings are not limited to words or sentences. The same principle extends to any modality:

| Unit | What it represents | Models | Use in RAG |
|---|---|---|---|
| **Words** | Single word in context | word2vec, GloVe | Context-blind; one vector per word |
| **Sentences** | Full sentence or query | SBERT, E5, GTE | Query and short passage embedding |
| **Chunks** | Paragraph of a document | Long-context embedders | **The primary unit in RAG** |
| **Anything** | Images, audio, code, tables | CLIP (images), CLAP (audio), CodeBERT | Multimodal RAG |

> **Key RAG insight:** The chunk is the unit of retrieval. When you embed a document for RAG, you embed its *chunks*, not the whole document. Each chunk gets one vector; that vector is what gets indexed and searched.

---

## 3.3 From Words to Sentences: The Representational Journey

### Step 1 — Word2Vec / GloVe (2013–2014): Static Word Embeddings

Word2Vec (Mikolov et al., 2013) trains shallow neural networks to predict word context. Each word gets one fixed vector regardless of context.

**Problem — context blindness:**
```
"I went to the river bank"     → vec("bank") = [0.3, -0.2, 0.8, ...]
"I withdrew money from the bank" → vec("bank") = [0.3, -0.2, 0.8, ...]
```

Both sentences produce **the same vector for "bank"** even though they mean different things. This is called a static or non-contextual embedding.

### Step 2 — BERT (2018): Contextual Embeddings

BERT (Devlin et al., 2018) introduced **contextual embeddings** — the vector for each word depends on its entire surrounding context.

```
"I went to the river bank"       → vec("bank") = [0.1, 0.7, -0.3, ...]   # river sense
"I withdrew money from the bank" → vec("bank") = [0.9, -0.1, 0.5, ...]   # finance sense
```

Different context → different vector for the same word. ✓

But BERT was designed for **token-level tasks** (NER, QA span extraction), not sentence-level retrieval.

### Step 3 — Sentence-BERT / SBERT (2019): Retrieval-Oriented Embeddings

SBERT (Reimers & Gurevych, 2019) fine-tuned BERT using a **Siamese network with contrastive loss** to produce sentence-level embeddings optimised for cosine similarity comparison.

This is the inflexion point. SBERT made semantic search practical:
- Fast: embed a query once, compare against millions of stored vectors
- Accurate: similar sentences produce similar vectors by construction

---

## 3.4 How BERT Produces Sentence Embeddings

### The [CLS] Token Approach

BERT prepends a special `[CLS]` (classification) token to every input. After the full transformer forward pass, the hidden state of `[CLS]` at the final layer is used as the sentence representation:

```
Input:  [CLS]  "So"  "long"  "and"  "thanks"  "for"  "all"  [SEP]
         ↓      ↓     ↓       ↓       ↓         ↓      ↓
Output: h_CLS  h_1   h_2    h_3    h_4        h_5    h_6

h_CLS = sentence embedding (768 dims for BERT-base)
```

`[CLS]` attends to every other token through self-attention across all 12 (base) / 24 (large) layers, so by the final layer it has "seen" the full context.

### Mean Pooling Approach

Instead of using only `[CLS]`, take the **average of all token vectors** at the final layer:

```
Sentence Embedding = mean(h_1, h_2, ..., h_N)
```

This is a 768-dimensional vector (for BERT-base). Mean pooling often works as well as or better than `[CLS]` for sentence similarity tasks.

**Process summary:**
```
Text input
    │
    ▼
Tokeniser → token IDs
    │
    ▼
BERT encoder (12 layers, self-attention)
    │
    ▼
Final hidden states: shape (N_tokens, 768)
    │
    ▼
Mean pooling → shape (768,) = sentence embedding
    │
    ▼  [optional]
Linear layer → shape (512,) = compressed embedding
```

---

## 3.5 The Challenge with Raw BERT

**BERT is NOT explicitly trained for cosine similarity.**

BERT was pretrained on Masked Language Modelling (MLM) and Next Sentence Prediction (NSP). Its representations are excellent for understanding language, but:

- Two sentences with similar meanings are **not guaranteed** to produce vectors pointing in similar directions
- The embedding space may be anisotropic (degenerate clustering) — all sentences cluster into a narrow cone

This is why you cannot just take BERT's `[CLS]` token and expect good cosine similarity for retrieval. You need **SBERT** or another retrieval-oriented fine-tuning.

---

## 3.6 Cosine Similarity — The Geometric Measure

Sentence embeddings are compared using **cosine similarity** — the cosine of the angle between two vectors:

```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)
                        = Σ Ai × Bi / (√Σ Ai² × √Σ Bi²)
```

- **Range:** −1 to +1
- **+1:** vectors point in exactly the same direction (identical meaning)
- **0:** vectors are orthogonal (unrelated)
- **−1:** vectors point in opposite directions (opposite meaning)

### Why Cosine, Not Euclidean Distance?

- Cosine is **scale-invariant**: a short and long document about the same topic score the same
- Euclidean distance penalises magnitude differences — a long document about "cats" would be "far" from a short document about "cats" in Euclidean space even though they discuss the same topic
- When embeddings are **L2-normalised** (unit length), cosine similarity = dot product — which is much faster to compute

```python
# Practical example
"How do I reset my password?" ←→ "I forgot my login credentials"  → cosine ≈ 0.78
"How do I reset my password?" ←→ "What's the weather in Tokyo?"   → cosine ≈ 0.05
```

---

## 3.7 Learning Retrieval-Oriented Embeddings — Contrastive Loss

To make embeddings useful for retrieval, we need to **explicitly train** them to satisfy:

> *Similar queries and relevant documents should be close; irrelevant documents should be far away.*

### The Training Objective — Hinge / Contrastive Loss

```
L(θ, q) = Σ_{d_pos ∈ D_pos} Σ_{d_neg ∈ D_neg}  max(0,  s(q, d_neg; θ) - s(q, d_pos; θ))
```

**Intuition:**
- **Pull** relevant documents closer to the query in embedding space
- **Push** irrelevant documents farther away
- The model learns embeddings optimised specifically for **retrieval and ranking tasks**

This is the fundamental insight behind DPR, SBERT, E5, BGE, and essentially every modern retrieval embedding model.

---

## 3.8 How to Get Negative Examples

Training requires positive pairs (query, relevant document) AND negative pairs (query, irrelevant document). Getting good negatives is a critical engineering challenge.

### In-Batch Negatives

The simplest approach: treat all other queries' positive documents in the same training batch as negatives for the current query.

```
Batch of 5 queries × 5 positive documents:
           d1  d2  d3  d4  d5
q1    →    1   0   0   0   0   ← d2-d5 are in-batch negatives for q1
q2    →    0   1   0   0   0
q3    →    0   0   1   0   0
q4    →    0   0   0   1   0
q5    →    0   0   0   0   1
```

**Advantage:** Scales efficiently — no manual negative mining, just use the batch.

**Limitation:** Most in-batch negatives are *easy negatives* — they are clearly irrelevant. Easy negatives provide a weak training signal; the model needs to learn to distinguish hard cases.

### Hard Negative Mining

Hard negatives are documents that **appear relevant but are actually incorrect** — they provide a much stronger training signal.

| Type | Definition | How to get |
|---|---|---|
| **In-batch negatives** | Other documents in the current batch | Free — from batch construction |
| **Hard negatives** | Documents that look relevant but aren't (retrieved by BM25 but not the correct answer) | Use a weak retriever (BM25) to find top-K, then exclude the true positive |

The standard recipe: **in-batch negatives + BM25 hard negatives** → best retrieval models (BGE, E5) are trained this way.

---

## 3.9 Dense Retrieval Architectures

### Architecture 1: Bi-Encoder

The standard architecture for production retrieval.

```
Query ──► BERT_Q ──► z_CLS_Q ──┐
                                 ├──► dot product ──► score(q, d)
Document ──► BERT_D ──► z_CLS_D ──┘
```

**How it works:**
- Encode query and document **independently** using separate (or shared) BERT encoders
- Score = dot product (or cosine) between the two [CLS] vectors
- Documents can be **pre-encoded offline** → fast at query time

**Trade-offs:**
- ✅ **Fast:** Documents are indexed once; query embedding + ANN search = milliseconds
- ✅ **Scalable:** Can index millions/billions of documents
- ❌ **Less accurate:** Query and document never "see" each other during encoding; interactions are approximate

**Examples:** DPR, SBERT, BGE, E5, OpenAI embeddings

### Architecture 2: Cross-Encoder (Single Encoder)

```
z = BERT( query ; [SEP] ; document ) [CLS]
score(q, d) = softmax(U × z)
```

**How it works:**
- Concatenate query and document with [SEP] separator
- Encode **jointly** in a single BERT forward pass
- The model can attend across both query and document tokens simultaneously
- Produces a relevance score via a linear layer over [CLS]

**Trade-offs:**
- ✅ **Highly accurate:** Full cross-attention between query and document
- ❌ **Too slow for retrieval:** Cannot pre-compute document representations; must run full BERT for every (query, document) pair
- ✅ **Perfect for reranking:** Run on a small shortlist (top 100–200 from bi-encoder)

**Production pattern:**
```
Bi-encoder: retrieve top 100-200 candidates  (fast)
    │
Cross-encoder: rerank top 100-200 to get top 5-20  (accurate)
    │
LLM: generate with top 5-20 as context
```

### Architecture 3: ColBERT — Late Interaction (Bonus)

ColBERT is a middle ground between bi-encoder and cross-encoder.

- Encode query and document independently (like bi-encoder) → but keep **all token vectors**, not just [CLS]
- Score = sum of max-similarity of each query token against all document tokens
- **More accurate** than bi-encoder, **faster** than cross-encoder
- Higher storage cost (store all token vectors, not just one per document)

---

## 3.10 Sentence-BERT (SBERT) — Architecture Deep Dive

SBERT uses a **Siamese network** (two identical BERT models sharing weights) trained with a regression or contrastive objective:

```
Sentence A ──► BERT ──► Pooling ──► Sentence Embedding A ──┐
                                                              ├──► Cosine Similarity ──► Loss (MSE vs target)
Sentence B ──► BERT ──► Pooling ──► Sentence Embedding B ──┘
(same architecture, same parameters, same weights)
```

**Training pairs examples:**
- ("My father plays with me at the park", "I play with my dad at the park") → target = 0.9 (highly similar)
- ("The cat sleeps", "The economy is struggling") → target = 0.1 (unrelated)

**Pooling options:**
- Mean pooling (average of all token embeddings) — generally best
- Max pooling (max across each dimension)
- [CLS] token

**Optional:** A linear projection layer can reduce embedding size (768 → 512 dims) for storage efficiency.

---

## 3.11 Common Embedding Models

| Model | Size | Architecture | Embedding Dims | Best For |
|---|---|---|---|---|
| `BAAI/bge-base-en-v1.5` | 110M | BERT | 768 | English retrieval, general purpose |
| `intfloat/e5-base-v2` | 110M | RoBERTa | 768 | English retrieval, needs query/passage prefix |
| `nomic-ai/nomic-embed-text-v1` | ~500M | GPT-style | 768 | Long-context, open-source |
| `sentence-transformers/all-MiniLM-L6-v2` | 22M | MiniLM | 384 | Prototyping, fast, CPU-friendly |
| `BAAI/bge-m3` | ~570M | BERT-based | 1024 | Multilingual (100+ languages) |
| `text-embedding-3-small` | — | OpenAI API | 1536 | Hosted, easy, Matryoshka-trained |
| `text-embedding-3-large` | — | OpenAI API | 3072 | Highest quality, hosted |

### Prototyping Choice: all-MiniLM-L6-v2

MiniLMv2 is a distilled model — it uses **self-attention relation distillation** (a teacher-student framework) to compress a large pretrained transformer into a 22M parameter model while retaining most of its retrieval quality.

```
Speedup: 5.3× vs original
Params: 107M (multilingual variant)
XNLI Acc: 69.3
```

For prototyping: fast, runs on CPU, good enough baseline. For production: upgrade to BGE-large or E5-large.

---

## 3.12 Learning Thoughts

> **Thought 1:** The representational journey (one-hot → Word2Vec → BERT → SBERT) is a story about progressively encoding more context. One-hot encodes identity only; Word2Vec encodes co-occurrence; BERT encodes sentence context; SBERT encodes retrieval relevance. Each step fixes the previous step's blind spot.

> **Thought 2:** The bi-encoder vs cross-encoder trade-off is fundamental to all production RAG. Bi-encoder = scalable but approximate. Cross-encoder = accurate but can't scale to a corpus. The standard production pattern (retrieve with bi-encoder, rerank with cross-encoder) is the best of both worlds.

> **Thought 3:** Contrastive learning is the secret behind all modern retrieval models. You are not training a model to "understand language" — you are training it to make correct documents closer to their queries than incorrect ones in vector space. The geometry is the semantics.

> **Thought 4:** Hard negatives are more important than most practitioners realise. A model trained only on easy negatives (random documents) will fail when two documents are both topically relevant but only one is the true answer. Hard negative mining is what separates mediocre retrieval from excellent retrieval.

> **Thought 5:** Common misconception: "higher cosine similarity = better answer." High cosine similarity means the embedding model thinks texts are similar — but if the model was trained for symmetric similarity (paraphrase detection), it may not align with asymmetric query-document retrieval.

---

## 3.13 Important Interview Questions

**Conceptual**

1. **Why can't we use raw BERT embeddings for retrieval?**
   - BERT is not trained to produce cosine-comparable sentence representations. Similar sentences are not guaranteed to produce vectors pointing in the same direction (the embedding space is anisotropic). SBERT fine-tunes BERT specifically for this with contrastive loss.

2. **What is the difference between a bi-encoder and a cross-encoder?**
   - Bi-encoder: encodes query and document independently → can pre-index documents → fast retrieval at scale. Cross-encoder: encodes query+document jointly → full cross-attention → more accurate but too slow to use as primary retriever; used as a reranker on a small shortlist.

3. **What is contrastive loss and why is it used for embedding training?**
   - Contrastive loss pushes (query, relevant doc) pairs closer and (query, irrelevant doc) pairs farther in embedding space. This explicitly trains the geometry to encode retrieval relevance, unlike MLM pretraining which only models language understanding.

4. **What is mean pooling and why is it used?**
   - Average all token hidden states at the final BERT layer to produce a single fixed-size sentence vector. It often outperforms [CLS] because it distributes information across all tokens rather than depending on one.

5. **What are hard negatives, and why are they important?**
   - Documents that look relevant to a query but are actually incorrect. They provide a stronger training signal than random negatives because they force the model to learn fine-grained distinctions. Models trained with only easy negatives fail on subtle relevance discrimination.

6. **How does Word2Vec differ from BERT? How does BERT differ from SBERT?**
   - Word2Vec: static, one vector per word, no context. BERT: contextual, one vector per word per context, but not optimised for sentence similarity. SBERT: fine-tuned BERT with contrastive loss, produces sentence-level embeddings optimised for cosine similarity.

**Applied / Design**

7. **In a RAG system, should you use a bi-encoder or cross-encoder for initial retrieval? Explain.**
   - Bi-encoder. You cannot run a cross-encoder over millions of documents per query — it requires a forward pass per (query, doc) pair. Bi-encoders pre-index all document embeddings, so retrieval is just a query embedding + ANN search (milliseconds).

8. **What is the ColBERT "late interaction" trick?**
   - Instead of compressing a document to one vector (bi-encoder) or requiring joint encoding (cross-encoder), ColBERT stores all token vectors per document. Score = sum of max-similarity of each query token over all document tokens. More expressive than bi-encoder, more scalable than cross-encoder.

9. **You have a cosine similarity of 0.85 between a query and a document. Is this necessarily good for your RAG pipeline?**
   - Not necessarily. It means the embedding model thinks they are similar, but if the model was trained for paraphrase detection (symmetric similarity), it may not align with asymmetric query→document retrieval relevance. Always use a model trained specifically for retrieval (not paraphrase detection) in RAG.

---

## 3.14 Section Summary

| Concept | One-line summary |
|---|---|
| Dense embedding | Short, dense vector where every dimension carries learned meaning |
| Geometry = semantics | `vec(king) - vec(man) + vec(woman) ≈ vec(queen)` — meaning encoded in geometric relationships |
| BERT [CLS] | Sentence embedding via special classification token attending to full context |
| Mean pooling | Average of all token embeddings — often better than [CLS] for similarity |
| Why not raw BERT | Not trained for cosine similarity — similar sentences may point in different directions |
| Contrastive loss | Pull positives close, push negatives away in embedding space |
| Hard negatives | Documents that look relevant but aren't — essential for sharp retrieval models |
| Bi-encoder | Independent encoding of query and document; fast at scale; less accurate |
| Cross-encoder | Joint encoding of query+document; accurate but not scalable; used as reranker |
| SBERT | Siamese BERT with contrastive fine-tuning — the foundation of modern semantic search |
| Production pattern | Bi-encoder retrieve top-100 → cross-encoder rerank → top-5 to LLM |

---

*Previous: [Section 2 — Sparse Retrieval](S2_Sparse_Retrieval.md)*
*Next: [Section 4 — Choosing an Embedding Model](S4_Choosing_Embedding_Model.md)*
