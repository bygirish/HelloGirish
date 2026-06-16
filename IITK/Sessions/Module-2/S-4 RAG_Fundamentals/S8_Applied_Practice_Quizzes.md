# Section 8 — Applied Practice & Quizzes

> **Session:** IIT-KGP Module 2 · Session 4 · Instructor: Pawan Goyal
> **Theme:** *Test your understanding, diagnose your blind spots, and build the intuition to debug RAG systems in production.*

---

## 8.1 How to Use This Section

This section serves three purposes:

1. **Assess understanding** — work through each quiz without looking at the answer first
2. **Diagnose** — note which topics your reasoning missed; those are your study targets
3. **Transfer** — the design questions mirror real production RAG interviews and on-call debugging

For each quiz: read the question → form your own answer → reveal the explanation.

---

## 8.2 Session Quizzes (from Lecture)

---

### Quiz 1 — The Silent Recall Bug

**Question:**
A team's RAG recall is mediocre. They swap their E5 embedding model for a fancier one and recall barely improves. What is the most likely real cause?

```
a. Their chunks are too large
b. They forgot the query: / passage: prefix the asymmetric model expects
c. They're using Euclidean distance
```

<details>
<summary>Answer & Explanation</summary>

**Answer: b — They forgot the query/passage prefix**

**Why b is correct:**

E5 (and BGE, GTE, Instructor, Jina, Qwen) models are *asymmetric* — they were trained with explicit prefixes to distinguish query-side from document-side encoding:

```python
# WRONG — same encoding for both
q_emb = model.encode(query)
d_emb = model.encode(documents)

# RIGHT — E5/BGE-style models need these exact prefixes:
q_emb = model.encode([f"query: {query}"],     normalize_embeddings=True)
d_emb = model.encode([f"passage: {d}" for d in documents], normalize_embeddings=True)
```

Without the prefix, both query and passage vectors are projected into the same undifferentiated space, producing *systematically misaligned representations*. The model cannot use the asymmetry it was trained on.

**Why a is not the primary cause:**
Chunks being too large would show up as *inconsistent* recall across query types and would improve with a better model's longer context handling — not be immune to a model swap.

**Why c is less likely:**
Euclidean vs cosine matters mainly when embeddings aren't normalised. Many modern pipelines normalise by default. A metric mismatch would also hurt uniformly, not prevent model swaps from helping.

**Key lesson:** The cheapest wins are configuration, not architecture. Before reaching for a better model, verify you're using the current one correctly. The prefix fix recovers more recall than most model swaps — at zero cost.

</details>

---

### Quiz 2 — The Poor PDF Parser

**Question:**
A team builds RAG for **PDFs with tables and multi-column layouts**. Accuracy is poor even though the embedding model is strong. What is the most likely cause?

```
a. The cosine similarity is incorrect
b. The document parsing / chunking step is losing structure
c. The vector store is too small
d. The LLM context window is too large
```

<details>
<summary>Answer & Explanation</summary>

**Answer: b — Document parsing/chunking is losing structure**

**Why b is correct:**

PDFs are notoriously hostile to naive text extraction:
- **Multi-column layouts:** `pypdf` reads coordinates left-to-right across the page, linearising "column 1 line 1, column 2 line 1, column 1 line 2..." — producing interleaved nonsense
- **Tables:** Flattened into a garbled word-salad: "Revenue Q1 100 Q2 120 Total 220..." — all row/column structure destroyed

The embedding model embeds this garbled text → vectors no longer represent the meaningful content → retrieval fails.

**The fix:**
```python
# For multi-column: use layout-aware parser
from unstructured.partition.pdf import partition_pdf
elements = partition_pdf("report.pdf", strategy="hi_res")

# For table-heavy: use pdfplumber
import pdfplumber
with pdfplumber.open("financials.pdf") as pdf:
    for page in pdf.pages:
        tables = page.extract_tables()  # preserves rows/columns
```

**Why a, c, d are incorrect:**
- Cosine similarity being incorrect would hurt all document types, not specifically those with tables and multi-column layouts
- Vector store size doesn't affect accuracy of retrieved content
- LLM context window size doesn't affect retrieval accuracy

**Key lesson:** A strong embedding model cannot compensate for broken input. Retrieval quality is capped by extraction quality — "garbage in, garbage out."

</details>

---

### Quiz 3 — IVF Computation Savings

**Question:**
N = 5,000,000 vectors, `nlist = √N` buckets, `nprobe = 2`. Roughly how many vectors get scanned vs brute force? Then: what does `nprobe = 10` cost?

```
a. 2, 10
b. 6,708 and 24,596  ← (Correct Answer)
c. 2,048 and 8,192
d. 5,000,000 and 500,000
```

<details>
<summary>Answer & Explanation</summary>

**Answer: b — 6,708 and 24,596**

**Working:**

```
nlist = √5,000,000 ≈ 2,236 centroids
Bucket size ≈ 5,000,000 / 2,236 ≈ 2,236 vectors per bucket

nprobe = 2:
  Work = nlist (centroid comparisons) + nprobe × bucket_size
       = 2,236 + 2 × 2,236 = 6,708 vectors

vs brute force: 5,000,000 vectors

Speedup: 5,000,000 / 6,708 ≈ 745×  ✓

nprobe = 10:
  Work = 2,236 + 10 × 2,236 = 24,596 vectors
  Still ~203× faster than brute force
  But significantly higher recall than nprobe=2
```

**Key insight from this calculation:**

The nprobe dial lets you slide along the recall-latency curve:
- `nprobe = 1`: ~745× speedup, lowest recall
- `nprobe = 2`: ~745× speedup, better recall
- `nprobe = 10`: ~203× speedup, much better recall
- `nprobe = 2236`: = brute force (full search)

**The standard production tuning procedure:** Profile at nprobe = 1, 2, 5, 10, 20, 50 → plot recall@10 vs latency → pick the knee of the curve.

</details>

---

### Quiz 4 — PQ Compression Ratio

**Question:**
A 768-dim float32 vector is split into m = 8 sub-vectors, each encoded to one of 256 centroids. What is the size per vector and the compression ratio?

```
a. 8 bytes, 384×  ← (Correct Answer)
b. 64 bytes, 48×
c. 32 bytes, 96×
d. 256 bytes, 12×
```

<details>
<summary>Answer & Explanation</summary>

**Answer: a — 8 bytes, 384×**

**Working:**

```
Sub-vector dimension: 768 / 8 = 96 dims each

Encoding: Each sub-vector maps to 1 of 256 centroids
          log2(256) = 8 bits = 1 byte per sub-vector

Size per vector:
  Original (float32):   768 × 4 bytes = 3,072 bytes
  PQ-compressed:        8 × 1 byte    = 8 bytes

Compression ratio: 3,072 / 8 = 384×
(~99.7% reduction)
```

**The memory implication at scale:**

```
At 100M vectors:
  float32:        100M × 3072 bytes = ~307 GB
  PQ (m=8):      100M × 8 bytes    = ~0.8 GB  ← fits in RAM
```

**Key insight:** PQ is a *lossy* compression — distances are approximated, not exact. The quality trade-off depends on m and nbits:
- Higher m (more sub-vectors) = higher quality but more memory
- Higher nbits (more centroids) = higher quality but more memory
- The bake-off determines the right m and nbits for your recall target

</details>

---

## 8.3 Extended Practice — Design Scenarios

These go beyond multiple-choice to mirror real production debugging and architecture decisions.

---

### Design Scenario 1 — Diagnosing RAG Failure

**Scenario:**
Your RAG system was working well. After adding 50,000 new documents (scanned PDFs of historical contracts), users report that answers about contract clauses are consistently wrong. The embedding model and vector store were not changed.

**Diagnose and fix:**

<details>
<summary>Analysis & Solution</summary>

**Root Cause Identification:**

The new documents are *scanned PDFs* — they have no text layer. `pypdf` returns empty strings silently. You indexed 50,000 empty or near-empty chunks. Retrieval for contract queries now surfaces these empty chunks (low signal vectors that may accidentally match).

**Verification:**

```python
def pdf_has_text_layer(path, sample=3):
    import pypdf
    r = pypdf.PdfReader(path)
    return any((p.extract_text() or "").strip() for p in r.pages[:sample])

# Probe all new PDFs
problem_files = [f for f in new_files if not pdf_has_text_layer(f)]
print(f"{len(problem_files)} scanned PDFs found with no text layer")
```

**Fix:**

```python
def load_document(path):
    if not pdf_has_text_layer(path):
        # Route to OCR
        from pdf2image import convert_from_path
        import pytesseract
        pages = convert_from_path(path, dpi=300)
        return "\n\n".join(pytesseract.image_to_string(p) for p in pages)
    else:
        import pypdf
        reader = pypdf.PdfReader(path)
        return "\n".join(p.extract_text() for p in reader.pages)
```

**Prevention:** Add a validation step after extraction that alerts if a document returns <10 non-whitespace characters. Never silently index empty chunks.

</details>

---

### Design Scenario 2 — Scaling from 100K to 100M Chunks

**Scenario:**
Your RAG system currently uses ChromaDB with HNSW and handles 100K chunks with excellent recall (recall@10 = 0.97) and 15ms latency. The business is adding data and you're projected to reach 100M chunks in 6 months. Your embedding model produces 768-dim float32 vectors. You have 128 GB RAM available.

**Design the migration path:**

<details>
<summary>Analysis & Solution</summary>

**Current state analysis:**
```
100K × 768 × 4 bytes = 0.3 GB — trivially fits in RAM
HNSW at 100K: excellent recall, low latency — correct choice
```

**Projected state:**
```
100M × 768 × 4 bytes = 307 GB — exceeds 128 GB RAM
→ Cannot use HNSW (graph must fit in RAM)
→ Must use IVF+PQ for memory compression
```

**Migration design:**

```python
import faiss, numpy as np

d = 768
m = 16           # PQ sub-vectors (tune for quality)
nbits = 8        # 256 centroids per sub-space

# Memory calculation:
# 100M × 16 bytes (m=16) = 1.6 GB — fits easily in 128 GB

nlist = 10000    # ≈ √100M = 10,000 clusters

quantizer = faiss.IndexFlatIP(d)
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)

# Training requires a sample (100K vectors is enough for 10K clusters)
index.train(sample_vecs.astype(np.float32))

# Add all 100M vectors in batches
for batch in batches(all_vecs, batch_size=100_000):
    index.add(batch.astype(np.float32))

# Tune nprobe via bake-off to meet your recall target
index.nprobe = 20    # start here, profile against your EVALSET
```

**Quality validation:**
- Measure recall@10 against your EVALSET after migration
- Tune `m` (8 vs 16 vs 32) and `nprobe` (5, 10, 20, 50) to find the recall/latency knee
- Target recall@10 > 0.90 at latency < 50ms

**Alternative:** If recall degrades too much with PQ, consider MRL truncation (if your embedding model supports it) to reduce dimensions from 768 → 256, enabling HNSW at `100M × 256 × 4 = 100 GB` — marginal but possibly under 128 GB.

</details>

---

### Design Scenario 3 — Multilingual RAG for Indic Languages

**Scenario:**
You are building a RAG system for a government healthcare portal in India. Documents are in Hindi, English, and occasionally Marathi. Queries come in Hindi and English mixed. The system must run on-premise (data cannot leave government servers). What is your full architecture?

<details>
<summary>Architecture Design</summary>

**Constraint mapping:**
- On-premise → No hosted APIs (no OpenAI, Cohere, Voyage)
- Multilingual (Hindi, English, Marathi) → Multilingual embedding model
- Healthcare → Accuracy and citation critical

**Embedding model choice:**
```
→ BGE-M3 (BAAI/bge-m3)
  - 100+ languages including Hindi and Marathi
  - Open-weight, self-hosted (runs on your GPU)
  - 8192 token context (handles long Hindi documents)
  - 1024-dim embeddings

Note: Always verify on your specific language with a bake-off
"supports 100+ languages" ≠ "good at Hindi medical terminology"
```

**Document processing (government PDFs):**
```
1. Probe for text layer first
2. Hindi scanned PDFs → Tesseract with Hindi language pack (hin)
3. Multi-column layouts → Docling or Unstructured
4. Tables (drug dosage tables) → pdfplumber → Markdown
```

**Chunking:**
```
- Government health circulars: Structure-aware on numbered clauses
- Long policy documents: Sliding window 512 tokens, 10% overlap (Hindi tokenises differently)
- Always count tokens with BGE-M3's tokeniser, not character count
```

**Vector store:**
```
- FAISS (on-premise, no managed SaaS)
- Start: IndexFlatIP (exact search) for < 1M chunks
- Scale: IndexIVFFlat or IndexIVFPQ when > 1M chunks
- Metadata: Store language, document_type, date for filtering
```

**Retrieval:**
```
Dense retrieval (BGE-M3) + BM25 (for exact medical terms, drug names, code numbers)
→ Hybrid via RRF
→ Cross-encoder reranker (BGE-reranker-m3) for top-20 to top-5
```

**Critical bake-off step:**
```
Build Hindi + English EVALSET with 200 question-answer pairs
Test BGE-M3 vs multilingual-e5-large vs Qwen3-Embedding-0.6B
Pick the model with highest Hit@10 on YOUR queries
Fine-tune on in-domain Hindi medical pairs if recall < 0.80
```

</details>

---

### Design Scenario 4 — The Chunk Size Experiment

**Scenario:**
You are evaluating chunking strategies for a technical documentation corpus (API docs, tutorials, how-to guides). You have an evaluation set of 500 question-answer pairs where you know the ground-truth passage for each question. Design the experiment to pick the best chunking strategy.

<details>
<summary>Experiment Design</summary>

**Metrics:**
```python
def hit_at_k(retrieved_chunks, relevant_chunk_id, k=10):
    """Did the relevant chunk appear in top-K?"""
    return relevant_chunk_id in retrieved_chunks[:k]

def mrr(retrieved_chunks, relevant_chunk_id):
    """1 / rank of first relevant chunk"""
    try:
        rank = retrieved_chunks.index(relevant_chunk_id) + 1
        return 1.0 / rank
    except ValueError:
        return 0.0
```

**Configurations to test:**
```python
chunking_configs = {
    "fixed_256":      TokenTextSplitter(chunk_size=256, chunk_overlap=0),
    "fixed_512":      TokenTextSplitter(chunk_size=512, chunk_overlap=0),
    "sliding_512_20": TokenTextSplitter(chunk_size=512, chunk_overlap=102),
    "recursive_600":  RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120),
    "structure_aware": split_by_markdown_heading,
    "semantic_p90":   SemanticChunker(..., breakpoint_threshold_amount=90),
}
```

**The sweep:**
```python
results = []
for name, splitter in chunking_configs.items():
    chunks = chunk_corpus(splitter, name)
    index, vecs = embed_and_index(chunks)            # use SAME embedding model
    
    hits, mrr_scores = [], []
    for q, answer_id in EVALSET:
        retrieved = retrieve(index, vecs, q, k=10)
        hits.append(hit_at_k(retrieved, answer_id, k=10))
        mrr_scores.append(mrr(retrieved, answer_id))
    
    results.append({
        "strategy": name,
        "hit@10": np.mean(hits),
        "mrr": np.mean(mrr_scores),
        "n_chunks": len(chunks),
        "avg_chunk_tokens": np.mean([count_tokens(c) for c in chunks])
    })

# Sort by hit@10
results_df = pd.DataFrame(results).sort_values("hit@10", ascending=False)
```

**Interpretation guidance:**
- If `recursive_600` wins over `fixed_512`: natural boundary preference is real for this corpus → use recursive
- If `sliding_512_20` wins over `fixed_512`: boundary facts are being cut → overlap is helping
- If `fixed_256` wins over `fixed_512`: smaller chunks are better (technical docs have short, dense facts)
- If `semantic` does not win despite its cost: do not pay 14× slower ingest for no recall gain

**Freeze the winner and reuse for all downstream experiments** (embedding model comparison, vector store comparison) — change only one variable at a time.

</details>

---

## 8.4 Conceptual Check — True / False

Work through these quickly to identify gaps:

| Statement | True / False |
|---|---|
| BM25 can find the document "Employees get 18 days of paid leave" for the query "vacation days" | **False** — zero shared tokens; score ≈ 0 |
| You can swap an embedding model after indexing without re-embedding | **False** — query and document vectors must live in the same space |
| Cosine similarity between two identical vectors is 1.0 | **True** — angle = 0, cos(0) = 1 |
| IVF reduces memory usage by compressing vectors | **False** — IVF reduces search time; PQ compresses vectors |
| PQ+IVF combine to solve both "too many vectors" and "vectors too big" | **True** — IVF = search scope; PQ = memory |
| HNSW requires storing the full index in RAM | **True** — the graph must be RAM-resident |
| "all-MiniLM-L6-v2" requires query/passage prefixes | **False** — symmetric model, no prefixes needed |
| Semantic chunking is ~14× slower than token splitting | **True** — embeds every sentence at ingest |
| You can truncate a non-MRL model's embedding to fewer dimensions | **False** (effectively) — truncating a non-MRL model shreds the embedding |
| ChromaDB's default index is HNSW | **True** |
| BM25 is usually a hard baseline to beat for exact-term queries | **True** |
| Overlap always improves retrieval recall | **False** — a Jan 2026 study found no measurable benefit in their setup |

---

## 8.5 End-to-End System Design Interview — Full Walkthrough

**Question (common in senior ML/AI engineer interviews):**

> "Design a RAG system for a legal firm's document search. They have 200,000 legal documents (mix of PDFs and Word files), updated monthly. Queries come from lawyers in natural language. Response must cite specific clause numbers. System must run on-prem. Budget latency < 200ms at P95."

**Structured Answer:**

<details>
<summary>Full Design</summary>

**1. Document Processing**

```
PDFs:
  - Plain PDFs (briefs, opinions) → pypdf
  - Table-heavy (financial exhibits, spreadsheets) → pdfplumber → Markdown tables
  - Scanned PDFs → probe first → OCR (tesseract, 300 DPI)
  - Multi-column (academic papers) → Unstructured with hi_res strategy

DOCX: python-docx → extract paragraphs + heading hierarchy

Clean: strip headers/footers, page numbers, "CONFIDENTIAL" watermarks
```

**2. Chunking**

```
Strategy: Structure-aware on legal clause boundaries

Legal docs have numbered clauses (§7.2, Article III, Section 4(b))
→ Split on clause markers using regex
→ Each chunk = one complete clause
→ Store metadata: {doc_id, clause_ref, clause_type, date}

Fallback for unstructured sections: recursive 600 tokens, 120 overlap
```

**3. Embedding Model**

```
On-prem → open-weight model required (no OpenAI API)
English only → BGE-large-en-v1.5 (top English retrieval quality)
Legal terminology → run bake-off on 200 legal Q&A pairs
Add query/passage prefixes (BGE requires them)
```

**4. Vector Store**

```
200K × 768-dim = 0.6 GB → easily fits in RAM
→ FAISS IndexHNSWFlat (d=768, M=32)
  - Best recall-per-ms when RAM allows
  - Recall@10 > 0.95 easily achievable
  - Latency << 200ms

Metadata store: PostgreSQL with (chunk_id, doc_id, clause_ref, text, date)
→ Post-retrieve: join on chunk_id to get clause_ref for citation
```

**5. Retrieval & Reranking**

```
Hybrid retrieval (Dense + BM25):
  - Legal jargon, case citations, statute numbers → BM25 catches exact tokens
  - Conceptual queries → dense catches meaning
  - Combine via RRF (k=60)

Reranking:
  - Retrieve top-50 from hybrid
  - Cross-encoder rerank (BGE-reranker-large) → top-5
  - Latency budget: embedding ~5ms, ANN ~1ms, reranking ~30ms total = well under 200ms
```

**6. Generation**

```
Pass top-5 chunks to LLM with:
  - Clause references from metadata
  - Source document name and date
  - Prompt: "Answer using only the passages below. Cite each clause reference used."
```

**7. Evaluation & Monitoring**

```
Offline: Hit@5, MRR on 200-item EVALSET
Online: User feedback (thumbs up/down), citation accuracy tracking
Monthly: Re-evaluate after new document ingestion
```

</details>

---

## 8.6 Common Mistakes and How to Avoid Them

| Mistake | Why It Fails | Fix |
|---|---|---|
| Using `pypdf` on scanned PDFs | Returns empty strings silently → empty chunks indexed | Probe text layer first; route to OCR |
| Missing query/passage prefixes on asymmetric models | Recall drops silently | Read model card; always add required prefixes |
| Using Euclidean distance with a cosine-trained model | Scores are wrong; recall drops | Match distance metric to model's training |
| Never testing BM25 as baseline | May waste months building dense retrieval that doesn't improve on BM25 | Always run BM25 first; measure the gap |
| Using character count for chunk size | Different from token count; may exceed model's token limit | Use the embedding model's own tokeniser |
| Truncating a non-MRL model's embedding | Shreds the embedding; recall collapses | Only truncate MRL-trained models; renormalise after |
| Defaulting to 20% overlap without measuring | May increase storage/cost with no recall benefit | Bake-off with overlap=0 vs overlap=20%; measure Hit@K |
| Choosing embedding model from MTEB leaderboard alone | High MTEB average ≠ good on your domain | Bake-off on your own queries and documents |
| Skipping chunking "because we have a long-context model" | Whole document → one diluted vector; poor retrieval precision | Chunk even with long-context models; they buy chunk-size flexibility, not chunking elimination |
| Building RAG when corpus fits in context | Adds failure modes (retrieval, chunking, embedding) for no benefit | If corpus < ~200K tokens, just use it as context |

---

## 8.7 The 10 Most Important Interview Concepts — Quick Reference

| # | Concept | One-sentence answer |
|---|---|---|
| 1 | What is RAG and why does it exist? | Retrieve relevant docs at inference time to augment the LLM prompt — solves knowledge cutoff, hallucination, private data, context limits |
| 2 | Sparse vs dense retrieval | Sparse matches strings (BM25); dense matches meaning (embeddings) — vocabulary mismatch is the gap |
| 3 | Bi-encoder vs cross-encoder | Bi: independent encoding, scalable, less accurate; Cross: joint encoding, accurate, can't scale — bi retrieves, cross reranks |
| 4 | Why not raw BERT for retrieval? | Not trained for cosine similarity; similar sentences not guaranteed to point in similar directions — need SBERT/contrastive fine-tuning |
| 5 | Contrastive loss | Pull positive (query, relevant doc) pairs close, push negative pairs far — trains geometry to encode retrieval relevance |
| 6 | Query/passage prefix | Asymmetric models (E5, BGE, GTE) need different prefixes for query vs document encoding — missing prefix = silent recall drop |
| 7 | IVF | Cluster vectors into Voronoi cells; at query time search only the nprobe nearest cells — reduces O(N) to O(k + nprobe × N/k) |
| 8 | PQ | Split vector into m sub-vectors; compress each to a centroid ID (1 byte) — 384× memory reduction for 768-dim vectors with m=8 |
| 9 | HNSW | Multi-layer graph index; best recall-per-ms; requires full index in RAM — default in ChromaDB, Qdrant, Weaviate |
| 10 | Document processing > model selection | 80% of bad RAG answers trace to poor document processing; fix the foundation before optimising the model |

---

## 8.8 Learning Thoughts

> **Thought 1:** The four quiz questions from the lecture are not random — they are the four most important debugging scenarios in production RAG: silent model misconfiguration (Q1), broken document parsing (Q2), ANN search efficiency calculation (Q3), and memory compression arithmetic (Q4). Master these four and you understand 80% of RAG production failures.

> **Thought 2:** The design scenarios reveal a consistent pattern: most RAG failures are NOT in the ML model. They are in document processing (empty OCR, scrambled tables), configuration (wrong prefix, wrong distance metric), or chunking (too large, wrong strategy). The discipline is to eliminate these first before concluding you need a better model.

> **Thought 3:** The True/False exercise is deliberately tricky. Read each statement carefully. The most commonly missed: "You can swap an embedding model after indexing without re-embedding" (False — you must re-embed everything) and "IVF reduces memory" (False — that's PQ). These confusions come up in interviews regularly.

> **Thought 4:** The full system design (legal firm) integrates all 7 sections. Notice the decision flow: data constraints first (on-prem → open-weight models), then format (PDF types → parsers), then structure (clause boundaries → structure-aware chunking), then scale (200K chunks → HNSW is fine), then retrieval quality (legal jargon → hybrid), then generation (citations from metadata). Every choice is driven by a condition — the unifying principle of the course.

---

## 8.9 Self-Assessment Rubric

Use this after completing all 8 sections:

| Topic | Beginner | Intermediate | Expert |
|---|---|---|---|
| RAG Architecture | Can draw the pipeline | Can explain naive vs advanced RAG trade-offs | Can design for real constraints (on-prem, multilingual, huge scale) |
| Sparse Retrieval | Know TF-IDF exists | Can explain BM25 formula and 4 limitations | Know when BM25 beats dense and how to combine via RRF |
| Dense Embeddings | Know embeddings are vectors | Understand bi-encoder vs cross-encoder | Can debug contrastive training issues, understand hard negatives |
| Embedding Model Selection | Know OpenAI and SBERT exist | Can apply the 7-branch decision tree | Can design a bake-off protocol and diagnose prefix/metric bugs |
| Document Processing | Know PDFs are hard | Know which tool for which format | Can build a multi-format router with OCR fallback and validation |
| Chunking | Know what a chunk is | Can choose fixed/sliding/structure-aware | Can design a bake-off, explain the overlap myth, pick semantic only when justified |
| Vector Stores & ANN | Know ChromaDB and FAISS exist | Can explain IVF phases and PQ compression | Can calculate IVF work savings, PQ memory, design a migration from 100K to 100M |
| Applied Practice | Can answer quiz questions | Can diagnose production failures | Can design full end-to-end RAG systems under real constraints |

---

*Previous: [Section 7 — Vector Stores & ANN Indexing](S7_Vector_Stores_ANN_Indexing.md)*
*Back to start: [Section 1 — RAG Architecture Recap](S1_RAG_Architecture_Recap.md)*
