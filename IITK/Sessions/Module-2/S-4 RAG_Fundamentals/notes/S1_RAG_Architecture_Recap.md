# Section 1 — RAG Architecture Recap

> **Session:** IIT-KGP Module 2 · Session 4 · Instructor: Pawan Goyal
> **Theme:** *Why RAG exists, what its architecture looks like, and how Naive vs Advanced RAG differ.*

---

## 1.1 Why Does RAG Exist?

### The Core Problem with Pretrained LLMs

A pretrained LLM is essentially a **compressed, lossy snapshot** of its training corpus. The weights memorise patterns, facts, and language from billions of documents — but that knowledge is **frozen at training time**. This creates four well-documented production limitations:

| Limitation | What Actually Happens |
|---|---|
| **Knowledge cutoff** | The model cannot answer questions about events after its training date. |
| **Hallucination** | When uncertain, the model fabricates plausible-sounding but factually wrong content. |
| **Private / proprietary data** | The model has never seen your company's wiki, contracts, internal codebase, or HR policies. |
| **Context-window finitude** | Even modern LLMs (128K–1M tokens) cannot ingest entire corporate knowledge bases on every query. |

### Two Solutions: Fine-tuning vs RAG

| Approach | How it works | Trade-offs |
|---|---|---|
| **Fine-tuning / continued pretraining** | Bake new knowledge into model weights | Expensive, slow, hard to update, hard to attribute which document sourced a fact |
| **Retrieval-Augmented Generation (RAG)** | At inference time, *retrieve* relevant documents and *augment* the LLM prompt with them | Cheap, fast to update, naturally attributable, scales to billions of documents |

> **Key insight:** RAG decouples **reasoning** (the LLM's job) from **knowledge** (the retrieval layer's job). This is why RAG has become the dominant pattern for knowledge-intensive LLM applications in production.

---

## 1.2 Historical Context — The Journey to RAG

Understanding history prevents you from reinventing the wheel and grounds your intuition.

| Era | Event | Significance |
|---|---|---|
| **1960s–1990s** | TF-IDF, BM25. Classical Information Retrieval (IR). | Sparse, keyword-based. Still highly competitive baselines. |
| **2013** | Word2Vec (Mikolov et al.) | Words become dense vectors; analogies emerge geometrically: `king − man + woman ≈ queen`. |
| **2018** | BERT (Devlin et al.) | Contextual embeddings; "bank" in *river bank* and *Bank of England* now have different vectors. |
| **2019** | Sentence-BERT / SBERT (Reimers & Gurevych) | Siamese-network fine-tuning produces *sentence-level* embeddings cheap enough for retrieval. **This is the inflexion point for semantic search.** |
| **2020** | DPR (Karpukhin et al.) + original RAG paper (Lewis et al.) | Dense retrieval + generation formalised as an end-to-end pattern. |
| **2022–present** | OpenAI `text-embedding-ada-002` → `text-embedding-3` family; BGE, E5, GTE, Nomic; Pinecone, Weaviate, ChromaDB, Qdrant emerge | Commoditised high-quality embeddings via API. Vector databases emerge as a mainstream category. |
| **2023–2025** | Long-context LLMs (128K–1M tokens) | Raises "is RAG still needed?" — consensus is **yes**, because cost, latency, and needle-in-haystack reliability still favour retrieval for large corpora. |

---

## 1.3 Where RAG Shows Up in Industry

RAG is not an academic toy — it is in production across every knowledge-intensive vertical:

| Domain | Example Use Case |
|---|---|
| **Enterprise search** | "Where is the Q3 sales playbook?" answered against internal Confluence/SharePoint |
| **Customer support** | Grounded chatbots that cite product documentation and refuse to invent policies |
| **Coding assistants** | Copilot-style tools that retrieve from a private codebase before answering |
| **Legal & compliance** | Contract review, case-law search; attribution is mandatory |
| **Healthcare & life sciences** | Querying clinical guidelines, drug interaction databases, literature |
| **Financial research** | Earnings calls, 10-K filings, analyst reports indexed for analyst chatbots |
| **Developer documentation** | "Docs assistants" on company websites |

The common thread: **a high-stakes domain where the right answer must be grounded in a specific corpus that the LLM was not trained on.**

---

## 1.4 The RAG Architecture — Component by Component

```
User Prompt
     │
     ▼
┌──────────────┐      ┌──────────────────┐
│  RAG System  │◄────►│  Knowledge Base  │
│  (Retrieval) │      │  (Documents /    │
└──────┬───────┘      │   Vector Store)  │
       │              └──────────────────┘
       ▼
┌──────────────────┐
│  Augmented Prompt│   ← original query + retrieved context
└──────┬───────────┘
       │
       ▼
     LLM
       │
       ▼
   Response
```

### The Two Phases in Detail

**Phase A — Indexing (Offline, runs once)**

1. **Document ingestion** — Load raw files (PDF, DOCX, HTML, scanned)
2. **Text extraction & cleaning** — OCR if needed, strip boilerplate
3. **Chunking** — Split documents into retrievable units
4. **Embedding** — Convert each chunk into a dense vector
5. **Storing** — Persist vectors + metadata in a vector database

**Phase B — Retrieval & Generation (Online, every query)**

1. **Embed the query** — Same embedding model used at index time
2. **ANN search** — Find top-K most similar chunks in vector database
3. **[Optional] Rerank** — Cross-encoder re-scores the candidates
4. **Augment** — Prepend retrieved chunks to the LLM prompt
5. **Generate** — LLM produces a grounded, attributable answer

---

## 1.5 Naive RAG vs Advanced RAG

This is one of the most commonly asked conceptual questions in interviews.

### Naive RAG

```
User Query → Indexing → Retrieval → [Query + Docs] → Frozen LLM → Output
```

Simple pipeline. Works for clean corpora with well-formed queries. **Breaks when:**
- User query is ambiguous or uses domain-specific terminology
- Retrieved documents contain noise or are irrelevant
- Answer spans multiple documents

### Advanced RAG

```
User Query
     │
     ▼
Pre-Retrieval ──► [Query Manipulation: rewrite, expand, decompose]
     │
     ▼
Retrieval ──────► [ANN search in vector store]
     │
     ▼
Post-Retrieval ──► [Reranking, Filtering, Fusion]
     │
     ▼
[Augmented Prompt + Context] → LLM → Output
```

| Stage | What Happens | Benefit |
|---|---|---|
| **Pre-Retrieval** | Query rewriting, HyDE (Hypothetical Document Embedding), query decomposition, query expansion | Bridges vocabulary mismatch; makes ambiguous queries explicit |
| **Retrieval** | Dense, sparse, or hybrid ANN search | Finds candidate documents |
| **Post-Retrieval** | Cross-encoder reranking, MMR diversity filtering, context compression | Rejects irrelevant documents; improves precision |

---

## 1.6 Key Terminology Quick Reference

| Term | Formal Definition | Intuition |
|---|---|---|
| **RAG** | Retrieval-Augmented Generation — pattern of retrieving context and passing it to an LLM | Open-book exam for the LLM |
| **Embedding** | Dense, fixed-dimensional real-valued vector representing text | A "coordinate" in semantic space |
| **Chunk** | A contiguous slice of a document used as the atomic retrieval unit | The "page" your retriever returns |
| **Vector database** | Storage + index system for high-dimensional vectors; primary operation is "find similar, not find equal" | ChromaDB, FAISS, Pinecone, Qdrant, Weaviate |
| **ANN** | Approximate Nearest Neighbour — finds close matches fast, sacrificing a small amount of exact recall | Close enough in milliseconds at billion-vector scale |
| **Bi-encoder** | Architecture where query and document are encoded independently | Encode once, compare cheaply many times |
| **Cross-encoder** | Architecture that takes (query, document) together and outputs a relevance score | Slower but more accurate; used as reranker |
| **Reranker** | Second-stage model that re-scores top-K candidates from the retriever | A "senior reviewer" reading the shortlist more carefully |

---

## 1.7 Learning Thoughts

> **Thought 1:** RAG's architectural elegance is the separation of concerns — the LLM does not need to know facts; it needs to reason. The retriever provides facts. When RAG fails, the failure is almost always in the retrieval layer, not the generation layer.

> **Thought 2:** The history matters. BM25 (1990s) is still a competitive baseline in 2025 for exact-token queries. Understanding why dense retrieval was invented (vocabulary mismatch) prevents you from blindly replacing BM25 when it actually works.

> **Thought 3:** "Sometimes the right RAG decision is no RAG." If your corpus is fewer than ~500 pages / 200K tokens, just put it in the prompt. RAG is the solution for corpora too big for context; below that threshold, it adds complexity you don't need.

> **Thought 4:** The two-phase (offline indexing / online retrieval) split is fundamental. Everything in indexing is a one-time cost; everything in retrieval is a per-query cost. This framing drives all engineering trade-offs in the rest of the course.

---

## 1.8 Important Interview Questions

**Conceptual**

1. **What problem does RAG solve that fine-tuning cannot?**
   - Fine-tuning bakes knowledge into weights — expensive, slow to update, hard to attribute. RAG retrieves from an external store at inference time — cheap to update, naturally attributable, scales to billions of docs.

2. **Explain the two phases of a RAG pipeline.**
   - Offline: ingest → chunk → embed → store in vector DB. Online: embed query → ANN search → (optional rerank) → augment prompt → generate.

3. **What is the difference between Naive RAG and Advanced RAG?**
   - Naive: query → retrieve → generate. Advanced adds a pre-retrieval step (query manipulation) and a post-retrieval step (reranking/filtering).

4. **Why does RAG still make sense even when LLMs have 1M-token context windows?**
   - Cost per token, latency, and needle-in-haystack reliability. Feeding an entire knowledge base into context on every query is prohibitively expensive and slow, and long-context models still degrade on "needle in a haystack" tasks.

5. **What are the four failure modes of pretrained LLMs that RAG addresses?**
   - Knowledge cutoff, hallucination, private/proprietary data blindness, context-window finitude.

**Applied / Design**

6. **You are building a customer support bot for a SaaS product. The company has 2,000 help articles updated weekly. Should you fine-tune or use RAG? Why?**
   - RAG. The knowledge updates weekly — reindexing is trivial. Fine-tuning would require weekly retraining runs. RAG also naturally cites the source article.

7. **What happens to RAG quality if the indexing phase is done poorly?**
   - Every downstream step inherits the damage. The instructor's rule: "80% of bad RAG answers start with poor document processing." Garbage in → garbage out.

8. **In a RAG pipeline, what is the role of the embedding model, and can you swap it after indexing?**
   - The embedding model maps text to vectors. You **cannot** swap it after indexing without re-embedding the entire corpus — query vectors must live in the same space as document vectors.

---

## 1.9 Section Summary

| Concept | One-line summary |
|---|---|
| Why RAG | LLMs are frozen at training; RAG injects live, private, attributable knowledge |
| Architecture | Two phases: offline indexing, online retrieve-augment-generate |
| Naive vs Advanced | Advanced adds pre/post retrieval steps to handle messy queries and noisy results |
| Industry reality | RAG powers enterprise search, support bots, legal AI, coding assistants across every vertical |
| The golden rule | When corpus < 200K tokens, skip RAG and just put it in the prompt |

---

*Next: [Section 2 — Sparse Retrieval & Its Limits](S2_Sparse_Retrieval.md)*
