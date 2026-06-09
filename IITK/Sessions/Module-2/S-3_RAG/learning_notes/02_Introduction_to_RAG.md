# Section 2: Introduction to RAG

> **Core Idea:** RAG is a three-stage pipeline — Index, Retrieve, Generate — that grounds the LLM's responses in external documents fetched at inference time. Understanding the architecture is the foundation for everything else.

---

## Topic 9: Overview of RAG — The Four-Step Solution

The original RAG paper (Lewis et al., NeurIPS 2020) proposed a clean, composable solution:

```
Step 1: User Query
         ↓
Step 2: Retrieve Relevant Documents  (from external knowledge base)
         ↓
Step 3: Augmented Prompt  (Query + Retrieved Documents)
         ↓
Step 4: LLM Generates a Grounded Response
```

### Why This Works
- The LLM is an excellent **in-context reasoner** — if the answer is in the context window, it will find and use it.
- Retrieval handles **what to know**; the LLM handles **how to express it**.
- You can update the knowledge base without retraining the model — a massive operational advantage.

### Benefits
| Benefit | Explanation |
|---------|-------------|
| **Factual Consistency** | Answer is grounded in retrieved text, not parametric memory |
| **Up-to-Date Information** | Retrieval index can be refreshed continuously |
| **Source Attribution** | Retrieved documents can be shown as citations |
| **Domain Adaptability** | Swap the knowledge base to change the domain, no fine-tuning needed |

---

## Topic 10: RAG Architecture — Retrieval Side vs Generation Side

RAG has two halves separated by a clear boundary:

```
┌─────────────────────────────┬──────────────────────────────┐
│        RETRIEVAL SIDE       │       GENERATION SIDE        │
│                             │                              │
│  User Prompt                │  Augmented Prompt            │
│       │                     │       │                      │
│       ▼                     │       ▼                      │
│  RAG System ──► Knowledge   │      LLM ──► Response        │
│       │         Base        │                              │
│       ▼                     │                              │
│  Augmented Prompt ──────────┼──►                           │
└─────────────────────────────┴──────────────────────────────┘
```

### Components Explained

| Component | Role |
|-----------|------|
| **User Prompt** | The raw natural-language question from the user |
| **RAG System / Retriever** | Searches the knowledge base for relevant documents |
| **Knowledge Base** | An indexed corpus (Wikipedia, company docs, PDFs, databases) |
| **Augmented Prompt** | User question + retrieved passages, formatted as a single prompt |
| **LLM (Parametric Model)** | Reads the augmented prompt and generates the final answer |
| **Response** | The grounded, natural-language answer returned to the user |

### Key Insight: The LLM is Frozen
In the classic RAG setup, the LLM is a **frozen parametric model** — it does not get fine-tuned on each new knowledge base. All adaptation happens via the retrieval step. This makes RAG operationally cheap and flexible.

> **Learning Thought:** The separation of retrieval and generation is what makes RAG so powerful. You can upgrade either side independently: swap a better retriever without changing the LLM, or upgrade the LLM without touching the index.

---

## Topic 11: Naive RAG Workflow

### The Three Stages

```
Indexing  ──►  Retrieval  ──►  Generation
```

#### Stage 1: Indexing (Offline, done once)
- Take your corpus of documents
- Split them into chunks
- Encode chunks as vectors (for dense) or build an inverted index (for sparse)
- Store in a vector database or search index

#### Stage 2: Retrieval (Online, per query)
- Receive user query
- Encode the query in the same representation as the index
- Retrieve top-k most relevant chunks
- Pass chunks to the generation stage

#### Stage 3: Generation (Online, per query)
- Combine the user query and retrieved chunks into an augmented prompt
- Feed to the frozen LLM
- Return the generated response

### How Naive RAG Works — Step by Step

| Step | Who does it | What happens |
|------|-------------|--------------|
| 1 | User | Provides a query (e.g., "What is the refund policy?") |
| 2 | Retriever | Searches the knowledge base, returns relevant documents |
| 3 | RAG system | Formats prompt: `Context: [doc1] [doc2]\nQuestion: [query]\nAnswer:` |
| 4 | LLM | Reads the augmented prompt, generates a grounded answer |

### Benefits of Naive RAG
- **More factual responses** — grounded in retrieved text
- **Access to up-to-date information** — no retraining required

### Limitations of Naive RAG
- No pre-processing of ambiguous or poorly-formed queries
- Retrieval quality depends entirely on the raw user query
- No post-processing to filter out irrelevant retrieved chunks

---

## Topic 12: Advanced RAG — Pre-retrieval and Post-retrieval

Advanced RAG adds two extra stages around the retrieval step:

```
[Pre-Retrieval] ──► Retrieval ──► [Post-Retrieval] ──► Generation
```

### Pre-Retrieval: Query Manipulation

**What:** Transform the raw user query *before* sending it to the retriever.

**Why it matters:**
- Users write conversational, ambiguous, or domain-generic queries
- The retriever needs precise, domain-specific search terms
- Pre-retrieval bridges the vocabulary gap

**What it does:**
- Rewrites the query to match domain-specific terminology
- Adds context to the query from prior conversation turns
- Expands or decomposes the query into sub-queries

**Example:**
```
Raw query:    "When can I get my money back?"
Rewritten:    "What is the online order return and refund window policy?"
```

### Post-Retrieval: Reranking and Filtering

**What:** Process retrieved chunks *before* passing them to the LLM.

**Why it matters:**
- The top-k retrieved chunks may include irrelevant or redundant content
- Feeding noise to the LLM degrades generation quality
- Fewer, better chunks produce better, more focused answers

**What it does:**
1. **Reranking** — Use a cross-encoder or relevance model to re-order results; the most relevant chunk goes first
2. **Filtering** — Remove chunks below a relevance threshold; discard duplicates

### Naive RAG vs Advanced RAG — Side by Side

| Feature | Naive RAG | Advanced RAG |
|---------|-----------|--------------|
| Query processing | Raw user query as-is | Query rewriting/expansion |
| Retrieval | Direct index lookup | Same |
| Post-processing | None | Reranking + filtering |
| Precision | Lower | Higher |
| Complexity | Simple | Moderate |

> **Learning Thought:** Advanced RAG is the production-grade version. Naive RAG is the research prototype. In real deployments, both pre- and post-retrieval steps are almost always needed to hit acceptable accuracy.

---

## Topic 13: Full RAG Example — End-to-End Pipeline

The slides show a concrete end-to-end example split into four quadrants:

### Quadrant 1: Pre-Retrieval (Offline Indexing)
```
Docs in JSON format
       ↓
Section-based chunking
       ↓
Embedding Generation
       ↓
Storing in Vector Database
```

### Quadrant 2: Retrieval (Online)
```
User ──► Query ──► ANN Search in VectorDB ──► Query-relevant Docs with similarity scores
```
*(ANN = Approximate Nearest Neighbour)*

### Quadrant 3: Post-Retrieval
```
Query + Query-relevant Docs ──► Reranker ──► Reranked Documents
```

### Quadrant 4: Augmentation and Generation
```
Query + Reranked Documents ──► Prompt Template ──► Fine-Tuned LLM ──► Answer
```

### The Full Flow in One Sentence
> Raw documents are chunked, embedded, and stored offline; at query time, relevant chunks are retrieved via ANN search, reranked for relevance, assembled with the query into a prompt template, and fed to an LLM that generates the final grounded answer.

---

## Interview Questions — Section 2

### Fundamental

**Q1. What are the three stages of the basic RAG pipeline?**
> **Indexing** (offline): documents are chunked and encoded into a searchable index.
> **Retrieval** (online): the user query is used to fetch the top-k most relevant chunks from the index.
> **Generation** (online): the retrieved chunks and the query are combined into an augmented prompt, which the LLM uses to generate a grounded response.

**Q2. What is the difference between parametric and non-parametric memory in RAG?**
> **Parametric memory** refers to knowledge baked into the LLM's weights during training — fixed and not easily updated. **Non-parametric memory** refers to the external knowledge base that RAG retrieves from — dynamic, updatable, and domain-specific. RAG augments the LLM's parametric memory with non-parametric evidence at inference time.

**Q3. Why is the LLM typically "frozen" in a RAG setup?**
> Because RAG is designed to be a retrieval-augmented inference framework, not a fine-tuning framework. Keeping the LLM frozen means: (a) no expensive retraining when the knowledge base changes, (b) the same general-purpose LLM can serve multiple domains just by swapping the retrieval index, (c) faster iteration and deployment.

### Intermediate

**Q4. What is pre-retrieval in Advanced RAG and why is it necessary?**
> Pre-retrieval transforms the raw user query before sending it to the retriever. It is necessary because users write natural, ambiguous questions that may not match the vocabulary or structure of the indexed documents. Query rewriting, expansion, or decomposition bridges this gap, improving retrieval recall and precision.

**Q5. What is post-retrieval and what two operations does it typically involve?**
> Post-retrieval processes the retrieved chunks before passing them to the LLM. It typically involves: (1) **Reranking** — using a cross-encoder to re-score and reorder the chunks so the most relevant appear first; (2) **Filtering** — removing chunks below a relevance threshold or that are duplicates, to reduce noise in the LLM's context.

**Q6. What is the difference between Naive RAG and Advanced RAG?**
> Naive RAG: raw query → retrieval → LLM generation. No query transformation, no post-processing of results. Advanced RAG adds a pre-retrieval step (query manipulation) and a post-retrieval step (reranking + filtering). Advanced RAG produces higher precision responses at the cost of added latency and complexity.

### Advanced

**Q7. How would you design a RAG system for a domain-specific enterprise chatbot?**
> Key decisions:
> 1. **Knowledge base**: internal docs, PDFs, databases — parse, clean, chunk.
> 2. **Indexing**: sparse (BM25 inverted index) for keyword precision, dense (embeddings) for semantic coverage, or hybrid.
> 3. **Pre-retrieval**: query rewriting to match internal terminology; query expansion for ambiguous queries.
> 4. **Retrieval**: top-k ANN search for dense; BM25 score for sparse; merge with hybrid.
> 5. **Post-retrieval**: cross-encoder reranker to select the best 3–5 chunks.
> 6. **Generation**: augmented prompt with system instructions, retrieved context, and user query.
> 7. **Evaluation**: RAGAS metrics (faithfulness, context precision, answer relevancy).

**Q8. What is the role of the augmented prompt template?**
> The prompt template structures how the retrieved context and user question are presented to the LLM. A well-designed template: (1) instructs the LLM to answer only from the provided context; (2) formats multiple retrieved chunks clearly; (3) adds system-level instructions (e.g., "If the answer is not in the context, say 'I don't know'"); (4) includes the user question at the end. Poor prompt templates lead to context-ignoring or hallucinated responses even when the right chunks were retrieved.

---

## Key Learning Thoughts — Section 2

> **Thought 1 — RAG = Separation of concerns:** The retriever knows *what's relevant*; the LLM knows *how to reason*. Keeping them separate lets you optimise each independently — the most important architectural principle in RAG.

> **Thought 2 — Indexing is offline; retrieval is online:** This distinction matters enormously for latency. You do the expensive work (embedding generation, index building) once offline. At query time, you only do fast lookups. Real-time performance is achievable because the heavy lifting is pre-computed.

> **Thought 3 — Naive RAG is a baseline, not a product:** Naive RAG demonstrates that retrieval-augmented generation works. But in practice, every production system needs pre- and post-retrieval steps. Think of Naive RAG as the proof of concept and Advanced RAG as the implementation.

> **Thought 4 — The frozen LLM is a feature, not a limitation:** Not needing to fine-tune the LLM on every new domain is what makes RAG economically viable. Fine-tuning is expensive and slow; RAG gives you domain adaptation for the price of building an index.

> **Thought 5 — Augmented prompt design is underrated:** Most attention goes to the retriever. But how you format the augmented prompt — how much context, in what order, with what instructions — has an enormous impact on generation quality. The prompt template is a first-class engineering artefact in a RAG system.

---

*Previous: [Section 1 — Motivating RAG](./01_Motivating_RAG.md) | Next: [Section 3 — The Retrieval Phase →](./03_Retrieval_Phase.md)*
