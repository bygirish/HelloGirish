# Section 3: The Retrieval Phase — Deep Dive

> **Core Idea:** The retrieval phase is where RAG earns its keep. The generation quality is *bounded* by what the retriever surfaces. This section covers every layer of the retrieval pipeline — from raw documents to ranked chunks — including the maths behind TF-IDF and BM25.

---

## Topic 14: Sparse Retrieval — Keyword-Focused Search

### What Is Sparse Retrieval?
Sparse retrieval matches documents to a query based on **shared keywords**. Documents are represented as high-dimensional sparse vectors — most dimensions (vocabulary terms) are zero, and only the terms that appear in the document have non-zero values.

### How It Works
```
Query ──► Sparse Retriever ──► Document [Sparse Vectors] ──► Top-N most similar documents
```
The retriever scores documents by how often and how *uniquely* the query terms appear in them (using algorithms like BM25).

### Strengths
- **Fast and efficient** — inverted index lookups are O(|query terms|), not O(|corpus|)
- **Interpretable** — you can explain exactly *why* a document was retrieved (it contains these terms)
- **No GPU required** — runs on standard search infrastructure (Elasticsearch, Solr, PyTerrier)
- **Excellent precision** for exact-match and keyword-heavy queries

### Weaknesses
- **Vocabulary mismatch** — "cardiac arrest" and "heart attack" are semantically identical but share no tokens; sparse retrieval misses the connection
- **No semantic understanding** — it counts tokens, it doesn't understand meaning
- **May miss paraphrases** — a document that answers the question using different words won't be found

> **Learning Thought:** Sparse retrieval is the workhorse of production search systems. Google, Elasticsearch, and most enterprise search are built on inverted index + BM25. It remains a very strong baseline that is hard to beat with dense methods in keyword-heavy domains.

---

## Topic 15: Dense Retrieval — Embedding-Based Semantic Search

### What Is Dense Retrieval?
Dense retrieval represents both queries and documents as **dense vectors** (embeddings) in a continuous semantic space, using deep learning models like BERT or T5. Documents that *mean* the same thing will have similar vectors, even if they use different words.

### How It Works
```
Query ──► Embedding Model ──► Query Vector
                                   │  cosine similarity
Vector Database ──────────────────►│
(all doc embeddings)               ↓
                          Top-N semantically similar documents
```

### Strengths
- **Semantic understanding** — captures meaning, not just keywords
- **Handles paraphrases and synonyms** — "cardiac arrest" and "heart attack" will have similar embeddings
- **Better for complex, nuanced queries** — questions that require conceptual matching

### Weaknesses
- **Computationally expensive** — requires GPU for embedding generation; ANN index adds infrastructure complexity
- **Less interpretable** — hard to explain why two vectors are similar
- **Requires domain-adapted embeddings** — general-purpose embeddings may underperform on niche domains

### Sparse vs Dense — Quick Comparison

| Dimension | Sparse (BM25) | Dense (Embedding) |
|-----------|--------------|-------------------|
| Representation | Sparse term vectors | Dense continuous vectors |
| Matching | Exact keyword overlap | Semantic similarity |
| Infrastructure | Inverted index (CPU) | Vector DB (GPU) |
| Vocabulary gap | Fails on synonyms | Handles them well |
| Interpretability | High | Low |
| Speed | Very fast | Slower (ANN search) |
| Best for | Keyword queries | Conceptual queries |

> **Learning Thought:** In practice, **hybrid retrieval** (BM25 + dense) outperforms either alone. BM25 catches exact-match precision; dense catches semantic recall. Most production RAG systems combine both via reciprocal rank fusion or a learned reranker.

---

## Topic 16: Augmentation and Generation — Grounding

### What Is Grounding?
**Grounding** means providing the LLM with use-case specific data and knowledge sources that are not part of its original training data. The RAG framework acts as the grounding mechanism.

### How the Generation Step Works
```
Retrieved Context (doc1, doc2, doc3)
         ↓
Prompt Template  ◄── Other instructions (system prompt, tone, format)
         │◄── User Query
         ↓
        LLM
         ↓
   Generated Response
```

### What the LLM Does
1. Reads the entire augmented prompt (context + query + instructions)
2. Identifies the parts of the context that are relevant to the query
3. Synthesises a coherent, contextually appropriate response
4. Leverages its parametric knowledge to write fluently, but prioritises the retrieved context for facts

### Key Principle: Context over Parameters
A well-prompted RAG system instructs the LLM: *"Answer using only the provided context. If the answer is not in the context, say 'I don't know.'"* This is the safety valve against parametric hallucination.

---

## Topic 17: The Retrieval Pipeline — 6 Steps

The complete retrieval pipeline converts raw documents into retrieved, ranked chunks:

```
Raw Documents
     │
     ▼
[1] Document Ingestion ──► [2] Chunking ──► [3] Encoding/Indexing
                                                     │
                                                     ▼
Query ──► [4] Query Processing ──► [5] Retrieval & Ranking ──► [6] Re-ranking (optional)
                                                                     │
                                                                     ▼
                                                              Retrieved Chunks
```

### Step 1: Document Ingestion
- **What:** Load raw documents from various sources
- **Formats:** PDFs, HTML pages, databases, JSON files, Word documents, Markdown
- **Key challenge:** Multi-format parsing — PDFs are notoriously hard to parse reliably; tables and images lose structure

### Step 2: Chunking
- **What:** Split documents into smaller, overlapping segments
- **Why:** LLMs have a limited context window; individual documents can be too long. Smaller chunks also improve retrieval precision.
- **Typical size:** 256–512 tokens with 10–50 token overlap between adjacent chunks
- **Overlap reason:** Prevents sentences from being split across chunk boundaries, preserving context

### Step 3: Encoding / Indexing
- **Dense path:** Each chunk is passed through an embedding model (e.g., sentence-transformers) to produce a fixed-size vector; vectors are stored in a vector database (Pinecone, Weaviate, FAISS)
- **Sparse path:** An inverted index is built mapping each term to the list of chunks containing it, with associated term frequencies

### Step 4: Query Processing
- **Dense:** Encode the user query using the same embedding model as the index (critical — must use identical model and normalisation)
- **Sparse:** Tokenise and normalise the query (stop-word removal, stemming optional)

### Step 5: Retrieval and Ranking
- **Dense:** ANN (Approximate Nearest Neighbour) search over the vector index using cosine similarity; return top-k
- **Sparse:** BM25 score computation over the inverted index; return top-k
- **Hybrid:** Merge results from both using reciprocal rank fusion

### Step 6: Re-ranking (Optional but High-Value)
- **What:** A cross-encoder model scores each (query, chunk) pair jointly and re-orders the top-k
- **Why:** Bi-encoders (used in step 5) embed query and doc independently — fast but less precise. Cross-encoders attend to both together — slower but much more precise.
- **Impact:** Re-ranking typically improves precision@1 by 10–20 percentage points.

> **Learning Thought:** Steps 1–3 are offline (done once). Steps 4–6 are online (done per query). This asymmetry is what makes RAG fast enough for real-time use — the expensive work is pre-computed.

---

## Topic 18: Chunking Strategies

Chunking is more nuanced than it appears — the wrong strategy can destroy the semantic coherence of retrieved passages.

### Strategy 1: Fixed-Size Chunking
```
"Cats are mammals. They sl | eep all day. Dogs are mam | mals too. They love long walks."
```
- **Method:** Sliding window of N tokens, with M% overlap
- **Pros:** Simple, fast, predictable chunk sizes
- **Cons:** May split mid-sentence, breaking semantic units; no awareness of document structure
- **When to use:** Large corpora where simplicity is needed; when documents lack clear structure

### Strategy 2: Sentence / Paragraph Chunking
```
"Cats are mammals." | "They sleep all day." | "Dogs are mammals too." | "They love long walks."
```
- **Method:** Split at natural sentence or paragraph boundaries (using a sentence detector like spaCy)
- **Pros:** Better semantic coherence; chunks are grammatically complete
- **Cons:** Variable chunk sizes; relies on accurate sentence detection
- **When to use:** Structured prose documents (articles, legal texts, reports)

### Strategy 3: Semantic Chunking
```
"Cats are mammals. They sleep all day." | "Dogs are mammals too. They love long walks."
```
- **Method:** Group sentences whose embeddings are semantically similar; create a new chunk when the embedding distance exceeds a threshold
- **Pros:** Thematically coherent chunks; naturally clusters related information
- **Cons:** Requires embedding model at indexing time; slower
- **When to use:** Long documents that mix topics; technical documents with distinct sections

### Strategy 4: Hierarchical / Parent-Child Chunking
```
Parent chunk: "Cats are mammals. They sleep all day. Dogs are mammals too. They love long walks."
    ↓ stores small children for retrieval, but returns parent for context
Child chunks: "Cats are mammals." | "They sleep all day." | "Dogs are mammals too." | "They love long walks."
```
- **Method:** Index small chunks for retrieval precision; return their parent (larger) chunk to the LLM for better context
- **Pros:** Best of both worlds — precise retrieval, rich context for generation
- **Cons:** More complex index structure; requires parent-child relationship tracking
- **When to use:** Production systems where both retrieval precision and generation quality matter

### Chunking Strategy Comparison

| Strategy | Coherence | Size Consistency | Complexity | Best For |
|----------|-----------|-----------------|------------|----------|
| Fixed-size | Low | High | Low | Large corpora, prototyping |
| Sentence/Paragraph | Medium | Medium | Low | Structured prose |
| Semantic | High | Low | Medium | Mixed-topic documents |
| Hierarchical | Very High | Variable | High | Production systems |

---

## Topic 19: Query Formulation Strategies

Even with a perfect index, a poorly formulated query returns irrelevant chunks. Query formulation bridges the gap between what the user *says* and what the retriever *needs*.

### Strategy 1: Naive — Query As-Is
```
User: "How do I reduce my electricity bill?"
Sent to retriever: "How do I reduce my electricity bill?"
```
- **Method:** Send the raw user question directly to the retriever
- **Pros:** Zero latency overhead; simplest to implement
- **Cons:** Conversational language may not match index vocabulary; implicit references ("what about its population?") fail
- **When to use:** Well-formed, self-contained queries

### Strategy 2: HyDE — Hypothetical Document Embeddings
```
User: "How do I reduce my electricity bill?"
LLM generates: "Lower your electricity bill by switching to LED bulbs, unplugging idle devices, 
                using energy-efficient appliances, and adjusting your thermostat."
Sent to retriever: [embedding of that hypothetical answer]
```
- **Method:** Ask the LLM to generate a hypothetical answer first, then retrieve using that answer's embedding
- **Why it works:** The hypothetical answer is in document-space (the same register as indexed documents), so it retrieves better than a question
- **Pros:** Dramatically improves dense retrieval for question-answering
- **Cons:** Adds one LLM call overhead; can hallucinate in the hypothetical answer

### Strategy 3: Multi-Query
```
User: "How do I reduce my electricity bill?"
Generated: "Ways to lower home energy costs"
           "Tips to save on power consumption"
           "How to make appliances more energy efficient"
Sent to retriever: all three queries; union of results
```
- **Method:** Generate multiple paraphrased versions of the query; retrieve for each; merge results
- **Why it works:** Different phrasings match different document styles; union increases recall
- **Pros:** Handles query ambiguity; better recall
- **Cons:** 3–5x retrieval cost; may introduce noise from irrelevant paraphrases

### Strategy 4: Step-Back Prompting
```
User: "How do I reduce my electricity bill?"
Step-back: "What factors affect household electricity usage?"
```
- **Method:** Abstract the query to a higher-level, more general question before retrieval
- **Why it works:** High-level queries often retrieve foundational documents that contain the answer as a sub-point
- **Pros:** Better coverage for complex questions that require background knowledge
- **Cons:** May return overly general documents; requires careful abstraction

---

## Topic 20: The Retriever — Tools and Parametric Models

### Retriever Tools (Search Backends)
| Tool | Type | Notes |
|------|------|-------|
| **BM25 / Lucene** | Sparse | Industry standard; used in Elasticsearch, Solr, PyTerrier |
| **Google Search API** | Sparse + Dense | Web-scale retrieval; used for real-time knowledge |
| **Elser (Elastic)** | Learned sparse | Elastic's neural sparse model; better than BM25 for semantic queries |
| **FAISS** | Dense | Facebook AI's vector similarity library; GPU-accelerated |
| **Pinecone / Weaviate** | Dense | Managed vector databases for production |

### Parametric Models (LLM Generators)
| Model | Provider | Notes |
|-------|----------|-------|
| GPT-4 / GPT-3.5 | OpenAI | Strong reasoning; large context window |
| Claude | Anthropic | Long context window; strong instruction following |
| Gemini | Google | Multimodal capable |
| Llama / Mistral | Meta / Mistral AI | Open-source; deployable on-premise |

> **Learning Thought:** The retriever and the LLM are independently replaceable. This modularity is a key operational advantage of RAG — you can upgrade from BM25 to dense retrieval, or from GPT-3.5 to GPT-4, without redesigning the whole system.

---

## Topic 21: TF-IDF — Term Frequency × Inverse Document Frequency

TF-IDF is the foundational term-weighting scheme in information retrieval and the conceptual predecessor to BM25.

### Intuition
A word is important to a document if it:
1. **Appears frequently in that document** (high term frequency = relevant to the document's topic)
2. **Appears rarely across all documents** (low document frequency = distinctive, not a stop word)

The product of these two factors gives the term's importance score for a document.

### Formulas

```
tf(t, d) = count(t in d) / total terms in d

idf(t) = log(N / df(t))
    where N = total number of documents
          df(t) = number of documents containing term t

tf-idf(t, d) = tf(t, d) × idf(t)
```

### Worked Example (from slides)

Three documents:
- d1: "The cat sat on the mat"
- d2: "The dog sat on the dog"
- d3: "The cat chased the dog"

For document d1:

| Term | TF | df | IDF | TF-IDF |
|------|----|----|-----|--------|
| the | 0.333 | 3 | log(3/3)=0.000 | 0.000 |
| cat | 0.167 | 2 | log(3/2)=0.176 | 0.029 |
| sat | 0.167 | 2 | log(3/2)=0.176 | 0.029 |
| on | 0.167 | 2 | log(3/2)=0.176 | 0.029 |
| mat | 0.167 | 1 | log(3/1)=0.477 | 0.080 |

**Key observations:**
- "the" has TF-IDF = 0 because it appears in all 3 documents — it is a stop word, tells us nothing distinctive
- "mat" has the highest TF-IDF because it only appears in d1 — it's the most distinctive term
- "cat" and "sat" are middle-ground — present in 2 documents, moderately distinctive

### How TF-IDF Is Used for Retrieval
1. Build a TF-IDF vector for every document in the corpus (one dimension per vocabulary term)
2. Represent the query as a TF-IDF vector
3. Compute cosine similarity between the query vector and each document vector
4. Return the top-k documents by similarity score

### Limitations of TF-IDF
- Does not account for document length — a long document with many mentions scores higher than a short, focused one
- IDF is global — doesn't capture local document structure
- Binary bag-of-words — ignores word order and grammar
- **BM25 was designed to fix these limitations**

> **Learning Thought:** TF-IDF is foundational — every interview on information retrieval will test it. Understand the intuition (frequent + distinctive = important) before memorising the formula.

---

## Topic 22: BM25 — The Nonbinary Ranking Model

BM25 (Best Match 25) is the industry-standard sparse retrieval function that improves on TF-IDF by:
1. **Saturating term frequency** — the score doesn't grow linearly with TF; diminishing returns kick in
2. **Normalising for document length** — a term appearing 5 times in a 100-word document is more significant than 5 times in a 1000-word document

### The BM25 Formula (RSV — Retrieval Status Value)

**Step 1: Simple IDF baseline**
```
RSV_d = Σ log(N / df_t)   for each term t in query q
```
This is just the sum of IDF weights for query terms found in document d.

**Step 2: Full BM25 with TF saturation and length normalisation**
```
RSV_d = Σ log(N / df_t) × [(k₁+1) × tf_td] / [k₁((1-b) + b×(L_d/L_ave)) + tf_td]
```

### Parameter Meanings

| Parameter | Meaning |
|-----------|---------|
| `tf_td` | Term frequency of term t in document d |
| `L_d` | Length of document d (in terms) |
| `L_ave` | Average document length in the collection |
| `k₁` | TF saturation parameter (typically 1.2–2.0) — controls how much weight TF gets before saturation |
| `b` | Length normalisation parameter (0–1, typically 0.75) — b=0 means no length normalisation, b=1 means full normalisation |

### Intuition Behind BM25

```
                    TF saturation curve
Score
  │         ●─────────── (BM25: plateaus)
  │      ●
  │   ●
  │●
  └──────────────── TF count
```

Without saturation (raw TF), a document that mentions "machine learning" 100 times would dominate. BM25 caps the benefit after a threshold controlled by k₁.

### BM25 vs TF-IDF

| Feature | TF-IDF | BM25 |
|---------|--------|------|
| TF saturation | No (linear) | Yes (via k₁) |
| Length normalisation | No | Yes (via b) |
| Tunable parameters | None | k₁, b |
| Performance | Good baseline | Better in practice |
| Standard in | Classic IR | Modern IR (Elasticsearch default) |

### Typical BM25 Values
- k₁ = 1.5 (term frequency matters but saturates early)
- b = 0.75 (75% length normalisation)

These are empirically validated defaults across many benchmarks.

> **Learning Thought:** BM25 is the default retrieval algorithm in Elasticsearch, Apache Solr, and PyTerrier. In LLM-era RAG systems, it remains competitive with dense methods for keyword-heavy domains. Understanding why TF saturation and length normalisation matter is the conceptual leap from TF-IDF to BM25.

---

## Interview Questions — Section 3

### Fundamental

**Q1. What is the difference between sparse and dense retrieval?**
> Sparse retrieval matches documents based on exact keyword overlap (e.g., BM25 with an inverted index). Documents are represented as sparse high-dimensional vectors where most values are zero. Dense retrieval uses deep learning models (BERT, T5) to embed both queries and documents into dense vectors, enabling semantic matching even when exact keywords differ.

**Q2. Explain TF-IDF — formula and intuition.**
> `TF(t,d) = count of t in d / total terms in d` — how frequent the term is in this document.
> `IDF(t) = log(N / df(t))` — how rare the term is across all documents; rare terms are more distinctive.
> `TF-IDF = TF × IDF` — a term is important if it's frequent in this document *and* rare overall.
> Intuition: "the" appears everywhere (low IDF → near-zero score); "photosynthesis" is rare and topical (high IDF → high score when it appears).

**Q3. What problem does BM25 solve that TF-IDF doesn't?**
> Two problems: (1) TF-IDF has linear TF growth — BM25 saturates TF so a document mentioning a term 50 times isn't 50× more relevant than one mentioning it once. (2) TF-IDF ignores document length — BM25 normalises scores relative to average document length, so shorter, focused documents aren't penalised vs. long documents that mention a term many times.

### Intermediate

**Q4. What are the four chunking strategies and when would you use each?**
> (1) Fixed-size: fast, simple, use for large corpora or prototyping.
> (2) Sentence/paragraph: better coherence, use for structured prose.
> (3) Semantic: thematic coherence using embedding similarity shifts, use for mixed-topic documents.
> (4) Hierarchical/parent-child: index small chunks for precision, return large parent for context — use in production systems requiring both retrieval precision and generation quality.

**Q5. What is HyDE and why does it improve dense retrieval?**
> HyDE (Hypothetical Document Embeddings) generates a hypothetical answer to the query using the LLM, then uses that answer's embedding as the retrieval query instead of the original question. It works because: questions and documents have different surface forms; the hypothetical answer is already in document-space (same register as indexed documents), producing embeddings closer to the relevant documents.

**Q6. What is the role of the re-ranker in the retrieval pipeline?**
> The re-ranker uses a cross-encoder model to jointly encode each (query, chunk) pair and produce a relevance score. Unlike bi-encoders used in ANN search (which encode query and document independently), the cross-encoder attends to interactions between the two, producing much higher precision scores. Re-ranking improves precision@1 by ~10–20% at the cost of O(k) cross-encoder forward passes.

### Advanced

**Q7. Why would you use hybrid retrieval (BM25 + dense) rather than either alone?**
> Sparse and dense methods have complementary failure modes. BM25 fails on vocabulary mismatch (synonyms, paraphrases) but excels at exact-match keyword precision. Dense retrieval handles synonyms but can miss documents with exact rare terms. Hybrid retrieval (merge via reciprocal rank fusion or a learned combiner) captures both and consistently outperforms either alone on standard benchmarks.

**Q8. Explain the BM25 parameters k₁ and b.**
> `k₁` controls TF saturation. Higher k₁ means term frequency keeps contributing longer before plateauing; lower k₁ means saturation kicks in sooner (approaching binary: did the term appear or not). Typical value: 1.2–2.0.
> `b` controls length normalisation. b=0 means document length is ignored; b=1 means full length normalisation (score is fully adjusted for document length relative to average). Typical value: 0.75. Setting b lower is appropriate for domains where longer documents are inherently more comprehensive (e.g., textbooks).

---

## Key Learning Thoughts — Section 3

> **Thought 1 — Retrieval quality is the ceiling for RAG quality:** No matter how good the LLM is, if the retriever returns the wrong documents, the answer will be wrong or fabricated. This is why the retrieval section is the longest and most technical in the lecture.

> **Thought 2 — TF-IDF → BM25 is a conceptual arc:** TF-IDF establishes the principle (frequent + distinctive). BM25 addresses TF-IDF's two practical failings (linear TF growth, ignoring document length). Understanding the arc helps you understand *why* BM25 is better, not just *that* it is.

> **Thought 3 — Chunking is a first-order design decision:** The most overlooked aspect of RAG engineering. The wrong chunking strategy can destroy retrieval precision even with a perfect embedding model. Hierarchical chunking is almost always worth the complexity in production.

> **Thought 4 — Query formulation is pre-retrieval engineering:** HyDE, multi-query, and step-back prompting are not tricks — they are systematic solutions to the vocabulary-mismatch problem between user questions and indexed documents. Understanding the problem they solve (not just the technique) is key.

> **Thought 5 — Sparse retrieval is not obsolete:** Despite the hype around vector databases, BM25 is still the default in Elasticsearch and remains competitive or superior for keyword-heavy queries, exact-match needs, and domains where interpretability matters. Production RAG systems almost always include sparse retrieval as a component.

---

*Previous: [Section 2 — Introduction to RAG](./02_Introduction_to_RAG.md) | Next: [Section 4 — Common Problems of RAG →](./04_Common_Problems_of_RAG.md)*
