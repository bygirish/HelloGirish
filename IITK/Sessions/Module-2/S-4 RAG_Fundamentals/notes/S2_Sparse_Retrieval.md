# Section 2 — Sparse Retrieval & Its Limits

> **Session:** IIT-KGP Module 2 · Session 4 · Instructor: Pawan Goyal
> **Theme:** *How keyword-based retrieval works, why it was the gold standard for decades, and the exact failure modes that motivated dense embeddings.*

---

## 2.1 What Is Sparse Retrieval?

Sparse retrieval is **keyword-focused search**: a document is relevant if it shares keywords with the query. The word "sparse" refers to the vector representation — a vocabulary of 100,000 words produces 100,000-dimensional vectors where **99%+ of entries are zero** (only words that actually appear in a document get non-zero values).

### The Pipeline

```
Query: "How much vacation do I get?"
    │
    ▼
Tokenise → Build sparse vector (TF-IDF / BM25 weights)
    │
    ▼
Compare against all document sparse vectors
    │
    ▼
Return Top-N most similar documents
```

### Why Was It the Default for Decades?

- **Fast and efficient:** Inverted index allows sub-millisecond lookups even at billions of documents
- **Interpretable:** You can trace exactly which term matched which document and why
- **No GPU needed:** Runs on commodity hardware
- **Hard baseline:** BM25 is notoriously difficult to beat even with modern neural models — especially for exact-token queries

---

## 2.2 TF-IDF — The Foundation

**TF-IDF (Term Frequency – Inverse Document Frequency)** was the dominant retrieval model from the 1970s through the 2010s.

### Term Frequency (TF)

How often does a word appear in a document?

```
TF(t, d) = count(t in d) / total_words(d)
```

If "neural" appears 5 times in a 500-word document: TF = 5/500 = 0.01

### Inverse Document Frequency (IDF)

How unique is this word across the entire corpus?

```
IDF(t) = log( N / df(t) )
```

Where N = total documents, df(t) = documents containing term t.

- "the" appears in every document → IDF ≈ 0 (useless signal)
- "BERT" appears in 100 of 10,000 documents → IDF = log(100) = 4.6 (useful signal)

### TF-IDF Score

```
TF-IDF(t, d) = TF(t, d) × IDF(t)
```

Documents are ranked by the sum of TF-IDF scores across all query terms.

---

## 2.3 BM25 — The Modern Sparse Standard

**BM25 (Best Match 25)** is a probabilistic ranking function from the Okapi BM25 paper (Robertson et al., 1994). It remains the **default ranking function in Elasticsearch, OpenSearch, and Apache Solr** today.

### Why BM25 Beats Naive TF-IDF

BM25 fixes two known flaws of TF-IDF:

1. **Term frequency saturation:** In TF-IDF, a word appearing 100 times gets 100× the score of a word appearing once. BM25 uses a saturation curve — after a term appears several times, additional occurrences add diminishing signal.

2. **Document length normalisation:** Long documents get artificially high TF scores just because they have more words. BM25 normalises by document length.

### BM25 Formula

```
BM25(d, Q) = Σ IDF(qi) × [ f(qi, d) × (k1 + 1) ]
                             / [ f(qi, d) + k1 × (1 - b + b × |d|/avgdl) ]
```

Where:
- `f(qi, d)` = term frequency of query term qi in document d
- `|d|` = document length in words
- `avgdl` = average document length in corpus
- `k1` ∈ [1.2, 2.0] = term frequency saturation parameter (typical: 1.5)
- `b` ∈ [0, 1] = length normalisation parameter (typical: 0.75)

**Intuition:** BM25 answers "how much does this document talk specifically about this query, adjusted for how long the document is and how rare these terms are?"

---

## 2.4 The Four Fundamental Limitations of Sparse Retrieval

These are the exact failure modes that motivated the entire field of dense embeddings.

### Limitation 1: No Semantic Meaning

```
"cat" and "kitten" are two completely unrelated dimensions in TF-IDF.
"king" and "queen" share zero information in TF-IDF.
```

TF-IDF treats every word as an atomic, independent symbol. Synonyms, hyponyms, paraphrases — all invisible.

**Practical failure:**
```
Query:    "How much vacation do I get?"
Document: "Employees accrue 18 days of paid leave annually."

BM25 / TF-IDF: zero shared words → score ≈ 0.  The right answer is invisible.
```

The words "vacation" and "paid leave" are semantically identical to a human but orthogonal symbols to TF-IDF.

### Limitation 2: Curse of Dimensionality

- A vocabulary of 100,000 words = 100,000-dimensional vectors
- For any single document, 99%+ entries are zero
- Storing and computing over these huge sparse matrices is memory-intensive
- Cosine similarity between sparse vectors is dominated by which words happen to appear

### Limitation 3: No Word Order — Bag of Words Assumption

Sparse retrieval treats text as a **bag of words**: order does not matter.

```
"dog bites man"  →  same TF-IDF vector as  "man bites dog"
```

These sentences have completely different meanings but identical sparse representations. Sentence structure, negation, and context are all lost.

### Limitation 4: Out-of-Vocabulary (OOV) Words

New words, proper nouns, technical terms, misspellings, abbreviations — anything not in the training vocabulary gets mapped to a **zero vector = no information**. The model cannot retrieve a document that uses a term it has never seen.

---

## 2.5 Why Sparsity Breaks Similarity — The Mathematical Proof

The dot product between any two one-hot (or sparse) vectors where the same term does not appear is always zero:

```
cat:    [1, 0, 0, 0, 0, 0, 0, 0]
kitten: [0, 0, 0, 0, 1, 0, 0, 0]

cat · kitten = 1×0 + 0×0 + 0×0 + 0×0 + 0×1 + ... = 0
cosine similarity = 0 / (||cat|| × ||kitten||) = 0.00
```

Same calculation for king/queen, happy/joyful, purchase/buy — all give **0.00**.

**The geometric insight:** One-hot vectors are orthogonal by construction. The model can *never* discover that "cat" and "kitten" are related, no matter how much text it has indexed. The representation space has no mechanism to encode semantic proximity.

---

## 2.6 The Vocabulary Mismatch Problem — Why Dense Retrieval Exists

This is the fundamental reason dense retrieval was invented:

```
Sparse retrieval matches STRINGS.
Dense retrieval matches MEANING.
```

| Retrieval Type | How it works | Example |
|---|---|---|
| **BM25 / TF-IDF** | Must share the exact same tokens | "vacation" query → misses "paid leave" documents |
| **Dense embeddings** | "vacation" and "paid leave" sit close in vector space → high similarity score | Retrieves the right document |

**Real-world vocabulary mismatch examples:**

| User Query | Document Uses | BM25 Result |
|---|---|---|
| "How do I terminate an employee?" | "Employee offboarding procedure" | Miss |
| "What's the refund policy?" | "Return and cancellation terms" | Miss |
| "myocardial infarction treatment" | "heart attack therapy" | Miss |
| "Python list append" | "Adding elements to a Python list" | Hit (mostly) |

The last row shows where BM25 still shines: when exact tokens matter (function names, error codes, SKUs, product IDs).

---

## 2.7 When Sparse Retrieval Is Still the Right Choice

Despite its limitations, **don't dismiss BM25**. It is still better than dense in specific conditions:

| Scenario | Why BM25 wins |
|---|---|
| Code / log search (exact function names, error codes) | Dense embeddings handle exact tokens poorly |
| Rare named entities (specific people, places, product SKUs) | Dense models may have never seen the entity |
| Legal text with exact clause references | "Section 7.2(b)" must match exactly |
| Short queries over a large corpus with exact terminology | BM25 retrieves in milliseconds; dense costs an embedding call |
| Explainability requirements | "I matched because the document contains 'fraud' 4 times" |

**The production pattern:** Hybrid retrieval (BM25 + dense) is often the best of both worlds — robust to vocabulary mismatch AND able to match exact tokens.

---

## 2.8 Hybrid Retrieval Preview — Reciprocal Rank Fusion

Since sparse and dense retrieval have complementary strengths, production RAG systems often combine them using **Reciprocal Rank Fusion (RRF)**:

```
RRF_score(d) = Σ_r  1 / (k + rank_r(d))
```

Where:
- `r` iterates over each retriever (BM25, dense)
- `rank_r(d)` = the rank of document d in retriever r's result list
- `k` = 60 (constant, typically)

RRF is robust to score-scale differences between retrievers — it only cares about rank position, not raw scores.

---

## 2.9 Learning Thoughts

> **Thought 1:** The vocabulary mismatch problem is not a failure of engineering — it is a fundamental mathematical property of one-hot representations. No amount of tuning BM25 parameters fixes it. You need a fundamentally different representation (dense embeddings) to solve it.

> **Thought 2:** BM25 is notoriously hard to beat. In practice, the gap between BM25 and dense retrieval shrinks significantly on corpora with consistent terminology. Always run BM25 as your baseline and measure the improvement dense retrieval actually gives you before committing to the added complexity.

> **Thought 3:** The four limitations (no semantics, dimensionality, no word order, OOV) are not independent — they all stem from the same root cause: treating words as independent atomic symbols with no relationship to each other. Dense embeddings fix this by learning a shared geometric space where proximity encodes semantic similarity.

> **Thought 4:** "Sparse retrieval matches strings, dense retrieval matches meaning" — this single sentence is the most concise summary of why the field moved from TF-IDF to neural embeddings. Memorise it; you will use it in every RAG interview.

---

## 2.10 Important Interview Questions

**Conceptual**

1. **Explain TF-IDF. What does each component measure?**
   - TF = how often a term appears in a specific document (term importance in document). IDF = log of N / df(t), how rare the term is across the whole corpus (term uniqueness). Their product scores how specifically a document talks about a term.

2. **What does BM25 improve over TF-IDF?**
   - (1) Term frequency saturation: additional occurrences beyond a threshold add diminishing signal. (2) Document length normalisation: long documents don't get unfairly boosted. Both controlled by parameters k1 and b.

3. **Why does cosine similarity between any two different one-hot vectors always equal zero?**
   - One-hot vectors occupy orthogonal axes. Their dot product is always 0 (no shared nonzero dimensions). Cosine similarity = 0 / (1 × 1) = 0. The entire vocabulary is orthogonal — no semantic relationship is representable.

4. **What are the four limitations of sparse retrieval?**
   - No semantic meaning (synonyms/paraphrases invisible), curse of dimensionality (100K-dim sparse vectors), no word order (bag-of-words), out-of-vocabulary (unseen terms = zero vector).

5. **What is the vocabulary mismatch problem? Give a concrete example.**
   - When the user's query and the correct document use different words for the same concept. E.g., "vacation" in query vs "paid leave" in document — BM25 gives score ≈ 0 because zero shared tokens.

6. **When would you prefer BM25 over dense retrieval?**
   - Code search, log search, exact-token queries (error codes, SKUs, legal clause numbers), rare named entities, explainability requirements.

**Applied / Design**

7. **Your RAG system works well for natural-language FAQ queries but badly for code-related questions. What is likely happening and how do you fix it?**
   - Dense embeddings handle semantic prose well but struggle with exact identifiers (function names, error codes). Fix: add BM25 as a hybrid retriever or use a code-specialized embedding model (Voyage code, Jina code).

8. **What is Reciprocal Rank Fusion and why is it used in hybrid retrieval?**
   - RRF combines rankings from multiple retrievers by summing `1 / (k + rank)` for each retriever. It's used because BM25 and dense scores are on incompatible scales — RRF only cares about rank position, making fusion stable and scale-invariant.

9. **If BM25 is still competitive, why bother with dense embeddings at all?**
   - Dense embeddings handle paraphrase, synonymy, cross-lingual queries, and conceptual questions where exact tokens never appear. For these cases (which are common in conversational and knowledge-base search), BM25 has zero recall regardless of quality.

---

## 2.11 Section Summary

| Concept | One-line summary |
|---|---|
| Sparse retrieval | Match documents by shared keywords; represent text as high-dimensional sparse vectors |
| TF-IDF | Score = term rarity (IDF) × term frequency in document (TF) |
| BM25 | TF-IDF with saturation + length normalisation; the gold standard sparse retriever |
| Limitation 1 | No semantic meaning — synonyms are orthogonal |
| Limitation 2 | Curse of dimensionality — 100K-dim vectors, 99% zeros |
| Limitation 3 | No word order — bag-of-words destroys sentence structure |
| Limitation 4 | OOV — unseen terms → zero vector → no retrieval |
| Core failure | "vacation" ≠ "paid leave" → the vocabulary mismatch problem |
| When BM25 wins | Exact tokens: code, error codes, rare entities, legal terms |
| Hybrid retrieval | BM25 + dense via RRF — best of both worlds |

---

*Previous: [Section 1 — RAG Architecture Recap](S1_RAG_Architecture_Recap.md)*
*Next: [Section 3 — Dense Embeddings](S3_Dense_Embeddings.md)*
