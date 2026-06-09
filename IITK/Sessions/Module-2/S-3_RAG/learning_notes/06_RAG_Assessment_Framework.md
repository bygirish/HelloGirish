# Section 6: RAG Assessment Framework

> **Core Idea:** Building a RAG system is not enough — you must measure whether it actually works. Without evaluation, silent failures (hallucinations, missed retrievals, attribution errors) go undetected until they cause real harm. RAGAS and TruLens are the two leading frameworks for systematically measuring RAG pipeline quality.

---

## Topic 33: Why RAG Needs Its Own Evaluation Framework

### The Evaluation Gap
Standard NLP evaluation (BLEU, ROUGE, accuracy) measures the final answer against a ground truth. But RAG has a **pipeline** with multiple stages, each of which can fail independently. A correct final answer doesn't tell you *why* it was correct — or whether the system will remain correct on the next query.

### The Three Failure Points RAG Evaluation Must Cover

| Failure Point | What to Measure |
|---------------|----------------|
| **Retrieval issues** | Were the right documents retrieved? Were key documents missed? |
| **Hallucination risk** | Does the generated answer stay within what the retrieved context actually says? |
| **Context utilisation** | Did the LLM effectively use the retrieved information, or ignore/misinterpret it? |

> Without evaluation, these failures lead to **inaccurate or fabricated responses** — a critical risk in domains like pharmaceutical research, legal advice, or financial services where accuracy is paramount.

### Two Leading Evaluation Frameworks

| Framework | Type | Focus |
|-----------|------|-------|
| **RAGAS** | Metric-based, open-source | Four quantitative metrics covering the full RAG pipeline |
| **TruLens** | Observability platform | Feedback functions, tracing, continuous monitoring |

---

## Topic 34: RAGAS — Retrieval-Augmented Generation Assessment Suite

### What Is RAGAS?
RAGAS is an **open-source evaluation framework** designed specifically for RAG systems. It measures how well a system:
- Retrieves relevant context
- Utilises that context during generation
- Produces accurate and faithful responses

RAGAS evaluates the **entire RAG pipeline** — from context retrieval through to answer generation — not just the final output.

### The RAGAS Evaluation Dataset Format
RAGAS requires a structured evaluation dataset with four fields:

| Field | What it represents | Example |
|-------|--------------------|---------|
| `question` | The user's query | "Who wrote Romeo and Juliet?" |
| `contexts` | Chunks retrieved by the RAG system | "Romeo and Juliet is a tragedy written by William Shakespeare." |
| `answer` | The LLM's generated response | "Romeo and Juliet was written by William Shakespeare in 1597." |
| `ground_truth` | Reference answer (optional but recommended) | "William Shakespeare wrote Romeo and Juliet." |

> **Note from slides:** The answer adds "in 1597" which is NOT in the retrieved context. That's a hallucinated detail — even if it's historically plausible. RAGAS is designed to catch exactly this.

---

## Topic 35: RAGAS Metrics — The Four Pillars

### Metric 1: Context Precision
**What it measures:** How relevant are the retrieved chunks to the query?

**Intuition:** If you retrieved 5 chunks and 4 of them are irrelevant to the question, you have low context precision. High precision means every retrieved chunk is genuinely useful.

**Formula:**
```
Context Precision = (Number of relevant retrieved chunks) / (Total retrieved chunks)
```

**Why it matters:** Low context precision leads to context overload (Section 4 failure mode) and distracts the LLM.

**Example (Romeo & Juliet):**
- Retrieved 1 chunk: "Romeo and Juliet is a tragedy written by William Shakespeare."
- This chunk is fully on-topic for the question "Who wrote Romeo and Juliet?"
- **Score: ~1.0** ← The one retrieved chunk is fully relevant.

---

### Metric 2: Context Recall
**What it measures:** Did the retrieval system find all the information needed to answer the question correctly?

**Intuition:** If the ground truth answer requires three facts, and your retrieval only found evidence for two of them, you have low context recall. You might get a partial or wrong answer.

**Formula:**
```
Context Recall = (Number of ground-truth claims supported by context) / (Total ground-truth claims)
```

**Why it matters:** Low context recall leads to incomplete retrieval failure (Section 4) — the answer is wrong or incomplete because critical information was never retrieved.

**Example (Romeo & Juliet):**
- Ground truth: "William Shakespeare wrote Romeo and Juliet."
- Context contains: "Romeo and Juliet is a tragedy written by William Shakespeare."
- The context contains everything the ground truth needs (the author).
- **Score: ~1.0** ← Context has complete recall of what's needed.

---

### Metric 3: Answer Relevancy
**What it measures:** How closely does the generated answer match the user's query intent?

**Intuition:** The user asked "Who wrote Romeo and Juliet?" An answer of "Romeo and Juliet is a 5-act play" is factually correct but doesn't answer the question. Low relevancy = the answer drifts from what was asked.

**Formula (RAGAS approach):**
RAGAS uses the LLM-as-judge approach: it generates N hypothetical questions that the answer could be a response to, then measures the cosine similarity between those hypothetical questions and the original question.

**Why it matters:** An irrelevant answer is a failure of the generation step regardless of how good the retrieval was.

**Example (Romeo & Juliet):**
- Question: "Who wrote Romeo and Juliet?"
- Answer: "Romeo and Juliet was written by William Shakespeare in 1597."
- This directly addresses "who wrote it." The "in 1597" part doesn't hurt relevancy — it's still about the same question.
- **Score: ~0.95** ← Answer directly addresses the query.

---

### Metric 4: Faithfulness
**What it measures:** Does the generated answer stay true to what is in the retrieved context? Does it add facts not in the context?

**Intuition:** Faithfulness is the anti-hallucination metric. A faithful answer only contains claims that can be verified in the retrieved documents.

**Formula:**
```
Faithfulness = (Number of answer claims supported by context) / (Total claims in answer)
```

**RAGAS approach:** Decomposes the answer into atomic claims, then checks each claim against the retrieved context.

**Why it matters:** Faithfulness = 1.0 means zero parametric hallucination. Lower scores mean the model added information from its weights, not the retrieved context.

**Example (Romeo & Juliet) — The Revealing Case:**
- Answer: "Romeo and Juliet was written by **William Shakespeare** in **1597**."
- Context: "Romeo and Juliet is a tragedy written by **William Shakespeare**."

Claim 1: "Written by William Shakespeare" → **Supported by context** ✓
Claim 2: "In 1597" → **NOT in context** ✗ (hallucinated, even if historically plausible)

```
Faithfulness = 1 supported / 2 total claims = 0.5
```

**Score: ~0.5** ← Half the claims are grounded; the other half is a hallucination.

> This is the key insight from the slides: Context Precision, Context Recall, and Answer Relevancy all scored near 1.0 for this example. Only **Faithfulness** caught the hallucination. This is why you need all four metrics.

---

### RAGAS Metrics — Full Scorecard for the Romeo & Juliet Example

| Metric | Score | Reason |
|--------|-------|--------|
| Context Precision | ~1.0 | The one retrieved chunk is fully on-topic |
| Context Recall | ~1.0 | Context contains everything the ground truth needs |
| Answer Relevancy | ~0.95 | Answer directly addresses "who wrote it" |
| **Faithfulness** | **~0.5** | "in 1597" is not in context — hallucinated detail |

**Critical observation:** If you only measured answer accuracy against ground truth ("William Shakespeare" is present in both the answer and ground truth), the system would appear correct. Only RAGAS's faithfulness metric reveals the hidden hallucination.

---

### How the Four Metrics Map to RAG Failure Modes

| RAGAS Metric | Failure Mode It Detects |
|-------------|------------------------|
| Context Precision | Wrong chunks retrieved (retrieval failure) |
| Context Recall | Incomplete retrieval (retrieval failure) |
| Answer Relevancy | Generation drifts from query (generation failure) |
| Faithfulness | Hallucination despite retrieval (generation failure) |

> **Learning Thought:** RAGAS provides a metric for each failure mode category from Section 4 (retrieval failures → context precision/recall; generation failures → faithfulness/answer relevancy). The two sections are designed to be read together.

---

## Topic 36: TruLens

### What Is TruLens?
TruLens is an **observability and evaluation platform** for LLM-based applications, with strong support for RAG pipelines. Where RAGAS is a batch evaluation library, TruLens is a **continuous monitoring system**.

### Key Capabilities
| Capability | Description |
|-----------|-------------|
| **Feedback functions** | Configurable evaluation functions (similar to RAGAS metrics) that score each LLM call |
| **Tracing** | Instruments each step in the RAG pipeline; records inputs, outputs, and metadata for every call |
| **Dashboard** | Web UI for exploring traces, comparing runs, and tracking metric trends over time |
| **LLM-as-judge** | Uses a separate LLM to evaluate answers; no ground truth required for most metrics |
| **Continuous monitoring** | Runs evaluation on every production query, not just on batch test sets |

### TruLens RAG Triad
TruLens uses a concept called the **RAG Triad** — three questions every RAG response should answer:
1. **Context Relevance** — Is the retrieved context relevant to the query? (≈ RAGAS Context Precision)
2. **Groundedness** — Is the answer grounded in the retrieved context? (≈ RAGAS Faithfulness)
3. **Answer Relevance** — Is the answer relevant to the query? (≈ RAGAS Answer Relevancy)

### RAGAS vs TruLens

| Dimension | RAGAS | TruLens |
|-----------|-------|---------|
| Type | Batch evaluation library | Observability + evaluation platform |
| When to use | Offline evaluation on test sets | Continuous monitoring in production |
| Ground truth required | For some metrics (context recall) | Not required for RAG Triad |
| Dashboard | No (Python library) | Yes (web UI) |
| Open source | Yes | Yes (core is open source) |
| Best for | Research, development evaluation | Production monitoring |

> **Learning Thought:** Think of RAGAS as your test suite and TruLens as your monitoring infrastructure. You use RAGAS to evaluate before deployment; you use TruLens to ensure quality doesn't degrade after deployment.

---

## The Complete RAG Evaluation Mental Model

```
Build RAG → Evaluate offline (RAGAS) → Deploy → Monitor in production (TruLens)
                    │                                       │
                    ▼                                       ▼
              Context Precision/Recall              Real-time faithfulness
              Faithfulness                          Context relevance
              Answer Relevancy                      Answer relevance
              (on curated test set)                 (on every production query)
```

---

## Interview Questions — Section 6

### Fundamental

**Q1. What are the four RAGAS metrics and what does each measure?**
> (1) **Context Precision** — what fraction of retrieved chunks are relevant to the query (retrieval precision). (2) **Context Recall** — what fraction of the information needed for the correct answer was retrieved (retrieval completeness). (3) **Answer Relevancy** — how closely the generated answer addresses the user's question intent. (4) **Faithfulness** — what fraction of claims in the generated answer are supported by the retrieved context (anti-hallucination metric).

**Q2. Why is faithfulness the most critical RAGAS metric?**
> Faithfulness is the only metric that directly detects LLM hallucination in a RAG context. The other three metrics (precision, recall, relevancy) can all score well even when the LLM adds fabricated details. In the Romeo and Juliet example, all other metrics scored ~1.0 while faithfulness scored ~0.5 because of the hallucinated "in 1597." A high faithfulness score is the primary guarantee that the system is not adding ungrounded claims.

**Q3. What is the RAGAS evaluation dataset and what four fields does it require?**
> The evaluation dataset contains: (1) `question` — the user's query; (2) `contexts` — the chunks retrieved by the RAG system; (3) `answer` — the LLM's generated response; (4) `ground_truth` — an optional reference answer. RAGAS uses these fields to compute all four metrics; ground_truth is required for context recall but optional for the others.

### Intermediate

**Q4. How does RAGAS's faithfulness metric detect hallucination?**
> RAGAS decomposes the generated answer into atomic claims (individual facts). For each claim, it checks whether that claim is supported by the retrieved context. Faithfulness = (supported claims) / (total claims). A score below 1.0 indicates hallucination — the LLM generated at least one claim not traceable to the retrieved documents.

**Q5. What is the difference between context precision and context recall?**
> Context precision is a quality measure: of the chunks you retrieved, how many are actually relevant? High precision = no noise in the context. Context recall is a completeness measure: of all the information needed to answer correctly, how much did you retrieve? High recall = you found everything. They trade off: retrieving more chunks tends to improve recall but hurts precision. A good RAG system optimises both via reranking and precise query formulation.

**Q6. How would you use RAGAS to diagnose a RAG system that gives wrong answers?**
> Analyse the four metrics:
> - Low Context Precision → retriever returning irrelevant chunks → fix chunking strategy or query formulation
> - Low Context Recall → relevant documents missed → improve retrieval (hybrid search, larger top-k, better embeddings)
> - Low Answer Relevancy → generation drifts from query → fix prompt template or system instructions
> - Low Faithfulness → LLM adding parametric facts → strengthen constrained generation instructions; consider smaller top-k to reduce context noise

### Advanced

**Q7. Compare RAGAS and TruLens. When would you use each?**
> RAGAS: Python library, batch evaluation, requires constructing a test dataset, best for development/research phases before deployment. Provides four clean metrics. Needs ground truth for context recall.
> TruLens: Observability platform with web dashboard, instruments the production pipeline, evaluates every query in real-time, doesn't require ground truth for most metrics. Best for production monitoring.
> **Use both**: RAGAS during development to tune the system; TruLens in production to ensure it doesn't regress.

**Q8. Can RAGAS metrics be gamed? What are their failure modes?**
> Yes, each metric has limitations:
> - Context Precision/Recall use LLM-as-judge for relevance assessment, which itself can hallucinate. Ground truth quality determines context recall accuracy.
> - Answer Relevancy uses embedding similarity of hypothetical questions — may miss subtle relevance failures for complex queries.
> - Faithfulness decomposes into atomic claims using an LLM — the claim extraction step can be imperfect; complex claims may not be cleanly verifiable against context.
> Best practice: use RAGAS metrics as signals, not ground truth. Combine with human evaluation for high-stakes use cases.

---

## Key Learning Thoughts — Section 6

> **Thought 1 — You can't improve what you don't measure:** A RAG system that "seems to work" on demo queries may be systematically failing on production queries. Without RAGAS or TruLens, you have no visibility. Evaluation is not optional — it is the engineering discipline that makes RAG trustworthy.

> **Thought 2 — Each metric catches a different failure mode:** Don't pick one metric. The Romeo and Juliet example perfectly illustrates this: 3 of 4 metrics looked perfect while the system was actively hallucinating. You need all four metrics to have complete visibility.

> **Thought 3 — Faithfulness is the LLM accountability metric:** In regulated domains (healthcare, finance, legal), you need to be able to say "this answer comes from this document." Faithfulness measures exactly that. A system with low faithfulness cannot be deployed in regulated contexts.

> **Thought 4 — RAGAS pairs with Section 4:** Every RAGAS metric corresponds directly to a failure mode category from Section 4. Context precision/recall → retrieval failures. Faithfulness → generation failures (hallucination). Answer relevancy → generation failures (drift). If you learned the failure modes first, the metrics are immediately intuitive.

> **Thought 5 — LLM-as-judge is powerful but not infallible:** RAGAS uses an LLM (GPT-4 or equivalent) to evaluate the RAG system's outputs. This is brilliant for scalability but introduces a meta-level concern: the judge LLM can also hallucinate or make errors. For high-stakes evaluation, supplement LLM-as-judge with human review on a sample.

---

## Session Summary — The Complete Picture

```
WHY RAG?          → Section 1: LLMs hallucinate, have knowledge cutoffs, can't access private data
WHAT IS RAG?      → Section 2: Index + Retrieve + Generate; naive vs advanced pipeline
HOW TO RETRIEVE?  → Section 3: Sparse (TF-IDF, BM25) vs Dense; chunking; query formulation
WHAT GOES WRONG?  → Section 4: Retrieval failures, Context failures, Generation failures
WHICH VARIANT?    → Section 5: Standard → Memory → Agentic → CoRAG (by complexity need)
HOW TO MEASURE?   → Section 6: RAGAS (offline) + TruLens (online monitoring)
```

**The Master Insight:**
> RAG turns the LLM from an **oracle of memorised facts** into a **reasoner over retrieved evidence**. Answer quality is bounded by retrieval quality. The retriever surfaces; the LLM reasons; RAGAS verifies.

---

*Previous: [Section 5 — RAG Variants](./05_RAG_Variants.md) | Back to [Index →](./00_Index.md)*
