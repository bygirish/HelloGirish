# Section 4: Common Problems of RAG

> **Core Idea:** RAG does not make your LLM infallible. It introduces a whole new set of failure modes on top of the existing LLM limitations. Understanding where things can go wrong is the prerequisite to building robust RAG systems.

---

## Topic 23: RAG Failure Modes — The Three Categories

RAG adds external retrieval to an LLM. Every link in the chain can fail. The lecture organises failures into three clean buckets:

```
┌──────────────────────────────────────────────────────────┐
│               RAG PIPELINE                                │
│                                                           │
│  User Query → [RETRIEVAL] → Context → [LLM] → Answer    │
│                   │              │         │              │
│              Retrieval       Context   Generation         │
│              Failures        Failures   Failures          │
└──────────────────────────────────────────────────────────┘
```

| Category | Where it breaks | Root cause |
|----------|----------------|------------|
| **Retrieval Failures** | Retriever returns wrong, incomplete, or outdated chunks | Gap between query and index |
| **Context Failures** | Right chunks retrieved, but LLM fails to use them correctly | LLM context window limitations |
| **Generation Failures** | Context is fine, but LLM generates wrong output | Parametric memory override / attribution errors |

> **Learning Thought:** These three categories map to the three stages of the RAG pipeline. When a RAG system produces a bad answer, the first debugging question is: *which stage failed?* This taxonomy is your diagnostic framework.

---

## Topic 24: Retrieval Failures

The retriever is the foundation. If it fails, everything downstream fails.

### Failure Mode 1: Wrong Chunk Retrieved (Vocabulary Mismatch)

**What happens:** The retriever returns a chunk that is superficially similar to the query (shares keywords or embedding similarity) but answers a *different* question.

**Example from slides:**
```
Query: "What is the return window for online orders?"
Relevant chunk: "Online orders can be returned for a refund within 30 days of delivery."
Retrieved chunk: "Our in-store exchange policy allows swaps within 14 days."
Answer: 14 days  ← WRONG
```

**Root cause:** "Online orders" and "in-store exchange" share no exact keywords, but their embeddings might be similar (both are about product returns). Semantic similarity ≠ query relevance.

**Mitigations:**
- Hybrid retrieval (sparse + dense) — sparse catches exact "online orders" match
- Better query formulation (add specificity to query before retrieval)
- Re-ranking with a cross-encoder to distinguish relevance from similarity

### Failure Mode 2: Incomplete Retrieval (Multi-Hop Queries)

**What happens:** The answer requires synthesising information from *multiple* documents, but the retriever only returns one relevant chunk, or returns related but incomplete chunks.

**Example from slides:**
```
Query: "What is the return window for online orders?"
Retrieved: "Online orders can be returned for a refund"  (chunk missing the "30 days" part)
Answer: "I don't know"  ← WRONG (incomplete context)
```

**Root cause:** Multi-hop questions require chaining facts across documents. Single-step retrieval finds one relevant document but misses the supporting facts in others.

**Mitigations:**
- Multi-query retrieval to fetch multiple relevant passages
- Chain-of-RAG (CoRAG) — iterative retrieval that decomposes the question into sub-queries
- Increasing top-k (retrieve more chunks, at risk of context overload)

### Failure Mode 3: Stale Knowledge Base

**What happens:** The knowledge base was indexed at a point in time, but the real-world fact has since changed. The retriever returns a chunk that was once correct but is now outdated.

**Example from slides:**
```
Query: "What is the return window for online orders?"
Retrieved: "Online return window: 15 days."  (last year's policy)
Answer: 15 days  ← WRONG (policy changed to 30 days)
```

**Root cause:** The index is a static snapshot. Unlike the LLM (which is frozen by design), the knowledge base *should* be updated but often isn't frequently enough.

**Mitigations:**
- Regular re-indexing schedules
- Metadata filters (chunk creation date; reject chunks older than N days)
- Freshness signals in ranking — down-weight older chunks
- Real-time data connectors for frequently-changing information

---

## Topic 25: Context Failures

The right chunks were retrieved, but the LLM fails to extract the answer from them correctly.

### Failure Mode 1: Lost in the Middle

**What happens:** The correct chunk is present in the retrieved context, but it's in the *middle* of a long context window. LLMs systematically attend more to the beginning and end of their context than to the middle — so the correct chunk is effectively invisible.

**Example from slides:**
```
Context:
[chunk 1] Office locations…
[chunk 2] Mission statement…
[chunk 5] Founded in 2008  ← correct chunk
[chunk 9] Product list…
[chunk 10] Contact info…

Query: "When was the company founded?"
Answer: "I don't know"  ← LLM missed chunk 5
```

**Root cause:** Transformer attention is not uniformly distributed across the context window. Research (["Lost in the Middle"](https://arxiv.org/abs/2307.03172)) confirms LLMs perform significantly worse on facts buried in the middle of long contexts.

**Mitigations:**
- Reduce top-k (fewer, better chunks — quality over quantity)
- Re-rank so the most relevant chunk appears first *and* last
- Use LLMs with longer context windows and better context utilisation
- Summarise retrieved chunks into a compact context before generation

### Failure Mode 2: Context Overload

**What happens:** Too many retrieved chunks overwhelm the LLM, causing it to produce a vague, confused, or mixed-up answer.

**Example from slides:**
```
Context: [5 chunks about history, HR policies, blog posts, FAQs…]
Query: "When was the company founded?"
Answer: "vague / mixed-up"  ← LLM couldn't isolate the relevant fact
```

**Root cause:** More context is not always better. The LLM's reasoning capacity is finite. Irrelevant chunks compete for attention, degrading generation quality.

**Mitigations:**
- Aggressive reranking and filtering before passing to LLM (post-retrieval step)
- Set a low top-k (3–5 chunks rather than 10+) for precise factual questions
- Use a summarisation step to compress retrieved content

### Failure Mode 3: Irrelevant Context (Distraction)

**What happens:** A retrieved chunk is factually correct but answers a *related but different* question. The LLM is distracted by this irrelevant information and produces the wrong answer.

**Example from slides:**
```
Context: "Our partner company was founded in 1999."
Query: "When was the company founded?"
Answer: 1999  ← WRONG (answered about the partner, not the company)
```

**Root cause:** LLMs can be easily distracted by plausible-sounding but incorrect context. Research (["Large Language Models Can Be Easily Distracted by Irrelevant Context"](https://arxiv.org/abs/2302.00093)) shows this is a systematic weakness.

**Mitigations:**
- Reranking to filter out tangentially related chunks
- System prompt instruction: "Answer only about [company name], ignore references to other entities"
- Entity-aware retrieval — filter chunks by entity metadata

---

## Topic 26: Generation Failures

The retrieval was good, the context was right, but the LLM still produces a wrong output.

### Failure Mode 1: Hallucination Despite Retrieval

**What happens:** The LLM generates facts that are *not* in the retrieved documents — it draws from its parametric memory instead of (or in addition to) the context.

**Example from slides:**
```
Retrieved: "Refundable within 30 days"
Query: "What is the refund window for online orders?"
Answer: "You get 30 days, and a free shipping label plus a 10% voucher."
           ← "free shipping label" and "10% voucher" NOT in context
```

**Root cause:** The LLM is conditioned to produce helpful, complete-sounding answers. When a response feels incomplete (just "30 days"), it tends to "fill in" with plausible additions from its parametric memory.

**Mitigations:**
- Explicit system prompt: "Only use information from the provided context. Do not add information not present in the context."
- RAGAS faithfulness metric to detect this automatically
- Constitutional AI / self-check prompting: ask the LLM to verify each claim against the context

### Failure Mode 2: Knowledge Conflict

**What happens:** The retrieved document says one thing; the LLM's parametric memory says something different. The LLM trusts its internal knowledge over the retrieved evidence.

**Example from slides:**
```
Retrieved context: "30 days"
LLM's parametric memory: "most stores use 15 days"
Answer: 15 days  ← LLM overrode the correct retrieved context
```

**Root cause:** LLMs can exhibit **parametric stubbornness** — strong priors from pretraining override provided context. This is especially likely when the parametric prior is confident and the context contradicts a commonly-held belief.

**Mitigations:**
- Reinforce context authority in the system prompt: "The provided context is always correct. If it contradicts your prior knowledge, trust the context."
- Use instruction-tuned models that are trained to respect context
- Faithfulness evaluation to detect context-overriding

### Failure Mode 3: Attribution Errors

**What happens:** The LLM cites the wrong source document for its answer, or correctly answers but attributes it to a different document than the one that actually contained the answer.

**Example from slides:**
```
Retrieved: "Refundable within 30 days" [Policy.pdf]
Answer: "30 days. [Source: Shipping-FAQ.pdf]"  ← wrong source cited
```

**Root cause:** When multiple documents are in the context, the LLM's cross-document attribution is unreliable. It may blend information from multiple sources and cite one arbitrarily.

**Mitigations:**
- Structured prompting: label each retrieved chunk with its source and ask the LLM to cite the chunk label
- Post-generation verification: programmatically check whether the cited source actually contains the claimed text
- Reduce context to fewer chunks so attribution is simpler

---

## RAG Failure Modes — Complete Reference Table

| Category | Failure Mode | Root Cause | Mitigation |
|----------|-------------|-----------|------------|
| **Retrieval** | Wrong chunk (vocabulary mismatch) | Semantic similarity ≠ relevance | Hybrid retrieval, reranking |
| **Retrieval** | Incomplete retrieval (multi-hop) | Single-step retrieval, missing supporting facts | CoRAG, multi-query, larger top-k |
| **Retrieval** | Stale knowledge base | Index not updated | Re-indexing schedules, freshness signals |
| **Context** | Lost in the middle | LLM attention bias to start/end | Reduce top-k, reorder by relevance |
| **Context** | Context overload | Too many chunks confuse LLM | Aggressive filtering, summarise context |
| **Context** | Irrelevant context (distraction) | Plausible-but-wrong chunks mislead LLM | Better reranking, entity-aware retrieval |
| **Generation** | Hallucination despite retrieval | LLM adds parametric facts to context | Constrained prompting, faithfulness check |
| **Generation** | Knowledge conflict | Parametric memory overrides context | Context authority in system prompt |
| **Generation** | Attribution errors | Cross-document blending | Structured chunk labelling, verify post-gen |

---

## Interview Questions — Section 4

### Fundamental

**Q1. What are the three categories of RAG failure modes?**
> (1) **Retrieval failures** — the retriever returns wrong, incomplete, or outdated chunks before the LLM ever sees them. (2) **Context failures** — the right chunks were retrieved but the LLM fails to extract the correct answer (lost in middle, context overload, distraction). (3) **Generation failures** — context is correct but the LLM still generates wrong output (hallucination despite retrieval, knowledge conflict, attribution errors).

**Q2. What is the "lost in the middle" problem?**
> LLMs systematically attend more to the beginning and end of their context window than to content in the middle. When a relevant document chunk is positioned in the middle of a long context, the LLM may effectively ignore it and produce a wrong or "I don't know" answer. Mitigation: reduce the number of retrieved chunks, rerank so the best chunk is first, or use summarisation to compress context.

**Q3. What is a knowledge conflict in RAG?**
> A knowledge conflict occurs when the retrieved document provides a fact that contradicts the LLM's parametric memory. Instead of trusting the provided context, the LLM falls back on its trained-in belief. For example, the retrieved document says "30-day refund window" but the LLM "knows" (from pretraining) that most stores use 15 days and answers 15.

### Intermediate

**Q4. Distinguish between "wrong chunk retrieved" and "incomplete retrieval."**
> "Wrong chunk" is a precision failure: a chunk was retrieved, but it answers a different question than the one asked (vocabulary mismatch or false semantic similarity). "Incomplete retrieval" is a recall failure: the answer exists in the corpus but requires facts from multiple documents; single-step retrieval only finds one of them. Wrong chunk → wrong answer. Incomplete retrieval → "I don't know" or partial answer.

**Q5. Why does hallucination persist even in RAG systems?**
> Because the LLM still draws on its parametric memory alongside the retrieved context. When the context provides a partial answer, the LLM tends to "complete" it with plausible additions from pretraining. Additionally, knowledge conflicts can cause the model to override correct context with incorrect parametric priors. RAG reduces hallucination but does not eliminate it — constrained prompting and faithfulness evaluation are needed as additional guards.

**Q6. How would you debug a RAG system that consistently gives wrong answers?**
> Apply the failure-mode taxonomy:
> 1. **Inspect retrieved chunks** — are they correct and relevant? (Retrieval failure?)
> 2. **Check if the answer is in the context** — is the relevant chunk present? (Lost in middle? Context overload?)
> 3. **Check if the generated answer follows the context** — does it add facts not in the chunks? (Hallucination? Knowledge conflict?)
> 4. **Run RAGAS metrics** — faithfulness score detects generation hallucination; context precision/recall diagnose retrieval.

### Advanced

**Q7. How do you address context overload without losing recall?**
> Context overload and recall are in tension: more chunks = better recall but worse generation quality. Solutions: (1) **Re-ranking** — retrieve a large initial set (top-20), then rerank to top-3 for generation. (2) **Compression** — use a small LLM to summarise retrieved chunks into a compact context before passing to the main LLM. (3) **Selective context** — for each query, use the cross-encoder score to dynamically set top-k rather than a fixed number.

**Q8. What is the difference between "hallucination despite retrieval" and "knowledge conflict"?**
> Both are generation failures, but the mechanism differs. Hallucination despite retrieval: the model *adds* new facts not present in the context (parametric memory leakage). Knowledge conflict: the context contains a correct fact, but the model *ignores or overrides* it in favour of a contradicting parametric belief. The first is an addition error; the second is a substitution error. Both are detected by the RAGAS **faithfulness** metric.

---

## Key Learning Thoughts — Section 4

> **Thought 1 — The failure mode taxonomy is a debugging tool:** When a RAG system produces a bad answer, don't just say "RAG failed." Classify the failure: retrieval, context, or generation. Each has different mitigations. The taxonomy turns debugging from art into engineering.

> **Thought 2 — Retrieval failures are the most common and the most fixable:** Most RAG failures in practice originate in the retrieval stage. Vocabulary mismatch and stale knowledge bases are addressable with engineering. Invest heavily in retrieval quality before optimising generation.

> **Thought 3 — "Lost in the middle" has profound implications for top-k:** Many practitioners set top-k=10 thinking more context is better. Research shows that above 5 chunks, LLM performance often *degrades* because of lost-in-the-middle and context overload effects. Quality beats quantity.

> **Thought 4 — Constrained generation is a first-line defence:** The single most impactful intervention against generation failures is a well-crafted system prompt that explicitly instructs the LLM to use only provided context and acknowledge uncertainty. This costs zero computation.

> **Thought 5 — Evaluation is non-negotiable:** These failure modes are invisible without systematic evaluation. A RAG system can appear to work on test queries while silently failing on production queries. RAGAS (Section 6) was built specifically to surface these failure modes — understanding the failures first makes the metrics meaningful.

---

*Previous: [Section 3 — The Retrieval Phase](./03_Retrieval_Phase.md) | Next: [Section 5 — RAG Variants →](./05_RAG_Variants.md)*
