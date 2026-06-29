# Section 4 — Request Shaping & Query Transformation

> Topics: request shaping (motivation + internal mechanics, 2 LLM calls) · query transformation overview · the 4 techniques: query rewriting, follow-up transformation, multi-query retrieval, step-back prompting.

---

## Big picture

> **Fix the *question* before you fix the retrieval.**

Users write vague, fragmented, or context-dependent queries ("Teach me attention", "Server down", "OOMKilled"). A great retriever over a bad query still returns bad chunks. **Request shaping** transforms the user's query *before* retrieval to improve relevance, precision, and recall.

This is the first thing that happens in the production pipeline (Section 3's diagram) — and the first **LLM call** in the system.

---

## 4.1 Request Shaping — the concept

**Motivation:** requests made by users are not always precise and usable.
**Example:** "Teach me attention." — this prompt doesn't specify what "attention" really means (the ML mechanism? paying attention? the *Attention Is All You Need* paper?).

> **Request shaping transforms a user query before retrieval to improve relevance, precision, and retrieval quality.**

Without vs with:
```
WITHOUT:  User query → Retriever → Poor results   (ambiguous query → irrelevant/incomplete)
WITH:     User query → Request Shaping → Improved query → Retriever → Better results
```

### How it works internally (the 2-LLM-call pattern)
```
1. User query → 2. Request Shaping (LLM Call #1) → 3. Retrieval (non-LLM) → 4. Answer Generation (LLM Call #2)
```

| Step | What | LLM? |
|------|------|------|
| 1 | User asks in natural language ("Why did APAC revenue drop in Q3?") | — |
| 2 | **LLM Call #1 — Request Shaper:** rewrite / expand / decompose / clarify → *Improved Query* | ✅ |
| 3 | **Retrieval:** vector search with the improved query → top-k chunks | ❌ |
| 4 | **LLM Call #2 — Answer Generator:** uses retrieved context + original query → final answer | ✅ |

**Inside the Request-Shaping LLM:** understand intent & context → expand & enrich (synonyms, metrics, business context) → rewrite / decompose → output optimized query.

**Same LLM or different?** Either works:
- **Same LLM** — simpler to manage, lower cost (one model).
- **Different LLMs** — small/cheap model for shaping, larger model for answer generation.

**Why multiple LLM calls?** Specialized tasks produce better results; shaping improves retrieval quality; the final answer is more accurate and grounded.

---

## 4.2 Query Transformation — overview

Using an LLM to transform the user query before retrieval to improve relevance and results.

```
NAIVE RAG:   "attention paper" → embed → vector search → often irrelevant results
WITH TRANSFORM: "attention paper" → LLM transformer →
   "Attention Is All You Need paper introducing the Transformer architecture"
   → embed → vector search → more relevant results
```

**Benefits:** improved relevance · better recall · handles ambiguity · captures intent.

The deck details **4 techniques**. They all live in the **"Request Shaping Layer"** of the pipeline (between user query and embedding/retrieval).

| # | Technique | One-liner | Trigger |
|---|-----------|-----------|---------|
| 1 | **Query Rewriting** | make a vague query clear & specific | shorthand, jargon, error codes |
| 2 | **Follow-up Transformation** | turn a contextual follow-up into a standalone query | multi-turn conversations with pronouns |
| 3 | **Multi-Query Retrieval** | generate several diverse queries, retrieve all, merge | complex / comparison / multi-source questions |
| 4 | **Step-Back Prompting** | ask a broader question first to get foundational context | hyper-specific "deep in the weeds" queries |

---

## 4.3 Technique 1 — Query Rewriting

**What:** use an LLM to rewrite ambiguous/vague/shorthand queries into clearer, highly specific search statements.
**When:** support channels (Slack, Teams) where fragmented sentences, error codes, or jargon hinder vector matching.

**DevOps Agent example:**
- User: `"OOMKilled"`
- Rewritten: `"What causes the OOMKilled container exit code in Kubernetes, and what are the remediation steps?"`
- **Impact:** instead of returning random docs containing "OOM", the agent retrieves specific internal **runbooks** for memory allocation and container scaling.

Why it helps: resolves ambiguity, adds missing context/specificity, improves lexical + semantic match, increases precision & recall. Used in RAG-Fusion, Azure AI Search, LangChain SelfQueryRetriever.

---

## 4.4 Technique 2 — Follow-up Question Transformation

**What:** convert contextual follow-up questions into complete, standalone queries by injecting the conversation history.
**When:** multi-turn troubleshooting/policy inquiries where the user relies on pronouns ("it", "they", "that server").

**HR/Support Agent example:**
- History — User: "What is the standard parental leave policy?" → Agent: [returns policy] → User: **"Does it apply to contractors?"**
- Transformed: `"Does the standard parental leave policy apply to contract employees?"`
- **Impact:** without this, the retriever searches for "Does it apply to contractors?" and **fails** (no anchor). The standalone query ensures high-accuracy retrieval.

Why it helps: resolves coreferences (it, they, there), converts follow-ups to standalone form, maintains conversation context. Used in conversational RAG, ChatGPT retrieval, LangChain Chat-History-Aware Retriever.

---

## 4.5 Technique 3 — Multi-Query Retrieval

**What:** generate multiple distinct, related queries from a single complex question and retrieve documents for all of them in **parallel**, then merge (union + dedup).
**When:** a prompt spans multiple microservices, compares entities, or requires joining info from separate documents.

**DevOps Agent example:**
- User: `"Did the latency spike happen in both the Mumbai and US-East Vertex AI deployments?"`
- Generated:
  - Q1: "What are the recent latency metrics and error logs for the **Mumbai** Vertex AI deployment?"
  - Q2: "What are the recent latency metrics and error logs for the **US-East** Vertex AI deployment?"
- **Impact:** guarantees the LLM gets context for **both** deployments before comparing, avoiding regional bias.

Why it helps: captures multiple aspects/phrasings, expands recall across angles, reduces embedding miss, improves answer completeness. Used in RAG-Fusion, LangChain MultiQueryRetriever, HyDE.

> Related idea — **HyDE** (Hypothetical Document Embeddings): have the LLM write a hypothetical *answer*, embed *that*, and search with it (answers look more like the target docs than questions do).

---

## 4.6 Technique 4 — Step-Back Prompting

**What:** create a broader, higher-level question to retrieve **foundational** context *before* attempting to answer a hyper-specific query.
**When:** an engineer's query is so deep in the weeds it misses underlying architectural rules → "no results" or hallucinations.

**DevOps Agent example:**
- User: `"Why can't I pull the switch association data for the rogue units on Floor 3?"`
- Step-back (broader): `"What is the overall architecture of our network troubleshooting system, and what data is collected from reporting access points versus rogue units?"`
- **Impact:** the step-back query retrieves the **foundational rule** — *switch association data is only available for reporting access points, not rogue units* — letting the agent explain the architectural limitation instead of failing.

Why it helps: moves to a higher abstraction level, retrieves foundational knowledge, bridges gaps in implicit assumptions, improves reasoning & answer quality. Used in Anthropic's step-back prompting, LangChain, advanced RAG.

> **Deck quiz:** user gets "no results"/hallucinations on hyper-specific queries; vector DB has the architectural rules but lacks the user's exact keywords → **Step-Back Prompting** (answer C). It abstracts the query to retrieve underlying rules first.

---

## How the system applies these (orchestration)

A single LLM orchestration step can apply *several* transformations at once. Example (deck): "Why is the payment failing in the checkout service?" expands into:
- a. "What error messages are returned when payment fails in checkout?"
- b. "Show recent logs and traces related to payment failures in checkout."
- c. "Are there any recent code changes in the payment module?"
- d. "What dependencies or external services are involved in payment?"
- + related queries: payment timeout issue, payment gateway errors.

All retrieved in parallel → results fused → grounded, cited answer.

> **The open question this raises:** *which* technique should the system pick for a given query, and *which* data source should it search? That's **Query Routing** — Section 5.

---

## 🎯 Interview questions

**Q1. What is request shaping and why is it the first step in production RAG?**
Transforming the raw user query (rewrite/expand/decompose/clarify) before retrieval, because vague queries cap retrieval quality regardless of index quality. It's the first LLM call; better queries → better chunks → better answers.

**Q2. Describe the two-LLM-call pattern.**
Call #1 (Request Shaper) turns the user query into an improved query; retrieval runs (non-LLM); Call #2 (Answer Generator) uses retrieved context + original query to answer. The two can be the same model or a cheap-shaper + strong-answerer split.

**Q3. Compare the four query-transformation techniques.**
Rewriting: vague → specific (single query). Follow-up: contextual → standalone via history. Multi-query: one complex → several parallel queries, merged. Step-back: specific → broader to fetch foundational context. Rewriting/follow-up reshape *one* query; multi-query *fans out*; step-back *abstracts up*.

**Q4. A user in a multi-turn chat asks "Does it apply to contractors?" — what breaks and what fixes it?**
The retriever has no anchor for "it" → fails. Follow-up Question Transformation injects history to produce a standalone query ("Does the parental leave policy apply to contract employees?").

**Q5. When is multi-query retrieval the right tool, and what's its cost?**
When the question spans multiple entities/services or compares things. It boosts recall/completeness but costs extra retrievals + an LLM call to generate the sub-queries, plus dedup/merge logic.

**Q6. What problem does step-back prompting uniquely solve?**
Hyper-specific queries that miss foundational rules and return nothing/hallucinate. Abstracting to a broader question retrieves the underlying architecture/rules first, which then explains the specific edge case.

**Q7. What is HyDE and why might it help?**
Hypothetical Document Embeddings: the LLM drafts a hypothetical answer, you embed and search with it. A pseudo-answer is geometrically closer to real answer-passages than a short question is, improving retrieval.

**Q8. Same vs different LLM for shaping and generation — trade-offs?**
Same: simpler, one model to manage, often cheaper to operate. Different: route shaping to a small/cheap model and generation to a strong one → cost-optimised without quality loss on the final answer.

---

## 🧠 Learning thoughts

- **The query is part of the system you design, not a fixed input.** Most RAG quality wins early on come from shaping queries, not swapping embedding models.
- The four techniques map to four failure modes: **vague** (rewrite), **context-dependent** (follow-up), **multi-faceted** (multi-query), **too-specific** (step-back). Diagnose the failure → pick the technique.
- **Step-back is the counter-intuitive gem:** to answer a narrow question, first ask a *broader* one. Humans do this — "let me back up and understand the system first."
- These techniques add **LLM calls = latency + cost**. That's exactly why Section 5 (routing) exists: decide *when* each is worth it instead of always running all of them.
- Everything here is "garbage-in prevention." It pairs with Section 2's "garbage-out detection" (reranking, silent-failure checks) and Section 6's evaluation.

## ✅ Self-check

1. Map each of the 4 techniques to the user-query failure mode it fixes.
2. Draw the two-LLM-call request-shaping flow and label which steps use an LLM.
3. Give the OOMKilled, contractor-follow-up, Mumbai/US-East, and Floor-3 examples and name the technique each illustrates.
4. Why does query transformation create the *need* for query routing?
