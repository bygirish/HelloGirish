# Module 2 · Lecture 5 — Retrieval Augmented Generation (RAG)

> Instructor: **Prof. Pawan Goyal**, Dept. of CSE, IIT Kharagpur (upGrad)
> Source: 85-slide deck + pre/post-read material + `Production_RAG_HandsOn.ipynb`

These notes turn the lecture into a **study path**: read top-to-bottom to build *understanding*, then use the interview questions and "Build expertise" boxes to convert understanding into *expertise*.

---

## How to use these notes

Each section file has the same skeleton:

1. **Big picture** — why this section exists in the RAG story.
2. **Topic-by-topic** — core idea → why it exists → worked example → mental model.
3. **🎯 Interview questions** — with model answers, ordered easy → hard.
4. **🧠 Learning thoughts** — the non-obvious insights worth remembering.
5. **✅ Self-check** — quick recall prompts.

The single thread running through the whole lecture is one case study: an **Enterprise DevOps & Support Agent** (troubleshoots incidents, queries codebase, answers HR questions, stays secure). Every abstract idea is grounded in it.

---

## The 6 sections (+ recap)

| # | File | Theme | One-line takeaway |
|---|------|-------|-------------------|
| 0 | [00_Recap.md](00_Recap.md) | RAG pipeline + IVF + PQ recap | RAG = retrieve relevant chunks, then let the LLM answer over them. |
| 1 | [01_Advanced_Vector_Indexing.md](01_Advanced_Vector_Indexing.md) | IVF+PQ, HNSW, FAISS, ChromaDB | Indexes trade **recall ↔ speed ↔ memory**; pick by corpus size. |
| 2 | [02_Retrieval_Pipeline_and_Reranking.md](02_Retrieval_Pipeline_and_Reranking.md) | Semantic search, bi/cross-encoder, ColBERT | **Retrieve cheap, rerank dear** — the two-stage pattern. |
| 3 | [03_Production_RAG_and_Agentic_Transition.md](03_Production_RAG_and_Agentic_Transition.md) | Enterprise / Production RAG | Production adds routing, gates, multi-source, observability. |
| 4 | [04_Request_Shaping_and_Query_Transformation.md](04_Request_Shaping_and_Query_Transformation.md) | Query rewriting, multi-query, step-back | Fix the **question** before you fix the retrieval. |
| 5 | [05_Query_Routing.md](05_Query_Routing.md) | Rule/embedding/classifier/LLM/hybrid routing | Send each query to the right source/model/prompt. |
| 6 | [06_Retrieval_Gateways_and_Agentic_RAG.md](06_Retrieval_Gateways_and_Agentic_RAG.md) | Gateway, access control, caching, agentic | A security+cost+observability layer; then loop instead of single-shot. |

---

## The mental model of the whole lecture

```
                 ┌─────────── REQUEST SHAPING (Sec 4) ───────────┐
 User query ───▶ │ rewrite · follow-up · multi-query · step-back  │
                 └───────────────────────┬───────────────────────┘
                                          ▼
                 ┌─────────── QUERY ROUTING (Sec 5) ─────────────┐
                 │ rule → embedding → classifier → LLM (hybrid)  │
                 └───────────────────────┬───────────────────────┘
                                          ▼
                 ┌─────────── RETRIEVAL GATEWAY (Sec 6) ─────────┐
                 │ access control · caching · quotas · logging   │
                 └───────────────────────┬───────────────────────┘
                                          ▼
   ┌──────── RETRIEVAL (Sec 1 indexing + Sec 2 pipeline) ────────┐
   │  embed → ANN search (IVF/PQ/HNSW) → rerank (cross-encoder)   │
   └───────────────────────┬─────────────────────────────────────┘
                           ▼
                    LLM generates grounded answer  ──▶  (loop? → Agentic RAG)
```

Sections 1–2 are the **engine** (how retrieval actually works). Sections 3–6 are the **production chassis** built around that engine, ending with the leap from single-shot RAG to **Agentic RAG**.

---

## Suggested learning order

1. **Foundations:** 0 → 1 → 2 (you can build a working RAG after these).
2. **Production:** 3 → 4 → 5 → 6 (what separates a demo from a deployed system).
3. **Practice:** open `Production_RAG_HandsOn.ipynb` and map each cell back to these notes.
