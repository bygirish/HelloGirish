# Section 3 — Production RAG & the Agentic Transition

> Topics: Enterprise RAG (text + tables + secure NL-to-SQL) · the DevOps & Support Agent case study · Normal RAG vs Production RAG.

---

## Big picture

Sections 1–2 give you a **working RAG demo**. This section is the jump to a **deployed enterprise system**. The difference isn't the retriever — it's everything *around* it: multiple data sources, security, query routing, request shaping, reranking, caching, observability, and cost control.

This section also introduces the **case study that drives the rest of the lecture**: the Enterprise DevOps & Support Agent.

---

## 3.1 Enterprise RAG: Text + Tables + Secure NL-to-SQL

Real enterprises have **two kinds of data**, and naive RAG only handles one:

| Data type | Examples | How to handle |
|-----------|----------|---------------|
| **Unstructured (text)** | PDFs, docs, emails | Standard RAG: extract → clean → chunk (semantic) → embed → vector store |
| **Structured (tables)** | databases, sheets, data warehouses | **NL-to-SQL** or table embeddings — *don't* embed raw numbers for semantic search |

### Why separate them?
> **Vectors are great for *meaning*, not for *exact numbers*.** Asking "what was APAC revenue in Q3?" against an embedded table gives fuzzy nonsense. SQL is reliable for calculations, aggregations, rankings.

### The architecture (high level)
```
DATA SOURCES → INGESTION → QUERY UNDERSTANDING → RETRIEVAL & EXECUTION → FUSION & GENERATION
(text+tables)  (2 pipelines) (intent + router)    (Text RAG | Table NL-to-SQL)  (merge, cite)
```

- **Query understanding & routing:** classify the query (Text / Table / Hybrid), extract entities, route to the right path.
- **Path A — Text RAG:** vector search → top relevant chunks.
- **Path B — Table RAG (NL-to-SQL):** LLM generates SQL → **SQL Validation Layer** → execute read-only → structured results.
- **Fusion & generation:** merge + rank info from all sources → LLM generates a **cited** answer.

### Table handling strategies
| Strategy | Best for |
|----------|----------|
| **Table summarization** (convert table → natural-language summary, embed) | explanations, trends, insights |
| **Row/cell embeddings** (embed each row, keep text+SQL) | lookups, filtering, entity-level retrieval |
| **Hybrid (Vector + SQL)** (summaries in vector DB, raw tables in SQL) | analytics, aggregations, precise numeric queries |

### Secure NL-to-SQL — the guardrails
Letting an LLM write SQL against your prod DB is dangerous. The **SQL Validation Layer** enforces:
- **Read-only access** — no writes.
- **Schema whitelisting** — only approved tables/columns/functions exposed.
- **Query type = SELECT only** — block `DROP/DELETE/UPDATE/INSERT/ALTER`.
- **Row-Level Security (RLS)** — users see only their own data.
- **Complexity & result-size limits** — prevent runaway queries.
- **Audit logs** — log every query for monitoring.

**Key takeaways:** use the right tool for the right data type; separate retrieval for text vs structured; always validate & secure NL-to-SQL; enforce least-privilege & governance; provide accurate, auditable, **cited** answers.

---

## 3.2 The Enterprise DevOps & Support Agent (the case study)

**Goal:** build an internal AI agent for software engineering teams.
**Key requirements:** troubleshoot incidents, query the codebase, answer HR queries, and ensure security.

This single agent is the backdrop for every later concept. It spans **multiple knowledge sources**, each with a purpose-built retrieval strategy:

| Knowledge source | Example content | Retrieval style |
|------------------|-----------------|-----------------|
| **Code repositories** (GitHub/GitLab) | proprietary code, PRs, docs | hybrid search (embedding + BM25) |
| **Live network & server data** | logs, metrics, traces, alerts | time-series / API queries |
| **HR policies & docs** | handbooks, policies, procedures | vector / semantic search |
| **Knowledge base & runbooks** | operational runbooks, FAQs, best practices | semantic search |
| **Third-party tools & APIs** | Jira, PagerDuty, GitHub, ServiceNow | API / graph queries |

The layered reference architecture (recurring diagram):
1. **Request shaping** — rewrite vague queries (Section 4).
2. **Agentic orchestration** — task planner, episodic + long-term memory, action executor.
3. **Routing & gateway** — intent classifier, query router, retrieval gateway (Sections 5–6).
4. **Knowledge sources** — the data silos above.
5. **Inference & evaluation** — context assembler, LLM inference, **LLM-as-a-judge** (grounding/accuracy/safety check) with pass/retry loop.

Plus **cross-cutting controls**: Identity & Access (RBAC/ABAC/SSO), policy & guardrails (PII redaction, content filtering), observability, cost/FinOps, audit & compliance.

---

## 3.3 Normal RAG vs Production RAG — what's different

| | **Normal RAG** | **Production RAG** |
|---|----------------|--------------------|
| Flow | Query → retriever → vector DB → LLM → answer | Query → **router** → **gateway** → multi-source → **reranker** → LLM → answer |
| Data sources | single vector DB | **multiple** (HR docs, Wiki/KB, codebase, logs…) — smart routing |
| Security | none | **Retrieval gates**: auth, access control, caching, logging |
| Query handling | raw query in | **Request shaping**: rewrite, expand, filter before retrieval |
| Retrieval quality | top-k similarity | **reranking, deduplication, context compression** |
| Concerns | "does it work?" | scalability, latency, reliability, **observability** |

**What's new in production RAG (the 5 additions):**
1. **Multiple data sources** — route queries to the right source.
2. **Retrieval gates** — auth, access control, caching, logging, monitoring.
3. **Request shaping** — rewrite/expand/filter queries before retrieval.
4. **Better retrieval quality** — reranking, dedup, context compression.
5. **Production concerns** — scalability, latency, reliability, observability.

> This table *is* the roadmap for Sections 4–6: request shaping (4), routing + gates (5–6), and the production concerns woven throughout.

---

## 🎯 Interview questions

**Q1. Why can't you treat structured tables like unstructured text in RAG?**
Embeddings capture semantic meaning, not exact numeric values/aggregations. "Sum of Q3 APAC revenue" needs SQL precision; an embedded table gives fuzzy, unreliable answers. Route numeric/analytical queries to NL-to-SQL, text queries to vector search.

**Q2. What guardrails make NL-to-SQL safe in production?**
Read-only connection, schema whitelisting (approved tables/columns/functions), SELECT-only enforcement (block DML/DDL), row-level security, complexity/result-size limits, and audit logging. The validation layer checks every generated query *before* execution.

**Q3. Name the main differences between a RAG prototype and production RAG.**
Prototype: single vector DB, raw query, no security. Production adds query routing across multiple sources, a retrieval gateway (auth/cache/logging), request shaping, reranking/dedup/compression, and observability/scalability/cost controls.

**Q4. What are the three table-handling strategies and when use each?**
Table summarization (→ NL summary, embed) for trends/explanations; row/cell embeddings for lookups/entity retrieval; hybrid vector+SQL (summaries in vector DB, raw in SQL) for analytics and precise numeric queries.

**Q5. In the DevOps agent, why does each knowledge source get a different retrieval strategy?**
Because the data shape differs: code → hybrid (semantic + keyword for symbols); logs → time-series/API; policies/runbooks → semantic; tickets → API/graph. One-size retrieval underperforms across heterogeneous sources.

**Q6. What does "cited / auditable answer" buy an enterprise?**
Trust and compliance: users can verify claims, auditors can trace decisions, and you can debug wrong answers back to the source document (ties into observability).

---

## 🧠 Learning thoughts

- **Production RAG is a systems problem, not a model problem.** The retriever is maybe 20% of the work; routing, security, caching, observability, and cost are the other 80%.
- **"Right tool for the right data" is a recurring theme** — text→vectors, numbers→SQL, code→hybrid. Forcing everything through one mechanism is the classic naive-RAG mistake.
- **Security can't be bolted on later.** Read-only, whitelisting, RLS, and access control must be architectural. (Section 6's gateway makes this concrete.)
- The **LLM-as-a-judge** evaluation loop hints at the agentic shift: the system *checks its own output* and retries — a precursor to Agentic RAG (Section 6).
- Keep the **DevOps agent case study** in your head as a worked example. Every later abstraction ("step-back prompting", "embedding router", "semantic cache") has a concrete instance in this agent — that's how to remember them.

## ✅ Self-check

1. Draw the Normal RAG vs Production RAG flows and list the 5 production additions.
2. List the NL-to-SQL guardrails from memory.
3. For each DevOps-agent knowledge source, name its retrieval strategy.
4. When would you summarize a table vs embed its rows vs keep it in SQL?
