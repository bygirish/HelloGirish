# Section 5 — Query Routing

> Topics: the need for dynamic decisions · query routing concept · 5 routing strategies (rule / embedding / classifier / LLM / hybrid) · 3 routing applications (data-source / model-&-workflow / prompt-&-index).

---

## Big picture

Section 4 gave us four query-transformation techniques and (in the case study) multiple data sources. That creates a **decision problem**:

> Which transformation? Which data source (HR vs codebase vs logs)? Which model (cheap vs expensive)?

Hardcoding `if/else` for every scenario is **brittle**. **Query Routing** is the intelligent layer that dynamically directs each query to the optimal workflow, tool, data source, model, or prompt.

**The goal:** *maximize accuracy while minimizing latency and API cost.*

---

## 5.1 The need for dynamic decisions

**Immediate problem:** we have four transformation techniques (rewriting, follow-ups, multi-query, step-back) — but how does the system know *which* to apply?

**Broader challenge** (as the DevOps agent scales):
- Which transformation technique should I use?
- Search the HR vector DB or the GitHub code repository?
- Use a fast/cheap LLM or a capable/expensive reasoning model?

**Solution = Query Routing.** Hardcoding `if/else` creates a brittle system; we need a layer that **dynamically routes** requests.

---

## 5.2 Query Routing — the "traffic cop"

**What:** routing directs a user query to the most relevant **knowledge source, workflow, or model** *before* execution.

**Production impact ("so what"):** if a user asks a simple question, searching the *entire* company infrastructure (HR, codebase, logs, wiki) is slow, expensive, and **increases hallucination** by pulling irrelevant data. **Routing isolates the search to only the necessary domains.**

**Routing destinations (examples):** Text RAG, SQL Engine, Knowledge Graph, Search Engine, Tool/API (weather/stocks/calculator), Code Interpreter.

**Example routing decisions:**
| Query | Route |
|-------|-------|
| "What is our PTO policy?" | Policy docs (Vector RAG — HR KB) |
| "What was APAC revenue in Q3?" | SQL Engine (Finance warehouse) |
| "Who reports to the VP of Engineering?" | Knowledge Graph (Org graph) |
| "What's the weather in London?" | External Tool (Weather API) |
| "Why did APAC revenue drop?" | **Multi-route** (SQL numbers + Docs commentary) |

---

## 5.3 The 5 routing strategies (simple → smart)

| Strategy | How it decides | Pros | Cons |
|----------|---------------|------|------|
| **1. Rule-Based (logical)** | hardcoded logic / regex / keywords | fast, deterministic, **$0**, ~1ms | brittle; fails on phrasing variation; no semantics |
| **2. Embedding-Based** | embed query, cosine-match against **route profiles** | fast, scalable, offline, no LLM | needs good profiles; struggles with true ambiguity |
| **3. Classifier-Based** | trained lightweight ML model (Random Forest, Naive Bayes, small BERT) → intent buckets | learns patterns, fast at runtime, confidence scores | needs labeled training data + retraining |
| **4. LLM-Based** | LLM reads query + options, outputs JSON decision | highly accurate on ambiguity; flexible | adds latency + API cost |
| **5. Hybrid (the pro approach)** | **waterfall:** rule → embedding → LLM | cost-efficient + robust | more components to build |

---

## 5.4 Rule-Based (Logical) Routing

**What:** the most fundamental technique — hardcoded logic, keywords, or regex mapping a raw query string against `if/else` conditions *before* AI processing.

**DevOps example:**
```python
IF query.contains(["PTO", "leave", "payroll", "manager"]): route_to = "HR_Database"
```
- User: "How many days of leave do I have left?" → instant route to HR docs. **Cost: $0 · Latency: ~1ms.**

**The catch — fragility:** brittle; no semantic understanding; fails if exact keywords are missing.
- User types: *"I need to take time off"* → **REGEX FAILS** (missing "leave"/"PTO").

> Profile: *Deterministic · Zero cost · Extremely low latency · Low generalization.*

---

## 5.5 Embedding-Based Routing

**What:** transform queries into vectors and similarity-check against **"route profiles."** Deterministic, fast, no LLMs, works offline.

**How it's done:** create a profile (20–50 sample questions) for each database/route. Embed real queries and match via **cosine similarity** to the closest profile.

```
Query "OOMKilled error code logic" → embed →
  Profile A (HR policies)     0.02
  Profile B (Company codebase) 0.92  ◀── highest → route here
  Profile C (Live network logs) 0.30
```

**DevOps example:**
- User: "I'm feeling burnt out and need a break next Tuesday."
- Profiles: HR_Profile (benefits, time off) vs Code_Profile (bugs, PRs).
- Result: semantic match favors **HR_Profile** — routed correctly **without a heavy LLM**. (Note: rule-based regex would *fail* here — no keyword "leave"/"PTO".)

> This is why embedding routing beats rule-based: it captures *meaning*, not just keywords.

---

## 5.6 Classifier-Based Routing

**What:** a trained, lightweight ML model (Random Forest, Naive Bayes, small BERT, XGBoost) categorizes query intent into predefined buckets — HR, Legal, Code, Ops.

**How it's done:** train on historical queries mapped to correct labels. At runtime, ingest a query → output **probability scores** per category.

**DevOps example:**
- Query: "My compensation seems wrong this month." → recognizes semantic pattern → **Intent: HR (95% confidence).**

> Sits between embedding routing (no training, profile-based) and LLM routing (no training, expensive): you invest in *training* once to get fast, confident, learned routing.

---

## 5.7 LLM-Based Routing

**What:** use an LLM as the router — simple queries → cheaper models; complex ones → stronger models. Ask the LLM to reason and output **structured JSON**.

**How it's done:** a prompt with the query + route descriptions asks: *"Analyze this query and return a JSON object with 'target_database' and 'reasoning'."*
```json
{ "selected_model": "GPT-4", "reason": "query requires detailed explanation and reasoning" }
```

**DevOps example:**
- Query: "The auth token is expiring too fast. Is this a new security policy thing or a bug?"
- Result: LLM recognizes technical execution over policy → outputs `{"route": "Codebase_DB"}`.

**The catch:** balances quality vs cost dynamically, but **adds latency and API cost** to the pipeline. (Don't pay it on every query — see hybrid.)

---

## 5.8 Hybrid Routing (the pro approach)

**What:** combine strategies to leverage each one's strengths while mitigating weaknesses.
**How:** routers structured **sequentially as a waterfall** — move to the next stage *only* on **low confidence**.

```
1. Rule-Based  ──fail──▶  2. Embedding-Based  ──low conf──▶  3. LLM-Based
```

**DevOps example — "Pipeline failed.":**
| Step | Result |
|------|--------|
| 1. Rule-Based | **FAIL** — no obvious keywords |
| 2. Embedding | **50% confidence** — ambiguous (CI/CD vs Sales) → fall through |
| 3. LLM-Based | **SUCCESS** — reviews query + metadata (Role: DevOps) → routes to Infrastructure_Logs |

> **Cost efficiency:** you only pay for the expensive LLM router *when the cheaper methods failed.* This is the same "cheap-first, escalate-on-uncertainty" pattern as the retrieve-cheap-rerank-dear idea from Section 2.

---

## 5.9 The 3 applications of routing

Routing isn't just for picking documents:

### Application 1 — Data Source Routing
Select the correct vector DB, SQL DB, or API by intent.
| Query | Route |
|-------|-------|
| "What is the maternity leave policy?" | **HR Docs Vector DB** |
| "How does the auth module handle token expiry?" | **Company Codebase Vector DB** |
| "Why did the payment container crash 5 min ago?" | **Live Splunk/Datadog logs** (bypass Vector DB entirely) |

> Intent-based routing targets the most relevant high-fidelity source for each request.

### Application 2 — Model & Workflow Routing
Dynamically pick *which LLM* or *which pipeline*, balancing quality vs cost.
| Query | Route |
|-------|-------|
| "Format this server log into a JSON block." (simple extraction) | smaller/cheaper model (Llama-3-8B, Haiku) — lowers latency, avoids wasting capacity |
| "Review this 500-line Python PR against our security guidelines, draft a summary." (complex reasoning) | premium reasoning model (GPT-4, Opus) → triggers multi-step **agentic orchestration** |

### Application 3 — Prompt & Index Routing
Decide *how* to search or *how* to prompt once data is found.
| Query | Route |
|-------|-------|
| "Find the specific error code in the network manual." | **Sparse index (BM25/keyword)** — exact match matters more than semantics |
| "Summarize the architectural changes made to the database last quarter." | **Summarization prompt template** — aggregate, don't extract a single fact |

### Advanced production routing techniques (deck)
- **Retrieval confidence routing** — check confidence after retrieval (e.g. top score 0.42 = low) → search another source / increase top-k / ask for clarification / escalate to a human.
- **Cost-aware routing** — simple query → small/cheap model; complex → large model.
- **Dynamic top-k routing** — adjust k by query type (fact/lookup k=3–5; general k=10–20; complex/analytical k=30–50).
- **Query decomposition routing** — break complex queries into sub-queries, route each, merge (overlaps Section 4 multi-query).
- **Agentic / planner routing** — LLM plans a multi-step plan (query SQL → query docs → query graph → synthesize).

**Failures to avoid:** wrong route → irrelevant results; over-routing → high latency/cost; under-routing → incomplete answers; no fallback → failed user experience.

---

## 🎯 Interview questions

**Q1. What is query routing and why does production RAG need it?**
A layer that sends each query to the optimal source/model/prompt before execution. Without it, you'd search all sources for every query → slow, costly, more hallucination. Routing isolates search to relevant domains, maximizing accuracy while minimizing latency/cost.

**Q2. Compare the five routing strategies.**
Rule-based (regex; fast/free/brittle), embedding-based (cosine vs route profiles; semantic, offline, no LLM), classifier-based (trained ML; learned, confidence scores, needs data), LLM-based (JSON decision; most flexible, costly/slow), hybrid (waterfall rule→embedding→LLM; cheap-first, escalate on low confidence).

**Q3. A support agent's rule-based router is too brittle but an LLM router is too expensive to run on every query. Best solution?**
Hybrid routing waterfall: rule → embedding → LLM. Try cheap deterministic methods first; only invoke the expensive LLM router when ambiguity (low confidence) requires it. (Deck quiz answer A.)

**Q4. Why does embedding routing succeed where rule-based fails?**
It matches on semantic meaning via cosine similarity to route profiles, so "I need to take time off" or "feeling burnt out" still routes to HR even without keywords like "PTO"/"leave."

**Q5. How are route profiles built for embedding routing?**
Create 20–50 representative sample questions per route/database, embed them as the profile; embed incoming queries and route to the highest cosine-similarity profile.

**Q6. Give three distinct applications of routing beyond data-source selection.**
Model/workflow routing (cheap vs premium model; simple vs agentic pipeline), prompt routing (extraction vs summarization template), index routing (dense vs sparse/BM25). Plus advanced: confidence-based, cost-aware, dynamic top-k, decomposition, agentic planner.

**Q7. What is cost-aware / model routing and why does it matter?**
Route simple tasks (log formatting) to small cheap models and complex reasoning (code review) to premium models. Optimizes the cost/quality trade-off instead of paying premium-model rates for trivial work.

**Q8. When would you route to a sparse (BM25) index instead of dense retrieval?**
When exact matches matter more than semantics — error codes, specific identifiers, rare jargon in a manual. Dense retrieval can miss exact tokens; sparse nails them.

---

## 🧠 Learning thoughts

- **Routing is the brain of production RAG.** Sections 1–2 retrieve; Section 4 reshapes; Section 5 *decides*. It's where cost, latency, and accuracy are actually traded.
- The **waterfall pattern (cheap → escalate on low confidence)** is the single most reusable idea here, and it echoes "retrieve cheap, rerank dear" (Sec 2) and "rule before LLM" everywhere. Spend expensive compute *only when uncertainty demands it.*
- **"Don't search everything"** is both a cost and a *quality* argument — irrelevant sources actively increase hallucination, not just latency.
- Routing decisions need **confidence scores**, not just labels — that's what makes the waterfall and fallbacks possible. A router that can't say "I'm unsure" can't escalate.
- The progression rule→embedding→classifier→LLM is a ladder of **(cost, capability)**. Know each rung so you can justify *why* you stopped at a given one.

## ✅ Self-check

1. List the 5 routing strategies with one pro and one con each.
2. Trace the "Pipeline failed." query through the hybrid waterfall.
3. Why does "feeling burnt out" route to HR under embedding routing but fail under rule-based?
4. Give a query that should route to: a SQL engine, a BM25 index, a cheap model, and a human escalation.
