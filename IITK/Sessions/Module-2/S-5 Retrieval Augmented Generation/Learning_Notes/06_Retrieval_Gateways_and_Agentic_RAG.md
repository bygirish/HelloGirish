# Section 6 — Retrieval Gateways & the Move to Agentic RAG

> Topics: securing the perimeter · the retrieval gateway · access control (RBAC/ABAC, least privilege) · cost management (FinOps) · semantic caching (+ threshold τ) · observability · the failure of single-shot RAG → multi-hop → Agentic RAG.

---

## Big picture

The router (Section 5) just mapped the query to the right database. But before we actually *touch* private company data, three enterprise risks appear — and then a deeper architectural limit: **single-shot RAG can't solve multi-step problems.** This section is the bridge from "good RAG pipeline" to **Agentic RAG**.

---

## 6.1 Securing the perimeter — three production risks

After routing maps a prompt to the correct database (e.g. `HR_Docs_DB`, `Live_Server_Logs`), three massive risks remain:

| Risk | The scary question |
|------|--------------------|
| 🔴 **Security** | "What if a junior engineer's query is routed to HR, but they ask for the CTO's salary?" |
| 🟡 **Efficiency** | "What if 50 engineers ask the same question in 10 seconds? Do we pay for 50 searches?" |
| 🔵 **Cost** | "What if a recursive bug triggers 10,000 expensive LLM evaluations?" |

**The solution: the Retrieval Gateway** — a protective "bouncer" layer between routing logic and private company data.

---

## 6.2 Retrieval Gateway — the enterprise bouncer

**What:** a centralized security & management layer sitting between the Query Router and your actual knowledge sources (vector DBs, APIs, SQL).

**Production impact:** in a corporate environment, not all data is public and not all AI queries are cheap. Without a gateway, **sensitive executive documents could be exposed** or **recursive loops could drain cloud budgets.**

**Core responsibilities:**
1. **Access Control** (AuthN / AuthZ) — §6.3
2. **Cost / Quota Management** (FinOps) — §6.4
3. **Semantic Caching** — §6.5
4. **Observability** — §6.6
5. (+ Source abstraction, query normalization)

```
User query → RETRIEVAL GATEWAY → [HR DB | Wiki/KB | Code Repo | Logs API ...]
              ├─ Access control (authN/authZ)
              ├─ Source abstraction (hide source-specific APIs)
              ├─ Caching (cut latency + cost)
              ├─ Monitoring & logging (usage, latency, errors, quality)
              └─ Query normalization (clean, standardize, enrich)
```

---

## 6.3 Access Control — enforcing least privilege

**Concept:** ensure the AI agent only retrieves documents the user has permission to read.
**Process:** the gateway intercepts identity tokens (SSO/AD) and **applies metadata filters in the Vector DB based on RBAC** *before* retrieval.

**The two-scenario test (same query, different users):**
| User | Query | Gateway action |
|------|-------|----------------|
| Junior Engineer | "What are the latest salary bands?" | **BLOCKED** — reads role, applies RBAC filter → retrieval returns **zero docs** → LLM declines politely |
| HR Director | "What are the latest salary bands?" | **APPROVED** — authorizes, retrieves sensitive PDF chunks → LLM summarizes |

**The full flow (RBAC + ABAC):**
1. User SSO login → identity provider (e.g. Azure AD) issues a **JWT** identity token.
2. User initiates query with the JWT.
3. Gateway **validates JWT, extracts roles**, evaluates policies (RBAC/ABAC) against a policy store.
4. **Dynamic metadata filter injected** into the Vector DB query:
   - Authorized (HR Director): `Search('CTO Salary') WHERE roles CONTAINS ['hr_director']` → confidential docs returned.
   - Unauthorized (Junior Eng): `Search('CTO Salary') WHERE roles CONTAINS ['junior_engineer']` → **access denied (zero docs)**, only public HR docs.
5. Route authorized query → execute vector search **with metadata filter** → auth-filtered chunks → LLM generates grounded answer.

> 🔑 **Why filter at the DB layer, not the LLM prompt?** Injecting metadata filters at the **database level** guarantees the retrieval engine *physically cannot return* unauthorized documents to the generation pipeline — a hard security barrier against data leakage. Relying on the LLM to "refuse" is unsafe (prompt-injection, jailbreaks). **(Deck quiz answer A.)**

---

## 6.4 Cost Management (FinOps)

**What:** monitor token usage, API calls, and compute cycles to prevent runaway costs — critical in multi-agent frameworks where recursion can explode.

**How:** the gateway tracks usage (per user/team/project), enforces rate limits, throttles requests, and **falls back to cheaper models.**

**DevOps example:**
- Scenario: agent runs intensive "LLM-as-a-Judge" loops on high-tier Vertex AI endpoints. Risk: accidental **$30,000 credit burn** from recursive bugs.
- Gateway action: enforces monthly token caps; at **90% budget**, throttles requests or routes non-critical tasks to cheaper local models.

---

## 6.5 Semantic Caching — speed & efficiency

**What:** store answers to previous questions. If a new query means the *exact same thing semantically*, serve the cached answer instantly.
**How:** the gateway embeds the query and checks it against a fast in-memory cache (e.g. Redis).

**DevOps example:**
- Scenario: 200 engineers ask a variation of "Is US-East down?" within 5 minutes.
- Gateway action: process the first query normally; recognize identical intent for the next 199 → serve the cached report.
- Impact: latency drops from **seconds → ms**; bypasses 199 expensive Vector DB lookups + LLM calls.

### Choosing the threshold τ (the subtle part)
A cache hit is a **similarity bet**. The threshold trades stale-answer risk against money saved:
- **Decision rule:** embed query, find nearest cached query, reuse its answer **only if** `cos(q, q_cache) ≥ τ`.
- **The asymmetry:** the two errors are *not* equal. A **miss** costs one extra lookup; a **false hit** returns a *wrong answer*. → **Bias τ high.** (`cost(false hit) ≫ cost(miss)`)
- **What you save:** if fraction `h` of queries hit, `savings = h · (retrieval + LLM)`, and latency drops to ms.

> **One line:** set τ high enough that a hit genuinely means the *same question* — the saved cost is large, but a confident wrong answer is expensive.
> **Deck quiz:** because a false hit damages trust far more than a miss costs, **bias the threshold high** (answer B) — accept a lower hit rate to guarantee accuracy on cached responses.

---

## 6.6 Observability — tracking the truth

**What:** the gateway is a **single pane of glass** for all RAG activity — it decouples the orchestration layer from the underlying databases and **logs every action**.
**How:** every query, retrieved chunk, citation, latency metric, and cache hit is logged to a central dashboard (Datadog, Splunk).

**DevOps example:**
- Scenario: an engineer complains the agent gave outdated steps to restart a Kubernetes cluster.
- Gateway action: the platform team looks up the exact **query ID**, traces the **outdated runbook** returned by the Vector DB → deletes the stale document.

> This closes the loop on Section 2's "silent failures": observability is how you *detect* and *trace* quality problems that don't throw errors.

---

## 6.7 From single-shot RAG to Agentic RAG

### The problem with "single-shot" RAG
Standard RAG is a strict **linear pipeline**: `Embed Query → Retrieve Top-K → Generate Answer`. It rests on two assumptions that break in production:
- **Assumption 1 — Perfect retrieval:** all needed context is found in a single search.
- **Assumption 2 — Complete context:** retrieved chunks contain the *final* answer, not a clue pointing elsewhere.

**Reality:** enterprise problems are rarely solved in one search. A developer runs a query, reads the result, realizes they need more context, and runs a *different* query. **Linear RAG cannot reflect, pause, or loop back.**

### Case study — the multi-hop query
> "Find the error in the payment container, check if it's related to yesterday's auth module commit, and if so, draft a Slack update for the QA team."

How linear RAG fails:
- **Vector dilution:** embedding the whole prompt at once confuses the vector DB (keywords from logs + codebase + HR tools) → irrelevant, blended results.
- **Missing sequential logic:** the system must know the *exact container error* **before** it knows what to search for in the auth commits. It cannot search both simultaneously.

### Why we can't `if/else` our way out
Engineers try massive rule-based routing trees:
```
IF query contains "error"        → Search Logs
THEN IF logs contain "auth"      → Search Codebase
THEN IF codebase contains "commit" → Draft Message
```
**The catch — fragility:** a developer typing *"payment pod crashed"* instead of *"error"* collapses the workflow. You cannot hardcode a path for every debugging scenario.

### The shift: from passive search to active problem-solving
Resolving an incident is an **iterative loop**:
1. Execute an initial search.
2. Evaluate the result.
3. Realize a piece of context is missing.
4. Plan a secondary action to fetch it.

> **The necessary shift:** move the responsibility of *how* to solve the problem away from static code and hand it to the **LLM itself**. We need a system that doesn't just *read* data, but **autonomously decides how to use our tools** — **Agentic RAG** (the next lecture / topic).

---

## 🎯 Interview questions

**Q1. What is a retrieval gateway and what does it protect against?**
A centralized layer between the router and knowledge sources handling access control, cost/quota, caching, and observability. It prevents data leakage (sensitive docs), runaway costs (recursive LLM loops), and redundant compute (duplicate queries).

**Q2. How should access control be enforced in RAG, and why there?**
By intercepting the user's identity token at the gateway and injecting a dynamic metadata filter into the vector DB query *before* retrieval (RBAC/ABAC). At the DB layer the engine physically cannot return unauthorized docs — far safer than instructing the LLM to refuse, which is vulnerable to prompt injection. (Quiz answer A.)

**Q3. Junior engineer and HR director ask the same salary question — how does the system respond differently?**
The gateway reads each user's role from their token and applies an RBAC metadata filter. Junior engineer → zero docs returned → LLM declines; HR director → authorized chunks returned → LLM summarizes. Same query, different retrieval scope.

**Q4. Explain semantic caching and how to choose its threshold.**
Cache answers keyed by query embedding; serve a cached answer if cosine similarity ≥ τ. Because a false hit (wrong cached answer) damages trust far more than a miss (just pay for the call), bias τ **high** — accept fewer hits to guarantee a hit means the same question. (Quiz answer B.)

**Q5. Why is observability essential in production RAG?**
Many RAG failures are silent (no errors). Logging every query, retrieved chunk, citation, latency, and cache hit lets you trace a bad answer to the exact stale document/query ID and fix it — closing the loop on silent failures.

**Q6. Why does single-shot/linear RAG fail on multi-hop queries?**
It assumes one search yields complete context. Multi-hop tasks need sequential dependencies (find error → then search related commit) and would suffer vector dilution if embedded all at once. Linear RAG can't reflect, pause, or loop back.

**Q7. Why not solve multi-hop with a big if/else routing tree?**
It's brittle: it relies on exact keywords ("error") and collapses on phrasing variation ("payment pod crashed"). You can't enumerate every debugging path. The fix is handing the "how" to the LLM — Agentic RAG.

**Q8. What defines Agentic RAG vs standard RAG?**
Agentic RAG lets the LLM autonomously plan and execute a multi-step loop — search, evaluate, decide what's missing, fetch more, use tools — instead of a fixed retrieve-then-generate pipeline. It adds reflection, planning, tool use, and iteration.

**Q9. How does FinOps/cost management work in the gateway?**
Track token/API/compute usage per user/team/project, enforce rate limits and budget caps, throttle at thresholds (e.g. 90%), and fall back to cheaper models for non-critical tasks — preventing runaway spend from recursive loops.

---

## 🧠 Learning thoughts

- **The gateway is where RAG becomes enterprise software.** Security, cost, caching, observability — these are the concerns that decide whether a system survives contact with real users and finance teams.
- **Enforce security at the data layer, never the prompt.** "Tell the LLM not to reveal X" is not a control; a metadata filter that returns zero rows *is*. This is the single most important security lesson in the lecture.
- **Error asymmetry drives design.** Semantic caching's τ, retrieval confidence thresholds, the hybrid router's escalation — all are decisions where one error type costs far more than the other. Always ask: *which mistake is more expensive?*
- **Observability is the cure for silent failures** (Section 2). You can't fix what you can't see; logging query IDs → traceable bad answers → deletable stale docs.
- **The whole lecture builds to one realization:** a fixed pipeline, no matter how well-tuned, can't handle problems that require *deciding what to do next*. That hands the wheel to the LLM → **Agentic RAG**. Everything (indexing, reranking, shaping, routing, gateways) becomes a *tool* the agent orchestrates.

## ✅ Self-check

1. Name the gateway's four core responsibilities and the risk each addresses.
2. Why inject access-control filters at the DB layer rather than the LLM prompt?
3. Explain the error asymmetry that makes you bias the cache threshold τ high.
4. Give the multi-hop payment-container example and explain the two ways linear RAG fails on it.
5. List the 4 steps of the iterative incident-resolution loop that motivate Agentic RAG.
