# Section 8 — Provenance & Citations

> **Goal of this section:** Learn how to make an autonomous multi-agent system **trustworthy and auditable** — what *provenance* is, why citations are non-negotiable in the enterprise, the **four citation architectures** (and their trade-offs), and how to **verify** citations programmatically.

**Topics covered**
1. Provenance & Citations — building trust
2. What is Provenance? / Why Citations Matter
3. The Spectrum: four citation architectures
4. Method 1: Inline Prompting
5. Method 2: Structured Tool Forcing
6. Method 3: Post-Hoc Attribution
7. Method 4: Agentic State Provenance
8. The Trade-off Matrix
9. Project citations, UI/UX, and Verification
10. ✅ Self-test MCQ

---

## 1. Provenance & Citations — Building Trust

> Building trust in autonomous multi-agent systems means establishing a **verifiable chain of custody** for every decision and discovery made within the swarm.

**The accountability requirement:** if the Triage agent attributes an incident to a specific code commit, you must be able to identify **exactly which sub-agent made the discovery** and the **specific location of the evidence.**

---

## 2. What is Provenance? / Why Citations Matter

### What is provenance?

| General Definition | AI Provenance | How It's Achieved |
|---|---|---|
| The **origin/source** of a piece of information | Mapping **every claim** in a generated answer back to specific components: specific **documents/log lines**, and **tool outputs / sub-agent actions** | Via **Citations** — establishing a verifiable **chain of custody** for every autonomous decision |

> Without provenance, a multi-agent system is just a **massive black box generating text.**

### Why citations matter

| Reason | Detail |
|---|---|
| **Verification** | Human engineers must be able to **audit the AI's logic** |
| **Hallucination Mitigation** | Forcing agents to cite sources **drastically reduces fabricated answers** |
| **Accountability** | Know whether an error came from the **Codebase Agent's analysis** or the **Infra Agent's log pull** |

> In enterprise support, **an uncited answer is an unusable answer.**

💡 **Learning Thought:** Citations do *double duty*. They're not just for the human reader's trust — the *act of forcing* the model to cite **reduces hallucination at generation time** (the model can't fabricate as freely when it must point to a real source). So provenance is both an *audit* feature and a *quality* feature. In multi-agent systems, provenance also answers "which agent is to blame?" — essential for debugging the *team*, not just the answer.

---

## 3. The Spectrum of Citation Architectures

> Generating reliable citations is **not a solved problem.** There are four primary approaches, ranging from **easy-but-flawed** to **complex-but-robust**:

```
1. Inline Prompting          2. Structured Tool Forcing
   (Zero/Few-Shot)              (JSON Schema)

3. Post-Hoc Attribution      4. Agentic State Provenance
   (Retroactive Verify)        (LangGraph tracing)
```

Each is judged on three axes: **latency**, **implementation cost**, and susceptibility to **"citation hallucination"** (citing a document that doesn't exist or doesn't support the claim).

---

## 4. Method 1 — Inline Prompting (The Baseline)

- **Mechanism:** instruct the LLM in the system prompt to **append bracketed citations directly into the text stream.**
- **Prompt example:** *"Use the provided context. After every sentence, cite the source like this: [Doc_1]."*

| Pros | Cons |
|---|---|
| Very **low latency** (streams naturally) | **Highly prone to hallucination** |
| **Easiest** to implement | Relies **entirely on instruction-following** |

> Early RAG systems relied on this, but performance **degrades quickly as context windows fill up.**

---

## 5. Method 2 — Structured Tool Forcing (The Enterprise Standard)

- **Mechanism:** force the LLM to output a **strict JSON schema** where the answer and citations are **separated into distinct fields.**

```json
{
  "answer": "The server failed due to an OOM error.",
  "citations": [ {"doc_id": "splunk_log_42", "exact_quote": "..."} ]
}
```

| Pros | Cons |
|---|---|
| Enables **programmatic verification** of exact quotes against source text before display | **Cannot be easily streamed**; higher latency (must wait for the full JSON object) |

> Forcing **`exact_quote`** creates a **deterministic bridge**: if the quote isn't found in the source, the UI can flag it as **invalid.**

💡 **Learning Thought:** The `exact_quote` field is the clever bit. A bare `doc_id` can still be hallucinated, but an *exact quote* can be **string-matched against the actual source** — turning a fuzzy trust problem into a deterministic check. This is why it's the "enterprise standard" for document RAG: it's verifiable without a second model.

---

## 6. Method 3 — Post-Hoc Attribution (The Fact-Checker)

- **Mechanism:** **decouple generation from citation.** A fast LLM generates the answer; **then** a secondary **NLI (Natural Language Inference) model** or smaller LLM injects citations retroactively by comparing the answer against source docs.

| Pros | Cons |
|---|---|
| **Highest accuracy** | **Doubles the compute cost** |
| Generator isn't distracted by citation-formatting rules | **Heavily increases latency** |

> Academic approach gaining traction in **high-stakes** environments (legal/medical): one model **creates fluently**, a strict logical model **fact-checks and cites.**

💡 **Learning Thought:** Post-Hoc embodies a "separation of concerns" *within citation* — the same pattern as planner/executor. One model optimizes for fluency, another for rigor. You pay double compute for the highest reliability. Use it where a wrong citation is catastrophic (a legal brief, a medical recommendation), not for a consumer chatbot.

---

## 7. Method 4 — Agentic State Provenance (The LangGraph Approach)

- **Mechanism:** citations are **NOT generated by the LLM's text output** — they're **derived from the execution graph itself.**
- **How it works:**
  ```
  Triage delegates to Codebase Agent
    > Agent calls GitHub_Tool & gets Commit_88X
    > State records: Message(content="Commit_88X...", name="Codebase_Agent")
  ```

| Pros | Cons |
|---|---|
| **100% deterministic** — the citation is a **literal log of the API call** that occurred in the state machine | Only works for **discrete tool calls**, not for synthesizing large paragraphs of retrieved text |

> **Most secure method for DevOps.** The UI builds a visual audit trail **from LangGraph State history**, bypassing LLM text uncertainty.

💡 **Learning Thought — the payoff of good state design:** This is where the `sender`/`name` fields (Section 6) and the provenance-tagged context vector (Section 4) pay off. Because every tool call and sub-agent action was *already recorded in the state*, the citation is just a **read of the execution log** — the LLM is never trusted to cite because *it doesn't have to.* The graph *is* the citation. This is the deepest idea in the section: **provenance as a byproduct of architecture, not of prompting.**

🎯 **Interview Question:** *"Why is Agentic State Provenance considered 100% deterministic while inline prompting is not?"*
**Model answer:** Inline prompting asks the LLM to *write* citations as text, so a citation is only as reliable as the model's instruction-following — it can fabricate a doc ID that doesn't exist. Agentic State Provenance doesn't ask the model at all: it **derives citations from the recorded execution graph** (which tool was called, by which agent, returning what). Since those are literal, logged facts of what the state machine actually did, the citation is a deterministic read of history — there's no generative step to hallucinate. The limitation is that it only covers **discrete tool calls**, not free-text synthesis of retrieved passages.

---

## 8. The Trade-off Matrix (memorize this)

| Method | Implementation | Latency | Accuracy / Reliability |
|---|---|---|---|
| **Inline Prompting** | Low (text only) | **Very Fast** | **Low** (prone to hallucination) |
| **Structured Output** | Medium (JSON parsing) | Moderate | **High** (exact-quote verification) |
| **Post-Hoc Attribution** | High (multi-model pipeline) | **Slow** | **Very High** (strict entailment) |
| **Agentic State** | High (LangGraph setup) | Graph-dependent | **Absolute** (deterministic tool logs) |

> **The practical recommendation:** for a **consumer chatbot**, Inline Prompting is often enough. But for the **Enterprise DevOps agent**, use a **combination**: **Structured Outputs** for *document RAG* and **Agentic State** for *API tool executions.*

💡 **Learning Thought:** There's no single "best" method — it's a **latency × cost × reliability** trade-off matched to stakes. The senior-engineer answer is almost always *"combine them"*: cheap methods where errors are tolerable, expensive/deterministic methods where they're not. Notice the recommended combo maps to the *two kinds of evidence* the DevOps agent produces — retrieved documents (→ structured/exact-quote) vs tool-call results (→ agentic state log).

---

## 9. Project Citations, UI/UX & Verification

### Project context — a multi-agent citation in the wild
> *"The payment container failed due to an Out-of-Memory error [Infra Agent: Splunk-pod-1, Line 452]. This aligns with yesterday's commit increasing the cache size [Codebase Agent: Git Commit 8f4b2a]."*

Every claim maps to **(which agent) + (which exact source location).** That's provenance realized.

### The UI/UX of citations
- **Don't just show `[1]`.** **Hyperlink to enterprise systems** — link directly to the live **Splunk dashboard** or **GitHub PR** so a human can verify in one click.
- *User experience drives adoption.* Make verification **frictionless** to build trust.

### Post-processing — verifying citations
- **Problem:** LLMs sometimes hallucinate citations (e.g., a commit hash that wasn't in the Codebase Agent's payload).
- **Solution — State-Verification Step:** add a lightweight code check ensuring **every cited ID actually exists in the State's message history.**
- **Core principle:** **Never trust the LLM blindly. Always verify the citation format programmatically before showing it to the user.**

### Connecting provenance to state machines
- **Native provenance:** LangGraph makes provenance easier via the **persistent state** of graph execution.
- **Chronological audit trail:** every sub-agent execution and tool call is saved in State → a **perfect chronological record** of what the team did.
- **Visualizing the "thought process":** render the LangGraph execution path in the UI to show the multi-agent reasoning journey.

🎯 **Interview Question:** *"Even with structured citations, an agent occasionally cites a source that wasn't retrieved. Cheapest robust safeguard?"*
**Model answer:** A **state-verification post-processing step**: before rendering, programmatically check that **every cited ID exists in the State's message history / retrieved payload**, and drop or flag any that don't. It's cheap (pure code, no extra model call), deterministic, and catches citation hallucination regardless of the generation method. Pair with `exact_quote` matching against source text for document citations. The governing rule: never trust the model's citation blindly — verify against the recorded state.

---

## 10. ✅ Self-test MCQ (from the lecture)

### MCQ 5 — Inline-only citations
> *A team uses only Inline Prompting for citations in its enterprise DevOps assistant, reasoning "the model is reliable enough." What two risks does this leave open, and what's a minimal fix (short of full Post-Hoc Attribution)?*
>
> **A)** Tool hallucination; state bloat. Fix: add Summarize node.
> **B)** Citation hallucination + stale memory cited as current. Fix: verify cited IDs vs. message history + timestamp/prune memory.
> **C)** Context token limits; recursion limit. Fix: raise both.
> **D)** Citations can't be hyperlinked; too slow. Fix: switch fully to Post-Hoc.

<details><summary>Answer & reasoning</summary>

**Answer: B.** These are the two failure modes tied to *this exact choice*: Inline Prompting has **no safeguard against fabricated citations**, and **uncurated long-term memory can surface outdated facts as current.** The minimal fix mirrors the lecture's countermeasures — **state-verification** of cited IDs + **timestamp/prune** memory — getting most of Post-Hoc's safety without its full cost.
- **A**'s risks are real elsewhere but are caused by tool count/message growth, not the citation method.
- **C** conflates unrelated capacity constraints with a citation decision.
- **D** is factually wrong (inline citations are *fast*, not slow) and proposes the non-minimal fix the question said to avoid.
</details>

---

## Section 8 — Consolidated Takeaways

- **Provenance** = mapping every claim back to its source (document/log line/tool output/sub-agent); achieved via **citations** = a verifiable chain of custody. *An uncited answer is unusable.*
- Citations serve **verification, hallucination mitigation, and accountability** — an audit feature *and* a quality feature.
- **Four architectures**, easy→robust: **Inline** (fast, hallucination-prone) → **Structured/exact-quote** (verifiable, enterprise standard) → **Post-Hoc/NLI** (highest accuracy, 2× cost + latency) → **Agentic State** (deterministic log of tool calls; only for discrete calls).
- Know the **trade-off matrix** (implementation × latency × reliability). Real systems **combine** methods: Structured for doc RAG + Agentic State for tool calls.
- **Agentic State Provenance** is the deepest idea: the **graph *is* the citation** — provenance as a byproduct of good state design, not of prompting.
- Always add a **state-verification step**: check every cited ID exists in state before display. **Never trust the LLM's citations blindly.** Hyperlink citations to live systems for frictionless verification.

**Next:** [Section 9 — Best Practices & Key Takeaways](10-Section-9-Best-Practices-and-Takeaways.md) — the production checklist and the four strategic pillars.
