# Module 2 · Lecture 6 — Agentic AI & Multi-Agent Orchestration

> **Instructor:** Prof. Pawan Goyal, Dept. of CSE, IIT Kharagpur
> **Deck title (as filed):** "Retrieval Augmented Generation (RAG) — Module 2 Lecture 6"
> **Real subject:** *Agentic* RAG — how retrieval stops being a fixed pipeline stage and becomes an **action an LLM chooses** inside a reasoning loop, and how you orchestrate a **team of specialized agents** to solve complex, multi-hop problems reliably.

These notes turn the 97-slide deck into a structured, self-study course. Each file is one **section**, and inside it every **topic** is elaborated with:

- **The core idea** (plain English first)
- **Deep elaboration** (the "why", the mechanics, the trade-offs)
- **The running example** — an *Enterprise DevOps Support Agent* that autonomously debugs failing production systems
- 💡 **Learning Thought** callouts — the mental models worth internalizing
- 🎯 **Interview Questions** with model answers
- ✅ **Self-test MCQs** (the exact scenario checkpoints from the lecture)

---

## The running example (used everywhere)

Throughout the lecture, one project ties every abstract concept to something concrete:

> **The Enterprise DevOps & Support Agent** — an AI system whose job is to *autonomously debug failing production pipelines*. It has tools like a Vector DB of **runbooks**, the **Splunk** API for logs, the **GitHub** API for commits, **Datadog**, **Kubernetes**, and company **HR/policy** documents. Its goal is to move from "fetch a document and read it" to "investigate, reason across systems, act, and remember."

Keep this project in mind — every concept (tools, planners, ReAct, LangGraph, memory, citations) is introduced as *"here's how the DevOps agent uses it."*

---

## How to use these notes

1. **Read in order.** The sections build on each other: motivation → single agent → multi-agent → orchestration → the ReAct engine → the LangGraph runtime → memory → trust/citations → best practices.
2. **Do the MCQs before reading the answer.** They are the highest-signal comprehension checks.
3. **Pair with the notebook.** `Lecture6_Agentic_RAG_Hands_On.ipynb` in the parent folder is where the theory becomes runnable LangGraph code.
4. **Rehearse the interview questions out loud.** They are written the way an interviewer would actually probe the concept.

---

## Section map

| # | File | What you'll master |
|---|------|--------------------|
| 0 | [`01-Section-0-Motivation-Agentic-RAG.md`](01-Section-0-Motivation-Agentic-RAG.md) | Why linear RAG breaks; the Pipeline→Agent paradigm shift; what "Agentic RAG" means |
| 1 | [`02-Section-1-Foundations-of-an-Agent.md`](02-Section-1-Foundations-of-an-Agent.md) | LLM vs Agent; the Observe–Think–Act loop; **tool calling**; how to design a tool |
| 2 | [`03-Section-2-Multi-Agent-Systems.md`](03-Section-2-Multi-Agent-Systems.md) | Single vs multi-agent; context exhaustion; the specialist-team pattern |
| 3 | [`04-Section-3-Planner-Executor-Orchestration.md`](04-Section-3-Planner-Executor-Orchestration.md) | Planner/Executor separation; re-planning; the orchestrator role |
| 4 | [`05-Section-4-ReAct-and-Hierarchical-ReAct.md`](05-Section-4-ReAct-and-Hierarchical-ReAct.md) | ReAct formalism; CoT's hallucination cliff; hierarchical/meta ReAct |
| 5 | [`06-Section-5-State-Machines-and-LangGraph.md`](06-Section-5-State-Machines-and-LangGraph.md) | State machines; LangChain vs LangGraph; State/Nodes/Edges |
| 6 | [`07-Section-6-Building-the-Multi-Agent-Graph.md`](07-Section-6-Building-the-Multi-Agent-Graph.md) | Building the graph end-to-end; `compile()`; HITL; loops & state-bloat caveats |
| 7 | [`08-Section-7-Agentic-Memory.md`](08-Section-7-Agentic-Memory.md) | Episodic vs semantic memory; self-healing; memory caveats |
| 8 | [`09-Section-8-Provenance-and-Citations.md`](09-Section-8-Provenance-and-Citations.md) | Provenance; the 4 citation architectures; verification |
| 9 | [`10-Section-9-Best-Practices-and-Takeaways.md`](10-Section-9-Best-Practices-and-Takeaways.md) | Error handling, system prompts, the 4 key takeaways |

---

## The one-paragraph summary of the whole lecture

> Linear RAG (embed → retrieve top-K → generate) assumes retrieval is perfect and context is complete — assumptions that collapse on real, multi-hop problems. **Agentic RAG** fixes this by making retrieval, routing, and query-rewriting into **tool calls inside an Observe–Think–Act (ReAct) loop**, so the LLM decides *whether, what, and when* to retrieve. As tool counts grow past ~20, a single agent starts hallucinating tool calls, so we split work into a **team of specialist agents** coordinated by a **Triage/orchestrator** — separating **planning** (expensive reasoning model) from **execution** (cheap, fast models). To run this reliably we replace brittle `while` loops with a **state machine** implemented in **LangGraph** (State + Nodes + Edges + cycles), which gives us checkpointing, human-in-the-loop pauses, and recursion limits. We add **memory** (episodic in-thread + semantic long-term vector store) so the system *self-heals* on recurring issues, and **provenance/citations** so every claim is auditable. The payoff: a stateful, self-improving, trustworthy autonomous team.

---

*Generated as study notes from the lecture deck. Cross-check any code idioms against the hands-on notebook and current LangGraph docs, since library APIs evolve.*
