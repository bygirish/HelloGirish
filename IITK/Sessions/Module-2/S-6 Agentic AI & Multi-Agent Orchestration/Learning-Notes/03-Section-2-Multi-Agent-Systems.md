# Section 2 — Multi-Agent Systems

> **Goal of this section:** Understand the single most important scaling insight of the lecture — *why one agent with many tools fails*, and how splitting into a **team of specialist agents** coordinated by a **Triage agent** solves it. This "context/tool exhaustion" argument is the recurring theme of the entire course.

**Topics covered**
1. Single-Agent vs Multi-Agent (the theory)
2. Project Context: the DevOps Multi-Agent Team
3. Transitioning to Orchestration

---

## 1. Single-Agent vs Multi-Agent — the Concept

| **Single Agent** | **Multi-Agent** |
|---|---|
| One massive prompt, *all* the tools | Several **specialized** agents (e.g., Log Analyzer, HR Policy) working together |
| Simple to build | Each agent has a narrow, focused toolset |
| **Prone to confusion** as it scales | Reliable at scale via **separation of concerns** |

**The core problem (memorize the mechanism):**

> As systems scale, giving ~50 tools to one LLM causes **context exhaustion**. So we split them up.

### What "context exhaustion" and "tool hallucination" actually mean

When a single agent is handed dozens of tools:
- All those tool schemas must sit in the prompt, **consuming the context window** and diluting attention.
- With many similar tools, the model starts **mixing up schemas** and **guessing/inventing parameters** — this is **tool hallucination**.
- The lecture gives a concrete threshold: **past roughly 20 tools (|A| > 20)**, standard single agents begin to hallucinate tool calls *regardless of how good the model or memory is*.

💡 **Learning Thought — the "20-tool rule":** This threshold is a load-bearing fact for the whole lecture and shows up in multiple exam questions. The failure at scale is a property of **tool count and reasoning load**, *not* of context-window size or memory quality. More memory or a bigger window does **not** fix it — only a *structural split* (multi-agent, or planner/executor) does. Burn this in.

🎯 **Interview Question:** *"Why does adding more tools to a single agent eventually degrade it, and at roughly what point?"*
**Model answer:** Each tool's schema occupies context and competes for the model's attention; with many overlapping tools the model confuses schemas and fabricates parameters ("tool hallucination"). The lecture pegs the danger zone at **>~20 tools**. Critically, this is a reasoning-load/action-selection problem, so it isn't cured by a bigger context window or better memory — you must **partition** the tools across specialized agents so each agent reasons over a small, coherent action space.

---

## 2. Project Context: the DevOps Multi-Agent Team

To avoid context exhaustion, the Enterprise DevOps Agent is divided into a **team of specialists**:

| Agent | Role | Owns |
|---|---|---|
| **Triage Agent** | The **front door**. Analyzes user queries and **delegates** to specialists. | Routing/orchestration (no domain tools) |
| **Infrastructure Agent** | Specializes in **live telemetry**. | Splunk, Datadog, Kubernetes clusters |
| **Codebase Agent** | Specializes in **source code**. | GitHub search, PR diffs, commits |
| **Policy & HR Agent** | Specializes in **RAG retrieval**. | Runbooks, company HR policies |

**The design principle:**

> Instead of giving **50 tools to one LLM** (and risking hallucination), we give **~3–5 highly relevant tools to each specialist.**

Each specialist reasons over a *small, coherent* action space — so it selects tools reliably. The Triage agent, meanwhile, doesn't touch raw tools at all; its "tools" are *the specialists themselves*.

> The lecture foreshadows: *"As we dive into LangGraph later, we'll learn how to wire these agents together so they can pass information back and forth to solve complex, multi-hop incidents."* (That's Sections 5–6.)

💡 **Learning Thought — separation of concerns:** This is the same principle as microservices vs a monolith, or a company org chart. You don't ask one engineer to be simultaneously the SRE, the backend dev, and the HR rep — you hire specialists and add a manager to coordinate. Multi-agent design is *software architecture applied to cognition*: partition responsibility so each unit stays within its competence.

🎯 **Interview Question:** *"How would you decompose a monolithic 30-tool agent into a multi-agent system? What's your partitioning criterion?"*
**Model answer:** Partition by **domain/data-source coherence** — group tools that operate on the same system or knowledge area so each specialist has a small, non-overlapping toolset (e.g., all telemetry tools → Infra agent; all VCS tools → Codebase agent; all document retrieval → Policy agent). Then add a **Triage/orchestrator** whose only job is to understand the request and delegate — it holds *no* domain tools, treating the specialists as its action space. Criterion: minimize each agent's tool count and inter-tool ambiguity while keeping each agent's tools mutually relevant.

---

## 3. Transitioning to Orchestration

Splitting into agents creates a *new* problem: coordination.

**The core challenge:**

> How do we manage multiple tools and steps **without getting stuck in infinite loops**?

**The solution — Orchestration:**

> Coordinating autonomous actions into a **cohesive workflow** to ensure **reliability and termination.**

And the technical foundation named here (previewing Sections 3 & 5):

> **Planners, Executors, and State Machines** form the foundation of modern AI agent orchestration.

💡 **Learning Thought:** Notice the intellectual arc: single agent → *too many tools* → split into specialists → *but now how do they coordinate and stop?* → **orchestration**. Every solution in this lecture creates the next problem, and the next slide solves it. Two words define the orchestration goal: **reliability** (right answer, no chaos) and **termination** (it must *stop* — no infinite loops burning tokens).

🎯 **Interview Question:** *"You split your agent into a team of specialists. What new failure modes did you just introduce?"*
**Model answer:** Coordination failures: (1) **infinite loops** — agents bouncing control back and forth (Triage↔specialist) without progress, burning tokens; (2) **state bloat** — inter-agent chatter inflating the shared context; (3) **routing errors** — the orchestrator sending work to the wrong specialist; and (4) **provenance loss** — harder to know which agent produced which claim. These are exactly why you need an *orchestration layer* (planner/executor + a state machine with recursion limits), covered in the next sections.

---

## Section 2 — Consolidated Takeaways

- A **single agent doesn't scale**: past **~20 tools**, it suffers **context exhaustion** and **tool hallucination** (guessing/mixing up schemas).
- This limit is about **tool count + reasoning load**, *not* context size or memory — so only a **structural split** fixes it.
- The fix is a **team of specialists** (Triage + Infra + Codebase + Policy/HR), each with a **small, coherent toolset**; the **Triage** agent delegates and holds no domain tools.
- Splitting introduces a *coordination* problem → solved by **orchestration**, whose goals are **reliability** and **termination**.
- Orchestration rests on three pillars introduced next: **Planners, Executors, and State Machines.**

**Next:** [Section 3 — Orchestration: Planner & Executor](04-Section-3-Planner-Executor-Orchestration.md) — separating the "thinking" from the "doing".
