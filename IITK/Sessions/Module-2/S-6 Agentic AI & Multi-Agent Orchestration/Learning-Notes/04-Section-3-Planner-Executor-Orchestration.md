# Section 3 — Orchestration: Planner & Executor

> **Goal of this section:** Learn the central production design pattern of agentic systems — **separating the "thinking" (Planner) from the "doing" (Executor)** — why it exists, how it maps onto the DevOps team, and how **re-planning** keeps agents robust in changing environments.

**Topics covered**
1. Agentic Orchestration — Planner + Executor
2. What is a Planner?
3. What is an Executor?
4. Why separate Planning and Execution?
5. Case Study: the Planner in Action
6. Specialist Agents in Action (Execution Phase)
7. Planner vs Executor — summary table
8. The Re-Planning Phase

---

## 1. Agentic Orchestration — the Two Phases

Orchestration splits an agent's work into two distinct phases handled by two distinct roles:

```
   ┌──────────────────────┐         ┌──────────────────────┐
   │      THE PLANNER      │         │     THE EXECUTOR     │
   │   "Thinking" phase    │  ────▶  │    "Doing" phase     │
   │  decides WHAT to do   │         │  decides HOW to do   │
   └──────────────────────┘         └──────────────────────┘
```

**Core principle:**

> To build reliable agents, we **separate the "thinking" (Planning) from the "doing" (Execution)** to ensure logical consistency and error handling.

---

## 2. What is a Planner?

| Aspect | Detail |
|---|---|
| **Role** | Breaks a complex user request into a **step-by-step execution plan** |
| **Mechanism** | Uses an **advanced reasoning model** (e.g., a top-tier GPT-4-class model) to generate a **DAG (Directed Acyclic Graph) of tasks** |
| **Key constraint** | The planner **does NOT run tools.** It just outputs a *recipe.* |

Example plan: *Step 1 — Check logs. Step 2 — Query codebase.*

> Planners focus on the **logic and structure** of the solution, delegating actual tool use to Executors.

💡 **Learning Thought:** The planner emits a *recipe*, not a *meal*. It's pure reasoning — no side effects, no API keys, no risk. This is why you can afford to run your **most expensive, most capable model** here: planning happens once per request and its quality determines everything downstream.

> ⚠️ Terminology nuance: the plan is called a **DAG of tasks** here (an acyclic *plan structure*). Don't confuse this with the earlier point that agentic *control flow* must be *cyclic*. The plan's dependency graph is acyclic; the runtime that executes and **re-plans** it is cyclic. Both statements are true at different levels.

---

## 3. What is an Executor?

| Aspect | Detail |
|---|---|
| **Role** | Takes a **single step** from the Planner and **executes it** |
| **Mechanism** | Uses **smaller, faster models** to format tool calls and handle the exact API interactions |
| **Key focus** | Narrowly on **safety and delivery** — get the job done safely, return the payload to the orchestrator |

> Executors are the **operational layer**, translating logical steps into concrete, safe system interactions.

💡 **Learning Thought:** The executor is a *specialist tradesperson*: it doesn't debate strategy, it does one well-scoped job correctly and safely, then reports back. Because its scope is tiny (one step, a handful of tools), a **cheap, fast model** is enough — and often *better*, because it isn't distracted by big-picture reasoning.

---

## 4. Why Separate Planning and Execution?

Three concrete benefits — a classic interview triad:

| Benefit | Why |
|---|---|
| **Accuracy** | Reduces **cognitive load** on the LLM — each model does one kind of thinking |
| **Cost Efficiency** | Use **expensive models for planning**, **cheaper models for routine execution** |
| **Fault Tolerance** | If **Step 2 fails, you don't need to re-plan Step 1** — only the failed step is retried/replanned |

> This is a crucial **production design pattern.**

🎯 **Interview Question:** *"Why separate planning from execution in an agent? Give three reasons."*
**Model answer:** (1) **Accuracy** — decomposing the labor reduces cognitive load; the planner reasons about strategy without formatting tool calls, and the executor formats calls without reasoning about strategy, so each does its job better. (2) **Cost** — planning is rare and high-stakes so you spend on a top-tier reasoning model, while execution is frequent and routine so a small fast model suffices, cutting cost dramatically. (3) **Fault tolerance** — failures are localized: a failed step is retried or re-planned in isolation without discarding valid completed steps. This mirrors a project manager (plans) vs engineers (execute).

---

## 5. Case Study: the Planner in Action

**User query:** *"Payment gateway failing after last night's update."*

The **Triage Agent (acting as the Planner)** produces this strategy:

```
01. Code Analysis   → Delegate to Codebase Agent: fetch the latest commit diff
                       from the payment-gateway repo.
02. Log Retrieval   → Delegate to Infrastructure Agent: pull error logs for
                       pay-pod-1 covering the last 12 hours.
03. Synthesis       → Compare code changes against live error traces for a
                       final root-cause answer.
```

**Under the hood:** the Triage Agent acts as an **orchestrator** — it does *not* run tools directly; it **formulates the recipe and routes steps to domain experts.**

---

## 6. The Specialist Agents in Action (Execution Phase)

Now the executors do the work the planner prescribed:

**Execution Step 1 — Codebase Agent** (receives directive, formats a GitHub tool call):
```
github.get_commit_diff(repo="payment-gateway", sha="latest")
Observation → Commit a1b2c3d, notes: "Added new strict rate-limiting logic".
```

**Execution Step 2 — Infrastructure Agent** (receives next directive, formats a Splunk tool call):
```
splunk.query_logs(service="pay-pod-1", level="ERROR")
Observation → TimeoutException at Auth.py line 42 for pay-pod-1.
```

The specialists operate in a **tight loop to accomplish narrow tasks safely.** By splitting the work, we avoid overwhelming a single agent and **reduce hallucinated API calls.** Each returns a **specific payload** to the orchestrator.

💡 **Learning Thought:** Watch the information flow: Planner → directive → Executor → *observation* → back to orchestrator. The executors return **clean, synthesized payloads**, not raw dumps. The planner reasons over *summaries*, never raw JSON. This "shielding" of the planner from low-level noise is exactly what keeps the top-level reasoning coherent (formalized as "hierarchical ReAct" in Section 4).

---

## 7. Planner vs Executor — the Summary Table

Memorize this table; it's the cleanest possible articulation of the pattern.

| Aspect | **Planner** | **Executor** |
|---|---|---|
| **Primary role** | Decides *what* should be done | Carries out the decided actions |
| **Main question** | "What is the best way to solve this task?" | "How do I perform this step?" |
| **Input** | User goal/task | A plan/action from the planner |
| **Output** | Sequence of steps (a plan) | Tool calls, actions, results |
| **Focus** | Reasoning & task decomposition | Execution & interaction with tools |
| **Uses tools?** | Usually *selects* tools | Actually *invokes* tools |
| **Example action** | "Search papers → Summarize → Compare" | Execute search, retrieve, summarize |
| **Failure mode** | Bad plan, missing steps | Wrong tool call, execution error |
| **Human analogy** | **Project Manager** | **Engineer / Worker** |
| **In our project** | **Triage Agent** | Tools, Codebase Agent, HR & Policy Agent |

---

## 8. The Re-Planning Phase

Static plans break in dynamic environments. Re-planning is what makes agents *robust*.

**Key question:** *What happens when an Executor fails or finds new information?*

```
01. Feedback Loop      → The Orchestrator feeds the result back to the Planner.
02. Dynamic Update     → The Planner updates the plan based on new constraints.
```

**Example — Network Failure:** if the server is unreachable, the plan must **pivot** to checking network status *before* retrying API calls.

> Dynamic environments require dynamic plans. **Continuous re-planning ensures agent robustness.**

💡 **Learning Thought:** This is the exact moment orchestration becomes a *loop* rather than a straight line. Plan → execute → observe → *maybe re-plan* → execute again. This cyclic feedback is why the runtime can't be a simple DAG and must be a **state machine** (Section 5). Re-planning is also where the fault-tolerance benefit pays off: only the affected part of the plan changes.

🎯 **Interview Question:** *"An executor step fails midway through a 5-step plan. What should a well-designed orchestrator do — and what should it NOT do?"*
**Model answer:** It should feed the failure/observation back to the planner and **re-plan only from the failure point forward**, preserving the results of the steps that already succeeded — e.g., insert a remediation step (check network) before retrying. It should **not** discard the whole plan and restart from Step 1 (wasteful, and the earlier results are still valid), nor should it let the executor silently retry forever (needs a bounded retry / recursion limit). This localized, feedback-driven re-planning is the fault-tolerance benefit of the planner/executor split.

---

## Section 3 — Consolidated Takeaways

- Orchestration separates **Planning ("what", expensive reasoning model)** from **Execution ("how", cheap fast model).**
- The **Planner outputs a recipe (a DAG of tasks) and runs no tools**; the **Executor runs one step at a time, safely.**
- Three reasons to separate: **Accuracy** (less cognitive load), **Cost** (right model for each job), **Fault tolerance** (localized failure recovery).
- In the project, the **Triage Agent is the Planner/orchestrator**; the **specialists are Executors**. Planner is the Project Manager, Executor is the Engineer.
- **Re-planning** (feedback loop → dynamic update) makes the system robust to failures and new information — and makes the runtime inherently **cyclic**, motivating state machines.

**Next:** [Section 4 — ReAct & Hierarchical ReAct](05-Section-4-ReAct-and-Hierarchical-ReAct.md) — the formal reasoning loop underneath all of this, and how it scales hierarchically.
