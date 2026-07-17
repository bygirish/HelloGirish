# Section 1 — Foundations of an Agent

> **Goal of this section:** Define precisely what an *agent* is (vs a bare LLM), understand its core reasoning loop, and master the **tool-calling** mechanism that is the literal engine of agency. This is the bedrock; every later concept is a variation on it.

**Topics covered**
1. Basics of Agentic AI — LLM vs Agent
2. Anatomy of an Agent — the Observe–Think–Act core loop (ReAct)
3. Example of Agentic AI in Project Context
4. Tool Calling — the Engine of Agency
5. Designing a Tool for the DevOps Agent

---

## 1. Basics of Agentic AI — LLM vs Agent

| | **LLM (the Foundation)** | **Agent (the Evolution)** |
|---|---|---|
| What it is | A **next-token predictor** | An **LLM equipped with Tools, Memory, and a Planning mechanism** |
| What it does | Reads input, produces text | *Acts on its environment* to accomplish a goal |

**The key architectural insight:**

> An agent is an **architectural wrapper around an LLM** that gives it **agency** — the ability to act on its environment.

So the equation to memorize:

```
Agent = LLM + Tools + Memory + Planning
```

The LLM is the "brain" (the reasoning core), but on its own it can only *talk*. Wrapping it with tools (hands), memory (recall), and a planning loop (executive function) turns talk into action.

💡 **Learning Thought:** "Agent" is not a different *model* — it's an *architecture*. You don't download an agent; you *build* one around an LLM. This is why prompt design, tool schemas, and the control loop matter so much: they're the parts *you* supply. The model is a component, not the whole system.

🎯 **Interview Question:** *"Is an LLM an agent? Justify."*
**Model answer:** No. An LLM is a next-token predictor — a pure function from text to text with no ability to act on the world or persist state. It becomes an agent only when wrapped in an architecture that adds tools (to affect the environment), memory (to persist context), and a planning/control loop (to decide the next action). Agency is a property of the surrounding architecture, not of the model weights.

---

## 2. Anatomy of an Agent — the Core Loop (ReAct)

Every agent runs the same three-beat loop:

```
   ┌──────────┐      ┌────────┐      ┌──────────┐
   │ OBSERVE  │ ───▶ │ THINK  │ ───▶ │   ACT    │ ──┐
   └──────────┘      └────────┘      └──────────┘   │
        ▲                                            │
        └──────────────── loop ──────────────────────┘
```

| Beat | What happens |
|---|---|
| **Observe** | Receive user input or system state (and, after the first loop, the result of the last action) |
| **Think** | The LLM decides what to do next, based on its prompt, the conversation, and available tools |
| **Act** | Execute a tool — query a DB, call an API, etc. — which produces a new observation |

**This is the ReAct framework** — *Reasoning and Acting* (Yao et al., ICLR 2023). We'll formalize it fully in Section 4, but the intuition is here: it **mimics how a human troubleshoots** — look, think, try something, look at what happened, think again.

💡 **Learning Thought:** The loop is the whole game. "Passive" systems do Observe → Think → *Answer*. Agents insert **Act → Observe** in the middle, and *repeat*. That single structural change — grounding each reasoning step in a real observation from the world — is what separates a chatbot from an agent.

---

## 3. Example of Agentic AI in Project Context

Grounding the loop in the DevOps agent:

- **Primary goal:** Autonomously debug failing pipelines.
- **Tools available:** Vector DB (Runbooks), Splunk API (Logs), GitHub API (Commits).
- **Capability evolution:**

| Previously (passive RAG) | Now (full agentic loop) |
|---|---|
| Just *fetched* runbooks | **Fetch runbook → Read log → Check history → Write summary** |

The transition: from **basic retrieval** to a **multi-step analytical reasoning loop.** The agent chains several tool calls, each informed by the previous observation, to reach a conclusion no single lookup could produce.

🎯 **Interview Question:** *"Walk me through how an agentic DevOps assistant would handle a 500-error incident differently from a RAG chatbot."*
**Model answer:** A RAG chatbot retrieves the most similar runbook and paraphrases it — one shot, no grounding in the live system. An agentic assistant runs a loop: fetch the runbook (context), then *act* by pulling the actual server logs, *observe* the specific error (e.g., "connection pool exhausted"), *think* to reinterpret (this is an infra/connectivity issue, not a code bug), possibly *act* again to check recent commits, and only then synthesize a root-cause answer grounded in real evidence from multiple systems.

---

## 4. Tool Calling — the Engine of Agency

This is *the* mechanism that lets an LLM "do" things. Understand it precisely because it's a very common interview topic.

### The three-step mechanic

```
1. TOOL DEFINITION            2. LLM RESPONSE                 3. EXECUTION
   Tools are passed to the       The LLM, instead of writing     Your application code runs
   LLM as JSON schemas           raw text, emits a structured    the actual function and
   describing each function      request to call a function      returns the result back
   and its parameters.           with specific arguments.        to the LLM as a new observation.
```

### The single most misunderstood point

> **The LLM does NOT run code.** It generates the *parameters* for the application to execute. Your app runs the function and hands the result back.

The LLM's "tool call" is just **structured text** — a JSON blob saying *"I would like to call `fetch_server_logs` with `server_id='db-12'`."* Your code (the harness/orchestrator) is what actually invokes the function, gets the return value, and feeds it back into the LLM's context as the next observation. The LLM never has direct execution privileges.

💡 **Learning Thought:** This separation is a *safety and control* boundary, not just an implementation detail. Because the app is the executor, you can validate arguments, enforce permissions, wrap calls in try/except, require human approval, and log everything — all *between* the LLM asking and the tool running. Keep this boundary in mind; it reappears as HITL (human-in-the-loop) and provenance later.

🎯 **Interview Question:** *"When an LLM 'calls a tool', what does it actually produce, and who executes the tool?"*
**Model answer:** The LLM produces **structured output** (typically JSON) naming a function and its arguments — it does *not* execute anything. The surrounding application ("harness") parses that request, runs the real function, and injects the return value back into the model's context as a new observation. The model only *requests*; the app *executes*. This boundary is what makes validation, permissioning, and human approval possible.

🎯 **Interview Question:** *"Why pass tools as JSON schemas rather than plain descriptions?"*
**Model answer:** A schema gives the model a *machine-checkable contract* — exact parameter names, types, and required fields — which (a) lets the model emit arguments that your code can parse deterministically, (b) constrains hallucination of parameters, and (c) enables programmatic validation before execution. Free-text descriptions still matter for *when* to use a tool, but the schema governs *how* to call it correctly.

---

## 5. Designing a Tool for the DevOps Agent

A worked example of *tool authoring* — and the lecture's key lesson about **descriptions**.

### The tool

```python
fetch_server_logs
"""Retrieve recent logs to diagnose failures, errors, or performance issues."""
# parameters: server_id: str, minutes_ago: int
```

### A troubleshooting call

```
Request:  "Investigate 500 errors on db-12"

Tool call:  fetch_server_logs(server_id="db-12", minutes_ago=10)

Result:   "Database connection pool exhausted"

Conclusion:  Connectivity issue, not a code failure.
```

Notice how the *observation* ("connection pool exhausted") redirected the diagnosis away from code and toward infrastructure — exactly the grounding effect of the Act→Observe step.

### Why descriptions matter (the core lesson)

> **LLMs choose tools based on their descriptions, not just their names.**

A good description:
- ✅ Helps the agent know **when** to use the tool
- ✅ Helps the agent understand the **returned data**
- ❌ Prevents **incorrect or hallucinated** calls

> **Key lesson:** The more clearly you describe a tool's purpose, the better the LLM can decide when to use it.

💡 **Learning Thought:** Tool descriptions are **prompt engineering for actions**. Treat every tool's docstring as a mini-prompt: state *what it does*, *when to use it*, *what it returns*, and *when NOT to use it*. Vague descriptions are the #1 cause of an agent calling the wrong tool or inventing parameters. In multi-agent systems this compounds — so this small discipline has outsized impact.

🎯 **Interview Question:** *"An agent keeps calling the wrong tool. Before touching the model, what do you inspect?"*
**Model answer:** The **tool descriptions/schemas**. Since the model selects tools purely from their names and descriptions, ambiguous or overlapping descriptions are the most likely cause. I'd rewrite each description to sharply state purpose, trigger conditions, return shape, and anti-conditions ("do not use this for X"), make parameter names/types unambiguous in the schema, and reduce overlap between similar tools — often fixing the behavior with no model change at all.

---

## Section 1 — Consolidated Takeaways

- **Agent = LLM + Tools + Memory + Planning.** "Agent" is an *architecture*, not a model.
- The universal engine is the **Observe → Think → Act** loop (**ReAct**): it grounds each reasoning step in a real observation.
- **Tool calling** mechanics: the LLM *emits structured arguments*; **your application executes** the function and returns the result. The LLM never runs code — a crucial control/safety boundary.
- Tools are passed as **JSON schemas**; but the **natural-language description** is what drives *when* the model uses a tool.
- **Well-written tool descriptions** are the cheapest, highest-leverage way to reduce wrong/hallucinated tool calls.

**Next:** [Section 2 — Multi-Agent Systems](03-Section-2-Multi-Agent-Systems.md) — why one agent with many tools breaks down, and how a *team* of specialists fixes it.
