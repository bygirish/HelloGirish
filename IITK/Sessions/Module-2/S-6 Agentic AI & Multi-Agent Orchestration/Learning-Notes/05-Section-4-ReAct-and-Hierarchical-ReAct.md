# Section 4 — Planning with ReAct (and Hierarchical ReAct)

> **Goal of this section:** Master **ReAct** — the seminal reasoning-and-acting framework that underpins every agent in this course — its formal definition, *why* it beats pure Chain-of-Thought, and how it **scales hierarchically** into a multi-agent system where the Triage agent treats specialists as its tools.

**Topics covered**
1. Single-Agent ReAct (the seminal work)
2. The Limitation of Pure Reasoning (CoT) — the "hallucination cliff"
3. The Formal Definition of the ReAct Loop
4. Agent Context Example (DevOps)
5. Scaling ReAct: the Context & Tool-Exhaustion problem
6. Hierarchical ReAct: Triage as Meta-Planner
7. Formalizing the Multi-Agent State Transition
8. ✅ Self-test MCQs

---

## 1. Single-Agent ReAct — the Seminal Work

> **Paper:** *"ReAct: Synergizing Reasoning and Acting in Language Models"* — Yao et al., **ICLR 2023.** (Worth knowing the citation for interviews.)

**The core thesis:** Human intelligence seamlessly integrates **task-oriented actions** with **verbal reasoning.** ReAct enables LLMs to **interleave** reasoning traces (**Thoughts**) with task-specific **Actions** (tool use).

**The key synergy:**
- **Reasoning guides action** (thinking decides what to do next)
- **Action grounds reasoning in external reality** (observations keep thinking honest)

> Before ReAct, reasoning (Chain-of-Thought) and acting (tool use) were treated as **separate domains.** ReAct's contribution was to *interleave* them in one loop.

---

## 2. The Limitation of Pure Reasoning (CoT)

**Chain-of-Thought (CoT)** lets a model solve complex problems by outputting intermediate steps. But used alone (closed-book, no tools), it has a fatal flaw:

### The "Hallucination Cliff"

> In closed-book CoT, if the model makes a factual error in **Step 2**, that error **compounds uncontrollably** through Step 10.

Because nothing external checks the reasoning, one bad assumption early on poisons every subsequent step. The model confidently reasons its way to a wrong answer.

### The ReAct solution

> ReAct forces the agent to **pause reasoning, take an Action, and wait for an Observation (ground truth) before thinking again.** It **fact-checks itself against reality via tools.**

```
CoT (open loop, error compounds):
   Think₁ → Think₂(wrong) → Think₃ → ... → Think₁₀ → Wrong Answer

ReAct (closed loop, grounded):
   Think₁ → Act₁ → Observe₁(reality) → Think₂ → Act₂ → Observe₂ → ...
                        ▲ error gets corrected here by real data
```

💡 **Learning Thought:** CoT reasons in a *vacuum*; ReAct reasons *against reality*. The observation after each action is a **correction signal** — it's the difference between a student solving a problem with their eyes closed vs. checking each step against a reference. This "grounding" is the entire value proposition of tool use.

🎯 **Interview Question:** *"What problem with Chain-of-Thought does ReAct specifically solve?"*
**Model answer:** The **compounding-error / hallucination-cliff** problem. In closed-book CoT there's no external check, so a factual mistake in an early step propagates and amplifies through the rest of the chain, yielding a confidently wrong answer. ReAct interrupts pure reasoning with **Action → Observation** steps: the model gets *ground truth* from a tool after each action, which corrects flawed assumptions before they compound. Reasoning guides the action; the action grounds the reasoning.

---

## 3. The Formal Definition of the ReAct Loop

This is the mathematical formalism — expect it in a rigorous interview.

- **Agent input:** the agent receives a user instruction **x**.
- **Agent context (cₜ):** the history of previous steps:

  ```
  cₜ = (x, t₁, a₁, o₁, …, t_{t-1}, a_{t-1}, o_{t-1})
  ```

- **Next-step generation (ŷ)** produces one of:
  - **Thought  tₜ ∈ T** — an internal state update to prepare for better actions
  - **Action   aₜ ∈ A** — an external interaction with the environment
  - **Observation oₜ** — the environment returns ground truth to ground reasoning

**The subtle but important point:**

> **Thoughts and Actions share the same context window.** Thoughts *prepare* the model; Actions *inject empirical facts (oₜ)* back into the context.

So the context grows as an interleaved transcript: instruction, then repeating (thought, action, observation) triples. The **action space A** is the set of available tools; the **thought space T** is free-form reasoning.

💡 **Learning Thought:** Everything the agent "knows" at step *t* lives in cₜ — the running transcript. This is why *context management* (Sections 6–7: state bloat, summarization, memory) becomes critical: the transcript is both the agent's working memory *and* a finite, expensive resource. The formalism also foreshadows the multi-agent extension: just expand the context vector to track *which agent* produced each entry.

---

## 4. Agent Context Example (DevOps translation)

What the formalism looks like in the actual project:

```
x  (User input):   "The auth-service is failing."

t₁ (Thought):      "I need to check the error logs for auth-service to see why."
a₁ (Action):       fetch_server_logs(service="auth-service", timeframe="1h")
o₁ (Observation):  "ConnectionTimeout: Cannot reach db-primary"

t₂ (Thought):      "The issue isn't the auth-service code, it's the DB connection.
                    I need to check db-primary status."
a₂ (Action):       query_infrastructure(target="db-primary")
o₂ (Observation):  ...
```

Notice how o₁ **redirected** the reasoning in t₂ — the agent abandoned the "code bug" hypothesis because reality (the observation) pointed at infrastructure. That is ReAct working exactly as intended.

---

## 5. Scaling ReAct: the Context & Tool-Exhaustion Problem

This ties back to Section 2's "20-tool rule", now framed formally.

**The single-agent bottleneck:**

> As enterprise systems grow, the **Action space A expands** to dozens of tools (Splunk, GitHub, Datadog, Jira). If **|A| > 20**, standard ReAct agents suffer **"Tool Hallucination"** — they guess API parameters or mix up tool schemas.

**The solution — Hierarchical Planning:**

> Multi-Agent systems enable **distributed focus.** In enterprise DevOps the environment is too vast for one ReAct loop — which is why we introduced the Triage, Infra, and Codebase agents.

> Context: the ReAct paper tested models on *simple* environments (like Wikipedia). **Enterprise scale requires specialized hierarchical structures.**

💡 **Learning Thought:** ReAct is *necessary but not sufficient* at scale. The loop is right, but a single loop over a huge action space collapses. The move to hierarchy is not a rejection of ReAct — it's ReAct **nested inside ReAct.**

---

## 6. Hierarchical ReAct: Triage as a Meta-Planner

**The Meta-ReAct loop:** the Triage Agent runs a ReAct loop where its **actions are delegations to specialists**, not raw API calls.

**Shift in action space A:**

> The action space consists of **Delegations to Specialist Agents** instead of raw API tools.

**Execution example:**
```
Thought:      "I need to know what code changed yesterday."
Action:       DelegateTo(CodebaseAgent, "Fetch yesterday's commits")
Observation:  Synthesized summary of commits returned by the specialist.
```

> The Triage Agent **never touches raw JSON or API keys.** It reasons purely at the **strategic level**, treating specialist sub-agents as its **"Tools."**

```
        ┌──────────── META ReAct loop (Triage) ────────────┐
        │  Think → DelegateTo(Specialist) → Observe(summary)│
        └──────────────────┬────────────────────────────────┘
                           │ each delegation triggers…
        ┌──────────────────▼──── micro ReAct loop (Specialist) ───┐
        │  Think → call raw API tool → Observe(raw data) → …       │
        │  …returns only a CLEAN SUMMARY up to Triage              │
        └──────────────────────────────────────────────────────────┘
```

💡 **Learning Thought — the fractal structure:** ReAct is *self-similar*. The same Observe–Think–Act loop runs at the strategic level (Triage delegating) and at the tactical level (a specialist calling Splunk). Each specialist's messy internal loop is **hidden** from Triage, which only sees the polished result. This is *abstraction* — the same principle that lets a CEO think in strategy while engineers handle syscalls.

---

## 7. Formalizing the Multi-Agent State Transition

To support hierarchy, we **expand the ReAct context vector to track provenance** — *which agent* did what:

```
c_global = (x, t₁^Triage, a₁^{Triage→Infra}, o₁^Infra, t₂^Triage, a₂^{Triage→Codebase}, …)
```

Each entry is now **tagged with the agent identity** and, for actions, the **delegation edge** (Triage→Infra). Key mechanics:

- **Isolated execution:** the Infra Agent runs its **own isolated micro-ReAct loop** to query Splunk. It returns **only the final o^Infra** back to the global state — **shielding Triage from raw system logs.**
- **LangGraph connection (preview):** this formalizes what LangGraph does "under the hood" — it *manages the global vector* and ensures the Triage agent receives **clean, synthesized observations** from specialists.

**Cognitive architecture summary:** The **Meta-Planner decomposes** the query, **delegates** to specialists via the shared **global context**, and **synthesizes** their observations into a final, **verifiable** answer.

💡 **Learning Thought:** Tagging every context entry with its author (`name="Codebase_Agent"`) isn't bookkeeping for its own sake — it's the seed of **provenance/citations** (Section 8). The moment you record *who observed what*, you can later cite it. Architecture decisions here pay off three sections later.

---

## 8. ✅ Self-test MCQs (from the lecture)

### MCQ 1 — The checkout latency scenario
> *The checkout API has been slowing down for two days. No deployment occurred, and there are no errors in the logs, but a feature flag controlling a third-party shipping-rate integration was changed three days ago. How should the Triage Agent most likely proceed, and why would a single flat ReAct agent (with all enterprise tools loaded) be riskier here?*
>
> **A)** Codebase Agent first, then Infra; flat agent fine — only two tools needed.
> **B)** Infra Agent + HR/Policy Agent; risk only applies to tool-heavy queries.
> **C)** Infra Agent (latency) + config/flag owner, then synthesize; flat agent with many tools more likely to mis-select/hallucinate params.
> **D)** Codebase Agent only — no deploy rules out config/infra; ReAct can't use Observations.

<details><summary>Answer & reasoning</summary>

**Answer: C.** It matches the actual symptom (a **config/flag** change, not code) and correctly explains that **tool-hallucination risk is a property of the agent's total tool count in the live system**, not of how many tools this particular query happens to need.
- **A** is wrong twice: misdiagnoses the cause (starts with code despite *no deploy*), and assumes a flat agent is "fine" just because this query is simple — ignoring that the same agent still carries all 30+ tools in production.
- **B** gets delegation right but reasoning wrong: risk is a *standing* property of tool count, not per-query.
- **D** misstates ReAct itself — its defining feature is that it *does* use Observations to ground reasoning.
</details>

### MCQ 2 — The 30-tool agent with good memory
> *A single ReAct agent has 30 tools and a correctly working long-term memory system. It still inefficiently resolves recurring multi-day incidents. What's structurally missing, and why doesn't more memory or more tools fix it?*
>
> **A)** Bigger context window for tool outputs.
> **B)** Faster model for quicker tool calls.
> **C)** Needs hierarchical agents (Triage + specialists) or Planner/Executor split — past ~20 tools, hallucination rises regardless of memory; memory ≠ action-selection.
> **D)** More frequent memory writes.

<details><summary>Answer & reasoning</summary>

**Answer: C.** ~20 tools is the threshold past which a single agent suffers tool hallucination (confused schemas, guessed params), **independent of memory quality**. Memory governs *what the agent knows*; tool count and reasoning load govern *whether it can reliably act.* Only a **structural split** (multi-agent or planner/executor) fixes it.
- **A** repeats the "more context is always better" misconception (the needle-in-a-haystack caveat) and doesn't fix schema confusion.
- **B** misreads "inefficiently" as a speed problem — a faster model with the same overload just hallucinates faster.
- **D** is contradicted by the premise (memory is already "correctly implemented").
</details>

---

## Section 4 — Consolidated Takeaways

- **ReAct** (Yao et al., ICLR 2023) **interleaves Thoughts and Actions**, using **Observations as ground truth** to fix reasoning.
- It cures **CoT's hallucination cliff** (errors compounding through an unchecked reasoning chain).
- Formally, the agent maintains a context `cₜ = (x, t₁, a₁, o₁, …)`; thoughts prepare, actions inject facts. **Thoughts and actions share one context window.**
- A single ReAct loop breaks when **|A| > ~20 tools** → **tool hallucination.** The fix is **Hierarchical ReAct.**
- In **Hierarchical/Meta ReAct**, the Triage agent's **action space is delegations to specialists**; specialists run **isolated micro-loops** and return **clean summaries**, shielding the meta-planner.
- The multi-agent context vector is **tagged with agent identity/provenance** — the foundation for LangGraph state and for citations.

**Next:** [Section 5 — State Machines & LangGraph](06-Section-5-State-Machines-and-LangGraph.md) — the runtime that executes all of this reliably.
