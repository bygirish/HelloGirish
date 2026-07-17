# Prompt Techniques — Master Comparison

**Course:** Advanced Prompt Engineering | IIT Kharagpur
**Instructor:** Prof. Koustav Rudra | Module 2, Session 1 (Lec 7)
**Purpose:** A single-glance comparison of every prompting technique covered across Sections 2–4, so you can pick the right tool for a task and understand how each one improves on the last.

---

## How to Read This Document

The techniques form a **ladder** — each one exists because it fixes a specific failure of the technique below it. Read the master table first for the at-a-glance comparison, then use the explanations and decision guide below to cement the *why*.

```
Zero-Shot        → raw instruction, no examples (baseline)
   ↑ fails on format-sensitive tasks
Few-Shot         → learn from a few examples
   ↑ fails on multi-step reasoning
Chain-of-Thought → reason step-by-step before answering
   ├── Zero-Shot CoT   (trigger phrase, no examples)
   ├── Few-Shot CoT    (worked reasoning examples)
   ├── Auto-CoT        (auto-generate the CoT examples)
   └── Self-Consistency (sample many chains, majority vote)
   ↑ fails when knowledge is stale / action needed
ReAct            → interleave reasoning with real tool calls
   ↑ needs higher-level control over strategy & workflow
Steering layer   → Meta Prompting · DSP · Prompt Chaining · Tree of Thoughts
```

---

## Master Comparison Table

| # | Technique | Core Idea | Key Mechanism / Trigger | Examples Needed? | Reasoning Style | Tool / External Use | Cost (relative) | Best For | Main Weakness | Fixes Which Predecessor |
|---|-----------|-----------|-------------------------|:----------------:|-----------------|:-------------------:|:---------------:|----------|---------------|--------------------------|
| 1 | **Zero-Shot** | Give an instruction, no examples | Direct task description | No (0) | Direct input→output | No | ★ (1×) | Simple, well-defined tasks (classification, translation) | Weak on reasoning, format, ambiguity | — (baseline) |
| 2 | **Few-Shot** | Show a few input→output demos | 2–10 `(input → output)` pairs (In-Context Learning) | Yes (2–10) | Direct, pattern-matched | No | ★ (1×) | Format-sensitive & style-sensitive tasks | Pattern-matches labels; fails multi-step reasoning | Zero-Shot's format/ambiguity gap |
| 3 | **Chain-of-Thought (CoT)** | Think step-by-step before answering | Intermediate reasoning steps in output | Optional | Linear chain | No | ★★ | Multi-step reasoning (math, logic) | Needs large models; single fragile path | Few-Shot's inability to reason |
| 4 | **Zero-Shot CoT** | Unlock reasoning with a phrase | *"Let's think step by step."* | No (0) | Linear, self-generated | No | ★★ | Reasoning when you have no examples | Unguided chain can talk itself into a wrong answer | CoT's need for hand-written examples |
| 5 | **Few-Shot CoT** | Demonstrate the reasoning, not just the answer | Full `(input → reasoning → output)` demos | Yes (worked chains) | Linear, guided | No | ★★ | Best accuracy when you *have* examples | Manual, expensive to author good chains | Zero-Shot CoT's unguided reasoning |
| 6 | **Auto-CoT** | Auto-generate the CoT examples | k-means cluster questions → Zero-Shot CoT per cluster | No (auto-built) | Linear, guided (auto) | No (uses embeddings) | ★★★ (build cost) | Scaling CoT without manual labeling | Auto-chains can be low quality; needs filtering | Few-Shot CoT's manual authoring cost |
| 7 | **Self-Consistency** | Sample many chains, take majority vote | Temperature > 0, run N times, aggregate | Optional | N parallel linear chains | No | ★★★★ (N×) | High-stakes reasoning where accuracy > cost | N× the cost/latency | CoT's single-fragile-path risk |
| 8 | **ReAct** | Interleave reasoning **and** acting | Thought → Action → Observation loop | Optional (1–6 demos) | Iterative + tool-grounded | **Yes** (Search, APIs, code…) | ★★★★ | Multi-hop, dynamic, real-time, multi-tool tasks | Latency, cost, error propagation, loops | CoT's stale/frozen-knowledge ceiling |
| 9 | **Meta Prompting** | Prompt the *structure*, not the content | Abstract templates + syntax anchors (e.g. `$\boxed{}$`) | No (abstract slots) | Structure-driven | No | ★★ | Consistent output structure across many domains | Broader but potentially shallower than domain examples | Few-Shot/CoT being tied to specific content |
| 10 | **Directional Stimulus Prompting (DSP)** | Small trained model generates per-input hints | Policy LM (SFT → RL, reward=ROUGE) feeds a frozen LLM | Yes (to train policy LM) | Hint-augmented | No (hints from input only) | ★★★★ (training) | Frozen/API LLMs needing per-instance guidance | Requires training a policy model | Static prompts being too generic per input |
| 11 | **Prompt Chaining** | Break one task into a pipeline of focused calls | Output of step A → input of step B → … | Per-step (optional) | Sequential pipeline | Optional | ★★★ (multi-call) | Complex tasks with verifiable, sequential sub-steps | Latency; steps must be cleanly separable | Single-prompt overload / no checkpoints |
| 12 | **Tree of Thoughts (ToT)** | Explore many reasoning paths, evaluate & prune | Branch → self-evaluate → prune → deepen (BFS/DFS/beam) | Optional | Branching tree + backtracking | No | ★★★★★ (highest) | Puzzles, planning, lookahead problems | Very high token/time cost | CoT/Self-Consistency having no lookahead or backtracking |

> **Cost legend:** ★ = single cheap call · ★★★★★ = many calls + evaluation. These are relative, not absolute.

---

## Quick Explanations — One Paragraph Each

**1. Zero-Shot** — The default way to use an LLM: state the task, provide no examples, and rely entirely on the model's pre-trained knowledge and instruction-following. Great for simple, common tasks; breaks down on reasoning, strict formats, and ambiguous asks. Its capability scales sharply with model size.

**2. Few-Shot** — Put a handful of `(input → output)` demonstrations in the prompt. This is **In-Context Learning (ICL)**: no weights change, yet behavior shifts because the model continues the demonstrated pattern. It nails *format and style* but still only pattern-matches — it doesn't actually *reason*.

**3. Chain-of-Thought (CoT)** — Instead of forcing the answer into a single prediction, let the model "think out loud" in intermediate steps. This exploits the model's strength (predicting the next logical step) to overcome its weakness (compressing a whole reasoning chain into one jump). Emerges mainly in large (~100B+) models.

**4. Zero-Shot CoT** — The remarkable finding that appending **"Let's think step by step"** alone triggers a reasoning chain — no examples required. The reasoning ability was already latent from pre-training; the phrase is a statistical trigger that activates it. Cheap and task-agnostic, but the unguided chain can go astray.

**5. Few-Shot CoT** — The gold standard for reasoning *when you have examples*: demonstrations include the **full reasoning chain**, not just the answer. Each worked example is a behavioral template — the model learns *how to think*, not just *what to output*. Downside: authoring correct chains is expensive.

**6. Auto-CoT** — Automates Few-Shot CoT. Cluster a dataset's questions by semantic similarity (embeddings + k-means), pick one representative per cluster, and use Zero-Shot CoT to auto-generate its reasoning chain. Clustering ensures **diversity**; heuristics filter bad chains. Matches manual CoT with no human labeling.

**7. Self-Consistency** — Treat one CoT answer as a single *sample*, not ground truth. Sample N diverse chains (temperature > 0) and take the **majority-vote** answer. Intuition: many roads lead to the correct answer, but only a few flawed roads lead to any specific wrong one — so voting amplifies signal over noise. Cost is N× inference.

**8. ReAct** — The point where prompting becomes **agentic**. The model interleaves **Thought → Action → Observation**: it reasons about what it needs, calls a tool (search, API, calculator, code), observes the real result, and adapts. It shatters CoT's ceiling of frozen, training-time knowledge and grounds answers in verified facts, reducing hallucination. Foundation of all modern agent frameworks.

**9. Meta Prompting** — Structure-driven rather than content-driven. Instead of "here are 3 examples of X," you give the model *the structure it must follow to solve any problem* (abstract slots, formatting anchors like `$\boxed{}$`). One meta prompt transfers across algebra, logic, coding, essays — you hand the model a **strategy**, not an answer.

**10. Directional Stimulus Prompting (DSP)** — The insight that the "best prompt" is a *function of the input*. A small **policy LM** is trained (supervised fine-tuning, then RL with ROUGE reward) to emit per-instance **hints** (e.g. keywords a summary should cover). These steer a large **frozen** LLM without touching its weights — an automatic, input-aware prompt engineer that works even with API-only models.

**11. Prompt Chaining** — The Unix "do one thing well" philosophy for LLMs. Decompose a complex task into a fixed sequence of focused calls where each output feeds the next. Each intermediate result is a **verifiable checkpoint**, making the pipeline accurate, debuggable, modular, and cost-tunable (cheap model for easy steps, expensive for hard). Deterministic — unlike ReAct's dynamic control flow.

**12. Tree of Thoughts (ToT)** — The closest technique to how a human expert tackles hard problems. Reasoning becomes a **tree**: generate multiple candidate thoughts, self-evaluate each, prune dead ends, and deepen promising branches using BFS/DFS/beam search. Unlike Self-Consistency (independent chains, vote at end), ToT has **lookahead and backtracking**. The accessible version is the "imagine three experts…" prompt. Highest cost — reserve for important decisions.

---

## Key Distinctions People Confuse

| Pair | The Difference That Matters |
|------|-----------------------------|
| **Few-Shot vs Few-Shot CoT** | Few-Shot shows the answer; Few-Shot CoT shows the *reasoning that leads to* the answer. |
| **Self-Consistency vs ToT** | Both use multiple paths, but Self-Consistency runs **independent** chains and votes only at the **end** — no intermediate evaluation, no backtracking. ToT evaluates **every node**, prunes early, and can backtrack (lookahead). |
| **CoT vs ReAct** | CoT = reasoning only over frozen internal knowledge. ReAct = reasoning **+** real tool calls that fetch current, verified facts. |
| **ReAct vs RAG** | RAG retrieves **once, before** generating. ReAct retrieves **iteratively, during** generation, guided by its own reasoning state, with any tool (not just a retriever). |
| **Prompt Chaining vs ReAct** | Chaining = a **fixed, predetermined** pipeline (deterministic, easy to debug). ReAct = the model **dynamically decides** its next action (flexible, harder to debug). |
| **Meta Prompting vs Few-Shot CoT** | Meta Prompting teaches *how to structure any response* (content-agnostic template). Few-Shot CoT teaches *how to reason about these specific problems* (content-specific chains). |
| **DSP vs Standard Prompting** | Standard prompt = one static string for all inputs. DSP = a trained model generates a **custom hint per input**. |

---

## Decision Guide — Which Technique When?

```
Is the task simple & well-defined?
├── Yes → Zero-Shot
└── No
    ├── Just need a specific format/style (no deep reasoning)?
    │       → Few-Shot
    │
    ├── Needs multi-step reasoning?
    │   ├── Have good worked examples?      → Few-Shot CoT
    │   ├── No examples?                    → Zero-Shot CoT
    │   ├── Large dataset, want to scale?   → Auto-CoT
    │   └── Accuracy is critical?           → wrap any of the above in Self-Consistency
    │
    ├── Needs current info / real-world actions / tools?
    │       → ReAct   (multi-hop, dynamic, multi-tool)
    │
    └── Needs higher-level steering?
        ├── Consistent structure across many domains?   → Meta Prompting
        ├── Per-input hints for a frozen/API LLM?        → DSP
        ├── Clean sequential sub-tasks, want checkpoints? → Prompt Chaining
        └── Many possible approaches / first idea often wrong / planning?
                → Tree of Thoughts
```

---

## Cross-Cutting Themes

- **The ladder principle:** every technique is a direct response to a failure mode of the one below it. Learning them in order makes each one's motivation obvious.
- **In-Context Learning:** Few-Shot through Few-Shot CoT all work *without changing model weights* — the model adapts purely from context tokens at inference time.
- **Reasoning vs Acting:** Sections 2 (CoT family) make a *single call* reason better; Section 3 (ReAct) lets the model *act* on the world; Section 4 (Meta/DSP/Chaining/ToT) *steer* the overall strategy and workflow.
- **The cost–accuracy trade-off:** moving up the ladder generally buys accuracy at the price of tokens, latency, and complexity. Use the cheapest technique that reliably solves your task.
- **Grounding beats guessing:** ReAct, RAG, and DSP all inject external or input-derived signal to reduce hallucination — a recurring theme in reliable LLM systems.

---

## Papers at a Glance

| Technique | Paper | Year |
|-----------|-------|------|
| Few-Shot / ICL | Brown et al. (GPT-3) | 2020 |
| Chain-of-Thought | Wei et al. | 2022 |
| Zero-Shot CoT | Kojima et al. | 2022 |
| Auto-CoT | Zhang et al. (arXiv:2210.03493) | 2022 |
| Self-Consistency | Wang et al. (arXiv:2203.11171) | 2022 |
| ReAct | Yao et al., ICLR | 2023 |
| Meta Prompting | Zhang, Yuan & Yao (arXiv:2311.11482) | 2023 |
| DSP | Li et al., NeurIPS | 2023 |
| Tree of Thoughts | Li et al., NeurIPS | 2023 |

---

*Companion to the Section 1–5 notes for Lec 7. Use this table to choose a technique; use the section notes to master each one.*
