# Lec 7 — Section 3: Acting & Tool Use
**Course:** Advanced Prompt Engineering | IIT Kharagpur  
**Instructor:** Prof. Koustav Rudra  
**Module:** 2, Session 1

---

## Topics Covered
8. ReAct Prompting — Reasoning + Acting with external tools

---

## Why This Section Exists: The Wall CoT Hits

By Section 2, Chain-of-Thought (CoT) was a major leap. It let models reason step by step, dramatically improving accuracy on multi-step problems. But CoT has a **hard ceiling**: it can only reason over knowledge that was frozen inside the model at training time.

Consider these failure modes CoT cannot fix:

| Problem | Why CoT fails |
|---|---|
| "What is today's stock price of NVIDIA?" | Knowledge is from training cutoff — stale |
| "Search for the latest paper on RAG from 2025" | Model has no access to post-training data |
| "Book a flight to Delhi for next Friday" | Model cannot take real-world actions |
| "Run this code and tell me the output" | Model cannot execute anything |

CoT is **reasoning without acting**. ReAct is the bridge that lets LLMs **reason AND act** — interleaving thought with tool use to solve tasks that require real-world interaction.

---

---

## Topic 8 — ReAct Prompting

### What is ReAct?

**ReAct** (Reasoning + Acting) is a prompting framework where an LLM generates **reasoning traces** (thoughts) and **task-specific actions** in an interleaved, cyclical manner.

The model doesn't just think — it acts. It calls tools (search engines, APIs, calculators, code executors), observes the results, and uses those results to refine its next thought.

**Paper:** Shunyu Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models*, ICLR 2023

### The Inspiration

ReAct is inspired by how **humans learn and make decisions**:
- We don't solve problems purely in our heads (reasoning only)
- We don't blindly take actions without thinking (acting only)
- We interleave thought and action: think → try → observe → revise thinking → try again

A doctor doesn't diagnose purely from memory — they order tests (actions), see results (observations), update their hypothesis (thought), and order more tests. ReAct encodes this cycle in a prompt.

---

### The Core Loop: Thought → Action → Observation

The ReAct cycle has exactly three components, repeated until the task is complete:

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   THOUGHT  →  What do I need to do next?            │
│      ↓         (internal reasoning, planning)       │
│   ACTION   →  Call a tool or take an action         │
│      ↓         (Search, Lookup, Calculate, etc.)    │
│ OBSERVATION → Feed the result back as context       │
│      ↓         (tool output becomes next input)     │
│   [REPEAT until FINISH]                             │
│                                                     │
└─────────────────────────────────────────────────────┘
```

Each component in detail:

| Component | Role | Example |
|---|---|---|
| **Thought** | The model reasons about what it knows, what it needs, and what to do next | *"I need to find which city the Taj Mahal is in, then find the river."* |
| **Action** | The model issues a specific tool call with parameters | `Search[Taj Mahal location]` |
| **Observation** | The tool's response is fed back into the model's context | *"The Taj Mahal is in Agra, on the banks of the Yamuna river."* |

The loop repeats — each Observation informs the next Thought, which triggers the next Action — until the model issues a `Finish[answer]` action with the final answer.

---

### Step-by-Step Walkthrough: Apple Remote Example

This is the HotpotQA (multi-hop question answering) example from the lecture.

**Question:**
> *"Aside from the Apple Remote, what other devices can control the program Apple Remote was originally designed to interact with?"*

This is a **multi-hop** question — you need to:
1. Find what program the Apple Remote was designed to control
2. Find what other devices can control that program

Neither fact is directly in the question. A model without external knowledge would hallucinate.

---

**Iteration 1:**

```
Thought 1: I need to search Apple Remote and find the program it was
           originally designed to interact with.

Action 1: Search[Apple Remote]

Observation 1: The Apple Remote is a remote control introduced in October 2005
               by Apple, originally designed to control the Front Row media
               center program...
```

The model correctly decomposes the question into sub-goals and searches for the first piece of information.

---

**Iteration 2:**

```
Thought 2: Apple Remote was originally designed to control the Front Row
           media center program. I need to search Front Row next and find
           what other devices can control it.

Action 2: Search[Front Row]

Observation 2: Could not find [Front Row]. Similar: ['Front Row Seat to Earth',
               'Front Row Motorsports', 'Front Row (software)', ...]
```

The search failed — the model hits an ambiguity. This is where ReAct's explicit reasoning loop shines: the model uses the Observation to adapt.

---

**Iteration 3:**

```
Thought 3: Front Row is not found. I need to search Front Row (software)
           specifically.

Action 3: Search[Front Row (software)]

Observation 3: Front Row is a discontinued media center software application
               developed by Apple. It can be controlled by an Apple Remote
               or the keyboard function keys...
```

The model pivoted correctly from the failed search and found the answer.

---

**Iteration 4:**

```
Thought 4: Front Row (software) is controlled by an Apple Remote or the
           keyboard function keys. So the answer is keyboard function keys.

Action 4: Finish[keyboard function keys]
```

**Answer: keyboard function keys** ✅

---

### What Made This Possible

Notice what happened across these 4 iterations:

1. **Decomposition** — Thought 1 broke the question into a clear sub-goal chain
2. **Tool use** — Action calls retrieved real knowledge the model didn't have
3. **Recovery** — When Search[Front Row] failed, the Observation surfaced similar results, and the model adapted its search term
4. **Synthesis** — The final Thought assembled pieces from multiple Observations into the answer

None of this is possible with pure CoT (no tool access) or pure Act (no intermediate reasoning).

---

### Comparison: Standard vs CoT vs Act-Only vs ReAct

The lecture presents a direct comparison on the same HotpotQA question:

| Method | Approach | Answer | Correct? |
|---|---|---|---|
| **Standard** | Direct answer, no reasoning, no tools | iPod | ❌ Wrong |
| **CoT (Reason only)** | Step-by-step reasoning, no tools | iPhone, iPad, iPod Touch | ❌ Wrong |
| **Act only** | Tool calls, no reasoning | yes | ❌ Wrong (no reasoning to interpret results) |
| **ReAct** | Interleaved Thought + Action + Observation | keyboard function keys | ✅ Correct |

**Why CoT fails here:** CoT can reason, but it's reasoning over stale or hallucinated knowledge. It "knows" Apple Remote was connected to iPod because that's a common association in training data — but it's wrong for this specific question about the program it was designed to control.

**Why Act-only fails:** Without reasoning, the model can call tools but can't decompose the multi-hop question or adapt when a search returns unexpected results.

**Why ReAct succeeds:** The reasoning keeps the model on track (knows what it's looking for), while the actions fetch accurate, current information.

---

### The Free-Form Thought: Four Uses

In ReAct, the Thought step can serve different roles depending on what the model needs at that moment:

| Thought Role | Example | When used |
|---|---|---|
| **Decomposition** | *"I need to find X first, then Y."* | At the start of a multi-step task |
| **Information extraction** | *"The observation says the remote controls Front Row."* | After receiving an Observation |
| **Search reformulation** | *"Front Row wasn't found, I should try Front Row (software)."* | After a failed tool call |
| **Synthesis / final answer** | *"Both facts together mean the answer is keyboard function keys."* | Before issuing Finish |

This flexibility is what makes ReAct powerful — the Thought is unconstrained natural language reasoning, not a rigid template.

---

### ReAct Prompt Structure

A ReAct prompt typically has two parts:

**Part 1 — Few-Shot Demonstrations** (optional but recommended)

Provide 1-6 complete Thought → Action → Observation trajectories on solved examples. This teaches the model the format and style of reasoning.

```
Example trajectory 1:
Question: [question]
Thought 1: [reasoning]
Action 1: Search[query]
Observation 1: [result]
Thought 2: [reasoning]
Action 2: Finish[answer]

Example trajectory 2:
...
```

**Part 2 — The Actual Query**

```
Question: [new question]
Thought 1:   ← model starts generating here
```

The model continues the pattern, generating Thoughts and Actions, with the system injecting Observations after each Action.

---

### Tool Types in ReAct

ReAct works with any tool that can receive a string query and return a string response. Common tools:

| Tool | Action syntax | Use case |
|---|---|---|
| **Search** | `Search[query]` | Retrieve web or knowledge base results |
| **Lookup** | `Lookup[entity]` | Find specific entity info (like Wikipedia Ctrl+F) |
| **Calculator** | `Calculate[expression]` | Arithmetic operations |
| **Code executor** | `Execute[code]` | Run and observe code output |
| **API call** | `Call[endpoint, params]` | Retrieve real-time data |
| **Finish** | `Finish[answer]` | Terminate with final answer |

The key requirement: each tool must return **deterministic, factual output** that the model can observe and reason over.

---

### ReAct vs RAG (Retrieval Augmented Generation)

Both ReAct and RAG address the "stale knowledge" problem, but differently:

| Dimension | RAG | ReAct |
|---|---|---|
| **When retrieval happens** | Once, before generation | Multiple times, during generation |
| **What drives retrieval** | The original query | The model's current reasoning state |
| **Adaptability** | Fixed single retrieval | Can pivot based on intermediate results |
| **Tool diversity** | Typically just a retriever | Any tool (search, APIs, code, calculators) |
| **Reasoning** | LLM reasons over retrieved docs | LLM reasons interleaved with retrieval |
| **Best for** | Single-hop factual queries | Multi-hop, multi-tool, dynamic tasks |

Think of RAG as "look it up once before answering" and ReAct as "look things up as you go, guided by reasoning."

---

### Hallucination Reduction in ReAct

One of the most practically important benefits of ReAct is its ability to reduce hallucination:

- In standard prompting, when the model doesn't know something, it fabricates an answer confidently
- In ReAct, the model is "allowed" to express uncertainty in a Thought and then take an Action to retrieve real information
- The Observation injects **ground truth** from external sources into the context, anchoring subsequent reasoning

**Example of hallucination prevention:**
```
Without ReAct:
Q: What year did Apple release AirTag?
A: Apple released AirTag in 2019.   ← Hallucination (actual: 2021)

With ReAct:
Thought: I should look up when Apple released AirTag.
Action: Search[Apple AirTag release date]
Observation: Apple AirTag was released on April 22, 2021.
Thought: The AirTag was released in 2021.
Finish: 2021   ← Correct ✅
```

---

### Limitations of ReAct

Despite its power, ReAct has real limitations worth knowing:

| Limitation | Description | Mitigation |
|---|---|---|
| **Latency** | Multiple tool calls add round-trip time | Parallelize independent actions |
| **Cost** | More tokens generated (reasoning traces + observations) | Use only for tasks that truly need it |
| **Error propagation** | A wrong Observation can mislead subsequent reasoning | Add verification steps |
| **Tool reliability** | Model is only as good as its tools | Use high-quality, reliable APIs |
| **Reasoning loops** | Model can get stuck searching in circles | Add iteration limit / fallback |
| **Prompt sensitivity** | ReAct requires carefully formatted demonstrations | Standardize trajectory format |
| **Not always needed** | Adds complexity for simple tasks | Use plain CoT when no tools are needed |

---

### ReAct in Modern AI Agents

ReAct is the **foundational architecture** of modern LLM agents. Today's AI agent frameworks (LangChain, LlamaIndex, AutoGPT, Claude's tool use) are all implementations of this Thought → Action → Observation loop:

```
Modern Agent = ReAct + Better Tools + Memory + Planning
```

Specifically:
- **LangChain's AgentExecutor** — implements ReAct loop with tool routing
- **OpenAI Function Calling / Tool Use** — structured ReAct where Actions are typed function calls
- **Anthropic's Claude tool use** — same pattern; Claude generates tool inputs, system injects outputs
- **AutoGPT / BabyAGI** — ReAct extended with long-term memory and task planning

Understanding ReAct means understanding the core of every LLM agent architecture.

---

> **Learning Thought:**  
> ReAct is the moment prompting becomes *agentic*. Before ReAct, an LLM was a very smart text predictor — you gave it a prompt, it gave you text. With ReAct, the LLM becomes an **agent** that perceives (Observations), reasons (Thoughts), and acts (Actions) in a loop. This is the same Perception → Cognition → Action cycle from classical AI agent theory, now implemented entirely through prompt structure. The beauty of ReAct is that it requires no architectural changes to the model — just a different prompt format. That's the power of prompting: changing *what the model is* without touching its weights.

---

> **Learning Thought:**  
> Notice that in the Apple Remote example, the model recovered from a failed search (Iteration 2) by reformulating its query. This error recovery behavior emerged purely from the reasoning trace — the model saw "Could not find [Front Row]" and used its own language understanding to try a more specific search term. This is not hard-coded logic — it's the model applying common sense reasoning to an API failure. That's what makes ReAct so robust: the Thought component gives the model agency to adapt when actions fail, rather than crashing.

---

### Interview Questions — Topic 8

**Q1. What is ReAct and what problem does it solve?**  
ReAct (Reasoning + Acting) is a prompting framework that interleaves language model reasoning traces (Thoughts) with external tool calls (Actions) and their results (Observations). It solves the fundamental limitation of pure CoT — that LLMs can only reason over knowledge frozen in their weights at training time. ReAct allows models to retrieve current information, perform real-world actions, and adapt based on tool outputs.

**Q2. Describe the Thought → Action → Observation loop in ReAct.**  
In each iteration: (1) **Thought** — the model reasons about what it knows and what it needs to do next; (2) **Action** — the model issues a tool call with specific parameters; (3) **Observation** — the tool's output is injected back into the model's context. This loop repeats until the model issues a `Finish` action with the final answer.

**Q3. Why does "Act only" (no reasoning) fail on multi-hop questions?**  
Without reasoning, the model has no mechanism to decompose a multi-hop question into sub-goals, track what it has already found, adapt when a tool call fails, or synthesize multiple pieces of information into a final answer. It can call tools but cannot interpret and chain the results intelligently.

**Q4. Why does "CoT only" (no tool use) fail on questions requiring current information?**  
CoT reasons over the model's internal knowledge, which is frozen at training time. For questions requiring real-time data, post-training facts, or retrieval from external systems, CoT will reason coherently but over incorrect or hallucinated premises — leading to confidently wrong answers.

**Q5. What are the four roles that the Thought step can serve in ReAct?**  
(1) **Decomposition** — breaking a complex question into sub-goals; (2) **Information extraction** — parsing relevant facts from an Observation; (3) **Search reformulation** — adapting the next Action when a previous one failed; (4) **Synthesis** — combining information from multiple Observations to form the final answer.

**Q6. How does ReAct reduce hallucination?**  
Instead of generating an answer from internal (potentially incorrect) knowledge, ReAct grounds the model's responses in Observations from external tools. The injected tool outputs act as factual anchors in the context window, and the model's subsequent reasoning is conditioned on these verified facts rather than on fabricated information.

**Q7. What is the difference between ReAct and RAG?**  
RAG performs a single retrieval step before generation — it fetches relevant documents and then generates a response. ReAct retrieves dynamically and iteratively — the model decides *when* and *what* to retrieve based on its current reasoning state, can make multiple retrievals, and can use diverse tools beyond just a vector store. RAG is better for simple single-hop factual queries; ReAct is better for multi-hop, dynamic, or multi-tool tasks.

**Q8. How does ReAct relate to modern LLM agent frameworks like LangChain or Claude's tool use?**  
Modern agent frameworks are direct implementations of the ReAct loop. LangChain's AgentExecutor runs the Thought-Action-Observation cycle. OpenAI's function calling and Anthropic's tool use are structured versions where Actions are typed function calls with JSON schemas rather than free-text tool invocations. ReAct is the foundational pattern — agents are ReAct + memory + planning + better tooling.

**Q9. What are the main limitations of ReAct?**  
Increased latency (multiple tool round-trips), higher token cost (reasoning traces + observations), potential error propagation from incorrect observations, risk of reasoning loops, dependency on tool reliability, and added complexity for tasks that don't require external knowledge.

**Q10. Why is ReAct considered "agentic" compared to standard prompting?**  
Standard prompting is a single-turn input-output function: you give a prompt, you get text. ReAct implements the Perception → Cognition → Action cycle — the model perceives its environment (Observations), reasons about it (Thoughts), and takes actions that change the environment (tool calls). This makes it a proper cognitive agent, not just a text predictor.

---

---

## Quick Reference Cheat Sheet — Section 3

| Concept | One-Line Summary |
|---|---|
| **ReAct** | Interleave Thought + Action + Observation in a loop to solve tasks requiring real-world tool use |
| **Thought** | Free-form reasoning: decompose, extract, reformulate, synthesize |
| **Action** | Tool call with specific parameters (Search, Lookup, Calculate, Finish) |
| **Observation** | Tool output injected back into context as new information |
| **Multi-hop question** | Requires chaining multiple facts/lookups to reach the answer |
| **Hallucination reduction** | Observations ground the model in verified external facts |
| **ReAct vs RAG** | RAG = single retrieval before answering; ReAct = iterative retrieval guided by reasoning |
| **ReAct vs CoT** | CoT = reasoning only; ReAct = reasoning + real-world actions |
| **Finish action** | Special action that terminates the loop and returns the final answer |
| **Agentic AI** | ReAct is the core loop of all modern LLM agent architectures |

---

## Key Paper — Section 3

| Paper | Year | Contribution |
|---|---|---|
| Shunyu Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models* | ICLR 2023 | Introduced the Thought-Action-Observation loop; demonstrated improvements on HotpotQA, FEVER, ALFWorld, WebShop |

---

*Next: Section 4 — Steering & Advanced Patterns (Meta Prompting, DSP, Prompt Chaining, Tree of Thoughts)*
