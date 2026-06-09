# Section 5: RAG Variants

> **Core Idea:** Standard RAG is stateless and single-hop. Real-world use cases demand more: multi-turn memory, autonomous planning, and iterative multi-hop reasoning. Each variant in this section is an answer to a specific limitation of the baseline.

---

## Topic 27: Types of RAG — Overview

The lecture covers four progressively more capable RAG architectures:

| Variant | Key Property | Limitation it Solves |
|---------|-------------|---------------------|
| **Standard RAG** | Stateless, single-step retrieval | Baseline; solves hallucination & knowledge cutoff |
| **RAG with Memory** | Tracks past conversation turns | Stateless RAG fails on multi-turn conversations |
| **Agentic RAG** | Plans, decomposes, uses multiple tools | Standard RAG fails on complex, multi-step questions |
| **Chain-of-RAG (CoRAG)** | Iterative, dynamic multi-hop retrieval | Single-step retrieval fails on multi-hop questions |

Each variant **extends** the previous one. Understanding them in order reveals the clean conceptual progression.

---

## Topic 28: Standard RAG — The Stateless Baseline

### What Makes It "Standard"?
Standard RAG is **stateless**: every query is treated completely independently. There is no memory of previous queries, previous answers, or previously retrieved documents.

```
Query₁ → Retrieve → Generate → Answer₁
Query₂ → Retrieve → Generate → Answer₂
(Query₁ and its context have no influence on Query₂)
```

### Architecture
```
Query ──────────────► Retrieval ──────────────► LLM ──► Generated Output
                          │                      ▲
                      Lookup               Relevant info
                          │
                       Database
```

### How It Works — Step by Step
1. System embeds the query using the retrieval model
2. Looks up the closest passage(s) in the document store
3. Constructs the augmented prompt (context + query)
4. LLM generates the answer grounded in the retrieved passage

**Example:**
```
Q: "What is the boiling point of water at sea level?"
→ System embeds: "boiling point water sea level"
→ Retrieves: physics reference passage about 100°C at 1 atm
→ LLM answers: "100°C" grounded in that passage
```

### When Standard RAG Works Well
- Self-contained, single-turn questions
- Fact lookup ("What is X?", "When did Y happen?")
- Document Q&A where each question is independent
- Customer support FAQs where each question is standalone

### When Standard RAG Fails
- Multi-turn conversations (no memory of "I'm a vegetarian" for the follow-up "Suggest high-protein foods")
- Questions that require combining facts from multiple documents
- Complex tasks that require planning and tool use

> **Learning Thought:** Standard RAG is the canonical "Lewis et al. 2020 NeurIPS" RAG. It proved the concept works. Everything that follows is an engineering extension to handle the cases it fails on.

---

## Topic 29: RAG with Memory

### The Problem Standard RAG Cannot Solve
Standard RAG treats each query independently. But real conversations have context:

```
Q1: "I'm a vegetarian."
Q2: "Suggest high-protein foods."
```

A stateless system answers Q2 as if Q1 never happened. The user gets a list of meat-based proteins — completely wrong for their dietary constraint.

### What Memory Adds
RAG with Memory **persists state across conversation turns**. It:
1. Stores past questions, answers, and retrieved documents in a memory store
2. When a new query arrives, retrieves *both* from the external knowledge base *and* from the conversation history
3. Rewrites ambiguous queries using the stored context

```
Query ──────────────────────────────────────────► Retrieval ──► LLM ──► Generated Output
  │                                                    │           ▲
  │         Lookup                                  Lookup    Relevant info
  │◄── Past conversation  ◄─────────────────────────────┘
              │
         Memory Store
         (past Q&A, retrieved docs)
```

### Memory Is Not Just History Replay
Memory in RAG is active, not passive:
- It **tracks** past interactions: prior questions, answers, retrieved documents
- It **understands how previous context influences new searches** — not just appending Q1 to Q2, but rewriting Q2 in light of Q1

### The Paris Example from Slides
```
Q1: "What is the capital of France?"
A1: "Paris."

Q2: "What about its population?"
```
The phrase "its population" is ambiguous without context. RAG with Memory:
1. Recalls: Q1 was about France/Paris
2. Determines: "its" refers to Paris
3. Rewrites query: "What is the population of Paris?"
4. Retrieves: Paris population data
5. Answers: "Paris has about 2.05 million people"

### Use Cases

| Domain | Use Case |
|--------|---------|
| Personal AI agents | Remember user preferences across sessions |
| Conversational chatbots | Multi-turn dialogue with context |
| Customer support | Remember what the customer already told the agent |
| Educational tutoring | Track what the student has already learned/asked |

> **Learning Thought:** Memory in RAG is the equivalent of working memory in human cognition. Stateless RAG is like having amnesia between every sentence. Adding memory is the minimum requirement for any real conversational application.

---

## Topic 30: Agentic RAG

### The Problem Memory-RAG Cannot Solve
Even with memory, standard RAG is **reactive** — it retrieves from one source in response to one query. Some questions require:
- Decomposing into sub-tasks
- Using multiple tools (search, calculator, APIs)
- Verifying intermediate results before proceeding
- Iterating when the first retrieval was insufficient

### What Agentic RAG Does
Instead of just retrieving documents, an Agentic RAG system:
1. **Plans** its approach to answering the question
2. **Decides** what to investigate and in what order
3. **Takes action** using a suite of tools
4. **Reflects** on whether what it found answers the question
5. **Keeps searching** if the answer is incomplete

```
Query ──► Retrieval Agent ──► Tools:
               │                ├── Vector Search Engine A ──► Collection A
               │                ├── Vector Search Engine B ──► Collection B
               ▼                ├── Calculator
              LLM               └── Web Search
               │
               ▼
           Response
```

### The Paris Population Example from Slides
```
Q: "How many more people live in Paris's metropolitan area than in the city proper?"

Agent plans:
1. Decompose: needs two figures (city proper, metro), then subtract
2. Web Search / collection lookup: city proper ≈ 2.1M; metro ≈ 11.3M
3. Reflect: "Do I have both numbers? Yes." → proceed
4. Calculator tool: 11.3M - 2.1M = 9.2M

Answer: "About 9.2 million more people live in the Paris metro than in the city proper."
```

A standard RAG would have retrieved a single passage and attempted one-shot computation — almost certainly failing.

### Key Behavioural Properties of Agentic RAG
- **Multi-source retrieval** — searches multiple collections or databases
- **Tool use** — not just retrieval; can call calculators, APIs, code interpreters
- **Self-verification** — checks whether retrieved information is sufficient before generating
- **Iterative refinement** — loops back to retrieve more if needed
- **Task decomposition** — breaks complex questions into answerable sub-questions

### Use Cases

| Domain | Use Case |
|--------|---------|
| Legal research | Agent retrieves case law, statutes, precedents from multiple databases |
| Financial analysis | Combines market data, regulatory filings, news articles |
| Scientific research | Aggregates papers, citations, and experimental data |
| Multi-step customer support | Checks order status, policy docs, and account history simultaneously |

> **Learning Thought:** Agentic RAG is where RAG meets the Agents paradigm. The LLM is no longer a generator — it is a planner and orchestrator. The key shift is from **reactive** (answer this query) to **proactive** (figure out what information is needed, get it, verify it, answer).

---

## Topic 31: Chain-of-RAG (CoRAG)

### The Problem Agentic RAG Still Has
Agentic RAG plans broadly. But for questions that require **strict sequential reasoning** — where each step's answer determines the next query — a more structured approach is needed.

### The Multi-Hop Problem
```
Q: "What's the nationality of the director of the film that won Best Picture the year Titanic was released?"
```

This question has four **nested dependencies**:
1. When was Titanic released? → 1997
2. What won Best Picture for 1997? → Titanic (at the 1998 ceremony)
3. Who directed Titanic? → James Cameron
4. What is James Cameron's nationality? → Canadian

A single retrieval step **cannot** answer this because the correct retrieval query for step 4 depends on the answer to step 3, which depends on step 2, which depends on step 1.

### Problems with Traditional RAG for Multi-Hop Questions
| Problem | Description |
|---------|-------------|
| **Imperfect Retrieval** | Single-step retrieval fails to gather all relevant information for complex tasks |
| **Static Reasoning** | No iterative refinement → suboptimal generation |
| **Contextual Misalignment** | Queries not adapted dynamically as new information surfaces |

### What CoRAG Does Differently
**Chain-of-RAG** extends RAG with **dynamic, multi-step retrieval** — each retrieval step is informed by the answers from the previous step:

```
Initial Query
     │
     ▼
Sub-query 1 → Retrieve → Sub-answer 1
                                │
                                ▼
Sub-query 2 → Retrieve → Sub-answer 2  (informed by Sub-answer 1)
                                │
                                ▼
...continue until sufficient information gathered...
                                │
                                ▼
                         Final Answer
```

### The Titanic Example from Slides
```
Sub-query 1: "When was Titanic released?" → 1997
Sub-query 2: "What won Best Picture for 1997 films?" → Titanic (1998 ceremony)
Sub-query 3: "Who directed Titanic?" → James Cameron
Sub-query 4: "What is James Cameron's nationality?" → Canadian

Final Answer: "Canadian"
```

Each sub-query is dynamically formulated based on the sub-answer from the previous step. This mirrors how humans approach complex research questions.

---

## Topic 32: CoRAG Mechanics — How It Works Internally

### Mechanism 1: Dynamic Retrieval Chains
Starting from the initial query, CoRAG **iteratively refines** retrieval:
- Generate a sub-query
- Retrieve relevant documents for that sub-query
- Extract a sub-answer
- Use the sub-answer to formulate the next sub-query
- Repeat until the system determines it has sufficient information for a final answer

**Stopping criterion:** The system checks at each step: "Do I now have enough information to answer the original question?" If yes, generate the final answer. If no, continue the chain.

### Mechanism 2: Rejection Sampling for Training
During training, CoRAG generates **intermediate retrieval chains** — sequences of (sub-query, retrieved doc, sub-answer) — and uses these to augment training datasets. This improves the model's ability to formulate good sub-queries. Think of it as teaching the model *how to decompose* questions by showing it many examples of successful decompositions.

### Mechanism 3: Flexible Decoding Strategies
At inference time, CoRAG can use different decoding strategies depending on the compute budget:
| Strategy | Description | Trade-off |
|----------|-------------|-----------|
| Greedy decoding | Always take the highest-probability next step | Fast; may miss optimal chains |
| Best-of-N sampling | Generate N chains, pick the best | Better quality; N× cost |
| Tree search | Explore multiple branches at each step | Highest quality; most expensive |

### More Examples from Slides

**Example 2: Gravitational Waves**
```
Q: "What experiments led to the discovery of gravitational waves?"
Step 1: "What are gravitational waves?" → "Ripples in spacetime caused by massive accelerating objects"
Step 2: "Which experiment first detected gravitational waves?" → "The LIGO experiment in 2015"
Final: "The LIGO experiment in 2015 detected gravitational waves, confirming Einstein's theory of general relativity"
```

**Example 3: Best Picture (1994)**
```
Q: "What is the nationality of the director of the 1994 film that won Best Picture?"
Step 1: "Which film won Best Picture in 1994?" → "Forrest Gump"
Step 2: "Who directed Forrest Gump?" → "Robert Zemeckis"
Step 3: "What is Robert Zemeckis's nationality?" → "American"
Final: "American"
```

> **Learning Thought:** CoRAG is essentially applying Chain-of-Thought reasoning to the *retrieval* step, not just the generation step. The insight is that complex questions require an iterative dialogue between the model and the retrieval system — not a single lookup followed by generation.

---

## RAG Variants — Comprehensive Comparison

| Dimension | Standard RAG | RAG with Memory | Agentic RAG | CoRAG |
|-----------|-------------|----------------|-------------|-------|
| State | Stateless | Stateful (conversation) | Stateful + planning | Stateful + iterative |
| Retrieval steps | 1 | 1 (with history context) | Multiple (parallel or planned) | Multiple (sequential, chained) |
| Query sources | Single | Single + conversation store | Multiple tools + databases | Sequential sub-queries |
| Planning | None | None | Yes (LLM plans) | Yes (chain structure) |
| Multi-hop support | No | No | Partial | Yes (designed for it) |
| Complexity | Low | Medium | High | High |
| Latency | Low | Low | High | High |
| Best for | Simple Q&A | Conversational agents | Complex multi-source tasks | Multi-hop reasoning questions |

---

## Interview Questions — Section 5

### Fundamental

**Q1. What is the key difference between Standard RAG and RAG with Memory?**
> Standard RAG is stateless: every query is independent, with no knowledge of prior interactions. RAG with Memory adds a conversation store that tracks past queries, answers, and retrieved documents. When a new query arrives, it retrieves from both the knowledge base *and* the conversation history, enabling context-aware responses and automatic query rewriting to resolve references like "it" or "the same."

**Q2. What is Agentic RAG and how does it differ from Standard RAG?**
> Agentic RAG replaces the passive retrieve-then-generate loop with an active agent that plans its approach, decides what to investigate, uses multiple tools (search engines, calculators, APIs), verifies intermediate results, and iterates until the question is answered. Standard RAG does a single retrieval lookup. Agentic RAG treats the LLM as an orchestrator, not just a generator.

**Q3. What problem does CoRAG solve that Standard RAG cannot?**
> Standard RAG does a single retrieval step, which fails for multi-hop questions where the correct retrieval query for step N depends on the answer retrieved in step N-1. CoRAG enables dynamic, sequential retrieval chains — each sub-query is formulated based on the sub-answer from the previous step, mirroring how humans reason through complex multi-step questions.

### Intermediate

**Q4. Walk through how CoRAG would answer: "Who wrote the novel that was adapted into the highest-grossing film of 1994?"**
> Sub-query 1: "What was the highest-grossing film of 1994?" → "The Lion King"
> Sub-query 2: "Is The Lion King based on a novel?" → "It is inspired by Hamlet, not directly adapted from a novel" — chain adjusts or backtracks
> Or alternative path: "Forrest Gump" (if the data differs) → Sub-query 2: "What novel is Forrest Gump based on?" → "Forrest Gump (1986) by Winston Groom" → "Winston Groom"

**Q5. What are the three decoding strategies in CoRAG and when would you choose each?**
> (1) **Greedy decoding** — take the highest-probability next step at each stage; fastest, lowest quality ceiling.
> (2) **Best-of-N sampling** — generate N complete chains and pick the best one; N× cost, significantly better quality for complex reasoning.
> (3) **Tree search** — explore multiple branches at each step; highest quality, most computationally expensive; use only when answer quality is critical and latency is not.

**Q6. What is rejection sampling training in CoRAG?**
> CoRAG generates intermediate retrieval chains (sub-query → retrieved doc → sub-answer sequences) and uses these as training data to augment the model's ability to decompose questions. The "rejection" aspect: chains that lead to incorrect final answers are rejected, while successful chains are kept. This teaches the model the reasoning patterns behind good question decomposition.

### Advanced

**Q7. How would you choose between Agentic RAG and CoRAG for a complex question-answering system?**
> Agentic RAG is better when: (a) the question requires multiple *parallel* information sources (market data + regulatory filings simultaneously), (b) tools beyond text retrieval are needed (calculators, code execution, APIs), (c) the plan is not strictly sequential.
> CoRAG is better when: (a) the question has strict sequential dependencies (each answer determines the next query), (b) the task is pure text-based multi-hop reasoning, (c) you want a principled, trainable framework with explicit chain structure.
> In practice: Agentic RAG is more flexible; CoRAG is more precise for multi-hop Q&A benchmarks.

**Q8. What are the latency implications of CoRAG vs Standard RAG?**
> Standard RAG: 1 retrieval call + 1 LLM generation call = ~2 API calls total.
> CoRAG: K retrieval calls + K sub-answer generation calls + 1 final generation call = ~2K+1 API calls, where K is the chain length (typically 2–5 steps).
> CoRAG is 3–10× slower. Mitigation strategies: (a) parallelize independent sub-queries; (b) cache intermediate results; (c) use fast local models for sub-answer generation and a powerful model only for the final answer; (d) set a maximum chain depth to bound latency.

---

## Key Learning Thoughts — Section 5

> **Thought 1 — Each variant solves one specific failure mode:** Standard RAG → hallucination + knowledge cutoff. Memory → multi-turn context loss. Agentic → complex multi-source tasks. CoRAG → sequential multi-hop reasoning. Don't apply a complex variant when a simpler one suffices.

> **Thought 2 — The spectrum from Naive to Agentic is a complexity trade-off:** Each step up the ladder adds capability and latency. Production systems often start with Standard RAG, measure failure modes, and only escalate to Memory/Agentic/CoRAG when the simpler variant demonstrably fails.

> **Thought 3 — CoRAG is Chain-of-Thought applied to retrieval:** If you understand CoT prompting (break reasoning into steps), CoRAG is the same insight applied to information gathering. The model doesn't just reason step-by-step — it *retrieves* step-by-step, with each step informed by the last.

> **Thought 4 — Agentic RAG blurs the line between RAG and AI agents:** The LLM in Agentic RAG is not a passive generator — it is a planning agent that decides what tools to use and when. This is the architectural bridge between RAG systems and autonomous AI agents. Understanding both is increasingly required.

> **Thought 5 — Memory is often the first upgrade you need:** Before jumping to agentic or CoRAG, most real-world applications fail simply because they are stateless. Adding conversation memory resolves a large class of practical failures at low engineering cost. Start here before tackling more exotic variants.

---

*Previous: [Section 4 — Common Problems of RAG](./04_Common_Problems_of_RAG.md) | Next: [Section 6 — RAG Assessment Framework →](./06_RAG_Assessment_Framework.md)*
