# Lec 7 — Section 4: Steering & Advanced Patterns
**Course:** Advanced Prompt Engineering | IIT Kharagpur  
**Instructor:** Prof. Koustav Rudra  
**Module:** 2, Session 1

---

## Topics Covered
9. Meta Prompting
10. Directional Stimulus Prompting (DSP)
11. Prompt Chaining
12. Tree of Thoughts (ToT)

---

## Why This Section Exists: Going Beyond Single-Step Reasoning

Sections 2 and 3 taught you how to make a single LLM call smarter — better reasoning (CoT), better knowledge access (ReAct). Section 4 is about a different class of problems:

| Challenge | Technique |
|---|---|
| The model reasons well but doesn't structure its approach optimally | **Meta Prompting** |
| The model needs task-specific guidance it can't generate itself | **Directional Stimulus Prompting** |
| The task is too complex for a single LLM call, needs a pipeline | **Prompt Chaining** |
| The problem has many possible paths and the first idea isn't always best | **Tree of Thoughts** |

These techniques are about **steering** the model's behavior at a higher level — not just improving individual reasoning steps, but shaping the overall strategy, structure, and workflow of problem-solving.

---

---

## Topic 9 — Meta Prompting

### The Core Idea

Standard prompting (zero-shot, few-shot, CoT) is **content-driven** — you give the model examples or instructions focused on *what* the task involves. **Meta Prompting** is **structure-driven** — instead of providing specific content, you instruct the model on *how to structure its entire problem-solving process*.

> Meta Prompting = Prompting the model to design its own reasoning strategy and response structure.

Think of it as the difference between:
- Teaching a student specific math problems (few-shot)
- Teaching a student *how to approach any math problem systematically* (meta prompting)

**Paper:** Yifan Zhang, Yang Yuan, and Andrew Chi-Chih Yao, *Meta Prompting for AI Systems*, arXiv:2311.11482, 2023

---

### Standard Prompting vs Meta Prompting

| | Standard Prompting | Meta Prompting |
|---|---|---|
| **Focus** | Specific content and examples | Structure, format, and reasoning workflow |
| **What you provide** | "Here are 3 examples of X, now do X" | "Here is the structure you must follow to solve any problem" |
| **Generalises to** | Problems similar to the examples | Any problem that fits the structural template |
| **Examples in prompt** | Task-specific input-output pairs | Abstract structural templates |
| **Syntax** | Natural language instructions | Formal structure markers, LaTeX, schemas |

---

### The Four Key Characteristics of Meta Prompting

**1. Abstract Examples**
Instead of concrete solved problems, meta prompts use *structural placeholders* that describe the shape of a solution without tying it to specific content.

```
Problem: [question to be answered]
Solution Structure:
  1. Begin with "Let's think step by step."
  2. Follow with clearly broken-down reasoning steps.
  3. End with the final answer in a LaTeX box: $\boxed{answer}$
  4. State "The answer is [final answer]."
```

The `[question]` and `[answer]` are structural slots — not concrete examples.

**2. Structure-Oriented**
The prompt prioritises *form* over *content*. It tells the model how to organise its response — what to say first, how to format intermediate steps, how to present the conclusion — without dictating what those steps should contain.

**3. Syntax-Focused**
Meta prompts often use formal syntax elements as structural anchors. For example, LaTeX `$\boxed{}$` is a concrete formatting directive. The model learns to produce structurally consistent output because the syntax constrains the shape of the response.

**4. Versatile**
Because meta prompts describe structure rather than content, the same meta prompt can apply across wildly different domains — algebra, logic, coding, essay writing — as long as the structural pattern fits.

---

### Worked Example from the Lecture

**The Meta Prompt (structure only):**

```
Problem: [question to be answered]
Solution Structure:
  1. Begin the response with "Let's think step by step."
  2. Follow with the reasoning steps, ensuring the solution process is
     broken down clearly and logically.
  3. End the solution with the final answer encapsulated in a
     LaTeX-formatted box: $\boxed{answer}$ for clarity and emphasis.
  4. Finally, state "The answer is [final answer].", presented in LaTeX.
```

**Combined with a few-shot example:**

```
Problem: Find the domain of √(x−2) / √(5−x).
Solution: The expressions inside each square root must be non-negative.
  Therefore x−2 ≥ 0, so x ≥ 2, and 5−x ≥ 0, so x ≤ 5.
  Also the denominator cannot equal zero, so 5−x > 0, giving x < 5.
  The domain is [2, 5). The final answer is [2, 5).

Problem: If det A = 2 and det B = 12, find det(AB).
```

**Model's Response:**

```
Let's think step by step.

1. We know det(AB) = det A × det B  [matrix multiplication property]
2. We are given det A = 2 and det B = 12.
3. Substituting: det(AB) = 2 × 12.
4. Calculate: det(AB) = 24.

So, the final answer is $\boxed{24}$.
```

The model followed the *structural template* perfectly — step-by-step reasoning, then LaTeX-boxed final answer — without the template itself knowing anything about linear algebra. The structure is domain-agnostic; only the content is domain-specific.

---

### Meta Prompting vs Few-Shot CoT

This distinction is subtle but important:

| | Few-Shot CoT | Meta Prompting |
|---|---|---|
| **What examples show** | Specific reasoning chains for specific problems | Abstract structural templates |
| **Model learns** | How to reason about *these types* of problems | How to structure *any* problem-solving response |
| **Flexibility** | Good within the domain of examples | Transfers across domains |
| **Depth** | Deep for specific domain | Broader but potentially shallower |

In practice, the lecture shows them combined: a meta prompt (structural template) + a few-shot example = the model gets both structural guidance AND a concrete worked instance.

---

### When to Use Meta Prompting

- You want **consistent response structure** across diverse inputs
- You're building a system where **output format matters** (APIs, grading, automated parsing)
- You want the model to **approach problems systematically** without being tied to specific examples
- The problem domain is too broad to cover with specific few-shot examples
- You want to encode your **problem-solving methodology** into the prompt itself

---

> **Learning Thought:**  
> Meta prompting makes explicit something that was implicit in CoT — that how you structure your reasoning process is itself a learnable, transferable skill. When you write a meta prompt, you're not teaching the model what to think about a specific problem; you're encoding a *cognitive procedure*. This is closer to how expert knowledge actually works: a great mathematician doesn't have every problem solved in memory — they have a repertoire of powerful problem-solving strategies that they apply to new problems. Meta prompting lets you hand the LLM a strategy, not just an answer.

---

### Interview Questions — Topic 9

**Q1. What is meta prompting and how does it differ from few-shot prompting?**
Meta prompting is structure-oriented — it provides abstract templates that dictate how to format and approach any problem, without specifying content. Few-shot prompting is content-oriented — it provides specific worked examples that the model uses to pattern-match its response. Meta prompting generalises better across domains; few-shot provides deeper guidance within a specific domain.

**Q2. What are the four key characteristics of meta prompting?**
(1) Abstract examples — structural placeholders rather than concrete content; (2) Structure-oriented — focuses on format and pattern rather than specific information; (3) Syntax-focused — uses formal syntax (LaTeX, schemas) as structural anchors; (4) Versatile — applicable across diverse domains because it describes structure, not content.

**Q3. In what scenarios would meta prompting outperform few-shot CoT?**
When the task domain is too broad for task-specific examples, when consistent output formatting across diverse inputs is required, when you want to encode a reusable problem-solving methodology, or when the downstream system needs to parse structured outputs reliably.

**Q4. Can meta prompting and few-shot CoT be combined? What does each contribute?**
Yes. Meta prompting contributes the structural template (how to organise the response). Few-shot CoT contributes a concrete worked example (what that structure looks like in practice). Combined, the model gets both format guidance and a grounded illustration of the expected reasoning style.

**Q5. Why is "versatile" listed as a key property of meta prompting?**
Because the prompt describes structure rather than content, the same meta prompt can apply to algebra, logic, coding, essay writing, or any structured domain. The structural template is domain-agnostic — only the content the model inserts into the template is domain-specific.

---

---

## Topic 10 — Directional Stimulus Prompting (DSP)

### The Core Problem DSP Solves

In standard prompting, the instructions you write directly into the prompt must work for all inputs uniformly. But many tasks are highly **instance-specific** — the guidance that would help the model on *this particular* input is different from what would help on *another* input.

Example: summarisation. One article is about a political scandal involving specific names. Another is about a scientific discovery. The keywords and focus points that should guide the summary differ completely per article.

**DSP insight:** Don't write one static prompt that must work for all inputs. Instead, **train a small model to generate customised hints for each specific input**, then use those hints to guide the large frozen LLM.

**Paper:** Li et al., *Guiding Large Language Models via Directional Stimulus Prompting*, NeurIPS 2023

---

### What is the Directional Stimulus?

The **directional stimulus** is a piece of **instance-specific guidance** — typically a set of keywords, dialogue acts, or hints — that is automatically generated for each input and injected into the prompt to steer the LLM toward the desired output.

Key properties:
- **Instance-specific** — different for every input, not a static system prompt
- **Generated automatically** — by a trained small policy model (T5, GPT-2), not written by a human
- **Acts as a hint** — nudges the LLM without controlling it fully
- **Not retrieved from external sources** — generated purely from the input text itself

---

### The Two-Component Architecture

DSP introduces a **two-model system**:

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   Input Text (article, story, dialogue)                     │
│         ↓                                                   │
│   ┌─────────────────┐                                       │
│   │  Policy LM      │  ← Small, trainable model            │
│   │  (T5 / GPT-2)   │    (learns to generate hints)        │
│   └────────┬────────┘                                       │
│            ↓                                                │
│   Directional Stimulus (keywords, hints)                    │
│            ↓                                                │
│   ┌─────────────────┐                                       │
│   │  Frozen LLM     │  ← Large, fixed model                │
│   │  (GPT-4, etc.)  │    (does the actual task)            │
│   └────────┬────────┘                                       │
│            ↓                                                │
│   Final Output (summary, dialogue, etc.)                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

The policy LM is small and cheap. The frozen LLM is large and expensive. DSP makes the expensive model perform better by having the cheap model generate optimal steering signals for each input.

---

### Stage 1 — Supervised Fine-Tuning of the Policy Model

The policy model is first trained with supervised learning on a dataset of (input → pseudo-stimulus) pairs.

```
Training Data:
  Input:  [Article text]
  Label:  [Keywords that a good summary should cover]

Example:
  Article: "Seoul (CNN) South Korea's Prime Minister Lee Wan-koo offered
            to resign on Monday amid a growing political scandal..."

  Keywords: "Lee Wan-koo; resign; South Korean tycoon; Sung Woan-jong;
             hanging from a tree; investigation; notes; top officials"
```

The policy model learns: given this input, which keywords are most important for the desired output?

After supervised fine-tuning, the policy model can generate reasonable keyword hints. But "reasonable" isn't always optimal — so the next stage refines it further.

---

### Stage 2 — Reinforcement Learning (RL) Fine-Tuning

The policy model is further fine-tuned using **Reinforcement Learning**, where the reward signal comes from evaluating the LLM's final output quality.

```
RL Loop:

  Input Text
      ↓
  Policy LM generates stimulus (keywords)
      ↓
  [Input + Stimulus] → Frozen LLM → Output (summary)
      ↓
  Reward = ROUGE Score (output vs reference summary)
      ↓
  Update Policy LM weights to maximise reward
```

**Why RL?** Supervised fine-tuning teaches the policy model to produce stimuli that *look like* good hints. RL teaches it to produce stimuli that *actually cause the LLM to generate better outputs*. This is a crucial distinction — the optimal stimulus to generate is determined by its downstream effect on the LLM's output, which only RL can capture.

**ROUGE Score** measures overlap between the generated summary and a reference summary — a standard automatic evaluation metric for summarisation tasks.

---

### Concrete Example: Article → Stimulus → Summary

```
Article:
  "Seoul (CNN) South Korea's Prime Minister Lee Wan-koo offered to
   resign on Monday amid a growing political scandal. Lee will stay
   in his official role until South Korean President Park Geun-hye
   accepts his resignation. He has transferred his role of chairing
   Cabinet meetings to the deputy prime minister for the time being..."

Policy LM generates Stimulus:
  Keywords: Lee Wan-koo; resign; South Korean tycoon; Sung Woan-jong;
            hanging from a tree; investigation; notes; top officials

Prompt to Frozen LLM:
  "Summarise the following article. Key points to cover:
   [Lee Wan-koo, resign, Sung Woan-jong, investigation, top officials]
   Article: [full text]"

LLM Output:
  A summary that correctly covers the resignation, the scandal's key
  figures, the investigation, and the political context.
```

Without the stimulus, the LLM might produce a generic summary that misses the most important specific details. The directional stimulus acts like a reporter's checklist — "make sure you cover these specific points."

---

### Why DSP is Different from RAG and Standard Prompting

| Dimension | Standard Prompting | RAG | DSP |
|---|---|---|---|
| **Guidance source** | Static human-written instructions | Retrieved external documents | Dynamically generated instance-specific hints |
| **Adapts per input?** | No | Partially (retrieves relevant docs) | Yes (policy model generates custom hints) |
| **External knowledge?** | No | Yes | No — generated from input only |
| **Trainable component?** | No | Retriever (optional) | Yes — policy LM is explicitly trained |
| **Interaction with LLM** | Direct prompt | Augmented prompt | Hint-augmented prompt |

---

### Key Insight: The Policy Model as a Prompt Optimiser

The trained policy model is essentially an **automatic prompt engineer**. It has learned, through thousands of training examples and RL feedback, what kind of hints lead to the best outputs for a given LLM on a given task. This represents a shift from:

> Human manually writes prompts → (static, labour-intensive)

To:

> Small model automatically generates optimal per-instance prompts → (dynamic, scalable)

---

> **Learning Thought:**  
> DSP reveals a deep insight: the "best prompt" for a task is not one fixed string — it's a function of the specific input. The optimal hints for summarising an article about a political scandal differ from those for a scientific paper. Humans intuitively adjust their attention and emphasis per document, but static prompts can't. DSP solves this by training a lightweight model to function as an automatic, input-aware prompt engineer. The policy model learns a mapping from input space to "optimal steering space" — making it one of the earliest examples of **learned prompting** as opposed to hand-crafted prompting.

---

### Interview Questions — Topic 10

**Q1. What is Directional Stimulus Prompting (DSP) and what problem does it solve?**
DSP introduces instance-specific "hints" (the directional stimulus) generated by a small trained policy model for each input. It solves the limitation of static prompts that apply the same guidance uniformly to all inputs — DSP dynamically tailors the steering signal per input, leading to better LLM performance without modifying the frozen LLM.

**Q2. What is the directional stimulus and how is it generated?**
The directional stimulus is a set of instance-specific hints (e.g., keywords the summary should cover). It is generated by a small policy model (T5, GPT-2) trained first with supervised fine-tuning on (input → hint) pairs, then refined with RL where the reward is the quality of the frozen LLM's final output.

**Q3. Why is RL used to fine-tune the policy model instead of just supervised learning?**
Supervised fine-tuning teaches the policy model to generate hints that *resemble* good hints. But the optimal hint is determined by its downstream effect on the frozen LLM's output — which only RL can capture. RL trains the policy model to maximise actual output quality (measured by ROUGE), not just hint superficial similarity.

**Q4. How is DSP different from RAG?**
RAG retrieves relevant external documents and adds them to the prompt. DSP generates hints purely from the input itself — no external knowledge is retrieved. DSP's guidance is about which aspects of the input to focus on; RAG's is about augmenting the input with additional information.

**Q5. What does it mean that DSP works with a "black-box frozen LLM"?**
The frozen LLM's weights are never modified — DSP doesn't fine-tune it. The policy model only controls what goes into the frozen LLM's prompt. This is crucial because it means DSP can work with any LLM, including API-only models where weight access is impossible.

**Q6. What is the reward signal used in DSP's RL stage, and why is it appropriate?**
ROUGE score — a standard metric that measures n-gram overlap between the generated summary and a reference summary. It's appropriate because it directly measures the quality of the frozen LLM's final output, which is exactly what the policy model is being trained to optimise.

---

---

## Topic 11 — Prompt Chaining

### The Core Problem

A single LLM call — no matter how cleverly prompted — has limits on the complexity of tasks it can reliably handle. When a task has many distinct steps, a single prompt forces the model to:
- Hold too many sub-tasks in mind simultaneously
- Perform too many different operations in one generation
- Produce a long, unverifiable output with no checkpoints

As task complexity grows, a single prompt becomes an unreliable black box. You can't isolate where it went wrong, and you can't intervene at intermediate steps.

---

### What is Prompt Chaining?

**Prompt Chaining** decomposes one complex task into a **sequence of smaller, focused LLM calls**, where the output of each step becomes the input to the next.

```
Input
  ↓
[PROMPT A] → Output A
                ↓
           [PROMPT B] → Output B
                            ↓
                       [PROMPT C] → Final Output
```

Each prompt in the chain has **one job**. It does that job well, and hands off a clean, structured result to the next step.

---

### Why Decompose? — The Single Prompt vs Chained Prompt Comparison

**Single complex prompt:**
```
"Consider the given text in Spanish. Translate it into English.
 Find all the statistics and facts used in this text and list them
 as bullet points. Translate them again into Spanish."
```

This asks the model to simultaneously handle: language translation, fact extraction, list formatting, and back-translation. Any failure in any step corrupts all subsequent steps, and you cannot see where it went wrong.

**Chained prompt — same task broken into 5 steps:**

```
Step 1 → Read the given Spanish text
Step 2 → Translate the text into English
Step 3 → Fetch statistics and facts from the translated text
Step 4 → Create a bullet point list of all these facts
Step 5 → Translate the bullet point list back into Spanish
```

Each step is verifiable. If Step 3 extracts the wrong facts, you catch it before Step 4 and 5 waste resources.

---

### The Workflow Pattern

```
                ┌───────────────────────────────────┐
                │         PROMPT CHAINING           │
                │                                   │
  Initial ─────►│  PROMPT A  │──► OUTPUT A ────────►│  PROMPT B  │──► OUTPUT B
   Task         │            │    (becomes           │            │
                │            │     input to B)       │            │
                └───────────────────────────────────┘
                         │                                  │
                         ▼                                  ▼
                  Intermediate                         Intermediate
                   Result (verifiable)                  Result (verifiable)
                                                            │
                                                            ▼
                                                       PROMPT C ──► FINAL OUTPUT
```

Each intermediate output is a **verifiable checkpoint** — you (or an automated validator) can inspect it before passing it to the next step.

---

### Real-World Example: Customer Review Analysis

```
Customer Reviews (raw)
        ↓
[PROMPT 1: Sentiment Analysis]
"Classify each review as positive, negative, or neutral."
        ↓
Labelled reviews with sentiment tags
        ↓
[PROMPT 2: Key Phrase Extraction]
"From the negative reviews, extract the key phrases describing issues."
        ↓
List of issue phrases
        ↓
[PROMPT 3: Summary Generation]
"Summarise these customer pain points into a product team report."
        ↓
Final Report
```

Doing this in one prompt would mix concerns — the model would have to simultaneously classify, extract, and summarise, increasing error rates and making debugging impossible.

---

### Benefits of Prompt Chaining

| Benefit | Why it matters |
|---|---|
| **Improved accuracy** | Each prompt focuses on one thing — less cognitive load per call |
| **Verifiable intermediate outputs** | You can validate at each step before proceeding |
| **Easier debugging** | When something goes wrong, you know exactly which step failed |
| **Modular and reusable** | Individual steps can be reused in different chains |
| **Conditional branching** | Output of one step can determine *which* prompt is used next |
| **Cost optimisation** | Use a cheap/fast model for simple steps, expensive model for hard steps |

---

### When to Use Prompt Chaining

Use chaining when:
- The task has **clean, sequential sub-tasks** (not circular or parallel)
- Each step's output is **verifiable** (you can check if it's correct before moving on)
- You want **intermediate checkpoints** to inspect or log
- The **latency budget allows** multiple sequential LLM calls
- You're building a **production pipeline** that needs reliability and debuggability

Don't use chaining when:
- Steps are inherently interdependent and can't be cleanly isolated
- Latency is critical (each call adds round-trip time)
- The task is simple enough to handle in one call reliably

---

### Prompt Chaining vs Agents (ReAct)

| Dimension | Prompt Chaining | ReAct (Agentic) |
|---|---|---|
| **Control flow** | Fixed, predetermined sequence | Dynamic — model decides next step |
| **Flexibility** | Low — steps are hardcoded | High — adapts based on observations |
| **Predictability** | High — same path every time | Lower — path varies by input |
| **Debugging** | Easy — fixed pipeline | Harder — dynamic execution path |
| **Best for** | Well-structured, repeatable tasks | Open-ended, exploratory tasks |

Chaining is **deterministic pipelines**. ReAct is **adaptive agents**. Choose based on how well-defined your task structure is.

---

> **Learning Thought:**  
> Prompt chaining is the LLM equivalent of the Unix philosophy: "Do one thing and do it well." Each prompt in the chain is a specialist — it receives clean input and produces clean output. The chain's overall reliability is much higher than a single monolithic prompt because errors are local and inspectable. This is also why prompt chaining naturally fits into software engineering workflows: each step can have tests, logging, and retry logic. Chaining is the bridge between prompting as a craft and prompting as an engineering discipline.

---

### Interview Questions — Topic 11

**Q1. What is prompt chaining and what problem does it solve?**
Prompt chaining decomposes a complex task into a sequence of smaller LLM calls, where each call's output feeds the next. It solves the unreliability of single-prompt approaches for complex tasks by reducing the cognitive load per call, enabling intermediate verification, and making the pipeline debuggable.

**Q2. What are the main advantages of prompt chaining over a single complex prompt?**
Improved accuracy (each step focuses on one task), verifiable intermediate outputs (inspect before proceeding), easier debugging (failures localised to a specific step), modularity (steps reusable), and cost control (use appropriate model per step).

**Q3. When should you NOT use prompt chaining?**
When steps are inherently circular or tightly coupled (can't be cleanly isolated), when latency is critical (each call adds time), or when the task is simple enough to handle reliably in a single call. The overhead of chaining isn't always worth it.

**Q4. How does prompt chaining differ from ReAct?**
Prompt chaining has a fixed, predetermined sequence of steps — the pipeline is hardcoded. ReAct is dynamic — the model decides its next action based on current observations. Chaining is more predictable and debuggable; ReAct is more flexible for open-ended tasks.

**Q5. How can prompt chaining enable cost optimisation in production?**
Because different steps in the chain can use different models. Simple steps (formatting, classification) can use small, cheap, fast models. Complex steps (reasoning, synthesis) can use expensive, capable models. A monolithic approach forces you to use the most powerful model for everything.

**Q6. What is a "verifiable intermediate output" and why is it important in chaining?**
An intermediate output is verifiable when you can programmatically or manually check if it's correct before using it as input to the next step. This is important because errors caught early prevent wasted computation on downstream steps and make the pipeline robust and trustworthy.

---

---

## Topic 12 — Tree of Thoughts (ToT) Prompting

### The Core Problem: Linear Reasoning Has Dead Ends

CoT (and its variants) uses **linear reasoning** — the model generates one thought after another in a single chain. This works well when the path to the answer is mostly forward-progress. It fails badly when:

- The first idea isn't the best approach
- The problem has multiple equally valid starting points
- An early wrong turn is hard to recover from
- The solution requires **lookahead** — evaluating where a path leads before committing

In these cases, a single linear chain gets stuck at dead ends with no recovery mechanism.

**Example:** The ball-in-cup puzzle from the lecture:
> *Bob is in the living room. He walks to the kitchen, carrying a cup. He puts a ball in the cup, turns the cup upside down, then carries the cup to the bedroom. Where is the ball?*

A zero-shot model says "bedroom" — the most naive answer. A CoT model might also say "bedroom" because it follows a single chain of reasoning. But the correct answer ("kitchen floor") requires exploring multiple physical interpretations simultaneously.

**Paper:** Zekun Li et al., *Tree of Thoughts: Deliberate Problem Solving with Large Language Models*, NeurIPS 2023

---

### What is Tree of Thoughts?

**Tree of Thoughts (ToT)** organises the LLM's problem-solving into a **tree structure** where:
- Each **node** is a partial solution or reasoning state ("thought")
- Each **branch** is a next possible step
- The model **explores multiple branches in parallel**, not just one
- The model **evaluates** its own thoughts at each node — deciding which branches are promising
- **Search algorithms** (BFS/DFS) guide which branches to explore

```
                         [Problem]
                             │
            ┌────────────────┼────────────────┐
            │                │                │
       [Thought A]      [Thought B]      [Thought C]
       (Med conf)       (High conf)      (Low conf)
            │                │                │
       [evaluated]      [evaluated]      [pruned ✗]
            │                │
        ┌───┤            ┌───┤
        │   │            │   │
    [A1]  [A2]        [B1]  [B2]
                          │
                     [Selected → Detailed Solution]
```

The model doesn't just generate one chain — it generates, evaluates, selects, and expands multiple chains simultaneously.

---

### The Three-Expert Prompt Pattern

The most accessible way to implement ToT is through a **multi-expert role-playing prompt** — a pattern used directly in the lecture:

```
Imagine three different experts are answering this question.
They will track the ball step-by-step, evaluate the physics of each
action, and branch out to cover different possible interpretations.
Each expert will share their approach, then the group will evaluate
and reach a consensus.
```

**Applied to the ball puzzle:**

```
EXPERT 1 (Physics perspective):
  - Cup turned upside down → gravity acts immediately
  - Ball falls out in the kitchen before being carried
  - Conclusion: Kitchen floor

EXPERT 2 (Language/narrative perspective):
  - The story never says the ball stayed inside
  - "Upside down" implies the ball dropped before moving rooms
  - Conclusion: Kitchen

EXPERT 3 (Skeptical/edge-case perspective):
  - The text doesn't specify cup type or ball size
  - If ball is small and lodged in the rim, or stuck by suction, or cup has a lid
  - It could remain inside while Bob carries it to the bedroom
  - Conclusion: Bedroom (ball still in cup) — if assumptions hold

FINAL CONSENSUS: The ball is in the kitchen.
  (Two experts agree on physics; the skeptic's case depends on
   unstated assumptions that aren't supported by the text.)
```

Zero-shot answer: **"bedroom"** ← Wrong  
ToT consensus: **"kitchen floor"** ← Correct ✅

The correct answer emerges from the **collision of multiple independent reasoning paths**, each catching what others miss.

---

### The Formal ToT Framework

In the full academic framework, ToT has four components:

**1. Thought Decomposition**
Break the problem into a sequence of "thought steps". What constitutes a "thought" depends on the task:
- For creative writing: a sentence or paragraph
- For math: an equation step
- For planning: an action
- For puzzles: a logical deduction

**2. Thought Generation**
At each state, generate multiple candidate next thoughts. Two strategies:
- **Sample**: Generate k thoughts independently (diverse, good for creative tasks)
- **Propose**: Generate all candidates in one structured prompt (efficient, good for constrained tasks)

**3. State Evaluation (Self-Evaluation)**
The model judges each candidate thought. Three approaches:
- **Value** (0–1 score per thought): "How promising is this partial solution?"
- **Vote**: Generate multiple completions and take the majority
- **Verify**: Check if the thought satisfies known constraints

This self-evaluation is what separates ToT from CoT — the model can recognise dead ends and prune them.

**4. Search Algorithm**
How to navigate the tree:

| Algorithm | How it works | Best for |
|---|---|---|
| **BFS (Breadth-First Search)** | Explore all thoughts at depth k before going to depth k+1 | Problems needing global perspective, comparing options at same level |
| **DFS (Depth-First Search)** | Fully explore one branch before backtracking to the next | Problems where solutions have clear early indicators; memory-efficient |
| **Beam Search** | Keep top-B thoughts at each level, prune the rest | Balance between breadth and depth; most common in practice |

---

### The Full ToT Prompt Pattern (from the lecture)

```
Let three different experts offer approaches for the problem.

→ All experts will write down their approaches, then share
  with the group for evaluation.

→ Suggested approach is the one with the highest confidence
  after evaluation.

→ The problem is [user_input]
```

In practice, the prompt structure for each expert:

```
Expert [N]:
  Approach: [reasoning path]
  Confidence: [High / Medium / Low]
  Evaluation: [why this is or isn't promising]

[After all experts respond]

Group Evaluation: Compare approaches.
Selected Approach: [highest-confidence approach]
Detailed Solution: [full solution following the selected approach]
```

---

### ToT vs CoT vs Self-Consistency

Understanding how these relate is important:

| Dimension | CoT | Self-Consistency | ToT |
|---|---|---|---|
| **Reasoning structure** | Linear chain | Multiple linear chains | Branching tree |
| **Paths explored** | 1 | N independent chains | Branching with evaluation at each node |
| **Evaluation** | None | Majority vote at the end | At every intermediate thought node |
| **Backtracking** | No | No | Yes — pruned branches can be revisited |
| **Lookahead** | No | No | Yes — evaluates where a path leads before committing |
| **Best for** | Sequential reasoning | Reducing variance in reasoning | Complex problems with many possible approaches |
| **Cost** | Low | Medium (N× calls) | High (many tokens + evaluations) |

**Key distinction from Self-Consistency:** Self-Consistency generates N independent chains and votes at the end. ToT generates branches, evaluates each intermediate step, prunes dead ends, and only continues promising branches. ToT has **lookahead and backtracking**; Self-Consistency does not.

---

### When to Use Tree of Thoughts

Use ToT when:
- **Planning tasks** — where early decisions constrain later options (game strategy, project planning)
- **Puzzles and riddles** — where the first interpretation is often wrong
- **Creative writing** — where you want to explore structurally different approaches before committing
- **Mathematical optimisation** — where multiple solution strategies exist
- **Multi-step reasoning with dead ends** — when CoT keeps getting stuck at the same wrong answer

Don't use ToT when:
- The task is simple and sequential (use CoT)
- Latency and cost are constrained (ToT requires significantly more tokens)
- The problem has one obvious solution path (overhead isn't justified)

The rule of thumb from the lecture: *"This pattern takes more tokens and time, so use it for important decisions where the quality improvement justifies the cost."*

---

### Key Features Summary

| Feature | Description |
|---|---|
| **Multiple Reasoning Paths** | Explores various lines of thought simultaneously, unlike linear CoT |
| **Self-Evaluation** | The LLM assesses the quality and progress of its own generated thoughts at each node |
| **Search Algorithms** | Uses BFS or DFS to navigate the thought tree, enabling lookahead and backtracking |
| **Deliberate Problem Solving** | Mimics how humans think about hard problems — exploring options, evaluating, and backtracking |

---

> **Learning Thought:**  
> Tree of Thoughts is the closest prompting technique to how a human expert actually thinks about hard problems. We don't commit to the first idea that comes to mind — we hold multiple partial solutions in mind simultaneously, evaluate their promise, abandon the ones that are going nowhere, and deepen the ones that look good. CoT is the internal monologue of a confident solver. ToT is the deliberative process of an expert who knows that confidence at step 1 doesn't guarantee a correct final answer. The cost is real — more tokens, more time — but for the right class of problems, the quality improvement is substantial.

---

> **Learning Thought:**  
> The three-expert prompt pattern (Imagine three experts...) is a brilliant practical instantiation of ToT that requires no infrastructure. Instead of calling the LLM multiple times with a search algorithm, you ask a single model to **simulate** multiple expert perspectives in one generation. Each "expert" is a different reasoning path. The model's self-evaluation appears as confidence ratings and group discussion. This is approximate ToT — cheaper, easier to implement, and surprisingly effective on problems where diversity of perspective matters.

---

### Interview Questions — Topic 12

**Q1. What is Tree of Thoughts and what limitation of CoT does it address?**
ToT organises LLM reasoning into a branching tree where multiple reasoning paths are explored in parallel, and the model evaluates the promise of each intermediate thought before deciding which to continue. It addresses CoT's linear reasoning limitation — CoT cannot backtrack or recover from early wrong turns, while ToT enables lookahead and explicit branch pruning.

**Q2. What are the four components of the formal ToT framework?**
(1) **Thought decomposition** — defining what constitutes a reasoning step for the task; (2) **Thought generation** — generating multiple candidate next steps at each node; (3) **State evaluation** — the model scores or votes on the promise of each thought; (4) **Search algorithm** — BFS, DFS, or beam search to navigate which branches to explore and which to prune.

**Q3. How is ToT different from Self-Consistency?**
Self-Consistency generates N independent reasoning chains and votes on the final answer at the end — no intermediate evaluation, no backtracking. ToT evaluates intermediate thoughts at every node, prunes unpromising branches early, and explores the tree with explicit lookahead. ToT has internal structure and backtracking; Self-Consistency is independent sampling with end aggregation.

**Q4. What search algorithms are used in ToT and when do you choose each?**
BFS explores all thoughts at a given depth before going deeper — good for problems needing global comparison at the same reasoning level. DFS fully explores one branch before backtracking — good for problems with clear early signals. Beam search keeps top-B thoughts per level — a practical middle ground between breadth and depth.

**Q5. What is the "three-expert prompt pattern" and how does it approximate ToT?**
The three-expert pattern asks the model to simulate three independent experts each offering a different reasoning approach, then evaluate and select the best. It approximates ToT in a single LLM call — each expert represents a different branch of the thought tree, and the group evaluation implements the state evaluation component. It's cheaper and simpler than formal multi-step ToT.

**Q6. For what types of tasks is ToT most appropriate?**
Planning tasks (early decisions constrain later ones), puzzles and riddles (first interpretation is often wrong), creative writing (structural alternatives worth exploring), mathematical optimisation (multiple solution strategies), and any multi-step reasoning where CoT consistently hits dead ends.

**Q7. What is the cost trade-off of ToT compared to CoT and Self-Consistency?**
ToT requires significantly more tokens and inference calls than CoT (many branches × evaluation steps). Self-Consistency is middle ground (N chains, no intermediate evaluation). For simple tasks, CoT's cost is justified by its accuracy; for complex problems where quality improvement matters, ToT's higher cost may be worthwhile. The rule: use ToT for important decisions where quality justifies cost.

**Q8. How does ToT mimic human expert reasoning?**
Experts don't commit to the first idea — they consider multiple hypotheses, evaluate which are most promising, abandon dead ends, and deepen productive lines of inquiry. ToT encodes this exact behaviour: generate alternatives → evaluate promise → prune → deepen. CoT is the internal monologue of someone who already knows the answer; ToT is the deliberative process of solving genuinely hard problems.

---

---

## Putting Section 4 Together: Which Technique When?

| Scenario | Recommended Technique |
|---|---|
| Need consistent output structure across diverse inputs | **Meta Prompting** |
| Need task-specific hints generated per input automatically | **DSP** |
| Task has distinct sequential steps, each verifiable | **Prompt Chaining** |
| Problem has many possible approaches, first idea often wrong | **Tree of Thoughts** |
| Need explainability + control over every step | **Prompt Chaining** |
| Hard puzzle or planning task requiring lookahead | **Tree of Thoughts** |
| Working with a frozen API LLM, need better task guidance | **DSP** |
| Building an automated, scalable reasoning pipeline | **Meta Prompting + Prompt Chaining** |

---

## Quick Reference Cheat Sheet — Section 4

| Technique | Core Idea | Key Mechanism | When to Use |
|---|---|---|---|
| **Meta Prompting** | Prompt the model with structural templates, not content examples | Abstract placeholders + syntax anchors | Consistent output structure across diverse domains |
| **DSP** | Train a small policy LM to generate instance-specific hints for a frozen LLM | Policy LM (SFT → RL) generates directional stimulus | When static prompts are too generic for varied inputs |
| **Prompt Chaining** | Break complex task into sequential focused LLM calls | Output A → input to next call | Complex pipelines with verifiable intermediate steps |
| **Tree of Thoughts** | Explore multiple reasoning paths in parallel with self-evaluation and pruning | Branch → Evaluate → Prune → Deepen | Hard problems with dead ends, planning, puzzles |

---

## Key Papers — Section 4

| Paper | Year | Contribution |
|---|---|---|
| Yifan Zhang et al., *Meta Prompting for AI Systems* | 2023 | Introduced structure-oriented prompting with abstract templates |
| Li et al., *Guiding LLMs via Directional Stimulus Prompting* | NeurIPS 2023 | Trained policy LM to generate per-instance hints for frozen LLMs |
| Zekun Li et al., *Tree of Thoughts: Deliberate Problem Solving with LLMs* | NeurIPS 2023 | Formalised tree-structured multi-path reasoning with search algorithms |

---

*Next: Section 5 — Limitations & Safety (Hallucination, Prompt Injection & Guardrails)*
