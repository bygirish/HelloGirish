# Lec 7 — Section 2: Core Prompting Techniques
**Course:** Advanced Prompt Engineering | IIT Kharagpur  
**Instructor:** Prof. Koustav Rudra  
**Module:** 2, Session 1

---

## Topics Covered
5. Zero-Shot Prompting
6. Few-Shot Prompting
7. Chain-of-Thought (CoT) Prompting
   - 7a. Zero-Shot CoT
   - 7b. Few-Shot CoT
   - 7c. Automatic CoT
   - 7d. Self-Consistency

---

## The Prompting Ladder

Before diving in, understand the hierarchy these techniques form. Each level addresses a limitation of the one below it:

```
Self-Consistency          ← most reliable for reasoning
        ↑
Auto-CoT / Few-Shot CoT   ← structured reasoning with examples
        ↑
Zero-Shot CoT             ← reasoning without examples
        ↑
Few-Shot Prompting        ← learning from examples
        ↑
Zero-Shot Prompting       ← raw instructions, no examples (baseline)
```

Each technique is a response to a failure mode of the previous one. Learning them in this order makes the motivation for each crystal clear.

---

---

## Topic 5 — Zero-Shot Prompting

### What is Zero-Shot Prompting?

**Zero-shot prompting** means giving the model a task instruction **with no examples** — you simply tell it what to do and expect it to figure out the rest from its pre-trained knowledge.

"Zero-shot" refers to zero training examples provided in the prompt.

### Basic Example

```
User: Classify the text into neutral, negative, or positive.
      Just give me the classification.
      Text: "I think the vacation is okay."
      Sentiment:

Mistral: Neutral
```

The model has never been shown what "Neutral" looks like in *this exact format* — it generalizes from its training data.

### How Does It Work?

The model relies entirely on:
1. **Pre-trained knowledge** — learned patterns, facts, and task formats from billions of tokens of training data
2. **Instruction following** — the model has been fine-tuned (via RLHF or instruction tuning) to follow natural language directives
3. **Generalization** — mapping your novel instruction to a task pattern it has seen before during training

### When Zero-Shot Works Well

- Simple, well-defined tasks (classification, translation, formatting)
- Tasks the model has clearly seen extensively in training (sentiment analysis, summarization)
- When you need a quick prototype or baseline result
- Tasks with universal, common-sense answers

### When Zero-Shot Fails

Zero-shot prompting struggles when:

| Failure Mode | Example | Why it fails |
|---|---|---|
| **Complex reasoning** | Multi-step math problems | No intermediate steps → model guesses |
| **Implicit knowledge needed** | Domain-specific terminology | Model may not have enough context |
| **Ambiguous instructions** | "Make this better" | Model doesn't know what "better" means to you |
| **Output format control** | Strict JSON output | No example → inconsistent formatting |

**Concrete failure example from the lecture:**

```
User: Solve the math problem. What is 3/7 of 84?

LLM: To find 3/7 of 84: multiply the numerators (3 × 84 = 252),
     then multiply the denominators (7 × 84 = 588),
     then divide: 252 / 588 = 0.4286

→ WRONG. Correct answer is 36.
```

The model confused the procedure for multiplying fractions with computing a fraction of a number.

### Model Size Matters for Zero-Shot

A critical insight: zero-shot capability scales with model size.

- **Large models** (e.g., Mistral 123B, GPT-4) → strong zero-shot performance
- **Small models** (e.g., Mistral 7B) → weak zero-shot, needs examples or CoT

This is because zero-shot relies entirely on emergent instruction-following ability, which only appears at sufficient scale.

### The Fix: Few-Shot Prompting

When zero-shot fails, the solution is to provide examples. This is exactly what Few-Shot Prompting does.

---

> **Learning Thought:**  
> Zero-shot prompting is the "default mode" of using an LLM. Its power comes from the enormous breadth of patterns baked into the model during pre-training — but its limitation is that it has no knowledge of *your specific task context*. Think of it like asking a highly educated stranger to help you: they're capable, but they don't know your preferences, your format requirements, or the edge cases in your domain. Examples (few-shot) are how you bridge that gap.

---

### Interview Questions — Topic 5

**Q1. What is zero-shot prompting and why is it called "zero-shot"?**  
Zero-shot prompting means providing a task instruction to an LLM with no examples. "Zero-shot" refers to zero demonstration examples in the prompt. The model relies entirely on knowledge from pre-training and instruction fine-tuning to generalize to the task.

**Q2. What are the key limitations of zero-shot prompting?**  
It struggles with complex reasoning tasks (multi-step math, logic), tasks requiring specific output formats, domain-specific queries, and any task where the model needs to understand subtle context that isn't universally encoded in its training data.

**Q3. Why does zero-shot performance improve with model size?**  
Zero-shot relies on emergent instruction-following capabilities that arise from scale. Larger models have seen more patterns during pre-training and have been more extensively fine-tuned, giving them better ability to generalize to unseen task descriptions without examples.

**Q4. In what scenarios would you still choose zero-shot over few-shot?**  
When the task is simple and well-defined (translation, basic classification), when context window space is limited, when you want a fast baseline, or when you have no labeled examples available.

**Q5. What does "instruction following" mean in the context of LLMs?**  
It refers to the model's ability to understand and execute natural language instructions without additional training. This capability is developed through instruction fine-tuning (supervised fine-tuning on instruction-output pairs) and RLHF (reinforcement learning from human feedback).

---

---

## Topic 6 — Few-Shot Prompting

### What is Few-Shot Prompting?

**Few-shot prompting** provides the model with **a small number of input-output examples** (called "demonstrations" or "shots") directly in the prompt before asking the actual question.

These examples serve as an **in-context teaching signal** — not modifying model weights, but shaping the model's interpretation of what it should produce.

### Basic Example

```
User: "This is awesome!" // Positive
      "This is bad!"     // Negative
      "Wow, that movie was rad!" // Positive
      "What a horrible show!" //

Mistral: Negative
```

The model observes the pattern from 3 examples and applies it to the 4th.

### What Examples Teach the Model

Few-shot examples communicate three things simultaneously:

| What it teaches | How |
|----------------|-----|
| **Intent** | The model sees what kind of task this is |
| **Output format** | The model mirrors the style, structure, and label format of examples |
| **Edge cases** | Examples can show how to handle ambiguous or tricky inputs |

This is why few-shot consistently outperforms zero-shot on format-sensitive tasks — it eliminates ambiguity about output structure.

### How Few-Shot Prompting Works Mechanically

From the autoregressive perspective, the examples are just tokens in the context window. The model learns to "continue the pattern" — it sees several `(input, output)` pairs and predicts what output should follow the final input.

```
[Example 1 input] → [Example 1 output]
[Example 2 input] → [Example 2 output]
[Example 3 input] → [Example 3 output]
[New input]       → [Model predicts output here]
```

This is fundamentally **in-context learning (ICL)** — learning from context without gradient updates.

### Prompt Structure Best Practices

1. **Consistency** — all examples should follow the exact same format
2. **Quality > Quantity** — 3-5 high-quality examples usually beat 10+ mediocre ones
3. **Coverage** — examples should cover the range of inputs the model will see
4. **Label balance** — for classification, include examples of all classes
5. **Order matters** — recent examples have slightly more influence; put the most representative ones last

### When Few-Shot Still Fails

Few-shot prompting dramatically improves performance, but it still fails on tasks requiring **complex multi-step reasoning**. Example from the lecture:

```
User: The odd numbers in this group add up to an even number:
      4, 8, 9, 15, 12, 2, 1. A: False
      17, 10, 19, 4, 8, 12, 24. A: True
      16, 11, 14, 4, 8, 13, 24. A: True
      17, 9, 10, 12, 13, 4, 2. A: False
      15, 32, 5, 13, 82, 7, 1. A: ?

Model: True   ← WRONG (correct is False: 15+5+13+7+1=41, which is odd)
```

Why does this fail? The examples show *what* to answer but not *how* to reason. The model is pattern-matching labels without actually doing the arithmetic. This is the motivation for Chain-of-Thought.

### One-Shot vs Few-Shot vs Many-Shot

| Variant | Shots | Notes |
|---------|-------|-------|
| **Zero-shot** | 0 | No examples, pure instruction |
| **One-shot** | 1 | Single example; reduces format ambiguity |
| **Few-shot** | 2–10 | Most common; strong performance improvement |
| **Many-shot** | 10–100+ | Possible with large context windows; research area |

### Context Window Considerations

Each example consumes tokens. For large tasks or long examples, you can run out of context window space quickly. This creates a trade-off:

- **More examples** → better in-context learning → fewer tokens left for the actual input
- **Fewer examples** → more space for the real input → less in-context signal

In production, this trade-off is managed by selecting representative examples dynamically (e.g., using embedding similarity to retrieve the most relevant examples for each query).

---

> **Learning Thought:**  
> Few-shot prompting reveals something profound about LLMs: they are not just retrieval systems — they are *in-context learners*. The examples don't modify the model's weights at all. Yet the model's behavior changes dramatically. This is called In-Context Learning (ICL), and it's one of the most surprising emergent properties of large transformers. The model is essentially "learning" a new task mapping from the context window alone, at inference time, with no backpropagation.

---

### Interview Questions — Topic 6

**Q1. What is few-shot prompting and how does it differ from zero-shot?**  
Few-shot prompting includes a small number of labeled input-output examples in the prompt before the actual query. Unlike zero-shot, which relies entirely on pre-trained knowledge, few-shot provides the model with task-specific demonstrations that shape its understanding of the desired output format, style, and behavior.

**Q2. What is In-Context Learning (ICL)?**  
ICL is the phenomenon where LLMs adapt their behavior based on examples provided in the context window — without any change to model weights. Few-shot prompting is the most common form of ICL. The model "learns" the task mapping purely from the input tokens, at inference time.

**Q3. Why does few-shot outperform zero-shot on format-sensitive tasks?**  
Because few-shot examples directly demonstrate the expected output format, eliminating ambiguity. Zero-shot must infer format from the instruction alone, which is often underspecified. Examples provide concrete anchors for format, style, and label vocabulary.

**Q4. What are best practices for selecting few-shot examples?**  
Use high-quality, representative examples. Maintain consistent formatting across all examples. Cover the range of input types the model will encounter. Balance class representation for classification tasks. Keep examples concise to minimize token usage.

**Q5. Why does few-shot prompting still fail on complex reasoning tasks?**  
Because it teaches the model *what* to output, not *how* to reason. On tasks requiring multi-step logic (arithmetic, logical deduction), the model can pattern-match the label format from examples without actually performing the underlying reasoning. Chain-of-Thought prompting addresses this.

**Q6. What is the trade-off between adding more few-shot examples and context window size?**  
More examples improve task understanding but consume more of the context window, leaving less space for the actual input and output. In production, this is managed by dynamically selecting the most relevant examples per query using embedding similarity, rather than using a fixed set.

**Q7. Does the order of few-shot examples matter?**  
Yes. More recent examples (closer to the query) tend to have slightly stronger influence on the output. Additionally, the last example especially influences format. Best practice is to put the most representative or challenging examples last.

---

---

## Topic 7 — Chain-of-Thought (CoT) Prompting

### The Core Problem CoT Solves

Both zero-shot and few-shot prompting treat LLM generation as a **direct input → output** mapping. For simple tasks, this works. For complex tasks requiring multi-step reasoning, it breaks down because the model must compress all reasoning into a single token prediction step.

**CoT insight:** Don't force the model to jump to the answer. Let it *think out loud* first.

### What is Chain-of-Thought Prompting?

**Chain-of-Thought (CoT) prompting** guides the LLM to solve complex tasks by generating **intermediate reasoning steps** before producing the final answer.

Instead of:
```
Q: [complex problem]
A: [direct answer]
```

CoT does:
```
Q: [complex problem]. Let's think step by step.
A: Step 1: [reasoning]
   Step 2: [reasoning]
   Step 3: [reasoning]
   Final Answer: [correct answer]
```

### Why CoT Works

1. **Forces decomposition** — the model breaks the problem into sub-problems it can handle one at a time
2. **Leverages sequential token generation** — each reasoning step conditions the next, building a logical chain
3. **Reduces the cognitive load per step** — instead of computing "what's the final answer to this complex problem?", the model computes "what's the next logical step given what I've reasoned so far?"
4. **Creates a scratchpad** — the reasoning tokens act as working memory for the model

### Concrete Example from the Lecture

**Without CoT (Few-Shot only):**
```
User: [4 examples of odd-sum problems with only A: True/False]
      15, 32, 5, 13, 82, 7, 1. A:

Model: True   ← WRONG
```

**With CoT (Few-Shot + reasoning):**
```
User: 4, 8, 9, 15, 12, 2, 1.
      A: Adding all the odd numbers (9, 15, 1) gives 25. The answer is False.

      17, 10, 19, 4, 8, 12, 24.
      A: Adding all the odd numbers (17, 19) gives 36. The answer is True.

      [2 more examples with full reasoning]

      15, 32, 5, 13, 82, 7, 1. A:

Mistral: The odd numbers are: 15, 5, 13, 7, 1.
         Adding: 15+5+13+7+1 = 41.
         41 is odd. The answer is False.   ← CORRECT ✅
```

The model now gets it right because it's performing the actual computation, not pattern-matching labels.

### Key Properties of CoT

- **Best with large models** — CoT prompting requires sufficient model capacity. Small models don't benefit as much because they lack the reasoning ability to produce valid chains.
- **Especially powerful with few-shot** — combining reasoning examples with CoT gives the model both format guidance and reasoning structure.
- **Emergent at scale** — CoT benefits appear primarily in models with ~100B+ parameters; smaller models show less improvement.

---

> **Learning Thought:**  
> Chain-of-Thought is arguably the most important prompting technique discovered so far. It reframes what an LLM is doing: instead of "answer this question," you're asking it to "narrate your problem-solving process." This works because the transformer's sequential generation mechanism is actually quite good at predicting the *next logical step* — it just can't compress an entire reasoning chain into a single prediction. CoT exploits the model's strength (step-by-step continuation) to overcome its weakness (long-range reasoning compression).

---

---

## Topic 7a — Zero-Shot CoT

### What is Zero-Shot CoT?

**Zero-Shot CoT** eliminates the need for manually crafted reasoning examples. Instead of providing worked-out examples, you simply append a trigger phrase to the prompt:

> **"Let's think step by step."**

This single phrase causes the model to generate its own reasoning chain before giving the answer.

### Example

**Zero-Shot (fails):**
```
Q: A juggler can juggle 16 balls. Half of the balls are golf balls,
   and half of the golf balls are blue. How many blue golf balls are there?
A: The answer is 8.   ← WRONG
```

**Zero-Shot CoT (correct):**
```
Q: A juggler can juggle 16 balls. Half of the balls are golf balls,
   and half of the golf balls are blue. How many blue golf balls are there?
A: Let's think step by step.

Output: There are 16 balls total.
        Half are golf balls → 16/2 = 8 golf balls.
        Half of the golf balls are blue → 8/2 = 4 blue golf balls.
        The answer is 4.   ← CORRECT ✅
```

### Why "Let's think step by step" Works

This phrase acts as a **cognitive anchor**. Because the model has seen this exact phrase (or variants) preceding systematic reasoning in its training data (tutorials, textbooks, walkthroughs), it has learned to associate this phrase with "generate an intermediate reasoning chain." It triggers a mode switch in the model's generation behavior.

### Other Trigger Phrases That Work

```
"Let's think step by step."
"Let's work through this carefully."
"First, let's understand the problem."
"Let me reason through this."
"Think about this step by step."
```

### Key Properties

- **Task-agnostic** — works across math, logic, commonsense reasoning, and more
- **No examples required** — great when you have no labeled demonstrations
- **Free** — only costs a few tokens
- **Consistent** — the same phrase reliably triggers the reasoning mode

### Limitations

- Less reliable than Few-Shot CoT because the model has no example of what good reasoning looks like for your specific task
- The reasoning chain can sometimes be flawed — the model "talks" itself into a wrong answer

---

> **Learning Thought:**  
> Zero-Shot CoT is a remarkable finding. It means that the reasoning capability was already present in the model from pre-training — it just needed a trigger phrase to activate it. The phrase "Let's think step by step" is not magic; it's a statistical pattern that correlates with high-quality sequential reasoning in the training corpus. This insight changed how the field thinks about prompting: sometimes you don't need to teach the model anything new, you just need to know the right words to unlock what's already there.

---

---

## Topic 7b — Few-Shot CoT

### What is Few-Shot CoT?

**Few-Shot CoT** combines the best of both worlds: you provide the model with **worked examples that include full reasoning chains**, not just input-output pairs.

### Comparison: Few-Shot vs Few-Shot CoT

**Standard Few-Shot:**
```
Q: Roger has 5 tennis balls. He buys 2 more cans (3 balls each). How many total?
A: The answer is 11.

Q: Cafeteria had 23 apples. Used 20, bought 6 more. How many?
A: The answer is 27.   ← WRONG (correct is 9)
```

**Few-Shot CoT:**
```
Q: Roger has 5 tennis balls. He buys 2 more cans (3 balls each). How many total?
A: Roger started with 5 balls. 2 cans × 3 balls = 6 balls.
   5 + 6 = 11. The answer is 11.

Q: Cafeteria had 23 apples. Used 20, bought 6 more. How many?
A: Started with 23. Used 20 → 23 - 20 = 3 left.
   Bought 6 more → 3 + 6 = 9. The answer is 9.   ← CORRECT ✅
```

### Why Few-Shot CoT Outperforms Both Alternatives

| Method | What model learns | Reasoning quality |
|--------|------------------|-------------------|
| Few-Shot | Output format and label | None — pattern matching |
| Zero-Shot CoT | To reason (but no example of how) | Self-generated, unguided |
| **Few-Shot CoT** | **Output format + how to reason step by step** | **Guided by demonstrated chains** |

Few-Shot CoT provides a **template for the reasoning process itself**, not just the answer.

### Designing Good CoT Examples

A high-quality CoT example should:
1. Show the **full reasoning process**, not just key steps
2. Make **intermediate computations explicit**
3. Use **clear, natural language** that the model can mimic
4. Handle **edge cases** relevant to your domain
5. Match the **complexity level** of real queries

---

> **Learning Thought:**  
> Few-Shot CoT is the gold standard of prompting for reasoning tasks. The key realization is this: the model doesn't just learn *what* the final answer format looks like — it learns *how* to think. Each reasoning chain in your examples is a behavioral template. When you write those examples, you're essentially writing an implicit algorithm that the model will instantiate for new inputs. This is prompt engineering at its most powerful.

---

---

## Topic 7c — Automatic CoT (Auto-CoT)

### The Problem with Manual CoT

Creating high-quality few-shot CoT examples is **expensive and time-consuming**:
- You need domain expertise to write correct reasoning chains
- You need many examples to cover diverse problem types
- The examples can have subtle errors that mislead the model
- Re-creating examples for every new dataset doesn't scale

### What is Automatic CoT?

**Auto-CoT** automates the creation of reasoning chains, eliminating the need for manually written examples. It's a two-stage process.

**Paper:** Zhang et al., *Automatic Chain of Thought Prompting in Large Language Models*, arXiv:2210.03493

### Stage 1 — Question Clustering

Questions from a dataset are **grouped into clusters** based on semantic similarity (using sentence embeddings + k-means clustering).

```
Dataset of 100 questions
         ↓
 Clustering (k clusters)
         ↓
 Cluster 1: [math word problems]
 Cluster 2: [logical reasoning]
 Cluster 3: [counting problems]
 ...
 Cluster k: [comparison problems]
```

**Why cluster?** To ensure the selected examples are **diverse** — covering different problem types, not just variations of the same problem.

### Stage 2 — Demonstration Sampling

From each cluster, **one representative question is selected**. Then **Zero-Shot CoT** (with "Let's think step by step") is applied to that question to **auto-generate a reasoning chain**.

```
For each cluster:
  1. Select representative question Q_i
  2. Run: Q_i + "Let's think step by step" → LLM generates chain C_i
  3. Apply selection heuristics to filter low-quality chains
     (e.g., chains that are too long, contain repetitive steps, etc.)
  4. Keep (Q_i, C_i) as a demonstration
```

### The Final Prompt

The auto-generated demonstrations are assembled into a few-shot CoT prompt:

```
[Auto-generated Demo 1: Q + full reasoning chain]
[Auto-generated Demo 2: Q + full reasoning chain]
...
[Auto-generated Demo k: Q + full reasoning chain]
[Test Question]
→ LLM generates reasoning + answer
```

### Performance

Auto-CoT achieves **comparable or superior performance** to manually crafted CoT on several reasoning benchmarks. This is significant because:
- No human annotation required
- Scales to new datasets automatically
- The diversity from clustering prevents over-fitting to one problem type

### Selection Heuristics for Chain Quality

Not all auto-generated chains are good. Chains are filtered out if:
- They are excessively long (likely rambling)
- They contain repetitive or circular reasoning
- The reasoning doesn't logically connect steps

---

> **Learning Thought:**  
> Auto-CoT closes the loop on making CoT practical at scale. The insight is elegant: if Zero-Shot CoT can generate reasoning chains (even imperfect ones), and if you ensure diversity through clustering, then you can build a few-shot CoT prompt without any human labeling. This is a microcosm of a bigger trend in AI: using LLMs to generate their own training/prompting data. It's the beginning of self-improving prompt pipelines.

---

---

## Topic 7d — Self-Consistency

### The Problem with Single-Chain CoT

Standard CoT (greedy decoding) generates **one reasoning path** and commits to it. But for complex problems, there are **multiple valid reasoning paths** — and the first path isn't always correct.

If the model takes one wrong turn early in the reasoning chain, the entire chain fails. There's no error correction.

### What is Self-Consistency?

**Self-Consistency** improves CoT by sampling **multiple diverse reasoning paths** for the same question, then selecting the answer that appears most frequently across all paths — the **majority vote**.

**Paper:** Wang et al., *Self-Consistency Improves Chain of Thought Reasoning in Language Models*, arXiv:2203.11171

### How It Works

```
Step 1: Set temperature > 0 (enable stochastic sampling)
Step 2: Run CoT prompt N times → get N different reasoning chains
Step 3: Extract the final answer from each chain
Step 4: Return the answer that appears in the majority of chains
```

### Concrete Example from the Lecture

**Question:** Janet's ducks lay 16 eggs/day. She eats 3 for breakfast, bakes muffins with 4. Sells the rest at $2/egg. How much does she make daily?

**Three independently sampled reasoning chains:**

```
Chain 1: She uses 3+4=7 eggs. Sells 16-7=9 at $2. → $18
Chain 2: She has 16-3-4=9 eggs left. 9×$2 = $18. → $18
Chain 3: She sells (16-4-3)×$2 = 9×$2 = $18. → $18
```

All three chains give $18, so the self-consistent answer is **$18** (correct ✅).

Even if one chain had an arithmetic error and gave $14, the majority vote would still yield $18.

### Why Self-Consistency Works

The key intuition: **a complex problem has many roads to the correct answer, but usually only a few roads to any given incorrect answer**.

- Correct answer: many diverse reasoning paths converge on it
- Wrong answer: only specific flawed reasoning paths lead to it

By aggregating across many paths, you amplify the signal (correct answer) and average out the noise (wrong reasoning chains).

### Visual Diagram

```
                    ┌─ Reasoning Path 1 → Answer: $18 ┐
                    │                                  │
Prompt + Question ──┼─ Reasoning Path 2 → Answer: $18 ┼──→ Majority Vote → $18 ✅
  (Temperature > 0) │                                  │
                    └─ Reasoning Path 3 → Answer: $18 ┘
```

vs. standard CoT (greedy):
```
Prompt + Question → [single reasoning path] → Answer (may be wrong)
```

### Cost vs. Benefit Trade-off

| Dimension | Standard CoT | Self-Consistency |
|-----------|-------------|-----------------|
| **Accuracy** | Good | Significantly better |
| **Cost** | 1× inference call | N× inference calls |
| **Latency** | Low | High (N parallel calls) |
| **Best for** | Simple reasoning | High-stakes, complex reasoning |

In practice, N=5 to N=20 paths is typical. Can be parallelized to reduce latency.

### When to Use Self-Consistency

- High-stakes tasks where accuracy matters more than cost (medical, legal, financial)
- Questions with known ambiguity in reasoning paths
- Evaluation benchmarks (research setting)
- When a single CoT run gives inconsistent results across different runs

---

> **Learning Thought:**  
> Self-Consistency is a brilliant application of a simple statistical principle: if you're uncertain, sample more and aggregate. It treats LLM inference not as a deterministic function but as a **stochastic process** — and uses ensemble logic to improve reliability. The deeper lesson is that a model's first answer is a sample from a distribution, not ground truth. Taking multiple samples and voting is how you get closer to the true signal. This is the LLM equivalent of asking 10 experts and going with the consensus.

---

---

## Putting It All Together: Which Technique When?

| Scenario | Recommended Technique |
|----------|----------------------|
| Simple task, no examples needed | Zero-Shot |
| Need specific format or style | Few-Shot |
| Multi-step reasoning, have examples | Few-Shot CoT |
| Multi-step reasoning, no examples | Zero-Shot CoT |
| Large dataset, don't want to write examples | Auto-CoT |
| High-stakes reasoning, accuracy critical | Self-Consistency |

### The Decision Flow

```
Is the task simple?
├── Yes → Zero-Shot
└── No (complex reasoning)
    ├── Do you have labeled examples?
    │   ├── Yes → Few-Shot CoT
    │   └── No
    │       ├── Large dataset available? → Auto-CoT
    │       └── No → Zero-Shot CoT
    └── Is accuracy critical?
        └── Yes → Wrap any of the above in Self-Consistency
```

---

## Quick Reference Cheat Sheet — Section 2

| Technique | Core Idea | Key Phrase / Mechanism | When to Use |
|-----------|-----------|----------------------|-------------|
| **Zero-Shot** | Instruction only, no examples | Direct task description | Simple, well-defined tasks |
| **Few-Shot** | Input-output examples in prompt | 3-5 (input → output) pairs | Format-sensitive, common tasks |
| **CoT** | Step-by-step reasoning in output | Intermediate reasoning steps shown | Complex multi-step reasoning |
| **Zero-Shot CoT** | Trigger phrase unlocks reasoning | "Let's think step by step" | Reasoning with no examples |
| **Few-Shot CoT** | Worked examples + reasoning chains | Full (input → reasoning → output) | Best accuracy with examples |
| **Auto-CoT** | Auto-generate CoT examples via clustering + Zero-Shot CoT | k-means clustering + Zero-Shot CoT | Scale CoT without manual labeling |
| **Self-Consistency** | Multiple reasoning paths + majority vote | Temperature > 0, N samples, aggregate | High-stakes accuracy |

---

## Key Papers — Section 2

| Paper | Year | Contribution |
|-------|------|-------------|
| Brown et al. (GPT-3) | 2020 | Demonstrated few-shot in-context learning at scale |
| Wei et al. | 2022 | Introduced Chain-of-Thought prompting |
| Kojima et al. | 2022 | Zero-Shot CoT ("Let's think step by step") |
| Zhang et al. | 2022 | Automatic CoT (Auto-CoT) |
| Wang et al. | 2022 | Self-Consistency for CoT |

---

*Next: Section 3 — Acting & Tool Use (ReAct Prompting)*
