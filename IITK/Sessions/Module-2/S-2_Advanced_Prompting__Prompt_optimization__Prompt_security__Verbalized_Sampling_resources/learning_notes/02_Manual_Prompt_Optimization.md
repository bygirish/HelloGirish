# Topic 2 — Manual Prompt Optimization

> **Session:** 2.2 — Prompt Optimization and Security | IIT Kharagpur × upGrad  
> **Instructor:** Koustav Rudra, Assistant Professor AI, IIT Kharagpur

---

## Overview

Manual Prompt Optimization is the human-driven craft of improving prompts through deliberate, structured techniques. Unlike automated methods (Topic 3), manual optimization relies on the practitioner's understanding of the task, the model's behavior, and language.

There are **two tiers** of manual techniques:

| Tier | Techniques |
|---|---|
| **Basic** | Instruction Refinement, Iterative Prompt Refinement, Few-shot Prompting |
| **Advanced** | Chain-of-Thought (CoT) Prompting, Prompt Chaining, Role Playing |

---

## 2.1 Instruction Refinement

### What it is

Instruction Refinement is the process of improving a prompt's **clarity, specificity, and constraints** so the model understands exactly what is expected.

The key idea: **vague instructions produce vague outputs**.

### The Principle

Every instruction should answer:
1. **What** should the model produce?
2. **How** should it produce it (format, length, style)?
3. **What to avoid** (hallucination, off-topic content, certain phrases)?

### Before & After Examples

**Example 1 — Summarization:**
```
❌ Vague:    "Summarize this article"
✅ Refined:  "Summarize this article in 5 bullet points under 120 words.
              Include key risks and conclusions. Avoid speculation."
```

**Example 2 — Medical Query:**
```
❌ Vague:    "Tell me about hepatomegaly"
✅ Refined:  "What are the three main causes of hepatomegaly and their
              specific impacts on the digestive system?"
```

**Example 3 — Customer Support:**
```
❌ Vague:    "Help the customer"
✅ Refined:  "You are a customer support agent for ShopNow.
              If the customer hasn't provided an order number, ask for it first.
              Explain refund eligibility clearly.
              Do not speculate about shipping timelines.
              Keep responses under 80 words."
```

### Refinement Checklist

- [ ] Is the task described precisely (what to do, not just what topic)?
- [ ] Is the expected output format specified (list, paragraph, JSON, etc.)?
- [ ] Is the length constrained?
- [ ] Are hallucination-prone areas explicitly told to avoid speculation?
- [ ] Are edge cases handled in the instructions?

---

## 2.2 Iterative Prompt Refinement

### What it is

Iterative Prompt Refinement is the process of rephrasing a prompt and progressively **adding more details** to guide the model toward the desired behavior. It is fundamentally a feedback loop:

```
Write initial prompt
      ↓
Run model, observe output
      ↓
Identify specific failure (too vague? wrong format? missing info?)
      ↓
Make ONE targeted change addressing that failure
      ↓
Re-run and compare
      ↓
Repeat until quality threshold is met
```

### Example

**Round 1:**
```
Prompt:  "How can I improve my public speaking?"
Output:  "Practice regularly, know your audience, use gestures..."
Problem: Generic. Not specific. No structure. Not tailored to context.
```

**Round 2 (more specific + constraints):**
```
Prompt:  "What are three advanced techniques to overcome nervousness during
          public speaking? Include one psychological tip, one physical exercise,
          and one preparation strategy."
Output:  Much more structured and actionable.
```

### Key Rule: One Change at a Time

Each iteration should change **one aspect** of the prompt. If you change multiple things simultaneously, you cannot attribute the improvement (or regression) to any specific change. This is the same principle as controlled experiments in science.

### What to Iterate On

- **Specificity:** Make vague terms more precise
- **Format specification:** Add explicit output format requirements
- **Length constraints:** Set word/token limits
- **Exclusion rules:** Tell the model what NOT to do
- **Context addition:** Provide background the model needs
- **Example quality:** Replace weak examples with better ones

---

## 2.3 Few-Shot Prompting

### What it is

Few-shot prompting provides **demonstrations** (input-output pairs) inside the prompt to show the model exactly what quality and format of output is expected, rather than just describing it in words.

The philosophy: **showing is more precise than telling**.

### Zero-Shot vs. Few-Shot

```
# Zero-Shot
Classify the sentiment of this review: "The product broke after one day."
Answer:

# Few-Shot (3 examples given)
Classify the sentiment of these reviews:

Review: "Amazing product, works perfectly!"
Sentiment: Positive

Review: "Terrible. Stopped working in two days."
Sentiment: Negative

Review: "It's okay, nothing special."
Sentiment: Neutral

Review: "The product broke after one day."
Sentiment:
```

### Why Few-Shot Works

The model uses the demonstrated examples to:
1. Understand the exact output format expected
2. Calibrate the "distribution" of acceptable answers
3. Learn task-specific nuances that are hard to describe verbally

### Critical Considerations for Few-Shot

**Example Selection Matters:**
- Choose **diverse** examples that cover different cases
- Include **difficult** or **edge cases**, not just easy ones
- Ensure examples are **correct** — bad examples actively harm performance

**Example Ordering:**
- Order matters! Models show recency bias — examples near the end of the prompt have slightly more influence
- Always test different orderings

**Example Count:**
- More is not always better — at some point, more examples just waste tokens
- Typically 3–8 examples strike the right balance

**Distribution:**
- Ensure your examples reflect the distribution of real inputs, not just one type

---

## 2.4 Chain-of-Thought (CoT) Prompting

### What it is

Chain-of-Thought prompting instructs the model to **reason step by step** before arriving at a final answer, rather than jumping directly to the answer.

> *"Let's think step by step."*

This simple addition dramatically improves performance on tasks requiring multi-step reasoning, math, logic, or planning.

### Why CoT Works

LLMs generate tokens left-to-right. If you ask for the answer directly, the model cannot "think" — it just pattern-matches to a plausible final token. CoT forces the model to produce intermediate reasoning tokens, which act as a working memory that makes the final answer more grounded.

### Types of CoT

**Zero-Shot CoT:**
```
Q: A store has 48 chocolates. It sells 3/4 of them. How many are left?
A: Let's think step by step.
```

**Few-Shot CoT (reasoning steps in examples):**
```
Q: Roger has 5 tennis balls. He buys 2 more cans of 3 balls each. How many?
A: Roger starts with 5 balls. He buys 2 × 3 = 6 new balls. 5 + 6 = 11 balls.

Q: A store has 48 chocolates. Sells 3/4. How many left?
A: 3/4 of 48 = 36 sold. 48 - 36 = 12 remaining.
```

### When to Use CoT

| Use CoT | Don't Use CoT |
|---|---|
| Math problems | Simple classification |
| Multi-step logical reasoning | Direct lookup tasks |
| Planning and decomposition | Single-sentence Q&A |
| Complex decision-making | Speed-critical production (adds tokens) |

### Pitfall: CoT ≠ Correct Reasoning

CoT makes the model's reasoning visible, but visible reasoning can still be wrong. The model can produce **plausible-looking but incorrect** chain-of-thought. Always evaluate CoT outputs against ground truth.

---

## 2.5 Prompt Chaining

### What it is

Prompt Chaining breaks a **complex, multi-step task** into a sequence of simpler sub-tasks, each handled by a separate, focused prompt. The output of one prompt becomes the input of the next.

```
Task → Prompt 1 → Output 1 → Prompt 2 → Output 2 → Prompt 3 → Final Output
```

### The Problem It Solves

Single-prompt approaches to complex tasks fail because:
- The model must juggle too many instructions simultaneously
- Error in any early step propagates and compounds
- The model's attention degrades for instructions far from the query
- Long prompts make evaluation and debugging hard

### Example: Multilingual Fact Extraction

**Single Prompt (problematic):**
```
Consider the given text in Spanish. Translate it into English. Find all the
statistics and facts used in this text and list them as bullet points. Translate
them again into Spanish.
```
This is too much for one call. The model often makes errors mid-chain.

**Chained Approach:**
```
Prompt 1: "Read this Spanish text and translate it into English."
          → Output: English translation

Prompt 2: "Extract all statistics and facts from this English text as bullet
           points. Only extract facts explicitly stated, do not infer."
          → Output: Fact list in English

Prompt 3: "Translate each of these bullet points into Spanish."
          → Output: Spanish fact list
```

### Benefits of Chaining

1. **Isolation:** Each step is independently verifiable
2. **Debuggability:** You can pinpoint exactly which step failed
3. **Modularity:** Reuse individual chain links for other tasks
4. **Quality control:** You can insert validation or filtering between steps

### When to Use Prompt Chaining

- Tasks with clearly separable sequential steps
- Long documents requiring stage-wise processing
- Multi-modal pipelines (extract → analyze → summarize → format)
- Workflows where intermediate human review is needed

---

## 2.6 Role Playing (Persona Assignment)

### What it is

Role Playing assigns the model a **specific persona, role, or expertise** to prime it to respond from a particular perspective, with appropriate vocabulary, depth, and tone.

```
"You are a [ROLE]. [Task]."
```

### Why It Works

When the model is assigned a role, it draws on training data associated with that role — the writing style, domain vocabulary, level of detail, and reasoning patterns typical of that persona. This acts as an implicit "domain filter."

### Before & After Examples

**Example 1 — Cybersecurity:**
```
❌ "Explain vulnerability assessment."
   → Generic, surface-level answer

✅ "You are a senior cybersecurity analyst presenting to a C-suite executive
    who needs to understand the business risk. Explain vulnerability assessment
    clearly without jargon."
   → Appropriately scoped, business-focused explanation
```

**Example 2 — Nutrition:**
```
❌ "Create a meal plan for someone with high cholesterol."
   → Generic plan, possibly unsafe without medical context

✅ "You are an experienced nutritionist specializing in heart health. Create a
    three-day meal plan for a 45-year-old patient with high cholesterol.
    Include explanations for why each meal is beneficial."
   → Evidence-based, specific, appropriately detailed
```

### Effective Persona Design

A strong persona specification includes:
1. **Role** — Who is this model supposed to be?
2. **Expertise level** — Senior? Expert? Beginner-friendly teacher?
3. **Audience** — Who is the model talking to?
4. **Constraints** — What must the persona avoid or always do?
5. **Tone** — Professional, empathetic, direct, technical?

### Combination: Role + Constraints + CoT

The most powerful prompts often combine persona assignment with specific constraints and chain-of-thought:

```
"You are an expert financial analyst reviewing a startup's pitch deck.
Your audience is a venture capital partner.
Think step by step:
1. Identify the core business model
2. Assess market size credibility
3. Evaluate the team's fit with the opportunity
4. Highlight the top 3 risks
5. Provide a one-paragraph investment recommendation."
```

---

## Learning Highlights

> **Instruction Refinement insight:** The more specific your prompt, the less the model has to "guess." Every ambiguous word in a prompt is a potential hallucination point.

> **Iterative Refinement insight:** Don't aim for a perfect prompt on the first try. Treat prompt writing like TDD — write the test (expected output), then refine the prompt until it passes.

> **Few-Shot insight:** Examples communicate constraints that are nearly impossible to express in words. If you can't describe the rule, show it.

> **CoT insight:** The model can't "think" without tokens to think in. CoT gives it a scratchpad — the quality of the scratchpad determines the quality of the final answer.

> **Chaining insight:** Complexity is the enemy of reliability. The moment a task has more than 3 sequential steps, break it into a chain.

> **Role Playing insight:** Personas are implicit retrieval of domain knowledge. A "senior cardiologist" role retrieves cardiology knowledge; a "marketing copywriter" role retrieves persuasive writing patterns.

---

## Interview Questions

### Foundational

**Q1. What is the difference between instruction refinement and iterative prompt refinement?**

*Answer:* Instruction refinement is a targeted improvement to the clarity, specificity, and constraints of an instruction in a single edit. Iterative prompt refinement is a multi-round feedback loop: you run the model, identify a specific failure, make a minimal targeted change, re-run, and repeat. Instruction refinement is one technique you apply within each iteration of the broader iterative refinement process.

---

**Q2. Why does few-shot prompting often outperform zero-shot prompting?**

*Answer:* Few-shot prompting provides concrete demonstrations of the expected input-output mapping, which communicates format, quality, and style more precisely than verbal description. LLMs are pretrained on enormous text corpora and are excellent at pattern-continuation — seeing examples of the target behavior primes the model's output distribution toward that behavior. Zero-shot relies entirely on the model's interpretation of verbal instructions, which introduces ambiguity.

---

**Q3. What is Chain-of-Thought prompting and when should you NOT use it?**

*Answer:* CoT prompting instructs the model to reason through intermediate steps before producing a final answer, which improves performance on multi-step reasoning tasks. You should NOT use CoT when: (1) the task is simple and doesn't require reasoning (e.g., classification, direct lookup), (2) latency is critical and you cannot afford extra tokens, or (3) intermediate reasoning isn't verifiable (the chain can look correct but be wrong, building false confidence).

---

**Q4. What is prompt chaining and what problem does it solve?**

*Answer:* Prompt chaining decomposes a complex task into a sequence of simpler sub-tasks, each handled by a separate prompt where the output of one prompt feeds as input to the next. It solves the problem of attention dilution in single long prompts, makes each step independently debuggable and verifiable, reduces compounding errors from overly complex single-call prompts, and enables modular reuse of individual pipeline stages.

---

### Intermediate

**Q5. How do you select good few-shot examples? What makes a bad example?**

*Answer:* Good few-shot examples are: diverse (covering different input types), challenging (include edge cases, not just easy cases), and correct (any error in an example will be learned and repeated). Bad examples are: all from the same narrow category (creates false confidence in easy cases), incorrect (directly trains the wrong behavior), or formatted inconsistently (confuses the model about the expected format). Example ordering also matters — examples closer to the query have slightly more influence.

---

**Q6. You are building a legal document summarizer. How would you use role playing and prompt chaining together?**

*Answer:* (1) Chain Step 1 — use a "legal analyst" persona prompt to identify and extract key clauses from the raw document. (2) Chain Step 2 — use a "plain English translator" persona to rewrite each extracted clause in accessible language. (3) Chain Step 3 — use a "risk assessor" persona to flag any clauses that represent unusual risk. This combination gives you role-appropriate behavior at each step, with clear separation of concerns and individual verifiability.

---

**Q7. What are the failure modes of Chain-of-Thought prompting?**

*Answer:* (1) **Plausible but wrong reasoning** — the chain looks logical but contains a subtle error that leads to a wrong final answer. (2) **Hallucinated intermediate steps** — the model fabricates facts in the reasoning chain to make the argument cohere. (3) **Over-explanation** — the model writes verbose reasoning that adds latency and cost without improving accuracy. (4) **Sycophancy** — when user provides hints in the question, the model reasons toward the hinted answer regardless of correctness.

---

### Advanced

**Q8. How would you use manual prompt optimization to reduce hallucinations in a RAG (Retrieval Augmented Generation) system?**

*Answer:* (1) **Instruction refinement:** Add explicit constraints like "Only use facts from the provided context. If the answer is not in the context, say 'I don't have enough information'." (2) **Role assignment:** Use a "research analyst who only cites provided sources" persona. (3) **CoT:** Ask the model to identify which sentence in the context supports each claim before writing the answer. (4) **Iterative refinement:** Build a regression set from known hallucination cases and iterate until all pass. (5) **Chaining:** Separate retrieval, relevance checking, and answer generation into separate prompts.

---

**Q9. Compare and contrast few-shot prompting with fine-tuning. When would you prefer one over the other?**

*Answer:* Few-shot prompting provides task demonstrations in-context at inference time — fast to iterate, no training needed, but consumes tokens and context window, and the model's weights don't change. Fine-tuning trains the model's weights on task-specific data — more efficient at inference (fewer/no examples needed), but expensive and slow to iterate, requires labeled data, and risks overfitting. Prefer few-shot when: you need rapid iteration, the dataset is small, or the task changes frequently. Prefer fine-tuning when: inference cost is critical (tokens are expensive at scale), the task is stable and well-defined, and you have sufficient labeled data.

---

## Quick Reference Summary

| Technique | Core Idea | Best For |
|---|---|---|
| Instruction Refinement | Increase clarity, specificity, constraints | Any prompt as a baseline improvement |
| Iterative Refinement | Feedback loop: run → identify failure → fix one thing | Continuous prompt improvement |
| Few-Shot Prompting | Show examples of correct input-output | Format-sensitive tasks, hard-to-describe patterns |
| Chain-of-Thought | Force step-by-step reasoning | Math, logic, planning, multi-step tasks |
| Prompt Chaining | Break complex task into sequential sub-prompts | Multi-step pipelines, long documents |
| Role Playing | Assign a domain expert persona | Domain-specific tasks, tone/audience alignment |
