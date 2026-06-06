# Lec 7 — Section 1: Foundations
**Course:** Advanced Prompt Engineering | IIT Kharagpur  
**Instructor:** Prof. Koustav Rudra  
**Module:** 2, Session 1

---

## Topics Covered
1. Introduction to Prompting
2. Autoregressive Language Models
3. What is Prompt Engineering?
4. Common LLM Inference Terms — Tokens, Temperature, Top-K, Top-P

---

---

## Topic 1 — Introduction to Prompting

### What is a Prompt?

A **prompt** is the initial text input you give to a language model. It is the primary interface between a human and an LLM — the model uses it to understand intent and generate a relevant response.

A prompt can take many forms:
- A **question** — *"Write me a story about a full moon"*
- A **command** — *"Translate this text to French"*
- A **text snippet** — *"Classify the sentiment: What a beautiful day"*
- A **code snippet** — *"Convert this Python function to C++"*
- A **complex paragraph** with constraints and context

### System Prompts vs User Prompts

| Type | Description | Example |
|------|-------------|---------|
| **System Prompt** | Instructions given to the model about how it should behave — its role, constraints, and style | *"You are a helpful assistant. Do not answer questions outside of finance."* |
| **User Prompt** | The actual input from the end-user to get a response | *"What is the P/E ratio of Apple?"* |

> The system prompt sets the **persona and guardrails**. The user prompt drives the **actual task**.

### Why Prompting = Programming with Natural Language

Traditional programming is rigid and deterministic:
```python
sort(numbers, descending=True)
```

Prompting is flexible and natural:
```
Sort these numbers from largest to smallest and explain the steps.
```

Both are giving instructions, defining inputs/outputs, and controlling behavior — but one uses code syntax and the other uses natural language. Prompt engineering is iterative, just like software development.

### Key Properties of a Good Prompt
- **Clarity** — unambiguous instruction
- **Context** — enough background for the model to understand intent
- **Specificity** — constrains the output to what you actually want
- **Format guidance** — tells the model how to structure its response

---

> **Learning Thought:**  
> A prompt is not just a question — it is a contract between you and the model. The model has no memory, no intuition, no common sense beyond its training. Everything it knows about *your* task comes from what you put in the prompt. Poor prompts produce poor outputs — not because the model is dumb, but because it was given an underspecified contract.

---

### Interview Questions — Topic 1

**Q1. What is a prompt in the context of LLMs?**  
A prompt is the text input provided to a language model that guides it to produce a specific output. It can be a question, command, code snippet, or any structured text that communicates the user's intent.

**Q2. What is the difference between a system prompt and a user prompt?**  
A system prompt sets the model's role, behavior, and constraints (set by the developer). A user prompt is the end-user's actual request or query at runtime. Together they shape the model's response.

**Q3. Why is prompting considered a form of programming?**  
Like programming, prompting involves giving structured instructions, defining expected inputs and outputs, controlling behavior, and iterating to improve results. The key difference is that prompts use natural language instead of code syntax.

**Q4. What makes a prompt effective?**  
Clarity of instruction, sufficient context, specificity about the desired output format, and constraints that prevent the model from going off-track.

---

---

## Topic 2 — Autoregressive Language Models

### What is an Autoregressive Language Model?

An **autoregressive language model** generates text **one token at a time**, always predicting the next token based on all previously generated tokens.

The core idea:

```
P(sentence) = P(t1) × P(t2 | t1) × P(t3 | t1, t2) × ... × P(tn | t1...tn-1)
```

Each token's probability is **conditioned on all previous tokens**. This is the "auto" in autoregressive — the model feeds its own outputs back as inputs for the next prediction.

### What is a Token?

A token is the **basic unit of text** that LLMs process. Tokens are not always whole words — they can be:
- A full word: `"transformer"`
- Part of a word (subword): `"methyl"` + `"ation"` → `"methylation"`
- Punctuation: `.`, `,`, `!`
- Spaces or special symbols

**Example:** The complex sentence:
> *"Pseudohypoparathyroidism and deoxyribonucleic acid methylation affect mitochondrial dysfunction in immunocompromised patients undergoing chemotherapeutic treatment."*

...uses **33 tokens** despite being long — because the tokenizer breaks it into subwords efficiently.

### Subword Tokenization

The most common approach is **Byte Pair Encoding (BPE)** or similar subword tokenization:

| Rule | Example |
|------|---------|
| Frequent words stay whole | `"the"`, `"transformer"`, `"model"` |
| Rare words split into subwords | `"methylation"` → `"methyl"` + `"ation"` |
| Unknown chars handled gracefully | `"🔥"` → special token |

This allows the model to handle **rare or unseen words** without a vocabulary blowup.

### How Generation Works

At each step, the model outputs a **probability distribution** over all possible next tokens:

```
Input: "The cat sat on the ..."

Predictions:
  "mat"    → 62%
  "floor"  → 15%
  "chair"  → 8%
  "roof"   → 3%
  ...
```

The model then **samples** from this distribution (based on temperature, top-k, top-p settings) to pick the next token. This repeats until an end-of-sequence token is generated or max length is hit.

So a full generation looks like:
```
"The" → "cat" → "sat" → "on" → "the" → "mat"
```

Each arrow is one prediction step.

### Why Autoregressive Matters for Prompting

- The model **reads your entire prompt** before generating anything
- It generates **left to right, one token at a time** — it cannot "go back and fix" earlier tokens
- This means the **beginning of your prompt heavily influences** what comes later
- Longer prompts = more context = generally better outputs, but also more compute

---

> **Learning Thought:**  
> Understanding autoregressive generation explains why prompt structure matters so much. When you write a prompt, you are literally setting the initial tokens that the model conditions all future tokens on. A vague start leads to a vague completion — because every subsequent token is probabilistically downstream of everything before it.

---

### Interview Questions — Topic 2

**Q1. What does "autoregressive" mean in the context of language models?**  
It means the model generates text sequentially — one token at a time — where each new token is predicted based on all previously generated tokens. The model's own outputs become part of its input for the next step.

**Q2. What is a token, and why don't tokens always equal words?**  
A token is the smallest unit of text the model processes. Models use subword tokenization (e.g., BPE) so common words stay whole while rare words are split into smaller subword pieces. This makes the vocabulary manageable while handling rare/unseen words.

**Q3. Why can't autoregressive models correct earlier parts of their output?**  
Because generation is strictly left-to-right. Once a token is generated, it becomes part of the fixed context for all future tokens. The model has no backtracking mechanism during inference — it committed to that token based on the probability distribution at that step.

**Q4. How does the autoregressive nature of LLMs affect how we write prompts?**  
Since each token conditions the next, putting the most important context and instructions early in the prompt tends to anchor the model's behavior better. The structure and ordering of information in a prompt directly shapes the probability distribution at each generation step.

**Q5. What is subword tokenization and why is it used?**  
Subword tokenization breaks words into smaller units (subwords) so the model can handle both common and rare words with a fixed, manageable vocabulary. Frequent words remain whole; rare words are split into known subword pieces — reducing out-of-vocabulary problems.

---

---

## Topic 3 — What is Prompt Engineering?

### Definition

**Prompt Engineering** is the systematic process of designing, structuring, and refining prompts to reliably get the desired output from a language model.

The "engineering" aspect reflects that this is:
- **Systematic** — follows principles, not just guessing
- **Iterative** — you refine prompts like you refine code
- **Task-optimized** — the best prompt for summarization differs from the best for code generation
- **Empirical** — you test and measure results

### Use Cases of Prompt Engineering

| Use Case | Prompt | Expected Output |
|----------|--------|----------------|
| **Summarization** | *"Summarize: 'The quick brown fox...'"* | A fox jumps over a dog. |
| **Sentiment Analysis** | *"Evaluate sentiment: 'This product exceeded my expectations!'"* | Positive sentiment |
| **Personalization** | *"Recommend a book for a mystery lover."* | *'The Silent Patient'* |
| **Code Generation** | *"Write a Python function that returns the sum of two numbers."* | Working Python code |
| **Code Conversion** | *"Convert this Python factorial function to C++."* | C++ equivalent |
| **Translation** | *"Translate to French: 'Hello, how are you?'"* | "Bonjour, comment allez-vous?" |

### The Iterative Refinement Loop

Prompt engineering is never "one and done." The process looks like:

```
Draft Prompt
     ↓
Run it → Observe output
     ↓
Identify what went wrong (too vague? wrong format? hallucinating?)
     ↓
Refine the prompt
     ↓
Repeat until the output is reliable
```

This mirrors software development's write → test → debug → iterate cycle.

### What Prompt Engineering is NOT

- It is **not fine-tuning** — you're not changing model weights
- It is **not magic** — there are principled techniques that work reliably
- It is **not permanent** — the same prompt may work differently across model versions

### Why Prompt Engineering Matters

The effectiveness of a prompt **directly determines** the relevance and accuracy of the model's output. A well-engineered prompt:
- Minimizes hallucination
- Gets the right format without extra post-processing
- Reduces the need for expensive fine-tuning
- Makes AI applications predictable and reliable

---

> **Learning Thought:**  
> Prompt engineering is the closest thing to "programming" a neural network without touching its weights. You're not changing what the model knows — you're changing how it accesses and applies what it knows. A great prompt engineer understands the model's strengths and limitations deeply enough to write prompts that navigate around failure modes.

---

### Interview Questions — Topic 3

**Q1. What is prompt engineering and why is it called "engineering"?**  
Prompt engineering is the systematic, iterative design of prompts to reliably elicit desired outputs from LLMs. It's called engineering because it involves principled design, testing, measurement, and refinement — not just informal trial and error.

**Q2. How is prompt engineering different from fine-tuning?**  
Fine-tuning modifies the model's weights using additional training data — it changes what the model knows. Prompt engineering only changes the input — it shapes how the model applies existing knowledge without any training.

**Q3. Name three practical use cases of prompt engineering.**  
Summarization, sentiment classification, code generation, translation, question answering, data extraction — any two or three with brief explanation.

**Q4. Why do well-structured prompts improve model output quality?**  
Because they reduce ambiguity — the model better understands the task, the expected format, and the constraints. Ambiguous prompts leave the model guessing, which increases variance and errors.

**Q5. Is prompt engineering a one-time task?**  
No. It requires continuous iteration. Prompts must be tested across many inputs, refined when failures occur, and updated when models are upgraded. It's a living artifact, similar to code.

---

---

## Topic 4 — Common LLM Inference Terms

This topic covers four critical parameters that control **how** an LLM generates text. Understanding these is essential for both using LLMs effectively and for interviews.

---

### 4.1 — Tokens (revisited in inference context)

At inference time, tokens matter because:
- **Cost** is calculated per token (input + output)
- **Context window** is measured in tokens
- **Latency** scales with number of tokens generated

Rule of thumb: **~1 token ≈ 0.75 words** in English.

---

### 4.2 — Temperature

**Temperature** controls the **sharpness (or flatness) of the probability distribution** over the next token. It is the single most important inference parameter.

#### How it works mathematically

The model produces raw scores called **logits** for each token. These are converted to probabilities via the **softmax function**:

```
P(token_i) = exp(logit_i / T) / Σ exp(logit_j / T)
```

Where **T is the temperature**.

#### Effect of Temperature

| Temperature | Effect | Use Case |
|------------|--------|----------|
| **T → 0** | Near-deterministic. Highest-probability token almost always picked. Very repetitive. | Factual Q&A, code generation |
| **T = 1** | Default. Uses raw logits directly. Balanced creativity. | General chat, summarization |
| **T > 1** | Flatter distribution. More random and diverse — but risks incoherence. | Creative writing, brainstorming |

#### Concrete Example

Given logits: `cat=4.0, dog=2.0, fish=1.0, banana=0.0`

| Temperature | cat | dog | fish | banana | Entropy |
|-------------|-----|-----|------|--------|---------|
| T = 0.5 | 97.9% | 1.8% | 0.2% | 0.03% | 0.110 (very sharp) |
| T = 1.0 | 83.1% | 11.2% | 4.1% | 1.5% | 0.594 (balanced) |
| T = 2.0 | 57.9% | 21.3% | 12.9% | 7.8% | 1.109 (flat, diverse) |

> Note: T = 0 is mathematically undefined (division by zero). In practice, it's approximated with a very small value (e.g., 0.000001) or replaced by **greedy decoding** (always pick the argmax token).

---

> **Learning Thought:**  
> Temperature is not a creativity dial in some mystical sense — it's a mathematical lever on the entropy of the token distribution. Lower temperature = lower entropy = the model is more "confident" (or more precisely, concentrates probability mass). Higher temperature = higher entropy = the model spreads probability across more choices. You're literally trading coherence for diversity.

---

### 4.3 — Top-K Sampling

**Top-K sampling** restricts the model to sampling only from the **K most probable tokens** at each step, discarding the rest.

#### How it works

```
At each step:
  1. Compute probabilities for all tokens in vocabulary (~50k tokens)
  2. Keep only the top K tokens
  3. Re-normalize their probabilities to sum to 1
  4. Sample from these K tokens
```

#### Example — "The sun rises in the ___"

The model might assign:
```
"east"    → 19%
"morning" → 17%
"sky"     → 15%
"distant" → 14%
"western" → 10%
"glowing" → 8%
...
```

With **K=5**, only `east, morning, sky, distant, western` are candidates. `glowing` and all other tokens are **clipped** (probability set to 0).

#### Pros and Cons

| Pros | Cons |
|------|------|
| Prevents very low-probability nonsense tokens | K is fixed regardless of distribution shape |
| More predictable and controlled output | May include bad tokens if top-K options are all weak |
| Reduces randomness from "long tail" tokens | May exclude good tokens if K is too small |

---

> **Learning Thought:**  
> Top-K solves a real problem: even with reasonable temperature, a vocabulary of 50,000 tokens means thousands of options have non-zero probability. Without truncation, a bad token can always slip through. Top-K creates a hard boundary. The weakness is that "K" is context-blind — sometimes the top 5 tokens are nearly equal in probability (broad distribution), sometimes one token dominates with 99% probability. A fixed K doesn't adapt to this.

---

### 4.4 — Top-P (Nucleus Sampling)

**Top-P sampling** (also called **nucleus sampling**) is a smarter alternative to Top-K. Instead of picking a fixed number of tokens, it picks the **smallest set of tokens whose cumulative probability exceeds threshold P**.

#### How it works

```
At each step:
  1. Sort tokens by probability, highest first
  2. Accumulate probabilities until the sum ≥ P
  3. This is the "nucleus" — the set of tokens selected
  4. Re-normalize and sample from the nucleus
```

#### Example — "The sun rises in the ___"

With **P = 0.65**:
```
"east"    → 19%  (cumulative: 19%)
"morning" → 17%  (cumulative: 36%)
"sky"     → 15%  (cumulative: 51%)
"distant" → 14%  (cumulative: 65%) ← threshold reached!
```

Only these 4 tokens form the nucleus. All others are discarded.

#### Why Top-P is better than Top-K

| Scenario | Top-K behavior | Top-P behavior |
|----------|---------------|----------------|
| One token has 99% probability | Still picks from K tokens unnecessarily | Nucleus = just that 1 token ✓ |
| Probability is spread evenly across many tokens | May be too restrictive | Nucleus adapts and includes more tokens ✓ |

Top-P is **dynamic** — the nucleus size adapts to the probability distribution shape. This is why it's preferred in most modern LLM APIs.

#### Typical Values

| Parameter | Common Range | Notes |
|-----------|-------------|-------|
| Temperature | 0.1 – 1.5 | Lower for factual tasks, higher for creative |
| Top-K | 10 – 100 | Often combined with Top-P |
| Top-P | 0.7 – 0.95 | 0.9 is a common default |

---

> **Learning Thought:**  
> Top-P is elegant because it respects the **shape** of the distribution rather than its **size**. Think of it this way: if the model is very confident (sharp distribution), it should pick from fewer options. If the model is uncertain (flat distribution), it should explore more. Top-K ignores this nuance; Top-P embraces it. This is why nucleus sampling tends to produce more natural-sounding text.

---

### Putting It All Together: How These Parameters Interact

In practice, temperature + top-k + top-p are often used together:

```
Step 1: Apply temperature → reshape the logit distribution
Step 2: Apply Top-K → clip to top K tokens
Step 3: Apply Top-P → further clip to nucleus
Step 4: Sample from the remaining candidates
```

For most production use cases:
- **Deterministic/factual tasks** → Low temp (0.1–0.3), low top-p (0.5–0.7)
- **Creative tasks** → Higher temp (0.7–1.2), higher top-p (0.9–0.95)
- **Code generation** → Very low temp (0.0–0.2), greedy or low top-p

---

### Interview Questions — Topic 4

**Q1. What is temperature in LLM inference and what happens when you set it to 0?**  
Temperature scales the logits before the softmax function. Higher temperature flattens the distribution (more random), lower sharpens it (more deterministic). T=0 is technically undefined; in practice it approximates greedy decoding — always picking the highest-probability token.

**Q2. What is the difference between Top-K and Top-P sampling?**  
Top-K keeps only the K most probable tokens at each step (fixed number). Top-P (nucleus sampling) keeps the smallest set of tokens whose cumulative probability exceeds P (dynamic number). Top-P adapts to the distribution shape; Top-K does not.

**Q3. When would you prefer Top-P over Top-K?**  
Almost always for general use, because Top-P adapts to whether the distribution is sharp or flat. When the model is confident, the nucleus naturally shrinks; when uncertain, it expands. Top-K would include unnecessary tokens in the first case and exclude good ones in the second.

**Q4. If temperature is increased, what happens to output diversity and coherence?**  
Diversity increases (the model explores more tokens) but coherence can decrease (low-probability tokens can be selected, leading to less predictable or incoherent text). There's a fundamental trade-off between creativity and reliability.

**Q5. What is entropy in the context of temperature scaling?**  
Entropy measures the spread of the probability distribution. Low entropy = probability concentrated on few tokens (model is confident). High entropy = probability spread across many tokens (model is uncertain). Temperature directly controls entropy — lower T reduces entropy, higher T increases it.

**Q6. What is greedy decoding?**  
Greedy decoding always picks the single highest-probability token at each step, with no sampling. It produces deterministic, repetitive output. Used when T≈0 or when strict reproducibility is needed.

**Q7. Why is the token count important from a cost and latency perspective?**  
LLM APIs typically charge per token (input + output). Longer prompts cost more and hit context window limits faster. Output token count directly determines generation latency — each token requires a full forward pass.

---

---

## Quick Reference Cheat Sheet — Section 1

| Concept | One-Line Summary |
|---------|-----------------|
| Prompt | Text input that controls what the LLM does |
| System Prompt | Developer-set role, rules, and constraints |
| User Prompt | End-user's actual query |
| Autoregressive LM | Generates one token at a time, each conditioned on all previous tokens |
| Token | Smallest unit of text (word, subword, or character) |
| Prompt Engineering | Systematic, iterative design of prompts for reliable outputs |
| Temperature | Controls sharpness of token probability distribution |
| Top-K | Sample only from the K most likely tokens |
| Top-P / Nucleus | Sample from tokens whose cumulative probability ≥ P |
| Greedy Decoding | Always pick the highest-probability token (T→0) |

---

## Key Papers Referenced in This Section

| Paper | Contribution |
|-------|-------------|
| Brown et al. (GPT-3, 2020) | Demonstrated zero-shot and few-shot prompting at scale |
| Wei et al. (2021) | Zero-shot instruction following |
| Holtzman et al. (2020) | Introduced nucleus (Top-P) sampling |

---

*Next: Section 2 — Core Prompting Techniques (Zero-Shot, Few-Shot, Chain-of-Thought)*
