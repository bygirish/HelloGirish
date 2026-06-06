# Section 3: Large Language Models — Response Generation & Decoding
**Lecture 4 | IIT Kharagpur × upGrad | Instructor: Prof. Sourangshu Bhattacharya**

> **Learning Goal:** Deeply understand how a language model turns logits into actual text. Master every generation parameter (temperature, top-k, top-p, penalties, max_tokens), know when to use each, and be able to configure them for any use case.

---

## Table of Contents
1. [How Text Generation Works — The Foundation](#foundation)
2. [What Are Logits?](#logits)
3. [Greedy Decoding](#greedy)
4. [The Problem with Greedy Decoding](#greedy-problem)
5. [Beam Search](#beam-search)
6. [Greedy & Beam Search Are Deterministic](#deterministic)
7. [Quality vs Diversity Trade-off](#qvd)
8. [Random Sampling with Temperature](#temperature)
9. [Top-k Sampling](#topk)
10. [Top-p (Nucleus) Sampling](#topp)
11. [Top-k vs Top-p: Key Comparison](#topk-vs-topp)
12. [Repetition Penalty & Frequency Penalty](#penalties)
13. [Max Tokens & Stop Sequences](#max-stop)
14. [Parameter Cheat Sheet](#cheatsheet)
15. [Use-Case Configurations](#usecases)
16. [Interview Questions](#interview)
17. [Key Learning Thoughts](#learning-thoughts)

---

## 1. How Text Generation Works — The Foundation {#foundation}

When an LLM generates text, it works **one token at a time** in a loop:

```
Input tokens → Transformer → Logits over full vocabulary 
                                        ↓
                              Decoding strategy applied
                                        ↓
                              One token selected
                                        ↓
                         Token appended → new input for next step
                                        ↓
                         Repeat until [EOS] or max_tokens reached
```

The Transformer outputs a **probability distribution over the entire vocabulary** (e.g., 50,000 possible next tokens). The **decoding strategy** decides which token to pick from that distribution.

> **Key insight:** The model itself is always the same. What changes is the *rule* for picking a token from the distribution. All the parameters (temperature, top-k, top-p, etc.) are about manipulating this selection rule.

---

## 2. What Are Logits? {#logits}

**Logits** are the raw, unnormalized scores the model gives to each possible next token *before* applying softmax.

Example for predicting the next word after "The capital of France is":
```
Logits:  "Paris" → 5.2    "London" → 3.1    "Berlin" → 2.8    ...
After softmax → probabilities:
         "Paris" → 0.85   "London" → 0.15   "Berlin" → 0.10   ...
```

Higher logit = higher probability after softmax. All temperature, top-k, and top-p operations happen on logits BEFORE softmax (or on the probabilities AFTER softmax, depending on implementation).

> **Why logits matter to you:** Every decoding parameter you set (temperature, penalties) directly manipulates logits. Understanding this gives you intuition for what each parameter actually does mathematically.

---

## 3. Greedy Decoding {#greedy}

### Definition
**Greedy decoding** always selects the single most probable next token at each step:

```
ŵ_t = argmax P(w | w_{<t})
       w ∈ V
```

In English: at every step t, pick the word `w` from vocabulary `V` that maximizes the conditional probability given all previous words.

### Algorithm
```
1. Start with input tokens
2. For each generation step:
   a. Compute probability distribution over all vocabulary tokens
   b. Select the token with the HIGHEST probability
   c. Append it to the sequence
3. Stop at [EOS] or max_tokens
```

### Properties
- **Deterministic:** Same input always produces same output
- **Fast:** No branching, no candidates to maintain
- **Simple:** One line of logic per step

### Example
Input: `"The weather is"`
- Step 1: `nice` (0.4), `cold` (0.3), `hot` (0.2) → picks `nice`
- Step 2: `today` (0.5), `but` (0.3) → picks `today`
- Output: `"The weather is nice today"`

---

## 4. The Problem with Greedy Decoding {#greedy-problem}

Greedy decoding is **locally optimal but globally suboptimal**.

### The Problem Illustrated

Consider token probabilities (simplified tree):

```
Start → "ok" (0.4) → "ok" (0.7) → EOS (1.0)   [sequence prob: 0.4 × 0.7 = 0.28]
      → "yes" (0.5) → "yes" (0.4) → EOS (1.0)  [sequence prob: 0.5 × 0.4 = 0.20]
                    → "ok" (0.3) → EOS (1.0)   [sequence prob: 0.5 × 0.3 = 0.15]
```

**Greedy chooses:** `"yes"` at step 1 (highest probability = 0.5), then `"yes"` at step 2 (0.4).
**Final sequence:** `"yes yes"` with probability 0.5 × 0.4 = 0.20

**Globally optimal:** `"ok ok"` with probability 0.4 × 0.7 = 0.28

Greedy committed to `"yes"` first because 0.5 > 0.4, but the sequence starting with `"ok"` was actually better overall. **Greedy locked in an early choice that led to a worse full sentence.**

### The Hiking Analogy (from class Q&A)
> Greedy decoding is like hiking and always taking the steepest upward path at every fork — you might end up on a small hill while a much bigger mountain was just one valley over.

### Real-world Consequence
- Greedy outputs tend to be **generic, repetitive, and safe** — the model keeps picking the statistically most common continuations
- Example failure: "The cat sat on the mat. The cat sat on the mat. The cat sat..."

---

## 5. Beam Search {#beam-search}

### Core Idea
Instead of committing to one path (greedy), **keep track of the k most probable partial sequences at each step** — where k is the *beam width* or *beam size*.

```
Core Idea:
At each step of the decoder:
  - Keep k most probable partial sequences (hypotheses)
  - k is the beam width (typically 5 to 10 in practice)
```

### Algorithm

```
Step 1: Initialize with k=2 beam. Start = ["", ""]

Step 2: For each existing beam hypothesis:
  - Compute probability distribution for next token
  - Extend hypothesis with ALL possible next tokens
  - Score each extended hypothesis by TOTAL sequence probability
  
Step 3: Keep only the TOP k hypotheses across all extensions

Step 4: Repeat until all beams hit [EOS] or max_tokens

Step 5: Return the hypothesis with the highest total probability
```

### Worked Example (Beam Size = 2)

Using the same tree from greedy example:

```
t=1: Beams = ["ok" (0.4), "yes" (0.5)]     [keep top 2]

t=2: Extend each beam:
  "ok" → "ok ok" (0.4×0.7=0.28), "ok yes" (0.4×0.2=0.08)
  "yes" → "yes ok" (0.5×0.3=0.15), "yes yes" (0.5×0.4=0.20)
  All 4 candidates: 0.28, 0.20, 0.15, 0.08
  Keep top 2: ["ok ok" (0.28), "yes yes" (0.20)]

Final: Return "ok ok" (0.28) — MATCHES the globally optimal!
```

Beam search with k=2 recovers the optimal sequence that greedy missed.

### Beam Search vs Greedy: The Key Difference

| | Greedy | Beam Search |
|---|---|---|
| Candidates tracked | 1 | k |
| Computational cost | 1× | k× |
| Deterministic? | Yes | Yes |
| Globally optimal? | No | Still heuristic (not guaranteed optimal) |
| Use in practice | Simple generation | Machine translation, speech recognition |

### Important: Beam Search is Still a Heuristic
Even beam search doesn't guarantee globally optimal sequences. Finding the exact best sequence would require exhaustive search over all possible sequences — computationally infeasible. Beam search is a practical approximation.

### Why GPT-4/Claude Don't Use Beam Search
Modern chatbots use **temperature + top-p sampling** instead. Beam search tends to produce:
- Safe, repetitive text that feels unnatural for conversation
- Text that "sounds" statistically correct but lacks creativity and variety
- Same output every time for the same input (both greedy and beam are deterministic)

For conversation, you want some controlled randomness — that's what sampling provides.

---

## 6. Greedy & Beam Search Are Deterministic {#deterministic}

Both greedy and beam search are **deterministic** — same input always produces same output. This is sometimes desirable (reproducible results) but:

1. No variety or creativity in outputs
2. Tends toward generic, repetitive text
3. Unsuitable for creative tasks

To introduce controlled randomness, we use **sampling methods**.

### Sampling: The Core Concept
Instead of picking the maximum probability token, **sample from the distribution**:

```
Random Sampling:
  i ← 1
  w_i ~ p(w)           ← sample from the distribution
  while w_i != EOS:
    i ← i + 1
    w_i ~ p(w_i | w_{<i})   ← sample from conditional distribution
```

The `~` symbol means "sample from" — we pick a token randomly, with higher-probability tokens more likely to be selected (but not guaranteed).

---

## 7. Quality vs Diversity Trade-off {#qvd}

This is the central tension in text generation:

```
QUALITY ◄──────────────────────────────► DIVERSITY
(deterministic, coherent, accurate)      (random, creative, varied)
    │                                           │
Greedy / Beam Search                    Pure Random Sampling
    │                                           │
"Paris is the capital of France."      "Paris is cheese-dream cobblestone forever!"
```

**Methods that emphasize high-probability words:**
- More coherent and accurate
- BUT more repetitive and boring

**Methods that give weight to middle-probability words:**
- More creative and diverse
- BUT potentially incoherent and less factual

All the sampling parameters (temperature, top-k, top-p) are **knobs on this dial**.

---

## 8. Random Sampling with Temperature {#temperature}

### The Thermodynamics Intuition
Temperature comes from physics: a system at **high temperature** is flexible (many states possible); at **low temperature** it settles into preferred (lower energy) states.

Applied to language models:
- **High temperature:** Model considers many tokens, output is varied and creative
- **Low temperature:** Model strongly prefers the highest-probability tokens, output is focused and deterministic

### The Mathematics

**Standard softmax:**
```
y = softmax(u)
```

**Temperature-scaled softmax:**
```
y = softmax(u / τ)    where τ ∈ (0, ∞)
```

Where `u` = logits vector, `τ` (tau) = temperature.

**Effect of dividing by τ:**
- τ < 1 (e.g., 0.3): Logits are scaled *up* (divided by small number = larger values). The gap between high and low logits widens. Softmax becomes more "peaked" — high probability tokens get even higher probability.
- τ = 1: No change. Natural distribution.
- τ > 1 (e.g., 1.8): Logits are scaled *down*. The gap narrows. Softmax becomes more "flat" — probabilities are more uniform.

### Temperature Examples (from slides)

| Temperature | Character | Behavior | Example Output |
|---|---|---|---|
| T = 0.3 | Very Cold | Almost deterministic. Always picks most likely token | "Paris is the capital of France." |
| T = 1.0 | Default | Natural distribution. Balanced creativity + coherence | "Paris is a stunning European capital." |
| T = 1.8 | Very Hot | High randomness. Creative but can be incoherent | "Paris is cheese-dream cobblestone forever!" |

### Practical Rules of Thumb (from lecture)
- **T ≈ 0.7** → factual tasks (QA, summarization, data extraction)
- **T ≈ 1.2** → creative writing (stories, marketing copy, brainstorming)
- **T > 1.5** → only for brainstorming / exploration
- **T = 0** → fully deterministic (always picks the single highest-probability token)

### Temperature for Numerical / Attribution Tasks (from Q&A)
> For drill-down analysis and numerical datasets: temp = 0–0.2, top_p = 0.7–0.9. You want accuracy, consistency, not creativity. Even simpler: temp = 0, top_p = 1 for maximum stability.

---

## 9. Top-k Sampling {#topk}

### Definition
Keep only the **k most probable next tokens**, discard the rest, renormalize, then sample.

```
Definition: Keep only the k most probable next tokens;
            redistribute probability mass among them.
```

### How It Works (Step by Step)

Given vocabulary probabilities over top 8 tokens:
```
dog   28%  ✓ (k=4)
cat   22%  ✓ (k=4)
bird  15%  ✓ (k=4)
fish  10%  ✓ (k=4)
horse  8%  ✗ (discarded)
rabbit 6%  ✗ (discarded)
turtle 4%  ✗ (discarded)
zebra  2%  ✗ (discarded)
```

**Steps:**
1. Compute logits over full vocabulary (~50,000 tokens)
2. Sort tokens by probability (descending)
3. Keep only the top-k tokens, discard the rest
4. Re-normalize the kept probabilities to sum to 1
5. Sample from this reduced distribution

**After renormalization (k=4):**
```
dog:  28/(28+22+15+10) = 37.3%
cat:  22/75 = 29.3%
bird: 15/75 = 20.0%
fish: 10/75 = 13.3%
```

### Effect of k Value

| Low k (e.g., k=5) | High k (e.g., k=80) |
|---|---|
| Conservative | Diverse |
| Predictable, focused | More varied, creative |
| Risk: repetitive | Risk: incoherent words |
| Good for: factual QA | Good for: creative writing |

### Important: top_k ≠ Output Length
> **Class Q&A clarification:** Setting `top_k = 50` does NOT mean the output will have 50 tokens. `top_k` controls **vocabulary diversity at every single step**. At each token generation, the model looks at ~50,000 candidates, keeps 50, and randomly picks one from those 50. This happens fresh for every token. To control output length, use `max_tokens` separately. They are completely independent parameters.

---

## 10. Top-p (Nucleus) Sampling {#topp}

### Definition
Sample from the **smallest set of tokens whose cumulative probability ≥ p**.

```
Definition: Sample from the smallest set of tokens whose 
            cumulative probability ≥ p.
            Dynamic vocabulary cutoff!
```

### How It Works

Example with p = 0.85:
```
Token    Prob    Cumulative
dog      28%     28%   ✓
cat      22%     50%   ✓
bird     15%     65%   ✓
fish     10%     75%   ✓
horse     8%     83%   ✓
rabbit    6%     89%   ← ✓ first time cumulative ≥ 85%
turtle    4%     93%   ✗ (below threshold)
zebra     2%     95%   ✗ (below threshold)
```

So with p=0.85, we use tokens {dog, cat, bird, fish, horse, rabbit} and discard {turtle, zebra}.

### Why "Nucleus" Sampling?
The selected tokens form the "nucleus" — the high-probability core of the distribution.

### The Key Advantage Over Top-k: Dynamic Cutoff

**Top-k problem:** k is fixed. When the model is very confident (one token has 99% probability), top_k=50 still includes 49 low-quality candidates. When the model is uncertain (uniform distribution), top_k=50 might cut off too aggressively.

**Top-p solution:** The cutoff **adapts** to the model's confidence level:
- **High confidence (model is certain):** The nucleus is small — only 1-2 tokens reach cumulative ≥ p
- **Low confidence (model is uncertain):** The nucleus is larger — need many tokens to reach cumulative ≥ p

This is why top-p tends to outperform top-k in practice for chat and creative tasks.

### Typical Values
- `p = 0.9–0.95` for creative tasks
- `p = 0.7` for factual tasks
- `p = 1.0` = no filtering (sample from full vocabulary)

---

## 11. Top-k vs Top-p: Key Comparison {#topk-vs-topp}

| Aspect | Top-k | Top-p (Nucleus) |
|---|---|---|
| **Cutoff type** | Fixed count (k tokens) | Dynamic (by probability mass) |
| **Low confidence** | Always uses k tokens | Restricts to fewer (auto-adapts) |
| **High confidence** | May include garbage | Stays focused |
| **Typical value** | k = 40–50 | p = 0.9–0.95 |
| **Best for** | Situations with known diversity needs | Adaptive chat/creative generation |

### The Beam Search vs Top-k Confusion (from class Q&A)
This is a common and important mix-up:

| | Beam Search | Top-k Sampling |
|---|---|---|
| **What "k" means** | Number of candidate *sequences* tracked | Number of candidate *tokens* per step |
| **Nature** | Deterministic search | Stochastic sampling |
| **Multi-sequence?** | Yes — k full sequences in parallel | No — generates one sequence |
| **Selection** | Picks highest-scoring final sequence | Samples randomly from filtered candidates |
| **Category** | Search algorithm | Sampling algorithm |

> Beam search keeps top-k *sequences* and picks deterministically.
> Top-k sampling filters to top-k *tokens per step* and picks randomly.
> Same letter "k," completely different mechanisms.

### Critical Warning: Never Combine top_k and top_p at Extreme Values
> Setting `top_k=5` AND `top_p=0.1` compounds the filtering — you might reduce the candidate pool to 1-2 tokens, producing degenerate, repetitive output. **Use one OR the other, then adjust temperature.** Most modern APIs default to `top_p` because it adapts dynamically.

---

## 12. Repetition Penalty & Frequency Penalty {#penalties}

Both penalties solve the same problem: **preventing repetitive, circular output**.

### Repetition Penalty

**Definition:** Divide the logit of any token that has **already appeared** in the context by the penalty value.

```
logit[t] /= penalty    (if token t was seen before in context)
```

**Effect:** Already-used tokens get lower probability → model avoids repeating them.

| Value | Effect |
|---|---|
| 1.0 | No effect (division by 1) |
| 1.2–1.5 | Typical useful range |
| > 2.0 | Avoids even necessary repeats (problematic) |

**Example:**
```
Without penalty: "The cat sat. The cat sat. The cat sat. The cat..."
With penalty=1.3: "The cat sat on the mat and then yawned slowly."
```

### Frequency Penalty

**Definition:** Subtract `frequency_penalty × count(t)` from the logit of token t, proportional to how many times it has already appeared.

```
logit[t] -= freq_penalty × count(t)
```

**Effect:** The *more* a token has been used, the *more* penalized it becomes. Unlike repetition penalty (binary: appeared or not), frequency penalty scales proportionally with count.

| Value | Effect |
|---|---|
| 0.0 | No effect |
| 0.4 | Typical creative setting |
| 0.1 | Typical factual setting |
| > 1.5 | Aggressive — may disrupt legitimate repetition |

**Example:**
```
Without penalty: "Paris Paris Paris is Paris the capital Paris Paris"
With penalty=0.5: "Paris is the capital of France and a major hub."
```

### Repetition vs Frequency: The Difference

| | Repetition Penalty | Frequency Penalty |
|---|---|---|
| **Applies** | If token appeared at all (binary) | Proportional to how often |
| **Effect** | Flat penalty for any repeat | Increasing penalty per repeat |
| **OpenAI name** | `repetition_penalty` | `frequency_penalty` |
| **Presence penalty** | Binary version (0 or 1) | — |

> **Presence penalty** (binary version): `logit[t] -= presence_penalty` if token t has appeared at all (0 = no penalty, 1 = subtract 1 from logit). This is a simpler binary version of frequency penalty.

---

## 13. Max Tokens & Stop Sequences {#max-stop}

### Max Tokens

**Definition:** Hard cap on the maximum number of tokens the model will generate in one response.

```python
max_tokens=20   → "The transformer architecture was introduced in 2017 by Google researchers in the..."
max_tokens=50   → "...paper Attention Is All You Need. It replaced RNNs with self-attention mechanisms,
                   enabling much faster training."
```

**Important:** `1 word ≈ 1.3 tokens` on average. `"ChatGPT"` = 3 tokens.
Token ≠ Word. Plan your `max_tokens` accordingly.

**Practical values:**
- Short answers / classifications: 50–100
- Paragraphs: 200–500
- Long essays / code: 1000–4000
- Large context work: up to 128k+

### Stop Sequences

**Definition:** Custom strings that signal the model to **stop generating immediately** when encountered. Essential for structured output.

```python
# Code generation: stop after code block ends
stop=["```", "def "]

# Q&A system: stop before next turn
stop=["\nUser:"]

# JSON output: stop after JSON closes
stop=["}\n"]
```

**Why this is powerful:**
- Prevents the model from generating more content than needed
- Enforces boundaries in structured multi-turn conversations
- Guarantees clean extraction of generated content

**Combine with max_tokens for full output control:**
```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[...],
    max_tokens=200,          # hard cap
    stop=["</answer>"]       # also stop on this string
)
```

---

## 14. Parameter Cheat Sheet {#cheatsheet}

| Parameter | Range | Default | Creative | Factual | Description |
|---|---|---|---|---|---|
| `temperature` | 0.0–2.0 | 1.0 | **1.2** | **0.3** | Scales randomness of output distribution |
| `top_k` | 1–1000+ | 50 | **80** | **10** | Keeps only k highest-probability tokens |
| `top_p` | 0.0–1.0 | 1.0 | **0.95** | **0.7** | Nucleus sampling: cumulative prob cutoff |
| `repetition_penalty` | 1.0–2.0 | 1.0 | **1.2** | **1.1** | Reduces logits for already-seen tokens |
| `frequency_penalty` | 0.0–2.0 | 0.0 | **0.4** | **0.1** | Proportional penalty based on token count |
| `max_tokens` | 1–128k+ | Model default | **1024** | **256** | Hard cap on generated token count |

> **Golden rule:** Never use top_k and top_p together at extreme values — they compound and may produce degenerate output. Use one or the other, then tune temperature.

---

## 15. Use-Case Configurations {#usecases}

### Factual Q&A
```python
# Goal: accuracy, consistency, no hallucination
temperature=0.3,
top_p=0.7,
max_tokens=256
```
Low temperature = near-deterministic. Tight top_p = focused nucleus. Short max_tokens = no rambling.

### Creative Writing
```python
# Goal: variety, creativity, engaging prose
temperature=1.2,
top_p=0.95,
repetition_penalty=1.2
```
Higher temperature = creative choices. Wide nucleus = diverse vocabulary. Repetition penalty = avoids loops.

### Code Generation
```python
# Goal: syntactically correct, deterministic, stop cleanly
temperature=0.2,
top_k=20,
stop=["```"]
```
Very low temperature = deterministic code. Low top_k = only syntactically likely tokens. Stop sequence = clean code block.

### Brainstorming
```python
# Goal: lots of diverse ideas, don't filter too much
temperature=1.5,
top_k=80,
max_tokens=1024
```
High temperature = wild ideas. High top_k = broad vocabulary. High max_tokens = let it run.

### Numerical / Attribution Analysis (from Q&A)
```python
# Goal: stability, accuracy, structured output
temperature=0.0,
top_p=1.0,
max_tokens=256
# Even simpler: temp=0, top_p=1 for maximum stability
```

---

## 16. Interview Questions {#interview}

**Q1: What is greedy decoding and what is its main weakness?**
> **Answer:** Greedy decoding always selects the single most probable token at each step: `ŵ_t = argmax P(w | w_{<t})`. It is fast, simple, and deterministic. Its main weakness is myopia — it makes locally optimal choices that can lead to globally suboptimal sequences. By committing to the highest-probability token at each step, it can lock into a path that has lower total sequence probability than an alternative path that started with a slightly lower-probability token. Example: choosing "nice" (40%) over "going" (35%) even though "going to be amazing tomorrow" would have been the globally better sequence.

**Q2: How does beam search improve on greedy decoding? Is it optimal?**
> **Answer:** Beam search keeps track of k candidate sequences (beams) at each step rather than just one. At each step, it extends all k beams with all possible next tokens, scores the resulting extended sequences by total probability, and retains only the top k. This "multi-path exploration" allows beam search to consider paths that greedy would discard early. However, beam search is still a heuristic — it is NOT guaranteed to find the globally optimal sequence, because the number of all possible sequences is exponential. In practice, beam search with k=5–10 significantly outperforms greedy and is widely used in machine translation and speech recognition.

**Q3: Explain temperature scaling in your own words. What does setting T=0.3 vs T=1.8 do?**
> **Answer:** Temperature divides the logits by τ before applying softmax: `P(token) = softmax(logits / τ)`. T=0.3 (cold): dividing by a small number amplifies differences between logits, making the softmax distribution very "peaked" — the highest-probability token gets even higher probability, effectively making the model almost deterministic. Output is focused but potentially repetitive. T=1.8 (hot): dividing by a large number compresses logits together, making the softmax distribution more "flat" — lower-probability tokens have a better chance, producing creative and varied output, but potentially incoherent.

**Q4: What is the difference between top-k and top-p sampling?**
> **Answer:** Top-k: keep only the k most probable tokens at each step, discard the rest, renormalize, sample. The cutoff is a fixed count — always exactly k candidates regardless of the model's confidence. Top-p (nucleus): keep the smallest set of tokens whose cumulative probability ≥ p, discard the rest, renormalize, sample. The cutoff is a dynamic probability mass — when the model is very confident (one token has 99% probability), p=0.9 gives a small nucleus of 1-2 tokens; when uncertain (flat distribution), p=0.9 gives a larger nucleus. Top-p adapts to confidence level; top-k does not.

**Q5: Explain the "quality vs diversity" trade-off in text generation.**
> **Answer:** Methods that emphasize high-probability tokens (greedy, beam search, low temperature, tight top-p) produce coherent, factually accurate outputs — but these tend to be repetitive and "safe," lacking creativity. Methods that give more weight to mid-probability tokens (high temperature, wide top-k, large top-p) produce more creative and varied outputs — but these can be incoherent, factually incorrect, or grammatically odd. All decoding parameters control where on this spectrum the model operates. The choice depends entirely on the use case: factual QA needs quality; creative writing needs diversity.

**Q6: What is the difference between repetition penalty and frequency penalty?**
> **Answer:** Both reduce the probability of already-seen tokens. Repetition penalty: binary — if a token appeared *at all* in the context, divide its logit by the penalty value (flat penalty regardless of how often it appeared). Frequency penalty: proportional — subtract `freq_penalty × count(token)` from the logit, so tokens that appeared 5 times get penalized 5× more than tokens that appeared once. In practice: repetition penalty is a blunt tool (any appearance triggers it), while frequency penalty is graduated and more nuanced.

**Q7: What does setting top_k = 50 do to the output length?**
> **Answer:** Nothing — top_k and output length are completely independent. top_k = 50 means at every single token generation step, the model looks at all ~50,000 possible next tokens, keeps the 50 most probable, and randomly samples one from those 50. This filtering happens fresh at every step. To get a 1,000-token response, you'd generate 1,000 tokens, each time selecting from a top-50 pool. Output length is controlled by max_tokens separately.

**Q8: When would you NOT use beam search in a production chatbot?**
> **Answer:** For conversation applications, beam search tends to produce text that is safe, repetitive, and unnatural because: (1) it always finds the statistically "most likely" sequence, which is often generic; (2) it's deterministic — every user gets the same response to the same question; (3) it lacks the natural variability of human conversation. Modern chatbots (GPT-4, Claude) use temperature + top-p sampling to produce varied, natural-sounding text. Beam search is better suited for structured output tasks like machine translation (where you want the single "best" translation) or speech recognition.

**Q9: What is a logit and how does it relate to temperature?**
> **Answer:** A logit is the raw, unnormalized score the model gives each possible next token before softmax. For example: `"Paris" → logit 5.2, "London" → logit 3.1`. After softmax these become probabilities: `Paris → 0.85, London → 0.15`. Temperature modifies logits before softmax by dividing them: `softmax(logits / T)`. T<1 amplifies the spread (peaked distribution), T>1 compresses it (flat distribution), T=1 leaves it unchanged.

**Q10: Why should you avoid combining top_k and top_p at extreme values?**
> **Answer:** The two filters compound — they both restrict the candidate pool. If you set top_k=5 (keep 5 tokens) AND top_p=0.1 (keep tokens summing to only 10% probability), the effective candidate pool might shrink to 1-2 tokens, producing extremely repetitive, degenerate output. The model has essentially no choices. Best practice: use either top_k or top_p, not both at extreme values. Then adjust temperature to tune creativity.

---

## 17. Key Learning Thoughts {#learning-thoughts}

> **Thought 1 — The Model Is Fixed; Decoding is Variable**
> People often say "GPT-4 gave me a different answer this time." The model's weights didn't change. The decoding parameters did — or temperature introduced stochasticity. When debugging inconsistent outputs, the first question is: what are my temperature/top-p settings?

> **Thought 2 — Logits Are the Universal Currency**
> Every single decoding parameter — temperature, top-k, top-p, repetition penalty, frequency penalty — operates on logits. If you understand logits, you can reason from first principles about what any parameter combination will do. High logit = high probability. Dividing all logits by T flattens/sharpens the distribution. Zeroing out logits (top-k/top-p) removes tokens from consideration.

> **Thought 3 — Greedy Is Not Dumb, Just Myopic**
> Greedy decoding is used in many production systems where latency matters and the task is well-defined (autocomplete, short responses). The issue is specifically with long-form generation where early choices compound. For short outputs (< 20 tokens), greedy often produces great results.

> **Thought 4 — Top-p Is Almost Always Better Than Top-k for Chat**
> Because top-p adapts to model confidence, it handles both high-confidence and low-confidence situations gracefully. Top-k with a fixed k can include garbage tokens when the model is confident, and may over-restrict when the model is uncertain. This is why most major APIs default to top-p.

> **Thought 5 — Temperature Is Your Most Important Knob**
> More than any other parameter, temperature shapes output character. Start by setting temperature for your use case (0.3 for factual, 1.0 for balanced, 1.2 for creative), then fine-tune with top-p. Don't reach for penalties and stop sequences before you've tuned temperature and top-p first.

> **Thought 6 — Stop Sequences Are Underused in Production**
> Most beginners use `max_tokens` to control output length, but `stop` sequences give you *semantic* control — "stop when the structure is complete." For code generation (`stop=["```"]`), JSON generation (`stop=["}\n"]`), or dialogue systems (`stop=["\nUser:"]`), stop sequences are cleaner and more reliable than max_tokens alone.

> **Thought 7 — Beam Search and Top-k Are NOT the Same Thing**
> This is the most common confusion in this section. Beam search tracks k *sequences* and is deterministic. Top-k sampling filters k *tokens* per step and is stochastic. They share the letter "k" but operate in fundamentally different ways. In interviews, clearly distinguish: beam search is in the *search* family (with greedy); top-k is in the *sampling* family (with temperature and top-p).

> **Thought 8 — Document Your Configs**
> In production AI systems, always version-control your prompt + decoding parameters together. A prompt that works beautifully at temp=0.7 may produce garbage at temp=1.5. When a change in model or API version breaks your system, the first thing to check is whether default parameter values changed.

---

## Quick Reference Card

```
FACTUAL WORK (QA, extraction, data analysis):
  temperature=0.2–0.3, top_p=0.7, top_k=10, max_tokens=256

BALANCED (general assistant, summarization):
  temperature=0.7, top_p=0.9, max_tokens=512

CREATIVE (stories, copy, ideation):
  temperature=1.0–1.2, top_p=0.95, repetition_penalty=1.2, max_tokens=1024

CODE GENERATION:
  temperature=0.1–0.2, top_k=20, stop=["```"], max_tokens=2000

BRAINSTORMING:
  temperature=1.3–1.5, top_k=80, max_tokens=1024

NEVER DO:
  top_k=5 + top_p=0.1  (compounds to near-zero candidates)
  temperature=2.0 in production (incoherent outputs)
  max_tokens without stop sequences for structured output
```
