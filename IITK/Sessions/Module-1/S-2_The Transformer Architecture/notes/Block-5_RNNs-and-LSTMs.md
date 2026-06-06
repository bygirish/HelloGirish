# Block 5: Recurrent Neural Networks (RNNs) & LSTMs
## Sequential Data, Vanilla RNN, BPTT, Gradient Problems, LSTM

> **Session:** Lecture 2 — The Transformer Architecture  
> **Topics covered:** 22–28

---

## Learning Roadmap for This Block

```
Why sequential data needs special treatment
→ Vanilla RNN (structure + math)
→ Training via BPTT
→ Vanishing/Exploding Gradient Problem
→ LSTM: The solution (Forget Gate, Input Gate, Cell State, Output Gate)
→ LSTM Mathematics (all 6 equations)
→ LSTM Worked Example (step-by-step numbers)
```

This block answers: **How do neural networks process sequences where order matters — text, speech, time-series?**

---

## Topic 22: What is Sequential Data?

### The Core Problem

Standard feedforward networks and CNNs treat every input independently. There is no concept of "what came before." This is fine for images (a pixel's meaning doesn't depend on the order it was processed). But for sequences, order is everything.

### Examples of Sequential Data

| Type | Example | Why order matters |
|---|---|---|
| Text / NLP | "The cat sat on the mat" | Each word's meaning depends on context |
| Speech / Audio | Waveform samples over time | Phonemes depend on surrounding sounds |
| Time Series | Stock prices, EEG signals | Past values predict future values |
| DNA | Nucleotide sequences | Patterns across positions encode genes |
| Video | Sequence of frames | Motion is captured by frame-to-frame changes |

### The Key Insight

Consider: "The cat sat on the ___"

To predict "mat," a network needs to remember "cat," "sat," "on," and "the" — words that appeared earlier in the sequence. A feedforward network processes each word independently, with no memory of previous words. **We need a model with memory.**

---

## Topic 23: Vanilla RNN

### Architecture

An RNN processes sequences one element at a time, maintaining a **hidden state** `hₜ` that acts as a "memory" — it carries information from previous time steps.

```
h₀ → [RNN cell] → h₁ → [RNN cell] → h₂ → [RNN cell] → h₃
      ↑                  ↑                  ↑                  ↑
      x₀                 x₁                 x₂                 x₃
```

The same RNN cell (same weights) is applied at every time step — this is **weight sharing across time**.

### The Update Rule

```
hₜ = tanh( Wₕ · hₜ₋₁  +  Wₓ · xₜ  +  b )
```

Where:
- `xₜ` = input at time step t
- `hₜ₋₁` = hidden state from previous step (the "memory")
- `hₜ` = new hidden state (updated memory)
- `Wₕ` = recurrent weight matrix (hidden → hidden)
- `Wₓ` = input weight matrix (input → hidden)
- `b` = bias
- `tanh` = activation function (squashes to [−1, 1])

### Output Computation

```
yₜ = g( Wᵧ · hₜ )
```

Where `g` is an output activation (e.g., Softmax for classification).

### Loss Function for Sequences

For sequence tasks (like language modeling), the loss is summed across all time steps:

```
L = −Σₜ  log P(yₜ | x₁, x₂, ..., xₜ)
```

The model predicts each next token given all previous tokens.

### Why tanh Instead of Sigmoid for Hidden State?

- tanh outputs ∈ [−1, 1], centered at 0 → zero-mean activations → better gradient flow
- Sigmoid outputs ∈ [0, 1], not centered → less ideal for hidden states
- In practice, tanh still saturates for large values, but is better than sigmoid for RNNs

### Unrolling the RNN

An RNN can be "unrolled" in time to visualize it as a very deep feedforward network:

```
Input:  x₀        x₁        x₂        x₃
         │         │         │         │
h₀ → [W,b] → h₁ → [W,b] → h₂ → [W,b] → h₃ → [W,b] → h₄
              ↓              ↓              ↓
             y₁             y₂             y₃
```

The same weights [W, b] appear at every step — this is what makes the gradient problem so severe.

---

## Topic 24: Training RNNs — Backpropagation Through Time (BPTT)

### What is BPTT?

To train an RNN, we "unroll" it for T time steps and apply standard backpropagation through the resulting deep network. The gradient flows backward through each time step.

```
Forward:  x₀ → h₁ → h₂ → ... → hₜ → Loss
Backward: Loss → ∂L/∂hₜ → ∂L/∂hₜ₋₁ → ... → ∂L/∂h₁ → ∂L/∂W
```

### The Gradient Chain

At each step, the gradient is multiplied by the Jacobian of the recurrent step:

```
∂L/∂h₁ = ∂L/∂hₜ × ∏ₖ₌₁ᵗ⁻¹ (∂hₖ₊₁/∂hₖ)
```

This product of Jacobians (one per time step) is what creates the gradient problem.

### Truncated BPTT

In practice, BPTT is expensive for long sequences (T = 1000 steps). A common solution is **Truncated BPTT**: unroll only k steps at a time (e.g., k=20), compute gradients for those k steps, then move the window forward.

---

## Topic 25: Vanishing & Exploding Gradient Problem

### The Math Behind the Problem

At each time step, the hidden state update involves multiplying by:

```
∂hₜ/∂hₜ₋₁ = Wₕ × diag(tanh'(Wₕ · hₜ₋₁ + Wₓ · xₜ))
```

The tanh derivative: `tanh'(z) = 1 − tanh²(z) ∈ [0, 1]`

So at each step, we multiply by a value ≤ 1 (and often much less).

**After T steps:**

```
∂L/∂h₁ ∝ (something)^T
```

### Vanishing Gradient

If `|∂hₜ/∂hₜ₋₁| < 1`, the product shrinks **exponentially** with T.

For T=50 steps and factor=0.9: `0.9^50 ≈ 0.005` — the gradient has essentially vanished.

**Effect:** The network can't learn long-range dependencies. Words/events more than ~10 steps ago are forgotten — gradients from those time steps are too small to update the weights.

### Exploding Gradient

If `|∂hₜ/∂hₜ₋₁| > 1`, the product grows **exponentially** with T.

For T=50 steps and factor=1.1: `1.1^50 ≈ 117` — the gradient explodes, causing NaN weights.

**Fix for exploding gradients: Gradient Clipping**

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

If the gradient norm exceeds `max_norm`, scale all gradients down so the norm equals `max_norm`. Simple but effective.

### Why Vanishing is Worse Than Exploding

Exploding gradients are obvious (NaN) and easy to fix (clip). Vanishing gradients are silent — the model trains without errors but silently fails to learn long-range dependencies. This is the harder and more fundamental problem.

### The Solution: LSTM

---

## Topic 26: LSTM — Long Short-Term Memory

### Motivation

Hochreiter & Schmidhuber (1997) proposed LSTM to solve the vanishing gradient problem. The key insight:

> Add a **dedicated long-term memory channel** (the cell state Cₜ) that information can flow through with minimal modification — like a conveyor belt that carries information across many time steps.

### The Four Gates

LSTM replaces the single tanh update of vanilla RNN with **four gating mechanisms**:

```
           ┌──────────────────────────────────────────────────────────┐
           │             Cell State Cₜ   (Memory Highway)             │
           │  Cₜ₋₁ → [ × fₜ ] → [ + (iₜ × C̃ₜ) ] → Cₜ              │
           └──────────────────────────────────────────────────────────┘
               ↑            ↑                ↑
           Forget Gate   Input Gate     Output Gate
               fₜ            iₜ             oₜ
                                            ↓
                                           hₜ = oₜ ⊙ tanh(Cₜ)
```

**Gate 1 — Forget Gate (fₜ):**
> "What fraction of the previous cell state should I erase?"

**Gate 2 — Input Gate (iₜ):**
> "What new information should I write into the cell state?"

**Gate 3 — Cell State Update:**
> Combine: erase old info (forget) + write new info (input)

**Gate 4 — Output Gate (oₜ):**
> "What part of the cell state should I output as the hidden state?"

### Intuition: The Cell State as a Conveyor Belt

Imagine the cell state as a conveyor belt running through the sequence. Items (memories) can be:
- **Removed** by the forget gate (items fall off the belt)
- **Added** by the input gate (items are placed on the belt)
- **Read** by the output gate (items are inspected before passing forward)

Information can travel across 100+ time steps with minimal degradation — this is what solves the vanishing gradient problem.

---

## Topic 27: LSTM Full Mathematics

### The Six LSTM Equations

**Notation:** `[hₜ₋₁, xₜ]` = concatenation of previous hidden state and current input

**1. Forget Gate:**
```
fₜ = σ( Wf · [hₜ₋₁, xₜ] + bf )
```
- Output ∈ [0, 1] (Sigmoid)
- 0 = forget everything from previous cell state
- 1 = keep everything

**2. Input Gate:**
```
iₜ = σ( Wi · [hₜ₋₁, xₜ] + bi )
```
- Output ∈ [0, 1] (Sigmoid)
- Controls how much new information to write

**3. Candidate Cell Values:**
```
C̃ₜ = tanh( Wc · [hₜ₋₁, xₜ] + bc )
```
- Output ∈ [−1, 1] (tanh)
- New candidate values to potentially add to cell state

**4. Cell State Update:**
```
Cₜ = fₜ ⊙ Cₜ₋₁  +  iₜ ⊙ C̃ₜ
```
- `⊙` = element-wise multiplication (Hadamard product)
- Erase: multiply old cell state by forget gate (fₜ ∈ [0,1])
- Write: add new information scaled by input gate

**5. Output Gate:**
```
oₜ = σ( Wo · [hₜ₋₁, xₜ] + bo )
```
- Controls what to output from the cell state

**6. Hidden State:**
```
hₜ = oₜ ⊙ tanh( Cₜ )
```
- tanh squashes cell state to [−1, 1]
- Output gate selects which parts to expose

### Why Does LSTM Solve Vanishing Gradients?

The gradient of the cell state update rule:
```
∂Cₜ/∂Cₜ₋₁ = fₜ
```

This is just a multiplication by fₜ — no matrix multiplication, no tanh squashing during backprop through the cell state. The forget gate values are ∈ (0, 1), but they are learned — the network can learn to keep fₜ ≈ 1 for important information, allowing gradients to flow unchanged.

Compare to vanilla RNN where gradients must multiply through tanh and Wₕ at every step.

---

## Topic 28: LSTM Worked Example (Step-by-Step)

**Setup from lecture:**
```
Previous hidden state: hₜ₋₁ = 0.5
Current input:         xₜ   = 1.0
Previous cell state:   Cₜ₋₁ = 0.3
All weights = 0.5, all biases = 0
```

**Step 1: Forget Gate**
```
fₜ = σ( 0.5 × [0.5 + 1.0] ) = σ( 0.5 × 1.5 ) = σ(0.75) ≈ 0.679
```
Interpretation: Keep 67.9% of the previous cell state.

**Step 2: Input Gate**
```
iₜ = σ( 0.5 × [0.5 + 1.0] ) = σ(0.75) ≈ 0.679
```
Interpretation: Write 67.9% of the candidate new values.

**Step 3: Candidate Cell Values**
```
C̃ₜ = tanh( 0.5 × [0.5 + 1.0] ) = tanh(0.75) ≈ 0.635
```

**Step 4: Cell State Update**
```
Cₜ = fₜ ⊙ Cₜ₋₁  +  iₜ ⊙ C̃ₜ
   = 0.679 × 0.3  +  0.679 × 0.635
   = 0.204        +  0.431
   = 0.635
```

**Step 5: Output Gate**
```
oₜ = σ( 0.5 × [0.5 + 1.0] ) = σ(0.75) ≈ 0.679
```

**Step 6: New Hidden State**
```
hₜ = oₜ ⊙ tanh(Cₜ)
   = 0.679 × tanh(0.635)
   = 0.679 × 0.561
   = 0.381
```

### Full Summary Table

| Step | Computation | Value |
|---|---|---|
| Forget gate (fₜ) | σ(0.75) | 0.679 |
| Input gate (iₜ) | σ(0.75) | 0.679 |
| Candidate (C̃ₜ) | tanh(0.75) | 0.635 |
| Cell state (Cₜ) | 0.679×0.3 + 0.679×0.635 | 0.635 |
| Output gate (oₜ) | σ(0.75) | 0.679 |
| Hidden state (hₜ) | 0.679 × tanh(0.635) | 0.381 |

*Note: All gates happen to be the same (0.679) because all weights and the input are identical — in a real trained LSTM they would all be different, learned from data.*

### GRU — A Simplified LSTM

The **Gated Recurrent Unit (GRU)** by Cho et al. (2014) simplifies LSTM by merging the forget and input gates into a single "update gate," eliminating the separate cell state:

```
Update gate: zₜ = σ( Wz · [hₜ₋₁, xₜ] )
Reset gate:  rₜ = σ( Wr · [hₜ₋₁, xₜ] )
Candidate:   h̃ₜ = tanh( W · [rₜ ⊙ hₜ₋₁, xₜ] )
New hidden:  hₜ = (1 − zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ
```

GRU has fewer parameters than LSTM → faster training. In practice, LSTM and GRU perform similarly — the choice is often empirical.

---

## Interview Questions — Block 5

**Q1: What is the vanishing gradient problem in RNNs? How does LSTM solve it?**

> In vanilla RNNs, gradients must flow backward through T time steps by multiplying through the Jacobian of the hidden state update at each step. Since tanh saturates (derivative ≤ 0.25) and weights are shared, this product shrinks exponentially — gradients from early time steps approach zero and those weights stop learning. LSTM solves this with a separate cell state (Cₜ) that has an additive (not multiplicative) update: Cₜ = fₜ⊙Cₜ₋₁ + iₜ⊙C̃ₜ. The gradient through the cell state is just ∂Cₜ/∂Cₜ₋₁ = fₜ — a single multiplication, not a chain of saturating operations. The network can learn fₜ ≈ 1 for important memories, enabling gradients to flow unchanged through long sequences.

**Q2: Explain the role of each LSTM gate.**

> Forget gate (fₜ): decides what percentage of the previous cell state to erase — a value of 0 means "forget everything," 1 means "remember everything." Input gate (iₜ): decides how much of the new candidate information to write into the cell state. Candidate cell (C̃ₜ): generates the new candidate values to potentially store. Output gate (oₜ): decides which parts of the cell state to expose as the hidden state (the output). Together they allow the LSTM to selectively read, write, and forget information over long sequences.

**Q3: What is the difference between LSTM and GRU? When would you prefer one over the other?**

> LSTM has 4 gates (forget, input, candidate, output) and a separate cell state — more expressive but more parameters. GRU has 2 gates (update, reset) and combines the cell state with the hidden state — simpler and faster to train. In practice, they perform similarly. Prefer LSTM when you have abundant data and longer sequences. Prefer GRU when training speed and parameter efficiency matter. For large-scale production systems, Transformers have largely replaced both.

**Q4: What is Backpropagation Through Time (BPTT)?**

> BPTT is the algorithm for training RNNs. The RNN is "unrolled" for T time steps, creating a T-layer deep feedforward network. Standard backpropagation is applied to this unrolled network — gradients are computed at each time step and then summed to update the shared weights. Truncated BPTT limits unrolling to k steps for efficiency.

**Q5: Why is gradient clipping used in RNN training?**

> In RNNs, gradients can explode exponentially when hidden state Jacobians have eigenvalues > 1. Gradient clipping caps the gradient norm: if ‖∇L‖ > threshold, scale all gradients by threshold/‖∇L‖. This prevents NaN weights from exploding gradients. It's applied during backward pass after computing gradients but before the optimizer step.

**Q6: What is the difference between a one-to-one, one-to-many, many-to-one, and many-to-many RNN?**

> One-to-one: standard feedforward (not RNN). One-to-many: single input, sequence output — e.g., image captioning (one image → a sentence). Many-to-one: sequence input, single output — e.g., sentiment analysis (a sentence → positive/negative). Many-to-many (same length): each input step produces an output — e.g., POS tagging. Many-to-many (different length): sequence input → sequence output of different length — e.g., machine translation (encoder-decoder architecture).

**Q7: What are the limitations of LSTMs that motivated Transformers?**

> (1) Sequential computation: LSTM processes tokens one-at-a-time, making it impossible to parallelize across sequence positions during training — very slow for long sequences. (2) Limited context: even with LSTM, the hidden state is a fixed-size vector that must compress all context — a bottleneck for very long sequences. (3) Quadratic memory growth with BPTT. (4) LSTMs still struggle with very long-range dependencies (thousands of tokens). Transformers address all of these with parallel self-attention.

---

## Key Learning Insights

> **Insight 1:** The RNN's core insight — **sharing weights across time** — is analogous to CNNs sharing weights across space. Both are applications of the same principle: if a pattern can occur anywhere (in time or space), share the parameters that detect it.

> **Insight 2:** The vanishing gradient problem is not specific to RNNs — it affects any very deep network. The difference is that in RNNs, depth equals sequence length (which can be thousands), while in feedforward networks, depth is the number of layers (typically tens to hundreds).

> **Insight 3:** LSTM's cell state is a "linear memory highway." Information can be copied unchanged through the forget gate (when fₜ ≈ 1) for many steps, allowing gradients to flow backward almost unimpeded. This is a beautiful mathematical solution to the vanishing gradient problem.

> **Insight 4:** In NLP, LSTMs were state-of-the-art from ~2015 to 2018. The Transformer (2017) began replacing them for most tasks by 2019, and by 2020-2021, LSTMs were rarely used for new NLP systems. However, for time series, signal processing, and edge devices (where sequential processing is acceptable), LSTMs remain relevant.

> **Insight 5:** The sequence of developments: Vanilla RNN → LSTM (1997) → GRU (2014) → Attention Mechanism (2015) → Transformer (2017) → BERT/GPT (2018) → ChatGPT (2022). Understanding each step reveals WHY each architecture was invented and what problem it solved.

---

## Quick Reference Cheatsheet

```
Vanilla RNN:
  hₜ = tanh( Wₕ·hₜ₋₁ + Wₓ·xₜ + b )

Problem: gradients shrink/explode exponentially over T steps

LSTM (6 equations):
  fₜ = σ( Wf·[hₜ₋₁, xₜ] + bf )          ← Forget gate
  iₜ = σ( Wi·[hₜ₋₁, xₜ] + bi )          ← Input gate
  C̃ₜ = tanh( Wc·[hₜ₋₁, xₜ] + bc )       ← Candidate
  Cₜ = fₜ⊙Cₜ₋₁ + iₜ⊙C̃ₜ                ← Cell state (memory highway)
  oₜ = σ( Wo·[hₜ₋₁, xₜ] + bo )          ← Output gate
  hₜ = oₜ ⊙ tanh(Cₜ)                    ← Hidden state

Key: ∂Cₜ/∂Cₜ₋₁ = fₜ (not tanh+Wₕ) → gradients can flow

GRU: simpler (2 gates), similar performance, faster training

Exploding gradient fix: gradient clipping (clip_grad_norm_)
Vanishing gradient fix: LSTM, skip connections, attention, Transformer
```
