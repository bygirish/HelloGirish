# Section 2: Self-Attention Mechanism

> **Learning Goal:** Deeply understand the Query-Key-Value framework, how self-attention computes contextual representations, and why it is the mathematical engine behind Transformers.

---

## 2.1 The Self-Attention Idea

### What is Self-Attention?

Self-attention is **encoder-encoder** (or decoder-decoder) attention where **each word attends to every other word within the same sequence**.

The key property:
> Every token can directly interact with every other token — no sequential propagation needed.

This solves three things simultaneously:
- **Long-range dependencies**: "it" and "animal" separated by 5 words connect in one step
- **Parallel computation**: all attention scores computed as one matrix operation
- **Dynamic context understanding**: the representation of each word changes based on its sentence context

### The Motivation — "The animal didn't cross the street because it was too tired."

What does **"it"** refer to — the animal or the street?

A human understands immediately: "it" = the animal (because animals get tired, not streets).

How does a model resolve this? Self-attention:

| Step | Operation | Result |
|------|-----------|--------|
| **1. Compare** | Compare "it" with every other word | Compute similarity scores |
| **2. Weight** | Assign higher weight to "animal" | Weight("animal") = 0.70, Weight("tired") = 0.20 |
| **3. Understand** | Use weights to build contextual meaning | "it" = blend of "animal" (70%) + "tired" (20%) + others |

After self-attention, the representation of "it" is no longer just "it" — it is **"something like animal, with context of being tired"**.

This is the fundamental power: tokens don't have static meanings, they have **contextual meanings**.

---

## 2.2 Attention vs Self-Attention — The Critical Distinction

| Property | Attention (Bahdanau) | Self-Attention (Transformer) |
|----------|---------------------|------------------------------|
| **Who attends to whom?** | Decoder → Encoder (cross-sequence) | Each word → every other word in same sequence (within-sequence) |
| **Used in** | Seq2Seq translation (RNN-based) | Transformers — BERT, GPT, T5 |
| **Replaces** | Fixed context vector bottleneck | Sequential LSTM entirely |
| **Parallelizable?** | Partially (encoder parallel, decoder sequential) | Fully — no recurrence at all |
| **Key formula** | c_t = Σ αᵢ·hᵢ, αᵢ = softmax(FF(s_{t-1}, hᵢ)) | Attention(Q,K,V) = softmax(QKᵀ/√d)·V |

> **Memory anchor:** Bahdanau attention asks "which source words matter for this output word?" — Self-attention asks "which words in this sentence matter for understanding each word?"

---

## 2.3 The Transformer Architecture — Where Self-Attention Lives

```
OUTPUT PROBABILITIES
        ↑
    [Softmax]
    [Linear]
        ↑
  ┌─────────────────────────────────────────────────────────┐
  │  DECODER (×6 layers)                                    │
  │  [Add & Norm] → [Feed Forward] → [Add & Norm]           │
  │  [Multi-Head Attention] → [Add & Norm]                  │
  │  [Masked Multi-Head Attention]                          │
  │  [Output Embedding + Positional Encoding]               │
  └─────────────────────────────────────────────────────────┘
            ↑ (encoder output)
  ┌─────────────────────────────────────────────────────────┐
  │  ENCODER (×6 layers)                                    │
  │  [Add & Norm] → [Feed Forward] → [Add & Norm]           │
  │  [Multi-Head Self-Attention]                            │
  │  [Input Embedding + Positional Encoding]                │
  └─────────────────────────────────────────────────────────┘
            ↑
        INPUTS
```

Self-attention is the core of **every encoder layer**. Each token builds a richer, context-aware representation by attending to all other tokens.

---

## 2.4 Query, Key, Value — The Mathematical Framework

### The Intuition: A Database Lookup

Think of self-attention as a **soft, differentiable database lookup**:

- **Query (Q)**: What I am looking for — "it asks: who am I?"
- **Key (K)**: What I represent / advertise — "animal says: I'm a noun, a living thing"
- **Value (V)**: The actual content I carry — "what information I provide"

In a hard lookup (Python dict): one key matches exactly → return that value.
In attention (soft lookup): every key partially matches → return a **weighted sum** of all values.

### The Analogy

Imagine searching on YouTube:
- Your **search query** = Q ("cat videos")
- Each video's **title/tags** = K ("funny cats", "dog tricks", "cat behaviour")
- Each video's actual **content** = V
- The **relevance score** of each video = Q · K
- The **result** = weighted blend of all videos by relevance

Self-attention does exactly this, but for every token simultaneously.

### Concrete Example — "Query, Key, Value Step by Step"

Sentence: "The animal didn't cross the street because it was too tired."

For the word **"it"**:
| Word | Key meaning (K) | Score (Q·K) | Weight (after softmax) |
|------|----------------|-------------|----------------------|
| animal | noun, living thing | 5.0 | **0.70** |
| tired | adjective | 3.0 | 0.20 |
| street | noun, location | 1.0 | 0.05 |
| others | ... | small | ~0 |

**Output for "it":**
```
Output_it = 0.70 × V_animal + 0.20 × V_tired + 0.05 × V_street + ...
```

After self-attention, "it" becomes: "something like animal, with context of being tired."

---

## 2.5 The (Self) Attention Mechanism — Attention as a Fuzzy Hashtable

### Hard Hashtable (exact lookup):
```
q → k₀ → v₀   (only ONE key matches)
    k₁ → v₁
    k₂ → v₂
```

### Soft Attention (fuzzy lookup):
```
q → k₀ → v₀  (weight 0.05)
    k₁ → v₁  (weight 0.70)  ← highest match
    k₂ → v₂  (weight 0.20)
    ...
Output = weighted sum of ALL values
```

Key insight: **Each query matches each key to varying degrees. The output is a sum of values weighted by the query-key match.**

This is why attention is differentiable — the "lookup" is a smooth function of the input, allowing gradients to flow through it during training.

---

## 2.6 Recipe for Self-Attention in the Transformer Encoder

For a single attention head, here are the 4 steps for each word xᵢ:

### Step 1: Compute Q, K, V for each word

Each word xᵢ produces three vectors via learned weight matrices:

```
qᵢ = W^Q · xᵢ     (query vector — what am I looking for?)
kᵢ = W^K · xᵢ     (key vector — what do I represent?)
vᵢ = W^V · xᵢ     (value vector — what content do I carry?)
```

W^Q, W^K, W^V are **learned parameters** (trained by gradient descent). They are shared across all positions.

### Step 2: Compute Attention Scores

For each pair (i, j), compute the raw score:
```
eᵢⱼ = qᵢ · kⱼ     (dot product — measures how much i should attend to j)
```

This produces an n×n matrix E (for a sequence of n tokens).

### Step 3: Normalize with Softmax

```
αᵢⱼ = softmax(eᵢⱼ) = exp(eᵢⱼ) / Σₖ exp(eᵢₖ)
```

Now αᵢⱼ ≥ 0 and Σⱼ αᵢⱼ = 1 for each row i. These are **attention weights** — probability distributions over all positions.

### Step 4: Weighted Sum of Values

```
Outputᵢ = Σⱼ αᵢⱼ · vⱼ
```

The output for token i is a weighted blend of all value vectors. The next layer's input: x_{l+1} = Output_l.

---

## 2.7 Vectorized Self-Attention — The Matrix Form

For an entire sequence at once (all n tokens stacked as rows):

```
X ∈ R^{n×d}   (n tokens, d-dimensional embeddings)

Q = X · W^Q   (shape: n × d_k)
K = X · W^K   (shape: n × d_k)
V = X · W^V   (shape: n × d_v)

Score matrix E = Q · Kᵀ   (shape: n × n)
                              ↑ element (i,j) = how much token i attends to token j

Attention matrix A = softmax(E)   (shape: n × n, rows sum to 1)

Output = A · V   (shape: n × d_v)
```

**The complete formula:**
```
Output = softmax(QKᵀ) · V
```

This is a **single matrix multiplication** — computable in O(1) parallel steps on a GPU.

---

## 2.8 Tiny Self-Attention — Numerical Walkthrough

Let's compute self-attention for 2 tokens: x₁ = "animal", x₂ = "it"
- Embedding dimension d = 2
- For simplicity: W^Q = W^K = I (identity), W^V given below

**Input:**
```
X = [[1, 0],   ← "animal" embedding
     [0, 1]]   ← "it" embedding
```

**Step 1: Q = K = X (since W^Q = W^K = I)**
```
Q = K = [[1, 0],
         [0, 1]]
```

**Step 2: Scores = Q · Kᵀ**
```
Scores = [[1,0],[0,1]] · [[1,0],[0,1]]ᵀ = [[1,0],[0,1]]
```

**Step 3: Softmax each row**
```
softmax([1, 0]) = [e¹/(e¹+e⁰), e⁰/(e¹+e⁰)] = [2.718/3.718, 1/3.718] = [0.73, 0.27]
softmax([0, 1]) = [0.27, 0.73]

A = [[0.73, 0.27],   ← "animal" attends 73% to itself, 27% to "it"
     [0.27, 0.73]]   ← "it" attends 73% to itself, 27% to "animal"
```

**W^V = [[1,1],[1,0]], so V = X · W^V:**
```
V = [[1,0],[0,1]] · [[1,1],[1,0]] = [[1,1],[1,0]]
```

**Step 4: Output = A · V**
```
"animal": 0.73·[1,1] + 0.27·[1,0] = [1.0, 0.73]
"it":     0.27·[1,1] + 0.73·[1,0] = [1.0, 0.27]
```

**Interpretation:**
- "it" has non-zero 2nd component → influenced by "animal" (which carries meaning [1,1])
- The output representation of "it" is now **partially "animal"** — coreference implicitly resolved

> This tiny example shows the core mechanism. In practice, d=512–1024, Q≠K≠X, and 8–16 parallel heads run simultaneously.

---

## Key Formulas Summary

| Step | Operation | Formula |
|------|-----------|---------|
| Project | Compute Q, K, V | Q=XW^Q, K=XW^K, V=XW^V |
| Score | Dot product similarity | E = QKᵀ |
| Normalize | Softmax over each row | A = softmax(E) |
| Aggregate | Weighted sum of values | Output = AV |
| **Full formula** | | **Output = softmax(QKᵀ)V** |

---

## Interview Questions

### Conceptual

**Q1. What is the Query-Key-Value framework in self-attention? Explain with an analogy.**

> **Answer:** In self-attention, each token generates three vectors: Query (what it's looking for), Key (what it advertises/represents), and Value (what content it provides). The analogy is a search engine: the Query is your search string, each document's title/tags is its Key, and the document's content is its Value. Unlike a hard lookup that returns one result, attention returns a weighted sum of all Values, where weights are determined by how well each Key matches the Query. This "soft" lookup is differentiable, enabling end-to-end gradient training.

**Q2. What is the difference between self-attention and cross-attention?**

> **Answer:** In self-attention, Q, K, and V all come from the same sequence — each token attends to all others in its own sequence. In cross-attention (as in the Transformer decoder), Q comes from one sequence (decoder) while K and V come from a different sequence (encoder). Self-attention builds intra-sequence context; cross-attention aligns two different sequences. BERT uses only self-attention; GPT uses only (masked) self-attention; the original Transformer encoder-decoder uses both.

**Q3. Why is self-attention computed as a dot product between Q and K? Why not other similarity measures?**

> **Answer:** The dot product is computationally efficient (parallelizable as matrix multiplication), differentiable, and captures the degree of alignment between two vectors (cos-similarity when normalized). It can be computed for all (i,j) pairs simultaneously as QKᵀ in O(n²d) time. Alternatives like cosine similarity require normalization overhead; additive attention (used in Bahdanau) requires a feedforward network and is slower. The dot product also has a natural geometric interpretation: high value = vectors point in similar direction = semantically related.

**Q4. What does the attention weight matrix tell you? What would a "perfect" attention pattern look like for coreference resolution?**

> **Answer:** The attention weight matrix A (shape n×n) tells you how much each token (row) attends to each other token (column). Row i, column j = how much token i focuses on token j when building its representation. For coreference: "The animal... it was tired" — a perfect attention pattern would have the row for "it" showing high weight (~0.8+) on "animal" and low weight on all other tokens. This is exactly what a well-trained self-attention head learns to do.

**Q5. Why is self-attention O(n²) in sequence length? Is this a problem?**

> **Answer:** For n tokens, we compute an n×n score matrix (every pair of tokens), so the time and memory complexity is O(n²d). For n=512 (BERT), this is manageable. For n=100,000 (long documents), it becomes prohibitive. This is why research has produced sparse attention variants (Longformer: O(n·w) with window attention), linear attention approximations (Performer, Linformer), and sliding window approaches. GPT-4 and Claude support very long contexts through various engineering tricks around this fundamental quadratic bottleneck.

---

### Deeper / Advanced

**Q6. Explain why softmax is used in the attention formula rather than just using raw dot products.**

> **Answer:** Raw dot products can be arbitrarily large or negative. Using them directly as weights would not guarantee they sum to 1 (no probability interpretation) and could cause gradient issues. Softmax: (1) normalizes weights to sum to 1 (attention as probability distribution), (2) amplifies the largest score and suppresses small ones (focus), (3) is differentiable everywhere (gradients flow through), (4) ensures non-negative weights. Without softmax, the attention output would not have a clean "blending" interpretation.

**Q7. What happens to the Q, K, V weight matrices during training? What do they learn?**

> **Answer:** W^Q, W^K, W^V start randomly initialized and are learned via gradient descent along with all other parameters. They learn to project token embeddings into a subspace where semantically/syntactically related tokens have high Q·K dot products. For example, a "coreference" head might learn projections where pronouns ("it", "they") have high Q similarity to their antecedents ("animal", "people"). W^V learns which information to aggregate. In multi-head attention, different heads learn different W^Q, W^K, W^V matrices, capturing different relationship types (syntax, semantics, position).

---

## Learning Thoughts

> **The "it" example is your anchor:** Whenever you need to explain self-attention, return to this example. "The animal... it was too tired." Every concept maps onto it — Q (it asks who I am), K (animal says I'm a noun), V (animal's content), softmax (70% animal, 20% tired), output (it becomes animal-like).

> **Attention is a re-weighting mechanism:** The input embeddings are static snapshots of word identity. The output of self-attention is a dynamic, context-aware re-weighting. The weights are not fixed — they are computed fresh for every input sentence. "Bank" in "river bank" vs "bank account" gets completely different attention patterns.

> **The fuzzy hashtable is the best mental model for interviews:** Hard database = one key matches. Attention = every key matches to different degrees, output is a weighted blend. This analogy is instantly understandable and technically precise.

> **Vectorization is what makes it fast:** The entire self-attention computation for n tokens collapses into 3 matrix multiplications (Q=XW^Q, K=XW^K, V=XW^V), one outer product (QKᵀ), one softmax, and one more matrix multiply (AV). GPUs are extraordinarily good at this. This is why Transformers, despite being more complex, train faster than RNNs on modern hardware.

> **Critical mental shift:** In RNNs, context is accumulated step-by-step (sequential). In self-attention, context is computed globally in one shot (parallel). This isn't just a speed improvement — it's a fundamentally different representational strategy.
