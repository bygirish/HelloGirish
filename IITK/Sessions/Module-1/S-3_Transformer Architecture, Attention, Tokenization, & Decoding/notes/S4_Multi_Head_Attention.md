# Section 4: Multi-Head Attention

> **Learning Goal:** Understand why one attention head is insufficient, how multiple heads capture different relationship types simultaneously, and how to work through the full multi-head computation by hand.

---

## 4.1 The Problem with Single-Head Attention

A single attention head computes one Q, K, V projection — it can only capture **one type of relationship** between tokens at a time.

But language is rich with simultaneously relevant relationship types:

| Relationship Type | Example |
|------------------|---------|
| **Coreference** | "it" → "animal" |
| **Verb-Object** | "cross" → "street" |
| **Semantic similarity** | "tired" → "animal" |
| **Positional adjacency** | nearby tokens |
| **Syntactic dependency** | subject → verb |
| **Temporal** | "before" → earlier event |

A single set of W^Q, W^K, W^V cannot learn all these patterns simultaneously. The projections would be pulled in contradictory directions during training.

> **The Solution:** Run h independent attention heads in parallel, each with its own projection matrices. Each head specializes in capturing a different relationship type.

---

## 4.2 Multi-Head Attention Architecture

### The Mechanism

For h attention heads, each head i has its own learned projection matrices:
```
W^Q_i ∈ R^{d_model × d_k}
W^K_i ∈ R^{d_model × d_k}
W^V_i ∈ R^{d_model × d_v}
```

Where d_k = d_v = d_model / h (each head works in a lower-dimensional subspace).

**Per-head computation:**
```
head_i = Attention(X·W^Q_i, X·W^K_i, X·W^V_i)
       = softmax(Q_i · K_i^T / √d_k) · V_i
```

**Combining all heads:**
```
MultiHead(X) = Concat(head₁, head₂, ..., head_h) · W^O
```

Where W^O ∈ R^{h·d_v × d_model} projects the concatenated outputs back to model dimension.

### The Full Formula

```
MultiHead(X) = Concat(head₁ ... head_h) · W^O
head_i = softmax(Q_i K_i^T / √d_k) · V_i
```

### Dimensions for Original Transformer (h=8, d_model=512)

```
d_k = d_v = 512 / 8 = 64 per head

Each head: Q_i, K_i, V_i ∈ R^{n × 64}
Concat output: R^{n × 512}  (8 heads × 64 dims each)
After W^O: R^{n × 512}  (back to model dim)
```

### Architecture Diagram

```
       Input X
      /   |   \
    Q₁   K₁   V₁     Q₂   K₂   V₂     ...    Qₕ   Kₕ   Vₕ
     \    |   /         \    |   /               \    |   /
   [Scaled Attn 1]    [Scaled Attn 2]    ...   [Scaled Attn h]
          |                  |                        |
        head₁              head₂          ...       head_h
          \                  |                      /
           └──────── Concat ──────────────────────┘
                        |
                 Linear (W^O)
                        |
                   Output (d_model)
```

---

## 4.3 Why Multiple Heads? — Each Head Learns Different Relationships

The key insight: **same input X, different projection matrices W^Q_i, W^K_i, W^V_i → different attention patterns → complementary information.**

In the original Transformer paper (and confirmed by many interpretability studies), different heads learn to capture:

| Head | What it learns | Example |
|------|----------------|---------|
| H1 | Coreference | "it" → "animal" |
| H2 | Verb-Object | "cross" → "street" |
| H3 | Semantic similarity | "tired" → "animal" |
| H4 | Positional | nearby tokens |
| H5 | Syntactic subject-verb | subject → verb |
| H6 | Long-range dependency | distant related tokens |

No single head is told to do any of these — they emerge from training by gradient descent.

---

## 4.4 2-Head Attention — Concrete Numerical Calculation

Let's work through a complete 2-head attention computation.

**Setup:**
- x₁ = "animal" → embedding [1, 0]
- x₂ = "it" → embedding [0, 1]
- X = [[1,0],[0,1]]
- d_model = 2, h = 2, d_k = 1 per head

### Head 1: Coreference (it → animal)

**Projection matrices:**
```
WQ₁ = [[1],[1]]    WK₁ = [[1],[0]]    WV₁ = [[2],[0]]
```

**Step 1: Compute Q, K, V**
```
Q₁ = X · WQ₁ = [[1,0],[0,1]] · [[1],[1]] = [[1],[1]]ᵀ
K₁ = X · WK₁ = [[1,0],[0,1]] · [[1],[0]] = [[1],[0]]ᵀ
V₁ = X · WV₁ = [[1,0],[0,1]] · [[2],[0]] = [[2],[0]]ᵀ
```

**Step 2: Score Matrix = Q₁ · K₁ᵀ**
```
[[1],[1]] · [[1, 0]] = [[1·1, 1·0],[1·1, 1·0]] = [[1, 0],[1, 0]]
```

Both rows = [1, 0] → both "animal" and "it" have the same attention distribution.

**Step 3: Softmax**
```
softmax([1, 0]) = [2.718/3.718, 1/3.718] = [0.73, 0.27]
A₁ = [[0.73, 0.27],[0.73, 0.27]]
```

**Step 4: Output₁ = A₁ · V₁**
```
V₁ = [[2],[0]]
animal: 0.73×2 + 0.27×0 = 1.46
it:     0.73×2 + 0.27×0 = 1.46
O₁ = [[1.46],[1.46]]
```

**Interpretation:** Both "animal" and "it" attend 73% to "animal" → Head 1 enforces coreference. Both tokens carry equal "animal" signal after this head.

---

### Head 2: Self/Context (it → itself)

**Projection matrices:**
```
WQ₂ = [[0],[1]]    WK₂ = [[0],[1]]    WV₂ = [[0],[3]]
```

**Step 1: Compute Q, K, V**
```
Q₂ = X · WQ₂ = [[1,0],[0,1]] · [[0],[1]] = [[0],[1]]ᵀ
K₂ = X · WK₂ = [[0],[1]]ᵀ   (same as Q₂)
V₂ = X · WV₂ = [[0],[3]]ᵀ
```

**Step 2: Score Matrix = Q₂ · K₂ᵀ**
```
Q₂ = [[0],[1]], K₂ᵀ = [[0, 1]]
Scores = [[0×0, 0×1],[1×0, 1×1]] = [[0,0],[0,1]]
```

**Step 3: Softmax**
```
Row 1 (animal): softmax([0,0]) = [0.50, 0.50]
Row 2 (it):     softmax([0,1]) = [0.27, 0.73]
A₂ = [[0.50, 0.50],[0.27, 0.73]]
```

**Step 4: Output₂ = A₂ · V₂**
```
V₂ = [[0],[3]]
animal: 0.50×0 + 0.50×3 = 1.50
it:     0.27×0 + 0.73×3 = 2.19
O₂ = [[1.50],[2.19]]
```

**Interpretation:** "it" attends 73% to itself → Head 2 captures self/context identity. "it" has a larger value (2.19) because it attends mostly to itself.

---

### Combining Heads: Concatenation & Final Output

```
O₁ = [[1.46],[1.46]]      O₂ = [[1.50],[2.19]]

Concat(O₁, O₂):
"animal" → [1.46, 1.50]
"it"     → [1.46, 2.19]
```

Multiply by W^O (output projection) for final representation.

**Final interpretations:**

| Token | H1 Output | H1 Meaning | H2 Output | H2 Meaning | Final Representation |
|-------|-----------|-----------|-----------|-----------|---------------------|
| "animal" | 1.46 | Attends to itself (main noun) | 1.50 | Equal mix → self-aware | Knows it is the coreference target |
| "it" | 1.46 | Attends to animal (coreference) | 2.19 | Attends to itself (identity) | "it" = animal + context — resolved! |

> The two heads together give "it" both its coreference identity (it = animal) and its distinct self-context. This is richer than either head alone.

---

## 4.5 Head 1 vs Head 2 — Side-by-Side Comparison

| Property | HEAD 1 (Coreference) | HEAD 2 (Self/Context) |
|----------|---------------------|----------------------|
| **WQ** | [1,1]ᵀ — both dims equally | [0,1]ᵀ — 2nd dim only |
| **WK** | [1,0]ᵀ — 1st dim only | [0,1]ᵀ — 2nd dim only |
| **WV** | [2,0]ᵀ — amplify 1st dim | [0,3]ᵀ — amplify 2nd dim |
| **Score pattern** | Both rows [1,0] → identical | Rows differ [0,0] vs [0,1] |
| **Attn matrix** | Both rows same [0.73, 0.27] | Rows differ [0.50,0.50] vs [0.27,0.73] |
| **Output pattern** | Both tokens equal | Tokens differ |
| **What it learns** | Coreference: it → animal | Self-awareness: it → itself |

**Key insight:** Identical projection matrices produce identical attention rows (all tokens attend the same way). Differential projections produce token-specific attention (each token attends differently). Both patterns are useful — together they are powerful.

---

## 4.6 Multi-Head Attention — PyTorch Implementation

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        self.h = n_heads
        self.d_k = d_model // n_heads   # 64 per head

        self.W_q = nn.Linear(d_model, d_model)  # projects to h×d_k total
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)  # output projection

    def split_heads(self, x, B):
        # x: (B, seq_len, d_model) → (B, h, seq_len, d_k)
        return x.view(B, -1, self.h, self.d_k).transpose(1, 2)

    def forward(self, Q, K, V):
        B = Q.size(0)
        Q = self.split_heads(self.W_q(Q), B)   # (B, h, T, d_k)
        K = self.split_heads(self.W_k(K), B)
        V = self.split_heads(self.W_v(V), B)

        scores = (Q @ K.transpose(-2,-1)) / self.d_k**0.5  # (B, h, T, T)
        out = F.softmax(scores, dim=-1) @ V                 # (B, h, T, d_k)

        # Concat heads: (B, h, T, d_k) → (B, T, d_model)
        out = out.transpose(1,2).contiguous().view(B, -1, self.h * self.d_k)
        return self.W_o(out)   # project back to d_model
```

**Key implementation details:**
- All heads' Q, K, V are computed with **one linear layer** (W_q, W_k, W_v are d_model×d_model — logically h×d_k heads)
- `split_heads` reshapes to make h attention operations run in parallel
- Final `contiguous().view()` concatenates all head outputs
- W_o is the output projection W^O

---

## 4.7 The Complete Transformer Stack (Encoder Only)

```python
class Transformer(nn.Module):
    def __init__(self, vocab_size=50000, d_model=512,
                 n_heads=8, n_layers=6, d_ff=2048, max_len=512):
        super().__init__()
        self.embedding = TransformerEmbedding(vocab_size, d_model, max_len)
        self.encoder   = nn.ModuleList(
            [TransformerEncoderLayer(d_model, n_heads, d_ff)
             for _ in range(n_layers)])
        self.fc_out    = nn.Linear(d_model, vocab_size)

    def forward(self, src):
        x = self.embedding(src)         # token + positional embedding
        for layer in self.encoder:
            x = layer(x)               # 6× self-attention + FFN
        return self.fc_out(x)           # project to vocab size
```

**GPT-2 style parameters:**
```python
model = Transformer(vocab_size=50257, d_model=768,
                    n_heads=12, n_layers=12, d_ff=3072)
params = sum(p.numel() for p in model.parameters())
# ≈ 117M (GPT-2 small)
```

**The full encoding pipeline:**
```
Input Tokens → + Pos Encoding → Self-Attention ×6 → FFN ×6 → Output Logits
```

---

## Key Formulas Summary

| Concept | Formula |
|---------|---------|
| Single head | `head_i = softmax(Q_i K_i^T / √d_k) V_i` |
| Multi-head | `MultiHead(X) = Concat(head₁...head_h) · W^O` |
| Per-head dim | `d_k = d_v = d_model / h` |
| Total params (MHA) | `4 × d_model² = 4 × W^Q, W^K, W^V, W^O` |

---

## Interview Questions

### Conceptual

**Q1. Why does multi-head attention use d_k = d_model/h per head, keeping total computation similar to single-head?**

> **Answer:** If each head used the full d_model dimension, multi-head attention would cost h times more computation. By using d_k = d_model/h per head, the total dimension across all h heads equals d_model — the same as a single head operating in full d_model space. The total matrix multiplication cost is similar, but the h parallel heads can capture h different relationship types. The information is split across heads (specialization) rather than all heads doing the same thing in full dimension.

**Q2. What is W^O in multi-head attention, and why is it needed?**

> **Answer:** W^O is the output projection matrix applied after concatenating all head outputs. It's necessary because: (1) the concatenation produces h·d_v = d_model dimensional vectors, and W^O projects back to d_model for the next layer; (2) more importantly, W^O allows the model to learn how to **combine and mix** information from different heads. Each head captured a different relationship type; W^O learns a weighted combination that is most useful for the next layer. Without W^O, you'd just be concatenating independent head outputs with no learned integration.

**Q3. In practice, what kinds of patterns do different attention heads learn?**

> **Answer:** Interpretability research (Voita et al., 2019; Clark et al., 2019) shows heads specialize in: (1) **positional heads** — attend to adjacent tokens (nearby context), (2) **syntactic heads** — capture subject-verb, noun-modifier dependencies, (3) **coreference heads** — link pronouns to their antecedents, (4) **rare word heads** — give high attention to rare/important words, (5) **separator heads** — attend heavily to [CLS], [SEP] tokens. Interestingly, ~10-20% of heads in BERT can be pruned without significant performance loss — many heads are redundant.

**Q4. Can multi-head attention be viewed as learning multiple "aspects" of the input simultaneously?**

> **Answer:** Yes, exactly. Each head projects the input into a different subspace and computes attention in that subspace. Head 1 might project "animal" and "it" into a subspace where noun-pronoun pairs are similar → coreference. Head 2 might project them into a subspace where sequential position is encoded → positional patterns. By learning different W^Q_i, W^K_i projections, each head literally sees the same input through a different "lens." The final W^O output projection combines these different perspectives into one rich representation.

**Q5. What is the time and space complexity of multi-head attention? How does it scale?**

> **Answer:** For sequence length n, model dimension d, h heads: Time: O(n²d) — the n×n score matrix requires O(n²) dot products, each of dimension d_k = d/h. Space: O(n²·h) = O(n²d/d_k) for storing all h attention matrices. The bottleneck is the n² term — for n=512, n²=262,144 which is manageable; for n=100,000 (long documents), n²=10B which is not. This is why sparse attention (Longformer), linear attention (Performer), and sliding window approaches exist for long-context models.

---

### Advanced

**Q6. How would you verify that different attention heads are actually learning different things?**

> **Answer:** Several methods: (1) **Attention visualization** — plot the n×n attention matrix for each head on a specific sentence; heads with similar matrices are doing similar things. (2) **Probing classifiers** — train a small linear classifier on the attention weights of head i to predict a linguistic property (syntax, coreference); if it works, that head encodes that property. (3) **Ablation** — zero out one head at a time and measure task performance; a head that causes large performance drops when removed encodes important information. (4) **Attention entropy** — heads with low entropy (focused attention) are more interpretable; high entropy heads may be attending to everything and contribute less specific information.

**Q7. Why does the Transformer use h=8 heads in the original paper? Is there an optimal number?**

> **Answer:** h=8 was empirically chosen and remained robust across tasks. The paper tried h=1, 4, 8, 16 and found 8 worked best for their translation task. More heads ≠ better: (1) diminishing returns — many heads learn redundant patterns, (2) per-head dimension shrinks (d_k = 512/8 = 64; at h=64, d_k=8 — too small for expressive projections), (3) more parameters in W^O. Modern models use h=12 (BERT-base), h=16 (BERT-large), h=32 (GPT-3 175B). The optimal number depends on d_model — a common heuristic is d_k ≥ 32 (so h ≤ d_model/32).

---

## Learning Thoughts

> **The heads-as-specialists analogy:** Think of multi-head attention as a team of specialists all reading the same document simultaneously. Specialist 1 is a linguist looking for grammatical dependencies. Specialist 2 is a logician looking for coreference. Specialist 3 is looking for topic relationships. Their individual analyses are then combined into one comprehensive understanding. No single specialist could do all jobs at once — but together they cover everything.

> **W^O is the "conference room":** Each specialist (head) prepares their analysis independently. W^O is the meeting where their findings are integrated. It learns how to weight and combine the different perspectives. This is why W^O is as important as the heads themselves.

> **The numerical walkthrough is your interview superpower:** Being able to work through the 2-head example (A, K, V → scores → softmax → output → concat) by hand demonstrates deep understanding. Most candidates explain attention conceptually; few can actually compute it. Practice this walkthrough until it becomes automatic.

> **Key insight from the example:** Head 1 produced identical rows in the attention matrix (both tokens attend the same way) → group-level coreference signal. Head 2 produced different rows (tokens attend differently) → individual identity signal. The same mechanism, different projections, completely different behaviors. This is the power of the projection matrices — they are what the model actually learns.

> **Connection to BERT/GPT:** BERT uses 12 heads in each of 12 layers = 144 total attention heads. Each develops its own specialty. The richness of BERT's representations comes from these 144 parallel views of each sentence, refined through 12 layers. This is impossible with a single-head model.
