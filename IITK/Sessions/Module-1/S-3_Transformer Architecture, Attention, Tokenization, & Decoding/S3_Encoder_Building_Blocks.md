# Section 3: Transformer Encoder Building Blocks

> **Learning Goal:** Understand every component that surrounds self-attention in an encoder layer — why each exists, what problem it solves, and how they combine into a stable, deep, trainable architecture.

---

## 3.1 The Encoder Layer — Overview

A single Transformer encoder layer consists of two sub-layers:

```
Input (from previous layer or embedding)
        ↓
[Multi-Head Self-Attention]  ← Section 2 + 4
        ↓
[Add & Norm]                 ← Residual + LayerNorm
        ↓
[Feed-Forward Network]       ← Per-token MLP
        ↓
[Add & Norm]                 ← Residual + LayerNorm
        ↓
Output (to next layer)
```

This block is **repeated 6 times** in the original Transformer (N=6). Modern models use 12, 24, 48, or 96 layers.

The supporting components — FFN, Residual Connections, LayerNorm, Scaled Attention, Positional Encoding — are what make deep stacking of self-attention **actually work** in practice.

---

## 3.2 Feed-Forward Layer (FFN)

### The Problem Self-Attention Alone Creates

After self-attention, each token representation is a **weighted average of value vectors**:
```
Outputᵢ = Σⱼ αᵢⱼ · vⱼ
```

This is a **linear** operation (a re-averaging). No non-linearity means the entire attention layer is equivalent to a linear transformation — regardless of how many stacked layers you have.

> Without non-linearity, the entire stack collapses to a single linear transformation. You'd lose the representational power of depth.

### The Solution: Feed-Forward Network After Attention

After attention, apply a 2-layer MLP to each token **independently**:

```
FFN(xᵢ) = ReLU(xᵢW₁ + b₁)W₂ + b₂
```

Where:
- W₁ ∈ R^{d×4d}: **expands** the representation from d → 4d (e.g., 512 → 2048)
- ReLU: non-linear activation, zeros out negative signals
- W₂ ∈ R^{4d×d}: **compresses** back from 4d → d

### The 4-Step Processing

| Step | Operation | What It Does |
|------|-----------|--------------|
| **1. Expand (W₁)** | d → 4d dimensions | Creates a large private workspace for complex transformations |
| **2. ReLU** | Zero out negatives | Introduces non-linearity; without this, the network is just linear |
| **3. Compress (W₂)** | 4d → d dimensions | Refines and projects back to model dimension |
| **4. Zero cross-token interaction** | Applied independently to each token | Unlike attention, FFN processes each token identically and independently |

### Key Architectural Insight

> **Attention = tokens talk to each other** | **FFN = each token thinks privately in 4d space**

The FFN parameters (W₁, W₂) account for approximately **⅔ of all Transformer parameters**. This is where most of the model's "knowledge" is stored — attention finds relationships, FFN transforms and stores information.

**Attention learns WHAT is relevant; FFN learns HOW to transform it.**

### GPT-2 Parameter Scale
For GPT-2 (d_model=768, d_ff=3072, 12 layers):
```
FFN params per layer = 768×3072 + 3072×768 = 4,718,592
Total FFN params = 4,718,592 × 12 = 56.6M (out of ~117M total)
```

---

## 3.3 Training Trick #1: Residual Connections

### The Problem

Deep networks (6+ layers) suffer from a counterintuitive problem: they are **surprisingly bad at learning the identity function**.

If a layer learns nothing useful, the ideal behavior is to pass input through unchanged (identity: x → x). But deep networks often corrupt or distort representations as they flow through layers.

Additionally, gradients must flow backwards through many layers. In deep networks, gradients get multiplied by many small values before reaching early layers — the vanishing gradient problem returns even without RNNs.

### The Solution: Add the Input Directly

```
x_l = F(x_{l-1}) + x_{l-1}
```

Where F is any sub-layer (self-attention or FFN). The input to each sub-layer is **added back** to the output.

```
      x (raw embedding)
      │
      ├────────────────────────────────────┐
      ↓                                    │ (skip connection)
[Self-Attention]                           │
      ↓                                    │
   F(x)                                    │
      ↓                                    │
   F(x) + x  ←────────────────────────────┘
```

### Why This Helps

**Problem 1 (Identity):** Now the network only needs to learn **residuals** (corrections to the identity), not the full transformation. Learning F(x) = 0 (correction ≈ zero) is much easier than learning F(x) = x (full identity). This is the origin of the name "residual."

**Problem 2 (Gradients):** The gradient through a residual connection is:
```
∂L/∂x_{l-1} = ∂L/∂x_l · (∂F/∂x_{l-1} + 1)
```

The "+1" term ensures that **gradients always have a direct path back** to early layers, even if ∂F/∂x_{l-1} is small. This is the "gradient highway."

**Real-world consequence:** Without residual connections, Transformers deeper than ~4 layers would be difficult to train. With them, 100+ layer models (GPT-4) train stably.

> **ResNets (He et al., 2015)** introduced this for computer vision. The Transformer adopted it directly.

---

## 3.4 Training Trick #2: Layer Normalization

### The Problem: Internal Covariate Shift

Training deep networks is difficult because the distribution of each layer's inputs keeps changing as the layers before it are updated. The weights of layer l were optimized for a certain distribution of inputs — if those inputs shift, layer l must re-adapt.

This is called **internal covariate shift**. It makes training slow and unstable, particularly for deep models.

### The Solution: Normalize Each Layer's Activations

For each token representation x ∈ R^d (a vector of d dimensions), normalize to zero mean and unit variance:

```
μˡ = (1/H) Σᵢ aᵢˡ           (mean of the d activations)
σˡ = √((1/H) Σᵢ (aᵢˡ - μˡ)²)  (standard deviation)

x' = (x - μ) / (σ + ε)       (normalize)
```

Where ε is a small constant (1e-8) to prevent division by zero.

**After normalization:** Each token's d-dimensional representation has mean ≈ 0 and std ≈ 1.

### Layer Norm vs Batch Norm

| Property | Batch Norm | Layer Norm |
|----------|-----------|-----------|
| **Normalizes over** | All samples in batch, per feature | All features in one sample |
| **Works with batch size = 1?** | No | Yes |
| **Works with variable-length sequences?** | Poorly | Yes |
| **Training vs inference behavior** | Different (running stats) | Same |
| **Used in** | CNNs, vision models | Transformers, RNNs |

Layer Norm is preferred for Transformers because sequences have variable length and batch sizes can be small.

### Where It's Applied

LayerNorm is applied **after** each sub-layer (the "Add & Norm"):
```
x → [Self-Attention] → Add → LayerNorm → [FFN] → Add → LayerNorm
```

This is called **Post-LN** (used in original Transformer). Modern models often use **Pre-LN** (normalize before each sub-layer) for more stable training of very deep networks.

---

## 3.5 Training Trick #3: Scaled Dot Product Attention

### Why Scaling is Needed

After LayerNorm, the mean of activations is ~0 and variance is ~1. But there's still a problem with the attention scores.

The dot product Q·K = Σᵢ qᵢkᵢ. For a vector of dimension d_k, if each component has variance 1:
```
Var(q·k) = Σᵢ Var(qᵢ·kᵢ) = d_k · Var(q) · Var(k) = d_k · 1 · 1 = d_k
```

**Problem:** The variance of the dot product grows linearly with d_k. For d_k = 64 (typical), the dot products have standard deviation 8 — they can be very large (e.g., 50, -30).

**Why this breaks softmax:** When softmax receives very large inputs, it produces near-one-hot distributions:
```
softmax([50, 1, 2]) ≈ [1.0, 0.0, 0.0]
```

The gradient of softmax near extreme values is ≈ 0 (it's saturated). This causes **vanishing gradients through the attention layer**.

### The Fix: Divide by √d_k

```
Output = softmax(QKᵀ / √d_k) · V
```

Dividing by √d_k brings the variance back to 1:
```
Var(q·k / √d_k) = Var(q·k) / d_k = d_k / d_k = 1
```

Now softmax operates on inputs with mean 0, variance 1 — it produces meaningful, non-saturated distributions.

### The Full Updated Formula

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V
```

This is called **Scaled Dot-Product Attention**. It is the complete, production-ready formula used in all Transformers.

| Without scaling | With scaling |
|-----------------|-------------|
| Large dot products | Controlled dot products |
| Saturated softmax → near one-hot | Smooth softmax → distributed attention |
| Vanishing gradients | Healthy gradients |

---

## 3.6 Positional Encoding — Injecting Order into Attention

### The Problem: Self-Attention is Permutation-Invariant

Self-attention computes:
```
Attention(X) = softmax(XW^Q (XW^K)ᵀ / √d_k) · XW^V
```

If you shuffle the rows of X (reorder the tokens), the output rows shuffle in the same way — but **each row's content is identical** to what it would have been with any other ordering.

**In other words:** "The cat chased the mouse" and "The mouse chased the cat" produce the same attention pattern (just with rows reordered). The model cannot distinguish word order!

This is called **permutation invariance** — a critical flaw for language understanding.

### The Solution: Add Positional Encodings

Add a **position vector** p_i to each token embedding before the first encoder layer:
```
ṽᵢ = v̂ᵢ + pᵢ     (token embedding + position signal)
```

Where pᵢ ∈ R^d is a vector encoding the position i.

After adding positional encodings:
- q_i = W^Q(v̂_i + p_i) = W^Q·v̂_i + W^Q·p_i  (position affects queries)
- k_i = W^K(v̂_i + p_i) = W^K·v̂_i + W^K·p_i  (position affects keys)
- The dot product q_i · k_j now depends on positions i and j

### Sinusoidal Positional Encoding (Original Transformer)

```
PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
```

Where:
- **pos** = token position (0, 1, 2, ..., max_len)
- **i** = dimension index (0, 1, 2, ..., d_model/2)
- **d_model** = embedding dimension (e.g., 512)

Each dimension oscillates at a different frequency — creating a **unique fingerprint** for each position.

### Why Sinusoidal? The 4 Reasons

| Property | Explanation |
|----------|-------------|
| **Deterministic** | No training needed; positions are fixed, not learned |
| **Unique per position** | Each pos has a unique vector (like a binary count in continuous space) |
| **Relative distances** | PE(pos+k) is a linear function of PE(pos) — model can learn relative positions |
| **Generalizes** | Works beyond training sequence lengths (sin/cos have infinite range) |

### Sinusoidal PE — Intuition

Think of it as a **binary counter in continuous space**:
- Low-frequency dimensions (large 10000^(2i/d)) change slowly → encode coarse position
- High-frequency dimensions (small 10000^(2i/d)) change fast → encode fine position

This is analogous to how a clock's hour hand changes slowly and minute hand changes fast — together they uniquely identify any moment.

### Learned vs Sinusoidal Positional Encoding

| Type | How | Used in | Pros |
|------|-----|---------|------|
| Sinusoidal | Formula | Original Transformer | Generalizes to longer sequences |
| Learned | Trained embedding table | BERT, GPT | Can adapt to data statistics |
| RoPE (Rotary) | Complex rotation | LLaMA, GPT-NeoX | Better relative position |
| ALiBi | Attention bias | BLOOM, MPT | Simple, efficient |

Modern LLMs mostly use **learned positional encodings** or **RoPE**, but sinusoidal is important to understand as the foundation.

---

## 3.7 The Complete Transformer Encoder — Putting It All Together

```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn       = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.self_attn(x, x, x)
        x = self.norm1(x + self.drop(attn_out))   # Add & Norm (residual)
        ffn_out  = self.ffn(x)
        x = self.norm2(x + self.drop(ffn_out))    # Add & Norm (residual)
        return x
```

### The Complete Stack (6 Encoder Layers):

```
Token IDs → Embedding (d=512) → + Positional Encoding
                    ↓
            ┌──────────────────┐
            │  Encoder Layer 1 │  ← Self-Attn + Add&Norm + FFN + Add&Norm
            └──────────────────┘
                    ↓
            ┌──────────────────┐
            │  Encoder Layer 2 │
            └──────────────────┘
                    ↓
               ... × 6 total
                    ↓
            Context-rich token representations
```

Each encoder layer refines the representations. Early layers tend to capture syntactic patterns; later layers capture semantic relationships.

---

## Building Block Summary

| Component | Problem Solved | Formula / Rule |
|-----------|---------------|----------------|
| **FFN** | No non-linearity in self-attention | FFN(x) = ReLU(xW₁+b₁)W₂+b₂ |
| **Residual Connection** | Vanishing gradients in deep networks; identity function problem | x_l = F(x_{l-1}) + x_{l-1} |
| **Layer Normalization** | Internal covariate shift; unstable training | x' = (x-μ)/(σ+ε) |
| **Scaled Attention** | Saturated softmax from large dot products | softmax(QKᵀ/√d_k)V |
| **Positional Encoding** | Self-attention is permutation-invariant | PE(pos,2i) = sin(pos/10000^(2i/d)) |

---

## Interview Questions

### Conceptual

**Q1. Why is the Feed-Forward Network applied per-token independently? Why not across tokens?**

> **Answer:** After self-attention, each token has already aggregated information from all other tokens. The FFN's job is to transform each token's representation independently — to apply non-linear filtering and increase model capacity. Applying it across tokens would duplicate what attention already does (inter-token interaction). The independence property means W₁ and W₂ are shared across all positions but applied separately to each — making it computationally efficient. Think of it as: attention lets tokens talk to each other, FFN lets each token think privately.

**Q2. What is the purpose of residual connections, and what would happen without them?**

> **Answer:** Residual connections serve two purposes: (1) they allow gradients to flow directly to early layers via an additive skip path (solving vanishing gradients), and (2) they make it easy for the network to learn identity functions by only needing to learn corrections (residuals). Without them, a 6-layer Transformer would be difficult to train because gradients would vanish before reaching the embedding layer. Practically, residual connections are why very deep models (100+ layers) are trainable at all.

**Q3. Why is Layer Normalization used instead of Batch Normalization in Transformers?**

> **Answer:** Batch Normalization normalizes across the batch dimension — it needs a consistent batch structure and has different behavior during training vs inference (requires running statistics). For Transformers: (1) sequences have variable lengths, making batch-wise normalization complicated, (2) in some inference scenarios (real-time generation), batch size = 1 — BN would have degenerate statistics, (3) in autoregressive generation, you process one token at a time. Layer Norm normalizes across the feature dimension for each sample independently — it works the same at any batch size and sequence length.

**Q4. Why do we scale dot products by 1/√d_k? What goes wrong without it?**

> **Answer:** Without scaling, the dot product q·k has variance = d_k. For d_k=64, std=8, so dot products can easily be ±30–50. When softmax receives such large values, it produces near-one-hot distributions (saturation), and the gradient of softmax at saturation is ≈0. This kills gradient flow through the attention layer. Dividing by √d_k brings variance back to 1, keeping softmax in its non-saturated regime where gradients are healthy and attention is distributed rather than one-hot.

**Q5. Why is sinusoidal positional encoding used? Why not just use position indices (0, 1, 2, ...)?**

> **Answer:** Raw position indices would have different magnitudes at different positions, biasing the model toward later positions. Sinusoidal encoding: (1) is bounded between -1 and 1 (no magnitude bias), (2) has a unique vector for each position (distinguishable), (3) encodes relative distance — PE(pos+k) is a linear function of PE(pos), enabling the model to learn relative attention patterns, (4) generalizes to sequence lengths beyond training (sin/cos are defined everywhere). Raw indices would not generalize and would be out-of-distribution for positions longer than seen during training.

---

### Advanced

**Q6. What is Pre-LN vs Post-LN, and why does Pre-LN train more stably for very deep networks?**

> **Answer:** Post-LN (original Transformer): LayerNorm after the residual addition: `LayerNorm(x + SubLayer(x))`. Pre-LN (modern practice): LayerNorm before the sub-layer: `x + SubLayer(LayerNorm(x))`. Pre-LN ensures that each sub-layer always receives normalized input, regardless of what happened in previous layers — the gradient scale is more consistent across layers. In Post-LN very deep networks (24+ layers), early layers can receive gradients with wildly different scales, causing instability. Pre-LN is why models like GPT-2, GPT-3, and BERT-large can train stably without specialized warmup schedules.

**Q7. The FFN dimension is 4× the model dimension (d_ff = 4 × d_model). Why 4?**

> **Answer:** The 4× expansion was empirically chosen in the original Transformer paper and has proven robust across many architectures. The expansion creates a larger "workspace" for each token to undergo complex transformations. The FFN can be interpreted as a key-value memory (Geva et al., 2021): the first layer detects patterns (like key lookup), ReLU activates matching patterns, and the second layer adds the corresponding values. The 4× factor represents a practical tradeoff — larger (8×, 16×) gives slightly better performance but quadratically more parameters; smaller (2×) reduces capacity. Many modern models still use 4× (GPT-2: 4×, BERT: 4×), though some use 8/3 × (LLaMA with SwiGLU).

---

## Learning Thoughts

> **Each building block solves one specific failure mode:**
> - FFN → linear collapse (no non-linearity)
> - Residual → gradient death in deep networks
> - LayerNorm → unstable training from shifting distributions
> - Scaling → softmax saturation from high-dimensional dot products
> - PE → position blindness (permutation invariance)
> Remove any one of them and the model either fails to train or fails to represent position.

> **The FFN is underrated:** Most explanations focus on attention, but the FFN holds ~⅔ of Transformer parameters. Recent research (Geva et al., 2021) shows FFN layers act as key-value memories — they store and retrieve factual knowledge. "Paris is the capital of France" is likely stored in an FFN layer, not the attention heads.

> **Residual connections are profound:** The mathematical insight is that learning F(x) = correction is easier than learning F(x) = transformation. This is why ResNets (vision) and Transformers (NLP) both scaled to hundreds of layers — without residuals, depth ≥ ~20 layers was practically untrainable.

> **Positional encoding is an injection:** The Transformer has no inherent sense of order — it's a set function. Positional encoding literally injects sequence order into what is otherwise an order-blind architecture. Modern RoPE and ALiBi encodings are more sophisticated but serve the same fundamental purpose.

> **Think of each encoder layer as a refinement step:** Layer 1 of a 12-layer BERT might learn basic morphology and syntax. Layer 6 captures long-range syntactic structure. Layer 12 captures semantic relationships and world knowledge. This hierarchical refinement is only possible because residuals + LayerNorm allow stable training of deep stacks.
