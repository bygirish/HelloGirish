# Section 5: The Transformer Decoder

> **Learning Goal:** Understand the decoder's three-layer structure, why masking prevents "cheating" during training, how cross-attention bridges encoder and decoder, and how the full encoder-decoder Transformer works together.

---

## 5.1 The Transformer Decoder — Architecture Overview

The decoder is more complex than the encoder. Each decoder block has **three sub-layers** instead of two:

```
Output Probabilities
        ↑
    [Softmax]
    [Linear]         ← Project to vocab size
        ↑
┌─────────────────────────────────────────────────────────────────┐
│  DECODER BLOCK (×6 layers)                                      │
│                                                                  │
│  [Add & Norm]                                                    │
│  [Feed-Forward Network]        ← same as encoder FFN            │
│                                                                  │
│  [Add & Norm]                                                    │
│  [Multi-Head Cross-Attention]  ← Q from decoder, K/V from encoder│
│                                                                  │
│  [Add & Norm]                                                    │
│  [Masked Multi-Head Self-Attention] ← attend only to past tokens │
│                                                                  │
│  Output Embedding + Positional Encoding                          │
└─────────────────────────────────────────────────────────────────┘
        ↑ (encoder output K, V)
┌─────────────────────────────────────────────────────────────────┐
│  ENCODER (×6 layers)                                            │
└─────────────────────────────────────────────────────────────────┘
```

Each sub-layer is wrapped with:
- **Residual connection**: x + SubLayer(x)
- **Layer Normalization**: stabilizes training

Repeated N=6 times. Final: **Linear + Softmax** → token probability distribution.

### The Three Sub-Layers

| Sub-Layer | Type | Purpose |
|-----------|------|---------|
| **1. Masked Multi-Head Self-Attention** | Self-attention with causal mask | Attend to generated output so far (no future leakage) |
| **2. Multi-Head Cross-Attention** | Cross-attention (enc-dec) | Read encoder output to use source information |
| **3. Feed-Forward Network (FFN)** | Position-wise MLP | Non-linear transformation per token |

---

## 5.2 Masked Self-Attention — Preventing Future Leakage

### The Training Problem

During **training**, the decoder has access to the full target sequence (teacher forcing). For example, translating "I love you" → "Je t'aime":

```
Input to decoder:  [<BOS>, Je, t'aime]     (shifted right)
Target output:     [Je, t'aime, <EOS>]
```

Without masking, when predicting "Je" (position 1), the decoder could **peek at "t'aime"** (position 2) and trivially learn to copy it — not actually learning translation.

This is called **future leakage** or "cheating."

### The Solution: Causal Mask

Set all future positions' attention scores to **-∞** before the softmax:

```
e_ij = q_i^T · k_j    if j < i   (attend to past)
e_ij = -∞             if j ≥ i   (block future)
```

After softmax: exp(-∞) = 0 → zero attention weight → future tokens are invisible.

### The Attention Mask Matrix

For a sequence of 4 tokens [<start>, Je, t'aime, <EOS>]:

```
              <start>   Je    t'aime  <EOS>
<start>      [ attn    mask    mask    mask  ]
Je           [ attn    attn    mask    mask  ]
t'aime       [ attn    attn    attn    mask  ]
<EOS>        [ attn    attn    attn    attn  ]
```

- `attn` = score computed normally
- `mask` = set to -∞ before softmax → becomes 0 after softmax

This produces a **lower-triangular attention pattern** (causal mask).

### What "t'aime" Attends To:

```
<start>  "Je"   "t'aime"   <EOS>
  a₁=0.15  a₂=0.45  a₃=0.40   blocked!
```

- "t'aime" builds a context-aware representation using only past tokens
- It cannot see <EOS> (its future)
- This enforces **left-to-right generation order** during training

### Key Benefit: Parallel Training, Causal Output

> The mask enforces left-to-right generation during training while still allowing full parallelism across the sequence.

Without masking: decoder trains in parallel but cheats (learns nothing).
With masking: decoder trains in parallel AND genuinely learns from past context only.

This is a critical insight — it's how Transformers achieve both fast training (parallel) and causal generation (left-to-right).

### Implementation: Causal Mask

```python
def causal_mask(seq_len):
    # Lower-triangular matrix: True = allowed, False = masked
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask == 0  # True = masked out

def masked_attention(Q, K, V, mask):
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / d_k**0.5

    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))  # block future

    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V)

# Usage
mask = causal_mask(5)  # shape: (5, 5) — lower triangular
```

---

## 5.3 Cross-Attention — The Bridge Between Encoder and Decoder

### What is Cross-Attention?

Cross-attention is how the decoder **reads** the encoder's output. The decoder asks: "Given what I've generated so far, which parts of the source sequence should I focus on?"

> **Cross-attention = the decoder asks questions of the encoder**

### The Q, K, V Sources — The Critical Difference

```
Q (Query)    = from DECODER    "What information do I need for this output token?"
K (Key)      = from ENCODER    "What information does each source token represent?"
V (Value)    = from ENCODER    "What content does each source token carry?"
```

**Formula:**
```
CrossAttention(Q, K, V) = softmax(Q · K^T / √d_k) · V

Where:
  Q = X_dec · W^Q   (decoder hidden state projected)
  K = X_enc · W^K   (encoder output projected)
  V = X_enc · W^V   (encoder output projected)
```

### Concrete Example: English → French Translation

**Source:** "The cat sat on the mat."
**Target:** "Le chat s'est assis sur le tapis."

When generating "chat" (French for "cat"):
- Q = "chat" decoder state (what French token am I generating?)
- K = all English token keys (what do "the", "cat", "sat", "on", "the", "mat" represent?)
- V = all English token values (what content do they carry?)

**Attention scores for "chat":**

| English Word | Key meaning | Score | Weight |
|-------------|-------------|-------|--------|
| cat | subject noun | 5.0 | **0.70** |
| sat | past verb | 3.0 | 0.20 |
| on | preposition | 1.0 | 0.05 |
| others | ... | small | ~0 |

**Output:**
```
Output_chat = 0.70 × V_cat + 0.20 × V_sat + ...
```

"chat" output becomes: "blends cat + sat, most weight on cat (subject)" — exactly right for translation.

### Cross-Attention: 3 Steps

| Step | Operation | Formula |
|------|-----------|---------|
| **1. Similarity** | Dot product between decoder query and each encoder key | score = Q_dec · K_enc |
| **2. Softmax** | Normalize to probability distribution over source tokens | weights = softmax(scores / √d_k) |
| **3. Weighted Sum** | Blend encoder values by relevance | output = Σ(weights × V_enc) |

**Combined:**
```
CrossAttention(Q, K, V) = softmax(Q_dec K_enc^T / √d) · V_enc
```

### Cross-Attention as Matrix Operations

For source length s and target length t:

```
X_enc ∈ R^{s×d}   (encoder output: s source tokens)
X_dec ∈ R^{t×d}   (decoder states: t target tokens)

Q = X_dec · W^Q     (t × d_k)
K = X_enc · W^K     (s × d_k)
V = X_enc · W^V     (s × d_v)

Scores = Q · K^T    (t × s) ← each decoder token vs each encoder token
A = softmax(Scores / √d_k)  (t × s, rows sum to 1)
Output = A · V_enc  (t × d_v) ← decoder tokens enriched with encoder context
```

The output matrix has shape t×d (target length × model dim) — each decoder token now has context from the relevant source tokens.

### Cross-Attention vs Self-Attention

| Property | Self-Attention | Cross-Attention |
|----------|---------------|----------------|
| **Q source** | Same sequence (X·W^Q) | Decoder (X_dec·W^Q) |
| **K source** | Same sequence (X·W^K) | Encoder (X_enc·W^K) |
| **V source** | Same sequence (X·W^V) | Encoder (X_enc·W^V) |
| **Scores shape** | n×n (src len × src len) | t×s (tgt len × src len) |
| **Attn A** | Token attends to all others in same seq | Each decoder token attends to all encoder tokens |
| **Output shape** | R^{n×d} — same shape as input | R^{t×d} — decoder shape, encoder content |
| **Learns** | Token understands itself in own context | Decoder token understands itself via encoder context |
| **When to use** | Inside encoder (all tokens ↔ all) + decoder (masked) | Decoder layers that need to read the encoder |

---

## 5.4 Tiny Cross-Attention — Numerical Walkthrough

**Setup:**
- Encoder: x₁="cat" → [1,0], x₂="sat" → [0,1]
- Decoder: y₁="chat" → [1,0] (generating first French word)
- d = 2, W^Q = W^K = I, W^V = [[1,1],[1,0]]

**Step 1: Q from decoder, K and V from encoder**
```
Q = Y · W^Q = [[1,0]] (decoder query)
K = X_enc · W^K = [[1,0],[0,1]] (encoder keys)
V = X_enc · W^V = [[1,1],[1,0]] (encoder values)
```

**Step 2: Scores = Q · K^T (1×2 shape: one decoder token, two encoder tokens)**
```
[[1, 0]] · [[1, 0],[0, 1]]^T = [[1, 0]]
```

**Step 3: Softmax → A**
```
softmax([1, 0]) = [0.73, 0.27]
"chat" attends 73% to "cat", 27% to "sat"
```

**Step 4: Output = A · V_enc**
```
0.73·[1,1] + 0.27·[1,0] = [1.0, 0.73]
```

"chat" output [1.0, 0.73] → 2nd dimension high → strongly influenced by "cat" (V=[1,1]).

---

## 5.5 Complete Decoder Layer — All Three Sub-Layers

```python
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d=512, h=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn  = MaskedMultiHeadAttention(d, h)   # sub-layer 1
        self.cross_attn = MultiHeadAttention(d, h)          # sub-layer 2
        self.ffn        = FFN(d, d_ff)                      # sub-layer 3
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.norm3 = nn.LayerNorm(d)

    def forward(self, x, enc_out, mask=None):
        # Sub-layer 1: Masked self-attention on decoder sequence
        x = self.norm1(x + self.self_attn(x, x, x, mask=mask))

        # Sub-layer 2: Cross-attention (Q from decoder, K/V from encoder)
        x = self.norm2(x + self.cross_attn(x, enc_out, enc_out))

        # Sub-layer 3: Feed-forward (per-token transformation)
        x = self.norm3(x + self.ffn(x))

        return x
```

### Information Flow Through the Decoder Block

```
1. Masked Self-Attention:
   "What have I generated so far, and how does it relate to itself?"
   (builds coherent partial output representation)
        ↓
2. Cross-Attention:
   "Given my current state, what parts of the source are relevant?"
   (incorporates source information)
        ↓
3. FFN:
   "Process this combined information into a refined representation."
   (non-linear transformation)
```

---

## 5.6 The Complete Transformer Decoder — Summary

| Component | Formula / Rule | Purpose |
|-----------|---------------|---------|
| **Masked Self-Attention** | Causal mask: e_ij=-∞ for j≥i | Attend only to past tokens; prevent future leakage |
| **Cross-Attention** | Q from decoder, K/V from encoder | Dynamically read source context |
| **Feed-Forward (FFN)** | ReLU(xW₁+b₁)W₂+b₂, d_ff=4×d_model | Non-linear per-token transformation |
| **Add & Norm** | x + SubLayer(LayerNorm(x)) | Stable training, gradient flow |
| **Positional Encoding** | sin/cos vectors injected into embeddings | Order-aware attention |
| **Autoregressive Loop** | One token per step, until <EOS> | Token-by-token generation |

---

## 5.7 What Cross-Attention Can't Do

Cross-attention is powerful but has limitations:

| Limitation | Explanation |
|-----------|-------------|
| **Encoder context is fixed** | Cross-attention reads encoder states. Within the encoder, tokens were processed independently (by self-attention). Cross-attention can't make encoder tokens re-attend to each other. |
| **No position awareness** | Without positional encoding, "The dog bit the man" and "The man bit the dog" look identical to attention. |
| **Separate Encoder & Decoder required** | For classification or language modeling (GPT, BERT), you don't need an encoder-decoder pair. Cross-attention is only for seq2seq tasks. |

---

## Interview Questions

### Conceptual

**Q1. Why is masking necessary in the Transformer decoder, and how is it implemented?**

> **Answer:** During training, the decoder receives the full target sequence (teacher forcing). Without masking, when predicting position i, the model could attend to positions i+1, i+2, ... — essentially seeing the answer it's supposed to predict. This is "cheating" that prevents genuine learning. The fix: set e_ij = -∞ for j ≥ i before softmax. After softmax, exp(-∞) = 0, so future positions have exactly zero attention weight. This creates a lower-triangular attention matrix. The key benefit: masking allows full parallelism during training (all positions processed simultaneously) while preserving left-to-right causal ordering.

**Q2. What is the difference between masked self-attention and cross-attention in the decoder?**

> **Answer:** Masked self-attention: Q, K, V all come from the decoder's own sequence; it lets the decoder attend to the tokens it has generated so far (with causal mask). This builds coherent representations of the partial output. Cross-attention: Q comes from the decoder, K and V come from the encoder; it lets the decoder "read" the source sequence. The two serve fundamentally different purposes — self-attention builds within-output coherence, cross-attention aligns output generation with the source.

**Q3. Why does cross-attention use Q from the decoder but K and V from the encoder?**

> **Answer:** The decoder is "asking a question" (Q) — "what source information do I need to generate this output token?" The encoder provides "answers" (K, V) — its hidden states represent what each source token contains. Q·K gives the alignment score (how relevant is each source token to the current decoder state). V provides the actual content to blend. If Q came from the encoder and K/V from the decoder, the meaning would be inverted — the encoder asking questions of the decoder, which makes no sense for generation. The direction of Q is always "what am I looking for?" from the current generation context.

**Q4. What is teacher forcing, and why is it used? What is its drawback?**

> **Answer:** Teacher forcing: during training, feed the decoder the ground-truth previous token as input (regardless of what the decoder actually predicted). For "I love you" → "Je t'aime": at step 2, feed "Je" (true token) as input even if the decoder had predicted "La" at step 1. **Why used:** stabilizes training — without it, early errors cascade, making training slow and unstable. **Drawback:** exposure bias — at inference time, the decoder must use its own predictions (not ground truth) as input. The train-test distribution mismatch can cause error accumulation. Solutions include scheduled sampling (gradually replace ground truth with model predictions during training) and REINFORCE-based training.

**Q5. Can you have a Transformer with only a decoder (no encoder)? What models use this?**

> **Answer:** Yes — this is the decoder-only architecture used by GPT, GPT-2, GPT-3, GPT-4, LLaMA, etc. In a decoder-only model, there is no encoder and therefore no cross-attention. Only masked self-attention + FFN + LayerNorm is used. The model processes the input sequence as the "prompt" through the same decoder stack, and generates the output autoregressively. This works for language modeling because the "task" is to predict the next token given all previous tokens — exactly what masked self-attention enables. BERT is encoder-only; T5, BART, the original Transformer are encoder-decoder; GPT is decoder-only.

---

### Advanced

**Q6. In the decoder, why does cross-attention appear after masked self-attention rather than before it?**

> **Answer:** The ordering is deliberate. Masked self-attention first builds a coherent, contextually-aware representation of the output generated so far — "what have I said, and how does it cohere?" Once the decoder has a refined representation of its own state, cross-attention uses that refined state as the Query to read the encoder — "given what I've said, what source information do I need next?" If cross-attention came first, the decoder query would be based on the raw (uncontextualized) token embedding rather than the contextually-refined decoder state. The self-attention first builds a richer Q for cross-attention to work with.

**Q7. What happens in the cross-attention matrix A (shape t×s) — what does each row and column represent?**

> **Answer:** Row i of A corresponds to target (decoder) token i — it shows which source (encoder) tokens that target token is attending to. The values sum to 1 (softmax normalization). Column j shows how much target attention goes to source token j. A well-trained translation model produces a nearly diagonal cross-attention matrix for word-by-word languages (French "chat" attends mostly to English "cat"), or a permuted near-diagonal for languages with different word orders. This cross-attention matrix is actually the alignment matrix used in classical MT — the Transformer learns implicit word alignment through end-to-end training.

---

## Learning Thoughts

> **Three sub-layers = three different information needs:** Masked self-attention handles "what have I said?" Cross-attention handles "what is the source saying?" FFN handles "how do I transform this combined information?" Each sub-layer addresses a distinct question. The ordering — self → cross → FFN — is the natural flow of information.

> **The mask is training infrastructure, not generation infrastructure:** At inference time (generating one token at a time), you never need the mask — you genuinely don't have future tokens. The mask only matters during training when the full target sequence is available. This is subtle but important: the mask doesn't change what the model does architecturally, it just prevents it from accessing information that wouldn't be available at generation time.

> **Cross-attention is alignment:** In classical machine translation, a separate alignment model would be trained to find which source word corresponds to each target word. The Transformer learns this alignment from scratch, implicitly, through the cross-attention weights. The attention matrix A in cross-attention IS the alignment matrix. This is a beautiful unification.

> **The decoder is where generation happens:** The encoder builds rich representations of the input. The decoder generates the output. But all the "interesting" generation logic — what to say next, based on what's been said and what the input says — lives in the decoder's three sub-layers. Understanding the decoder is understanding how LLMs actually produce text.

> **BERT vs GPT architectural choice:** BERT uses encoder only — no masking, bidirectional self-attention. It's excellent for understanding tasks (classification, NER, QA). GPT uses decoder only — causal masking, left-to-right. It's excellent for generation tasks. The architectural choice (encoder vs decoder) directly determines the model's strengths.
