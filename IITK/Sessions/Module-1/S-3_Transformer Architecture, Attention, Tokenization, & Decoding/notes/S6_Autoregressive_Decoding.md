# Section 6: Autoregressive Decoding / Generation

> **Learning Goal:** Understand how the Transformer decoder generates text token-by-token, the mechanics of the decoding loop, and the different strategies for selecting the next token (greedy, beam search, sampling).

---

## 6.1 What is Autoregressive Generation?

**Autoregressive** means: generate one token at a time, where each new token is conditioned on all previously generated tokens.

```
P(y₁, y₂, ..., yₙ | x) = P(y₁|x) × P(y₂|y₁,x) × P(y₃|y₁,y₂,x) × ...
```

This is the **chain rule of probability** applied to sequence generation:
- Token 1 depends on: source x
- Token 2 depends on: source x + token 1
- Token 3 depends on: source x + tokens 1,2
- Token n depends on: source x + all previous tokens

The decoder implements this by:
1. Feeding the source through the encoder (once)
2. Generating the target left-to-right, one token per step

---

## 6.2 The Decoder Loop — Step by Step

### The Full Algorithm

```python
def generate(model, src, max_len=50, bos=1, eos=2):
    # Step 1: Encode source ONCE
    enc_out = model.encoder(src)   # fixed throughout generation

    # Step 2: Start with <BOS> (beginning of sequence) token
    tgt = torch.tensor([[bos]])

    for _ in range(max_len):
        # Step 3: Build causal mask for current target length
        mask = causal_mask(tgt.size(1))

        # Step 4: Forward pass through decoder
        logits = model.decoder(tgt, enc_out, mask)[:, -1, :]  # last token's logits

        # Step 5: Select next token (greedy: take argmax)
        next_tok = logits.argmax(dim=-1).unsqueeze(0)

        # Step 6: Append to sequence
        tgt = torch.cat([tgt, next_tok], dim=-1)

        # Step 7: Check stop condition
        if next_tok.item() == eos:
            break

    return tgt   # everything between <BOS> and <EOS>
```

### The 6-Stage Loop Explained

| Stage | Operation | Details |
|-------|-----------|---------|
| **Start Token** | Feed `<BOS>` as first decoder input | `<BOS>` = "beginning of sequence" marker tells decoder to start generating |
| **Forward Pass** | Masked Self-Attn → Cross-Attn → FFN | Decoder processes current sequence through all 6 decoder layers |
| **Linear + Softmax** | Project to vocab; get probability distribution | Linear layer maps d_model → vocab_size (50,000); softmax gives probabilities |
| **Sample / Argmax** | Select next token | Greedy (argmax), beam search, or temperature sampling |
| **Append & Repeat** | Append token; feed back as input | The new token becomes part of the next step's input — the sequence grows |
| **Stop Condition** | Halt at `<EOS>` or max length | `<EOS>` = end of sequence marker; generation stops when model predicts this |

### Critical Design Choice: Encoder Runs ONCE

The encoder processes the source sequence **only once** and produces `enc_out` — a fixed set of key-value pairs used by cross-attention at every decoding step.

This is computationally efficient: if you generate 50 tokens, the encoder only runs 1 time, not 50 times.

```
Encoder: ONE forward pass → enc_out (fixed)
Decoder: 50 forward passes (one per token) → each reads enc_out via cross-attention
```

---

## 6.3 Step-by-Step Generation Example: "I love you" → "Je t'aime"

### Setup
- Source: "I", "love", "you" → Encoder produces enc_out (fixed)
- Target: generate "Je", "t'aime", `<EOS>`

### Step 1: Generate First Token

```
Decoder input:  [<BOS>]
Self-attn scope: just <BOS>
Cross-attn:      <BOS> query attends to "I", "love", "you" encoder states
                 → Cross-attn focuses on "love" (most semantically relevant for "Je"?)
Linear + Softmax → probability distribution over vocab
Prediction:      "Je"
```

Append "Je" to sequence: `[<BOS>, Je]`

### Step 2: Generate Second Token

```
Decoder input:  [<BOS>, Je]
Self-attn scope: <BOS> and Je (mask 2×2)
Cross-attn:      "Je" state queries encoder
                 → Cross-attn focuses on "love"
Linear + Softmax → probability distribution
Prediction:      "t'aime"
```

Append "t'aime": `[<BOS>, Je, t'aime]`

### Step 3: Generate Third Token

```
Decoder input:  [<BOS>, Je, t'aime]
Self-attn scope: all 3 tokens (mask 3×3)
Cross-attn:      "t'aime" state queries encoder
                 → Cross-attn focuses on "you" / "love"
Prediction:      <EOS>
```

Loop ends. Output: "Je t'aime" ✓

### Why the Mask Grows Each Step

```
Step 1: Mask is 1×1 → <BOS> sees only itself
Step 2: Mask is 2×2 → Je sees <BOS> and itself
Step 3: Mask is 3×3 → t'aime sees <BOS>, Je, itself
Step n: Mask is n×n → token n sees all tokens 1...n
```

The causal mask is **rebuilt at every step** as the target sequence lengthens.

---

## 6.4 Decoding Strategies — How to Select the Next Token

The decoder outputs logits (unnormalized scores) for each vocab token. How you convert these into a chosen token determines the character of the generated text.

### Strategy 1: Greedy Decoding (Argmax)

```python
next_tok = logits.argmax(dim=-1)  # always pick highest probability
```

**Pros:** Simple, fast, deterministic
**Cons:** Locally optimal choices can lead to globally suboptimal sequences. Tends to generate repetitive, generic text.

**Example:** "The best thing about this is that it is **the best** thing about this..."
(gets stuck in loops because "the best" always looks like the highest probability next token)

### Strategy 2: Beam Search

Instead of keeping one candidate sequence, maintain **k beams** (partial sequences):

```
At each step, expand all k sequences by all vocab words,
score each extended sequence (sum of log probabilities),
keep only the top k sequences.
```

For k=3, beam search explores 3 paths simultaneously and returns the globally highest-probability complete sequence.

**Pros:** Better sequence-level optimization than greedy, produces more coherent text
**Cons:** Computationally k× more expensive; tends toward safe, generic sentences; "beam search curse" — high-probability sequences are often bland

**Used in:** Machine translation (k=4-5 is common), summarization

### Strategy 3: Temperature Sampling

```python
# Divide logits by temperature T before softmax
probs = F.softmax(logits / T, dim=-1)
next_tok = torch.multinomial(probs, num_samples=1)
```

**Temperature T controls sharpness:**

| Temperature | Effect | Use case |
|-------------|--------|---------|
| T → 0 | → greedy (deterministic) | When you need exact, reproducible output |
| T = 1.0 | Raw model probabilities | Balanced creativity/coherence |
| T > 1.0 | Flatter distribution → more random | Creative writing, diversity |
| T < 1.0 | Sharper distribution → more focused | Code generation, factual Q&A |

**Example:** Claude and ChatGPT use T≈0.7-1.0 for conversational responses.

### Strategy 4: Top-k Sampling

```python
# Keep only top k most probable tokens, redistribute probability
top_k_logits, top_k_ids = logits.topk(k)
probs = F.softmax(top_k_logits, dim=-1)
next_tok = top_k_ids[torch.multinomial(probs, 1)]
```

**Why:** Pure sampling can occasionally pick very improbable words. Top-k filtering prevents absurd selections while maintaining diversity. Typical k=50.

### Strategy 5: Nucleus (Top-p) Sampling

```python
# Keep smallest set of tokens whose cumulative probability ≥ p
sorted_probs, sorted_ids = logits.sort(descending=True)
cumulative_probs = torch.cumsum(F.softmax(sorted_probs), dim=-1)
# Keep tokens until cumulative prob ≥ p (e.g., p=0.95)
```

**Why it's better than top-k:** k is a fixed count. But sometimes top-5 words account for 99% of probability (simple prediction), sometimes top-100 words share it (complex prediction). Top-p dynamically adjusts — keeps more options when the model is uncertain, fewer when it's confident.

**Used in:** GPT-3, ChatGPT API (top_p parameter), LLaMA

### Comparison Table

| Strategy | Deterministic? | Quality | Diversity | Speed |
|----------|---------------|---------|-----------|-------|
| Greedy | Yes | Medium | None (repetitive) | Fastest |
| Beam Search | Yes | High (for translation) | Low (generic) | k× slower |
| Temperature Sampling | No | Variable | High | Fast |
| Top-k | No | Good | Medium | Fast |
| Top-p (Nucleus) | No | Best | Adaptive | Fast |

---

## 6.5 Generation in Practice

### The KV Cache Optimization

**Problem with naive generation:** At each step, the decoder runs a forward pass on the full sequence so far. For step 50, it recomputes attention over tokens 1-49 — redundant work.

**KV Cache:** Cache the Key and Value vectors from all previous steps. At step t, only compute Q, K, V for the new token, and reuse cached K, V for positions 1...t-1.

```
Without KV cache: Step t → O(t²) attention computations
With KV cache:    Step t → O(t) attention computations (only new token queries all cached KVs)
```

This reduces inference time from O(n²) to O(n) — crucial for long generation.

**KV cache memory:** For 50 tokens, d_model=512, 8 heads, 6 layers:
```
Cache size = 2 (K,V) × 50 (tokens) × 512 (dim) × 6 (layers) × 8 (heads) = ~24MB
```
For GPT-4 (~96 layers, d_model~12800), KV cache for 10K tokens ≈ ~50GB.

### Stop Conditions

| Condition | Description |
|-----------|-------------|
| **`<EOS>` token** | Model explicitly predicts end-of-sequence |
| **Max length** | Hard cap (e.g., max_new_tokens=512) — prevents infinite loops |
| **Stop sequences** | Specific token patterns that halt generation (API feature) |
| **Repetition penalty** | Reduce probability of recently generated tokens |

---

## 6.6 Generation — The Full Picture

```
Source Text
    ↓
[ENCODER — runs once]
    ↓
enc_out (K, V pairs for cross-attention)
    ↓ (fixed throughout decoding)
[DECODER LOOP]
    <BOS>
      → [Decoder Layer × 6] → predict token 1
    <BOS>, token1
      → [Decoder Layer × 6] → predict token 2
    <BOS>, token1, token2
      → [Decoder Layer × 6] → predict token 3
    ...
    until <EOS> or max_len
    ↓
Output sequence
```

---

## Interview Questions

### Conceptual

**Q1. What is autoregressive generation, and why does the Transformer generate text one token at a time?**

> **Answer:** Autoregressive generation applies the chain rule of probability: P(y₁,...,yₙ|x) = P(y₁|x)·P(y₂|y₁,x)·... Each token is generated conditioned on all previous tokens. The Transformer must generate one token at a time at inference because: (1) future tokens genuinely don't exist yet — unlike training (where masking prevents cheating), at inference you truly don't have them, (2) the probability distribution for token t depends on tokens 1...t-1, which don't exist until you generate them. This sequential dependency is fundamental and cannot be parallelized away during generation (only during training via masking).

**Q2. What is the difference between greedy decoding and beam search? When would you use each?**

> **Answer:** Greedy decoding picks the single highest-probability token at each step — fast but locally optimal (can lead to globally suboptimal sequences and repetition). Beam search maintains k candidate sequences simultaneously, always keeping the k highest-probability partial sequences at each step — finds better global solutions but is k× slower. Use greedy for: real-time generation (speed matters), when beam search shows minimal improvement. Use beam search for: machine translation (k=4-5), text summarization, tasks where sequence-level quality is critical and latency allows. Modern conversational AI (ChatGPT, Claude) uses sampling-based methods (temperature + top-p) rather than beam search for diversity and naturalness.

**Q3. What is temperature sampling and how does T affect generation?**

> **Answer:** Temperature T divides logits before softmax: probs = softmax(logits/T). At T=1: raw model probabilities. T<1 (e.g., 0.3): sharpens the distribution → high-probability tokens become even more dominant → more deterministic, focused output (good for code, facts). T>1 (e.g., 1.5): flattens the distribution → all tokens become more equally likely → more random, creative, surprising output (good for creative writing). T→0: approaches greedy decoding. T→∞: approaches uniform random sampling. API users can tune temperature: Claude/GPT default ~0.7-1.0 for conversation.

**Q4. Why is the encoder run only once during generation while the decoder runs once per token?**

> **Answer:** The encoder processes the source and produces enc_out — a static set of contextual representations for all source tokens. This source context doesn't change during generation. Cross-attention in the decoder reads these fixed K, V pairs at every step, but enc_out itself never changes. Running the encoder once is therefore sufficient and efficient. The decoder, however, must be run once per token because its output at each step depends on all previously generated tokens — it needs to process the growing target sequence to predict the next token.

**Q5. What is the KV cache, and why is it important for efficient inference?**

> **Answer:** During autoregressive generation, at step t, the decoder's self-attention attends over tokens 1...t. Without caching, you recompute K and V for tokens 1...t-1 redundantly at every step — time complexity O(t²). The KV cache stores the K and V vectors for all past tokens. At step t, you only compute K, V for the new token, then append to the cache. New Q for token t can attend over all cached K, V in O(t) time — overall generation is O(n) instead of O(n²). This is critical for long generation: generating 1000 tokens is 1000× faster with caching vs without.

---

### Advanced

**Q6. What is top-p (nucleus) sampling, and why is it generally preferred over top-k?**

> **Answer:** Top-k keeps the k most probable tokens and renormalizes; top-p keeps the smallest set of tokens whose cumulative probability ≥ p. The advantage of top-p: k is fixed regardless of probability distribution shape. When the model is confident (e.g., after "The Eiffel Tower is located in"), maybe top-3 tokens cover 99% of probability — k=50 would include many ridiculous tokens. When uncertain (e.g., after "I like"), maybe 200 tokens all have similar probability — k=50 would be too restrictive. Top-p dynamically adjusts: it uses fewer tokens when the model is confident, more when uncertain. This adaptive behavior makes top-p more robust across different prediction contexts.

**Q7. What is exposure bias, and how does it affect the gap between training and inference?**

> **Answer:** Exposure bias (Bengio et al., 2015): during training with teacher forcing, the decoder always sees ground-truth previous tokens. At inference, it sees its own predictions. If the model makes an error at position i, position i+1 onward must condition on a wrong token — a distribution shift the model has never experienced during training. This causes error accumulation, especially problematic for long sequences. Solutions: (1) Scheduled sampling — gradually replace ground-truth tokens with model predictions during training. (2) Minimum risk training — train directly on sequence-level metrics (BLEU, ROUGE). (3) Reinforcement learning (RLHF, REINFORCE) — train the model to generate good complete sequences, not just next tokens. (4) Diffusion-based non-autoregressive models — generate all tokens simultaneously without left-to-right dependency.

---

## Learning Thoughts

> **Generation is just the same forward pass, repeated:** There's no new architecture at inference. The same decoder layers run over and over, each time with one more token appended. The generation loop is pure engineering on top of the decoder architecture you already understand.

> **The mask grows, and that's the key:** At training, you mask all future positions. At inference, you never have future positions — every position you can see IS the past. So the mask is irrelevant at inference, but the autoregressive structure (left-to-right generation) remains. The training-time mask teaches the decoder to generate left-to-right; inference follows that learned behavior naturally.

> **Temperature is the most important hyperparameter you control:** Almost every LLM API exposes temperature. Understanding what it does mathematically (divides logits, reshapes the probability distribution) gives you precise control over generation quality vs diversity. For code generation: low temperature (0.2-0.4). For creative tasks: high temperature (0.8-1.2). For factual Q&A: medium (0.5-0.7).

> **KV cache = inference efficiency at scale:** Every production deployment of LLMs uses KV caching. Without it, generating a 2000-token response on a single query would require computing attention over millions of token pairs. With it, it's manageable. Understanding KV cache is essential for ML engineering interviews at companies running LLM inference.

> **The "<EOS> stops the loop" detail matters:** The model learns when to stop — it's not hardcoded. The model is trained so that <EOS> becomes high-probability when the task is complete. This is why ChatGPT knows when to stop answering. If the model never learned to predict <EOS>, the max_len hard cap would be the only stopping condition.
