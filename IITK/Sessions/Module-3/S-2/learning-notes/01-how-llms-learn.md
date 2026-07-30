# Section 1 — How LLMs Learn (Slides 4–13)

> **Why this matters:** Every later topic — instability, forgetting, LR schedules,
> batch size — is a *consequence* of this one training loop. If you truly understand
> what happens in a single step, the rest of the lecture becomes "what can go wrong
> here, and which knob fixes it."

---

## 1.1 The big picture: training is one loop, run millions of times

Training is **a single loop executed millions of times**. Each pass over one mini-batch
nudges *every* parameter slightly downhill on the loss. The six stages of one step:

```
text ─► tokens ─► predict next token ─► softmax ─► cross-entropy loss ─► backprop ─► update ─► (repeat)
```

Nothing about this changes between pretraining and fine-tuning. Fine-tuning is the *same
loop* on a smaller, more specific dataset, usually with a smaller learning rate.

---

## 1.2 Tokenization — text becomes numbers (Slide 6)

The model never sees characters or words. Text is converted into **integer token IDs**,
then packed into **fixed-length windows of `T` tokens** — `T` is the **context length**
(e.g., 2048, 4096, 8192).

- "What training sees" = long ID sequences, chopped and packed into equal-length windows.
- Packing matters for efficiency: short examples get concatenated so no GPU cycles are
  wasted on padding.

> 💡 **Learning Thought:** The tokenizer is *frozen* during fine-tuning. If your domain
> has rare vocabulary (chemistry, code, another language), the tokenizer may shatter it
> into many sub-tokens, inflating sequence length and cost, and making the model work
> harder to learn it. This is a real-world reason some teams extend the tokenizer.

---

## 1.3 Mini-batches — how much data per step (Slide 7)

You never compute the gradient from one example or the whole dataset. Three regimes:

| Regime | Gradient quality | Speed | Verdict |
|--------|------------------|-------|---------|
| **Full-batch** (all data) | Exact gradient | One step needs the entire dataset — impossibly slow | ✗ |
| **Stochastic** (B = 1) | Wildly noisy | Very cheap per step | ✗ |
| **Mini-batch** (B = tens–thousands) | Stable **averaged** gradient | Full GPU parallelism | ✓ **sweet spot** |

The training set is **shuffled** so each mini-batch is a fresh random sample.

> 💡 **Learning Thought:** The word **"averaged"** is the key. A mini-batch gradient is
> the *average* of per-example gradients. Bigger batch → average of more samples → less
> noise. This single fact explains the entire batch-size section (§5) and why one weird
> example can wreck a small batch (§2).

---

## 1.4 Predict the next token at every position (Slide 8)

For a window of `T` tokens, the model predicts the next token **at every position
simultaneously** — position *i* predicts token *i+1*. This is *causal / autoregressive*
language modeling.

> 💡 **Learning Thought:** One sequence of length `T` yields ~`T` training signals, not
> one. That's why LLMs are so sample-efficient at the token level and why "number of
> tokens" — not "number of documents" — is the currency of training scale.

---

## 1.5 Softmax — logits become probabilities (Slide 9)

The final layer outputs a raw score (a **logit**) for *every token in the vocabulary*.
**Softmax** converts that vector of logits into a probability distribution that is
non-negative and sums to 1:

$$p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

- Large logit → large probability; the exponential makes the biggest logit dominate.
- The output is a full distribution over the vocab (e.g., 50k+ numbers), not a single guess.

---

## 1.6 Cross-entropy — scoring the prediction (Slide 10)

For each position, cross-entropy asks a single question: **how much probability did the
model put on the token that actually came next?** Then it averages that penalty over all
positions into one scalar `L`:

$$L = -\frac{1}{N}\sum_{i} \log p_i(\text{correct token})$$

- If the model gave the correct token probability 1.0 → penalty 0.
- If it gave it 0.001 → large penalty (−log of a tiny number is big).

> **Lower loss ⇔ the model assigns higher probability to the actual text.**
> This single scalar `L` is exactly what gradient descent minimizes.

> 💡 **Learning Thought:** Cross-entropy loss is measured in *nats* (or bits). `exp(L)`
> is **perplexity** — the model's "average branching factor," i.e., how many tokens it's
> effectively choosing between. A loss of 2.0 ≈ perplexity 7.4. Interviewers love this link.

---

## 1.7 Backpropagation — one gradient per parameter (Slide 11)

Backprop computes, for **every one of the billions of parameters**, the partial
derivative ∂L/∂θ — "if I nudge this weight up a hair, does `L` go up or down, and how
fast?" It does this efficiently via the chain rule, reusing intermediate results from
the forward pass (which is why activations must be stored — the source of most training
memory cost).

> 💡 **Learning Thought:** The softmax + cross-entropy pairing is not an accident. Their
> combined gradient simplifies to the beautifully clean `(predicted_prob − true_label)`.
> That clean, well-scaled gradient is a big reason this exact pairing is universal.

---

## 1.8 The update — stepping downhill (Slide 12)

The core equation of all deep learning:

$$\theta \leftarrow \theta - \eta \cdot \frac{\partial L}{\partial \theta}$$

Applied to **every parameter at once**. Three things to internalize:

1. **η — the learning rate.** How big a step to take. *Too large diverges, too small
   crawls.* Typically ~1e-4 with warmup + decay. → This one number gets its own section (§4).

2. **In practice: adaptive SGD, not plain SGD.** Plain SGD is rarely used for LLMs.
   **Adam / AdamW** rescales each parameter's step by running averages of its recent
   gradients (first moment = mean, second moment = variance). Parameters with consistent
   gradients move confidently; noisy ones move cautiously.

3. **Then loop.** Load the next mini-batch and repeat.

> 💡 **Learning Thought:** Adam keeps *two extra numbers per parameter* (the moment
> estimates). That's why the optimizer state is ~2× the model size in memory, and why
> optimizer memory — not the weights — often dominates fine-tuning cost. (This is the
> whole motivation for LoRA / PEFT, covered elsewhere.)

---

## 1.9 Millions of steps — watching the loss fall (Slide 13)

Vocabulary you will reuse for the entire lecture:

- **Step** = one mini-batch → one parameter update.
- **Epoch** = one full pass over the training set. *(Frontier LLMs often pretrain for
  only ~1 epoch on huge corpora.)*
- **When to stop** = when loss on **held-out validation data flattens.** Training past
  that point risks *memorizing* the training set.
- **The wiggle is not a bug.** Each mini-batch is a random sample, so its loss is a
  *noisy estimate* of the true loss. Expect jitter; watch the trend, not the noise.

> 💡 **Learning Thought:** Always distinguish **training loss** (the batch you just
> learned from — optimistic) from **validation loss** (held-out data — honest). The gap
> between them is the single most informative signal in the whole lecture, and Section 7
> is entirely about reading it.

---

## 🎯 Interview Questions

**Q1. Walk me through one training step of an LLM.**
> Tokenize text into IDs packed into length-`T` windows → forward pass predicts a
> next-token distribution at every position → softmax turns logits into probabilities →
> cross-entropy scores each prediction against the true next token and averages to a
> scalar `L` → backprop computes ∂L/∂θ for every parameter → the optimizer (AdamW) steps
> each parameter downhill by `η·∂L/∂θ` → load next mini-batch and repeat.

**Q2. Why mini-batches instead of full-batch or single-sample updates?**
> Full-batch gives the exact gradient but needs a full dataset pass per step —
> prohibitively slow. Single-sample (B=1) is cheap but its gradient is a very noisy
> estimate, so training is unstable and slow to converge. Mini-batch averages over many
> samples: it's a low-variance gradient estimate *and* saturates GPU parallelism.

**Q3. What is the loss function and what does minimizing it actually do?**
> Cross-entropy: the negative average log-probability the model assigns to the true next
> tokens. Minimizing it forces the model to place more probability mass on the text that
> actually occurs — i.e., become a better next-token predictor.

**Q4. Difference between a step and an epoch? Why do big LLMs train for ~1 epoch?**
> A step is one optimizer update on one mini-batch; an epoch is one full pass over the
> data. Frontier models have such enormous corpora that a single pass already contains
> billions of steps' worth of signal, and repeating data risks memorization — so ~1 epoch.

**Q5. Why is the training loss curve noisy even when training is healthy?**
> Each batch is a random sample of the data, so its measured loss is a stochastic
> estimate of the true loss. The per-batch value wiggles; only the trend is meaningful.

**Q6. (Senior) Why AdamW over plain SGD for LLMs? What's the memory cost?**
> AdamW adapts the per-parameter step using running estimates of gradient mean and
> variance, which handles the heterogeneous, sparse gradient scales across an LLM's
> layers far better than a single global LR. Cost: it stores two moment tensors per
> parameter (~2× model size), often making optimizer state the dominant memory term —
> the core motivation for PEFT methods like LoRA.

**Q7. (Senior) Relate cross-entropy loss to perplexity.**
> Perplexity = `exp(loss)` when loss is average per-token cross-entropy in nats. It's the
> effective number of choices the model is deciding between per token; lower is better.

---

## One-line takeaway

**An LLM learns by repeatedly taking a small, averaged, downhill step (`θ ← θ − η·∂L/∂θ`)
on cross-entropy loss — and every problem in the rest of this lecture is that step being
too big, too noisy, or pointed the wrong way.**
