# Section 5 — Choosing Batch Size (Slides 46–53)

> **Why this matters:** Batch size is the *other* half of step control (LR is the first). It
> sets how *noisy* each gradient is, how well you use the GPU, and — coupled with LR — how
> stably you converge. Crucially, on real hardware you rarely set it directly; you *assemble*
> it from three sub-knobs. Getting the mental model right prevents both out-of-memory crashes
> and mysteriously bad convergence.

---

## 5.1 What you're actually trading off (Slide 47)

Four factors pull against each other:

- **GPU memory** — bigger batch = more activations stored = more VRAM.
- **Training stability** — bigger batch = less noisy gradient.
- **Throughput** — bigger batch = better hardware utilization (to a point).
- **Model generalization** — *too* big can hurt generalization.

**Goal:** *maximize hardware utilization while maintaining good convergence.*

---

## 5.2 The three sub-knobs and effective batch size (Slide 48)

You almost never set "batch size" as one number. You compose it:

- **Micro-batch size (per GPU):** samples processed in one forward/backward pass. *Bounded by
  VRAM.*
- **Gradient accumulation steps:** number of micro-batches to process *before* one optimizer
  update. *Free in memory, costs time.*
- **Number of GPUs:** data-parallel replicas.

$$\textbf{Effective batch} = \text{Micro-batch} \times \text{Grad-accum} \times \text{GPUs}$$

**Example:** micro-batch 2 × grad-accum 16 × 4 GPUs = **128**.

> 💡 **Learning Thought — the single most important idea in this section:** **Gradient
> accumulation decouples the batch size that matters for *learning* (effective batch) from the
> batch size that fits in *memory* (micro-batch).** You accumulate gradients over several
> small forward/backward passes and only *then* step. This is how a 24 GB consumer GPU can
> train at an effective batch of 128 that "should" need far more VRAM.

**In HF Trainer, it's two arguments** (effective batch = product of these × #GPUs):

```python
TrainingArguments(
    per_device_train_batch_size = 4,     # micro-batch — as big as VRAM allows
    gradient_accumulation_steps = 8,     # accumulate 8 micro-batches before stepping
    # → effective batch = 4 × 8 × (num GPUs) = 32 on 1 GPU
)
```

**The manual-loop version** makes the mechanic explicit — call `backward()` every micro-batch
but `step()` only every *k*-th one, and scale the loss by `k` so the gradient is an *average*:

```python
ACCUM = 8
for i, batch in enumerate(loader):
    loss = loss_fn(model(**batch), batch["labels"]) / ACCUM   # scale so grads AVERAGE
    loss.backward()                                           # accumulates into .grad
    if (i + 1) % ACCUM == 0:                                  # step once every 8 micro-batches
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # clip on the FULL accumulated grad (§2)
        optimizer.step()
        optimizer.zero_grad()
```

---

## 5.3 How to select batch size — the procedure (Slide 49)

A clean, repeatable recipe:

1. **Decide the maximum sequence length.** (Longer sequences cost more memory *per sample*, so
   this must come first — it sets your memory budget per sample.)
2. **Increase the micro-batch until GPU memory is nearly full.** Find the biggest micro-batch
   that fits.
3. **Use gradient accumulation to reach the desired effective batch.** Make up the rest with
   accumulation steps.

> 💡 **Learning Thought:** The order is deliberate: *seq-len → micro-batch (memory) →
> accumulation (the rest).* You max out hardware first, then use the "free" knob (accumulation)
> to hit your target. Never pick effective batch first and hope it fits.

---

## 5.4 The core trade-off: small vs. large batch (Slide 50)

| | **Smaller batch** | **Larger batch** |
|---|---|---|
| **Pros** | Better generalization; more frequent updates | Stable gradients; faster throughput; better GPU utilization |
| **Cons** | Noisy gradients; slower convergence | Can reduce generalization; fewer optimizer updates; may require higher LR |

> 💡 **Learning Thought:** Note the symmetry with §1.3 — the mini-batch gradient is an
> *average*. More samples → lower-variance average → smoother but "less exploratory" steps. A
> little gradient noise is actually *helpful* for generalization (it nudges the model out of
> sharp minima), which is why the biggest possible batch isn't always the best batch.

---

## 5.5 Batch size ↔ learning rate: the Linear Scaling Rule (Slide 51)

Batch size and LR are **coupled** — you cannot tune one in isolation.

> **Increasing batch size usually requires increasing the learning rate.**

**[Linear Scaling Rule](https://arxiv.org/abs/1706.02677):** *if the effective batch doubles,
the learning rate ≈ doubles.* (From Goyal et al., *"Accurate, Large Minibatch SGD"* — the
paper that trained ImageNet in 1 hour and popularized warmup + linear scaling together.)

| Effective Batch | Learning Rate |
|-----------------|---------------|
| 32 | 2e-5 |
| 64 | 4e-5 |

> 💡 **Learning Thought — the intuition:** a larger batch gives a *less noisy* gradient, so you
> can *trust it more* and take a proportionally bigger step. Same total "distance" per unit of
> data, fewer-but-larger steps. (In practice, add warmup when scaling LR up, and the rule is a
> starting point, not a law — very large batches often need a square-root, not linear, scaling.)

---

## 5.6 Practical recommendations by GPU memory (Slide 52)

| GPU memory | Micro-batch | Grad-accum | Effective batch |
|------------|-------------|-----------|-----------------|
| **24 GB** | 1–4 | 8–32 | 16–64 |
| **40 GB** | 4–8 | 4–16 | 32–128 |
| **80 GB** | 8–32 | 1–8 | 64–256 |

Notice the pattern: as VRAM grows, micro-batch grows and accumulation *shrinks* to hit a
similar effective batch — you're substituting real parallelism for simulated parallelism.

---

## 5.7 Let the loss curves judge your batch size (Slide 53)

The empirical feedback loop — read the *train loss curve* and adjust:

| Symptom | Diagnosis | Action |
|---------|-----------|--------|
| **Smooth, steady decay**, noise shrinking over time | Batch is **fine** — clean gradient signal, well-sized steps | **Keep it.** |
| **Jitter that never shrinks** | Batch **too small** (or LR too high) — noisy gradients | **Raise effective batch** (↑ grad-accum) or lower LR until the *trend*, not the noise, dominates |
| **Both losses plateau high** | Batch **too large** — dataset ÷ batch left only a handful of updates → underfit | **Shrink the batch** until the run gets **≥ 500–1,000 optimizer steps** |

> 💡 **Learning Thought:** The "too large" failure is the sneaky one. A huge batch gives
> beautiful smooth curves — but if `dataset_size ÷ effective_batch` yields only a few dozen
> optimizer *steps*, the model barely updates and underfits. **Always sanity-check your total
> optimizer-step count** (aim for ≥ 500–1,000); smooth curves alone don't mean healthy training.

---

## 🎯 Interview Questions

**Q1. What is effective batch size and how is it computed?**
> The batch size that actually determines the gradient used for an optimizer update:
> `micro-batch × gradient-accumulation-steps × number-of-GPUs`. E.g., 2 × 16 × 4 = 128.

**Q2. What problem does gradient accumulation solve?**
> It decouples the *learning* batch size from the *memory* batch size. You run several small
> micro-batches (each fitting in VRAM), accumulate their gradients, and only then step —
> simulating a large batch on limited memory. Costs wall-clock time, not memory.

**Q3. Walk through how you'd pick a batch size on a fixed GPU.**
> Fix max sequence length first (sets per-sample memory), grow the micro-batch until VRAM is
> nearly full, then use gradient accumulation to reach the target effective batch. Finally,
> verify the run still has enough optimizer steps (≥500–1,000).

**Q4. Trade-offs of small vs. large batches?**
> Small: noisier gradients and slower convergence, but more frequent updates and often better
> generalization. Large: stable gradients, high throughput and GPU utilization, but fewer
> updates, possible generalization loss, and typically needs a higher LR.

**Q5. State the Linear Scaling Rule and the intuition behind it.**
> If effective batch doubles, scale LR by ~2×. A larger batch yields a lower-variance gradient
> you can trust more, so a proportionally larger step keeps the effective progress-per-datum
> roughly constant. (Add warmup; very large batches may need √-scaling.)

**Q6. Your loss curve is smooth but both train and val loss plateau high. What's likely wrong?**
> Batch is probably too large: dataset ÷ effective batch leaves too few optimizer steps, so the
> model underfits. Shrink the effective batch to get ≥500–1,000 updates. (Also rule out LR too
> low.)

**Q7. Your loss curve jitters persistently and never smooths. What do you do?**
> Gradients are too noisy — either the batch is too small or the LR is too high. Raise the
> effective batch via gradient accumulation, and/or lower the LR, until the trend line rather
> than the noise dominates.

**Q8. (Senior) Why isn't "biggest batch that fits" always optimal?**
> Two reasons: (1) fewer optimizer steps for a fixed dataset → risk of underfitting; (2) large-
> batch training reduces beneficial gradient noise, which can steer the model into sharp minima
> that generalize worse. You want the *largest batch that still yields enough well-generalizing
> updates*, coupled with an appropriately scaled LR.

---

## One-line takeaway

**Assemble effective batch = micro-batch × grad-accum × GPUs — max out micro-batch for memory,
use accumulation to hit your target, scale LR with batch (linear rule), and let the loss curve
referee: jitter → batch too small, high plateau → batch too large (too few of your ≥500–1,000
needed steps).**

---

## 🔗 Further reading

- **The linear scaling rule:** [Accurate, Large Minibatch SGD (Goyal et al., 2017)](https://arxiv.org/abs/1706.02677)
  — origin of §5.5, including why you *must* add warmup when scaling LR up.
- **Batch size vs. generalization:** [On Large-Batch Training for Deep Learning: Generalization
  Gap and Sharp Minima (Keskar et al., 2017)](https://arxiv.org/abs/1609.04836) — the evidence
  behind "too big can hurt generalization" in §5.4.
- **The dual view:** [Don't Decay the Learning Rate, Increase the Batch Size (Smith et al., 2018)](https://arxiv.org/abs/1711.00489)
  — batch size and LR are two handles on the *same* thing (gradient noise).
- **Gradient accumulation, practically:** [HF — Methods and tools for efficient training on a single GPU](https://huggingface.co/docs/transformers/perf_train_gpu_one)
  and [Performance & Scalability](https://huggingface.co/docs/transformers/performance) — memory
  math, accumulation, checkpointing, and the recent
  [gradient-accumulation loss-scaling fix](https://huggingface.co/blog/gradient_accumulation).
- **Critical batch size (when scaling stops helping):** [An Empirical Model of Large-Batch Training (McCandlish et al., 2018)](https://arxiv.org/abs/1812.06162)
  — introduces the "gradient noise scale" that predicts the largest *useful* batch.
