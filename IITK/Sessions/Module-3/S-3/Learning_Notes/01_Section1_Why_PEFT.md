# Section 1 — Motivation: Why PEFT? (Slides 3–14)

> **Goal of this section:** Understand *precisely* why full fine-tuning of large LLMs is impractical, so that every PEFT method that follows feels like an obvious answer to a real pain.

---

## 1.1 Recap — Transfer learning & full fine-tuning

**Transfer learning** = take a model pre-trained on a huge, general corpus (which has already learned grammar, facts, reasoning patterns) and *adapt* it to a narrower downstream task instead of training from scratch.

**Full fine-tuning (FFT)** is the classic way to do this: unfreeze **all** the model's weights and continue training on the task dataset. Every parameter is updated by gradient descent.

FFT works and often gives the best raw accuracy. But at LLM scale it hits two walls: **cost** and **catastrophic forgetting**. The rest of this section quantifies both.

> 💡 **Learning Thought:** PEFT is *not* a new idea about learning — it's an engineering response to the fact that model **scale** grew faster than the hardware most people can afford. Frame PEFT as "democratizing fine-tuning," and the design of every method makes sense.

---

## 1.2 LLM training stages & Post-trained Language Models (PoLMs)

The lecture situates PEFT inside the modern LLM lifecycle (Tie et al., *A Survey on Post-training of LLMs*, 2025):

1. **Pre-training** — self-supervised next-token prediction on trillions of tokens. Learns general language capability. (Enormously expensive, done once by the model provider.)
2. **Post-training** — everything done *after* pre-training to make the model useful and aligned:
   - Supervised Fine-Tuning (SFT) on instruction data
   - Alignment (RLHF / DPO)
   - **Task adaptation ← this is where PEFT lives**

**Emergence with scale:** capabilities like in-context learning and reasoning *emerge* only past certain model sizes. This is the double edge: you *need* big models for these abilities, but big models are *exactly* what you can't afford to fully fine-tune. Hence *"when scale is a curse."*

> 💡 **Learning Thought:** "Emergence" is the reason we can't just use a small model. We are stuck adapting large models, so we need cheap ways to adapt them. PEFT = "keep the emergent abilities of the big model, pay only a little to specialize it."

---

## 1.3 When scale is a curse — the GPU/VRAM wall

The single most important number in this section: **memory needed to fully fine-tune a model is several times its parameter count**, because for every parameter you must simultaneously hold in GPU memory:

| What you store per parameter | Bytes (mixed precision, Adam) |
|---|---|
| Weight (fp16) | 2 |
| Gradient (fp16) | 2 |
| Optimizer state 1 — Adam momentum (fp32) | 4 |
| Optimizer state 2 — Adam variance (fp32) | 4 |
| (Often) fp32 master copy of weight | 4 |
| **≈ Total** | **~16 bytes/param** |

Plus **activations** (scale with batch size × sequence length × layers), which can rival or exceed the above.

**Worked example from the deck (7B model):**
- Full fine-tuning ⇒ **~92 GB** VRAM. Requires **multiple GPUs**.
- The GPU/VRAM table on slide 11 shows the reality: a 24 GB RTX 4090 *can't do it at all*; even A100 40 GB needs several cards; only top-end H100 NVL (188 GB) fits it on one.

> **Reconciling 92 GB with the table:** 16 bytes × 7B = **112 GB**, not 92. The deck's figure excludes the fp32 master copy: `2 + 2 + 4 + 4 = 12 bytes → 84 GB`, plus ~8 GB of activations ≈ **92 GB**. Both numbers are right; they just count different rows. Quote the *range* (~12–16 bytes/param) rather than a single figure.

**Contrast (slide 14):**
- **PEFT ⇒ ~15 GB, one GPU.** Because the frozen base weights need no gradients and no optimizer state — you only pay the full ~16 bytes/param overhead on the **<1%** of parameters you actually train.

> 💡 **Learning Thought:** The savings do **not** come from making the model smaller — the frozen 7B weights still sit in memory (as ~14 GB in fp16). The savings come from *eliminating gradients + optimizer state + activation checkpoints for 99% of the model*. This is the crux most people get wrong in interviews.

**Rule of thumb to memorize:**
- Inference of an N-billion param model in fp16 ≈ **2N GB**.
- Full fine-tuning ≈ **~12–16N GB**.
- PEFT ≈ **2N GB + a small constant**.

---

## 1.3.1 What Adam actually stores (the two 4-byte rows)

Those two optimizer-state rows are **half the training memory bill**, so they're worth understanding rather than memorizing. The load-bearing phrase in the table above is *"per parameter"*: **Adam keeps two extra numbers for every single weight.** For a 7B model that isn't "some optimizer overhead" — it's **14 billion additional floats**.

### The name says it: Adam = **ADA**ptive **M**oment estimation

The two states *are* the two moments:

| Row in the table | Real name | What it tracks |
|---|---|---|
| "momentum" | **1st moment**, `m` | running average of the gradient — **direction** |
| "variance" | **2nd moment**, `v` | running average of the gradient **squared** — **magnitude** |

Both are **exponential moving averages** — running averages that quietly forget old history:

```
m ← β₁·m + (1−β₁)·g          β₁ = 0.9    → remembers ~10 steps
v ← β₂·v + (1−β₂)·g²         β₂ = 0.999  → remembers ~1000 steps
```

And the weight update itself:

$$\theta \leftarrow \theta - \text{lr} \cdot \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}$$

(`m̂`, `v̂` are the bias-corrected versions — `m/(1-β₁ᵗ)`, `v/(1-β₂ᵗ)` — which stop the EMAs reading near-zero during the first few steps.) Everything interesting is in that fraction.

### `m` — "how consistently have we been pushed this way?"

`m` averages the **signed** gradient, so signs cancel:

- Last ~10 gradients all pointed the same way → they **reinforce** → `m` is large.
- Last ~10 gradients flip-flopped +/−/+/− → they **cancel** → `m` is near zero.

The analogy is a **ball rolling downhill**: it builds speed across a consistent slope and coasts through small bumps instead of reacting to every local wobble. Plain SGD responds to whatever the *last* batch said; `m` responds to what the last ten batches **agreed** on.

### `v` — "how big have this parameter's gradients been lately?"

`v` averages the gradient **squared**, so the sign is destroyed and only magnitude survives. The update then **divides by `√v`** — and that division is the whole point:

| This parameter's gradients are… | `√v` | Effect on its step |
|---|---|---|
| consistently large | large | **damped** — divided down |
| consistently tiny | small | **amplified** — divided up |

> 💡 **Learning Thought — this is what "adaptive" means:** Adam gives **every parameter its own learning rate**, namely `lr / √vᵢ`. Which reframes the hyperparameter entirely: `learning_rate=2e-5` is **not** the step size. It is a **global multiplier** sitting on top of ~7 billion individually-computed adaptive rates.

### The two together: a signal-to-noise ratio

$$\frac{\hat{m}}{\sqrt{\hat{v}}} \;\approx\; \frac{\text{mean of recent gradients}}{\text{typical size of recent gradients}}$$

That is a **signal-to-noise ratio**, and it is naturally bounded around ±1. Watch it work on one weight at `lr = 2e-5`, run to steady state:

| Gradient pattern | `m̂` | `√v̂` | ratio | **actual step** |
|---|---|---|---|---|
| **Consistent** `+0.01` every step | 0.01 | 0.01 | **1.00** | 2.0e-5 — full step |
| **Alternating** `±0.01` | 5.3e-4 | 0.01 | **0.053** | 1.1e-6 — **19× smaller** |
| **Consistent** `+0.000001` | 1e-6 | 1e-6 | **1.00** | 2.0e-5 — **full step** |

Two comparisons carry the whole idea:

- **Rows 1 vs. 2:** *identical* gradient magnitude, **19× difference in step size.** Adam moves confidently on agreement and shuffles cautiously through noise.
- **Rows 1 vs. 3:** gradients **10,000× smaller**, *exactly the same step.* The scale divides straight out.

That second one is why Adam handles transformers at all — gradient magnitudes differ by orders of magnitude between embeddings, attention, and LayerNorm — and why a rare token's embedding still learns instead of being drowned out.

> 💡 **Learning Thought — what the LR number literally means.** Because that ratio is bounded near 1, **with Adam the learning rate is approximately the furthest any single weight can move in one step.** `2e-5` isn't an abstract scaling factor; it's roughly *"each weight moves by at most 0.00002 per update."* That is what makes the standard peak-LR ranges concrete.

### Why fp32, when the weights and gradients are fp16?

The table asserts the precision without saying why — and the *why* is the better interview answer.

**`v` would underflow to zero.** fp16's smallest representable positive value is `2⁻²⁴ ≈ 5.96e-8`. Since `v` stores `g²`:

$$g < \sqrt{5.96\times10^{-8}} \approx 2.4\times10^{-4} \;\Longrightarrow\; g^2 \text{ rounds to } \mathbf{0}$$

Typical fine-tuning gradients run **1e-3 to 1e-5**, so a large share of them would square to exactly zero. And zero in `v` is catastrophic, because it sits in the denominator:

```
step = m / (√0 + ε) = m / 1e-8 = m × 100,000,000
```

An instant, enormous update — divergence triggered by nothing but a numeric format. **`v` is fp32 purely for dynamic range.**

**`m` would be swamped.** fp16 has a 10-bit mantissa (~0.05% relative resolution). The EMA adds `0.1 × g` onto a running value each step; once the increment is small relative to the accumulator it **rounds away entirely** and the average silently stops updating. fp32's 24-bit mantissa (~1e-7 relative) has the headroom. **`m` is fp32 for accumulation precision.**

> 💡 **Learning Thought:** the same range-vs-precision trade-off as bf16-vs-fp16, one level down: **`v` needs range, `m` needs precision**, and fp32 is the only 4-byte format with enough of both.

### The memory bill — and why this *is* the PEFT argument

For a 7B model:

| Item | Bytes/param | 7B total |
|---|---|---|
| Weight (fp16) | 2 | 14 GB |
| Gradient (fp16) | 2 | 14 GB |
| **`m` — 1st moment (fp32)** | **4** | **28 GB** |
| **`v` — 2nd moment (fp32)** | **4** | **28 GB** |
| fp32 master weight | 4 | 28 GB |

**The two optimizer states are 8 of the ~16 bytes — half the training memory, and 4× the size of the model's own fp16 weights. Adam costs more memory than the model does.**

That single fact is the quantitative core of §1.3's Learning Thought. When PEFT freezes 99% of the model the frozen weights **stay resident** (the forward pass still needs them); what evaporates is gradients + `m` + `v` for everything frozen. LoRA training 4M parameters keeps `m` and `v` at **32 MB instead of 56 GB.**

It also makes optimizer choice a genuine memory lever:

| Optimizer | State bytes/param | 7B state |
|---|---|---|
| SGD | 0 | 0 |
| SGD + momentum | 4 | 28 GB |
| **Adam / AdamW** | **8** | **56 GB** |
| Adafactor | ~0.5 (factors `v` into row/col) | ~4 GB |
| 8-bit Adam (bitsandbytes) | 2 | 14 GB |

> ⚠️ **Terminology trap:** `v` is the **uncentered second moment** `E[g²]`, *not* the variance `E[g²] − E[g]²`. The actual variance would be `v − m²`; the two coincide only when the mean gradient is zero. "Adam variance" is the common informal label (used in the table above) — know the correct term when asked.

---

## 1.4 Catastrophic forgetting in multi-task fine-tuning

Second wall (slides 13, 31). If you sequentially fine-tune the *same* weights on Task 1 → Task 2 → … → Task n:

- Gradient updates for Task 2 **overwrite** the weight configuration that solved Task 1.
- The model's performance on earlier tasks **degrades sharply** — this is **catastrophic forgetting**.

Naïve workaround: keep a **full separate copy** of the model per task. For a 7B model that's ~14 GB *per task* just to store, plus you lose any sharing. Unworkable at scale.

**How PEFT solves it:** freeze the shared base model (it never changes, so it *cannot* forget), and give each task its **own tiny module** (an adapter or a soft prompt, often <1–2% of params). To switch tasks you swap the small module, not the model.

> 💡 **Learning Thought:** Notice PEFT solves *two different problems with one mechanism*. Freezing the base → (a) removes the optimizer/gradient memory cost, and (b) removes catastrophic forgetting, because the shared knowledge is immutable. Cost and forgetting are two symptoms of the same disease: "we keep mutating a giant shared object."

---

## 1.5 The PEFT core idea (Slide 14)

> *"Why update 7 billion parameters when 4 million will do?"*

**Definition.** PEFT adapts a large pre-trained model to a new task by training **less than 1%** of its parameters, while **freezing the rest**, and reaches accuracy close to full fine-tuning.

The three benefits, together:
1. **Memory/compute:** 92 GB & multi-GPU → ~15 GB & single GPU.
2. **Storage per task:** ship a few MB adapter instead of a full model copy.
3. **No catastrophic forgetting:** base is frozen; tasks are isolated modules.

The (small) cost: a usually-negligible accuracy gap vs. FFT, and some method-specific overheads (e.g., a little extra inference latency for adapters — covered in Section 3).

---

## 🎯 Interview Questions — Section 1

**Q1. Why does full fine-tuning need ~12–16× the model's weight-memory, not just 1×?**
*A:* Because training holds, per parameter, not just the weight but also its gradient and the optimizer states (Adam keeps momentum + variance), often an fp32 master weight, plus activation memory for backprop. Inference needs only the weight (~2 bytes/param in fp16). So training memory ≈ 6–8× the raw fp16 weights, and with activations the effective figure lands around the deck's 92 GB for a 7B model.

**Q2. If PEFT freezes 99% of the model, why doesn't VRAM drop by 99%?**
*A:* The frozen weights still must be *resident* in memory to run the forward pass (~2N GB in fp16). What's eliminated is gradients, optimizer state, and most activation-checkpoint memory for the frozen portion. So you drop the *training overhead* (the ~6–8× multiplier), not the base weights themselves.

**Q3. What is catastrophic forgetting, and how does PEFT structurally prevent it?**
*A:* Sequentially fine-tuning shared weights on new tasks overwrites the configuration that solved earlier tasks, degrading old performance. PEFT freezes the shared base (immutable → can't be overwritten) and stores each task's adaptation in a separate small module, so tasks are isolated and swappable.

**Q4. Does PEFT make the model smaller or faster at inference?**
*A:* Not inherently smaller — the frozen base is still full size. Some PEFT methods (adapters, prefix tuning) *add* a little inference latency/params; others (LoRA) can be *merged* back into the weights for zero inference overhead. PEFT's win is training cost and per-task storage, not model size.

**Q5. Name the two independent problems with full fine-tuning at LLM scale and the single mechanism PEFT uses to address both.**
*A:* (1) Prohibitive training memory/cost; (2) catastrophic forgetting in multi-task settings. Single mechanism: **freeze the base model and train only small task-specific parameters.** Freezing kills the optimizer-memory overhead *and* protects shared knowledge from being overwritten.

**Q6 (deep). When would you still prefer full fine-tuning over PEFT?**
*A:* When (a) you have a very large, distribution-shifted dataset where the extra capacity of updating all weights measurably helps; (b) you serve a single task so per-task storage/forgetting don't matter; (c) you have the hardware budget and want the last fraction of accuracy; or (d) you plan to distill/quantize afterward and want the fully-adapted weights. PEFT's small accuracy gap can matter in high-stakes or heavily shifted domains.

**Q7. What are Adam's two optimizer states, and what does each contribute to the update?**
*A:* The **1st moment `m`** (EMA of the gradient, β₁=0.9) tracks *direction* — how consistently recent gradients agreed; signs cancel, so oscillating gradients give a small `m`. The **2nd moment `v`** (EMA of the gradient *squared*, β₂=0.999) tracks *magnitude*, sign-free. The update is `lr · m̂/(√v̂ + ε)`, so dividing by `√v` gives every parameter its **own effective learning rate** (`lr/√vᵢ`) — that's the "adaptive" in Adam. Together the ratio behaves like a **signal-to-noise ratio** bounded near ±1: consistent gradients get a full-size step, noisy ones get a heavily damped step, and a parameter whose gradients are 10,000× smaller still gets a full step because the scale divides out.

**Q8. Why are Adam's states kept in fp32 when weights and gradients are fp16?**
*A:* Two different numeric failures. **`v` needs range:** fp16's smallest positive value is `2⁻²⁴ ≈ 5.96e-8`, so any gradient below ~2.4e-4 squares to *exactly zero* — and typical fine-tuning gradients are 1e-3 to 1e-5. A zero in the denominator gives `m/ε` = a step ~10⁸× too large, i.e. instant divergence. **`m` needs precision:** the EMA repeatedly adds a small increment to a running value, and with fp16's 10-bit mantissa the increment eventually rounds away entirely, silently freezing the average. Same range-vs-precision trade-off as bf16 vs. fp16, one level down.

**Q9 (deep). Your 7B full fine-tune OOMs. Which single change frees the most memory, and what does it cost?**
*A:* Attack the optimizer state — it's **8 of the ~16 bytes/param (56 GB of 112 GB), more than the model's own fp16 weights.** Cheapest wins in order: **8-bit Adam** (bitsandbytes) drops 8 → 2 bytes/param, saving ~42 GB at near-zero quality cost; **Adafactor** factors `v` into row/column statistics for ~0.5 bytes/param but is noisier and less battle-tested for SFT; **SGD+momentum** halves the state but usually converges worse on transformers. The structural answer, though, is **PEFT** — freeze the base and you pay gradients + `m` + `v` on <1% of parameters (e.g. 32 MB instead of 56 GB), while the frozen weights stay resident for the forward pass.

---

## ✅ Section 1 takeaways

- Full fine-tuning fails at scale for **two** reasons: **cost** (~12–16× weight memory) and **catastrophic forgetting**.
- Memory cost comes from **gradients + optimizer state + activations**, not the weights alone.
- **Adam's two states (`m` + `v`) alone are ~half the training memory** — 8 bytes/param, 4× the fp16 weights. `m` = direction (did recent gradients agree?), `v` = magnitude (giving each parameter its own effective LR). Both fp32: `v` for range, `m` for precision.
- **PEFT = freeze base, train <1%.** One mechanism, two problems solved.
- Mental model: *inference ≈ 2N GB; FFT ≈ ~12–16N GB; PEFT ≈ 2N GB + small constant.*

➡️ **Next:** [Section 2 — the PEFT taxonomy](02_Section2_PEFT_Taxonomy.md): the four families of PEFT and where every named method fits.
