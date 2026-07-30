# 09 — Making Full Fine-Tuning Efficient

> **Why this matters:** File 04 said full FT is "expensive." This file makes that *quantitative*
> — you'll be able to compute the GPU memory a fine-tune needs on a napkin — and then covers the
> three cheap tricks (layer-wise, block-wise, progressive) that keep full FT's *reach* while
> cutting its *peak memory*. The memory-math question is one of the most common LLM interview
> questions, period.

---

## 1. The Cost Reality (slide 62)

Full-parameter fine-tuning means **updating all parameters** — and that's brutal at scale:

- **InstructGPT has 175 billion parameters.** Fine-tuning updates *all* of them.
- Storing + updating these weights needs **a lot of GPU memory**, **a lot of compute**, and
  therefore **a lot of money**.
- **The eye-watering fact (slide 62):** GPT-4-scale training used roughly **25,000 A100 GPU
  servers, for months, costing ~$100 million USD.**

That's *pre-training* scale, but it frames why even *fine-tuning* a 10B–175B model is a serious
budget line — and why PEFT (file 03) exists.

---

## 2. The Memory Math ⭐ (slides 63–64) — learn to derive this

**Setup:** a **10-billion-parameter** model, trained in **FP16** (2 bytes per number) with the
**Adam optimizer**.

Adam doesn't just store weights — for *every* parameter it also stores a gradient and **two**
running statistics (momentum *m* and variance *v*). The per-parameter tally:

| What is stored | Precision | Bytes / param |
|----------------|-----------|---------------|
| Weights | FP16 | 2 |
| Gradients | FP16 | 2 |
| Adam momentum (*m*) | FP32 | 4 |
| Adam variance (*v*) | FP32 | 4 |
| **Total** | | **12 bytes / param** |

**⇒ 12 bytes × 10 billion params = 120 GB** (matches slide 63/64 exactly).

> **Note the two figures the deck uses:** slide 33 quotes **~16 bytes/param**, slides 63–64 give
> **120 GB = 12 bytes/param**. The difference is the **FP32 master copy of the weights** (+4
> bytes): 12 bytes (grad + m + v + fp16 weights) → **16 bytes** once you add the fp32 master →
> ~160 GB for 10B. Either number is "correct" depending on whether you count the master copy;
> quote 120 GB if asked to match this lecture, and be ready to explain the +4.

**Now fit it on hardware (slide 64):** a single GPU tops out around **24 GB**. To hold ~120 GB
(plus activations and overhead) you need a **multi-GPU pack** — the slide shows a **7-GPU pack**
of 24 GB cards (7 × 24 = 168 GB, giving headroom over the raw 120 GB).

> 💡 **Learning Thought:** The memorable ratio is **"Adam ≈ 12–16 bytes per parameter."** Weights
> are only 1/6 of the bill — **the optimizer state (m, v) dominates.** This single fact explains
> (a) why full FT needs so many GPUs and (b) why PEFT/LoRA — which trains a tiny number of params
> and thus keeps optimizer state tiny — is such a massive memory win.

---

## 3. The Ultimate Reality (slide 65)

> **Full fine-tuning is very expensive — time and memory both. *But* the adaptation to the new
> task is better.**

This is the tension the next three techniques attack: they try to **keep full FT's adaptation
quality while lowering peak memory**, by never having *all* layers "active" (holding gradients +
optimizer state) at the same moment.

---

## 4. Three Efficiency Strategies (slides 66–68)

All three still ultimately tune the whole network — they differ in **how much is trainable at
once**. Fewer active layers → fewer gradients + optimizer states in memory at any instant → lower
peak memory (and often more stable training).

### (a) Layer-wise — "one layer at a time" (slide 66)
Fine-tune **all** layers, but **one layer at a time**: unfreeze a single decoder layer, train,
freeze it, move to the next. At any moment only *one* layer carries gradients + optimizer state.
- **Pro:** minimal peak memory.
- **Con:** many sequential passes; slow.

### (b) Block-wise (slide 67)
Instead of individual layers, train **contiguous blocks** of layers together (e.g. decoders 1–3,
then 4–6).
- **Pro:** a middle ground — fewer passes than layer-wise, more parallel learning within a block.
- **Con:** higher peak memory than single-layer, lower than full.

### (c) Progressive layer-wise (slide 68)
Rather than training all selected layers from the start, **progressively unfreeze deeper layers**
over the course of training — start with some layers trainable and *add more* as you go.
- **Pro:** eases the model in gradually (a *depth* curriculum — cf. file 08); can improve
  stability.
- **Con:** requires a schedule for when to unfreeze.

| Strategy | What's trainable at once | Analogy |
|----------|--------------------------|---------|
| Layer-wise | exactly one layer | tune one instrument at a time |
| Block-wise | a contiguous block | tune sections of the orchestra |
| Progressive | a growing set (unfreeze deeper over time) | add instruments as the piece builds |

> 💡 **Learning Thought:** All three exploit the same lever — **you don't need every layer's
> gradient and optimizer state resident simultaneously.** By making only part of the network
> trainable at a time, you cut the dominant cost (optimizer state) *without* giving up on
> eventually adapting the whole model. It's full FT's reach at a fraction of the *peak* memory.

---

## 5. How This Connects to the Rest of the Course

- **PEFT (file 03)** is the *other* answer to this memory problem: instead of tuning layers in
  turns, freeze the base entirely and train tiny low-rank matrices (LoRA). Layer/block/progressive
  reduce *when* you pay; PEFT reduces *how much* you ever pay.
- **Progressive unfreezing** is a **curriculum over model depth** — same philosophy as file 08.
- **The `Multi_Task_FFT.ipynb`** notebook and the S-2 `LW-BW-P-FT` demo let you *watch* these
  layer/block/progressive schedules run.
- **Part II (S-2)** picks up here with *how to keep the run stable* (LR schedules, batch size,
  reading loss curves).

---

## 🎯 Interview Questions

**Q1. Estimate the GPU memory to fully fine-tune a 10B-parameter model with Adam in FP16.**
~**120 GB**. Per parameter: 2 (fp16 weights) + 2 (fp16 grads) + 4 (Adam *m*, fp32) + 4 (Adam *v*,
fp32) = **12 bytes**; ×10B = 120 GB. Add a 4-byte fp32 master weight copy and it's ~16 bytes/param
≈ 160 GB.

**Q2. Why does full fine-tuning need so much more memory than just storing the model?**
Because the **optimizer state dominates**. Weights are ~2 bytes/param but Adam adds gradients plus
two fp32 moments (m, v) — roughly 12–16 bytes/param total, so the model weights are only ~1/6 of
the memory bill.

**Q3. The deck says 120 GB but also "~16 bytes per parameter." Reconcile.**
120 GB uses **12 bytes/param** (fp16 weights + fp16 grads + fp32 m + fp32 v). The 16-byte figure
**adds the fp32 master copy of the weights** (+4 bytes), giving ~160 GB. Same model, different
accounting of the master copy.

**Q4. If one GPU has 24 GB, how many do you need for that 10B fine-tune, and why more than 120/24?**
The deck uses a **7-GPU pack** (168 GB). You size above the raw 120 GB to leave headroom for
**activations, communication buffers, and fragmentation** — 120/24 = 5 is the floor, not a safe
allocation.

**Q5. Explain layer-wise vs. block-wise vs. progressive fine-tuning.**
**Layer-wise:** train one layer at a time (all eventually), minimal peak memory, slow.
**Block-wise:** train contiguous blocks of layers together — a middle ground. **Progressive:**
gradually unfreeze deeper layers over training — a depth curriculum that eases the model in.

**Q6. How do these three strategies save memory if they still tune the whole model?**
They keep only *part* of the network trainable **at any one moment**, so only those layers hold
gradients and optimizer state. Peak memory is set by the largest simultaneously-trainable set, not
the whole model.

**Q7. Contrast this family of tricks with PEFT/LoRA as memory solutions.**
Layer/block/progressive **reduce when you pay** the optimizer-state cost (tune the full model in
turns). PEFT/LoRA **reduce how much you ever pay** — freeze the base entirely and train tiny
low-rank adapters, so optimizer state is negligible and you store a small adapter, not a full
checkpoint.

**Q8. Why might progressive unfreezing also improve training stability, not just memory?**
Adapting shallow layers first and easing into deeper ones is a **curriculum over depth**: it avoids
large simultaneous updates across the whole network, reducing the chance of destabilizing the
pretrained features early on.

---

**One-line takeaway:** *Full FT's memory is dominated by Adam's optimizer state (~12–16 bytes/param
→ ~120–160 GB for a 10B model), and layer-wise / block-wise / progressive fine-tuning cut the
**peak** by making only part of the network trainable at a time — while PEFT cuts the cost
altogether.*
