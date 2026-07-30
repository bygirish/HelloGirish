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

**Contrast (slide 14):**
- **PEFT ⇒ ~15 GB, one GPU.** Because the frozen base weights need no gradients and no optimizer state — you only pay the full ~16 bytes/param overhead on the **<1%** of parameters you actually train.

> 💡 **Learning Thought:** The savings do **not** come from making the model smaller — the frozen 7B weights still sit in memory (as ~14 GB in fp16). The savings come from *eliminating gradients + optimizer state + activation checkpoints for 99% of the model*. This is the crux most people get wrong in interviews.

**Rule of thumb to memorize:**
- Inference of an N-billion param model in fp16 ≈ **2N GB**.
- Full fine-tuning ≈ **~12–16N GB**.
- PEFT ≈ **2N GB + a small constant**.

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

---

## ✅ Section 1 takeaways

- Full fine-tuning fails at scale for **two** reasons: **cost** (~12–16× weight memory) and **catastrophic forgetting**.
- Memory cost comes from **gradients + optimizer state + activations**, not the weights alone.
- **PEFT = freeze base, train <1%.** One mechanism, two problems solved.
- Mental model: *inference ≈ 2N GB; FFT ≈ ~12–16N GB; PEFT ≈ 2N GB + small constant.*

➡️ **Next:** [Section 2 — the PEFT taxonomy](02_Section2_PEFT_Taxonomy.md): the four families of PEFT and where every named method fits.
