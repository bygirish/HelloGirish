# Section 2 — The PEFT Taxonomy (Slides 5, 15–17)

> **Goal:** Build the mental "map" of PEFT so every method later has a home. If you can place a new method into this taxonomy on sight, you understand PEFT structurally.

---

## 2.1 The organizing question

All PEFT methods freeze the base and add/select a small set of trainable parameters. They differ on **one question:**

> **Where do the trainable parameters live, and what form do they take?**

The concept map (slide 5) answers this with **four families**:

```
                          PEFT
   ┌──────────────┬───────────────┬────────────────────┬───────────────┐
 Additive       Selective     Re-Parameterization    Soft Prompting
   │               │                  │                    │
 Adapters,       BitFit,           LoRA,              Prefix Tuning,
 Adapter         Diff Pruning      QLoRA              Prompt Tuning,
 Fusion                                               SMoP, (APT, IDPG)
```

*(This session teaches **Additive** and **Soft Prompting** in depth; LoRA/QLoRA under Re-Parameterization are covered in a sibling session.)*

---

## 2.2 The four families

### A) Additive PEFT — *"insert new small modules"*
Add brand-new trainable layers (**adapters**) *inside* the frozen network. The base weights never change; the added modules learn the task.
- **Examples:** Sequential Adapter (Houlsby), Residual/Parallel Adapter (Lin), AdapterFusion, Tiny-Attention Adapter.
- **Trainable params live in:** new bottleneck sub-layers in the architecture.
- **Signature trade-off:** extra parameters ⇒ some **inference latency** (you run through extra layers).

### B) Selective PEFT — *"train a subset of existing weights"*
Don't add anything; just **choose a small subset** of the model's own parameters to unfreeze.
- **BitFit:** train *only the bias terms* (a tiny fraction of params) — surprisingly effective.
- **Diff Pruning:** learn a sparse "diff" vector added to the frozen weights; a learned mask keeps it sparse.
- **Trainable params live in:** a chosen subset of original weights.
- **Trade-off:** zero added inference latency, but limited capacity/expressiveness.

### C) Re-Parameterization PEFT — *"learn a low-rank delta"*
Represent the weight update as a **low-rank** decomposition. Freeze `W`, learn `ΔW = B·A` where `A`, `B` are skinny matrices (rank `r ≪ d`).
- **LoRA:** `h = Wx + (BA)x`. Train only `A`, `B`.
- **QLoRA:** LoRA on top of a 4-bit **quantized** frozen base → fine-tune huge models on one GPU.
- **Trainable params live in:** low-rank factors of the weight update.
- **Killer feature:** `ΔW` can be **merged** into `W` after training ⇒ **zero** extra inference latency.
- *(Taught in the sibling session — named here for completeness.)*

### D) Soft Prompting PEFT — *"prepend trainable virtual tokens"*
Don't touch the architecture at all. **Prepend a sequence of trainable "virtual" token embeddings** to the input (or to each layer's key/value). The model is *steered* by this learned context.
- **Examples:** Prefix Tuning (Li & Liang), Prompt Tuning, SMoP, APT, IDPG, SPT/LPT.
- **Trainable params live in:** the **token/embedding space**, not the weight space.
- **Trade-off:** very few params; but consumes context length and adds some latency per extra token.

> 💡 **Learning Thought — the deepest axis:** *Additive/Selective/Re-param all modify the **function** (the weights/architecture). Soft Prompting modifies the **input** (the context) while leaving the function untouched.* This is why slide 34 contrasts them as **"architectural space"** vs **"token space."** Two fundamentally different levers to specialize the same model.

---

## 2.3 A comparison table to memorize

| Family | What's trainable | Added inference cost | Mergeable? | Example params |
|---|---|---|---|---|
| **Additive (Adapters)** | New bottleneck layers | Yes (extra layers) | No | ~1–4% |
| **Selective (BitFit)** | Subset of existing (biases) | No | N/A (in-place) | ~0.1% |
| **Re-param (LoRA)** | Low-rank `A,B` of ΔW | No (after merge) | **Yes** | ~0.1–1% |
| **Soft Prompt (Prefix)** | Virtual token embeddings | Yes (extra tokens) | No | ~0.1–1% |

> 💡 **Learning Thought:** Interviewers love "LoRA vs Adapters." The crisp differentiator: **LoRA merges away to zero inference cost; adapters add permanent layers.** Adapters are more *modular* (hot-swap per task without touching weights); LoRA is *cheaper at serve time*. Different optimization targets.

---

## 2.4 Where this session goes

The rest of the lecture drills into the two families that best illustrate the *"architectural space vs token space"* split:

- **Section 3 → Additive / Adapters** (architectural space)
- **Sections 4–5 → Soft Prompting** (token space), including its modern refinements (SMoP, APT, IDPG, SPT).

---

## 🎯 Interview Questions — Section 2

**Q1. Name the four PEFT families and the one distinguishing idea of each.**
*A:* **Additive** — insert new small modules (adapters); **Selective** — unfreeze a subset of existing weights (e.g., biases in BitFit); **Re-parameterization** — learn a low-rank delta of the weights (LoRA/QLoRA); **Soft Prompting** — prepend trainable virtual tokens (Prefix/Prompt Tuning).

**Q2. Which families add inference latency and which don't? Why?**
*A:* Additive (adapters) and Soft Prompting add latency — adapters insert extra layers to run; soft prompts add extra tokens to attend over. Selective adds none (it just changes existing weights in place). Re-param (LoRA) adds none *after merging* `BA` back into `W`.

**Q3. Contrast "architectural space" vs "token space" adaptation.**
*A:* Architectural-space methods (adapters, LoRA, BitFit) change the model's *function* — its weights or layers. Token-space methods (soft prompting) leave the function untouched and instead change the *input context* with learned virtual tokens that steer the frozen model.

**Q4. LoRA vs Adapters — when would you pick each?**
*A:* Pick **LoRA** when serving latency matters and you can merge the delta into the base (single-task serving, or swap merged checkpoints). Pick **Adapters** when you want strong **modularity** — hot-swappable, composable per-task modules (e.g., AdapterFusion) without ever recomputing base weights, at the cost of some inference latency.

**Q5. Where does QLoRA's efficiency come from beyond LoRA?**
*A:* QLoRA quantizes the *frozen* base model to 4-bit (NF4), slashing the resident weight memory (~2N GB → ~0.5N GB), and trains LoRA adapters in higher precision on top. This lets very large models be fine-tuned on a single consumer GPU.

**Q6 (deep). Why can soft prompting use so few parameters yet still specialize a model?**
*A:* Because a large pre-trained model already contains the needed capabilities; the task is to *elicit* the right behavior. A learned continuous prefix conditions the frozen attention distribution to route into the right latent "skill," so you only need enough parameters to specify a good context, not to relearn the skill.

---

## ✅ Section 2 takeaways

- Four families: **Additive, Selective, Re-parameterization, Soft Prompting.**
- The deepest split: **weight/architecture space** (Additive/Selective/Re-param) vs **token space** (Soft Prompting).
- Memorize the **latency + mergeability** table — it's the source of most comparison questions.
- This session = **Adapters** (Section 3) + **Soft Prompting** (Sections 4–5).

➡️ **Next:** [Section 3 — Adapters](03_Section3_Adapters.md).
