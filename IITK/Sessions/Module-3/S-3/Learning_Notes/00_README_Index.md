# Module 3 · Session 3 — PEFT: Additive & Soft Prompting

> **Course:** Executive Certificate Programme — Generative AI and Agentic AI, IIT Kharagpur
> **Instructor:** Dr. Plaban Kumar Bhowmick
> **Session topic:** Parameter Efficient Fine-Tuning (PEFT) — Additive methods (Adapters) and Soft Prompting
> **Source deck:** `3-2a-PEFT-Adapter-Soft-Prompt.pptx` (55 slides)

These notes turn the lecture deck into a self-study track. Read them in order to **develop understanding**, then use the interview questions + demo notebooks to **build expertise**.

---

## How to use these notes

1. **Read section-by-section** — each file is self-contained and builds on the previous.
2. **"Learning Thoughts"** boxes (💡) capture the deep insight an interviewer or a strong practitioner cares about — not just *what* a method does but *why* it exists.
3. **"Interview Questions"** at the end of each section range from warm-up → deep. Try answering before reading the model answer.
4. **Run the demos** when pointed to them:
   - `Residual_Adapter_Demo_1.ipynb` → Section 3 (Adapters)
   - `Prefix_Tuning_Demo_2.ipynb` → Sections 4–5 (Soft Prompting)

---

## Section map

| # | File | Topics covered | Slides |
|---|------|----------------|--------|
| 1 | [01_Section1_Why_PEFT.md](01_Section1_Why_PEFT.md) | Full fine-tuning limits, GPU/VRAM wall, catastrophic forgetting, PEFT core idea | 3–14 |
| 2 | [02_Section2_PEFT_Taxonomy.md](02_Section2_PEFT_Taxonomy.md) | PEFT taxonomy: Additive / Selective / Re-param / Soft-Prompt; concept map | 5, 15–17 |
| 3 | [03_Section3_Adapters.md](03_Section3_Adapters.md) | Adapter setup, two requirements, Sequential & Residual/Parallel adapters, AdapterFusion | 18–32 |
| 4 | [04_Section4_SoftPrompt_Foundations.md](04_Section4_SoftPrompt_Foundations.md) | Token-space adaptation, discrete vs continuous prompts, Prefix vs Prompt Tuning, cost | 33–40 |
| 5 | [05_Section5_Advanced_SoftPrompt.md](05_Section5_Advanced_SoftPrompt.md) | SMoP, APT (adaptive prefix), IDPG (instance-dependent), SPT/LPT | 41–53 |
| 6 | [06_Section6_Wrapup_Resources.md](06_Section6_Wrapup_Resources.md) | Summary, decision guide, full reading list | 54–55 |

---

## The one-slide mental model

```
                    Full Fine-Tuning  (update 100% params — costly, forgets)
                              │
                              ▼
                     PEFT  (freeze base, train <1%)
        ┌───────────────┬───────────────┬──────────────────┐
     Additive        Selective     Re-Parameterization   Soft Prompting
   (add modules)   (pick subset)   (low-rank deltas)     (add tokens)
        │               │                │                    │
   Adapters,         BitFit,          LoRA,             Prefix Tuning,
   AdapterFusion    Diff Pruning      QLoRA             Prompt Tuning,
                                                        SMoP, APT, IDPG
```

**This session focuses on the two left/right pillars: Additive (Adapters) and Soft Prompting.**
LoRA/QLoRA (re-parameterization) are named here but taught in a sibling session.

---

## The three questions that unify the whole session

Every method below is an answer to one of these:

1. **Where do we put the trainable parameters?** (in the architecture vs. in the token stream)
2. **How few parameters can we get away with?** (and are they all useful?)
3. **Should the adaptation be fixed per-task, per-layer, or per-input?**

Keep these in mind — they are the thread connecting Adapters → Prefix Tuning → SMoP → APT → IDPG.
