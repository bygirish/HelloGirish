# 03 — Fine-tuning Approaches (The Taxonomy)

> **Why this matters:** "Fine-tuning" is not one thing. Once you've *decided* to fine-tune
> (file 02), you make **two independent choices**: *what objective* to train on, and *how many
> weights* to move. Confusing these two axes is the most common beginner mistake. This file is
> the map; files 04–09 walk each branch.

---

## 1. The Two Axes ⭐

The professor explicitly splits fine-tuning approaches into two **orthogonal** categories
(slide 27). This is the mental scaffold for the entire lecture.

```
                        FINE-TUNING APPROACHES
                                 │
        ┌────────────────────────┴────────────────────────┐
        │                                                  │
  AXIS A: OBJECTIVE                          AXIS B: PARAMETER-UPDATE STRATEGY
  "based on task/style adaptation"          "based on how many weights change"
        │                                                  │
  ┌─────┼─────────────┐                          ┌─────────┴─────────┐
  │     │             │                          │                   │
 Unsup- Supervised  Safety/                    Full FT             PEFT
 ervised  (SFT)     Alignment              (update ALL)      (update FEW)
   FT               FT                                              │
                                                    ┌──────────────┼──────────────┐
                                                 Additive      Selective    Reparameterization
                                                                              (LoRA, QLoRA)
```

> The slide's own caption: *"Not an exhaustive list, but broad categories."* Treat this as the
> skeleton, not the full anatomy.

> 💡 **Learning Thought:** **Objective (Axis A) and strategy (Axis B) are independent.** You can
> run *any* objective with *any* strategy — e.g. "Supervised + Full FT" or "Supervised + LoRA".
> Slide 33 says it directly: *"Any of the previous recipes (unsupervised, SFT, alignment) can be
> run as full fine-tuning — it describes **what is trained, not the objective**."*

---

## 2. Axis A — By Objective (task / style adaptation)

### Unsupervised Fine-Tuning
- **What:** further training on **raw, unlabeled domain text**, using the *same
  self-supervised objective as pre-training* (next-token or masked-token prediction).
- **Same objective, new data:** no annotations needed — the model just keeps predicting tokens,
  now on *your* corpus.
- **What it buys you:** absorbs **domain knowledge & style** — vocabulary, jargon, formats, and
  facts get baked into the weights.
- **Why it's attractive:** **cheapest data to gather.** Any raw corpus works — internal wikis,
  filings, papers, support tickets — and it's usually abundant.
- *(Also called domain-adaptive pre-training / continued pre-training.)*

### Supervised Fine-Tuning (SFT)
- **What:** learn from **demonstrations** — humans (or a stronger model) write the *target
  outputs*, and the model **imitates them token by token**.
- **Teaches:** behavior and format — instruction following, tone, structure (JSON, bullet
  lists), and task skills.
- **Landmark example:** human-written demonstrations powered the **SFT stage of InstructGPT**.
- **→ Full detail in file 05.** Its specialized form (IFT) is file 06.

### Safety / Alignment Fine-Tuning
- **What:** make the model's outputs safe and preference-aligned. Three flavors named on slide 32:
  1. **Human feedback → reward → optimize (RLHF):** humans *rank* candidate answers; a **reward
     model** learns the ranking; **PPO** steers the policy toward high reward.
  2. **DPO (Direct Preference Optimization):** trains **straight on (chosen, rejected) pairs** —
     no separate reward model.
  3. **Revision / Constitutional-style:** the model **critiques and revises its own outputs**
     against written principles.
- **Depth:** *"Details you will learn in lecture 3/4/5."* Just know it *exists* and *where it
  sits* for now.

> 💡 **Learning Thought:** The three objectives form a natural progression of *supervision
> signal*: Unsupervised (no labels) → Supervised (gold outputs) → Alignment (relative
> preferences between outputs). Each needs richer, more expensive human input than the last.

---

## 3. Axis B — By Parameter-Update Strategy

### Full Fine-Tuning
- Update **every parameter** of the model on the new data.
- Maximum adaptation capacity, but high memory/compute cost and risk of forgetting.
- **→ Full detail in file 04**, and efficiency tricks in file 09.

### Parameter-Efficient Fine-Tuning (PEFT)
- **Core idea:** **preserve most of the model's original weights**; train only a small number
  of new/selected parameters.
- Broadly **three methods** (slide 34–35):

| PEFT method | What it does |
|-------------|--------------|
| **Additive** | *Add* one or more trainable layers / a new set of parameters; **only the new ones are trained.** The base model is frozen. (e.g. adapters, prompt/prefix tuning.) |
| **Selective** | *Select a subset* of the existing parameters to fine-tune; freeze the rest. (e.g. tune only bias terms, or only the top layers.) |
| **Reparameterization** | Don't update all params and don't add physical adapter layers — instead use a **new low-dimensional representation** during training. Uses **low-rank matrix transformations**. Prominent techniques: **LoRA** (Low-Rank Adaptation) and **QLoRA** (Quantized LoRA). |

- **Reparameterization depth:** *"Details you will learn in lecture 3/4/5."* For now, know that
  **LoRA/QLoRA freeze the big model and learn tiny low-rank update matrices**, giving most of
  full-FT's benefit at a fraction of the memory.

> 💡 **Learning Thought:** The three PEFT families answer *"where do the trainable parameters
> come from?"* — **Additive = add new ones**, **Selective = reuse a chosen few existing ones**,
> **Reparameterization = express the update in a smaller basis (low rank).** LoRA is the famous
> member of the third family and the reason PEFT went mainstream.

---

## 4. Putting the Two Axes Together

Any real fine-tuning run is a **pair**: *(objective, strategy)*.

| Example run | Axis A (objective) | Axis B (strategy) |
|-------------|--------------------|--------------------|
| Continued pre-training a Llama on legal text | Unsupervised | Full FT (or LoRA) |
| InstructGPT SFT stage | Supervised | Full FT |
| A cheap instruction-tuned chatbot on one GPU | Supervised (IFT) | PEFT — LoRA |
| RLHF/DPO safety pass | Alignment | Full FT or PEFT |

> 💡 **Learning Thought:** When someone says *"I fine-tuned the model,"* the two questions that
> pin it down are: **"On what objective?"** and **"How many weights moved?"** If you can always
> answer both, you understand this lecture.

---

## 🎯 Interview Questions

**Q1. What are the two orthogonal axes along which fine-tuning approaches are categorized?**
**Axis A — objective** (unsupervised, supervised/SFT, safety-alignment): *what* the model learns.
**Axis B — parameter-update strategy** (full FT vs. PEFT): *how many* weights change. They're
independent — any objective can run under any strategy.

**Q2. How does unsupervised fine-tuning differ from pre-training?**
Same self-supervised objective (next/masked-token prediction), but on a *smaller, domain-specific*
corpus starting from an already pre-trained model, to absorb domain jargon/style. It's continued
pre-training, not from scratch.

**Q3. What is PEFT and what problem does it solve?**
Parameter-Efficient Fine-Tuning keeps most original weights frozen and trains only a small set of
new/selected parameters. It solves full FT's huge memory/compute/storage cost while retaining most
of the adaptation benefit.

**Q4. Name the three PEFT families and one technique or example of each.**
**Additive** — add trainable layers/params, train only those (adapters, prefix tuning).
**Selective** — fine-tune a chosen subset of existing params (bias-only, top layers).
**Reparameterization** — low-rank update in a new basis (**LoRA, QLoRA**).

**Q5. "Full fine-tuning is an objective." True or false, and why?**
**False.** Full FT is a *parameter-update strategy* — it describes *what is trained* (all
weights), not the *objective*. You can run unsupervised, SFT, or alignment objectives as full FT.

**Q6. In one line each, how does LoRA differ from an additive adapter?**
An **additive adapter** inserts new physical layers into the network and trains them. **LoRA**
adds *no* physical layers — it learns low-rank matrices that reparameterize the weight update and
can be merged back into the original weights at inference.

**Q7. Why is unsupervised fine-tuning data the "cheapest to gather"?**
It needs **no human annotation** — any raw domain corpus (wikis, filings, tickets) works, because
the label is just the next token. SFT and alignment require expensive human-written outputs or
preference judgments.

---

**One-line takeaway:** *Every fine-tuning run is a pair — an **objective** (unsupervised /
supervised / alignment) crossed with a **parameter strategy** (full FT vs. PEFT's additive /
selective / reparameterization) — and the two choices are independent.*
