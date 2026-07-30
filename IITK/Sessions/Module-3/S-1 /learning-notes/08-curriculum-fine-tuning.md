# 08 — Curriculum Fine-Tuning

> **Why this matters:** The *order* in which you show examples changes how well and how fast a
> model learns — not just *which* examples. Curriculum learning is a cheap, data-side lever
> (no architecture change) that can speed up convergence and stabilize training. It's the most
> conceptually rich topic in this lecture, so this file is the longest.

---

## 1. What Is Curriculum Learning?

**A training strategy that presents examples in a meaningful order — typically from *easy to
hard* — instead of uniformly at random.**

Three motivating ideas (slide 56):
- **Inspired by human education:** students master arithmetic before calculus; models can benefit
  from the same progression.
- **Shapes the loss landscape:** early easy examples guide optimization toward *smoother regions*
  before harder ones arrive.
- **Two core ingredients:** a **difficulty measure** (what is "hard"?) and a **schedule** (when is
  harder data introduced?).

**The progression:** `Easy (short, common) → Medium (longer, mixed) → Hard (rare, complex)`, with
**model competence → data difficulty** kept roughly in sync.

> 💡 **Learning Thought:** Curriculum learning changes **none** of the model, the objective, or
> even the dataset contents — only the **order and timing** of examples. It's the cheapest
> possible intervention: pure data scheduling.

---

## 2. Why Curriculum for LLMs? (slide 57) — the four benefits

| Benefit | Mechanism |
|---------|-----------|
| **Faster convergence** | Easy examples give **clean gradients early**, so the model reaches a target loss in fewer steps and less compute. |
| **More stable training** | Fewer loss spikes/divergences — noisy, hard, or out-of-distribution samples arrive **only once the model can absorb them**. |
| **Better generalization** | Easy→hard ordering acts like a **continuation method**, steering optimization toward **flatter minima** that transfer better. |
| **Data & compute efficiency** | High-value ordering squeezes more from **limited** fine-tuning data — critical when instruction data is scarce/expensive. |

> 💡 **Learning Thought:** "Continuation method" is the deep idea: solve an *easy* version of the
> optimization problem first, then gradually morph it into the *hard* one. Starting easy lands you
> in a good basin; the hard examples then refine within it, instead of throwing the model into a
> jagged landscape cold. (Flatter minima ≈ better generalization ties back to Part II.)

---

## 3. The Pipeline (slide 58) — five stages with a feedback loop

```
1. Pool data        → collect the fine-tuning set (instructions, code, domain corpora)
2. Score difficulty → assign each example a difficulty score (length, loss, perplexity, labels)
3. Order / bucket   → sort or bin into easy → hard stages (or a continuous ranking)
4. Scheduled train  → fine-tune in phases, expanding the data pool as competence grows
5. Evaluate & adapt → track validation loss per stage; re-pace or re-score if progress stalls
        │
        └──── feedback: evaluation results update difficulty scores and pacing ────┘
```

The **feedback loop** (5 → 2) is what makes it adaptive: if the model stalls, you **re-score
difficulty and re-pace** rather than blindly following the initial ordering.

---

## 4. Designing a Curriculum: Three Key Decisions (slide 59) ⭐

This is the interview-favorite framework. A curriculum is fully specified by three choices:

### (1) Difficulty measure — *how do we quantify "hard"?*
- **Static heuristics:** length, rarity, syntactic depth.
- **Model-based:** loss, perplexity, margin (the model tells you what *it* finds hard).
- **External:** human labels, grade level.

### (2) Schedule / pacing — *when does harder data arrive?*
- **Stage boundaries:** fixed steps vs. adaptive.
- **Pacing function:** linear, root, exponential.
- **Trigger:** step count vs. a **competence signal**.

### (3) Data mixing — *what does each phase contain?*
- **Cumulative:** keep easy data in the mix as you add harder data.
- **Replacement:** swap stages entirely.
- **Replay** easy samples to **avoid forgetting**.

> 💡 **Learning Thought:** Memorize the triple **(difficulty measure, pacing schedule, data
> mixing)** — *what's hard*, *when hard arrives*, *what each phase holds*. Any curriculum method
> (including the scheduling strategies below) is just a specific setting of these three dials.

---

## 5. Scheduling Strategies (slide 60) — the named recipes

| Strategy | How it works | Trade-off |
|----------|--------------|-----------|
| **Baby Steps** | Train on the easiest bucket, then **add** the next bucket while **keeping earlier data**. Cumulative. | Robust to forgetting; more data per stage. |
| **One-Pass (staged)** | Train on each difficulty bucket **once, in order, discarding** previous stages. | Cheap, but **risks catastrophic forgetting**. |
| **Self-Paced Learning** | The **model** picks its curriculum: include samples with **loss below a rising threshold** — "easy" is defined by the *learner*. | Adaptive to the model's actual competence. |
| **Competence-Based** | A pacing function **c(t)** grows over time; at each step sample from the **easiest c(t) fraction** of data. | Smooth, tunable progression. |
| **Anti-Curriculum (Hard-First)** | Start with **hard** examples. | Occasionally useful for robust features, or when easy data dominates and adds little signal. |

**Map to the three decisions:** Baby Steps vs. One-Pass differ in **data mixing** (cumulative vs.
replacement); Self-Paced vs. Competence-Based differ in the **pacing trigger** (learner loss vs. a
time function c(t)); Anti-Curriculum inverts the **difficulty ordering**.

> 💡 **Learning Thought:** **Baby Steps = cumulative (safe), One-Pass = replacement (cheap but
> forgets).** That single contrast — *do you keep the old easy data or throw it away?* — is the
> most likely exam question, and it directly connects to catastrophic forgetting (file 04, and
> Part II §3). Replaying easy data is the forgetting antidote.

---

## 6. How It Connects

- **Forgetting (file 04 / Part II §3):** One-Pass discards stages → forgetting; Baby Steps / replay
  keep old data → mitigates it. Curriculum design *is* a forgetting-management problem.
- **Multi-task FT (file 07):** you can run a curriculum *over* a multi-task mixture (easy tasks or
  easy examples first).
- **Difficulty via loss/perplexity** is a *model-based* signal — the same loss curves you'll learn
  to read in Part II §7.

---

## 🎯 Interview Questions

**Q1. What is curriculum learning, and what does it change vs. standard training?**
Presenting training examples in a meaningful order (typically easy→hard) instead of random. It
changes only the **order/timing** of examples — not the model, objective, or dataset contents.

**Q2. What are the two core ingredients of any curriculum?**
A **difficulty measure** (what counts as hard) and a **schedule/pacing** (when harder data is
introduced).

**Q3. Give the four benefits of curriculum learning for LLMs.**
Faster convergence (clean early gradients), more stable training (hard/noisy data arrives when the
model can handle it), better generalization (continuation method → flatter minima), and data/compute
efficiency (more from limited data).

**Q4. Describe the three key design decisions for a curriculum.**
(1) **Difficulty measure** — static heuristics (length/rarity), model-based (loss/perplexity), or
external (human labels). (2) **Pacing schedule** — stage boundaries, pacing function
(linear/root/exponential), trigger (steps vs. competence). (3) **Data mixing** — cumulative,
replacement, or replay.

**Q5. Contrast Baby Steps and One-Pass scheduling.**
**Baby Steps** adds each harder bucket while keeping earlier data (cumulative) → robust to
forgetting. **One-Pass** trains each bucket once and discards previous ones → cheap but risks
catastrophic forgetting.

**Q6. What is self-paced learning?**
A curriculum where the **model** defines difficulty: include only samples whose **loss is below a
threshold that rises over time**, so "easy" is set by the learner's current competence rather than
a fixed external measure.

**Q7. When would anti-curriculum (hard-first) make sense?**
When you need **robust features**, or when **easy data dominates and adds little signal**, so
starting with hard examples forces the model to learn the discriminating cases first.

**Q8. How does curriculum learning relate to catastrophic forgetting?**
Directly: discarding earlier (easy) stages (One-Pass) causes forgetting, while keeping/replaying
easy data (Baby Steps, replay) mitigates it. Choosing the **data-mixing** dial *is* managing
forgetting.

---

**One-line takeaway:** *Curriculum fine-tuning orders examples easy→hard to converge faster, train
more stably, and generalize better — fully specified by three dials (difficulty measure, pacing
schedule, data mixing), where the cumulative-vs-replacement choice is really a forgetting-management
decision.*
