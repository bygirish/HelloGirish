# 07 — Multi-Task Fine-Tuning

> **Why this matters:** Instead of maintaining four separate models for four tasks, you train
> **one** model on **all** of them. This is how you get a single deployable model that's a
> jack-of-many-trades, and it's the foundation of instruction-tuned generalists (Flan-T5, etc.).
> It powers the `Multi_Task_FFT.ipynb` notebook in this session.

---

## 1. Definition

**Multi-task fine-tuning is an approach where a single model is fine-tuned on *multiple related
tasks simultaneously*, instead of training it for just one task.**

The payoff (slide 52): *"Instead of fine-tuning four separate models, you fine-tune **one** model
using data from all four tasks."* One model to train, deploy, monitor, and update — not four.

---

## 2. How It Works

1. **Start from a pretrained model** (e.g. BERT, T5, Llama).
2. **Build a mixed training dataset** containing examples from **several tasks**.
3. The model **learns a shared representation** while **optimizing for each task's objective**.

**The training data looks like this (slide 52):**

| Input | Task | Output |
|-------|------|--------|
| "I love this movie." | Sentiment | Positive |
| "Who discovered gravity?" | QA | Isaac Newton |
| *(a long article)* | Summarization | *(a short summary)* |
| "Google is in California." | NER | Google (Organization), California (Location) |

All four tasks flow through **one** set of weights during training. Often each example carries a
task tag or is framed as an instruction (which is exactly why **multi-task FT and IFT overlap** —
see file 06).

> 💡 **Learning Thought:** The magic phrase is **"shared representation."** The model builds
> internal features useful across *all* the tasks; skills learned on one task (e.g. parsing
> entities for NER) can **transfer** and help another (e.g. QA). This positive transfer is the
> whole reason to co-train rather than train separately.

---

## 3. Why Do It — Benefits

- **Efficiency:** one model instead of *N* — less training overhead, one artifact to store,
  serve, and update. (Recall from file 04 that each full-FT model is a full checkpoint; N of them
  is N× the storage.)
- **Positive transfer / better generalization:** related tasks reinforce shared features; the
  model often generalizes better than *N* narrow specialists, especially where per-task data is
  scarce.
- **Foundation for instruction generalists:** training on many tasks at once (framed as
  instructions) is exactly what yields models that follow *arbitrary* instructions (link to IFT,
  file 06).

---

## 4. The Trade-offs (know these for interviews)

The slides sell the upside; a strong answer also names the tensions:

- **Task balancing / interference:** if one task dominates the data, the model skews toward it;
  poorly related tasks can *hurt* each other ("negative transfer"). Mitigated by **weighting /
  sampling** tasks appropriately.
- **Data-mixture design:** deciding *how much* of each task to include is a real tuning problem.
- **Conflicting objectives:** different tasks may pull the shared weights in different
  directions; the shared representation is a compromise.

> 💡 **Learning Thought:** Multi-task FT is a **balancing act on a shared representation**.
> Related tasks → positive transfer (1+1 > 2). Unrelated/imbalanced tasks → negative transfer
> (1+1 < 2). The art is choosing the task mix and their sampling proportions.

---

## 5. Relationship to Neighboring Recipes

- **vs. single-task SFT (file 05):** SFT on one task = one specialist; multi-task FT = one
  generalist across tasks.
- **vs. IFT (file 06):** if the many tasks are each framed as natural-language instructions,
  multi-task FT *becomes* instruction tuning. IFT is essentially "multi-task FT where the task is
  specified by an instruction."
- **vs. Curriculum FT (file 08):** curriculum orders examples *easy→hard over time*; multi-task
  mixes *tasks* together. They can be combined (a curriculum *over* a multi-task mixture).

---

## 🎯 Interview Questions

**Q1. What is multi-task fine-tuning and what's its main practical benefit?**
Fine-tuning a *single* model on *several related tasks at once* (with a mixed dataset) so it
learns a **shared representation**. Main benefit: one model to train/deploy/maintain instead of N,
plus positive transfer across related tasks.

**Q2. What does "shared representation" mean and why is it valuable?**
The model develops internal features useful across all the trained tasks. It's valuable because
skills learned on one task transfer to others, improving generalization — especially when some
tasks have little data.

**Q3. Give an example training dataset for multi-task FT.**
A mixed set spanning tasks: sentiment ("I love this movie."→Positive), QA ("Who discovered
gravity?"→Isaac Newton), summarization (article→summary), NER ("Google is in California."→Google
[ORG], California [LOC]) — all fed through one model.

**Q4. What is negative transfer, and how do you mitigate it?**
When unrelated or imbalanced tasks pull the shared weights in conflicting directions and *hurt*
each other. Mitigate by **task weighting/sampling**, curating a related task mix, and tuning the
data proportions.

**Q5. How does multi-task fine-tuning relate to instruction tuning?**
IFT is essentially multi-task FT where each task is expressed as a natural-language instruction.
Training on many instruction-framed tasks simultaneously is what produces instruction-following
generalists.

**Q6. Why choose one multi-task model over N single-task models?**
Lower training/storage/serving overhead (one artifact vs. N full checkpoints) and positive
transfer that can beat N narrow specialists — at the cost of managing task balance and possible
interference.

**Q7. Name a concrete architecture/model family associated with multi-task/instruction fine-tuning.**
T5 / **Flan-T5** (and BERT/Llama as starting points) — Flan-T5 in particular is instruction-tuned
across a large collection of tasks.

---

**One-line takeaway:** *Multi-task fine-tuning trains one model on many related tasks at once,
learning a shared representation that enables positive transfer — one artifact instead of N, and
the direct precursor of instruction-following generalists.*
