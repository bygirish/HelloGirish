# LLM Fine-Tuning Fundamentals – I
## Decision Framework & Fine-tuning Approaches — Study Notes

> Lecture by Jiaul Paik, Associate Professor, Dept. of AI, IIT Kharagpur.
> This is **Part I** of the fine-tuning series. It answers two big questions:
> **(a) *Should* you fine-tune at all** (RAG vs. Prompt vs. Fine-tuning), and
> **(b) *What kinds* of fine-tuning exist** (the full taxonomy + the classic recipes).
> **Part II** (S-2) then goes into *how training actually works mechanically* and how to
> keep it from breaking (instability, forgetting, LR schedules, batch size, data prep).

---

### How to use these notes

Each file is a self-contained lesson for one topic. Read them **in order** the first
time — later files assume vocabulary from earlier ones. Every file follows the same
structure (the same one your S-2 notes use):

1. **Why this matters** — the one-paragraph motivation.
2. **Concept-by-concept elaboration** — intuition → mechanism → the knob you actually set.
3. **💡 Learning Thoughts** — highlighted mental models worth memorizing.
4. **🎯 Interview Questions** — with model answers, basic → senior level.
5. **One-line takeaway.**

---

### The 10 files

| # | File | Section in deck | Core question |
|---|------|-----------------|---------------|
| 0 | [00-overview.md](00-overview.md) | — | How do the pieces fit together? |
| 1 | [01-fundamentals.md](01-fundamentals.md) | Fundamentals (slides 4–20) | What is an LM/LLM, and where does fine-tuning live? |
| 2 | [02-decision-framework.md](02-decision-framework.md) | Decision Framework (21–25) | Prompt vs. RAG vs. Fine-tuning — pick one. |
| 3 | [03-fine-tuning-approaches.md](03-fine-tuning-approaches.md) | FT Approaches (26–38) | The full taxonomy: two axes, PEFT families. |
| 4 | [04-full-fine-tuning.md](04-full-fine-tuning.md) | Full Fine-tuning (33, 39) | Update *every* weight — capacity, cost, forgetting. |
| 5 | [05-supervised-fine-tuning.md](05-supervised-fine-tuning.md) | SFT (40–44) | Learning from human-written input→output pairs. |
| 6 | [06-instruction-fine-tuning.md](06-instruction-fine-tuning.md) | IFT (45–50) | Teaching the model to *follow instructions*. |
| 7 | [07-multi-task-fine-tuning.md](07-multi-task-fine-tuning.md) | Multi-Task FT (51–53) | One model, many tasks, shared representation. |
| 8 | [08-curriculum-fine-tuning.md](08-curriculum-fine-tuning.md) | Curriculum FT (55–60) | Training easy→hard, and why it helps. |
| 9 | [09-making-full-ft-efficient.md](09-making-full-ft-efficient.md) | Efficient Full FT (61–68) | The memory math + layer/block/progressive tricks. |

### The 3 hands-on notebooks in this session

| Notebook | Topic | Reinforces file |
|----------|-------|-----------------|
| `SFT_Demo.ipynb` | Supervised fine-tuning | 05 |
| `IFT_Demo.ipynb` | Instruction fine-tuning | 06 |
| `Multi_Task_FFT.ipynb` | Multi-task full fine-tuning | 07 |

---

### The single mental model that ties Part I together

There are **three levers** to change an LLM's behavior, in increasing order of cost and permanence:

```
Prompt Engineering  →  RAG  →  Fine-tuning
(change instructions)  (add facts)  (change the weights)
```

Once you've decided to **fine-tune**, you make two independent choices:

1. **What objective?** (the *recipe*) — Unsupervised · Supervised (SFT/IFT) · Alignment
2. **How many weights move?** (the *strategy*) — Full FT vs. PEFT (LoRA & friends)

> **The one sentence to remember:** *Fine-tuning = taking a pre-trained generalist and
> nudging some of its weights toward a specialist, where you separately pick the **objective**
> (what it learns) and the **parameter strategy** (how much of it changes).*

Everything in Part I is a branch of that tree. Part II is about walking down the branch safely.

---

### Prerequisites this lecture assumes (from earlier modules)
- Deep learning basics (gradients, backprop, loss).
- The Transformer (self-attention, feed-forward, layer norm).
- Prompt Engineering and Retrieval-Augmented Generation (RAG) — covered so you can
  *contrast* them against fine-tuning in file 02.
