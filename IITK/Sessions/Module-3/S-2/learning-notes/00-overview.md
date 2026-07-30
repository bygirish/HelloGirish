# LLM Fine-Tuning Fundamentals – II
## Data Preparation and Training Mechanisms — Study Notes

> Lecture by Jiaul Paik, Associate Professor, Dept. of AI, IIT Kharagpur.
> This is **Part II** of the fine-tuning series. Part I covered *when* to fine-tune
> (RAG vs. Prompt vs. Fine-tuning) and the *broad approaches* (full FT, layerwise,
> block-wise, progressive). Part II is about **how training actually works and how to
> keep it from breaking**.

---

### How to use these notes

Each file is a self-contained lesson for one section. Read them in order the first
time — later sections assume the vocabulary from earlier ones. Every file follows the
same structure:

1. **Why this matters** — the one-paragraph motivation.
2. **Concept-by-concept elaboration** — intuition → mechanism → the knob you actually set.
3. **💡 Learning Thoughts** — highlighted mental models worth memorizing.
4. **🎯 Interview Questions** — with model answers, ranging from basic to senior-level.
5. **One-line takeaway.**

---

### The 7 sections

| # | File | Section | Core question |
|---|------|---------|---------------|
| 1 | [01-how-llms-learn.md](01-how-llms-learn.md) | How LLMs Learn | What is the training loop, mechanically? |
| 2 | [02-training-instability.md](02-training-instability.md) | Training Instability | Why do runs blow up, and how do I stabilize them? |
| 3 | [03-catastrophic-forgetting.md](03-catastrophic-forgetting.md) | Catastrophic Forgetting | Why does the model forget, and how do I prevent it? |
| 4 | [04-learning-rate-scheduling.md](04-learning-rate-scheduling.md) | Learning-Rate Scheduling | How should the step size change over a run? |
| 5 | [05-choosing-batch-size.md](05-choosing-batch-size.md) | Choosing Batch Size | How big should each gradient step's sample be? |
| 6 | [06-preparing-the-data.md](06-preparing-the-data.md) | Preparing the Data | What silently breaks models before training even starts? |
| 7 | [07-reading-loss-curves.md](07-reading-loss-curves.md) | Reading the Loss Curves | How do I diagnose what's going wrong from the curves? |

### The 3 hands-on notebooks

| Notebook | Topic | Reinforces |
|----------|-------|-----------|
| Demo 1 — `LW-BW-P-FT-and-LRs` | Layer/Block/Progressive FT + LR schedules | Sections 1 & 4 |
| Demo 2 — `sst2-with-cleanlab` | Finding label errors with confident learning | Section 6 |
| Demo 3 — `Catastrophic-Forgetting` | Measuring & mitigating forgetting | Section 3 |

---

### The single mental model that ties it all together

Fine-tuning is **repeatedly taking small, averaged, downhill steps on cross-entropy loss.**
Everything else in this lecture is a lens on that one sentence:

- **Instability** (§2) = a step was too **big**.
- **Forgetting** (§3) = steps dragged the weights too **far** from where they started.
- **LR scheduling** (§4) = controlling step **size** over time.
- **Batch size** (§5) = controlling step **noise**.
- **Data prep** (§6) = making sure the downhill direction points somewhere **worth going**.
- **Loss curves** (§7) = reading the **feedback** to know which of the above is wrong.
