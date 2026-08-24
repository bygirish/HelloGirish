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

Each section now has a **🧪 block** that pulls the *actual code* from these notebooks so you
can connect theory to a runnable line. Where the real snippet lives:

| Notebook | Topic | Reinforces | Real code appears in |
|----------|-------|-----------|----------------------|
| Demo 1 — `LW-BW-P-FT-and-LRs` | Layer/Block/Progressive FT + LR schedules | Sections 1, 3, 4, 7 | §3.8 (freezing), §4.6 (scheduler A/B), §7.4 (plotting) |
| Demo 2 — `sst2-with-cleanlab` | Finding label errors with confident learning | Section 6 | §6.1 (k-fold OOF + cleanlab) |
| Demo 3 — `Catastrophic-Forgetting` | Measuring & mitigating forgetting | Sections 1, 3 | §1.1 (training loop), §3.3 (forget→replay demo) |

> **Running them:** all three are Colab-ready. They install `transformers`, `datasets`,
> `evaluate`, `cleanlab` and pull models/data from the Hugging Face Hub (`distilbert-base-uncased`,
> `bert-base-uncased`, SST-2 / AG News / CoLA / STS-B). A free Colab T4 GPU is enough — they use
> small subsets (2k–10k examples) so each finishes in minutes.

> **Seeing the diagrams:** the `mermaid` flowcharts render automatically on **GitHub** and in
> **VS Code** with the *Markdown Preview Mermaid Support* extension. The ASCII diagrams render
> everywhere. Open any `.md` in VS Code and press `Cmd+K V` for a live preview.

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

---

## 🔗 Global resources (start here if you want the big picture first)

Each section file ends with its own focused **🔗 Further reading**. These are the broad,
cross-cutting resources for the *whole* topic of fine-tuning:

- **Andrej Karpathy — [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html):**
  the best free course for genuinely understanding §1's training loop from scratch.
- **Hugging Face — [LLM Course](https://huggingface.co/learn/llm-course) & [Trainer docs](https://huggingface.co/docs/transformers/main_classes/trainer):**
  the library every demo here uses; the `TrainingArguments` reference is your knob catalog for
  §2, §4, §5, §7.
- **Sebastian Raschka — [Blog](https://magazine.sebastianraschka.com/) & [*Build a Large Language
  Model (From Scratch)*](https://github.com/rasbt/LLMs-from-scratch):** superb, practical,
  frequently-updated writing on fine-tuning, LoRA, and instabilities.
- **[The Novice's LLM Training Guide](https://rentry.org/llm-training)** and
  **[Weights & Biases — LLM fine-tuning guides](https://wandb.ai/site/articles):** end-to-end,
  hands-on walkthroughs that touch every section here.
- **[Hugging Face PEFT](https://huggingface.co/docs/peft)** & **[LoRA paper](https://arxiv.org/abs/2106.09685):**
  the parameter-efficient methods that make most of §2/§3/§5's pain cheaper to manage — the
  natural "Part III" after these notes.
