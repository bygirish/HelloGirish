# 06 — Instruction Fine-Tuning (IFT)

> **Why this matters:** IFT is *why modern LLMs feel like they "just understand what you ask."*
> It's the difference between a model that rambles about French grammar and one that actually
> *translates your sentence*. It is a **special case of SFT** (file 05), and it powers the
> `IFT_Demo.ipynb` notebook in this session.

---

## 1. Definition

**Instruction Fine-Tuning is a specific type of Supervised Fine-Tuning** where the model is
trained to **follow natural-language instructions**, rather than just perform a single fixed task.

The distinction from plain SFT (slide 46):
- **SFT:** the model learns from input→output pairs (the input can be *plain text*).
- **IFT:** the input is **explicitly written as an instruction**, and the model learns to
  **execute that instruction correctly.**

> 💡 **Learning Thought:** IFT ⊂ SFT. All IFT is SFT, but not all SFT is IFT. The magic
> ingredient is that the *input is phrased as a command*, so the model learns the general skill
> of **"do what the prompt tells me,"** not just one hard-coded task.

---

## 2. Training Data Format ⭐

IFT data is typically a triple: **instruction · input · output** (slide 47).

```json
{
  "instruction": "Translate the following sentence to French.",
  "input": "I love machine learning.",
  "output": "J'aime l'apprentissage automatique."
}
{
  "instruction": "Summarize the following article.",
  "input": "Large language models are pretrained on massive datasets...",
  "output": "LLMs are pretrained on large datasets before being specialized."
}
```

- **`instruction`** — the natural-language command (the new, essential field vs. plain SFT).
- **`input`** — optional context the instruction operates on (may be empty for some tasks).
- **`output`** — the gold response the model learns to imitate.

By training over **many different instructions**, the model generalizes to instructions it has
**never seen** — the hallmark capability of an instruction-tuned model.

---

## 3. The Effect of IFT (slide 48) — the killer demo

| Prompt | **Before IFT** (base model) | **After IFT** |
|--------|-----------------------------|---------------|
| "Translate to French: Good morning" | "French is a Romance language spoken by millions…" | **"Bonjour"** |
| "Write a Python function to compute factorial." | "Python is a programming language. A factorial is a mathematical function…" | **actual working `def factorial(n): …` code** |

The base model **completes text about the topic**; the instruction-tuned model **does the task**.
Same weights family, transformed behavior — purely from the *format* of the training data.

> 💡 **Learning Thought:** Before IFT, an LLM treats your instruction as *text to continue*.
> After IFT, it treats your instruction as *a command to execute*. That single shift is what made
> GPT-3 → InstructGPT → ChatGPT usable by non-experts.

---

## 4. SFT vs. IFT — the comparison table (slide 50)

| | **Supervised Fine-Tuning (SFT)** | **Instruction Fine-Tuning (IFT)** |
|---|---|---|
| **Data** | General labeled input–output pairs | A specialized form using **instruction–response** pairs |
| **Task breadth** | May train on **one** task (e.g. sentiment classification) | Trains on **many** tasks expressed as natural-language instructions |
| **Input** | Can be plain text | Includes an **instruction** (and often extra context) |
| **Goal** | Improve **task performance** | Improve **instruction-following behavior** |

**The essence:** SFT optimizes *"be good at this task."* IFT optimizes *"be good at doing whatever
the instruction says."* IFT generalizes across tasks *because* every example is framed as an
instruction and there are *many* of them.

> 💡 **Learning Thought:** The reason IFT generalizes to unseen instructions is **task diversity
> framed uniformly**: hundreds of different tasks all wear the same `instruction/input/output`
> costume, so the model learns the *meta-skill* of reading and obeying instructions rather than
> memorizing any one task.

---

## 5. Relationship to the Rest of the Lecture

- **IFT is SFT** (Axis A objective = supervised), and can be **full FT or PEFT** (Axis B) — axes
  independent (file 03).
- Doing IFT over **many tasks simultaneously** is essentially **Multi-Task Fine-Tuning** (file
  07) — the two ideas overlap heavily.
- IFT produces the "**instruct**" models (e.g. InstructGPT, Flan-T5) that alignment (RLHF/DPO)
  then polishes further.

---

## 🎯 Interview Questions

**Q1. How is IFT related to SFT?**
IFT is a **special case of SFT**: it's supervised fine-tuning where every input is phrased as a
**natural-language instruction**, so the model learns the general skill of *following
instructions* rather than one fixed task. All IFT is SFT; not all SFT is IFT.

**Q2. Describe the IFT training data format.**
Triples of **instruction / input / output**: a natural-language command, optional context it
operates on, and the gold response to imitate. Training over many diverse instructions yields
generalization to unseen ones.

**Q3. What behavioral change does IFT produce? Give an example.**
It turns *text completion* into *task execution*. "Translate to French: Good morning" → a base
model rambles about the French language; an IFT model outputs **"Bonjour."** Same for "write a
factorial function" → prose vs. actual code.

**Q4. Why does IFT generalize to instructions never seen in training?**
Because many *different* tasks are all framed uniformly as instructions, the model learns the
**meta-skill of reading and obeying an instruction**, not the specifics of any single task — so
new instructions in the same format are handled.

**Q5. Give two concrete differences between SFT and IFT.**
(1) SFT input can be plain text; IFT input is explicitly an instruction. (2) SFT may target a
single task and optimizes *task performance*; IFT spans many tasks and optimizes
*instruction-following behavior*.

**Q6. Is IFT a full-fine-tuning or a PEFT technique?**
Neither — it's an **objective** (a kind of SFT). It can be executed as full FT *or* PEFT/LoRA; the
objective and the parameter strategy are independent choices.

**Q7. Name a model whose usability jump came largely from instruction tuning.**
InstructGPT (instruction-tuned GPT-3) — and the Flan-T5 family. Instruction tuning is what made
these models follow user commands out of the box.

---

**One-line takeaway:** *IFT is SFT whose inputs are framed as natural-language instructions across
many tasks — teaching the model the meta-skill of "do what I ask," which is exactly what turned raw
LLMs into usable assistants.*
