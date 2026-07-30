# 05 — Supervised Fine-Tuning (SFT)

> **Why this matters:** SFT is the workhorse recipe of modern LLMs — it's the stage that turns a
> raw next-token predictor into something that *behaves* like an assistant. Every chat model you
> use went through SFT. It's also the direct parent of Instruction Fine-Tuning (file 06) and the
> base for the `SFT_Demo.ipynb` notebook in this session.

---

## 1. Definition

> **Supervised fine-tuning (SFT) is the process of taking a pretrained language model and
> teaching it to perform a specific task by training it on input–output pairs created by
> humans.**

Unpack the three load-bearing words:
- **Pretrained** — you start from a model that already knows language (file 01). SFT doesn't
  teach English; it teaches *how to respond*.
- **Input–output pairs** — the training data is `(prompt, desired response)` tuples. This is the
  "supervised" part: every example has a **gold target**.
- **Created by humans** — the targets are demonstrations of the behavior you want. The model
  learns to **imitate them token by token** (standard cross-entropy on the output tokens).

> 💡 **Learning Thought:** SFT is **imitation learning for text**. The model isn't told *why* an
> answer is good — only shown *what* a good answer looks like — and it learns to reproduce that
> distribution. (Contrast: alignment/RLHF in file 03 teaches it *which of two answers is
> better*.)

---

## 2. The Simple Analogy (slide 43) — worth remembering verbatim

Imagine you hire someone who **already knows English very well**:
- During **pretraining**, they read millions of books and articles → they know the language.
- During **supervised fine-tuning**, you hand them a **manual**:

  | User asks | Correct response |
  |-----------|------------------|
  | "What are your business hours?" | "We're open from 9 AM to 6 PM." |
  | "How do I reset my password?" | "Click 'Forgot Password' on the login page." |

- After seeing **thousands of examples**, they learn **how you want them to answer.**

> 💡 **Learning Thought:** Pretraining hires a fluent generalist; SFT hands them your company
> playbook. You're not teaching *language*, you're teaching *your* expected input→output behavior.

---

## 3. How SFT Works Mechanically

1. Start from pretrained weights.
2. Format each example as `(input, target output)`.
3. Feed the input, let the model predict the output tokens.
4. Compute **cross-entropy loss on the target tokens** (typically only the *response* tokens are
   used for loss, not the prompt).
5. Backprop and update weights (full FT *or* PEFT — recall the axes are independent, file 03).
6. Repeat over thousands of examples until the behavior generalizes.

It is "supervised" in the classic ML sense: **labeled data**, gold targets, a loss against them.
The novelty over ordinary supervised learning is only that the "label" is a *token sequence* the
model generates autoregressively.

---

## 4. What SFT Teaches

- **Task skills** — how to do the specific task (classification, extraction, drafting, etc.).
- **Behavior and format** — instruction following, tone, and structure (JSON, bullet lists).
- **The "assistant" persona** — turning a text-completer into something that *answers*.

**Landmark:** the **SFT stage of InstructGPT** was powered by **human-written demonstrations** —
labelers wrote ideal responses to prompts, and the model imitated them. That SFT stage is what
first made GPT-3 usable as an instruction-following assistant (before the later RLHF stage).

> 💡 **Learning Thought:** SFT is stage *one* of the modern alignment pipeline: **SFT (imitate
> good answers) → RLHF/DPO (prefer better answers).** SFT gets you 80% of the way to a usable
> assistant; alignment polishes the rest.

---

## 5. SFT's Limits (which motivate later recipes)

- **Data cost:** every example needs a **human-written gold output** — far more expensive than
  the raw text used in unsupervised FT (file 03).
- **Only imitates, never judges:** SFT can't learn from *"answer A is better than answer B"* —
  that requires preference/alignment methods.
- **Single-task narrowness:** vanilla SFT may train on *one* task; generalizing to *follow any
  instruction* is exactly what **Instruction Fine-Tuning (file 06)** adds, and handling *many
  tasks at once* is **Multi-Task FT (file 07)**.

---

## 🎯 Interview Questions

**Q1. Define SFT in one sentence.**
Taking a pretrained model and teaching it a task by training it on **human-created input–output
pairs**, so it learns to imitate the demonstrated outputs token by token.

**Q2. What makes SFT "supervised," and what's the loss?**
Each training example has a **gold target output** (the label). The loss is **cross-entropy on
the target/response tokens** between the model's predicted distribution and the human-written
answer.

**Q3. Using the hiring analogy, what do pretraining and SFT each contribute?**
Pretraining = hiring someone already fluent in English (general language ability). SFT = handing
them your company manual of question→answer examples so they learn *how you want things
answered*.

**Q4. Where does SFT sit in the modern LLM training pipeline, and what comes after?**
It's the first behavior-shaping stage after pretraining. It's typically followed by
**alignment** (RLHF/DPO), which improves on SFT by learning from *preferences between* answers
rather than just imitating one gold answer.

**Q5. Why is SFT data more expensive than unsupervised fine-tuning data?**
Unsupervised FT reuses raw unlabeled text (label = next token, free). SFT requires **humans to
write the correct output** for each input, which is slow and costly.

**Q6. Is SFT necessarily full fine-tuning?**
No. SFT is an *objective* (Axis A); it can be run as full FT *or* PEFT/LoRA (Axis B). The two
choices are independent.

**Q7. What's one thing SFT fundamentally cannot learn, and which method fixes it?**
It can't learn *relative quality* ("A is better than B") — it only imitates a single gold answer.
**Preference-based alignment** (RLHF / DPO) fixes this by training on ranked or (chosen, rejected)
pairs.

---

**One-line takeaway:** *SFT is imitation learning on human-written input→output pairs — it turns
a fluent pretrained model into a task-performing assistant by showing it thousands of "here's how
to answer" examples, and it's the first stage of the modern alignment pipeline.*
