# Module 3 · Session 6 — LLM Fine-Tuning & Small Language Models
### Learning Notes — Index

**Lecture:** *Revision of Fine-tuning Concepts and Hands-on Small Language Models*
**Instructor:** Jiaul Paik, Associate Professor, Dept of AI, IIT Kharagpur
**Deck:** 64 slides · **Notebook:** SFT / DPO / PII / Red-Team (71 cells) · **Duration:** ~3h

---

## How to use these notes

Each section file follows the same structure:

- **Topic-by-topic elaboration** — expanded beyond the slide text, with transcript quotes where the instructor's phrasing was clarifying
- **📊 Slide figures** — the actual lecture diagrams, extracted from the deck into [`assets/`](assets/)
- **💻 Runnable code** — notebook cells where they exist, plus from-scratch implementations (LoRA, NF4, DPO loss) written to be pasted into a REPL
- **💡 Learning thoughts** — the insight behind the mechanism; *the highest-value paragraphs in each file*
- **📚 Go deeper** — papers, docs, and videos for each topic
- **🎯 Interview questions with full answers** — 6–12 per section
- **✅ Self-check** — test yourself before moving on

**Suggested path:** read a section → run its code → do the self-check from memory → cover the interview answers and give them out loud. Return to Sections 4–6 twice; they carry the most weight.

---

## Sections

| # | File | Slides | Core question it answers |
|---|---|---|---|
| 1 | [Foundations & Framing](01-foundations-and-framing.md) | 2–6 | *When should I fine-tune at all?* |
| 2 | [Taxonomy & SFT](02-finetuning-taxonomy-and-sft.md) | 7–16 | *What kind of fine-tuning, and how do I format data?* |
| 3 | [PEFT: Why & Which](03-peft-why-and-which.md) | 17–20 | *Why can't I just fine-tune everything?* |
| 4 | [LoRA](04-lora.md) | 21–30 | *How do I train 0.5% of parameters and still adapt the model?* |
| 5 | [QLoRA](05-qlora.md) | 31–47 | *How do I fit a 65B model on one GPU?* |
| 6 | [Alignment: RLHF, PPO, DPO](06-alignment-rlhf-ppo-dpo.md) | 48–63 | *How do I teach preference, not just correctness?* |
| 7 | [Hands-on: Guardrails & Red-Team](07-handson-guardrails-and-redteam.md) | Notebook | *How do I ship this safely?* |

---

## Code you can run

Each file contains working snippets. The ones worth actually typing out:

| Section | Snippet | What it teaches |
|---|---|---|
| 1 | `named_parameters()` inspection | Which matrices exist and how big they are |
| 2 | Loss masking with `-100` labels | Why SFT trains on completions only |
| 2 | `apply_chat_template()` across 3 models | Why template mismatch is the #1 bug |
| 3 | `training_memory_gb()` calculator | Full vs LoRA vs QLoRA, any model size |
| 3 | Adam optimizer state inspection | Where the 8 bytes/param actually go |
| 4 | **`LoRALinear` from scratch** | The whole method in 25 lines |
| 4 | Zero-init gradient demo | Why B=0 and A~Normal, provably |
| 4 | SVD of ΔW | *Verify* the low-rank hypothesis, don't assume it |
| 5 | **NF4 quantize/dequantize** | Reproduces the slide's exact indices `[0,3,5,8,11,14]` |
| 5 | Outlier quantization demo | Why block-wise quantization exists (200× error) |
| 5 | `BitsAndBytesConfig` | All four QLoRA ideas as four arguments |
| 6 | **`dpo_loss()` implementation** | Run it on slide 62's numbers |
| 6 | Bradley–Terry loss table | Why `−log σ(difference)` behaves correctly |
| 7 | Full SFT → DPO → guard → red-team | The complete pipeline |
| 7 | Preference-pair confound check | Catch length bias before you train |

---

## Full topic list (40 topics)

<details>
<summary><b>Section 1 — Foundations &amp; Framing</b> (Slides 2–6)</summary>

1. Session goals · 2. Three pathways: Prompting vs RAG vs Fine-tuning · 3. What exactly do we fine-tune? · 4. Training stages · 5. Dermatology example
</details>

<details>
<summary><b>Section 2 — Taxonomy &amp; SFT</b> (Slides 7–16)</summary>

6. Two-axis taxonomy · 7. Unsupervised / continued pre-training · 8. SFT · 9. Alignment overview · 10. Data format · 11. Instruction fine-tuning · 12. Multi-task fine-tuning
</details>

<details>
<summary><b>Section 3 — PEFT</b> (Slides 17–20)</summary>

13. Full-FT memory math (120–160 GB) · 14. PEFT taxonomy · 15. Soft prompting · 16. Prefix tuning
</details>

<details>
<summary><b>Section 4 — LoRA</b> (Slides 21–30)</summary>

17. Reparameterization PEFT · 18. Gradient descent recap · 19. Freeze W, train B·A, merge · 20. Initialization & gradient flow · 21. The remaining 40 GB
</details>

<details>
<summary><b>Section 5 — QLoRA</b> (Slides 31–47)</summary>

22. Four key ideas · 23. What is quantization · 24. FP32→INT8 formula · 25. NF4 (equal-sized buckets) · 26. Codebook & worked example · 27. Double quantization · 28. Gradient checkpointing · 29. Paged optimizer
</details>

<details>
<summary><b>Section 6 — Alignment</b> (Slides 48–63)</summary>

30. HHH · 31. RLHF motivation · 32. RLHF pipeline · 33. Why a reward model · 34. Bradley–Terry loss · 35. PPO pieces · 36. PPO ratio & clipping · 37. DPO: no reward model · 38. DPO data · 39. Relative probabilities · 40. DPO loss
</details>

<details>
<summary><b>Section 7 — Hands-on</b> (Notebook)</summary>

Small language models · tokenization & chat templates · LoRA/SFT with TRL · adapter merging · DPO · PII detection (regex + Luhn) · content filtering · GuardedLLM wrapper · red-team suite & scoring · hardening against obfuscation
</details>

---

## The four ideas to carry out of this session

**1. The two-axis taxonomy.** Every technique has two coordinates: *what you teach* (unsupervised / supervised / alignment) and *how many weights move* (full / PEFT). "QLoRA DPO" = alignment × PEFT. Decompose any technique name this way.

**2. Training memory has four components, and each technique kills exactly one.**
```
Optimizer states  →  LoRA (freeze base, train adapters)
Gradients         →  LoRA (same mechanism)
Weights           →  Quantization / NF4
Activations       →  Gradient checkpointing
(+ spikes         →  Paged optimizer)
```
This table answers almost any LLM memory-optimisation question.

**3. Alignment exists because judging is easier than writing.** SFT maximises the likelihood of one gold answer and has no notion of "better." Most prompts have many correct answers of differing quality. Humans are far better at ranking than authoring — preference learning monetises that asymmetry. DPO then removes the reward model by proving the optimal reward is an analytic function of the policy-to-reference log-ratio.

**4. Training is not a safety mechanism.** Alignment shifts a distribution; guardrails are deterministic code. Ship both, then red-team the combination — with benign cases included, scored per category, and every attack you've seen kept as a permanent regression suite.

---

## 📚 Core resource library

**Read these five papers in this order** — they *are* the session:
1. [LoRA (Hu et al., 2021)](https://arxiv.org/abs/2106.09685) — Section 4
2. [QLoRA (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314) — Section 5
3. [InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155) — Sections 2 & 6
4. [DPO (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290) — Section 6
5. [LIMA (Zhou et al., 2023)](https://arxiv.org/abs/2305.11206) — data quality over quantity

**Libraries used throughout:**
- [PEFT](https://huggingface.co/docs/peft) · [TRL](https://huggingface.co/docs/trl) · [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) · [Transformers](https://huggingface.co/docs/transformers)

**Best free explainers:**
- [Illustrating RLHF (HF)](https://huggingface.co/blog/rlhf) — the alignment pipeline, visually
- [LoRA Insights (Raschka)](https://lightning.ai/pages/community/lora-insights/) — practical ablations on r, α, targets
- [Transformer Math 101 (EleutherAI)](https://blog.eleuther.ai/transformer-math/) — the memory arithmetic reference
- [The Illustrated Transformer (Alammar)](https://jalammar.github.io/illustrated-transformer/)

**Hands-on next steps:**
- [HF Alignment Handbook](https://github.com/huggingface/alignment-handbook) — production SFT+DPO recipes
- [Unsloth](https://github.com/unslothai/unsloth) — 2× faster LoRA/QLoRA, same API
- [garak](https://github.com/NVIDIA/garak) — automated LLM red-teaming
- [Microsoft Presidio](https://microsoft.github.io/presidio/) — production PII detection

---

## Source files in this folder

| File | What it is |
|---|---|
| `691f7877-…-llm-fine-tuning-15-08-26.pptx` | The 64-slide deck |
| `78f7b4fa-…-llm-hands-on-sft-dpo-pii-rtt.ipynb` | The hands-on notebook (71 cells) |
| `transcript.text` | Full session transcript with timestamps |
| `81a9bd6a-…-Q-A.xlsx` | Live Q&A — 22 questions with instructor answers |
| `1ce975da-…-pre-read.pdf` / `33ab21f2-…-post-read.pdf` | Pre/post reads |
| `learning-notes/assets/` | 27 figures extracted from the deck, embedded in the section files |

> Q&A answers are quoted throughout where they clarify a slide — they're often sharper than the slides, because they answer what people actually got stuck on.
