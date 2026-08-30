# Module 3 · Session 4 — PEFT: Re-Parameterization (LoRA & QLoRA)

**Instructor:** Dr. Plaban Kumar Bhowmick, IIT Kharagpur
**Date:** 1 August · Executive Certificate Programme, Generative AI and Agentic AI
**Source deck:** `3-2b-PEFT-Reparameterization.pptx` (58 slides)

---

## How to use these notes

Six files, one per section of the lecture. Each file has the same shape:

| Block | What it gives you |
|---|---|
| **Topic walkthrough** | Every slide-topic elaborated, with the math worked end-to-end |
| 🖼️ **Slide diagrams** | The actual figures from the deck, extracted into [`assets/`](assets/) |
| 💻 **Code** | Runnable snippets — notebook code where it exists, purpose-written where it doesn't |
| 💡 **Learning Thought** | The idea underneath the mechanics — the thing worth remembering in 6 months |
| ⚠️ **Trap** | Misconceptions from the live Q&A, and why they're wrong |
| 🎯 **Interview Questions** | Asked-in-the-wild questions with model answers |
| 📚 **Go deeper** | Papers, blog posts and docs for advancing past the lecture |
| ✅ **Self-check** | Questions to test yourself before moving on |

**Suggested path:**
1. **Understanding pass** — read A → F, skipping the interview blocks. Look at
   every diagram. Don't run code yet.
2. **Practical pass** — run every code snippet. The short ones are designed to
   be pasted into a REPL and produce output you can reason about.
3. **Expertise pass** — interview blocks, the demo notebook end-to-end, the
   ablations in [Section F](F-synthesis-and-practice.md#-where-to-go-next).

---

## Section files

| File | Section | Slides | Core question it answers |
|---|---|---|---|
| [A — Motivation & Taxonomy](A-motivation-and-taxonomy.md) | Orientation | 1–8 | Why does re-parameterization exist as a category at all? |
| [B — LoRA Theory](B-lora-theory.md) | Re-parameterization | 9–17 | Why is a low-rank update *enough*? |
| [C — Quantization Foundations](C-quantization-foundations.md) | Quantized PEFT | 18–28 | How do you store a number in fewer bits without wrecking it? |
| [D — QLoRA's Three Ingredients](D-qlora-three-ingredients.md) | QLoRA core | 29–40 | NF4, double quantization — what exactly do they buy? |
| [E — Memory Engineering](E-memory-engineering.md) | Fitting it | 41–48 | Gradient checkpointing and paged optimizers |
| [F — Synthesis & Practice](F-synthesis-and-practice.md) | Putting it together | 49–57 | Can a 70B model fit on one GPU? |

---

## The spine of the whole session (memorize this chain)

```
Intrinsic dimension is small
        ↓
so ΔW has low intrinsic rank
        ↓
so ΔW = BA  with r ≪ d              ← LoRA  (fewer params to TRAIN)
        ↓
but the frozen W₀ still occupies FP16
        ↓
so quantize W₀ to 4 bits            ← QLoRA (fewer bits to STORE)
        ↓
uniform quantization breaks on outliers  → block-wise quantization
block-wise costs constants               → double quantization
activations still blow up memory          → gradient checkpointing
spikes still OOM                          → paged optimizer
        ↓
70B fine-tuning on ONE GPU
```

Every topic in this session is a node on that chain. If you can redraw this
from memory and say *what problem each arrow solves*, you have the session.

---

## The three numbers that anchor everything

| Recipe | Bits / param | 70B model | GPUs |
|---|---|---|---|
| Full fine-tuning | 96 (12 bytes) | **840 GB** | ~20 |
| LoRA | ~17.6 | **154 GB** | ~4 |
| QLoRA | ~5.2 | **46 GB** | **1** |

If you can derive each row from first principles (see [Section C](C-quantization-foundations.md)),
you can answer most memory-budget interview questions on the spot.

---

## The one equation that contains the session

```
Y^BF16 = X^BF16 · doubleDequant(c₁^FP32, c₂^k-bit, W₀^NF4)
         + (α/r) · X^BF16 · B^BF16 · A^BF16
```

Every superscript is a decision from a different section — see
[§F1](F-synthesis-and-practice.md#the-full-equation-slide-55) for the
term-by-term trace.

---

## Code index — where to find each implementation

| What | File | Source |
|---|---|---|
| The four PEFT families, side by side | [A](A-motivation-and-taxonomy.md#seeing-the-difference-in-code) | written for these notes |
| Multi-tenant adapter hot-swap | [A](A-motivation-and-taxonomy.md#what-hot-swapping-looks-like-in-code) | `peft` API |
| Memory accounting (`RECIPES`) | [A](A-motivation-and-taxonomy.md#-notebook-link) | notebook §1 |
| Intrinsic-dimension subspace probe | [B](B-lora-theory.md#how-li-et-al-actually-measured-this) | written for these notes |
| Bottleneck rank demonstration | [B](B-lora-theory.md#verify-it-and-see-what-you-gave-up) | written for these notes |
| **Is ΔW actually low-rank?** (SVD test) | [B](B-lora-theory.md#see-it-for-yourself--is-δw-really-low-rank) | written for these notes |
| `MinimalLoRA` — 15-line version | [B](B-lora-theory.md#the-minimal-implementation) | written for these notes |
| α/r magnitude experiment | [B](B-lora-theory.md#measure-the-effect-in-five-lines) | written for these notes |
| **Initialization gradient probe** | [B](B-lora-theory.md#prove-the-alternatives-break) | written for these notes |
| `LoRALinear` — production version | [B](B-lora-theory.md#-notebook-link--the-production-grade-version) | notebook §2 |
| `find_lora_targets` / `inject_lora` | [B](B-lora-theory.md#-notebook-link--the-production-grade-version) | notebook §3 |
| `merge_lora_` — zero-latency fold | [B](B-lora-theory.md#b7b-no-additional-inference-latency-pre-read) | notebook §2 |
| Adapter-size calculator | [B](B-lora-theory.md#as-a-reusable-function) | written for these notes |
| Optimizer-state measurement | [C](C-quantization-foundations.md#where-the-8-bytes-of-optimizer-state-comes-from) | written for these notes |
| `quantize_int8` / `dequantize_int8` | [C](C-quantization-foundations.md#in-code--6-lines-and-you-should-be-able-to-write-them-cold) | written for these notes |
| **Outlier collapse demonstration** | [C](C-quantization-foundations.md#demonstrate-the-collapse) | written for these notes |
| Block-wise quantization | [C](C-quantization-foundations.md#implement-it-and-watch-the-fix-work) | written for these notes |
| `measure_quantization` (FP16 vs NF4) | [C](C-quantization-foundations.md#-notebook-link) | notebook §4 |
| NF4 codebook derivation | [D](D-qlora-three-ingredients.md#derive-the-nf4-levels-yourself) | written for these notes |
| **Full NF4 6-step pipeline** | [D](D-qlora-three-ingredients.md#d5d8-the-six-step-nf4-pipeline-slides-3235) | written for these notes |
| NF4 vs INT4 head-to-head | [D](D-qlora-three-ingredients.md#prove-nf4-beats-int4-on-real-weights) | written for these notes |
| Error-accumulation demonstration | [D](D-qlora-three-ingredients.md) | written for these notes |
| Double quantization | [D](D-qlora-three-ingredients.md#implement-the-recursion) | written for these notes |
| Inspecting `quant_state` | [D](D-qlora-three-ingredients.md#-notebook-link) | `bitsandbytes` API |
| Activation-memory probe | [E](E-memory-engineering.md#measure-activation-memory-yourself) | written for these notes |
| √n optimality check | [E](E-memory-engineering.md#the-arithmetic-that-makes-n-optimal) | written for these notes |
| **Checkpointing benchmark** | [E](E-memory-engineering.md#measure-the-real-trade-off) | written for these notes |
| `prepare_for_training` | [E](E-memory-engineering.md#-notebook-link) | notebook §3 |
| `build_optimizer` (paged) | [E](E-memory-engineering.md#using-it) | notebook §3 |
| **Complete production recipe** | [F](F-synthesis-and-practice.md#the-complete-recipe-in-production-code) | written for these notes |
| Three-way comparison | [F](F-synthesis-and-practice.md#the-controlled-comparison-notebook-7) | notebook §7 |
| Rank sweep | [F](F-synthesis-and-practice.md#f5-reading-the-rank-sweep--the-practical-payoff) | notebook §8 |

---

## Session materials in this folder

| File | Use |
|---|---|
| `3-2b-PEFT-Reparameterization.pptx` | The 58-slide deck these notes track |
| `transcript.text` | Lecture transcript — source of the exact worked numbers |
| `Q-A.xlsx` | **41 live questions with answers** — mined into the ⚠️ Trap blocks |
| `Copy_of_LoRA_and_QLoRA_Demo.ipynb` | 8-section runnable demo (LoRA from scratch, no `peft`) |
| `Pre-read-Material-01-August.pdf` | Intrinsic dimension, matrix decomposition, no-added-latency |
| `post-read-material-1st-Aug.pdf` | Primary paper list |
| [`learning-notes/assets/`](assets/) | 30 diagrams extracted from the deck, referenced inline |

## Primary papers (from post-read)

1. [**Li et al., 2018**](https://arxiv.org/abs/1804.08838) — *Measuring the Intrinsic Dimension of Objective Landscapes* — the conceptual foundation
2. [**Hu et al., 2022**](https://arxiv.org/abs/2106.09685) — *LoRA: Low-Rank Adaptation of Large Language Models* — the method
3. [**Dettmers et al., 2023**](https://arxiv.org/abs/2305.14314) — *QLoRA: Efficient Finetuning of Quantized LLMs* — NF4 + double quant + paging
4. [**Dettmers & Zettlemoyer, 2023**](https://arxiv.org/abs/2212.09720) — *The Case for 4-bit Precision*
5. [**Chen et al., 2016**](https://arxiv.org/abs/1604.06174) — *Training Deep Nets with Sublinear Memory Cost* — gradient checkpointing
6. [**Biderman et al., 2023**](https://arxiv.org/abs/2304.01373) — *Pythia* — the model family used in the demo notebook
