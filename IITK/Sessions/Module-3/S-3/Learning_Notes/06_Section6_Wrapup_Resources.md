# Section 6 — Wrap-up, Decision Guide & Resources (Slides 54–55)

> **Goal:** Consolidate the whole session into a single mental model, a practical decision guide, a rapid-revision sheet, and the reading list.

---

## 6.1 "In a Nutshell" — the session in five lines (Slide 54)

1. **Prefix-tuning** adds **virtual (learnable) tokens** to enrich the input context — without touching weights.
2. **Not all prompt tokens suit a given input** → choose **shorter** prompts automatically → **SMoP**.
3. **Budget a greater prefix length to bottom layers, shorter to top layers** → **APT**.
4. **A single soft prompt shouldn't fit all inputs** → make prompts **input-dependent** → **IDPG**.
5. (Underlying all of it) **Freeze the base, train a tiny task-specific module** — the PEFT thesis.

---

## 6.2 The whole session as one story

```
Full fine-tuning is too costly & forgets  (Sec 1)
        │
        ▼
PEFT: freeze base, train <1%   ── taxonomy: Additive · Selective · Re-param · Soft-Prompt  (Sec 2)
        │
   ┌────┴─────────────────────────────┐
   ▼                                   ▼
ADAPTERS (architectural space, Sec 3)   SOFT PROMPTS (token space, Sec 4)
 • bottleneck d→m→d, near-identity init   • discrete → continuous (differentiable)
 • Sequential (Houlsby): latency          • Prompt Tuning (input only)
 • Residual/Parallel (Lin): fix latency   • Prefix Tuning (K/V, every layer)
 • AdapterFusion: compose tasks           • cost: not all tokens help every input
                                            │
                                            ▼
                              ADVANCED SOFT PROMPTS (Sec 5)
                              • SMoP  — route to short prompt (per input)
                              • APT   — budget prefix length (per layer)
                              • IDPG  — generate prompt (per instance)
                              • SPT   — select layers to prompt (placement)
```

**One sentence:** *Every method is a different answer to "given a tiny parameter budget on a frozen model, where and how much should I adapt — architecturally or in the token stream — and should that adaptation vary by task, layer, or input?"*

---

## 6.3 Practical decision guide

| Situation | Recommended method | Why |
|---|---|---|
| Need **zero** extra inference latency, single task | **LoRA** (merged) | delta folds into weights |
| Many tasks, want **hot-swappable modules** | **Adapters / AdapterFusion** | modular, composable, no forgetting |
| Very large model, **tight GPU** | **QLoRA** or **Prompt Tuning** | 4-bit base / minimal params |
| Cheapest possible per-task footprint | **Soft prompting** | just `k×d` params/task |
| Heterogeneous inputs within a task | **SMoP / IDPG** | per-input specialization |
| Task relies on specific layers (interpretability) | **APT / SPT** | per-layer budgeting/selection |
| Big distribution shift, need real capacity | **Adapters** or **full FT** | non-linear transform capacity |

> 💡 **Learning Thought:** In practice, **LoRA/QLoRA dominate industry** today for their zero-latency merge and strong performance; **adapters** win when modularity/composition matters; **soft prompting** shines for ultra-cheap, many-task, or research/interpretability settings. Know *why* each is chosen, not just that it exists.

---

## 6.4 Rapid revision — one-liners for every topic

- **PEFT** = freeze base, train <1%; fixes **cost** + **catastrophic forgetting**.
- **Training memory** ≈ 12–16× weights (weight+grad+Adam states+activations); **PEFT** removes that for 99% of params.
- **Four families:** Additive, Selective, Re-param, Soft-Prompt.
- **Adapter** = `x + f(x·W_down)·W_up`, bottleneck `m≪d`, **near-identity init** (`W_up≈0`).
- **Sequential adapter** limits: latency, depth-scaling, no parallelism, costly multi-task serving.
- **Residual/Parallel adapter** = parallel branch fixes latency.
- **AdapterFusion** = compose frozen task-adapters without forgetting.
- **Soft prompt** = steer frozen LM via learned **virtual tokens** (token space).
- **Discrete→continuous** = relax brittle hand-prompts into differentiable vectors.
- **Prompt Tuning** = input-only; **Prefix Tuning** = K/V at every layer (`K'=[P_k;K]`).
- **SMoP** = route to 1-of-k short prompts (MoE for prompts).
- **APT** = token-gate (length) + layer-gate (prefix-vs-input); more prefix to bottom layers.
- **IDPG** = generate an instance-specific prompt via a small generator.
- **SPT/LPT** = learn which/where layers to inject prompts under a budget.

---

## 6.5 Reading list (Slide 55)

### General PEFT surveys
- Wang et al., *Parameter-efficient fine-tuning in large language models: a survey of methodologies*, 2025.
- Xu et al., *Parameter-Efficient Fine-Tuning Methods for Pretrained Language Models: A Critical Review and Assessment*, 2023.

### Additive PEFT (Adapters)
- **Houlsby et al., 2019** — *Parameter-Efficient Transfer Learning for NLP* (Sequential Adapters).
- **Lin et al., 2020** — *Exploring Versatile Generative Language Model via Parameter-Efficient Transfer Learning* (Residual/Parallel Adapters).

### Soft Prompting
- **Li & Liang, 2021** — *Prefix-Tuning: Optimizing Continuous Prompts for Generation*.
- **Choi et al., 2023** — *SMoP: Towards Efficient and Effective Prompt Tuning with Sparse Mixture-of-Prompts*.
- **Zhang et al., 2023** — *Towards Adaptive Prefix Tuning for Parameter-Efficient Language Model Fine-tuning* (APT).
- **Wu et al., 2022** — *IDPG: An Instance-Dependent Prompt Generation Method*.
- **Zhu & Tan, 2023** — *SPT: Learning to Selectively Insert Prompts for Better Prompt Tuning*.

### Background referenced in deck
- Tie et al., 2025 — *A Survey on Post-training of Large Language Models*.
- (Sibling session) Lester et al., 2021 — *The Power of Scale for Parameter-Efficient Prompt Tuning* (Prompt Tuning); Hu et al., 2021 — *LoRA*; Dettmers et al., 2023 — *QLoRA*.

---

## 6.6 Hands-on next steps

1. Run **`Residual_Adapter_Demo_1.ipynb`** → confirm near-identity init + trainable-param %.
2. Run **`Prefix_Tuning_Demo_2.ipynb`** → vary prefix length, watch params vs. accuracy.
3. **Stretch:** implement a toy **SMoP router** (average-pool input → softmax over 4 short prompts → top-1) on top of the prefix-tuning demo.
4. Cross-read the **pre-read** and **post-read** PDFs and the **Q&A** sheet in this folder to catch nuances the slides compress.

---

## 6.7 Mock "exit interview" — can you answer all 8?

1. Why does full FT need ~12–16× the weight memory, and how does PEFT dodge it?
2. Give the four PEFT families with one distinguishing idea each.
3. Write the adapter equation and explain near-identity init.
4. Why do adapters add inference latency but LoRA doesn't?
5. Prompt Tuning vs Prefix Tuning — where does each inject, and how does Prefix enter attention?
6. Derive SMoP from the three costs of soft prompting.
7. Why does APT give longer prefixes to bottom layers?
8. How does IDPG make prompts instance-dependent, and why is that principled?

*(If you can answer these cold, you've mastered the session. Answers live across Sections 1–5.)*

---

## ✅ You've completed Module 3 · Session 3

**The through-line to remember:** PEFT freezes a powerful frozen model and spends a tiny, carefully-placed parameter budget to specialize it — either in **architectural space** (adapters) or **token space** (soft prompts) — and the frontier is about making that spend **adaptive** to task, layer, and input.

⬅️ Back to [index](00_README_Index.md).
