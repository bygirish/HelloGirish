# Section 5 — Advanced Soft Prompting (Slides 41–53)

> **Goal:** Understand the three refinements that fix the three weaknesses of vanilla soft prompting. Each answers a different "not all prompt X is useful" problem:
> - **SMoP** — not all prompt *tokens* help every **input** → *route to a short prompt*.
> - **APT** — not all *layers* need the same prompt *length* → *budget prefix per layer*.
> - **IDPG** — one fixed prompt can't fit all **instances** → *generate the prompt per input*.
> - **SPT/LPT** — not every *layer* needs prompting at all → *select where to inject*.

---

## 5.1 The problem recap (from Section 4)

A vanilla soft prompt is:
- **fixed** for the whole task,
- the **same length at every layer**,
- **identical for every input instance**, and
- injected at the **input layer only**.

Slide 40 asked: *are all prompt tokens effective for a given input?* Section 5 is four "no"s, each with a fix.

---

## 5.2 SMoP — Sparse Mixture of Prompts (Choi et al., 2023) (Slides 41–42)

**Problem it solves:** long soft prompts are costly, and a single prompt can't fit heterogeneous data. **Idea:** keep **several short** prompts and **route** each input to the most relevant one — a *sparse* Mixture-of-Experts, but the "experts" are prompts.

**Mechanics (slide 42):**
- Maintain `k` short **soft-prompt units**, each a small prompt embedding.
- A lightweight **router model** looks at the input and produces routing probabilities.
  - Let `x̄` = **average of the input** embeddings (a cheap summary of the input).
  - **Routing probability:** `p = softmax(x̄ · W_router)` over the `k` prompts.
- **Sparse selection:** route to the **top-1** (or top-few) prompt(s) — only the selected short prompt is prepended. So each input pays for a *short* prompt, not a long one.
- **Training objective:** standard task loss over the routed prompt (often with a load-balancing term so the router uses all prompts, à la MoE).

```
input ──► average x̄ ──► Router ──► top-1 prompt ─┐
                                                  ├─► prepend ► frozen LM ► output
                     (k short prompts pool)  ─────┘
```

> 💡 **Learning Thought:** SMoP directly implements slide 40's three fixes at once: **short** prompts + **multiple** prompts + **gating**. It's the exact same "route to the right small module" pattern you saw in **Adapter Routing** (Section 3.3). *Sparse* routing means only one small prompt is active per input ⇒ efficiency *and* specialization together.

---

## 5.3 APT — Adaptive Prefix Tuning (Zhang et al., 2023) (Slides 44–48)

**Problem it solves:** vanilla prefix tuning uses the **same prefix length (same # trainable params) at every layer** — but layers aren't equal.

**The key observation (slide 44):**
- **Bottom layers** capture **concrete, shallow, phrase-level** features → they carry **more** information that could use a **greater** prefix length.
- **Top layers** deal with **abstract semantic** information → **shorter** prefix suffices.

So a uniform prefix length is wasteful. APT makes the **effective prefix length dynamic across layers** via **two gates**:

### (a) Fine-grained weighting — *token-level gate* (slide 46)
A **token-level gate** decides **how many** of the pseudo/virtual prefix tokens are actually needed at a given layer. The gated weights at layer *i* scale each prefix token's contribution — effectively pruning/downweighting unneeded pseudo-tokens per layer. This makes the "number of trainable pseudo-tokens" adaptive rather than fixed.

### (b) Coarse-grained weighting — *layer-level gate* (slide 47)
A **coarse (layer-level) gate** balances information from the **task-specific prefix tokens** vs. the **original input tokens** at each layer. The prefix key–value pair derived from the pseudo prefix tokens at layer *i* is **rescaled** by a learned layer-level weight before being merged with the real tokens' K/V.

**Empirical payoff (slide 48):** the learned prefix-weight distribution matches task nature:
- **CoNLL04** (Entity–Relation Extraction) → weight concentrates on **bottom layers** (phrase-level features).
- **COPA** (Commonsense causal reasoning) → weight concentrates on **higher layers** (semantic reasoning).

> 💡 **Learning Thought:** APT is the soft-prompt echo of the adapter "food for thought" (Section 3.7): *different layers deserve different amounts of adaptation.* The two-gate design (token-level = "how many tokens", layer-level = "prefix vs input balance") is a clean example of **learned, data-driven allocation of a parameter budget**. The interpretability bonus (slide 48) — being able to *see* which layers a task relies on — is a strong talking point.

---

## 5.4 IDPG — Instance-Dependent Prompt Generation (Wu et al., 2022) (Slides 49–52)

**Problem it solves (slide 49):** soft prompts are **instance-independent** — the *same* prompt is prepended to **every** input of a task. This is odd:
- It **contradicts the LM objective**: during pre-training you'd rarely see many different sentences sharing an identical prefix.
- One fixed prefix can't be optimal for every diverse instance.

**Question:** *can we generate an **input-dependent** soft prompt?* IDPG says yes.

### General prompt-tuning framework (slide 50)
For a task `T` with data, the input is formatted (with soft prompts `P`) as:
- **sentence-pair:** `[P ; sentence_A ; sentence_B]`
- **single-sentence:** `[P ; sentence]`

Task-specific prompt `P ∈ ℝ^{m×d}`, where `m` = number of soft-prompt tokens, `d` = hidden dim. *(Note the deck's point: in earlier methods `m` and the fact that `P` is fixed were left implicit — IDPG makes `P` a function of the input.)*

### IDPG mechanism (slides 51–52)
```
input ─► Encode with PTM (pre-trained model) ─► hidden summary
                                                      │
                                                      ▼
                                         Neural Prompt Generator  G(·)
                                                      │
                                                      ▼
                                      instance-specific soft prompt P(x)
                                                      │
                                     prepend to input ► frozen LM ► output
```
- **Encode input with a PTM** to get a representation of the specific instance.
- A **Neural Prompt Generator** (a small module, e.g., a bottleneck two-layer MLP) maps that representation to a **time-stamped / instance-specific soft prompt**.
- **Parameter count (slide 52):** the generator `G` is deliberately small (a down-up bottleneck like an adapter) so the added params stay tiny while now the prompt **varies per input**.

> 💡 **Learning Thought:** IDPG is the conceptual inverse of the fixed-prompt assumption. Instead of *learning one prompt vector*, it *learns a function that produces a prompt from the input*. Notice this reintroduces an adapter-like generator (bottleneck MLP) — **the token-space and architecture-space families quietly merge here.** IDPG = "conditional soft prompting."

---

## 5.5 SPT / LPT — Where should prompts be injected? (Slide 53)

**Problem it solves:** classic prompt tuning adds soft prompts **only at the input layer**. Is that optimal?

- **LPT (Late Prompt Tuning):** showed prompts can be inserted at **intermediate layers**, not just the input — sometimes better and cheaper.
- **SPT (Selective Prompt Tuning; Zhu & Tan, 2023):** *learn* the **optimal set of layers** at which to inject prompts, given the task.
- **Budget constraint:** what if you can only afford prompts at **max K layers**? SPT frames prompt injection as a **selection problem** under a layer budget and learns where the prompts matter most.

> 💡 **Learning Thought:** SPT/LPT close the loop with APT and the adapters section: the recurring meta-question of this whole lecture is **"given a fixed small budget, *where* and *how much* should I adapt?"** APT answers *how much per layer*; SPT answers *at which layers at all*; SMoP answers *which prompt per input*; IDPG answers *what prompt per input*. Four axes of the same allocation problem.

---

## 5.6 Unifying view — the four refinements

| Method | "Not all ___ is useful" | Fix | Adaptivity axis |
|---|---|---|---|
| **SMoP** | prompt fits every **input** | route to 1 of k **short** prompts | per-input **selection** of prompt |
| **APT** | layers need equal prefix **length** | token+layer gates budget prefix | per-**layer** amount |
| **IDPG** | one prompt fits every **instance** | generate prompt from input | per-**instance** content |
| **SPT/LPT** | every **layer** needs a prompt | learn which layers to inject | per-**layer** placement |

> 💡 **Learning Thought (capstone):** Vanilla soft prompt = *fixed content, fixed length, fixed location, fixed per input.* Each advanced method **relaxes exactly one of those "fixed"s into "learned."** If an interviewer asks "how would you improve prompt tuning?", you can *derive* SMoP/APT/IDPG/SPT from first principles by asking "which fixed assumption can I make adaptive?"

---

## 🎯 Interview Questions — Section 5

**Q1. What problem does SMoP solve and how?**
*A:* Long single prompts are costly and can't fit heterogeneous data. SMoP keeps **k short** prompts and uses a **router** (softmax over an average-pooled input summary `x̄`) to **sparsely** select the top prompt per input — combining short + multiple + gating for efficiency and specialization.

**Q2. Why does APT vary prefix length across layers, and in which direction?**
*A:* Because bottom layers encode concrete **phrase-level** features (need **more** prefix) while top layers encode **abstract semantics** (need **less**). APT budgets **greater** prefix length to bottom layers, **shorter** to top layers.

**Q3. Describe APT's two gates.**
*A:* (1) **Fine-grained token-level gate** — decides how many pseudo prefix tokens are effectively used at each layer (adaptive prefix length). (2) **Coarse-grained layer-level gate** — rescales the prefix-derived K/V to balance task-prefix information vs. original-input information at each layer.

**Q4. What does APT's learned weight distribution reveal (slide 48)?**
*A:* Task-dependent layer reliance: entity–relation extraction (CoNLL04) leans on **bottom** (phrase-level) layers; commonsense causal reasoning (COPA) leans on **top** (semantic) layers. It's interpretable evidence that different tasks need adaptation at different depths.

**Q5. Why are vanilla soft prompts "instance-independent," and why is that a weakness?**
*A:* The same learned prefix is prepended to every input of a task. It's a weakness because it contradicts the LM's pre-training (rarely do many different sentences share one prefix) and one fixed prompt can't be optimal across diverse instances.

**Q6. How does IDPG make prompts input-dependent?**
*A:* It **encodes the input with a PTM**, then passes that representation through a small **neural prompt generator** (bottleneck MLP) that outputs an **instance-specific** soft prompt, which is prepended before the frozen LM. The generator is tiny, so params stay low while the prompt now adapts per input.

**Q7. What question do SPT and LPT address?**
*A:* **Where** to inject soft prompts. LPT shows intermediate layers work (not just input); SPT *learns* the optimal layers to prompt, often under a **budget of at most K layers**.

**Q8 (deep). Unify SMoP, APT, IDPG, SPT as relaxations of vanilla prompt tuning.**
*A:* Vanilla soft prompt fixes content, length, placement, and is input-agnostic. SMoP relaxes *which prompt per input* (selection); IDPG relaxes *prompt content per instance* (generation); APT relaxes *prompt length per layer* (budgeting); SPT relaxes *placement across layers* (selection). Each makes one fixed assumption learnable — a general recipe for improving any PEFT method.

**Q9 (deep). SMoP's router vs Adapter Routing vs MoE — what's common?**
*A:* All three learn to **route each input to a small specialized module** (a prompt, an adapter, or an FFN expert) and activate only a sparse subset, trading a cheap routing computation for input-specific specialization and efficiency. SMoP is MoE applied to soft prompts.

---

## ✅ Section 5 takeaways

- The four advanced methods each **relax one "fixed" assumption** of vanilla soft prompting into a **learned** one.
- **SMoP:** route input → 1 of k **short** prompts (MoE for prompts).
- **APT:** two gates budget prefix **length per layer** (more at bottom, less at top); interpretable.
- **IDPG:** a small generator produces an **instance-specific** prompt from the encoded input.
- **SPT/LPT:** learn **which layers** to prompt under a budget.
- Meta-lesson: PEFT progress = *deciding where/how much/what to adapt, adaptively, under a budget.*

➡️ **Next:** [Section 6 — Wrap-up & Resources](06_Section6_Wrapup_Resources.md).
