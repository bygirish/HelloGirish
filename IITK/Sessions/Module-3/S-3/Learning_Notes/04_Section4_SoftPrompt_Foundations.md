# Section 4 — Soft Prompting: Foundations (Slides 33–40)

> **Goal:** Understand the *second* PEFT family — steering a frozen model through **learned virtual tokens** instead of new architecture. Master the discrete-vs-continuous distinction and the Prefix-Tuning vs Prompt-Tuning mechanics.
> **Demo:** `Prefix_Tuning_Demo_2.ipynb`

---

## 4.1 The core idea (Slide 34)

Two ways to specialize a frozen model:

| | Adapters | Soft Prompt |
|---|---|---|
| **Where** | Add elements **vertically** in the architecture | Add elements in the **input sequence** |
| **Space** | **Architectural space** | **Token space** |
| **Mechanism** | New bottleneck sub-layers | Task-specific (trainable) tokens |

**The governing principle (slide 34):**
> *"Conditioning on a proper context can steer the LM without changing its parameters."*

This is the entire thesis of prompting: a large LM is a conditional distribution `p(output | context)`. If you can find the *right context*, you change the output distribution **without touching a single weight**. Soft prompting *learns* that context.

> 💡 **Learning Thought:** Adapters change *the function*; soft prompts change *the input to the function*. That's why soft prompting can have even fewer trainable params than adapters — you're not learning a transformation, just an optimal "question" to ask the frozen model.

---

## 4.2 Discrete vs. Continuous prompts (Slide 35)

| | **Discrete (hard) prompt** | **Continuous (soft) prompt** |
|---|---|---|
| What it is | A sequence of **real token** embeddings (actual words: "Summarize:") | A sequence of **virtual, trainable** embeddings (no word maps to them) |
| Chosen by | Humans, by hand (prompt engineering) | **Gradient descent** |
| Constraint | Must correspond to vocabulary tokens | Free vectors in embedding space ℝ^d |

**Why not just hand-write prompts?** (slide 35): *manual prompts are brittle* — performance **varies greatly** with tiny wording changes. There's no gradient signal to optimize a discrete prompt directly (discrete tokens aren't differentiable). Soft prompts remove the constraint that each prompt vector be a real word, making the prompt **continuous and differentiable** — so we can *optimize* it.

> 💡 **Learning Thought:** The leap "discrete → continuous" is the same move deep learning always makes: **relax a discrete search into a continuous optimization.** Instead of searching the finite vocabulary for the best prompt words, we do gradient descent in the *infinite* continuous embedding space — a strictly larger, differentiable search space. **PT = Prefix Tuning** (slide 35, Li & Liang 2021).

---

## 4.3 Prefix Tuning vs Prompt Tuning (Slides 36–39)

Both **prepend trainable virtual tokens**, but they differ in **where** the tokens act:

### Prompt Tuning (Lester et al., 2021) — the *shallow* version
- Prepend `k` trainable embeddings **only at the input layer**.
- The rest of the network processes them like normal tokens.
- Fewest parameters (`k × d`). Works well mainly at **large model scale**.

### Prefix Tuning (Li & Liang, 2021) — the *deep* version
- Prepend trainable **prefix vectors at every transformer layer**, specifically into the **keys and values** of attention.
- More expressive (influences every layer's attention directly), more parameters than Prompt Tuning.

**The attention mechanics (slide 39):** at each layer, trainable prefix key/value matrices `P_k`, `P_v` are **concatenated** with the actual tokens' keys/values:

```
K' = [ P_k ; K ]        V' = [ P_v ; V ]
Attention(Q, K', V') = softmax( Q·K'ᵀ / √d ) · V'
```
where `;` denotes **concatenation**. The query comes from the real tokens; the prefix injects extra "virtual context" that every real token can attend to, at every layer.

> 💡 **Learning Thought:** Prefix Tuning acts on **keys/values**, not on the input embeddings alone. That means the learned prefix directly reshapes the **attention distribution** at every layer — a much stronger lever than only prepending at layer 0 (Prompt Tuning). Trade-off: more params, more expressiveness. *"Prompt Tuning = prefix at input only; Prefix Tuning = prefix in K/V at all layers"* is the crisp exam answer.

**Stability note:** Prefix Tuning often re-parameterizes the prefix through a small MLP during training (to stabilize optimization), then keeps only the resulting key/value vectors at inference.

---

## 4.4 The cost of prompt tuning (Slide 40)

Soft prompts aren't free. Key issues:

1. **Longer prefix ⇒ more parameters and more compute.** Increasing the number of soft tokens increases the prefix key/value sizes (and eats context-window budget).
2. **Large numbers of soft tokens** (e.g., 100) are sometimes used — but…
3. **Are all prompt tokens effective for a given input?** Probably not — a fixed long prompt wastes capacity on inputs that don't need it.

**Three remedies the deck previews (→ Section 5):**
- **Use a short prompt** for an optimal parameter increment.
- **Use multiple prompts** to cover different parts/subsets of the data.
- **Use a gating mechanism** to align the right prompt with the right data subset.

> 💡 **Learning Thought:** These three bullets are literally the roadmap to the advanced methods: *short + multiple + gating* = **SMoP** (Sparse Mixture of Prompts). "Not all prompt tokens are useful for all inputs" is the problem statement that all of Section 5 tries to solve, from three different angles (per-input routing, per-layer budgeting, per-instance generation).

---

## 🧪 Tie-in: `Prefix_Tuning_Demo_2.ipynb`

When running the demo, observe:
1. **Trainable prefix tensors** are created and prepended; **base model frozen**.
2. Where the prefix enters — input embeddings (prompt tuning style) vs. per-layer key/value (prefix tuning style).
3. **Param count**: prefix params vs. total → typically well under 1%.
4. Effect of **prefix length** (`k`) on both performance and memory — connect to slide 40.

---

## 🎯 Interview Questions — Section 4

**Q1. What does soft prompting change, and what does it leave frozen?**
*A:* It learns a set of **virtual token embeddings** prepended to the input/attention (token space), while leaving **all model weights frozen**. It steers the model via *context*, not via weight changes.

**Q2. Discrete vs continuous prompts — define and give the key advantage of continuous.**
*A:* Discrete = real vocabulary-token embeddings chosen by hand (brittle, non-differentiable). Continuous = free trainable vectors in embedding space, **optimized by gradient descent**. Advantage: differentiable optimization over an unrestricted (not vocabulary-limited) space ⇒ better, more robust prompts.

**Q3. Why are manual (hard) prompts problematic?**
*A:* Performance varies greatly with small wording changes (brittle), and you can't directly optimize them because token choice is discrete/non-differentiable — you're stuck with heuristic search over the vocabulary.

**Q4. Prefix Tuning vs Prompt Tuning — what's the difference?**
*A:* Prompt Tuning prepends trainable embeddings **only at the input layer** (fewest params, needs scale to shine). Prefix Tuning inserts trainable prefixes into the **keys and values at every layer** (more expressive, more params), directly shaping attention throughout the network.

**Q5. In Prefix Tuning, how does the prefix enter the attention computation?**
*A:* Trainable prefix key/value matrices are **concatenated** with the real tokens' K and V: `K'=[P_k;K]`, `V'=[P_v;V]`, then standard attention runs over the extended K'/V'. Real-token queries attend over both real and virtual context at every layer.

**Q6. Name three costs/limits of soft prompting and the fix each motivates.**
*A:* (1) Long prefixes cost params/compute/context → use **short** prompts; (2) a single fixed prompt underfits diverse data → use **multiple** prompts; (3) not all tokens help every input → use a **gating/router**. Together these motivate **SMoP**.

**Q7 (deep). Why does Prompt Tuning need large models to work well, while Prefix Tuning works at smaller scale?**
*A:* Prompt Tuning only perturbs the *input* layer, so its influence must propagate through the frozen stack — small models don't have enough capacity/robustness to be steered by an input-only nudge. Prefix Tuning injects trainable context into **every layer's** attention, giving direct control at all depths, so it's effective even without huge scale.

**Q8 (deep). Compare soft prompting to adapters in parameter count and expressiveness.**
*A:* Soft prompts typically use the fewest params (just `k×d` per task, or per-layer K/V) and don't alter the function — great for cheap task-switching, but limited when the task needs a genuinely new transformation. Adapters use more params (bottleneck per layer) but can learn new **non-linear** transformations, so they're more expressive for larger distribution shifts. Soft prompt = cheapest lever; adapter = stronger lever.

---

## ✅ Section 4 takeaways

- Soft prompting adapts in **token space**: learn a context that steers a **frozen** model.
- **Discrete → continuous** = relaxing brittle hand-prompts into differentiable, optimizable vectors.
- **Prompt Tuning** = trainable embeddings at input only; **Prefix Tuning** = trainable K/V prefixes at **every layer** (`K'=[P_k;K]`, `V'=[P_v;V]`).
- Costs: prefix length ↔ params/compute/context; **not all tokens help every input** → motivates SMoP/APT/IDPG.

➡️ **Next:** [Section 5 — Advanced Soft Prompting](05_Section5_Advanced_SoftPrompt.md): SMoP, APT, IDPG, SPT.
