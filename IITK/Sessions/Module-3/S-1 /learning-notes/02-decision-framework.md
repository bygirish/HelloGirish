# 02 — The Decision Framework
### RAG vs. Prompt Engineering vs. Fine-tuning

> **Why this matters:** Fine-tuning is expensive and permanent. Before you touch a single
> weight, you should be able to justify *why not just prompt it* or *why not just retrieve the
> facts*. This is the single most practical, most interview-tested part of the lecture: given a
> business problem, pick the right lever. Getting this wrong wastes GPU budget on a problem a
> better prompt would have solved.

---

## 1. Three Pathways to Get Things Done with an LLM

There are exactly three levers to change an LLM's output. They differ in **what** you change.

| Pathway | What you change | The loop |
|---------|-----------------|----------|
| **Prompt Engineering** | The **instructions** (not the model, not the data) | Define what you want → write a clear prompt with context/instructions/examples → check output → refine & iterate |
| **RAG** (Retrieval-Augmented Generation) | The **facts available at inference** | User query → retrieve top-*k* relevant chunks from a Vector DB → feed them to the LLM → grounded response |
| **Fine-tuning** | The **model weights** themselves | Take a pre-trained LLM → train on domain-specific data → updated weights → fine-tuned LLM |

> 💡 **Learning Thought:** Order them by *what they touch*: **Prompt = instructions, RAG =
> knowledge, Fine-tuning = weights.** Cost and permanence increase left → right. Always try the
> cheaper lever first.

---

## 2. The Blueprint — What Each Is *Best For*

**Prompt Engineering**
- **Best for:** quick prototyping, standard tasks, **zero setup**.
- **Key idea:** you change the *instructions*, not the model or the data.

**RAG**
- **Best for:** dynamic, massive, or **proprietary** knowledge bases.
- **Key idea:** connects the model to a **database to fetch fresh facts** *before* responding.

**Fine-tuning**
- **Best for:** deep customization, **strict output formatting**, teaching domain-specific
  terminology on a *foundational* level.
- **Key idea:** bake behavior/vocabulary into the weights.

> 💡 **Learning Thought:** RAG changes *what the model knows right now*; fine-tuning changes
> *how the model behaves in general*. A model that must cite **today's** filing needs RAG. A
> model that must always answer in a **fixed JSON style** needs fine-tuning. They solve
> different problems and are frequently **combined**.

---

## 3. The Decision Matrix ⭐ (memorize this table)

This is the slide most likely to become an interview question. Five dimensions:

| Dimension | Prompt Engineering | RAG | Fine-Tuning |
|-----------|--------------------|-----|-------------|
| **Primary goal** | Directing behavior & task framing | Accessing external/updated facts | Adapting style, format & core behavior |
| **External knowledge** | Very Low (limited by context window) | **Extremely High** | Low → Medium |
| **Style/Format control** | Medium | Low | **Extremely High** |
| **Cost to set up** | Minimal | Medium | **High** (needs clean datasets & GPU time) |
| **Latency impact** | Low | Medium (adds a retrieval step) | **Very Low** (shorter prompts = faster inference) |

**How to read the extremes:**
- Need **fresh/proprietary facts**? → RAG wins on *external knowledge*.
- Need **strict style/format** or deep behavior change? → Fine-tuning wins on *style control*.
- Need **cheapest, fastest to ship**? → Prompt engineering wins on *setup cost*.
- Note the subtle one: **fine-tuning has the *lowest* latency at inference** because you no
  longer need a long few-shot prompt — the behavior is in the weights. (Its cost is paid
  *upfront* in training, not per-call.)

> 💡 **Learning Thought:** People wrongly assume fine-tuning is "slow." At **training** time,
> yes. At **inference** time it's the *fastest* of the three, because it removes prompt bloat
> and the retrieval hop. Separate **build cost** from **run cost** in your head.

---

## 4. When to Choose What — A Practical Decision Tree

```
Is the needed knowledge changing frequently, huge, or proprietary?
        │
        ├── YES ──►  Use RAG  (ground on a vector DB; update data, not weights)
        │
        └── NO
             │
   Can a well-crafted prompt (with a few examples) already get it right?
        │
        ├── YES ──►  Use Prompt Engineering  (cheapest, zero setup, ship it)
        │
        └── NO
             │
   Do you need a consistent style/format/behavior, or deep domain adaptation,
   AND do you have (or can build) a clean labeled dataset + GPU budget?
        │
        └── YES ──►  Fine-tune
```

**Rules of thumb (say these in an interview):**
- **Facts that change** → RAG. **Behavior that's fixed** → fine-tune.
- Reach for the **cheapest lever that works**: Prompt → RAG → Fine-tune.
- They are **not mutually exclusive.** The strongest production systems often **fine-tune for
  style/format *and* use RAG for facts *and* still prompt-engineer** on top.
- Fine-tune when prompting "can't guarantee" the behavior — recall from file 01 that
  *consistency* is a core reason to fine-tune.

> 💡 **Learning Thought:** A clean litmus test — *"If the right answer depends on information
> that could change tomorrow, no amount of fine-tuning fixes it; you need retrieval."* Weights
> are a snapshot; a database is live.

---

## 🎯 Interview Questions

**Q1. A bank wants a chatbot that answers questions about a customer's *latest* transactions.
Prompt, RAG, or fine-tune?**
**RAG.** The data is proprietary and changes constantly; it must be fetched at query time.
Fine-tuning would bake in stale data and leak private info into weights.

**Q2. You need every response in a strict JSON schema with a fixed brand tone. Which lever?**
**Fine-tuning** — style/format control is its strongest dimension, and it *guarantees* the
behavior in a way prompting can't. (You might still RAG for the factual content inside the JSON.)

**Q3. Fine-tuning is often called expensive, yet the matrix says its latency impact is "Very
Low." Reconcile that.**
Two different costs. **Setup/training cost is high** (clean data + GPU time, paid once).
**Inference latency is low** because the behavior lives in the weights, so prompts are short and
there's no retrieval hop. High build cost, low run cost.

**Q4. Why isn't "just use a bigger context window / more few-shot examples" always the answer
instead of fine-tuning?**
Long prompts raise **per-call cost and latency**, the model can still be inconsistent, and
external knowledge in the prompt is bounded by the context window. Fine-tuning moves the
behavior into weights (cheaper per call) and RAG scales knowledge beyond any context window.

**Q5. Can RAG and fine-tuning be used together? Give an example.**
Yes, and it's common. Fine-tune a model to always answer in your support-doc style and JSON
format, then use RAG to inject the *current* product manual so the facts are fresh. Style from
weights, facts from retrieval.

**Q6. Give the one-sentence heuristic for RAG vs. fine-tuning.**
If the answer depends on **knowledge that changes or is too large to bake in → RAG**. If you
need to change **how the model behaves, formats, or specializes → fine-tune**. Facts vs. behavior.

**Q7. Which lever has the lowest setup cost and why would you still not use it?**
**Prompt engineering** (minimal setup). You'd move past it when prompts can't *guarantee*
consistency, when you need large external/proprietary knowledge (→ RAG), or when long prompts
make inference too costly (→ fine-tune).

---

**One-line takeaway:** *Prompt changes instructions, RAG changes available facts, fine-tuning
changes weights — reach for the cheapest lever that works, use RAG for knowledge that moves and
fine-tuning for behavior that must be fixed, and remember they compose.*
