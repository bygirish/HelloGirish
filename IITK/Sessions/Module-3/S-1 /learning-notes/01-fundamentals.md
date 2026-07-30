# 01 — Fundamentals
### Language Model, LLM, and where fine-tuning lives

> **Why this matters:** You cannot reason about *fine-tuning* until you are crisp on three
> things — what a model fundamentally *is* (a next-token probability machine), what *kind*
> of model you're holding (encoder / decoder / encoder–decoder), and *which training stage*
> fine-tuning occupies. This file builds that vocabulary. Everything later is a variation on it.

---

## 1. What is a Language Model?

A **language model (LM)** is a **probabilistic model of text**: given a sequence of words, it
estimates the probability of the *next* word.

- **Core task — predict the next token:**

  ```
  P(wₙ | w₁, w₂, …, wₙ₋₁)
  ```

- **Learned from data:** trained on large text corpora, absorbing grammar, facts, and style.
- **Why it matters:** next-word prediction is the single engine behind autocomplete,
  translation, and chatbots.

**The canonical example (slide 5):**

> "The cat sat on the ___"
> `mat` → 62%  ·  `sofa` → 21%  ·  `roof` → 9%

The model assigns a probability to *every* possible next token, then we sample or take the top one.

> 💡 **Learning Thought:** An LLM never "looks up" an answer. It computes *the most probable
> continuation of the text so far*. Every strength (fluency, reasoning) **and** every failure
> mode (hallucination, confident-but-wrong) is a direct consequence of this one mechanism.

---

## 2. What is a *Large* Language Model?

An **LLM** is a language model **scaled up massively** — in parameters, training data, and
compute — almost always built on the **Transformer** architecture.

| Axis of scale | Magnitude | Detail |
|---------------|-----------|--------|
| **Parameters** | Billions+ | GPT-3 = 175B; frontier models larger still |
| **Training tokens** | Trillions | web pages, books, code, articles |
| **Architecture** | Transformer | self-attention layers stacked dozens–hundreds deep |

The word "large" is doing real work: it's not a different algorithm, it's the *same*
next-token objective at a scale where new behaviors appear.

---

## 3. Why Scale Changes Everything

Capabilities that **emerge** with scale (they are weak-to-absent in small models):

- **Fluent generation** — coherent essays, summaries, dialogue.
- **In-context learning** — learn a new task from a few examples *in the prompt*, with **no
  retraining**. (This is exactly why prompt engineering works — see file 02.)
- **Reasoning & knowledge** — multi-step problem solving, world knowledge, code, math.

**Well-known LLM families to name-drop correctly:**

| Family | Maker | Note |
|--------|-------|------|
| **GPT** | OpenAI | GPT-3 → GPT-4 and beyond |
| **Claude** | Anthropic | assistant models focused on safety |
| **Gemini** | Google | multimodal frontier models |
| **Llama** | Meta | open-weight, popular for research & industry (and for *fine-tuning*) |

> 💡 **Learning Thought:** "Emergence" is why fine-tuning a large base model is so powerful —
> the general capability is *already there*; fine-tuning just *points it* at your task instead
> of teaching it from scratch.

---

## 4. Three Types of LLMs ⭐

This is one of the most common interview topics in the whole lecture. The distinguishing
feature is the **attention mask** (what each token is allowed to look at).

### Encoder-Only (BERT family)
- **Attention:** *fully bidirectional* — every token attends to its full left **and** right
  context. **No causal mask.**
- **Pre-training objective:** **Masked Language Modeling (MLM)** — hide ~15% of tokens and
  predict them from surrounding context. (BERT also used next-sentence prediction.)
- **Strengths:** deep *understanding* — classification, named-entity recognition (NER),
  sentence similarity, embeddings for search / retrieval / reranking. **Not built to generate
  free text.**
- **Representative models:** BERT, RoBERTa, ALBERT, ELECTRA, DeBERTa.

### Decoder-Only (GPT family) — *the dominant modern architecture*
- **Attention:** *causal (masked)* — a triangular mask blocks every position from seeing the
  future, so token *t* attends only to tokens 1…*t*.
- **Pre-training objective:** **next-token prediction** over raw text. Generation is
  **autoregressive**: sample a token, append it, predict again. (KV-caching makes this fast.)
- **Strengths:** open-ended *generation* — chat, code, reasoning, summarization, agents.
  Scales exceptionally well.
- **Representative models:** GPT series, Claude, Llama, Mistral, Gemini — nearly all frontier
  chat models.

### Encoder–Decoder (T5, original Transformer)
- **Attention:** encoder is bidirectional over the input; decoder is causal on the output so
  far **plus cross-attention** that queries the encoder's memory (its keys & values).
- **Pre-training objective:** denoising / span corruption (T5, BART) or paired data (e.g.
  translation).
- **Strengths:** explicit **input → output transformations** — machine translation,
  summarization, question answering, structured rewriting.
- **Representative models:** original Transformer (Vaswani et al., 2017), T5 / Flan-T5, BART,
  mT5, Marian NMT.

| | Encoder-only | Decoder-only | Encoder–Decoder |
|---|---|---|---|
| **Mask** | none (bidirectional) | causal | bidirectional + causal + cross-attn |
| **Objective** | Masked LM | Next-token | Denoising / seq2seq |
| **Superpower** | Understanding | Generation | Transformation |
| **Example** | BERT | GPT / Llama | T5 |

> 💡 **Learning Thought:** *The attention mask defines the model type.* Full visibility →
> understanding. Masked future → generation. Both + a bridge (cross-attention) → seq2seq.
> **When people say "fine-tune an LLM" today they almost always mean a decoder-only model.**

---

## 5. LLM Training Stages

Every LLM walks through these stages. Knowing *which stage fine-tuning is* is the point of
this whole file.

1. **Pre-training**
   - **Start state:** *zero* world knowledge; doesn't understand word meaning.
   - **Task:** next-token prediction (decoder) or masked-token prediction (encoder, e.g. BERT
     predicting the `[MASK]` in "what `[MASK]` you doing").
   - **Data:** a giant corpus of web text — **unlabeled**.
   - **Method:** **self-supervised learning**.
   - **Outcome:** learns language + knowledge. *Most expensive stage.*

2. **Fine-tuning** ← **this entire lecture lives here**
   - Take the **pre-trained, general-purpose** model and **train some of its weights**.
   - **A general-purpose base model → a specialized model for a particular use case.**

3. **Safety / Alignment**
   - **Goal:** make outputs safe and matched to human preference.
   - **How:** further fine-tuning using a **feedback mechanism on model outputs**.
   - **Outcome:** safer model, less biased content, aligned with human judgment.
   - (Mechanics — RLHF, DPO — are covered in later lectures 3/4/5.)

> 💡 **Learning Thought:** "Self-supervised" sounds fancy but just means *the label is free*:
> the next token in the text **is** the target. No human annotation needed — which is exactly
> why pre-training can consume trillions of tokens.

---

## 6. Why is Fine-Tuning Necessary?

Four justifications (slide 16) — memorize these four; they're the "why fine-tune" interview answer:

| Reason | What it buys you |
|--------|------------------|
| **Domain expertise** | Base models are *generalists* — they miss the jargon/conventions of medicine, law, finance. |
| **Consistent behavior** | Enforce a specific tone, output format, or company policy that **prompting alone can't guarantee**. |
| **Higher accuracy** | A small fine-tuned model often **beats a much larger general model** on a *narrow* task. |
| **Lower cost & latency** | Shorter prompts + smaller specialized models → cheaper, faster inference. |

> 💡 **Learning Thought:** The last two reasons are the ones people forget. Fine-tuning isn't
> only about *quality* — a small specialized model that needs no giant few-shot prompt is
> **cheaper and faster** than calling a frontier model every time.

---

## 7. Proof It Works — Domain Specialists

- **Med-PaLM (Google):** base model PaLM, fine-tuned on medical Q&A + clinician-written
  answers. **Med-PaLM 2 reached ~86% (expert-level)** on USMLE-style medical-licensing questions.
- **BloombergGPT (Finance):** trained on decades of financial documents, filings, and news;
  specialized for sentiment analysis, entity recognition, financial Q&A. **Domain adaptation
  beat general models of similar size** on finance tasks.

**Concrete before/after (slides 18–19):**
- *"How to fine-tune a model?"* — base model gives a vague, generic response; the fine-tuned
  model gives an actual step list (choose base model → prepare dataset → modify top layers →
  train → evaluate).
- A dermatology-tuned model turns *"skin irritation, redness, itching"* into a specific
  *"mix of non-inflammatory + inflammatory acne"* diagnosis, instead of the base model's vague
  "probably acne."

> 💡 **Learning Thought:** The pattern in every example is the same — the base model isn't
> *wrong* so much as *generic*. Fine-tuning trades breadth for **precision on your slice**.

---

## 🎯 Interview Questions

**Q1. What fundamentally *is* a language model?**
A probabilistic model of text that estimates `P(next token | previous tokens)`. All downstream
behavior is next-token prediction applied repeatedly.

**Q2. Why is pre-training called "self-supervised" if the corpus is unlabeled?**
Because the supervision signal comes from the data itself: the next (or masked) token *is* the
label. No human annotation is required, which is what lets pre-training scale to trillions of tokens.

**Q3. Contrast encoder-only, decoder-only, and encoder–decoder models.**
Encoder-only (BERT): bidirectional attention + masked-LM → *understanding* tasks (classification,
NER, embeddings). Decoder-only (GPT): causal attention + next-token → *generation* (chat, code).
Encoder–decoder (T5): bidirectional encoder + causal decoder + cross-attention → *transformation*
(translation, summarization). The differentiator is the attention mask + objective.

**Q4. Why can't BERT generate fluent long-form text?**
Its bidirectional attention and masked-LM objective never train it to produce a continuation
from left context alone; it fills blanks given both sides, so it lacks the autoregressive
left-to-right generation skill decoder models are trained for.

**Q5. Give a case where a small fine-tuned model beats a much larger general model, and say why.**
Med-PaLM 2 on USMLE (~86%) or BloombergGPT on finance tasks. A narrow, well-defined domain lets
the smaller model concentrate all capacity on the target distribution, while the larger general
model spreads capacity across everything.

**Q6. Where does fine-tuning sit relative to pre-training and alignment, and how does its data differ?**
It's the middle stage: pre-training (unlabeled web text, self-supervised) → fine-tuning
(smaller, task-specific data, updating *some* weights) → alignment (feedback/preference data to
make outputs safe). Data gets smaller and more curated at each stage.

**Q7. Name two reasons to fine-tune that are about economics rather than quality.**
Lower **cost** and lower **latency**: a specialized model needs shorter prompts (no big few-shot
block) and can be smaller, cutting per-call cost and response time.

---

**One-line takeaway:** *An LLM is a scaled-up next-token predictor; fine-tuning is the middle
training stage that moves some of its weights to turn a generalist into a precise specialist —
and the model you're tuning is almost always decoder-only.*
