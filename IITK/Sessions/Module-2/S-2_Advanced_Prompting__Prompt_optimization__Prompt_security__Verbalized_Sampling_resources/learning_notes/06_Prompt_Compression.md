# Topic 6 — Prompt Compression

> **Session:** 2.2 — Prompt Optimization and Security | IIT Kharagpur × upGrad  
> **Instructor:** Koustav Rudra, Assistant Professor AI, IIT Kharagpur

---

## Overview

**Prompt Compression** is an optimization technique that **reduces the size of input prompts while preserving essential information**, enabling more efficient, faster, and cheaper LLM inference.

As LLM applications grow in complexity — multi-turn conversations, long documents, RAG pipelines with large retrieved contexts — prompt sizes grow rapidly. Managing this is now a first-class engineering concern.

---

## 6.1 The Problem: Why Prompt Size Matters

### Growing Prompt Sizes in Modern Applications

LLM systems continuously process:
- Multi-turn conversation histories (growing with each turn)
- Retrieved document chunks from RAG systems
- Long tool outputs from agentic pipelines
- Structured logs and contextual data

As prompt size grows, it drives up:

| Impact | Description |
|---|---|
| **Computational cost** | Token count directly determines API cost ($/1M tokens) |
| **Response latency** | Longer context = slower time-to-first-token |
| **Context window strain** | Models have fixed context windows (32K, 128K, 200K tokens) |
| **Truncated inputs** | Long prompts may be silently cut off, losing critical information |
| **Inefficient token use** | Many tokens in the prompt carry redundant or low-value information |

### The Core Engineering Question

> *"How do we preserve useful information while reducing token count?"*

This is the fundamental trade-off: **quality vs. compression ratio**.

---

## 6.2 What is Prompt Compression?

> **Prompt Compression:** An optimization technique that reduces the size of the input prompt while preserving essential information.

### The Architecture

**Without compression:**
```
Original Prompt (2000 tokens) ──→ LLM Inference
```

**With compression:**
```
Original Prompt (2000 tokens) ──→ Compressor ──→ Optimized Prompt (400 tokens) ──→ LLM Inference
```

The compressor is a separate module (algorithm, model, or system) that takes the full prompt and produces a compressed version. The target LLM only sees the compressed prompt.

### The Key Insight

Most prompt tokens are **redundant**:
- Filler phrases ("As mentioned earlier...", "It is worth noting that...")
- Repeated structural patterns (same few-shot format repeated many times)
- Verbose explanations that could be condensed
- Low-information connectors ("Furthermore", "In addition to this")

Compression aims to prune these while retaining the high-information tokens.

---

## 6.3 Basic Prompt Compression Techniques

These three techniques can be applied sequentially as a pipeline. Consider the running example throughout:

**Original:**  
> *"Customer John reported unstable internet for 3 days with video call disruptions."*

---

### Technique 1: Extractive Compression

**How it works:**  
Select the most relevant parts of the input while removing redundant or low-value content. Supported by NLP techniques:

- **Named Entity Recognition (NER):** Identifies important entities (people, places, dates, organizations)
- **Keyword Extraction:** Highlights critical content terms using TF-IDF or similar

**Example:**
```
Original:  "Customer John reported unstable internet for 3 days with video call disruptions."

NER identifies:     John (person), 3 days (duration)
Keywords identify:  unstable internet, video call disruptions

Extracted:  "John: unstable internet, 3 days, video issues"
```

**Token reduction:** ~60% (18 tokens → ~7 tokens)  
**Information preserved:** All critical facts (who, what, duration, impact)  
**Information lost:** Formality, natural language structure

**When to use:** Pre-processing structured data, customer support tickets, log entries, factual records

---

### Technique 2: Summarization

**How it works:**  
Converts extracted content into a **concise, readable natural language representation** using abstractive summarization. Unlike extractive compression (which selects existing text), summarization may **rewrite** in shorter form, preserving intent while reducing complexity.

Modern LLMs use **abstractive summarization** — they understand the meaning and rewrite rather than copy-pasting fragments.

**Example (continuing from extractive):**
```
After extraction:   "John: unstable internet, 3 days, video issues"

After summarization: "John: unstable internet (3 days), video issues"
```
(Minor but meaningful improvement — adds parenthetical context, cleaner format)

**More impactful for longer texts:**
```
Original 500-word policy document
        ↓
Summarized: "Refunds allowed within 30 days with receipt. No refunds on
             digital items. Contact support@company.com for exceptions."
```

**When to use:** Long documents, meeting transcripts, email threads, knowledge base articles passed as context

---

### Technique 3: Token-Level Optimization

**How it works:**  
Reduces token count by simplifying at the **word and sub-word level** — abbreviations, removing filler words, compacting phrases — without semantic summarization.

Techniques include:
- Abbreviations: "days" → "d"
- Removing filler words: "as well as" → "and", "in order to" → "to"
- Contracting phrases: "in the year" → "in"
- Removing punctuation noise where safe

**Example (continuing pipeline):**
```
After summarization:  "John: unstable internet (3 days), video issues"
After token-level:    "John: unstable internet, 3d, video issues"
```

**Impact:** LLM cost and latency are **directly proportional** to token count. Even 20-30% token reduction across millions of API calls compounds into significant savings.

**When to use:** Any prompt where cost/latency optimization is required, as a final compression pass

---

## 6.4 Advanced Prompt Compression: The 4-Quadrant Taxonomy

Advanced compression techniques can be organized along two axes:

```
                    REPRESENTATION SPACE
                          ↑
              ┌───────────────────────────────┐
              │  Soft / Latent  │  Semantic   │
LEXICAL       │ (hidden space)  │(summaries)  │  SEMANTIC
SPACE ───────►│─────────────────────────────── │──────────► SPACE
              │ Lexical / Token │Retrieval    │
              │(token pruning)  │-side        │
              └───────────────────────────────┘
                          ↓
                       INPUT SPACE
```

| Quadrant | Technique | Description |
|---|---|---|
| **Lexical/Token** | LLMLingua | Token pruning in the input (lexical) space |
| **Soft/Latent** | gist tokens, ICAE | Compress into hidden representation vectors |
| **Semantic** | Selective-Context, RECOMP | Summarize into human-readable abstractions |
| **Retrieval-side** | Dense retrieval, re-ranking | Reduce context by selecting less (RAG optimization) |

---

## 6.5 LLMLingua — Token-Level Prompt Compression (Microsoft)

### What is LLMLingua?

LLMLingua is a **prompt-compression framework from Microsoft** designed to reduce the number of tokens sent to LLMs while preserving most useful information. The main goal: make inference faster, cheaper, and more context-efficient.

**Key components:**
1. **Budget Controller** — determines how many tokens to allocate to each section
2. **Iterative Token-Level Compression** — prunes tokens based on importance scoring
3. **Distribution Alignment** — aligns small and large model probability distributions

### How LLMLingua Works

```
Original Prompt (2366 tokens)
        ↓
[I] Budget Controller
  - Analyzes prompt structure
  - Sets compression budget per section (instruction, context, question)
        ↓
[0] Distribution Alignment
  - Small model learns to approximate large model distribution
  - Ensures compressed prompt preserves information in the large model's terms
        ↓
[II] Iterative Token-Level Compression
  - Scores each token by its perplexity/importance
  - Prunes low-importance tokens iteratively
        ↓
Compressed Prompt (117 tokens)
        ↓
[III] Compressed Prompt Execution on Black-box LLM (GPT-4, Claude, etc.)
```

### LLMLingua Example (from slides)

**Original Prompt (778 tokens):**
```
Instruction: Please reference the following examples to answer the math question.

[Context: Two detailed multi-step word problems with full reasoning chains]

Question: Josh decides to try flipping a house. He buys a house for $80,000
and puts in $50,000 in repairs. This increased the value by 150%. How much profit?
```

**Compressed Prompt (379 tokens) — Compression ratio: 2.1×:**
```
Instruction: Please reference the following examples to answer the math question.

[Context: Same examples, compressed — key numbers and answers preserved,
verbose reasoning reduced]

Question: [unchanged — instructions and questions are NOT compressed]
```

**Critical observation:** The instruction and question remain intact. Only the **context** (few-shot examples, retrieved documents) is compressed. This is by design — the question and instruction are the most information-dense parts.

### Token Importance Scoring

LLMLingua uses **perplexity** from a small language model (e.g., GPT-2, Llama-7B) to score each token's importance:

- **Low perplexity token** = predictable given context → redundant → prune
- **High perplexity token** = surprising given context → carries information → keep

This is an elegant insight: tokens that a small model can easily predict from context are likely redundant — the LLM can reconstruct them. Surprising tokens are the ones that carry the actual signal.

### LLMLingua Performance

| Metric | Result |
|---|---|
| Token reduction | 2–5× with minimal task degradation |
| Inference cost reduction | Proportional to token reduction |
| Latency reduction | Proportional to token reduction |
| Task performance | Minimal degradation at 2×; some degradation at 5× |

**Compression effectiveness depends on:**
- **Task type:** Factual tasks tolerate more compression; reasoning tasks less so
- **Reasoning depth:** Shallow Q&A compresses better than multi-step math
- **Retrieval quality:** Better retrieved context compresses better (less noise to start)

---

## 6.6 LLMLingua-2 — Learned Prompt Compression

### What's New in LLMLingua-2?

LLMLingua uses **heuristic pruning** based on perplexity — a statistical measure that may not directly optimize task performance. LLMLingua-2 replaces this with a **learned compressor** using:

1. **Data Distillation** — use a teacher LLM to generate high-quality compressed versions of training prompts
2. **Supervised Compression Learning** — train a small compressor model on the distillation data
3. **Task-Aware Token Retention** — the learned model retains tokens based on actual task relevance, not just perplexity

### LLMLingua-2 Pipeline

```
Teacher LLM (GPT-4, etc.)
        ↓
Data Distillation (create training set of original → compressed pairs)
        ↓
Data Annotation (quality control)
        ↓
Quality Control & Filtering
        ↓
Train Compressor (token classifier — each token: preserve or discard)
        ↓
Fast Token Selection at inference
```

The compressor is a **token classifier**: for each input token, it outputs a probability of whether to preserve (p_preserve) or discard (p_discard). Tokens above a threshold are kept; the rest are pruned.

### LLMLingua-2 Example (from slides)

**Original (778 tokens):**
```
Please reference the following examples to answer the math question.

[Full word problems with context and reasoning]

Question: Josh decides to try flipping a house...
```

**Compressed (433 tokens, compression rate 0.5):**
```
reference examples answer math question,
Angelo Melanie plan hours next week study test. 2 chapters textbook study
4 worksheets memorize.

[Severely compressed — articles, prepositions, verbose phrases removed]

Josh flipping house. buys house for $80,000 $50,000 repairs. increased value
150%. profit?
```

**Key difference from LLMLingua:** In LLMLingua-2, **everything** (instruction, context, question) gets compressed since the compressor has been trained to know what's truly essential.

### LLMLingua vs. LLMLingua-2

| Dimension | LLMLingua | LLMLingua-2 |
|---|---|---|
| Compression method | Perplexity-based heuristic pruning | Supervised learned token classifier |
| Protects instruction/question | Yes — only context is compressed | No — everything can be compressed |
| Speed | Slower (iterative compression) | Faster (3-6× inference speedup) |
| Faithfulness | Good | Stronger (learned from task data) |
| Reasoning quality | Some degradation at high ratios | Better preservation |
| Training required | No | Yes (needs distillation data) |
| Compute overhead | Higher (small model scoring every token) | Lower (fast token classifier) |

---

## 6.7 The Quality vs. Compression Trade-Off

This is the fundamental tension in prompt compression:

```
Quality
  │
  │  ●  (No compression — full quality, full cost)
  │  
  │        ●  (2× compression — slight quality drop, half the cost)
  │
  │                ●  (5× compression — moderate drop, 80% cost reduction)
  │
  │                              ●  (10× — significant quality degradation)
  └─────────────────────────────────────────────────────► Compression Ratio
```

**Practical guidance:**
- 2× compression: Generally safe for most tasks, minimal quality degradation
- 5× compression: Task-dependent; acceptable for RAG retrieval contexts, risky for complex reasoning
- 10×+ compression: Only for very long, repetitive contexts with low information density

**The sweet spot** depends on:
- Task sensitivity (customer support vs. medical diagnosis)
- Cost budget
- Latency requirements
- Quality floor (minimum acceptable quality)

---

## Learning Highlights

> **Core insight:** Most LLM prompts contain 50-80% redundant tokens. This isn't a problem with the user's writing — it's structural: natural language is inherently redundant, and LLMs were trained on natural language, so they can reconstruct it.

> **Perplexity insight:** Using a small model's perplexity to score token importance is a beautiful information-theoretic trick. Boring (low-perplexity) tokens are the ones we don't need. Surprising (high-perplexity) tokens are the signal.

> **LLMLingua-2 insight:** Learning to compress is fundamentally better than heuristic compression because it can learn what YOUR specific task cares about, not just what's generally surprising linguistically.

> **Production insight:** For RAG systems specifically, retrieval-side compression (better retrieval selectivity) is often more effective and less risky than post-retrieval token compression. Get better context before compressing it.

> **Trade-off insight:** Compression ratio and quality are inversely related, but the curve is not linear. The first 2× compression is nearly "free" (the pruned tokens were genuinely redundant). Beyond 5×, you start cutting into load-bearing tokens.

---

## Interview Questions

### Foundational

**Q1. What is prompt compression and why is it important?**

*Answer:* Prompt compression reduces the token count of input prompts while preserving essential information. It's important because token count directly determines LLM API cost and response latency; large prompts strain context windows; and modern LLM applications (RAG, multi-turn conversations, agentic pipelines) routinely generate prompts with thousands of tokens. Compression makes these applications economically viable and latency-acceptable at production scale.

---

**Q2. Describe the three basic prompt compression techniques and their order in a pipeline.**

*Answer:* (1) **Extractive compression** — use NER and keyword extraction to select only the most relevant tokens from the original text, discarding low-value words. (2) **Summarization** — rewrite the extracted content into a concise natural language form, preserving intent while reducing length. (3) **Token-level optimization** — reduce token count through abbreviations, removing filler words, and compacting phrases. Applied in this order, each technique further reduces the prompt without losing the essential signal.

---

**Q3. What is LLMLingua and how does it use perplexity to compress prompts?**

*Answer:* LLMLingua is Microsoft's prompt compression framework that uses a small language model to score each token's importance by its perplexity. Low-perplexity tokens (predictable from context) are considered redundant and pruned; high-perplexity tokens (surprising) carry information and are preserved. LLMLingua also uses a budget controller to decide how many tokens to allocate to each prompt section, and iteratively compresses until the budget is met.

---

**Q4. What is the key difference between LLMLingua and LLMLingua-2?**

*Answer:* LLMLingua uses heuristic perplexity-based pruning — it scores tokens using a small model's next-token prediction difficulty. LLMLingua-2 replaces this with a **learned compressor**: a teacher LLM generates high-quality compressed versions of training prompts, which are used to train a token classifier that predicts whether each token should be preserved or discarded. LLMLingua-2 is faster (3-6× inference speedup), more faithful to the task, and better at preserving reasoning quality — but requires training data and a trained compressor model.

---

### Intermediate

**Q5. Describe the four-quadrant taxonomy of advanced prompt compression techniques.**

*Answer:* The taxonomy spans two axes — representation space (lexical vs. latent/embedding) and input handling (token-level vs. semantic level): (1) **Lexical/Token** (LLMLingua) — prune tokens directly in the input text space; (2) **Soft/Latent** (gist tokens, ICAE) — compress into dense vector representations in the model's hidden space; (3) **Semantic** (Selective-Context, RECOMP) — rewrite into concise natural language summaries; (4) **Retrieval-side** — reduce the amount retrieved from external sources via better retrieval selectivity or re-ranking, before any compression is applied.

---

**Q6. In a RAG (Retrieval-Augmented Generation) pipeline, where would you apply prompt compression and which technique would you prefer?**

*Answer:* In a RAG pipeline, compression is most impactful on the **retrieved context** — the retrieved documents are typically the largest portion of the prompt and often contain redundant or marginally relevant content. First try retrieval-side compression: improve retrieval selectivity (better embeddings, re-ranking) to retrieve fewer but more relevant chunks. If further compression is needed, apply LLMLingua to the retrieved context with the instruction and question protected from compression (since they're already concise and information-dense). Avoid compressing the instruction or the user query — these are already minimal.

---

**Q7. At what compression ratio does quality typically start degrading significantly, and why?**

*Answer:* Quality degradation becomes significant around 5× compression for most tasks. Below 2×, pruned tokens are genuinely redundant (filler, connectors, repeated structure), so quality is maintained. Between 2× and 5×, some task-supporting tokens are pruned, causing moderate degradation — acceptable for retrieval contexts but risky for complex reasoning. Above 5×, load-bearing tokens — those that carry the logical connectives, numerical values, and semantic relationships needed for reasoning — are pruned, causing significant task degradation. The exact threshold is task-dependent: shallow Q&A tolerates higher ratios than multi-step math.

---

### Advanced

**Q8. How would you design a prompt compression strategy for a high-volume enterprise customer support chatbot processing 1 million conversations per day?**

*Answer:* (1) **Profile the prompt structure:** Measure the average token distribution across instruction, conversation history, knowledge base context, and user query for your actual traffic. (2) **Apply tiered compression:** Conversation history beyond the last 3 turns → aggressive semantic summarization (10× target). Knowledge base context → LLMLingua with 3× target. Instruction → no compression. User query → no compression. (3) **Quality gate:** For every compression tier, measure task performance on a held-out evaluation set. Set a minimum acceptable quality floor per tier. (4) **Cost-quality Pareto analysis:** Plot quality vs. token cost at different compression ratios; identify the operating point that meets quality floor within cost budget. (5) **Monitor in production:** Track quality metrics on a 1% sample with human review, adjust compression ratios dynamically.

---

**Q9. What is the information-theoretic basis for using perplexity as a token importance score?**

*Answer:* Perplexity measures a language model's surprise at a token given its context. High perplexity = the model found this token unexpected = it carries information not predictable from context. Low perplexity = the model could predict it easily = it's redundant given what came before. This follows directly from Shannon's information theory: the information content of a symbol is proportional to its unexpectedness (−log P). A token that a model can predict with 99% confidence contributes ~0.01 bits of information; a token it finds completely surprising contributes ~6.6 bits. LLMLingua exploits this: use a small model's perplexity as a proxy for token information content, then prune the low-information tokens.

---

## Quick Reference Summary

| Technique | Type | Compression Ratio | Trade-Off |
|---|---|---|---|
| Extractive Compression | Basic | 2-5× | Loses natural language fluency |
| Summarization | Basic | 3-10× | May lose fine-grained detail |
| Token-level optimization | Basic | 1.2-2× | Risk of ambiguity from abbreviation |
| LLMLingua | Advanced (lexical) | 2-5× | Task degradation at high ratios |
| LLMLingua-2 | Advanced (learned) | 2-6× | Requires training data + compressor |
| Soft/Latent | Advanced (embedding) | Up to 10× | Requires model access, not portable |
| Retrieval-side | Advanced (selection) | Indirect | Depends on retrieval quality |
