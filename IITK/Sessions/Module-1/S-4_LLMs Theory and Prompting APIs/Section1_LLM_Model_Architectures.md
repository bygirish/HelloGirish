# Section 1: Large Language Models — Model Architectures
**Lecture 4 | IIT Kharagpur × upGrad | Instructor: Prof. Sourangshu Bhattacharya**

> **Learning Goal:** Understand how LLMs evolved from BERT (2018) to today's multimodal reasoning systems. Be able to place any model on the architecture/training/alignment timeline and explain *why* each generation was a step forward.

---

## Table of Contents
1. [The Big Picture: Four Axes of Progress](#big-picture)
2. [BERT — Contextual Pretraining (2018)](#bert)
3. [GLUE — The Benchmark That Defined an Era](#glue)
4. [GPT-1 & GPT-2 — Decoder-Only Generation (2018–2019)](#gpt12)
5. [LAMBADA — Measuring Long-Range Understanding](#lambada)
6. [T5 & BART — Text-to-Text and Seq2Seq (2019–2020)](#t5bart)
7. [GPT-3 — The Few-Shot Paradigm (2020)](#gpt3)
8. [GPT-3: In-Context Learning Deep Dive](#icl)
9. [InstructGPT & RLHF — Alignment (2022)](#rlhf)
10. [The Open-Weights Wave — LLaMA, Mistral, Falcon (2023–24)](#openweights)
11. [MMLU — The Modern Capability Benchmark](#mmlu)
12. [The Multimodal & Reasoning Era (2023–2025)](#multimodal)
13. [Map of Innovations — Summary Table](#map)
14. [Interview Questions](#interview)
15. [Key Learning Thoughts](#learning-thoughts)

---

## 1. The Big Picture: Four Axes of Progress {#big-picture}

> **Core Insight:** LLM progress is NOT one single breakthrough. It compounds along four independent axes simultaneously.

| Axis | What it means | Example leap |
|---|---|---|
| **Architecture** | How the model is wired | Encoder-only → Decoder-only → Vision-Language |
| **Training Data** | What the model learned from | 3.3B tokens → 300B tokens → multimodal corpora |
| **Alignment** | Making outputs *useful* to humans | Base LM → RLHF → Constitutional AI |
| **Inference-Time Compute** | Thinking harder at generation time | Single pass → Chain-of-thought search |

**Why this matters for interviews:** When asked "why is GPT-4 better than GPT-2?", the answer is never just "more parameters." You need to cite all four axes.

---

## 2. BERT — Contextual Pretraining (2018) {#bert}

### What is BERT?
**BERT** = **B**idirectional **E**ncoder **R**epresentations from **T**ransformers.

It is a **Transformer encoder-only** model. It *reads* the full sentence in both directions (left AND right context at once) and produces rich vector representations of text — it does NOT generate new text.

### Architecture
- **BERT-Base:** 110M parameters, 12 Transformer encoder layers
- **BERT-Large:** 340M parameters, 24 Transformer encoder layers
- Each layer = Multi-Head Self-Attention + Feed-Forward Network + LayerNorm + Residual connections
- Uses **WordPiece tokenization**

> **Class Q&A insight:** "12 layers" does NOT mean 12 simple ReLU hidden layers. Each of those 12 layers is a full Transformer encoder block. ReLU (or GELU in BERT) is used only *inside* the feed-forward sub-network within each block.

### The [CLS] Token
BERT prepends a special `[CLS]` (classification) token to every input. After the model runs, the **[CLS] vector aggregates the full-sentence meaning** and is used for classification tasks (e.g., sentiment, topic).

Example: Input `"[CLS] I love pizza"` → after attention, the CLS vector = summary of the whole sentence → feed to a classifier head.

> **Analogy:** [CLS] is like the "chairman" of a board meeting — it gathers and summarizes everything discussed.

### Two Pre-training Objectives

#### 1. Masked Language Modeling (MLM)
- Randomly mask 15% of input tokens
- Ask the model to predict the masked words
- **Goal:** Forces bidirectional understanding — to predict `[MASK]`, you must read left AND right
- Example: `"I [MASK] food"` → model predicts `"indian"` using both `"I"` and `"food"` as context

#### 2. Next Sentence Prediction (NSP)
- Given two sentences A and B, predict whether B actually follows A in the corpus
- **Goal:** Teaches sentence-level coherence understanding
- Used for tasks like question answering (is the answer in this passage?)

> **Important distinction — masking in BERT vs GPT:**
> - BERT masking (MLM) = hide *random* words → goal is *understanding*
> - GPT masking (Causal) = hide *all future* tokens → goal is *generation*
> These are opposite purposes — don't confuse them.

### Why BERT Mattered
1. **Bidirectional context in one pass** — previous models (like ELMo) were shallow bidirectional; BERT is deep bidirectional
2. **Transfer via fine-tuning** — add a small task-specific head + a few epochs of fine-tuning = new SOTA on 11 NLP benchmarks
3. **Established pretrain-then-finetune as the dominant paradigm** — everything that came after used this pattern
4. **GLUE score:** BERT-Large hit 82.1 (vs 75.1 for OpenAI GPT at the time)

### When is BERT Still Used Today?
Despite generative models dominating, BERT-style models are still preferred when:
- **Classification** — spam detection, sentiment analysis (faster, cheaper, no need for generation)
- **Named Entity Recognition (NER)** — finding names, places in text
- **Search & Retrieval** — semantic similarity matching
- **Extractive Question Answering** — picking the answer *from* a passage, not generating it
- **Why?** Faster, cheaper, more stable for fixed tasks. No need to generate long text.

---

## 3. GLUE — The Benchmark That Defined an Era {#glue}

**GLUE** = **G**eneral **L**anguage **U**nderstanding **E**valuation

Introduced in 2018 as a collection of **9 diverse NLU tasks** to test a model's language understanding across multiple levels of abstraction. Results reported as a **single average score** (like an exam grade for language understanding).

### Task Categories

| Category | Tasks | Example |
|---|---|---|
| **Single-Sentence** | CoLA (linguistic acceptability), SST-2 (sentiment) | "He go to school" → low probability |
| **Similarity & Paraphrase** | MRPC (paraphrase pairs), STS-B (semantic similarity), QQP (duplicate questions) | Are these two sentences saying the same thing? |
| **Inference** | MNLI (multi-genre NLI), QNLI, RTE, WNLI | Does sentence A entail/contradict sentence B? |

### Why It Mattered
- Before GLUE, each paper tested on a different benchmark — no fair comparison
- GLUE created a **standardized leaderboard** that everyone raced to top
- BERT-Large immediately pushed SOTA from 71.0 (BiLSTM+ELMo) to 82.1
- **SuperGLUE** (harder version) came later; T5-11B reached 88.9 vs 89.8 human baseline

> **Learning thought:** Benchmarks are the "exams" of AI. A model's GLUE score isn't a real application — it's a standardized test. The value is in comparing models fairly, not in deploying GLUE itself.

---

## 4. GPT-1 & GPT-2 — Decoder-Only Generation (2018–2019) {#gpt12}

### The Key Architectural Shift
While BERT uses the **encoder** (reads full sentence, understands), GPT uses only the **decoder** (predicts next token, generates). This is the split that still defines LLMs today:
- **Encoder-only** (BERT) → Understanding tasks
- **Decoder-only** (GPT) → Generation tasks

### What "Causal" Means
The GPT decoder is **causal** — at each step, it can ONLY see past tokens, never future ones. This is enforced by **causal masking** (future tokens are masked to -∞ before softmax).

> Example: Generating "I like pizza" token by token:
> - Step 1: `I` → model sees `[BOS]`, predicts `I`
> - Step 2: `like` → model sees `[BOS] I`, predicts `like`
> - Step 3: `pizza` → model sees `[BOS] I like`, predicts `pizza`

### GPT-1 (June 2018)
- **117M parameters**, decoder-only Transformer
- Trained on BookCorpus (~800M words), then fine-tuned on task-specific data
- Outperformed task-specific SOTA on 9/12 tasks at the time
- **Key innovation:** A single generative pre-trained model can match task-specific architectures after fine-tuning

### GPT-2 (February 2019)
- **1.5B parameters** (13× GPT-1), trained on WebText (40GB of Reddit-linked text)
- **Dropped the fine-tuning step** — showed that a big enough model can do tasks from natural language prompts alone
- Achieved **zero-shot performance** on translation, summarization, QA
- **LAMBADA SOTA surpassed zero-shot:** 63.2% accuracy without task-specific training
- First widely-felt hint of **emergent capability** — capability that appears suddenly at scale

> **Common misconception:** "GPT-2 doesn't require training datasets." FALSE. GPT-2 was trained on 40GB of WebText. "Zero-shot" means: no *task-specific examples* at *inference time* — not that the model has no training.

> **Is GPT-2 overfitting?** No. Overfitting = memorize training data, fail on new data. GPT-2 was trained on huge, diverse web text and shows generalization to tasks never explicitly trained on (QA, summarization, translation). It generalizes.

---

## 5. LAMBADA — Measuring Long-Range Understanding {#lambada}

**LAMBADA** = **L**anguage **M**odeling **B**roadened to **A**ccount for **D**iscourse **A**spects

### What It Tests
A model's ability to **predict the last word of a narrative passage** where local context alone is INSUFFICIENT. The entire multi-sentence discourse is required.

Example passage (paraphrased): `"Alice opened the door. She walked into the living room. The smell of coffee hit her. She called out—'Is anyone home?' Silence. She sat down feeling..."` → predict: `"alone"`.

The last word cannot be predicted from just the final sentence — you need to track the whole story.

### Why N-gram Models Fail
N-gram models score near **0% accuracy** on LAMBADA — they only use local context windows. LAMBADA specifically tests **discourse-level comprehension**, not just co-occurrence patterns.

### GPT-2's Significance
GPT-2 (1.5B) was the **first model to surpass LAMBADA prior SOTA zero-shot** — meaning scale alone (without task supervision) was enough to learn discourse structure. This was early evidence that LLMs encode world knowledge and story structure, not just word patterns.

> **LAMBADA vs CBOW:** Both predict a word from context, but CBOW uses a small fixed window of nearby words. LAMBADA requires multi-sentence understanding. They're in completely different leagues.

---

## 6. T5 & BART — Text-to-Text and Seq2Seq (2019–2020) {#t5bart}

### T5 (Google, Raffel et al., 2019)
**T5** = **T**ext-**t**o-**T**ext **T**ransfer **T**ransformer

**Core idea:** Cast EVERY NLP task as a string-to-string prediction problem using a single model.

| Task | Input | Output |
|---|---|---|
| Sentiment | `"classify sentiment: food was bad"` | `"negative"` |
| Translation | `"translate English to French: hello"` | `"bonjour"` |
| Summarization | `"summarize: [long article]"` | `"short summary"` |
| QA | `"question: capital of France? context: Paris is..."` | `"Paris"` |

- **Architecture:** Full encoder–decoder Transformer
- **Training objective:** Span-corruption (randomly mask contiguous spans, predict them)
- **Data:** C4 (Colossal Clean Crawled Corpus, 750 GB)
- **Scale:** 60M → 11B parameters
- **Result:** T5-11B scored 88.9 on SuperGLUE vs 89.8 human baseline — nearly closing the gap to human performance

### BART (Facebook AI, Lewis et al., 2019)
**BART** = **B**idirectional **A**uto-**R**egressive **T**ransformer

- Architecture: **BERT-like encoder** (bidirectional) + **GPT-like decoder** (autoregressive)
- Pre-training: Denoising objective — introduce varied corruptions (deletion, masking, permutation), train to reconstruct original text
- Strong at **summarization and generation** tasks
- Template for later models: mBART (multilingual), Pegasus (summarization)

### Main Difference: BERT vs T5
| Aspect | BERT | T5 |
|---|---|---|
| Architecture | Encoder-only | Encoder-decoder |
| Output | Label / embedding | Generated text |
| Example | `"food was bad"` → `negative` | `"text: food was bad"` → `"negative"` (as text) |
| Use case | Understanding | All tasks as text generation |

> **Are T5/BART still used?** Yes — for structured text generation (summarization, translation, Q&A with context) where they are often faster and cheaper than giant LLMs.

---

## 7. GPT-3 — The Few-Shot Paradigm (2020) {#gpt3}

### Scale Leap
GPT-3 scaled the GPT-2 recipe by **two orders of magnitude**:

| Metric | GPT-2 | GPT-3 |
|---|---|---|
| Parameters | 1.5B | **175B** |
| Training tokens | 40GB | **~300B tokens** |
| Context window | 1,024 | **2,048 tokens** |

### The Key Insight
> A sufficiently large language model can perform novel tasks from **a handful of demonstrations in the prompt — no gradient updates, no fine-tuning**.

This is the **prompting paradigm** — the era of API-served foundation models began here.

### What "Scaling the recipe by 2 orders of magnitude" means
It's not just more parameters — it's the entire training setup:
- Same Transformer architecture (decoder-only)
- Same training objective (next-token prediction)
- Same optimization strategy (Adam, LR schedules, etc.)
- But: 117× more parameters, ~7× more data, larger context window

### TriviaQA Results (closed-book)
| Setting | Accuracy |
|---|---|
| Prior fine-tuned SOTA | 68.0% |
| GPT-3 zero-shot | 64.3% |
| GPT-3 one-shot | 68.0% |
| GPT-3 few-shot (64 examples) | **71.2%** |

GPT-3 with 64 examples **beat fine-tuned SOTA** — without any weight updates. This was a landmark result.

---

## 8. GPT-3: In-Context Learning (ICL) Deep Dive {#icl}

### The Three Modes

#### Zero-Shot Learning
- **Only a task description** in the prompt; no examples provided
- Model uses world knowledge captured during pre-training
- Useful baseline; no prompt engineering required
- Example: `"Translate to French: The sky is blue"`

#### Few-Shot Learning (In-Context Learning)
- **10–100 input-output examples** precede the query
- **No weight updates** — the model adapts purely via attention over context
- Approaches fine-tuned smaller models on many tasks
- Example: Show 10 English→French pairs, then ask for the 11th

#### Scale and Emergence
- Smaller GPT-3 variants show **weaker in-context learning**
- The capability **emerges sharply with scale** — not a smooth trend
- This "phase transition" behavior hints at qualitative shifts beyond mere parameter count

### How Does ICL Work Without Gradient Updates?
> **Class Q&A insight:** In few-shot learning, weights ARE fixed — gradient descent happened only during training. At inference, the model "reads" the examples in the prompt and uses attention to recognize the input-output pattern. Example: seeing `"2 → 4, 3 → 6, 5 → ?"` the model recognizes a doubling pattern via attention — no learning happening, just pattern matching from training.

### Dense Models
A **dense model** is one where **every parameter activates for every input** — nothing is skipped. GPT-3, GPT-4, LLaMA, Mistral (base) are dense models. (Contrast with Mixture-of-Experts where only a subset of parameters activate per token.)

---

## 9. InstructGPT & RLHF — Aligning Models with Human Intent (2022) {#rlhf}

### The Problem with Base LMs
A raw base language model (like GPT-3) **predicts likely text continuations** — it doesn't necessarily give *helpful answers*. Ask it a question; it might respond with another question (that's how the web works — questions are often followed by questions).

### The RLHF Solution
InstructGPT aligned GPT-3 using **Reinforcement Learning from Human Feedback (RLHF)** — a three-stage pipeline:

#### Stage 1: Supervised Fine-Tuning (SFT)
- Collect **human-written prompt/response pairs** demonstrating helpful behavior
- Fine-tune the base GPT-3 model on these demonstrations
- Teaches the model the **format and style of helpful replies**

#### Stage 2: Reward Model Training
- Annotators rank multiple model outputs for the same prompt (pairwise comparisons)
- Train a separate **reward model** to predict human preference from these rankings
- The reward model = a proxy for "what humans find helpful, harmless, honest"

#### Stage 3: RL with PPO (Proximal Policy Optimization)
- Use the reward model to guide generation: outputs that score higher on the reward model get reinforced
- **KL penalty** keeps the fine-tuned policy close to the SFT model (prevents reward hacking — model finding degenerate ways to score high)
- Produces **helpful, harmless, honest** model behavior

### Key Result
Annotators preferred the **1.3B InstructGPT over the 175B base GPT-3** — alignment beats raw scale on helpfulness. A smaller, aligned model outperforms a 100× larger unaligned model in practical use.

### Win-Rate Comparison
| Model | Win-rate vs 175B SFT baseline |
|---|---|
| GPT-3 175B (default) | 37% |
| GPT-3 175B (prompted) | 58% |
| InstructGPT 1.3B (PPO-ptx) | **71%** |
| InstructGPT 175B (PPO-ptx) | 85% |

### Released as ChatGPT
This technique was released publicly as ChatGPT in late 2022 and defined what "using an LLM" feels like today.

---

## 10. The Open-Weights Wave — LLaMA, Mistral, Falcon (2023–24) {#openweights}

### Context
Through 2022, the frontier was locked behind commercial APIs. Meta's LLaMA changed everything.

### LLaMA / LLaMA 2 / LLaMA 3 (Meta, Feb 2023 → 2024)
- **7B – 70B dense models**
- Weights distributed to researchers; LLaMA 2 added commercial licensing
- Enabled a rich ecosystem of fine-tunes: **Alpaca, Vicuna, and thousands more**
- Template for modern open models
- Made LLM customization accessible to any lab with a **single GPU server**

### Mistral & Mixtral
- **Mistral 7B (Sep 2023):** Compact dense model with strong quality-per-parameter; Apache-2.0 license
- **Mixtral 8×7B (Dec 2023):** Open **Mixture-of-Experts (MoE)** model — only 2 of 8 expert layers activate per token, reducing compute while maintaining capacity
- Reached **70.6 on MMLU** (5-shot) — matching GPT-3.5 (closed)

### Falcon (TII, UAE)
- Falcon-40B, Falcon-180B
- Training data (RefinedWeb) released publicly — emphasis on **transparency**
- Later joined by Qwen, DeepSeek, Yi as the open-weights ecosystem expanded

### Why This Mattered
- Democratized LLM research — no longer dependent on OpenAI/Google APIs
- Enabled fine-tuning, auditing, and on-device deployment
- Created competitive pressure that accelerated frontier model releases

---

## 11. MMLU — The Modern Capability Benchmark {#mmlu}

**MMLU** = **M**assive **M**ultitask **L**anguage **U**nderstanding

### What It Tests
- **57 diverse subjects** including STEM, humanities, and social sciences
- **Multiple-choice questions** (4 options) testing both knowledge and reasoning
- Used as a **standard measure of general knowledge** in AI model reports
- Typical evaluation: **5-shot** (5 examples provided in the prompt)

### Scope of Subjects
Mathematics, physics, chemistry, biology, computer science, history, law, philosophy, medicine, economics, and more.

### Why It Became the Standard
- Single number summarizing broad knowledge coverage
- Easy to compare across models fairly
- Human performance ≈ 89.8% on MMLU (expert level)
- Modern models: Mixtral 8×7B = 70.6, LLaMA-2 70B = 68.9, GPT-3.5 = 70.0

---

## 12. The Multimodal & Reasoning Era (2023–2025) {#multimodal}

### Two Frontier Axes

#### Natively Multimodal Models
The frontier bifurcated: models became natively multimodal on the **input side**.

| Model | Modalities |
|---|---|
| GPT-4 / GPT-4V (OpenAI, Mar 2023) | Text + Image |
| Gemini 1.0/1.5/2 (Google DeepMind) | Text + Image + Audio + Video |
| Claude 3 & 4 (Anthropic) | Long context + Vision |

**How multimodal works:** A vision encoder (like CLIP or ViT) converts images into "visual tokens" that look (to the Transformer) similar to text tokens. The Transformer then processes text+visual tokens as one unified stream.

**Text-only vs Multimodal:**
- Text-only models have no mechanism to convert pixels into tokens — they are "blind" to images
- Multimodal models use unified tokenizers over text, image, and audio modalities

#### Inference-Time Reasoning Models
A new generation of systems spent **significant inference-time compute** on chain-of-thought search:

| Model | Year | Key innovation |
|---|---|---|
| OpenAI o1 | Sep 2024 | Chain-of-thought as first-class output; RL on verifiable outcomes |
| OpenAI o3 | 2025 | Scaled inference compute further |
| DeepSeek-R1 | 2025 | Open reasoning via RL |

**What changed:** Previously, compute was spent at *training time*. Now, compute can be spent at *inference time* — the model "thinks longer" on hard problems. For the first time, models routinely solved competition-level mathematics and programming problems.

> **Major disadvantage of reasoning models:** They are **time/token consuming**. Extended thinking means more tokens generated, higher latency, higher cost per query. This is the primary trade-off vs non-reasoning models.

---

## 13. Map of Innovations — Summary Table {#map}

| Era | Architecture | Training / Data | Alignment / Inference |
|---|---|---|---|
| Pre-2018 (word2vec, ELMo) | Shallow embeddings; biLSTM LM | Co-occurrence statistics; small corpora | Feature-based transfer |
| BERT (2018) | Deep bidirectional encoder | MLM + NSP on 3.3B tokens | Fine-tune with small task head |
| GPT-1/GPT-2 (2018–19) | Decoder-only causal Transformer | BookCorpus → WebText (40 GB) | Zero-shot task transfer via prompts |
| T5/BART (2019–20) | Encoder–decoder seq2seq | C4 span-corruption; denoising | Unified text-to-text interface |
| GPT-3 (2020) | 175B dense Transformer | ~300B tokens, Common Crawl mix | In-context / few-shot learning |
| Scaling / Chinchilla (2020–22) | Same arch, principled sizing | ~20 tokens/parameter optimal | Predictable loss curves |
| InstructGPT / RLHF (2022) | Base LM + reward model | Human demos + preference data | SFT + RLHF produce helpful assistants |
| Open-weights (2023–24) | LLaMA, Mistral (MoE), Falcon | Open corpora; permissive licenses | On-device, fine-tunable, auditable |
| Multimodal / Reasoning (2023–25) | Vision-language; deliberative policies | Text + image + audio; RL on outcomes | Inference-time chain-of-thought search |

> **Key takeaway:** Progress compounds along all four axes — not any single breakthrough.

---

## 14. Interview Questions {#interview}

### Foundational Architecture Questions

**Q1: What is the difference between an encoder-only and decoder-only model? Give use cases for each.**
> **Answer:** Encoder-only (BERT): sees the full sentence at once, learns bidirectional context, outputs representations — used for understanding tasks (classification, NER, retrieval, extractive QA). Decoder-only (GPT): sees only past tokens, predicts next token, outputs generated text — used for generation tasks (chatbots, code generation, summarization, translation). The key insight: *understanding needs to see the whole sentence; generation must not see the future.*

**Q2: Explain Masked Language Modeling vs Next Token Prediction. Why does each suit its model family?**
> **Answer:** MLM (BERT): randomly hides 15% of tokens and asks the model to predict them. This forces bidirectional reasoning — to predict a masked word, you need both left and right context. Perfect for an encoder that produces rich representations. NTP/Causal LM (GPT): predicts the next token given all preceding tokens. This is inherently left-to-right and trains generation. You can't use MLM for a decoder because revealing future tokens during training would make generation trivially easy.

**Q3: What is the CLS token in BERT and what is it used for?**
> **Answer:** [CLS] is a special classification token prepended to every BERT input. Unlike other tokens that represent individual words, the CLS token is designed (through pre-training) to aggregate the overall meaning of the entire input sequence. Its final-layer embedding is used as a sentence-level representation for downstream classification tasks like sentiment analysis, topic classification, and NLI.

**Q4: What are the three stages of RLHF and why is each necessary?**
> **Answer:** (1) SFT: fine-tune on human demonstrations to teach the model format and style of helpful responses — without this, the reward model has no good baseline. (2) Reward Model: train a model on pairwise human rankings — this creates a scalable proxy for human preference since you can't ask humans to evaluate every generated output. (3) PPO: optimize the policy against the reward model with a KL penalty — the KL penalty prevents reward hacking (finding degenerate outputs that fool the reward model). Each stage solves a failure mode of the previous stage.

**Q5: Why did annotators prefer 1.3B InstructGPT over 175B base GPT-3?**
> **Answer:** Raw scale gives language modeling ability (predicting likely text), not *helpfulness*. A base model responds in ways that are statistically likely given the training corpus (questions followed by questions, code followed by code). RLHF explicitly optimizes for human preference — helpfulness, harmlessness, honesty. Alignment fundamentally changes the objective, not just the capability level.

### Benchmark & Evaluation Questions

**Q6: What is GLUE and what types of tasks does it cover?**
> **Answer:** GLUE (General Language Understanding Evaluation, 2018) is a collection of 9 NLU tasks covering: single-sentence tasks (CoLA for linguistic acceptability, SST-2 for sentiment), similarity/paraphrase tasks (MRPC, STS-B, QQP), and inference tasks (MNLI, QNLI, RTE, WNLI). Results are reported as a single average score, enabling fair model comparisons. BERT-Large's release immediately pushed GLUE from 71.0 to 82.1, cementing pretrain-then-finetune as the dominant paradigm.

**Q7: What is the LAMBADA benchmark and why is it hard for n-gram models?**
> **Answer:** LAMBADA tests whether a model can predict the last word of a narrative passage where local context is insufficient — the whole discourse is required. N-gram models score near 0% because they only look at a fixed window of nearby words, missing the multi-sentence story context. LAMBADA specifically tests genuine discourse comprehension, not word co-occurrence.

**Q8: What is MMLU and why is it used as a standard benchmark?**
> **Answer:** MMLU (Massive Multitask Language Understanding) covers 57 subjects (STEM, humanities, social sciences) with multiple-choice questions. It is the standard benchmark because: (1) broad coverage tests general knowledge, not just one skill; (2) multiple-choice format is unambiguous to score; (3) it captures both knowledge and reasoning; (4) 5-shot setting tests in-context learning. Human expert performance ≈ 89.8%.

### Evolution & Scaling Questions

**Q9: What was the key insight of GPT-2 vs GPT-1?**
> **Answer:** GPT-1 showed a single pre-trained model could match task-specific architectures after fine-tuning. GPT-2 removed the fine-tuning step entirely — scale alone (1.5B params, 40GB web text) was enough for zero-shot task transfer. This was the first widely-felt evidence of emergent capability: the model was never trained for QA, translation, or summarization explicitly, yet performed all three from natural-language prompts.

**Q10: What is in-context learning and how does it differ from fine-tuning?**
> **Answer:** In-context learning (ICL): provide examples inside the prompt itself; the model adapts by attention over the context with NO weight updates. Fine-tuning: run gradient descent on examples, updating the model's weights permanently. ICL is temporary (only for that conversation), cheap (no GPU training), and flexible (swap tasks by changing the prompt). Fine-tuning is permanent, more compute-intensive, but can achieve deeper adaptation for a specific domain.

**Q11: What is Mixture-of-Experts (MoE) and how does Mixtral use it?**
> **Answer:** MoE is an architecture where the model has multiple "expert" feed-forward networks at each layer, but only activates a subset (typically 2) per token via a routing mechanism. Mixtral 8×7B has 8 experts per layer but routes each token to only 2, so the effective compute per token ≈ a 12-13B dense model, but the model has 47B total parameters (full capacity for diverse knowledge). This gives strong quality-per-compute-dollar.

**Q12: What is the main disadvantage of reasoning models like o1?**
> **Answer:** They are **time/token consuming**. Extended chain-of-thought reasoning generates many intermediate tokens before producing the final answer. This means: higher latency per query, significantly higher cost per token compared to non-reasoning models, and unsuitability for real-time interactive applications. Use them only when the task genuinely requires deep multi-step reasoning (math proofs, complex code).

---

## 15. Key Learning Thoughts {#learning-thoughts}

> **Thought 1 — The Encoder/Decoder Split is Fundamental**
> Every LLM you encounter can be categorized along this axis. Encoder = understand (BERT, RoBERTa, DeBERTa). Decoder = generate (GPT family, LLaMA, Mistral). Encoder-decoder = translate/transform (T5, BART). When you see a new model, ask first: "What architecture family?"

> **Thought 2 — Benchmarks Are Not Applications**
> GLUE, MMLU, LAMBADA are measurement tools. A model achieving 88.9 on SuperGLUE doesn't "do SuperGLUE tasks" in production. The benchmark tells you the model's capability ceiling. Know the benchmarks, but build applications.

> **Thought 3 — Zero-Shot ≠ No Training**
> This is the #1 misconception beginners have. Zero-shot means no task-specific examples at *inference time*. The model was extensively trained (GPT-2 on 40GB of text). "Zero-shot" is about what you provide in the prompt, not about whether the model was trained.

> **Thought 4 — Alignment Is a Separate Axis from Capability**
> You can have a highly capable but useless model (base GPT-3) or a less capable but very useful model (InstructGPT 1.3B). When evaluating an LLM for your application, ask both: "How capable is it?" AND "How well is it aligned to helpful responses?"

> **Thought 5 — Emergent Capabilities Are Scale-Dependent**
> Some capabilities (in-context learning, zero-shot translation, chain-of-thought reasoning) don't improve smoothly with scale — they jump suddenly at certain size thresholds. This makes it hard to predict what a 10× larger model will be able to do. It also means: your application may suddenly "work" when you switch from a small model to a large one.

> **Thought 6 — Open Weights Changed the Power Dynamic**
> Before LLaMA (Feb 2023), serious LLM work required OpenAI or Google API access. After LLaMA, a single-GPU researcher could fine-tune a 7B model. This democratized AI development and accelerated the pace of innovation dramatically.

> **Thought 7 — Inference-Time Compute is the New Frontier**
> For decades, the question was "how do we train better models?" The new question is "how do we *use* models more intelligently at inference?" Chain-of-thought, tree-of-thought, and reasoning models like o1 represent this shift. Compute is increasingly shifting from training to inference.

---

## References (from lecture)
- Devlin et al. (2018/2019). BERT. arXiv:1810.04805
- Radford et al. (2018). GPT-1. OpenAI Technical Report
- Radford et al. (2019). GPT-2. arXiv: Language Models are Unsupervised Multitask Learners
- Brown et al. (2020). GPT-3: Language Models are Few-Shot Learners. arXiv:2005.14165
- Ouyang et al. (2022). InstructGPT. arXiv:2203.02155
- Touvron et al. (2023). LLaMA. arXiv:2302.13971
- Jiang et al. (2023). Mistral 7B. arXiv:2310.06825
- OpenAI (2023). GPT-4 Technical Report. arXiv:2303.08774
- DeepSeek-AI (2025). DeepSeek-R1. arXiv:2501.12948
