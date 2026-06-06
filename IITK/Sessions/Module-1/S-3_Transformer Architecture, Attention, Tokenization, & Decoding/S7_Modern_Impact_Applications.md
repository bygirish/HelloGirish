# Section 7: Broader Context & Modern Impact

> **Learning Goal:** Understand how the Transformer's self-attention led to the pretraining paradigm, why the modern AI landscape (BERT, GPT, LLaMA, ViT, Whisper, CLIP) is built on this one architecture, and how to connect all concepts into a unified worldview.

---

## 7.1 The Evolution of Representations

Before Transformers, NLP went through several representational eras. Each solved problems of the previous, and introduced new ones.

| Representation | Strength | Weakness |
|---------------|----------|----------|
| **One-Hot** | Gave structure to discrete tokens | Sparse; no semantics ("cat" and "kitten" are orthogonal) |
| **Dense Embeddings** | Compact vectors in continuous space | Same vector for all contexts |
| **Word2Vec / GloVe** | Captured semantic meaning (king - man + woman ≈ queen) | Static: "bank" = same vector in "river bank" and "bank account" |
| **ELMo / BiLSTM** | Contextual vectors (different "bank" in each context) | Sequential, slow, limited depth |
| **Cross-Attention** | Alignment across sequences (Bahdanau) | No within-sequence attention |
| **Self-Attention** | Every token ↔ every token in same sequence | Position-agnostic (permutation invariant) |
| **Self-Attention + Positional Encoding** | Order-aware, fully contextual | The current state of the art ✓ |

### The Key Insight Behind Pretraining

Self-attention learns **general-purpose representations** of language:
- Q, K, V weight matrices learn universal patterns: syntax, semantics, coreference
- These patterns are **task-independent** — "cat attends to animal" is useful for translation, QA, summarization, etc.

> **The paradigm shift:** If attention learns general representations, why train from scratch for every new task? Learn once on massive data. Reuse everywhere.

This is the birth of **pretraining + fine-tuning** — the dominant paradigm for the past 6 years.

```
Raw Text (billions of tokens)
        ↓
Transformer (self-attention learns)
        ↓
Pretrained Weights (Q, K, V, FFN)
        ↓
Fine-tune on small labeled data    OR    Prompt / in-context learning
        ↓
Deployed Task-Specific Model
```

---

## 7.2 Transformers — "Attention Is All You Need" (2017): Why It Changed Everything

### The Self-Attention Mechanism

| Property | What it gives you |
|----------|------------------|
| **Context-aware representations** | Every token sees every other token; output encodes full sentence context |
| **Multi-head = multiple views** | Syntax, semantics, coreference captured simultaneously |
| **Long-range dependencies — O(1)** | Any two tokens connect in one attention step regardless of distance |
| **Fully parallel computation** | No sequential bottleneck; all tokens processed at once |

### Why Transformers Enabled Scale

RNNs couldn't scale because:
1. Sequential processing → GPU underutilization → slow training
2. Vanishing gradients → couldn't train very deep models

Transformers solved both. The result: you could train larger models on more data faster.

```
GPT-1 (2018):    117M parameters
BERT (2018):     340M parameters
GPT-2 (2019):    1.5B parameters
GPT-3 (2020):    175B parameters
GPT-4 (2023):    ~1.76T parameters (estimated)
Claude 3 (2024): undisclosed, likely similar scale
```

### The Emergence Phenomenon

As models scale, they acquire capabilities that smaller models lack — seemingly without being explicitly trained for them:
- Few-shot learning
- Chain-of-thought reasoning
- Code generation
- Mathematical problem solving

These **emergent capabilities** are only possible because the Transformer architecture scales so well.

---

## 7.3 From Attention to Pretrained Models

### BERT (2018) — Encoder-Only, Bidirectional

**Architecture:** Encoder only, no masking → every token sees every other token bidirectionally
**Pretraining task:** Masked Language Modeling (MLM) — mask 15% of tokens, predict them
**What it learns:** Bidirectional contextual representations — "bank" gets different vectors based on full context
**Use cases:** Classification, Named Entity Recognition, Question Answering, Semantic Similarity

```
BERT training:
Input:  "The [MASK] sat on the mat."
Target: "cat"    (predict the masked token)
```

BERT learns to predict masked tokens by attending to the full context — this forces it to understand bidirectional relationships.

### GPT (2018 onwards) — Decoder-Only, Causal

**Architecture:** Decoder only, causal masking → each token sees only previous tokens
**Pretraining task:** Causal Language Modeling (CLM) — predict the next token
**What it learns:** Left-to-right generative patterns — how to complete text coherently
**Use cases:** Text generation, chatbots, code generation, reasoning

```
GPT training:
Input:  "The cat sat"
Target: "on"    (predict next token)
```

By predicting billions of next tokens, GPT learns the structure and content of language implicitly.

### T5 / BART — Encoder-Decoder

**Architecture:** Full encoder-decoder (like original Transformer)
**Pretraining:**
- T5: "Text-to-Text" — every task is framed as text→text
- BART: Corrupted text → original text
**Use cases:** Translation, summarization, Q&A (tasks requiring seq2seq mapping)

### Fine-Tuning Steers Attention

The pretrained Q, K, V weights (learned from massive data) are adjusted for your specific task using small labeled datasets. Fine-tuning doesn't rebuild from scratch — it steers the pretrained attention patterns toward task-specific behavior.

### PEFT — Freeze Attention, Tune Adapters (LoRA)

**Problem:** Full fine-tuning of a 7B-parameter model requires storing and updating 7B parameters per task — expensive.

**LoRA (Low-Rank Adaptation):**
```
Instead of updating W (d×d), learn a small ΔW = A·B (d×r × r×d, where r << d)
W_fine-tuned = W_pretrained + A·B
```

- Pretrained W is frozen (unchanged)
- Only A and B are trained (r=8 → 0.1% of parameters)
- Quality is ~equal to full fine-tuning

> LoRA makes fine-tuning accessible: fine-tune a 7B model on a consumer GPU in hours.

---

## 7.4 The Modern Transformer Zoo

All of these models share the same core: **Embeddings + Positional Encoding + Self-Attention + FFN, stacked**.

| Model | Architecture | Task | Notes |
|-------|-------------|------|-------|
| **BERT** | Encoder only | Classification, NER, QA | Bidirectional; best for understanding |
| **GPT-4** | Decoder only | Generation, chat, code | Unidirectional; best for generation |
| **T5 / BART** | Encoder-Decoder | Translation, summarization | Seq2seq; full encoder-decoder |
| **ViT** | Encoder only (patches) | Image classification | Self-attention on image patches |
| **Whisper** | Encoder-Decoder | Speech → Text | Enc: audio features; Dec: transcript |
| **CLIP** | Dual encoder | Image-text matching | Separate image/text encoders; aligned |

### Vision Transformer (ViT)

Applied the exact same Transformer encoder to images:
1. Split image into 16×16 patches
2. Each patch = a token (embedded as a vector)
3. Add positional encoding
4. Apply standard Transformer encoder

Result: patches attend to each other via self-attention — "which parts of the image are relevant for understanding this patch?"

This shows the Transformer is not just an NLP architecture — it's a **general sequence processing architecture** applicable to any modality.

### Whisper

Applied the encoder-decoder Transformer to audio:
- Encoder: processes mel-spectrogram (audio features)
- Decoder: autoregressively generates transcript tokens

Cross-attention in the decoder aligns audio segments with words — when generating "cat", the decoder attends to the audio segment where "cat" is spoken.

### CLIP (Contrastive Language-Image Pretraining)

Two Transformers working together:
- Text encoder (BERT-like): encodes captions
- Image encoder (ViT-like): encodes images
- Trained to bring matching text-image pairs close in embedding space

```
"a photo of a cat" ↔ [cat image]   → aligned embeddings
"a photo of a dog" ↔ [cat image]   → misaligned embeddings
```

CLIP enables zero-shot classification: embed class names ("cat", "dog") and find which image embedding is closest.

---

## 7.5 Quiz Reviews — Conceptual Anchors

### Quiz 1: Self-Attention vs Cross-Attention

**Question:** In a Transformer, what is the primary difference between self-attention and cross-attention in terms of Q, K, V sources?

**Answer:** **A** — In Self-Attention, Q, K, and V all come from the same input sequence; in Cross-Attention, Q comes from the decoder and K and V come from the encoder.

**Why the others are wrong:**
- B: Q and K from encoder in self-attention? Wrong — all from same sequence
- C: No difference? Wrong — they have completely different Q/K/V sources
- D: Self-attention only in decoder? Wrong — encoder uses self-attention; decoder uses masked self-attention AND cross-attention

---

### Quiz 2: Positional Encodings

**Question:** Why are positional encodings added to input embeddings in a Transformer?

**Answer:** **C** — To provide the model with information about the order/position of words, since the attention mechanism itself is permutation-invariant.

**Why the others are wrong:**
- A: To reduce parameters? Wrong — PE adds no parameters (sinusoidal) or small embedding table (learned)
- B: To process sequentially like RNN? Wrong — Transformers are still fully parallel after PE; PE doesn't make them sequential
- D: To normalize before MHA? Wrong — LayerNorm does normalization; PE has no normalization purpose

---

## 7.6 The Full Architecture — Everything Connected

```
                    "I love you"
                         ↓
              ┌─────────────────────────┐
              │  TOKEN EMBEDDING         │  Word IDs → dense vectors
              │  + POSITIONAL ENCODING   │  Add position signals
              └─────────────────────────┘
                         ↓
              ┌─────────────────────────┐
              │  ENCODER (×6 layers)    │
              │  Self-Attention          │  Tokens attend to all tokens
              │  Add & Norm             │  Residual + LayerNorm
              │  FFN                    │  Non-linear per-token transform
              │  Add & Norm             │
              └─────────────────────────┘
                    enc_out (K, V)
                         ↓
              ┌─────────────────────────┐
              │  DECODER (×6 layers)    │
              │  Masked Self-Attention   │  Attends to past output only
              │  Add & Norm             │
              │  Cross-Attention         │  Q from decoder, K/V from encoder
              │  Add & Norm             │
              │  FFN                    │
              │  Add & Norm             │
              └─────────────────────────┘
                         ↓
              ┌─────────────────────────┐
              │  LINEAR + SOFTMAX       │  → token probabilities
              └─────────────────────────┘
                         ↓
              "Je", "t'aime", <EOS>
```

---

## Interview Questions

### Conceptual

**Q1. What is the pretraining + fine-tuning paradigm, and why did Transformers enable it?**

> **Answer:** Pretraining: train a Transformer on massive unlabeled data (next-token prediction or MLM), learning general language representations. Fine-tuning: adapt the pretrained model to a specific task (classification, translation, QA) using small labeled datasets. Transformers enabled this because: (1) self-attention learns general, task-agnostic representations — the Q, K, V weights learned on Wikipedia/internet text are useful for any language task; (2) Transformers scale to billions of parameters and trillions of tokens because they're fully parallelizable; (3) learned representations improve monotonically with scale, making larger pretrained models always better starting points. Before Transformers (RNNs), pretraining was much less effective because sequential training limited scale.

**Q2. What is the difference between BERT and GPT architecturally, and what tasks is each suited for?**

> **Answer:** BERT uses only the encoder with bidirectional (unmasked) self-attention — every token sees every other token. Pretrained with Masked LM (predict masked tokens). Best for understanding tasks: classification, NER, QA, semantic similarity — because bidirectional context is needed to understand a token. GPT uses only the decoder with causal (masked) self-attention — each token sees only past tokens. Pretrained with CLM (predict next token). Best for generation tasks: text completion, dialogue, code, translation — because generation is inherently left-to-right. The architectural choice directly encodes the task assumption: BERT assumes you have the full text and need to understand it; GPT assumes you're building the text incrementally.

**Q3. How does LoRA (Low-Rank Adaptation) enable efficient fine-tuning of large models?**

> **Answer:** Full fine-tuning updates all W parameters (billions) — requires enormous GPU memory and compute. LoRA observes that fine-tuning updates are often low-rank: the change in weights ΔW ≈ A·B where A ∈ R^{d×r} and B ∈ R^{r×d} with r << d. Instead of updating W, freeze it and add a trainable ΔW = A·B. For r=8, d=4096: full update needs 4096² = 16.7M params; LoRA update needs 2×4096×8 = 65K params — 256× fewer. Quality is near-identical to full fine-tuning. This makes fine-tuning a 7B model feasible on a single consumer GPU.

**Q4. What is ViT (Vision Transformer), and what insight does it reveal about the Transformer?**

> **Answer:** ViT applies the Transformer encoder to images by dividing them into 16×16 patches, treating each patch as a token. After embedding patches and adding positional encodings, standard self-attention is applied — each patch attends to all other patches. ViT shows that the Transformer architecture is not NLP-specific — it's a **general sequence modeling architecture** applicable to any data that can be represented as a sequence of vectors. Images, audio (Whisper), proteins (AlphaFold2), video, code, multimodal inputs (CLIP) — all can be processed with the same architecture. "Attention Is All You Need" applies beyond language.

**Q5. What does "emergent capabilities" mean in the context of large language models?**

> **Answer:** Emergent capabilities are behaviors that appear suddenly when a model reaches a certain scale threshold, without being explicitly trained for them. Examples: few-shot learning (GPT-3 — solving novel tasks from a few examples), chain-of-thought reasoning (solving multi-step math problems), code generation, translation between unseen language pairs. These don't exist in small models (1B params) and appear at large scale (100B+ params). The Transformer enables emergence because: (1) scale is achievable via parallelism, (2) deep stacking of self-attention + FFN layers creates increasingly abstract representations, (3) cross-layer attention patterns enable compositional reasoning that flat architectures cannot.

---

### Advanced

**Q6. How does the Transformer compare to CNNs and RNNs on the three dimensions of: (1) dependency path length, (2) computational complexity, (3) parallelism?**

> **Answer:**

| Model | Dependency Path Length | Computation per Layer | Parallelism |
|-------|----------------------|----------------------|-------------|
| RNN | O(n) — must traverse n steps | O(n) | Sequential — none |
| CNN | O(log_k n) — with dilated | O(k·n·d) | Across positions |
| Transformer | O(1) — direct attention | O(n²·d) | Full — all positions |

The Transformer trades computational complexity (O(n²) vs O(n)) for O(1) dependency paths and full parallelism. For typical sequence lengths (n≤1024), n²·d is acceptable. For very long sequences (n=100K), the O(n²) bottleneck requires architectural modifications (sparse attention, linear attention).

**Q7. What would a world without positional encodings look like — what tasks would and wouldn't work?**

> **Answer:** Without positional encodings, the Transformer is a **bag-of-words model with attention** — permutation invariant. "The cat chased the mouse" and "The mouse chased the cat" would produce identical representations (same tokens, different order = same attention matrix). Tasks that would still work: sentiment analysis (bag-of-words is sufficient for "great movie" = positive), topic classification, keyword-based retrieval. Tasks that would fail: machine translation (word order critically differs between languages), Q&A (subject/object order matters: "Who bit whom?"), any task requiring syntactic understanding. In practice, models without positional encoding degrade severely on almost all tasks beyond bag-of-words difficulty.

---

## Learning Thoughts

> **The unifying insight:** BERT, GPT, ViT, Whisper, CLIP — these look like completely different models. But they're all `Embeddings + Positional Encoding + [Self-Attention + FFN] × N`. The only differences are: (1) which architecture (encoder, decoder, enc-dec), (2) what modality (text, images, audio), (3) what pretraining task. The Transformer architecture is a universal sequence model.

> **The pretraining insight is the most important idea in modern AI:** Why does ChatGPT know about quantum physics, cooking, history, programming? Not because OpenAI labeled those topics — but because the self-attention mechanism, trained on internet text, learned to capture the statistical patterns of all human knowledge. Pretraining + attention = learning from all of humanity's written output simultaneously.

> **Scale is not magic — it's engineering:** The reason GPT-4 is so much better than GPT-2 is not a new architecture. It's the same Transformer at massively larger scale, trained on orders of magnitude more data. This was only possible because the Transformer's parallel training allows scaling. Understand this: better AI in the last 6 years has come primarily from scale enabled by architecture, not from fundamentally new algorithms.

> **LoRA is the democratization of LLMs:** Before LoRA, fine-tuning required GPUs that only large companies could afford. With LoRA, any developer can fine-tune LLaMA-7B or Mistral-7B on a gaming GPU. Understanding LoRA is increasingly essential for applied ML engineers.

> **The evolution table is your narrative arc:** One-Hot → Word2Vec → ELMo → Self-Attention tells the story of how we progressively solved the "static representation" problem. Each step adds context: Word2Vec adds semantics (but static), ELMo adds context (but sequential), Self-Attention adds full context with no sequential limitation. This narrative is the best answer to "how did we get to LLMs?" in any interview.
