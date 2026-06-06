# Block 6: Road to Transformers & Embeddings
## NLP Evolution, One-Hot, Distributional Semantics, Word2Vec, ELMo, Attention

> **Session:** Lecture 2 — The Transformer Architecture  
> **Topics covered:** 29–39

---

## Learning Roadmap for This Block

```
Why RNNs are not enough
→ The full evolution of NLP (Statistical → Deep → Transformers)
→ How text becomes numbers: One-Hot Encoding
→ Why one-hot fails (sparsity, no semantics)
→ The big idea: Distributional Semantics
→ Dense Embeddings (Word2Vec, GloVe)
→ Static embeddings can't handle polysemy
→ Contextual embeddings: ELMo (BiLSTM)
→ The Transformer: Attention-based solution to all problems
```

This block answers: **How has the AI community gone from counting word frequencies to building ChatGPT?**

---

## Topic 29: Evolution of Language Processing

### The Full Timeline

```
1950s-2000s    Statistical NLP
               ├── N-gram language models (count co-occurrences)
               ├── Hidden Markov Models (HMM)
               ├── Conditional Random Fields (CRF)
               └── TF-IDF for document retrieval

2000s-2015     Deep NLP
               ├── Word2Vec (2013) — dense word embeddings
               ├── CNN for text (2014) — local n-gram features
               ├── RNN / LSTM (2013-2015) — sequential processing
               └── GRU (2014) — simplified LSTM

2015-2017      Attention Mechanism
               ├── Bahdanau Attention (2015) — for machine translation
               └── Pointer Networks, Copy Mechanisms

2017-2018      Transformer Era Begins
               ├── "Attention Is All You Need" (Vaswani et al., 2017)
               ├── BERT (2018) — bidirectional encoder
               └── GPT (2018) — autoregressive decoder

2019-2022      Foundation Models
               ├── GPT-2 (2019), GPT-3 (2020) — scaling laws
               ├── T5, RoBERTa, ALBERT — BERT variants
               └── Llama, Falcon, Mistral — open-source LLMs

2022-Present   Agentic AI
               ├── ChatGPT, Claude, Gemini
               └── GPT-4, Claude 3, Gemini Ultra
```

The central paradigm shift: **Pre-train once on massive data → Fine-tune for many downstream tasks**.

---

## Topic 30: Limitations of RNN Models

Before moving to embeddings, understand WHY we need something beyond RNNs:

### 1. Slow Sequential Computation

```
Time step: t₁ → t₂ → t₃ → ... → tₙ
```

Each step depends on the previous one — **no parallelism**. For a sentence of 100 words, you need 100 sequential steps. Training GPT-3 on RNNs would have been impossibly slow.

**Transformer fix:** Self-attention processes ALL positions simultaneously.

### 2. Information Bottleneck

In an encoder-decoder RNN (for translation), the entire source sentence is compressed into a single fixed-size hidden state vector. For long sentences, this vector can't hold everything.

**Transformer fix:** Attention allows the decoder to look directly at all encoder states.

### 3. Vanishing Gradient / Limited Context

Even with LSTM, dependencies beyond ~100 tokens are hard to capture reliably.

**Transformer fix:** Attention directly connects any two positions with O(1) operations.

### 4. No Direct Long-Range Dependency Modeling

RNNs must "carry" information step-by-step. If a word at position 1 is needed at position 500, it must survive 499 updates without being overwritten.

**Transformer fix:** Attention directly computes relationships between any two positions.

---

## Topic 31: Feature Learning Through Embeddings — The Need

### The Representation Problem

Any ML model needs numbers as input. Text is symbols. The critical question:

> **How do you convert a word into a number (or vector) in a way that preserves meaning?**

The journey:
```
Raw Text → ???  → ML Model → Prediction
```

The `???` is the **representation step** — and it's the most important design choice in NLP.

---

## Topic 32: One-Hot Vector Representation

### How It Works

1. Build a vocabulary of all unique words
2. Assign each word a unique integer index
3. Represent each word as a vector of length |V| with a 1 at its index, 0 everywhere else

**Example with vocabulary of 8 words:**

```
Index: 0=cat, 1=dog, 2=king, 3=queen, 4=apple, 5=runs, 6=happy, 7=the

cat   = [1, 0, 0, 0, 0, 0, 0, 0]
dog   = [0, 1, 0, 0, 0, 0, 0, 0]
king  = [0, 0, 1, 0, 0, 0, 0, 0]
queen = [0, 0, 0, 1, 0, 0, 0, 0]
```

### Python Code

```python
from sklearn.preprocessing import OneHotEncoder
import numpy as np

vocab = ["king", "queen", "man", "woman", "cat"]
encoder = OneHotEncoder(sparse_output=False)
vectors = encoder.fit_transform([[w] for w in vocab])

# king   = [1, 0, 0, 0, 0]
# queen  = [0, 1, 0, 0, 0]
# man    = [0, 0, 1, 0, 0]

# Similarity check:
cos = lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
print(cos(vectors[0], vectors[1]))  # king vs queen → 0.0 !!
```

### What One-Hot Gets Right

- Simple to implement
- No training required
- Works for word frequency signals (Bag-of-Words)
- Interpretable (each dimension = one word)

---

## Topic 33: Limitations of Sparse (One-Hot) Representations

### 4 Fundamental Flaws

**Flaw 1: No Semantic Meaning**
```
dot(cat, kitten) = 0   ← synonyms treated as unrelated
dot(king, queen) = 0   ← related words treated as unrelated
dot(cat, dog)    = 0   ← same category, zero similarity
```
One-hot vectors are **orthogonal by construction**. The model can NEVER discover that "cat" and "kitten" are semantically related — their vectors are perpendicular.

**Flaw 2: Curse of Dimensionality**
```
Vocabulary size |V| = 50,000 words (a modest vocabulary)
One-hot vector dimension = 50,000
A document with 100 words = 50,000 × 100 = 5M sparse entries
99.998% of values are 0
```
The embedding matrix (|V| × |V|) would require 50,000² ÷ 8 bytes ≈ **23 GB** just for storage.

**Flaw 3: No Word Order**
Bag-of-Words representations (using one-hot) ignore word order:
```
"Dog bites man" and "Man bites dog" → IDENTICAL representation
```
Order encodes meaning, but one-hot has no mechanism to capture it.

**Flaw 4: Out-of-Vocabulary (OOV) Problem**
Words not in the training vocabulary have **no representation** — the model is completely blind to them. In production, new proper nouns, slang, technical terms constantly appear.

### The Verdict

One-hot encoding is useful as a baseline or for low-cardinality categorical features. For natural language (large vocabularies with rich semantics), it fails fundamentally. We need something smarter.

---

## Topic 34: Distributional Semantics — The Big Idea

### The Hypothesis

> "You shall know a word by the company it keeps."  
> — J.R. Firth (1957)

This is the **Distributional Hypothesis** — one of the most powerful ideas in computational linguistics:

> **A word's meaning is defined by the contexts (surrounding words) in which it appears.**

### The Intuition

Consider the word "banking":
```
Context 1: "...government debt problems turning into banking crises as happened in 2009..."
Context 2: "...Europe needs unified banking regulation to replace the hodgepodge..."
Context 3: "...India has given its banking system a shot in the arm..."
```

"Banking" consistently appears with "regulation," "crises," "system," "debt" — these are its typical context words. Any word with similar context words should have similar meaning.

### Context Windows

Given a target word, look at the k words on either side:

```
Sentence: "The quick brown fox jumps over the lazy dog"

Target: "fox", context window k=2:
Context words: "quick", "brown", "jumps", "over"

These 4 words DEFINE the meaning of "fox" for our model.
```

### From Co-occurrence to Vectors

Build a **co-occurrence matrix**: for every pair of words (w₁, w₂), count how often w₂ appears within k words of w₁. This matrix encodes the distributional profile of every word. Then compress (via PCA/SVD) to get dense vectors.

---

## Topic 35: Dense Representations — Word Embeddings

### The Key Shift

```
SPARSE (One-Hot, 50,000 dims):
[0, 0, 1, 0, 0, ..., 0, 0]   ← 99.998% zeros, no meaning

DENSE (Word Embedding, 300 dims):
[0.32, -0.71, 0.15, 0.88, -0.43, 0.07, 0.61, ...]   ← every value carries meaning
```

### The Magic of Embedding Geometry

Dense embeddings are learned so that **geometric relationships encode semantic relationships**:

```
vec("king") − vec("man") + vec("woman") ≈ vec("queen")
vec("Paris") − vec("France") + vec("Germany") ≈ vec("Berlin")
cosine( vec("cat"), vec("kitten") ) ≈ 0.92    ← synonyms close together
cosine( vec("cat"), vec("dog") )    ≈ 0.78    ← same category
cosine( vec("cat"), vec("democracy")) ≈ 0.05  ← unrelated far apart
```

This is profoundly different from one-hot: the geometry of the vector space now reflects the geometry of meaning.

### Why Dense?

| Property | One-Hot | Dense Embedding |
|---|---|---|
| Dimension | |V| (50,000+) | 100–1000 (tunable) |
| Sparsity | 99.99% zeros | No zeros |
| Similarity | All pairs = 0 | Meaningful cosine similarity |
| Analogies | Impossible | Works (king−man+woman≈queen) |
| Memory | Huge | Compact |
| Learned? | No | Yes (end-to-end or pre-trained) |

---

## Topic 36: Embedding Lookup Tables in PyTorch

### How Embeddings Are Implemented

An embedding is simply a **learnable lookup table** — a matrix of shape (|V|, d) where d is the embedding dimension. Given a word index, you look up the corresponding row.

```python
import torch
import torch.nn as nn

VOCAB_SIZE = 50_000
EMBED_DIM  = 300      # was 50K dimensions (one-hot), now just 300!

# This is a learnable weight matrix: shape (50000, 300)
embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)

# Forward pass: integer indices → dense vectors
token_ids = torch.tensor([42, 1337, 7])        # word indices for "king", "queen", "man"
vectors   = embedding(token_ids)               # shape: (3, 300)

# During training, gradients flow back and update the embedding weights
# Similar words end up with similar vectors — AUTOMATICALLY!
```

### What Changed?

```
50,000 dimensions  →  300 dimensions    ✓ 167× smaller
All zeros          →  Dense floats      ✓ Information-rich
Fixed              →  Learned           ✓ Task-aware, updatable
```

### How Are Embeddings Learned?

Two main approaches:

**1. As part of training:** The embedding layer is randomly initialized and updated via backpropagation alongside the rest of the model. Similar words get similar embeddings because they appear in similar contexts and thus receive similar gradient updates.

**2. Pre-trained (transfer learning):** Train embeddings on a massive corpus (billions of words), then use them as a starting point for downstream tasks. This is the approach of Word2Vec, GloVe, fastText, and (contextually) BERT.

---

## Topic 37: Word2Vec — Skip-gram & CBOW

### The Word2Vec Insight (Mikolov et al., 2013)

Rather than building a co-occurrence matrix and compressing it (computationally expensive), directly train a neural network to **predict context from a center word** (or vice versa). The hidden layer weights become the embeddings.

### Two Training Strategies

**Skip-gram:**
```
Input: center word → Predict: surrounding context words

"The [cat] sat on the mat"
     ↓
Predict: "The", "sat", "on"

Better for rare words (explicitly generates context for each center word)
```

**CBOW (Continuous Bag of Words):**
```
Input: context words → Predict: center word

"The [?] sat on the mat"
              ↑
Context: "The", "sat", "on"

Faster, better for frequent words
```

### The Training Objective (Skip-gram)

Maximize the probability of seeing context words given the center word:

```
Objective: maximize Σ Σ log P(context_word | center_word)
                   sentence word
```

Using Negative Sampling for efficiency: instead of computing softmax over all 50,000 words, sample a few "negative" (wrong) context words and train to distinguish correct context from noise.

### Gensim Code

```python
from gensim.models import Word2Vec

sentences = [
    ["king", "rules", "the", "kingdom"],
    ["queen", "rules", "the", "kingdom"],
    ["man", "is", "a", "human"],
    ["woman", "is", "a", "human"]
]

model = Word2Vec(
    sentences,
    vector_size=100,  # embedding dimensions
    window=3,         # context window size (±3 words)
    min_count=1,      # minimum word frequency
    sg=1              # sg=1: Skip-gram, sg=0: CBOW
)

# The famous analogy test:
result = model.wv.most_similar(
    positive=["king", "woman"],
    negative=["man"]
)
print(result[0])   # ('queen', 0.9991...)
```

### GloVe — An Alternative

GloVe (Pennington et al., 2014) trains on the global co-occurrence matrix directly:

```
Loss = Σᵢⱼ f(Xᵢⱼ) × (vᵢ · ṽⱼ + bᵢ + b̃ⱼ − log Xᵢⱼ)²
```

where `Xᵢⱼ` is the co-occurrence count between words i and j. GloVe is competitive with Word2Vec and often preferred for its mathematical interpretability.

---

## Topic 38: Static Embeddings & the Polysemy Problem

### The Critical Limitation of Word2Vec

Word2Vec assigns **one single vector per word** — regardless of context.

```python
# Word2Vec gives ONE vector per word, no matter how it's used:
bank_vector = model.wv["bank"]   # one vector

# But "bank" has completely different meanings:
# "I deposited money at the bank"   → financial institution
# "She sat on the river bank"       → geographical feature
# "The plane had to bank left"      → aeronautical maneuver

# Word2Vec collapses ALL meanings into ONE point in vector space!
```

This is the **polysemy problem** — a word with multiple meanings gets a single averaged vector that may represent none of its meanings well.

### Real-World Impact

The embedding for "bank" ends up somewhere between "financial institution" and "river bank" — a vector that doesn't cleanly represent either meaning. Downstream models that use this embedding receive confused input.

### Other Static Embedding Limitations

**No morphology:** "run," "running," "ran," "runner" each get separate embeddings, even though they're clearly related. (FastText addresses this by embedding character n-grams.)

**No context:** "hot" in "hot dog" and "hot weather" get the same vector.

**Domain-specific:** Word2Vec trained on Wikipedia poorly represents medical, legal, or scientific vocabulary.

---

## Topic 39: ELMo & Contextual Embeddings

### ELMo: Embeddings from Language Models (Peters et al., 2018)

**The key insight:** Instead of one vector per word, compute a **different vector for each occurrence** of a word based on its surrounding context.

```python
# pip install allennlp
from allennlp.commands.elmo import ElmoEmbedder

elmo = ElmoEmbedder()

sent1 = ["I", "deposited", "money", "at", "the", "bank"]
sent2 = ["She", "sat", "on", "the", "river", "bank"]

emb1 = elmo.embed_sentence(sent1)  # shape: (3, 6, 1024)
emb2 = elmo.embed_sentence(sent2)  # shape: (3, 6, 1024)

# "bank" in financial context vs. river context:
bank_finance = emb1[2][5]   # layer 2, position 5
bank_river   = emb2[2][5]   # layer 2, position 5

import numpy as np
from numpy.linalg import norm
sim = np.dot(bank_finance, bank_river) / (norm(bank_finance) * norm(bank_river))
print(f"Similarity: {sim:.3f}")   # ~0.54 — different! (vs 1.0 for Word2Vec)
```

### How ELMo Works

ELMo uses a **bidirectional LSTM (BiLSTM)** language model trained on a large corpus:

```
Forward LSTM:  I → went → to → the → bank
               →    →     →    →      →
Backward LSTM: I ← went ← to ← the ← bank

For each word:
  ELMo representation = weighted combination of:
    - Character-level embeddings (captures morphology)
    - Forward LSTM hidden states
    - Backward LSTM hidden states
```

By combining both directions, "bank" in the financial context gets influence from "deposited" and "money" (forward) and sentence end (backward) — different from the river context which has "river" nearby.

### ELMo's Three-Layer Architecture

ELMo produces embeddings from 3 layers:
- Layer 0: Character-level representation (morphology)
- Layer 1: Syntax-level representation (grammar, POS)
- Layer 2: Semantic-level representation (meaning)

Different downstream tasks use different layers: syntax tasks prefer layer 1, semantic tasks prefer layer 2.

### BERT: The Next Step (2018)

ELMo uses LSTMs; BERT (Bidirectional Encoder Representations from Transformers) uses the **Transformer** architecture instead:

```
ELMo:  BiLSTM-based contextual embeddings (limited by LSTM's sequential nature)
BERT:  Transformer-based contextual embeddings (fully parallel, better long-range context)
```

BERT pre-trains on Masked Language Modeling (predict randomly masked words) and Next Sentence Prediction, then fine-tunes on downstream tasks. It set state-of-the-art on 11 NLP benchmarks upon release.

### The Path from Static to Contextual

```
Static embeddings:
  Word2Vec (2013), GloVe (2014), fastText (2016)
  → One vector per word → can't handle polysemy

Contextual embeddings (LSTM-based):
  ELMo (2018)
  → Different vector per occurrence → solves polysemy
  → But still sequential (slow)

Contextual embeddings (Transformer-based):
  BERT (2018), GPT (2018) → fully parallel, better context
  → State of the art in NLP
  → Used as base for ChatGPT, Claude, Gemini
```

### The Transformer: Attention Is All You Need

The lecture concludes by pointing to the **Attention-based Transformer** as the culmination of this journey. The Transformer:

1. Replaces RNN sequential processing with **parallel self-attention**
2. Allows each word to directly attend to all other words in the sequence
3. Scales to billions of parameters on billions of tokens
4. Achieves superior performance on ALL NLP benchmarks

This is covered in detail in the next lecture (Module 2: Advanced Prompting). The key formula:

```
Attention(Q, K, V) = Softmax(QKᵀ / √d_k) × V
```

Where:
- Q (Query), K (Key), V (Value) are learned linear projections of the input
- The dot product QKᵀ measures how much each word should "attend to" every other word
- √d_k is a scaling factor to prevent vanishing gradients

---

## Interview Questions — Block 6

**Q1: What is the distributional hypothesis and why is it fundamental to NLP?**

> The distributional hypothesis (Firth, 1957) states that words appearing in similar contexts have similar meanings. This is fundamental because it provides a computational, corpus-based way to define word meaning without human annotation — you just need a large corpus and the ability to count co-occurrences. It's the theoretical foundation behind Word2Vec, GloVe, BERT, and all modern embedding approaches.

**Q2: Why do we use 300-dimensional embeddings instead of one-hot vectors?**

> Three reasons: (1) Dimensionality — one-hot for a 50,000-word vocabulary is 50,000-dimensional, embedding reduces this to 300 (167× smaller). (2) Semantics — one-hot vectors are orthogonal, so cosine similarity between any two is 0; embedding geometry captures meaning (similar words are nearby). (3) Learned representation — embeddings are trained to capture the statistical structure of language, encoding analogies, synonymy, and semantic relationships. The tradeoff is a training cost and the need for sufficient data to learn good representations.

**Q3: What is the difference between Word2Vec Skip-gram and CBOW?**

> Skip-gram takes a center word and predicts surrounding context words. CBOW takes surrounding context words and predicts the center word. Skip-gram treats each (center, context) pair independently, giving it more training examples for each word — better for rare words. CBOW averages context word representations before predicting, making it faster but less precise for rare words. In practice, Skip-gram with negative sampling is more widely used.

**Q4: What is the polysemy problem and how does ELMo solve it?**

> Polysemy means a word has multiple meanings (e.g., "bank" = financial institution or river bank). Static embeddings like Word2Vec assign ONE vector per word regardless of context — the vector for "bank" is a blend of all its meanings, representing none well. ELMo solves this by using a bidirectional LSTM to compute a different vector for each occurrence of a word, conditioned on the surrounding context. "bank" in "river bank" gets a different vector than "bank" in "savings bank."

**Q5: How does BERT improve on ELMo?**

> Both produce contextual embeddings, but BERT uses the Transformer architecture instead of LSTM. Key advantages: (1) Fully parallel — BERT processes all positions simultaneously vs. LSTM's sequential processing; (2) Deeper bidirectionality — BERT's self-attention sees the full context in all layers, not just two directional LSTMs combined at the end; (3) Scale — Transformers scale better to more parameters and data; (4) Pre-training objectives — BERT's masked language modeling and next sentence prediction provide richer supervision signals.

**Q6: Explain the progression from statistical NLP to foundation models.**

> (1) Statistical NLP (1990s-2000s): Hand-crafted features + probabilistic models (HMMs, CRFs, n-grams). Interpretable but limited by feature engineering. (2) Deep NLP (2013-2017): Learned features via CNNs/RNNs + static embeddings (Word2Vec). Better representations but still task-specific models. (3) Transformers (2017-2018): Self-attention replaces sequential processing. BERT/GPT emerge. (4) Foundation Models (2019+): Pre-train once on massive data → fine-tune or prompt for any task. GPT-3, ChatGPT, Claude. The key shift at each stage: from hand-crafted → learned features, from task-specific → transfer learning, from task-specific → few-shot/zero-shot generalization.

**Q7: What is negative sampling in Word2Vec and why is it needed?**

> Training Word2Vec naively requires computing a Softmax over all |V| words (50,000+) for each training example — prohibitively expensive. Negative sampling replaces this: for each (center, true_context) positive pair, sample k random "negative" words that are NOT in the context. Train a binary classifier to distinguish true context words from random noise. This reduces computation from O(|V|) to O(k) per training step (k is typically 5-20). The theoretical justification: negative sampling approximates the full softmax by sampling from a noise distribution.

**Q8: What does the Transformer's attention formula compute?**

> `Attention(Q, K, V) = Softmax(QKᵀ / √d_k) × V`. The Query (Q) represents "what am I looking for?" The Key (K) represents "what do I have to offer?" The dot product QKᵀ computes similarity scores between every Query-Key pair — how much each position should attend to every other position. Dividing by √d_k prevents vanishing gradients when d_k is large (dot products can get very large). Softmax converts scores to a probability distribution (attention weights). Multiplying by V computes a weighted average of Values — the actual information retrieved.

---

## Key Learning Insights

> **Insight 1:** The evolution of NLP representation is a story of moving from **explicit, human-designed features to learned, data-driven representations**. Every step (one-hot → Word2Vec → ELMo → BERT) reduces the need for human knowledge while increasing model performance.

> **Insight 2:** Word2Vec's famous analogy (`king − man + woman ≈ queen`) isn't just a party trick — it reveals that the embedding space has structured, linear representations of semantic relationships. This means arithmetic in embedding space corresponds to reasoning about meaning.

> **Insight 3:** The ELMo paper introduced the concept of **probing** — testing which layers of a deep model capture which linguistic properties (syntax vs. semantics). This became a fundamental technique for understanding what neural NLP models learn.

> **Insight 4:** BERT changed the paradigm from "train a model per task" to "pre-train once, fine-tune for any task." The fine-tuning cost is typically minutes vs. days for pre-training. This made powerful NLP accessible to practitioners without billion-dollar compute budgets.

> **Insight 5:** From the Q&A session: embeddings don't need to be built from scratch — pre-trained embeddings (Word2Vec, GloVe, BERT, APIs) are readily available and fine-tunable. In production NLP, you almost never train embeddings from scratch unless you have domain-specific vocabulary not covered by public embeddings.

> **Insight 6:** The Transformer architecture's success isn't just about attention — it's about **scalability**. The fact that all positions process in parallel means you can throw more GPU/TPU hardware at training, and the model keeps improving. This scalability property, plus the attention mechanism, is what enabled GPT-3, GPT-4, Claude, Gemini.

> **Insight 7:** We are living through the shift from "fine-tuning paradigm" (BERT era: pre-train + fine-tune per task) to "prompting paradigm" (GPT-3+ era: pre-train once + prompt for any task). Module 2 of your course covers this — Prompt Engineering is now a core skill.

---

## Quick Reference Cheatsheet

```
NLP Representation Evolution:
  One-hot → Static Embeddings → Contextual Embeddings → Full Pre-training

One-hot problems:
  No semantics (cos sim = 0), sparse (|V| dims), no order, OOV

Distributional Hypothesis: "A word is known by the company it keeps"

Word2Vec:
  Skip-gram: center → context   (rare words better)
  CBOW:      context → center   (frequent words, faster)
  Negative Sampling: approximates full softmax

Famous analogy: king − man + woman ≈ queen

Polysemy problem: 1 vector per word can't capture multiple meanings

ELMo: BiLSTM → different vector per occurrence (contextual)
BERT: Transformer → fully parallel contextual embeddings

Transformer attention:
  Attention(Q,K,V) = Softmax(QKᵀ/√d_k) × V

RNN Limitations (why Transformer was needed):
  1. Sequential → can't parallelize
  2. Fixed-size hidden state → bottleneck
  3. Vanishing gradients → limited long-range context
  4. Slow training on long sequences
```
