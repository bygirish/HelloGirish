# Part D — Embeddings & Representation Learning

> **Course:** Executive PGP in Generative AI & Agentic AI | IIT Kharagpur × upGrad
> **Instructor:** Prof. Niloy Ganguly, IIT Kharagpur
> **Session:** Module 1 — AI & Deep Learning Essentials

---

## Learning Compass

> Representation is everything. The same data, represented differently, can be trivially easy or impossibly hard to learn from. Embeddings are the technology that transformed how machines understand language, images, and knowledge — and they are the foundation on which all modern AI (including LLMs) is built.

By the end of this section you will be able to:
- Explain why the choice of representation is the most fundamental decision in ML
- Describe the feature extraction bottleneck and why it limits classical ML
- Understand one-hot encoding and its deep limitations
- Explain distributional semantics and why context defines meaning
- Compare sparse vs dense representations
- Explain what embeddings are, how they are learned, and what makes them powerful
- Understand the famous vector algebra of word2vec: king − man + woman = queen

---

## Topic 28 — The Feature Extraction Bottleneck (Recap and Deeper Dive)

### The Central Problem

The full classical ML pipeline is:

```
Raw Data  ──►  Feature Extractor  ──►  Supervised/Unsupervised Model  ──►  Predictions
                 (handcrafted)
```

The **feature extractor** is the step where raw data (images, text, audio) is converted into a numerical vector that the ML model can process.

In classical ML, this step requires **human expertise**:
- Computer vision: facial landmark detection, edge detectors, HOG (Histogram of Oriented Gradients)
- NLP: word counts, TF-IDF, n-gram features
- Audio: Mel-frequency cepstral coefficients (MFCCs)

### Why Hand-crafted Features Fail

**Problem 1 — Domain-specific and non-transferable**
A feature engineering pipeline built for medical imaging cannot be reused for satellite imagery. Every new domain requires a new expert.

**Problem 2 — Incomplete**
Experts don't know what they don't know. Important patterns may exist in raw data that no human would think to encode as a feature.

**Problem 3 — Static**
Hand-crafted features don't adapt. If the domain changes (new types of tumors appear), the feature extractor must be redesigned.

**Problem 4 — Expensive**
Building, validating, and maintaining feature pipelines costs hundreds of engineering hours per domain.

### The Solution: Let the Model Learn its Own Features

The vision of **representation learning** (deep learning):

```
Raw Data  ──►  [Learned Feature Extractor]  ──►  Model  ──►  Predictions
                    (learned jointly with
                     the rest of the model)
```

Key properties of **learned representations (embeddings)**:
1. **Automatic** — no expert feature design
2. **Task-optimized** — features are tuned for what actually matters for the prediction
3. **Transferable** — one representation model works across many tasks
4. **Dense** — compact, no wasted dimensions
5. **Semantic** — similar objects cluster together in the embedding space

---

## Topic 29 — One-Hot Vector Representation

### What is a One-Hot Vector?

Given a vocabulary of V words, each word is represented as a vector of length V where:
- All entries are 0
- Exactly one entry is 1 (the entry at the word's index)

**Example vocabulary (V=8):**

```
Index:    0      1      2      3      4      5      6      7
Word:    cat    dog   king  queen  apple  runs  happy   the
```

**Encodings:**
```
cat   → [1, 0, 0, 0, 0, 0, 0, 0]
king  → [0, 0, 1, 0, 0, 0, 0, 0]
queen → [0, 0, 0, 1, 0, 0, 0, 0]
```

**Key property:** Each word is a point in V-dimensional space, and each point is at the tip of a coordinate axis — orthogonal to all other words.

### Bag-of-Words (BoW) Representation

To represent an entire sentence, sum the one-hot vectors:

```
Sentence: "The cat is happy"
Token:     the    +    cat   +    is   +  happy
Vector:  [0,0,0,0,0,0,0,1] + [1,0,0,0,0,0,0,0] + [0,...] + [0,0,0,0,0,0,1,0]
Result:   [1, 0, 0, 0, 0, 0, 1, 1]  (ignoring "is" for simplicity)
```

This produces a **bag of words** — we know which words appeared, not in what order.

### Using One-Hot for Classification

For sentiment classification ("happy sentence" → +1, "sad sentence" → -1):

```
Positive: "The cat is happy"   → [1, 0, 1, 0, 0, 0, 1, 1] → Linear Classifier → +1
Negative: "The dog runs away"  → [1, 1, 0, 0, 1, 1, 0, 0] → Linear Classifier → -1
```

**What works:**
- Simple and fast
- Interpretable (each weight corresponds to a word's contribution)
- Works when word frequency signals matter (spam: "Nigeria", "prize", "wire transfer")

---

## Topic 30 — Limitations of Sparse Representations

### The Four Fundamental Flaws

**Flaw 1 — No Semantic Meaning**

In one-hot encoding, "king" and "queen" occupy orthogonal axes. The model has no way to know they are related. Every word is equally unrelated to every other word.

```
cat    → [1, 0, 0, 0, 0, ...]
kitten → [0, 0, 0, 0, 1, ...]
Cosine similarity(cat, kitten) = 0.00  ← catastrophic
```

Similarly in TF-IDF: "cat" ≠ "kitten" numerically, even though semantically they are nearly synonymous.

**Flaw 2 — Curse of Dimensionality**

English vocabulary ≈ 100,000 words. Each sentence becomes a 100,000-dimensional vector where 99.9%+ entries are zero.

- Memory: storing millions of 100,000-dim vectors is prohibitive
- Computation: matrix multiplications with mostly-zero inputs are inefficient
- Distance: in high-dimensional sparse space, distances lose meaning

**Flaw 3 — No Word Order**

"Dog bites man" and "Man bites dog" have the same bag-of-words representation. They are completely different sentences but indistinguishable to a BoW model.

```
"Dog bites man" → {dog:1, bites:1, man:1}
"Man bites dog" → {dog:1, bites:1, man:1}
```

This is the **bag-of-words assumption** — words are an unordered set, not a sequence. It loses syntactic and semantic information.

**Flaw 4 — Out-of-Vocabulary (OOV) Words**

Words not in the training vocabulary get assigned a zero vector — carrying no information. New product names, slang, technical terms, foreign words: all invisible to a sparse BoW model.

### Why These Are Fundamental Flaws

These are not engineering problems fixable with better preprocessing. They are **representational limitations** of the sparse paradigm:

1. Sparsity kills similarity
2. Dimensionality kills scalability
3. Order-blindness kills syntax
4. Fixed vocabulary kills generalization

> We need a smarter representation. We need embeddings.

---

## Topic 31 — Why Sparsity Breaks Similarity

### The Mathematics

**Cosine similarity** is the standard metric for text similarity:

```
cosine(u, v) = (u · v) / (||u|| × ||v||)
```

For two one-hot vectors:
- Their dot product = 1 only if they share an index = only if they are the **same word**
- For any two **different** words: dot product = 0 → cosine similarity = 0

```
cosine(cat, kitten) = 0.00
cosine(king, queen) = 0.00
cosine(happy, joyful) = 0.00
```

**The implication:** A model using one-hot features can never discover that "cat" and "kitten" are related, because they are orthogonal by construction.

### The Geometric Insight

One-hot vectors live in V-dimensional space where each word is on a different axis. This space has no geometry that reflects meaning. Two words can be "close" in this space only if they are the same word.

**What we need:** A space where **similar words are geometrically close**. This is what word embeddings provide.

---

## Topic 32 — Distributional Semantics: Words by Context

### The Foundational Idea

> *"You shall know a word by the company it keeps."*
> — J. R. Firth, 1957

This is the **distributional hypothesis**: the meaning of a word can be derived from the contexts in which it appears. Words that appear in similar contexts have similar meanings.

### The Mechanism

**Context window:** For each target word, look at the ±k words surrounding it.

**Example (k=2):**

```
"...government debt problems turning into [banking] crises as happened..."
                                            ▲
                                         target

Context words: "government", "debt", "problems", "crises", "as"
```

```
"...saying that Europe needs unified [banking] regulation to replace..."
Context words: "unified", "regulation", "Europe", "needs", "replace"
```

```
"...India has just given its [banking] system a shot in the arm..."
Context words: "India", "system", "shot", "arm", "its"
```

**Observation:** The word "banking" consistently co-occurs with financial and regulatory terms. These co-occurrence patterns **are** the meaning of "banking."

### Building a Representation from Context

The process:

1. For each word in the vocabulary, count how often each other word appears within its context window across a large corpus
2. Build a **co-occurrence matrix**: rows = target words, columns = context words, cells = co-occurrence count
3. This raw count matrix is still sparse and high-dimensional
4. **Compress** it (via SVD or neural methods) to a dense, low-dimensional vector

**Key shift:**
> Instead of encoding a word's position in a fixed vocabulary (one-hot), we encode **what words it co-occurs with**.

Count co-occurrences → compress → dense vector. This is the road to word embeddings.

---

## Topic 33 — Dense Representations: Distributed Semantics

### Sparse vs Dense: The Comparison

**Sparse (One-Hot):**
```
"king" → [0, 0, 1, 0, 0, 0, 0, ..., 0]   (50,000 dims, one 1)
Problems: 99%+ zeros, no similarity, huge memory
```

**Dense (Word Embedding):**
```
"king" → [0.32, -0.71, 0.15, 0.88, -0.43, 0.07, 0.61, ...]  (300 dims)
Properties: all values active, geometry = meaning, compact and fast
```

### What Makes Dense Vectors Powerful?

**1. Every dimension carries information**

In a 300-dimensional word embedding, every one of the 300 values contributes to representing the word's meaning. There are no wasted dimensions.

**2. Geometry encodes meaning — the famous examples**

```python
vec("king") - vec("man") + vec("woman") ≈ vec("queen")
```

**Interpretation:** The vector from "man" to "king" (the "royalty" direction) is the same as the vector from "woman" to "queen." The embedding space has learned to encode the concept of royalty as a **direction in vector space**.

Other examples:
```python
vec("Paris") - vec("France") + vec("Germany") ≈ vec("Berlin")
vec("walked") - vec("walking") + vec("swimming") ≈ vec("swam")
```

The embedding space has learned:
- Capital city relationships
- Verb tense
- Gender relationships
- Country–language relationships
- ...all without being explicitly taught these concepts

**3. Cosine similarity captures semantic relatedness**

```python
cosine(vec("cat"), vec("kitten")) ≈ 0.92  ← near-synonyms cluster
cosine(vec("cat"), vec("dog"))    ≈ 0.76  ← related (both pets)
cosine(vec("cat"), vec("finance"))≈ 0.02  ← unrelated
```

### Why Does Geometry Encode Meaning?

Because word meaning IS defined by distributional context. Words that appear in similar contexts have similar contexts vectors, and similar context vectors end up as similar points in embedding space. The embedding space is a learned compression of the co-occurrence statistics of language.

---

## Topic 34 — Feature Learning through Embeddings

### What is an Embedding?

Formally, an embedding is a **dense, fixed-length vector** that represents an object (word, sentence, image, user, product) in a continuous vector space such that **similar objects are geometrically close**.

```
Objects  ──►  Embedding Model  ──►  Dense Vectors
  (any raw object:                  (fixed-length, e.g., 300 dims)
   word, image, user ID)
```

### How the Embedding Model is Trained

Embedding models are neural networks trained on a task that forces the network to learn useful representations:

- **Word2Vec:** Predict the target word from its context words (CBOW), or predict context words from the target (Skip-gram)
- **BERT:** Predict masked words in a sentence (masked language modeling)
- **ResNet:** Predict image class (ImageNet classification)
- **Sentence-BERT:** Predict whether two sentences are semantically similar

The embedding is the **hidden representation** learned by the network — the intermediate vector before the final classification layer.

### Three Key Properties

**Automatic:** The model learns which features matter directly from raw data. No human labeling of features required. The same architecture that learns "king-queen" analogies also learns medical synonyms, code function similarity, product associations.

**Transferable:** A model trained on one task learns general representations usable in other tasks. A BERT model trained on general English can be fine-tuned for:
- Medical NLP
- Legal document classification
- Sentiment analysis
- Question answering
...with very little task-specific data. This is **transfer learning**.

**Dense and Scalable:** 300-dimensional vectors vs 100,000-dimensional sparse vectors.
- 333× compression for vocabulary of 100K
- All distances meaningful
- Efficient matrix operations

---

## Topic 35 — The Embedding Geometry: king − man + woman = queen

### The Visual Intuition

The word2vec embedding space organizes words such that meaningful **semantic relationships correspond to geometric directions**:

**Gender relationship:**
```
         man
          │
     ─────┼──────────────► [gender direction]
          │
        woman

    king ─────────────────► [gender direction]
      │
      ▼
    queen
```

`king - man = queen - woman` means the "royalty" direction is the same regardless of gender.

**Tense relationship:**
```
walking ─────► walked   (past tense offset)
swimming ────► swam     (same offset applied)
```

**Country-Capital relationship:**
```
France ──► Paris
Germany ─► Berlin
Japan ───► Tokyo
```

All these arrows point in the **same direction** in embedding space — the capital-of direction.

### Why This Emerges from Training

These geometric relationships were **not explicitly programmed**. They emerged from training on billions of words of text. The model discovered them because:

1. "King" and "queen" appear in the same types of contexts (royal, political, medieval)
2. "King" and "man" share masculine contexts; "queen" and "woman" share feminine contexts
3. The optimization process finds the embedding space that best compresses these co-occurrence patterns

The analogy arithmetic works because **linear directions in embedding space encode semantic dimensions**.

### From Word Embeddings to Sentence and Document Embeddings

| Level | Method | Example Models |
|---|---|---|
| Word | Word2Vec, GloVe, FastText | word2vec, GloVe-6B |
| Subword | BPE, WordPiece | BERT, GPT tokenizers |
| Sentence | Mean pooling, [CLS] token, contrastive | Sentence-BERT, E5 |
| Document | Hierarchical, chunk + pool | Longformer embeddings |
| Image | CNN feature maps | ResNet, ViT |
| Multimodal | Joint text-image embedding | CLIP |

### The Road to Deep Learning

Embeddings are learned by neural networks. The embedding model is itself a **deep neural network**. Understanding embeddings naturally leads to the question: **how do deep neural networks learn these representations?**

This is what Part E (Deep Learning) addresses.

> **Embeddings are the bridge between symbolic AI's brittle representations and deep learning's flexible, data-driven intelligence.**

---

## Interview Questions — Part D

**Q1: What is the distributional hypothesis and how does it motivate word embeddings?**

> The distributional hypothesis (Firth, 1957) states: "You shall know a word by the company it keeps." Words that appear in similar contexts have similar meanings. Word embeddings operationalize this by representing each word as a dense vector derived from its co-occurrence patterns in a large corpus. Words with similar contexts end up as similar vectors — enabling the model to capture semantic relationships that one-hot encoding completely misses.

**Q2: Why does cosine similarity between one-hot vectors always equal zero?**

> One-hot vectors are **orthogonal by construction** — each word occupies a unique axis with a 1 in exactly one position. For any two different words, their dot product is 0 (no shared non-zero position), so cosine similarity = 0. This means one-hot encoding cannot represent semantic similarity: "cat" and "kitten" are as "different" as "cat" and "quantum physics." Word embeddings fix this by placing semantically similar words near each other in the vector space.

**Q3: Explain the four fundamental limitations of sparse representations.**

> (1) **No semantic meaning** — every word is orthogonal, capturing no similarity; (2) **Curse of dimensionality** — 100K-dim sparse vectors are memory-intensive and computationally expensive; (3) **No word order** — bag-of-words loses syntax ("dog bites man" = "man bites dog"); (4) **Out-of-vocabulary** — new words get zero vectors, carrying no information. Together, these make sparse representations fundamentally unsuitable for tasks requiring semantic understanding.

**Q4: What is the difference between one-hot encoding and word embeddings, and when would you use each?**

> **One-hot:** Binary, sparse, V-dimensional (V = vocabulary size). Every word is equidistant from every other. Use for: simple frequency-based models where semantic similarity doesn't matter (e.g., a spam filter that just needs to know "prize" appeared). **Word embeddings:** Dense, low-dimensional (typically 50-300 dims), semantically structured. Use for: any task requiring language understanding — sentiment analysis, similarity search, machine translation, question answering, RAG systems.

**Q5: Explain the geometric interpretation of king − man + woman ≈ queen.**

> In word embedding space, semantic relationships correspond to geometric directions. The vector from "man" to "king" encodes the concept of "royalty for males." Adding this same vector to "woman" produces "woman + royalty = queen." More precisely: the embedding space has learned that `king - man = queen - woman` — the offset vector that encodes the male-to-female gender relationship is consistent across the royalty axis. This emerges from training, not programming — the model discovered the gender dimension from statistical co-occurrence patterns in text.

**Q6: What does "transferable" mean in the context of embeddings, and why is it economically important?**

> A transferable embedding is one trained on one task (e.g., general language modeling) that generalizes to other tasks without requiring full retraining. BERT, trained on Wikipedia and BookCorpus, can be **fine-tuned** on medical records, legal contracts, or social media with just a small task-specific dataset. Economically, this is transformative: instead of training a model from scratch for each domain (expensive: millions of $ and months of compute), you fine-tune a pre-trained model in hours for thousands of $. Transfer learning made state-of-the-art NLP accessible to organizations without massive compute budgets.

**Q7: How are word embeddings trained? Explain Word2Vec's key idea.**

> Word2Vec (Mikolov et al., 2013) trains a shallow neural network on a self-supervised task: either **CBOW** (predict the target word given surrounding context words) or **Skip-gram** (predict context words given the target word). The task forces the network to encode words in a way that allows predicting their contexts. The learned weights of the network's hidden layer **are** the word embeddings. The insight: training to predict context forces the model to encode meaning in the representation — because words with similar meanings appear in similar contexts.

**Q8: Why are embeddings "the only scalable way forward" in ML?**

> Three reasons: (1) **Data scale** — the internet has petabytes of text but almost none is labeled. Embeddings can be learned from unlabeled data via self-supervised objectives; (2) **Domain generality** — one embedding model trained on general data transfers across domains; hand-crafted features require a new expert per domain; (3) **Representation quality** — learned embeddings capture complex nonlinear patterns that no human-designed feature could express. The combination makes embeddings the foundation of modern ML: LLMs, image models, search engines, recommendation systems — all rely on learned embeddings.

---

## Key Learning Thoughts — Part D

> **Thought 1:** Representation is not a preprocessing step — it IS the model's understanding. The choice of representation determines what the model can and cannot learn. A great algorithm with a bad representation will always lose to a mediocre algorithm with a great representation.

> **Thought 2:** The king−man+woman=queen example is not a party trick. It is evidence that embedding spaces encode **conceptual structure** — that mathematical operations on vectors correspond to logical operations on meanings. This is a profound and surprising result.

> **Thought 3:** The distributional hypothesis is the secret ingredient of modern AI. All LLMs (GPT, Claude, Gemini) are fundamentally built on the insight that meaning is co-occurrence. They are giant, sophisticated implementations of Firth's 1957 observation.

> **Thought 4:** The bottleneck shifted, not disappeared. Classical ML had a feature engineering bottleneck (expert time). Modern ML has a data labeling / RLHF bottleneck (human feedback at scale). Understanding the bottleneck in each era helps you understand where the work is.

> **Thought 5:** Cosine similarity in embedding space is the most important distance metric in modern AI. RAG (Retrieval-Augmented Generation) systems, semantic search, recommendation engines — all are "find the vectors with highest cosine similarity to the query." Understanding why cosine similarity works for embeddings is foundational knowledge for building AI systems.

> **Thought 6:** Embeddings are not just for text. Image embeddings (ResNet, ViT), protein embeddings (AlphaFold), molecular embeddings (drug discovery), user embeddings (recommender systems) — the same principle applies everywhere. Any object that can be characterized by its relationships to other objects can be embedded.

---

*Previous: [Part C — Machine Learning](Part-C_Machine_Learning.md)*
*Next: [Part E — Deep Learning](Part-E_Deep_Learning.md)*
