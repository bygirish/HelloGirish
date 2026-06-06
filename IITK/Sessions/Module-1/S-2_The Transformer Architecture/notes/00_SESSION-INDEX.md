# Session 2: The Transformer Architecture — Study Index

> **Instructor:** Prof. Niloy Ganguly, IIT Kharagpur  
> **Course:** Executive PGP in Generative AI & Agentic AI (upGrad × IITK)  
> **PPT:** Upgrad Slides Lecture 2 — v2.pptx

---

## Study Files

| Block | File | Topics |
|---|---|---|
| **Block 1** | [Block-1_Loss-Functions-and-Fundamentals.md](Block-1_Loss-Functions-and-Fundamentals.md) | Loss functions (MSE, BCE, CCE), Parameters vs Hyperparameters |
| **Block 2** | [Block-2_Training-Neural-Networks.md](Block-2_Training-Neural-Networks.md) | Forward pass, Backprop, Overfitting, Bias-Variance, Remedies |
| **Block 3** | [Block-3_Evaluation-Metrics.md](Block-3_Evaluation-Metrics.md) | Confusion Matrix, Precision, Recall, F1, AUC-ROC |
| **Block 4** | [Block-4_Convolutional-Neural-Networks.md](Block-4_Convolutional-Neural-Networks.md) | CNN Architecture, Convolution, Pooling, FC Layer, PyTorch, ResNet |
| **Block 5** | [Block-5_RNNs-and-LSTMs.md](Block-5_RNNs-and-LSTMs.md) | Sequential data, RNN, BPTT, Vanishing gradient, LSTM, GRU |
| **Block 6** | [Block-6_Road-to-Transformers-and-Embeddings.md](Block-6_Road-to-Transformers-and-Embeddings.md) | NLP evolution, One-Hot, Word2Vec, ELMo, Transformers |

---

## Topic-to-Block Mapping (All 39 Topics)

| # | Topic | Block |
|---|---|---|
| 1 | What is a Loss Function? | Block 1 |
| 2 | MSE — Mean Squared Error | Block 1 |
| 3 | BCE — Binary Cross-Entropy | Block 1 |
| 4 | CCE — Categorical Cross-Entropy | Block 1 |
| 5 | Comparing the Three Loss Functions | Block 1 |
| 6 | Parameters vs Hyperparameters | Block 1 |
| 7 | Forward Pass (step-by-step) | Block 2 |
| 8 | Computing MSE Loss | Block 2 |
| 9 | Backpropagation — Output Delta & Chain Rule | Block 2 |
| 10 | After 1 Epoch — what changes? | Block 2 |
| 11 | Underfitting, Good Fit & Overfitting | Block 2 |
| 12 | Bias-Variance Tradeoff | Block 2 |
| 13 | Detecting Overfitting & Remedies | Block 2 |
| 14 | Confusion Matrix & Evaluation Metrics | Block 3 |
| 15 | What is a CNN? | Block 4 |
| 16 | Convolution as a Matrix Operation | Block 4 |
| 17 | Pooling Layer (Max Pooling) | Block 4 |
| 18 | Fully Connected Layer | Block 4 |
| 19 | Full CNN Architecture & Flow | Block 4 |
| 20 | CNN in PyTorch (code) | Block 4 |
| 21 | Famous CNN Architectures & Drawbacks | Block 4 |
| 22 | What is Sequential Data? | Block 5 |
| 23 | Vanilla RNN | Block 5 |
| 24 | Backpropagation Through Time (BPTT) | Block 5 |
| 25 | Vanishing & Exploding Gradient Problem | Block 5 |
| 26 | LSTM — Motivation & Four Gates | Block 5 |
| 27 | LSTM Full Mathematics | Block 5 |
| 28 | LSTM Worked Example | Block 5 |
| 29 | Evolution of Language Processing | Block 6 |
| 30 | Limitations of RNN Models | Block 6 |
| 31 | Feature Learning Through Embeddings — The Need | Block 6 |
| 32 | One-Hot Vector Representation | Block 6 |
| 33 | Limitations of Sparse Representations | Block 6 |
| 34 | Distributional Semantics | Block 6 |
| 35 | Dense Representations / Word Embeddings | Block 6 |
| 36 | Embedding Lookup Tables in PyTorch | Block 6 |
| 37 | Word2Vec — Skip-gram & CBOW | Block 6 |
| 38 | Static Embeddings & Polysemy Problem | Block 6 |
| 39 | ELMo & Intro to Transformers / Attention | Block 6 |

---

## Key Equations to Memorize

```
MSE:   L = (1/n) Σ (y − ŷ)²
BCE:   L = −(1/n) Σ [y·log(ŷ) + (1−y)·log(1−ŷ)]
CCE:   L = −(1/n) Σᵢ Σc yᵢc·log(ŷᵢc)

Backprop: ∂L/∂w = (∂L/∂ŷ)(∂ŷ/∂z)(∂z/∂w)
Update:   w ← w − η × ∂L/∂w

RNN:   hₜ = tanh(Wₕ·hₜ₋₁ + Wₓ·xₜ + b)

LSTM:  fₜ = σ(Wf·[hₜ₋₁,xₜ]+bf)          ← forget gate
       iₜ = σ(Wi·[hₜ₋₁,xₜ]+bi)          ← input gate
       C̃ₜ = tanh(Wc·[hₜ₋₁,xₜ]+bc)       ← candidate
       Cₜ = fₜ⊙Cₜ₋₁ + iₜ⊙C̃ₜ            ← cell state
       oₜ = σ(Wo·[hₜ₋₁,xₜ]+bo)          ← output gate
       hₜ = oₜ⊙tanh(Cₜ)                 ← hidden state

Attention: Attention(Q,K,V) = Softmax(QKᵀ/√d_k)×V
```

---

## Recommended Study Order

**For understanding (building from fundamentals):**
Block 1 → Block 2 → Block 3 → Block 4 → Block 5 → Block 6

**For interview prep (by importance):**
Block 2 (backprop) → Block 5 (LSTM) → Block 6 (embeddings/Transformers) → Block 1 (losses) → Block 3 (metrics) → Block 4 (CNNs)

**For quick revision:**
Use the "Quick Reference Cheatsheet" at the bottom of each block file.

---

## Additional Resources (from Q&A session)

- **Recommended book:** Fundamentals of ML by Apress Publications
- **Practice problems:** Kaggle ML problems for unsupervised learning
- **Reference video:** https://www.youtube.com/watch?v=LWMzyfvuehA (Attention/Transformer intro)
- **Post-read materials:** Available on upGrad platform under Live Learning → Past Sessions → Resources
