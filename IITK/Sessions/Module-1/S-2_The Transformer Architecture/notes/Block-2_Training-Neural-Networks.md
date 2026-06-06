# Block 2: Training a Neural Network
## Forward Pass, Backpropagation, Overfitting & Remedies

> **Session:** Lecture 2 — The Transformer Architecture  
> **Topics covered:** 7–13

---

## Learning Roadmap for This Block

```
Forward Pass → Compute Loss → Backprop (Chain Rule) → Weight Update
     ↓
Understanding Overfitting/Underfitting → Bias-Variance → Detection → Remedies
```

This block answers: **How does a neural network actually learn?** Everything here is the engine room — the math and mechanics behind gradient descent.

---

## Topic 7: The Forward Pass

### What is the Forward Pass?

The forward pass is a single left-to-right evaluation of the network: input → hidden layers → output prediction. No weights are updated here — this is pure **inference**.

### The Lecture's Worked Example

**Network setup (from slides):**

```
Input: x = 1.0    Target: y = 1.0    Learning rate: η = 0.5

Weights:
  w₁ = 0.6  (input → hidden neuron 1)
  w₂ = 0.4  (input → hidden neuron 2)
  w₃ = 0.8  (hidden neuron 1 → output)
  w₄ = 0.5  (hidden neuron 2 → output)
```

**Architecture:**
```
         h₁ (sigmoid)
x → w₁ ↗            ↘ w₃
                         → z₃ → σ(z₃) → ŷ
x → w₂ ↘            ↗ w₄
         h₂ (sigmoid)
```

### Step 1: Compute Hidden Neuron 1

```
z₁ = w₁ × x = 0.6 × 1.0 = 0.6
h₁ = σ(z₁) = 1 / (1 + e^(−0.6)) = 0.6457
```

The Sigmoid squashes z₁ = 0.6 to h₁ = 0.6457. Since z₁ > 0, h₁ > 0.5.

### Step 2: Compute Hidden Neuron 2

```
z₂ = w₂ × x = 0.4 × 1.0 = 0.4
h₂ = σ(z₂) = 1 / (1 + e^(−0.4)) = 0.5987
```

h₂ < h₁ because w₂ < w₁ → smaller pre-activation → smaller post-activation.

### Step 3: Compute Output

```
z₃ = w₃ × h₁ + w₄ × h₂
   = 0.8 × 0.6457 + 0.5 × 0.5987
   = 0.5166 + 0.2994
   = 0.8159

ŷ = σ(z₃) = σ(0.8159) = 0.6934
```

**Prediction: ŷ = 0.6934, Target: y = 1.0**  
The prediction is too low by 0.3066 — the network needs to learn.

### The Sigmoid Function (Activation)

```
σ(z) = 1 / (1 + e^(−z))
```

Properties:
- Output range: (0, 1) — always between 0 and 1, never exactly 0 or 1
- S-shaped (sigmoidal) curve
- Derivative: `σ'(z) = σ(z) × (1 − σ(z))` — maximized at z=0 (value: 0.25)
- Problem: saturates at extremes (gradient ≈ 0 for large |z|) → vanishing gradient

### Why Do We Need Non-Linear Activations?

Without activation functions, stacking layers does nothing — a chain of linear transformations is still just a linear transformation. Non-linearity is what gives neural networks their expressive power to model complex, non-linear patterns.

---

## Topic 8: Computing MSE Loss

From the example above:

```
L = ½ × (y − ŷ)²
  = ½ × (1.0 − 0.6934)²
  = ½ × (0.3066)²
  = ½ × 0.0940
  = 0.0470
```

The ½ is a notational convenience that cancels the 2 produced during differentiation (see backprop below).

**Loss = 0.0470.** Not zero — the model needs to update its weights to reduce this.

---

## Topic 9: Backpropagation — Output Delta & Weight Updates

### What is Backpropagation?

Backpropagation (backprop) is the algorithm that computes **how much each weight contributed to the loss**. It applies the **chain rule of calculus** starting from the output layer and moving backward through the network.

Invented in the context of neural networks by Rumelhart, Hinton, and Williams (1986) — though the math predates this.

### The Chain Rule (Core Concept)

If loss L depends on ŷ, which depends on z₃, which depends on w₃:

```
∂L/∂w₃ = (∂L/∂ŷ) × (∂ŷ/∂z₃) × (∂z₃/∂w₃)
```

This is the chain rule: multiply partial derivatives along the path from L back to w₃.

### Step 1: Compute Output Error Signal (δ_out)

```
σ'(z₃) = ŷ × (1 − ŷ) = 0.6934 × 0.3066 = 0.2126

δ_out = −(y − ŷ) × σ'(z₃)
      = −0.3066 × 0.2126
      = −0.0652
```

`δ_out` is the **error signal** at the output neuron. Negative value means: weights feeding into the output should increase (prediction needs to go up toward 1.0).

### Step 2: Update Output Weights w₃ and w₄

Weight update rule:
```
w_new = w_old − η × δ_out × h
```

**For w₃:**
```
w₃_new = 0.8000 − 0.5 × (−0.0652) × 0.6457
        = 0.8000 + 0.0210
        = 0.8210
```

**For w₄:**
```
w₄_new = 0.5000 − 0.5 × (−0.0652) × 0.5987
        = 0.5000 + 0.0195
        = 0.5195
```

**Notice:** h₁ > h₂ (0.6457 > 0.5987) so Δw₃ > Δw₄. Neurons with higher activation receive larger weight updates. This makes intuitive sense — a neuron that fires more has more influence on the error.

### Step 3: Backprop Through Hidden Layers (w₁, w₂)

The error signal propagates further backward through the hidden neurons. Their delta values are computed using the chain rule again — multiplied by the corresponding output weight and the hidden neuron's sigmoid derivative.

### After 1 Epoch — The Results

| | Before (epoch 0) | After (epoch 1) | Change |
|---|---|---|---|
| w₁ | 0.6000 | 0.6060 | +0.0060 |
| w₂ | 0.4000 | 0.4039 | +0.0039 |
| w₃ | 0.8000 | 0.8210 | +0.0210 |
| w₄ | 0.5000 | 0.5195 | +0.0195 |
| ŷ | 0.6934 | 0.6987 | +0.0053 |
| Loss | 0.0470 | 0.0454 | −3.5% |

One epoch reduced the loss by 3.5%. Real training runs thousands to millions of epochs.

### The Gradient Descent Update Rule

```
w ← w − η × ∂L/∂w
```

- `η` = learning rate (how big a step to take)
- `∂L/∂w` = gradient (which direction makes loss increase)
- We subtract because we want to **decrease** the loss

### Variants of Gradient Descent

| Variant | Description | Pros | Cons |
|---|---|---|---|
| Batch GD | Gradient over ALL data before 1 update | Stable convergence | Slow for large datasets |
| Stochastic GD (SGD) | Gradient over 1 sample per update | Fast, can escape local minima | Noisy updates |
| Mini-batch GD | Gradient over a batch (e.g., 32 samples) | Balance of speed and stability | Standard in practice |

### Popular Optimizers Beyond SGD

| Optimizer | Key Idea |
|---|---|
| SGD with Momentum | Accumulates velocity in gradient direction — smooths updates |
| Adam | Adaptive learning rates per parameter — most widely used today |
| AdamW | Adam + weight decay regularization — preferred for LLMs |
| RMSProp | Scales gradients by running average of recent magnitudes |

---

## Topic 10: After 1 Epoch — The Big Picture

One epoch = one pass through the entire dataset. Training involves:

```
Repeat for N epochs:
    For each batch in dataset:
        1. Forward pass → compute ŷ
        2. Compute loss L(y, ŷ)
        3. Backward pass → compute ∂L/∂w for all w
        4. Update all weights
```

Key observations from the worked example:
- Loss decreases each epoch (if learning rate is appropriate)
- The prediction (ŷ) moves toward the target (y = 1.0)
- **All weights update simultaneously** — a key property of backprop (weights are frozen during the forward pass, then all updated together at the backward pass)

---

## Topic 11: Underfitting, Good Fit & Overfitting

This is one of the most fundamental concepts in all of machine learning.

### The Three Regimes

**Underfitting (High Bias)**
```
Training error:   HIGH
Validation error: HIGH
Behaviour: Model is too simple to capture data patterns
Cause: Too few parameters, too little training, bad features
Fix: More layers, more neurons, train longer, better features
```

**Good Fit**
```
Training error:   LOW
Validation error: LOW (comparable to training)
Behaviour: Model generalizes well to unseen data
Cause: Right balance of model complexity and data
Goal: This is what we want!
```

**Overfitting (High Variance)**
```
Training error:   VERY LOW (near zero)
Validation error: HIGH
Behaviour: Model memorizes training data, fails on new data
Cause: Model too complex, too little data, too much training
Fix: Regularization, dropout, more data, early stopping
```

### Intuition: The Curve Fitting Analogy

Imagine fitting a polynomial to data points:
- **Underfitting:** Linear fit (degree 1) through data that follows a quadratic curve — misses the pattern
- **Good fit:** Quadratic fit (degree 2) — captures the true underlying curve
- **Overfitting:** Degree-20 polynomial that passes through every training point — memorizes noise, fails on new points

---

## Topic 12: The Bias-Variance Tradeoff

### Decomposing Prediction Error

Total prediction error = Bias² + Variance + Irreducible Noise

```
Bias²:           Error from wrong model assumptions (underfitting)
Variance:        Error from sensitivity to training data (overfitting)
Irreducible:     Error from inherent data noise — can't be eliminated
```

### The Tradeoff

```
Simple Model:  High Bias,   Low Variance  → Underfits
Complex Model: Low Bias,    High Variance → Overfits
Sweet Spot:    Low Bias,    Low Variance  → Generalizes
```

You cannot simultaneously minimize both bias and variance — every modeling decision shifts this balance. The sweet spot must be found empirically.

### Visual Mental Model

```
Target:
         ●●●                  ←— High Bias, Low Variance
         (centered but far from true target: biased model,
          consistent predictions = underfitting)

   ●         ●               ←— Low Bias, High Variance  
      ●   ●                   (scattered around true target:
         ●                     sensitive to training data = overfitting)

         ●                   ←— Low Bias, Low Variance  ✓
       ●   ●                  (tight cluster around true target
         ●                     = good generalization)
```

---

## Topic 13: Detecting Overfitting & Remedies

### The Train vs. Validation Loss Curve

This is the most reliable diagnostic tool in deep learning:

```
Loss
 │
 │  Training Loss ──────────────────────────
 │                    Validation Loss ─────┐
 │                                         │ overfitting gap
 │                               ┌─────────┘
 │                               ↑ "elbow" — optimal stopping point
 └────────────────────────────────────────── Epochs
```

**Signal 1:** Train loss ↓ continuously, Val loss starts ↑ after some point
→ Classic overfitting. Stop before the elbow (early stopping).

**Signal 2:** Both losses stay high, barely improving
→ Classic underfitting. Model needs more capacity or training.

**Signal 3:** Train loss ≈ Val loss, both low
→ Good generalization. 

### Remedy 1: Regularization (L1 / L2)

Adds a penalty term to the loss function that discourages large weights:

**L2 (Ridge):**
```
L_total = L_original + λ × Σ w²
```

**L1 (Lasso):**
```
L_total = L_original + λ × Σ |w|
```

- L2 encourages weights toward zero (small weights → simpler model)
- L1 encourages **exactly zero weights** (sparse model — acts as feature selection)
- `λ` is the regularization strength hyperparameter

### Remedy 2: Dropout

During training, randomly **zero out** a fraction of neurons at each forward pass.

```python
nn.Dropout(p=0.5)  # p = probability of a neuron being zeroed out
```

**Why it works:** Prevents neurons from co-adapting — each neuron must learn features independently, without relying on other neurons to compensate. Forces robust representations.

**Important:** Dropout is ON during training, OFF during inference (evaluation). PyTorch handles this with `model.train()` and `model.eval()`.

*From Q&A session: Dropout is best when the model is big/complex. For small models it can hurt.*

### Remedy 3: Early Stopping

Monitor validation loss during training. Stop when validation loss starts increasing:

```python
if val_loss > best_val_loss:
    patience_counter += 1
    if patience_counter > patience_limit:
        stop_training()
else:
    best_val_loss = val_loss
    save_checkpoint(model)
    patience_counter = 0
```

*From Q&A session: Early stopping is best when overfitting shows during training (e.g., small dataset where val loss starts rising).*

### Remedy 4: More Data

The most effective remedy when possible. More diverse training data:
- Reduces variance (model has more examples to generalize from)
- Makes the model's learned patterns more reliable

**Data Augmentation** — if you can't collect more data, artificially expand your dataset:
- Images: rotation, flipping, cropping, color jitter
- Text: paraphrasing, synonym replacement, back-translation
- Time series: noise injection, time warping

### Remedy 5: Cross-Validation

Instead of a single train/val split, use **k-fold cross-validation**:
1. Split data into k equal folds (typically k=5 or 10)
2. Train k models, each using k−1 folds for training and 1 fold for validation
3. Average validation scores across all k runs

Provides a more reliable estimate of true generalization performance, especially on small datasets.

### Remedy 6: Reduce Model Complexity

- Fewer layers
- Fewer neurons per layer
- Smaller embedding dimensions
- Simpler architecture

---

## Interview Questions — Block 2

### Fundamental Questions

**Q1: What is backpropagation? Explain it in simple terms.**

> Backpropagation is an algorithm that computes how much each weight in a neural network contributed to the prediction error. It applies the chain rule of calculus to propagate the gradient of the loss function backward through the network, from the output layer to the input layer. Once gradients are computed for all weights, an optimizer (like SGD or Adam) uses them to update the weights in the direction that reduces the loss.

**Q2: What is the vanishing gradient problem?**

> In deep networks, gradients are multiplied together as they flow backward through layers (chain rule). If activations like Sigmoid are used, their derivatives saturate near 0 or 1, producing very small values (< 0.25). Multiplying many small numbers makes the gradient exponentially small — weights in early layers barely update, so they don't learn. Solutions: use ReLU activations, batch normalization, residual connections, or LSTM-style architectures.

**Q3: How do you detect overfitting?**

> The clearest signal is the divergence between training loss and validation loss: training loss continues to decrease while validation loss starts increasing. You can also look at the gap between training accuracy and validation accuracy. Other signals: near-zero training loss with high test loss, model performs much worse on held-out data than training data.

**Q4: What is the difference between L1 and L2 regularization?**

> L2 (Ridge) adds `λ × Σ w²` to the loss — it penalizes large weights quadratically, pushing all weights toward (but rarely exactly to) zero. L1 (Lasso) adds `λ × Σ |w|` — it creates a linear penalty that can drive weights to exactly zero, producing sparse models. L1 is effectively a feature selector; L2 produces distributed, small weights. In deep learning, L2 (weight decay) is more commonly used.

**Q5: What is dropout and why does it prevent overfitting?**

> Dropout randomly sets a fraction p of neurons to zero during each training forward pass. This prevents neurons from co-adapting (relying on specific other neurons). The network is forced to learn more robust, redundant representations since it can't count on any neuron always being present. At test time, dropout is turned off and all weights are scaled by (1-p) to compensate.

**Q6: Explain the bias-variance tradeoff.**

> Every model's generalization error decomposes into bias (systematic error from model assumptions — underfitting), variance (sensitivity to training data fluctuations — overfitting), and irreducible noise. Increasing model complexity typically reduces bias but increases variance, and vice versa. The goal is to find the complexity sweet spot where both bias and variance are simultaneously low.

### Advanced Questions

**Q7: What is the difference between batch gradient descent, stochastic gradient descent, and mini-batch gradient descent?**

> Batch GD computes gradients over the entire dataset before updating — gives accurate gradient estimates but is slow and memory-intensive for large datasets. SGD updates after every single sample — fast and can escape local minima due to noise, but updates are very noisy. Mini-batch GD (standard in practice) computes gradients over a subset (batch) of 16–512 samples — balances accuracy and speed, fits on GPU memory, and exploits hardware parallelism.

**Q8: Why are weights initialized randomly? What happens if all weights are zero?**

> If all weights start at zero, all neurons in a layer produce identical outputs (same activation), receive identical gradients, and update identically — they never differentiate. This is the "symmetry problem." Random initialization breaks symmetry, allowing different neurons to specialize in learning different features.

**Q9: What is early stopping's relationship to regularization?**

> Early stopping is effectively a form of regularization. When you stop training before full convergence, you implicitly prevent the model from overfitting to the training data by limiting the optimization steps. It has been shown to be roughly equivalent to L2 regularization in certain settings. It's also computationally free — no extra math needed.

---

## Key Learning Insights

> **Insight 1:** The forward pass and backward pass together constitute one training step. The forward pass is just a series of matrix multiplications and activation functions — computationally cheap to understand. Backprop is where the "learning" actually happens.

> **Insight 2:** Gradient descent doesn't find the global minimum — it finds a local minimum. In practice, deep networks have many local minima, but they tend to be similar in quality (the "flat minima" hypothesis). The noise in mini-batch GD actually helps escape sharp, bad minima.

> **Insight 3:** The "elbow" in the training curve (where val loss starts rising) is the model's optimal state. Everything after that is memorization, not learning.

> **Insight 4:** Overfitting and underfitting are not about the model being wrong — they're about **generalization**. A model that memorizes 1 million training examples perfectly is useless if it can't handle a single new example.

> **Insight 5:** Dropout has an elegant interpretation: training with dropout is like training an ensemble of `2^n` different network architectures (where n is the number of neurons) and averaging their predictions at test time.

> **Insight 6:** In modern deep learning, batch normalization (normalizing layer inputs) has largely replaced dropout for CNNs and transformers. Understanding why: batch norm reduces internal covariate shift AND has a slight regularizing effect. But for dense layers, dropout still has a place.

---

## Quick Reference Cheatsheet

```
Forward Pass:   x → z₁ = Wx + b → h₁ = σ(z₁) → z₂ → ... → ŷ
Loss:           L = ½(y − ŷ)²  or  BCE  or  CCE
Backprop:       ∂L/∂w = (∂L/∂ŷ)(∂ŷ/∂z)(∂z/∂w)  [chain rule]
Update:         w ← w − η × ∂L/∂w

Underfitting:   train ↑, val ↑       → more capacity
Overfitting:    train ↓↓, val ↑      → regularize/dropout/more data
Good Fit:       train ↓, val ↓ (close) → ship it

Remedies:       L1/L2 regularization, Dropout, Early Stopping,
                More Data, Data Augmentation, Cross-Validation
```
