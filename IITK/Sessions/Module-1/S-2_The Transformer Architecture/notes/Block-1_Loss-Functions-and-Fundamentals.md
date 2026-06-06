# Block 1: Neural Network Fundamentals
## Loss Functions, Parameters vs Hyperparameters

> **Session:** Lecture 2 — The Transformer Architecture  
> **Topics covered:** 1–6

---

## Learning Roadmap for This Block

```
Loss Functions → Why they exist → MSE → BCE → CCE → Compare all three → Parameters vs Hyperparameters
```

The core idea: a neural network needs a **signal** to know how wrong it is and in which direction to improve. That signal is the **loss function**. Without it, learning is impossible.

---

## Topic 1: What is a Loss Function?

### The Core Idea

A loss function (also called a cost function or objective function) answers one question:

> **"How far is the model's prediction from the correct answer?"**

It turns a model's mistake into a **single number**. The optimizer then uses this number to nudge the model's internal weights in the direction that reduces the mistake. This cycle — predict, measure error, update — is the heartbeat of all supervised learning.

### The Training Loop (Memorize This)

```
1. Forward Pass   → model receives input x, produces prediction ŷ
2. Compute Loss   → L(y, ŷ) — how far off is ŷ from true label y?
3. Backpropagation → compute gradients ∂L/∂w for every weight w
4. Update Weights  → w ← w − η × ∂L/∂w  (gradient descent step)
5. Repeat          → thousands of times until loss ≈ 0 (or plateaus)
```

### Intuition: The GPS Analogy

Think of loss as a GPS telling you "you are 47 km from your destination." The optimizer is the driver adjusting the route. The loss function doesn't tell the driver HOW to drive — that's the optimizer's job — but it always tells the driver HOW FAR OFF they are.

### What Makes a Good Loss Function?

| Property | Why It Matters |
|---|---|
| Differentiable | Backprop requires gradients — non-differentiable = no learning |
| Sensitive to errors | Small mistakes should produce small loss, big mistakes big loss |
| Task-appropriate | Regression ≠ classification — wrong choice = poor training |
| Non-negative | L = 0 means perfect prediction; L > 0 means error |

### Choose Based on Task

| Task | Loss Function | Output Activation |
|---|---|---|
| Regression | MSE | None / Linear |
| Binary Classification | Binary Cross-Entropy | Sigmoid |
| Multi-Class Classification | Categorical Cross-Entropy | Softmax |

---

## Topic 2: Loss Function 1 — Mean Squared Error (MSE)

### Formula

```
MSE = (1/n) × Σᵢ (yᵢ − ŷᵢ)²
```

Where:
- `yᵢ` = actual (true) value
- `ŷᵢ` = model's predicted value
- `n` = number of training samples

*Note: In backprop derivations, MSE is often written as `½ × (y − ŷ)²` for a single sample — the ½ cancels the 2 that appears when differentiating.*

### Step-by-Step Intuition

1. For each sample, compute the **error** = `(y − ŷ)`
2. **Square it** — this eliminates negative values AND penalizes large errors more
3. **Average** across all samples

### Why Squaring?

Two reasons:
- **Sign:** An error of −5 is just as bad as +5. Squaring makes both 25.
- **Penalty:** An error of 10 becomes 100; an error of 2 becomes 4. Large errors are punished disproportionately, forcing the model to prioritize fixing the biggest mistakes.

### Worked Example from Lecture

| House | Actual (y) | Predicted (ŷ) | Error | Error² |
|---|---|---|---|---|
| 1 | 300K | 280K | 20K | 400M |
| 2 | 450K | 460K | −10K | 100M |
| 3 | 200K | 250K | −50K | 2500M |

```
MSE = (400M + 100M + 2500M) / 3 ≈ 1,000M
```

The 50K miss on House 3 dominates — MSE forces the model to fix its worst predictions first.

### When to Use MSE

- Output is a **continuous number** (house price, temperature, stock value, age)
- You want to penalize large errors heavily
- Your data doesn't have many outliers (outliers inflate MSE unfairly)

### When NOT to Use MSE

- Binary or multi-class outputs — squaring probabilities doesn't make mathematical sense
- When you have extreme outliers — consider MAE (Mean Absolute Error) instead

### Gradient of MSE (for backprop)

```
∂L/∂ŷ = −(y − ŷ)   [for the ½(y−ŷ)² version]
```

This gradient flows backward through the network during backpropagation.

---

## Topic 3: Loss Function 2 — Binary Cross-Entropy (BCE)

### Formula

```
BCE = −(1/n) × Σᵢ [ yᵢ · log(ŷᵢ) + (1 − yᵢ) · log(1 − ŷᵢ) ]
```

Where:
- `yᵢ ∈ {0, 1}` — true label (binary: yes/no, spam/not spam)
- `ŷᵢ ∈ (0, 1)` — predicted probability (output of Sigmoid)

### Why Not MSE for Classification?

If you use MSE with a Sigmoid output, the loss surface becomes non-convex with many local minima, and gradients near 0 and 1 become vanishingly small (the "saturated Sigmoid" problem). BCE is purpose-built for probability outputs.

### Unpacking the Formula

The formula has two cases baked in:

**When y = 1 (true label is Positive):**
```
Loss = −log(ŷ)
```
- If ŷ = 0.95 (confident & correct): loss = −log(0.95) ≈ 0.05 — very low ✓
- If ŷ = 0.05 (confident & WRONG): loss = −log(0.05) ≈ 3.0 — very high ✗

**When y = 0 (true label is Negative):**
```
Loss = −log(1 − ŷ)
```
- If ŷ = 0.05 (confident it's negative, correct): loss ≈ 0.05 — very low ✓
- If ŷ = 0.95 (confident it's positive, WRONG): loss ≈ 3.0 — very high ✗

### The Key Insight: Logarithmic Penalty

```
log(0.95) ≈ −0.05   → small penalty for being right
log(0.50) ≈ −0.69   → medium penalty for being uncertain
log(0.05) ≈ −3.0    → HUGE penalty for being confidently wrong
```

BCE **brutally punishes confident mistakes**. This is exactly what you want — it forces the model to be both accurate AND well-calibrated (uncertain when it should be).

### Worked Example

| Sample | True Label (y) | Predicted (ŷ) | Loss |
|---|---|---|---|
| Email 1 | 1 (Spam) | 0.95 | ≈ 0.05 (correct & confident) |
| Email 2 | 1 (Spam) | 0.50 | ≈ 0.69 (uncertain) |
| Email 3 | 1 (Spam) | 0.05 | ≈ 3.0 (wrong & confident) |

### Output Activation: Sigmoid

BCE requires the output to be a probability between 0 and 1. The **Sigmoid** function does this:

```
σ(z) = 1 / (1 + e^(−z))
```

It maps any real number to (0, 1). BCE + Sigmoid = the standard recipe for binary classification.

---

## Topic 4: Loss Function 3 — Categorical Cross-Entropy (CCE)

### Formula

```
CCE = −(1/n) × Σᵢ Σc [ yᵢc · log(ŷᵢc) ]
```

Where:
- `yᵢc = 1` if sample `i` belongs to class `c`, else `0`  (one-hot encoded)
- `ŷᵢc` = predicted probability for class `c`
- Sum is over all samples `i` and all classes `c`

### Relationship to BCE

CCE is the **generalization of BCE to K > 2 classes**. When K=2, CCE reduces to BCE exactly.

### Why Only the True Class Matters

Because `yᵢc = 0` for all wrong classes, only the log probability of the **correct class** contributes to the loss:

```
Loss for sample i = −log(ŷᵢ_correct_class)
```

This simplification means: **maximize the probability assigned to the correct class**.

### Output Activation: Softmax

CCE requires class probabilities that sum to 1. **Softmax** does this:

```
Softmax(zc) = e^(zc) / Σk e^(zk)
```

It converts raw scores (logits) into a probability distribution across all classes.

### Worked Example from Lecture

Image classifier: Cat / Dog / Bird

| Class | Softmax Output (ŷ) | True Label (y) |
|---|---|---|
| Cat | 0.70 | 1 |
| Dog | 0.20 | 0 |
| Bird | 0.10 | 0 |

```
Loss = −log(0.70) ≈ 0.36
```

Only Cat's probability (the true class) matters. The higher the model's confidence in the correct class, the lower the loss.

### What Would Perfect Prediction Look Like?

If ŷ_cat = 1.0: Loss = −log(1.0) = 0 → perfect.  
If ŷ_cat = 0.01: Loss = −log(0.01) ≈ 4.6 → terrible.

---

## Topic 5: Comparing All Three Loss Functions

| | MSE | Binary Cross-Entropy | Categorical Cross-Entropy |
|---|---|---|---|
| **Formula** | (1/n)Σ(y−ŷ)² | −[y·log(ŷ) + (1−y)·log(1−ŷ)] | −Σc y_c·log(ŷ_c) |
| **Task** | Regression | Binary Classification | Multi-Class Classification |
| **Output Range** | Any real number | Probability [0,1] | Probability per class [0,1] |
| **Output Activation** | None / Linear | Sigmoid | Softmax |
| **Penalty Style** | Squares the error | Log penalizes confident mistakes | Only true class log probability |
| **Example Use** | House prices, temperature | Spam detection, fraud | MNIST digits, image classification |

### The Deeper Connection: Information Theory

Both cross-entropy losses come from **information theory**. The idea is:

> Loss = how many bits of "surprise" does the true label add given the model's prediction?

If the model confidently predicts the correct answer, there's no surprise → low loss.  
If the model confidently predicts the WRONG answer, there's maximum surprise → high loss.

This is why log is used: `−log(p)` is the "surprise" or "self-information" of an event with probability `p`.

---

## Topic 6: Parameters vs Hyperparameters

This distinction is fundamental and appears in nearly every ML interview.

### Parameters

**Definition:** Values learned **automatically** by the optimizer during training. The model updates these through gradient descent.

**Examples:**
- Weights (`W`) in every layer of a neural network
- Biases (`b`) in every layer
- Coefficients in linear regression

**Key property:** You never set these manually. The optimizer finds their optimal values through backpropagation.

```python
# Parameters are inside the model — you don't set them, the optimizer does
model = nn.Linear(10, 1)  # W (10×1) and b (1) are parameters
optimizer.step()           # optimizer updates parameters every iteration
```

### Hyperparameters

**Definition:** Configuration values you **set before training begins**. They control HOW the training happens, not what is learned.

**Examples:**

| Hyperparameter | What it controls |
|---|---|
| Learning rate (η) | How big each gradient descent step is |
| Batch size | How many samples per gradient update |
| Number of epochs | How many full passes through the dataset |
| Number of layers | Model depth/capacity |
| Number of neurons per layer | Model width |
| Dropout rate | Regularization strength |
| Kernel size (CNNs) | Size of the convolution filter |

**Key property:** You set these. Bad hyperparameter choices = bad model, regardless of training.

### The Crucial Difference

```
Parameters: model learns them   → stored inside model weights
Hyperparameters: YOU choose them → stored in your training script/config
```

### Hyperparameter Tuning

Finding good hyperparameters is itself an art/science:
- **Grid Search** — try all combinations of a hyperparameter grid
- **Random Search** — sample randomly from the hyperparameter space
- **Bayesian Optimization** — use a probabilistic model to guide the search
- **Learning Rate Schedulers** — dynamically adjust learning rate during training

### The Learning Rate: The Most Critical Hyperparameter

```
η too high: overshoots the minimum, loss oscillates or diverges
η too low:  converges extremely slowly, may get stuck in local minima
η just right: steady, smooth convergence to a good minimum
```

Common starting values: 1e-3 for Adam, 1e-2 for SGD.

---

## Interview Questions — Block 1

### Fundamental Questions

**Q1: What is the difference between a loss function and a metric?**

> A loss function is used during training — it must be differentiable so gradients can flow backward. A metric is used for evaluation and can be non-differentiable (e.g., accuracy, F1). For example, you train with BCE loss but evaluate with accuracy. Sometimes they align (e.g., lower BCE → higher accuracy), but not always.

**Q2: Why can't you use MSE for binary classification?**

> Two reasons: (1) MSE combined with Sigmoid creates a non-convex loss surface with many local minima, making optimization unreliable. (2) Sigmoid saturates near 0 and 1 — its gradient is near zero there. BCE's gradient doesn't suffer this problem: `∂BCE/∂z = (ŷ − y)`, which remains non-zero even when Sigmoid saturates.

**Q3: Why does BCE use log?**

> The log creates an asymmetric, information-theoretic penalty. A small probability for the correct class incurs a massive penalty (−log(0.01) = 4.6), while a high probability incurs almost none (−log(0.99) ≈ 0.01). This property forces the model to be both accurate AND calibrated. Mathematically, BCE is derived from Maximum Likelihood Estimation under a Bernoulli distribution.

**Q4: What's the difference between parameters and hyperparameters? Give 3 examples of each.**

> Parameters: (1) weights in a neural network layer, (2) biases, (3) embedding vectors in an embedding layer. Hyperparameters: (1) learning rate, (2) number of layers, (3) dropout rate. Parameters are updated by the optimizer; hyperparameters are set by the practitioner before training.

**Q5: If your model has high training loss AND high validation loss, what does that indicate?**

> Underfitting. The model lacks the capacity or training to learn the patterns in the data. Fix: increase model complexity, train longer, add more features, reduce regularization.

**Q6: What happens if the learning rate is too large?**

> The model overshoots the loss minimum. Instead of converging, the loss may oscillate or even diverge (explode to infinity). In gradient descent terms: `w_new = w_old − η × gradient`. If η is too large, the step goes past the minimum and ends up on the other side, then overshoots again.

### Advanced Questions

**Q7: Why is the `½` sometimes included in MSE?**

> The ½ is a mathematical convenience for backpropagation. `d/dŷ [½(y−ŷ)²] = −(y−ŷ)`. Without the ½, the derivative is `−2(y−ŷ)` — the factor of 2 would need to be absorbed into the learning rate anyway, so it's cleaner to include it upfront. It doesn't affect the location of the minimum.

**Q8: Can you use CCE for binary classification?**

> Yes, but it's redundant. With K=2 classes, CCE reduces to exactly BCE. BCE is preferred for binary tasks because it's simpler and computationally cheaper (one output neuron with Sigmoid vs two with Softmax).

**Q9: What is label smoothing and why is it used with CCE?**

> Label smoothing replaces hard one-hot targets (0 or 1) with soft targets like 0.1 and 0.9. This prevents the model from becoming overconfident (assigning probability ≈ 1.0 to the true class), which tends to cause overfitting. It's a regularization technique that improves generalization.

---

## Key Learning Insights

> **Insight 1:** The choice of loss function is not optional — it's a **design decision** that must match the problem type. Using the wrong loss function is one of the most common beginner mistakes.

> **Insight 2:** All three loss functions have the same goal: **push the model's prediction toward the true label**. They differ only in HOW they measure distance from the truth.

> **Insight 3:** Cross-entropy is derived from **maximum likelihood estimation** — minimizing BCE/CCE is mathematically equivalent to maximizing the likelihood of the training data under the model's distribution. This is a deep connection between loss functions and probability theory.

> **Insight 4:** Hyperparameters have an enormous impact on model quality — sometimes more than architectural choices. A simple model with well-tuned hyperparameters often outperforms a complex model with poor ones.

> **Insight 5:** The learning rate is the single most important hyperparameter. A useful trick: start with a **learning rate range test** — plot loss vs. learning rate over a few iterations to find the sweet spot.

> **Insight 6:** In production systems, the loss function you optimize is often different from the business metric you care about (e.g., you optimize BCE but you care about precision at a specific threshold). Understanding this gap is crucial for ML engineering.

---

## Quick Reference Cheatsheet

```
Task Type              → Loss Function            → Output Activation
Regression             → MSE (or MAE/Huber)       → Linear
Binary Classification  → Binary Cross-Entropy     → Sigmoid
Multi-Class            → Categorical Cross-Entropy → Softmax
Multi-Label            → Binary Cross-Entropy      → Sigmoid (per label)

Parameter  = learned by optimizer = weights, biases
Hyperparameter = set by you = learning rate, batch size, layers, dropout
```
