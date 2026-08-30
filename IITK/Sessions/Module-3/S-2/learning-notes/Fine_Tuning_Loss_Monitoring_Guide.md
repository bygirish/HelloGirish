# Fine-Tuning LLMs: Ensuring Loss Reduction Across Batches

**Course:** Generative AI & ML — IIT Kharagpur × upGrad  
**Topic:** Loss Tracking and Validation During Fine-Tuning  
**Date:** August 2026

---

## Table of Contents

1. [Layer 1: Conceptual Understanding](#layer-1--conceptual-understanding)
2. [Layer 2: Metrics to Track](#layer-2--metrics-to-track)
3. [Layer 3: Runnable Code with Monitoring](#layer-3--runnable-code-with-monitoring)
4. [Visualization Code](#visualization-code)
5. [Key Patterns to Watch](#key-patterns-to-watch)
6. [Exam & Interview Quick Reference](#exam--interview-quick-reference)

---

## Layer 1 — Conceptual Understanding

### The Core Problem

After you fine-tune an LLM on your task, how do you know it actually improved? You need to measure something.

The loss at **one batch** is noisy (random variation in samples). So you must:

1. **Track training loss across all batches** in an epoch (smoothed)
2. **Validate on held-out data** (data the model never saw during training)
3. **Compare** train loss vs. validation loss to detect overfitting

### The Analogy

Think of it like studying for an exam:

- **Training loss** = your scores on practice problems you study
- **Validation loss** = your score on unseen practice tests
- If train loss ↓ but validation loss ↑ = you memorized answers, didn't learn (overfitting)
- If both ↓ = you actually learned

### Why Single Batch Loss Is Not Enough

```
Batch 1: loss = 2.34
Batch 2: loss = 2.89  ← Is this worse? Or just random variation?
Batch 3: loss = 2.10
```

Different batches contain different data. One batch's loss is just one data point — it tells you nothing about overall progress. You need to:

- **Average across all batches in an epoch** for training loss
- **Test on completely separate validation data** for true performance

---

## Layer 2 — Metrics to Track

### The Decision Table

| Metric | What It Measures | Healthy Pattern |
|---|---|---|
| **Training Loss (batch)** | Loss on current batch | Noisy, but downward trend over epochs |
| **Training Loss (epoch avg)** | Average loss across all batches in epoch | Should decrease each epoch |
| **Validation Loss** | Loss on held-out data (unseen during training) | Should also decrease, but slower than train |
| **Validation Accuracy** | % correct predictions on validation set | Should increase as loss decreases |
| **Perplexity** | exp(loss) — intuitive measure for LLMs | Lower is better; tells you "how confused" model is |

### Signs of Good Fine-Tuning

```
Epoch 1: Train Loss = 2.50, Val Loss = 2.45
Epoch 2: Train Loss = 2.10, Val Loss = 2.08  ← Both decreasing ✓
Epoch 3: Train Loss = 1.80, Val Loss = 1.85  ← Train ↓, Val slightly ↑ (start of overfit)
Epoch 4: Train Loss = 1.40, Val Loss = 2.20  ← Train ↓, Val ↑ (overfitting) ✗ STOP HERE
```

### Signs of Bad Fine-Tuning

```
Epoch 1: Train Loss = 2.50, Val Loss = 2.50
Epoch 2: Train Loss = 2.48, Val Loss = 2.51  ← No progress (wrong learning rate?)
Epoch 3: Train Loss = 2.50, Val Loss = 2.52  ← Stuck (gradient too small)
```

---

## Layer 3 — Runnable Code with Monitoring

### Complete Fine-Tuning Script with Loss Tracking

```python
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# ============================================================================
# SETUP: Toy Dataset
# ============================================================================

class SimpleTextDataset(Dataset):
    """Toy dataset for demonstration"""
    def __init__(self, texts, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts, 
            truncation=True, 
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )
    
    def __len__(self):
        return len(self.encodings['input_ids'])
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'labels': self.encodings['input_ids'][idx]  # For LLM, labels = input (next token prediction)
        }


# ============================================================================
# TRAINING LOOP WITH LOSS TRACKING
# ============================================================================

def fine_tune_with_monitoring(
    model_id='gpt2',
    train_texts=None,
    val_texts=None,
    num_epochs=3,
    batch_size=8,
    learning_rate=5e-5,
    device='cpu'
):
    """
    Fine-tune an LLM and track training/validation loss.
    
    Args:
        model_id: HuggingFace model ID
        train_texts: list of text strings for training
        val_texts: list of text strings for validation
        num_epochs: number of training epochs
        batch_size: samples per batch
        learning_rate: learning rate for optimizer
        device: 'cpu' or 'cuda'
    
    Returns:
        dict with training history and model
    """
    
    # Default toy data if not provided
    if train_texts is None:
        train_texts = [
            "The cat sat on the mat.",
            "Dogs are friendly animals.",
            "Machine learning is amazing.",
            "Fine-tuning improves model performance.",
        ] * 10  # 40 samples
    
    if val_texts is None:
        val_texts = [
            "The dog played in the park.",
            "Learning is a continuous process.",
        ] * 5  # 10 samples
    
    print("=" * 70)
    print("FINE-TUNING WITH LOSS MONITORING")
    print("=" * 70)
    
    # Load model and tokenizer
    print(f"\n1. Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model = model.to(device)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create datasets
    print("2. Preparing datasets...")
    train_dataset = SimpleTextDataset(train_texts, tokenizer)
    val_dataset = SimpleTextDataset(val_texts, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # History tracking
    history = {
        'train_loss_per_batch': [],
        'train_loss_per_epoch': [],
        'val_loss_per_epoch': [],
        'best_val_loss': float('inf'),
        'best_epoch': 0
    }
    
    print("\n3. Starting fine-tuning...\n")
    
    # ========== TRAINING LOOP ==========
    for epoch in range(num_epochs):
        # --- TRAINING PHASE ---
        model.train()
        train_losses = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track batch loss
            batch_loss = loss.item()
            train_losses.append(batch_loss)
            history['train_loss_per_batch'].append(batch_loss)
            
            # Update progress bar with current batch loss
            progress_bar.set_postfix({'loss': batch_loss:.4f})
        
        # Average training loss for this epoch
        avg_train_loss = np.mean(train_losses)
        history['train_loss_per_epoch'].append(avg_train_loss)
        
        # --- VALIDATION PHASE ---
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass (no gradient tracking)
                outputs = model(input_ids, labels=labels)
                val_loss = outputs.loss.item()
                val_losses.append(val_loss)
                
                progress_bar.set_postfix({'loss': val_loss:.4f})
        
        avg_val_loss = np.mean(val_losses)
        history['val_loss_per_epoch'].append(avg_val_loss)
        
        # --- EPOCH SUMMARY ---
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        
        # Check for improvement
        if avg_val_loss < history['best_val_loss']:
            history['best_val_loss'] = avg_val_loss
            history['best_epoch'] = epoch + 1
            print(f"  ✓ Validation loss improved! (best: {avg_val_loss:.4f})")
        else:
            epochs_since_improvement = (epoch + 1) - history['best_epoch']
            print(f"  ⚠ No improvement for {epochs_since_improvement} epoch(s)")
            if epochs_since_improvement >= 2:
                print(f"  → Consider stopping early to avoid overfitting")
        print(f"{'='*70}\n")
    
    return history, model
```

### Code Breakdown

#### Part 1: Forward Pass
```python
# STEP 1: Forward Pass
logits = model(batch.input_ids)  # input_ids shape: (32, 512) for 32 samples
loss = loss_fn(logits, batch.labels)

# Memory at this point:
# ✓ Model Weights: 14 GB (7B params in float16)
# ✓ Activations (h1, h2, ...): 25 GB (stored from each layer)
# ✓ Batch data (input_ids, labels): ~1 GB
# Total: ~40 GB VRAM
```

#### Part 2: Backward Pass
```python
# STEP 2: Backward Pass
loss.backward()  # Compute ∂L/∂W using stored activations

# Memory at this point (same as above):
# ✓ Model Weights: 14 GB
# ✓ Activations: 25 GB (still needed for gradient computation)
# ✓ Gradients: 14 GB (computed, temporary)
# Total: ~53 GB VRAM (peak memory usage)
```

#### Part 3: Weight Update
```python
# STEP 3: Update Weights
optimizer.step()  # W = W - lr * ∂L/∂W

# STEP 4: Clear
optimizer.zero_grad()  # Delete gradients, clear for next batch

# Memory after step 4:
# ✓ Model Weights: 14 GB (updated)
# ✗ Activations: gone
# ✗ Gradients: gone
# Total: ~14 GB (back to baseline)
```

### Key Code Features

1. **`model.train()` and `model.eval()`**: Switches between training mode (gradients tracked, dropout active) and evaluation mode (no gradients, deterministic)

2. **`torch.no_grad()`**: During validation, don't compute gradients to save memory

3. **`optimizer.zero_grad()`**: Clear old gradients before computing new ones (critical!)

4. **`tqdm`**: Progress bars showing real-time batch loss

5. **History tracking**: Store losses per batch and per epoch for visualization

---

## Visualization Code

### Function to Plot Training History

```python
def plot_training_history(history, save_path='training_history.png'):
    """Visualize training progress"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss per batch (noisy but shows immediate feedback)
    ax = axes[0]
    ax.plot(history['train_loss_per_batch'], alpha=0.5, linewidth=1, label='Per-batch loss (noisy)')
    
    # Overlay smoothed training loss (per epoch)
    ax.plot(range(len(history['train_loss_per_epoch'])), 
            history['train_loss_per_epoch'], 
            'o-', linewidth=2.5, markersize=8, label='Epoch avg (train)', color='#1976d2')
    
    ax.set_xlabel('Batch / Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax.set_title('Training Loss: Batch-Level Noise vs. Epoch Trend', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Train vs. Validation Loss (key indicator of overfitting)
    ax = axes[1]
    epochs = range(1, len(history['train_loss_per_epoch']) + 1)
    
    ax.plot(epochs, history['train_loss_per_epoch'], 'o-', linewidth=2.5, markersize=8, 
            label='Training Loss', color='#388e3c')
    ax.plot(epochs, history['val_loss_per_epoch'], 's-', linewidth=2.5, markersize=8, 
            label='Validation Loss', color='#d32f2f')
    
    # Highlight best epoch
    best_epoch = history['best_epoch']
    best_val_loss = history['best_val_loss']
    ax.scatter([best_epoch], [best_val_loss], s=200, marker='*', color='#ffc107', 
               edgecolors='black', linewidth=2, label=f'Best (Epoch {best_epoch})', zorder=5)
    
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax.set_title('Train vs. Validation Loss: Detecting Overfitting', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to {save_path}")
    plt.show()
```

### Running the Complete Example

```python
if __name__ == "__main__":
    history, model = fine_tune_with_monitoring(
        model_id='gpt2',
        num_epochs=3,
        batch_size=4,
        learning_rate=5e-5,
        device='cpu'  # Use 'cuda' if you have GPU
    )
    
    # Visualize
    plot_training_history(history)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"Best validation loss: {history['best_val_loss']:.4f} (Epoch {history['best_epoch']})")
    print(f"Final training loss:  {history['train_loss_per_epoch'][-1]:.4f}")
    print(f"Final validation loss: {history['val_loss_per_epoch'][-1]:.4f}")
    print(f"\nGap (Val - Train): {history['val_loss_per_epoch'][-1] - history['train_loss_per_epoch'][-1]:.4f}")
    print("  → Positive gap = overfitting (memorizing training data)")
    print("  → Small gap = good generalization")
```

### Expected Output

```
======================================================================
FINE-TUNING WITH LOSS MONITORING
======================================================================

1. Loading model: gpt2
2. Preparing datasets...
   Train batches: 10
   Val batches: 2
3. Starting fine-tuning...

======================================================================
Epoch 1/3
  Train Loss: 2.8934
  Val Loss:   2.7821
  ✓ Validation loss improved! (best: 2.7821)
======================================================================

======================================================================
Epoch 2/3
  Train Loss: 2.1234
  Val Loss:   2.3456
  ⚠ No improvement for 1 epoch(s)
  → Consider stopping early to avoid overfitting
======================================================================

======================================================================
Epoch 3/3
  Train Loss: 1.6789
  Val Loss:   2.8901
  ⚠ No improvement for 2 epoch(s)
  → Consider stopping early to avoid overfitting
======================================================================

TRAINING SUMMARY
======================================================================
Best validation loss: 2.7821 (Epoch 1)
Final training loss:  1.6789
Final validation loss: 2.8901

Gap (Val - Train): 1.2112
  → Positive gap = overfitting (memorizing training data)
```

---

## Key Patterns to Watch

### ✅ Healthy Fine-Tuning

```
Epoch 1: Train 2.89, Val 2.78  ← Both decreasing
Epoch 2: Train 2.12, Val 2.35  ← Both still decreasing
Epoch 3: Train 1.68, Val 2.10  ← Gap widening, stop here
```

**Action:** Stop at Epoch 2. Save the best validation checkpoint.

**Why:** The validation loss is still decreasing in Epoch 2, indicating real learning. By Epoch 3, validation loss increases while training loss continues to decrease — classic sign of overfitting. The model is memorizing the training set rather than learning generalizable patterns.

### ⚠️ Early Stopping (Overfitting)

```
Epoch 1: Train 2.89, Val 2.78  ← Good
Epoch 2: Train 2.12, Val 2.35  ← Still good
Epoch 3: Train 1.68, Val 2.89  ← Val loss jumped! Overfitting.
```

**Action:** Revert to Epoch 2's weights. Don't use Epoch 3.

**Why:** The sharp jump in validation loss indicates the model stopped generalizing and started memorizing. This is the point to stop training.

### ❌ Bad Learning Rate (Too High)

```
Epoch 1: Train NaN, Val NaN  ← Exploded
```

**Action:** Reduce learning rate and restart.

**Why:** With too high a learning rate, gradients become so large that weights diverge, causing numerical instability (NaN = Not a Number).

**Fix:** Try 0.1x or 0.01x the current learning rate.

### ❌ Bad Learning Rate (Too Low)

```
Epoch 1: Train 2.50, Val 2.50
Epoch 2: Train 2.48, Val 2.49  ← Barely improving
Epoch 3: Train 2.46, Val 2.48  ← Too slow
```

**Action:** Increase learning rate.

**Why:** With too low a learning rate, gradient updates are tiny. The model barely learns anything meaningful per epoch.

**Fix:** Try 10x the current learning rate.

### ❌ Stuck / No Convergence

```
Epoch 1: Train 2.50, Val 2.50
Epoch 2: Train 2.50, Val 2.50  ← No change whatsoever
Epoch 3: Train 2.50, Val 2.50
```

**Possible causes:**
- Learning rate is 0 or extremely small
- Gradient is exactly 0 (rare, but happens with dead ReLUs)
- Model is already converged
- Data preprocessing issue (all inputs identical)

**Debug steps:**
1. Print `model.layers[0].weight.grad` — is it actually computing gradients?
2. Check if loss is finite: `print(f"Loss: {loss.item()}")`
3. Visualize a batch: `print(batch['input_ids'][0])`

---

## Exam & Interview Quick Reference

### 🎯 Exam Likely to Ask

**Q: How do you detect if your model is overfitting during fine-tuning?**

**A:** Validation loss increases while training loss decreases. The gap between them grows. This indicates the model is memorizing training data rather than learning generalizable patterns. The standard approach is to use early stopping — save the model at the epoch with the lowest validation loss and stop training at that point, reverting to that checkpoint.

**Q: Why can't you just use training loss to evaluate fine-tuning?**

**A:** Training loss can decrease indefinitely through memorization. A model can achieve zero training loss by simply memorizing each training example, but this provides zero ability to generalize to new, unseen data. Validation loss on held-out data reveals whether the model is truly learning or just overfitting.

**Q: What is the purpose of a validation set?**

**A:** The validation set is a portion of data (typically 10-20% of total) that the model never sees during training. It serves as an unbiased estimate of how well the model will perform on truly new data. By monitoring validation loss, you can detect when the model stops improving on unseen data, indicating overfitting.

**Q: What does `optimizer.zero_grad()` do and why is it necessary?**

**A:** `zero_grad()` clears the gradients from the previous backward pass. Without it, PyTorch accumulates gradients across multiple backward passes (which is occasionally useful but usually a bug). For standard training, you must zero gradients before each backward pass to compute fresh gradients for the current batch.

### 💼 Interview Signal (Strong Answer)

**"I track three key metrics during fine-tuning:**

1. **Training loss per epoch** — smoothed average across all batches to filter noise
2. **Validation loss per epoch** — performance on held-out data the model never sees
3. **The gap between them** — when gap grows large, overfitting is occurring

My workflow:
- Save a checkpoint at the epoch with the lowest validation loss
- Once validation loss stops improving for N epochs (typically 2-3), stop training
- Revert to the best checkpoint and use that for inference

This approach ensures I'm not overfitting. I also monitor learning rate — if training loss doesn't decrease in the first epoch, the learning rate is likely too small; if loss becomes NaN, it's too large."

### ⚠️ Common Traps

**Trap 1:** "If training loss is decreasing, fine-tuning is working"
- **Reality:** Training loss can decrease even while overfitting. Always validate on held-out data.

**Trap 2:** "I should train until training loss reaches 0"
- **Reality:** Zero training loss means memorization, not learning. Stop when validation loss plateaus.

**Trap 3:** "Validation loss fluctuates, so ignore it"
- **Reality:** Some fluctuation is normal (due to batch randomness), but the overall trend matters. Smooth it or look at epoch averages.

**Trap 4:** "The model is overfitting, so I need a bigger dataset"
- **Reality:** Sometimes. But first try: reduce learning rate, add dropout, reduce model size, or use early stopping.

**Trap 5:** "One good epoch means the fine-tuning is successful"
- **Reality:** Run multiple seeds and validate over multiple epochs. One good epoch could be luck.

---

## Additional Advanced Techniques

### Early Stopping Implementation

```python
class EarlyStopping:
    def __init__(self, patience=3, min_delta=1e-4):
        """
        Args:
            patience: Stop training if val loss doesn't improve for this many epochs
            min_delta: Minimum change to count as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
    
    def __call__(self, val_loss):
        """
        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False  # Continue training
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # Stop training
            return False


# Usage in training loop
early_stopping = EarlyStopping(patience=3)

for epoch in range(num_epochs):
    train_loss = train_one_epoch(...)
    val_loss = validate(...)
    
    if early_stopping(val_loss):
        print(f"Early stopping at epoch {epoch}")
        break
```

### Calculating Perplexity (More Intuitive Than Loss)

```python
def calculate_perplexity(loss):
    """
    Perplexity = exp(loss)
    
    Interpretation:
    - Perplexity 100 = model is 100x "confused" compared to random guessing
    - Lower perplexity = better predictions
    - For language models, good perplexity is typically 20-100
    """
    return np.exp(loss)


# During validation
val_loss = np.mean(val_losses)
perplexity = calculate_perplexity(val_loss)
print(f"Validation Perplexity: {perplexity:.2f}")
```

### Learning Rate Scheduling

```python
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

# Reduce learning rate by 0.1 every 2 epochs
scheduler = StepLR(optimizer, step_size=2, gamma=0.1)

# Or use cosine annealing (gradually decrease LR over time)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

for epoch in range(num_epochs):
    train_one_epoch(...)
    validate(...)
    
    scheduler.step()  # Adjust learning rate
    print(f"Current LR: {optimizer.param_groups[0]['lr']}")
```

---

## Summary Table: What to Monitor

| Metric | What to Look For | Action if Problem |
|---|---|---|
| Train Loss per Epoch | Should decrease smoothly | ↓ Learning rate too low, ↑ if NaN |
| Val Loss per Epoch | Should decrease, then plateau | Stop when it plateaus (early stopping) |
| Train-Val Gap | Should stay small (<0.5) | Reduce epochs, add regularization |
| Gradient Magnitude | Should be ~0.001-0.01 | ↓ LR if too large, ↑ if too small |
| Training Speed | Should be consistent | Check GPU/CPU usage |

---

*Guide compiled for Generative AI & ML program — IIT Kharagpur × upGrad*  
*All code is Jupyter-compatible and uses standard ML libraries (PyTorch, HuggingFace, NumPy, Matplotlib)*
