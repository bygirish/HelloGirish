# Section 7 — Reading the Loss Curves (Slides 67–69)

> **Why this matters:** This is the capstone — *"Your Ultimate Guide."* Every earlier section
> gave you a knob; this section teaches you to **read the one instrument** that tells you which
> knob to turn. The relationship between **training loss** and **validation loss** over steps is
> the single richest diagnostic in all of fine-tuning. Learn to read these two curves and you
> can debug almost any run at a glance.

---

## 7.1 The two curves and the three failure modes (Slide 68)

You plot **training loss** vs. **validation loss** across fine-tuning steps. Their *relationship*
tells you which failure mode you're in.

| Mode | Also called | Signal on the curves |
|------|-------------|----------------------|
| **Underfitting** | High **bias** | Both losses stay **high** and plateau early |
| **Overfitting** | High **variance** | Train loss → 0 while **val loss turns back up** |
| **No Learning** | — | Both losses **flat at the initial value** — nothing updates |

The mental decoder:
- **Both high & flat** → the model isn't capable/trained enough yet → *underfitting.*
- **Train great, val worsening** → the model is memorizing the training set → *overfitting.*
- **Both frozen at start value** → gradients aren't flowing at all → *no learning* (a bug).

> 💡 **Learning Thought — the bias/variance lens:**
> - **Underfitting = high bias:** the model is too simple / under-trained to capture the signal.
>   Fix by training *more* / bigger LR / more capacity / less regularization.
> - **Overfitting = high variance:** the model captures noise specific to the training set. Fix
>   with *more/better data*, regularization, early stopping, or *less* training.
> The train–val **gap** is your variance meter; the *absolute level* of the training loss is your
> bias meter. Read both dimensions at once.

### The "No Learning" mode is special — it's a *bug*, not a tuning issue
Flat-at-initial curves mean **nothing is updating.** Usual culprits: LR is effectively zero,
frozen/`requires_grad=False` on everything, a broken data pipeline feeding no signal, gradients
not connected, or `loss.backward()`/`optimizer.step()` missing. Don't tune hyperparameters —
**find the bug.**

---

## 7.2 What good learning looks like (Slide 69)

The target picture — **both losses fall together with a small, stable gap; stop at the
validation-loss minimum.** Four hallmarks of a healthy fine-tune:

1. **Curves move together** — validation loss follows training loss *down.* The model
   **generalizes, not memorizes.**
2. **Small, stable gap** — a modest train–val gap that *stops widening* is normal and healthy. (A
   gap isn't bad; a *growing* gap is.)
3. **Diminishing returns** — loss flattens *smoothly*; no oscillation, spikes, or divergence.
4. **Stop at the val minimum** — **checkpoint where validation loss bottoms out; training longer
   buys nothing** (and risks overfitting/forgetting).

> 💡 **Learning Thought — the golden stopping rule:** **Stop at the validation-loss minimum, not
> the training-loss minimum.** Training loss almost always keeps falling (the model can always
> memorize more); validation loss is the honest signal of generalization. The point where val
> loss bottoms and starts to rise is precisely where memorization overtakes learning — that
> checkpoint is your model. This is what `load_best_model_at_end=True` automates.

---

## 7.3 The unified diagnostic table (synthesizing the whole lecture)

This is the payoff — read the curve, name the mode, turn the right knob (with the section it
came from):

| What you see | Mode | Likely cause | Fix (→ section) |
|--------------|------|--------------|-----------------|
| Both losses high, flat early | Underfitting (bias) | Too few steps, LR too low, batch too large, model frozen too much | Train longer / raise LR (§4) / shrink batch for more steps (§5) / unfreeze more (§3) |
| Both losses flat at *initial* value | No learning (bug) | LR≈0, everything frozen, broken data/grad flow | Debug the pipeline — not a tuning problem |
| Train↓ to ~0, val↑ back up | Overfitting (variance) | Too many epochs, dataset too small/undiverse | Early-stop at val min (§7), more/better data (§6), regularize (§3) |
| Sudden vertical jump then recovery | Loss spike | Outlier batch / peak LR | Gradient clipping, warmup (§2) |
| Grad-norm ↑↑, loss rising, NaN | Explosion → divergence | LR too high, fp16 overflow | Lower peak LR, bf16, clip, restore checkpoint (§2) |
| Persistent jitter that never shrinks | Noisy gradients | Batch too small / LR too high | ↑ effective batch (§5), ↓ LR (§4) |
| Target metric ↑ but general evals ↓ | Catastrophic forgetting | Narrow data, high LR, full FT | Rehearsal, freeze, regularize, conservative HPs (§3) |
| Both fall together, small stable gap, flatten | **Healthy** ✓ | — | Stop at val minimum, ship it |

> 💡 **Learning Thought — this table IS the course.** Notice how *every* section resolves to a
> pattern in these two curves. Sections 1–6 taught you the mechanisms and knobs; Section 7 is the
> feedback loop that tells you *which* knob. In practice you'll spend most of your fine-tuning
> life staring at exactly this plot and mapping what you see back to these fixes.

---

## 🎯 Interview Questions

**Q1. You have training and validation loss curves. How do you diagnose the run?**
> Read two dimensions. Absolute training-loss level = bias: if both curves are high and flat →
> underfitting. Train–val gap = variance: if train→0 while val rises → overfitting. If both are
> frozen at their initial value → nothing is learning (a pipeline/config bug). Healthy = both
> fall together with a small, stable gap.

**Q2. Distinguish underfitting, overfitting, and "no learning" from the curves.**
> Underfitting: both losses high, plateau early (high bias). Overfitting: train loss keeps
> dropping while val loss turns upward (high variance / memorization). No learning: both curves
> flat at the *initial* value — gradients aren't updating the weights at all.

**Q3. Why stop at the validation-loss minimum instead of training longer?**
> Training loss keeps falling because the model can always memorize more, but generalization is
> measured by validation loss. Once val loss bottoms and rises, further training only memorizes —
> it buys nothing on held-out data and risks overfitting and forgetting. Checkpoint at the val
> minimum (`load_best_model_at_end=True`).

**Q4. Is a gap between train and validation loss always bad?**
> No. A modest, *stable* gap is normal and healthy. What matters is whether it's *widening* — a
> growing gap signals overfitting; a small constant gap just reflects that the model does slightly
> better on data it's seen.

**Q5. Your curves are perfectly flat at the starting loss from step 0. What do you check?**
> This is "no learning" — a bug, not a hyperparameter. Check: LR isn't effectively zero, params
> aren't all frozen (`requires_grad`), the data loader is actually yielding labeled signal, the
> loss is connected to the graph, and `loss.backward()` / `optimizer.step()` are being called.

**Q6. Map bias/variance to fixes.**
> High bias (underfit): train more, raise LR, add capacity, unfreeze layers, reduce
> regularization, shrink batch for more update steps. High variance (overfit): get more/better/
> more diverse data, regularize, early-stop, reduce epochs.

**Q7. (Senior) Target-task accuracy is climbing beautifully but you're worried. What curve do the
loss plots *not* show, and what do you add?**
> The train/val loss on the *target task* won't reveal **catastrophic forgetting** of general
> abilities. Add a fixed general "canary" eval suite (reasoning, code, instruction-following,
> multilingual) evaluated at every checkpoint, and early-stop on *that*, not on target loss — so
> you catch general-capability collapse that the target curves hide.

---

## One-line takeaway

**Train loss reads *bias* (absolute level), the train–val gap reads *variance* (widening =
overfit), both-flat-at-start means a *bug* — so watch two curves, and stop exactly at the
validation-loss minimum, because that's where learning ends and memorizing begins.**

---

## 🏁 Course wrap-up — the whole lecture in one breath

> Fine-tuning takes small, averaged, downhill steps on cross-entropy loss (**§1**). Keep each
> step from blowing up with clipping, warmup, bf16, and a conservative peak LR (**§2**). Keep the
> weights from drifting so far they forget everything else — measure a general eval, and use
> rehearsal/freezing/regularization/merging (**§3**). Shape the step *size* over time with a
> schedule — cosine + warmup by default (**§4**). Shape the step *noise* with an effective batch
> assembled from micro-batch × accumulation × GPUs, scaling LR with it (**§5**). Point the whole
> descent somewhere worth going by cleaning, deduplicating, balancing, and leak-free splitting the
> data (**§6**). And read it all off two loss curves, stopping at the validation minimum (**§7**).**
