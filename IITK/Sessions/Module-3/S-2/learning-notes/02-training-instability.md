# Section 2 — Training Instability (Slides 14–21)

> **Why this matters:** Fine-tuning a pretrained model is like doing surgery on a
> delicately balanced system. One oversized step can undo hours of training. This section
> is the *pathology* (how runs die) followed by the *treatment* (a stability checklist you
> can apply on every run).

---

## 2.1 Why fine-tuning becomes unstable (Slide 15)

The root feedback loop: **one oversized update damages the weights → produces worse
predictions → larger loss → even bigger gradients on the next step.** Left unchecked, this
runs away. Five forces that trigger or amplify it:

1. **Sharp loss landscapes.** Pretrained optima sit in *narrow valleys*. A step sized for
   gentle terrain overshoots the valley walls.
2. **Low-precision arithmetic.** fp16 overflows above ~65,504. One large activation becomes
   `Inf`, which poisons the gradient for every downstream parameter.
3. **Heterogeneous batches.** A single outlier batch — extreme length, corrupted text,
   unusual tokens — can produce a gradient **10–100× the typical norm**.
4. **Scale amplifies everything.** Deeper networks multiply perturbations layer by layer, so
   a small early error compounds into a large one by the output.
5. (The **feedback loop** itself from the top — bad update begets a worse one.)

> 💡 **Learning Thought:** Instability is almost always a **step-size problem in
> disguise.** Sharp valleys, outlier batches, and fp16 overflow all end the same way: an
> update far too large for where the model currently sits. Keep that framing — it makes
> the mitigations obvious.

---

## 2.2 The three failure modes (Slides 16–18)

These are three points on one spectrum — from recoverable to fatal.

### (a) Loss Spike (Slide 16) — recoverable
- **What happens:** One batch produced an outsized gradient; the update knocked weights
  off the low-loss valley. Loss jumps, then **recovers** as later steps repair the damage.
- **Cost:** Each spike still wastes hundreds of steps.
- **Where:** Spikes cluster **where LR is highest — just after warmup.**

### (b) Gradient Explosion (Slide 17) — the danger zone
- **What happens:** The feedback loop in action: *bad update → worse loss → bigger
  gradient → worse update.* Gradient **norms grow exponentially over tens of steps.**
- **Key insight:** This is the phase **before** loss visibly diverges — the gradient norm
  spikes *first*, which is why you monitor it as an early warning.
- **In fp16:** ends in overflow → gradients become `Inf`, then `NaN`.

### (c) Divergence (Slide 18) — terminal
- **What happens:** The terminal state. Weights have left every good region; updates no
  longer point downhill. Unlike a spike, **loss keeps rising (or flatlines at
  random-guess level) with no self-repair.**
- **Symptoms:** Model outputs degenerate — gibberish, repetition, empty strings.

> 💡 **Learning Thought:** The ordering is your diagnostic clock:
> **grad-norm explodes → loss spikes → NaN → divergence.** Monitoring the *gradient norm*
> buys you the earliest warning, often tens of steps before the loss looks wrong. By the
> time you see `NaN` in the loss, you're already too late — restore from a checkpoint.

---

## 2.3 Mitigation #1: Gradient Clipping (Slide 19)

**Rescale any gradient whose norm exceeds a cap — direction preserved, magnitude
bounded.** This directly defuses the outlier-batch and explosion problems.

```python
# Manual training loop
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()

# HF Trainer: one argument
TrainingArguments(
    max_grad_norm=1.0,   # default — keep it
)
```

**Tuning the cap:**
- `1.0` is the standard default for LLM fine-tuning.
- **Log the pre-clip norm.** If **>20% of steps are clipping, your LR is too high** —
  clipping is a *seatbelt, not a steering wheel.*

> 💡 **Learning Thought:** Clipping bounds the *worst case* but doesn't fix a systematically
> too-large LR. If you're clipping constantly, you're driving with the handbrake on — lower
> the learning rate instead.

---

## 2.4 Mitigation #2: Learning-Rate Scheduling (Slide 20)

The most powerful single lever (and the subject of all of §4). Preview here:

```python
TrainingArguments(
    learning_rate=2e-5,
    lr_scheduler_type="cosine",   # smooth decay to ~0 at run end
    warmup_ratio=0.03,            # ramp up from 0 to avoid step-zero spikes
    num_train_epochs=2,
)
# linear = same idea, straight line;  constant = only for short LoRA runs
```

**Real example from the slide:** Legal-domain SFT at a *constant* 2e-5 spiked repeatedly in
the last 20% of training. Switching to **cosine** (identical *peak* LR) removed every
late spike **and** improved final validation loss by **4%**.

> 💡 **Learning Thought:** The valley narrows as training progresses. A step size that was
> safe at 30% of the run overshoots at 90%. Decaying the LR isn't just about final polish —
> it's about *keeping the step proportional to how sharp the local landscape has become.*

---

## 2.5 The Stability Checklist That Compounds (Slide 21)

No single setting saves you — they **stack**. This is the money slide of the section:

| Layer | Setting | What it prevents |
|-------|---------|------------------|
| **Clipping** | `max_grad_norm=1.0` | Loss spikes; starves the explosion loop |
| **Schedule** | `lr_scheduler_type="cosine"` | Late-run spikes as the valley narrows |
| **Warmup** | `warmup_ratio=0.03` | Step-zero spikes and instant divergence |
| **Precision** | `bf16=True` (not fp16) | Overflow → Inf/NaN cascade |
| **Peak LR** | `1e-5–2e-5` full FT / `1e-4–2e-4` LoRA | Overshooting sharp valley walls |
| **Monitoring** | log grad-norm; alert at 5–10× median | Catching explosion before divergence |
| **Checkpoints** | save every N steps; keep last 3 | Losing the run when divergence strikes |

> 💡 **Learning Thought — the bf16 vs fp16 point is a favorite interview trap.** Both are
> 16-bit. But **bf16** keeps fp32's *exponent range* (8 bits) while sacrificing mantissa
> precision, so it **can't overflow** at 65,504 the way fp16 does — it trades numerical
> *precision* for numerical *range*, and range is what prevents the Inf/NaN cascade. On
> hardware that supports it (Ampere+), **always prefer bf16 for training.**

---

## 🎯 Interview Questions

**Q1. Describe the feedback loop that causes training to diverge.**
> An oversized update damages the weights → predictions worsen → loss rises → gradients
> grow larger → the next update is even more oversized. Without intervention (clipping, LR
> decay) this compounds exponentially into gradient explosion and then divergence.

**Q2. What's the difference between a loss spike, gradient explosion, and divergence?**
> A **spike** is a single bad step that self-repairs over subsequent steps. **Explosion** is
> the gradient norm growing exponentially over tens of steps — the pre-divergence phase.
> **Divergence** is terminal: weights have left all good regions, loss keeps rising with no
> recovery, outputs degenerate. They're three points on one severity spectrum.

**Q3. Why do spikes tend to occur right after warmup?**
> That's where the LR is at its peak. Warmup ramps LR up; immediately after, step sizes are
> largest, so an outlier batch is most able to knock weights off the valley.

**Q4. How does gradient clipping work, and what does it *not* fix?**
> If the global gradient norm exceeds a cap (e.g., 1.0), it's rescaled down to the cap,
> preserving direction. It bounds outlier batches. It does **not** fix a systematically too-
> high LR — if you clip on >20% of steps, lower the LR; clipping is a seatbelt, not steering.

**Q5. Why prefer bf16 over fp16 for fine-tuning?**
> fp16 overflows above ~65,504; one large activation becomes Inf and poisons gradients.
> bf16 has the same exponent range as fp32, so it doesn't overflow — it trades mantissa
> precision for dynamic range, which is exactly what prevents the Inf/NaN cascade.

**Q6. Which metric gives the earliest warning of instability, and why?**
> The **gradient norm**. It grows exponentially during the explosion phase *before* the loss
> visibly diverges, so alerting at ~5–10× its running median catches trouble tens of steps
> early — well before you'd see NaN in the loss.

**Q7. (Senior) Why does a cosine schedule with the *same peak LR* as constant improve
stability and final loss?**
> Pretrained optima are sharp valleys that narrow as training proceeds. A constant LR keeps
> full-size steps even late, causing overshoot spikes near convergence. Cosine shrinks the
> step in step with the narrowing valley, removing late spikes and letting the model settle
> deeper into the minimum — better final loss at identical peak LR.

**Q8. (Senior) You hit a NaN loss at step 8,000. What's your recovery playbook?**
> Restore the last good checkpoint (before the grad-norm blew up), then lower peak LR and/or
> lengthen warmup, confirm bf16 (not fp16), verify `max_grad_norm` is set, and inspect the
> batches around the failure for corrupted/extreme-length outliers. Resume from checkpoint,
> not from scratch.

---

## One-line takeaway

**Instability is a step-size problem: the fix is a *compounding stack* — clip the outliers,
warm up then decay the LR, train in bf16, keep the peak LR conservative, monitor the
gradient norm as an early alarm, and checkpoint so a blow-up costs minutes, not the run.**
