# Section 4 — Learning-Rate Scheduling (Slides 32–45)

> **Why this matters:** The lecture literally calls this *"a panacea for many troubles."*
> The learning rate is the **single most important hyperparameter in fine-tuning**, and how
> it *changes over the run* fixes instability, improves final loss, and adapts to the phases
> of training. If you master one knob, master this one.

---

## 4.1 The core dilemma: no single LR is right all run (Slide 33)

$$\theta \leftarrow \theta - \eta \cdot \frac{\partial L}{\partial \theta}$$

- **High LR:** learns fast early, but then *bounces around the minimum forever* — a noisy
  plateau that's spike-prone.
- **Low LR:** perfectly stable, but *still far from converged* when the compute budget runs out.
- **You need both:** big steps early, small steps later.

That "big early, small late" requirement is *exactly* what a schedule delivers.

---

## 4.2 Training has phases — the LR should track them (Slide 35)

| Phase | When | What's happening | LR behavior |
|-------|------|------------------|-------------|
| **1 · Fragile start** | 0 – ~3% | New data distribution *shocks* the pretrained weights; full-size steps cause instant spikes | LR **ramps up from 0** (warmup) |
| **2 · Rapid learning** | ~3% – ~60% | Gradients are large, consistent, informative; model crosses easy terrain — *most of the task is learned here* | LR sits at **peak** |
| **3 · Convergence** | ~60% – 100% | Model sits inside a *narrowing valley*; safe step size shrinks with progress | LR **decays** toward floor |

The LR-vs-progress arc for the default **cosine + warmup**, in ASCII:

```
LR
peak ┤        ______
     │      /        \___
     │    /              \___
     │  /                    \____
     │ /                          \____
   0 ┼/─────────────────────────────────\___  → training progress
     0%   3%        ~60%                 100%
     └warmup┘└──── peak (rapid learning)────┘└─decay─┘
       ↑ fragile start        ↑ productive middle   ↑ narrowing valley
```

> 💡 **Learning Thought:** A schedule is not arbitrary curve-fitting — it's *matching the step
> size to the physics of each phase.* Warmup protects the fragile start, the peak exploits the
> productive middle, and decay respects the narrowing valley at the end. Every schedule shape
> is just a different way to trace this arc.

---

## 4.3 Anatomy: four numbers define ANY schedule (Slide 36)

Memorize these four — every schedule is a choice of them:

1. **Warmup length** — how long LR ramps 0 → peak. Default **3%** of steps; raise to **5–10%**
   for short or unstable runs.
2. **Peak LR** — the maximum. *The single most important hyperparameter.*
   **1e-5–2e-5 full FT, 1e-4–2e-4 LoRA.**
3. **Decay shape** — the path down from peak: linear, cosine, step, or restarts. *This is where
   the named strategies differ.*
4. **Floor (min LR)** — where decay ends: **0**, or **~10% of peak** to keep learning alive for
   continual training.

> 💡 **Learning Thought:** When someone says "I used a cosine schedule," they've only told you
> *one* of these four numbers. Always pin down all four — warmup, peak, shape, floor — or you
> can't reproduce or debug a run.

---

## 4.4 Warm-up (Slide 37)

During the first *k* steps, the LR **increases linearly from 0 (or a very small value) to the
maximum.** This is the antidote to the "fragile start" spikes from §2 — you don't hit the
pretrained weights with a full-size step on the very first, highest-noise batch.

---

## 4.5 The six decay shapes (Slides 38–44)

### (1) Constant (± Warmup) (Slide 38)
The simplest policy: one LR for the whole run (ideally after a short warmup). **No decay means
full-size steps even at convergence** — fine for short runs, spike-prone in long ones.
```python
TrainingArguments(lr_scheduler_type="constant_with_warmup", learning_rate=1e-4, warmup_ratio=0.03)
# plain "constant" skips even warmup
```

### (2) Linear Decay (Slide 39)
After warmup, LR falls in a **straight line to zero** at the final step. The no-surprises default.
```python
TrainingArguments(lr_scheduler_type="linear", learning_rate=2e-5, warmup_ratio=0.03, num_train_epochs=2)
```

### (3) Cosine Decay — the modern default (Slide 40)
LR follows a **half-cosine**: lingers near the peak early (more time at productive step sizes),
then glides gently into the floor (extra-fine steps at the very end). **The standard choice for
LLM pretraining and fine-tuning alike.**
```python
TrainingArguments(lr_scheduler_type="cosine", learning_rate=2e-5, warmup_ratio=0.03)
# "cosine_with_min_lr" adds a floor
```

### (4) Step Decay (Slide 41)
LR stays flat, then **drops by a factor (e.g., ×0.5) at chosen milestones** — a staircase.
Inherited from computer vision; its main *modern* form is **reactive** — drop LR when validation
loss plateaus.
```python
sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[2000, 3500], gamma=0.5)  # fixed
sched = ReduceLROnPlateau(opt, factor=0.5, patience=3)                                   # reactive
```

### (5) [Cosine with Restarts / Cyclical](https://arxiv.org/abs/1608.03983) (Slide 42)
Cosine decay that **periodically snaps back to the peak** and decays again. Each restart *kicks
the model out of its current basin* — a cheap way to explore several solutions in one run.
Introduced as **SGDR** (Loshchilov & Hutter); closely related to
[Cyclical Learning Rates](https://arxiv.org/abs/1506.01186) (Leslie Smith).
```python
TrainingArguments(lr_scheduler_type="cosine_with_restarts", learning_rate=2e-5, warmup_ratio=0.02)
# cycles via lr_scheduler_kwargs={"num_cycles": 3}
```

### (6) [Warmup–Stable–Decay / WSD](https://arxiv.org/abs/2404.06395) (Slide 43)
A **trapezoid**: warm up, *hold the peak for most of the run*, then decay sharply in the final
~10–20%. Its superpower: you can **decay on demand from any point** — perfect when you don't
know the run length up front. Popularized by the **MiniCPM** report as a way to train
open-ended without committing to a total step count.
```python
TrainingArguments(lr_scheduler_type="warmup_stable_decay", learning_rate=2e-5,
                  warmup_ratio=0.03, lr_scheduler_kwargs={"num_decay_steps": 500})
```

---

## 4.6 Six shapes at a glance (Slide 44)

| Shape | Use it for |
|-------|-----------|
| **Constant + Warmup** | short runs only |
| **Linear** | no-surprises default |
| **Cosine** | the modern default |
| **Step Decay** | val-driven drops |
| **Cosine Restarts** | multi-solution runs |
| **WSD** | open-ended budgets |

---

## 🧪 Comparing schedules head-to-head in Demo 1 (`LW-BW-P-FT-and-LRs`)

Demo 1's whole point is to train the **same model, same data, same peak LR** under three
different schedules and plot the difference. The mechanism is just swapping one string in
`TrainingArguments` — the HF `Trainer` builds the schedule for you:

```python
SCHEDULERS_TO_COMPARE = ["linear", "cosine", "constant_with_warmup"]
WARMUP_RATIO = 0.1            # identical warmup across all three, so only the DECAY differs

# warmup is specified in *steps*; derive it from the total-step estimate
total_steps  = approx_steps_per_epoch * EPOCHS
warmup_steps = max(1, int(total_steps * WARMUP_RATIO))

for scheduler_name in SCHEDULERS_TO_COMPARE:
    training_args = TrainingArguments(
        learning_rate = LR,                     # SAME peak LR (2e-5) for every run
        lr_scheduler_type = scheduler_name,     # ← the only thing that changes
        warmup_steps = warmup_steps,
        num_train_epochs = EPOCHS,
        eval_strategy = "epoch",
        ...
    )
    trainer = MultiTaskTrainer(model=fresh_model, args=training_args, ...)
    trainer.train()
    # log_history now holds per-step 'loss' and per-epoch 'eval_loss' to plot (see §7)
```

> 💡 **Learning Thought:** This is the *cleanest possible controlled experiment* — hold
> everything fixed and vary a single knob. Because warmup and peak LR are identical, any
> difference in the loss curves is attributable **purely to the decay shape**. That's exactly
> the discipline you want when tuning schedules on your own runs: change one thing, plot,
> compare. (Demo 1 saves the loss/val-loss curves — those plots belong to §7.)

Under the hood, `Trainer` calls
[`transformers.get_scheduler`](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules)
— you can use the same schedules in a manual loop:

```python
from transformers import get_scheduler
scheduler = get_scheduler(
    name="cosine", optimizer=optimizer,
    num_warmup_steps=warmup_steps, num_training_steps=total_steps,
)
# inside the loop, after optimizer.step():
scheduler.step()     # advance the LR one step along the schedule
```

---

## 4.7 Match the schedule to the run (Slide 45) — the decision table

| Your situation | Schedule | Why |
|----------------|----------|-----|
| Standard SFT, known length | **Cosine (or linear) + 3% warmup** | Best default; tail stabilizes convergence |
| Run length unknown / continual training | **WSD** | Decay on demand from any point in the stable phase |
| Data arrives in waves; val-driven control | **Step decay (ReduceLROnPlateau)** | Drops react to plateaus, not the calendar |
| Want several candidate models in one budget | **Cosine with restarts** | Each cycle yields a checkpoint at an LR minimum |
| Instability at any point | **Lengthen warmup, then lower peak** | *Shape tweaks beat shape swaps for stability* |

> 💡 **Learning Thought — the most practical line in the section:** *"Shape tweaks beat shape
> swaps for stability."* When a run is unstable, don't reach for an exotic schedule — first
> **lengthen the warmup**, then **lower the peak LR.** Those two adjustments fix the vast
> majority of instability, whereas swapping cosine→WSD rarely does.

---

## 🎯 Interview Questions

**Q1. Why not just pick one good learning rate and keep it constant?**
> No single value serves the whole run. A high constant LR learns fast but bounces around the
> minimum and is spike-prone; a low one is stable but under-converges within budget. You need
> big steps early and small steps late — which requires a schedule.

**Q2. What is warmup and what problem does it solve?**
> Warmup linearly ramps the LR from ~0 up to peak over the first few percent of steps. It
> protects the *fragile start*: the pretrained weights are shocked by the new data
> distribution, and a full-size step on the first, noisiest batches causes instant spikes or
> divergence. Ramping up avoids that.

**Q3. Name the four numbers that fully specify a schedule.**
> Warmup length, peak LR, decay shape, and floor (min LR). Everything else is a specific choice
> of these four.

**Q4. Why is cosine the modern default over linear?**
> Cosine lingers near the peak early (more time at productive step sizes) and then decays
> gently into a very fine tail, letting the model settle deep into the minimum. It empirically
> gives smoother convergence and slightly better final loss than a straight linear ramp-down,
> at no extra cost.

**Q5. What are typical peak LRs for full fine-tuning vs. LoRA, and why do they differ?**
> ~1e-5–2e-5 for full FT, ~1e-4–2e-4 for LoRA — roughly 10× higher for LoRA. LoRA updates only
> a tiny set of low-rank adapter parameters (base weights frozen), so larger steps are safe and
> necessary to move the effective function meaningfully.

**Q6. When would you choose WSD over cosine?**
> When the run length is unknown or training is continual/open-ended. WSD holds the peak
> through a stable phase and can decay *on demand* from any point, so you don't need to commit
> to a total step count in advance (cosine's shape depends on knowing the endpoint).

**Q7. What does cosine-with-restarts buy you?**
> Each restart snaps the LR back to peak, kicking the model out of its current basin to explore
> a different solution; and each cycle *ends* at an LR minimum, giving you a good checkpoint per
> cycle — several candidate models from one training budget.

**Q8. (Senior) A run is stable early but spikes in the last 20%. Fixes?**
> The valley narrows late while a constant/too-flat LR keeps steps too large. Switch to a
> decaying schedule (cosine) or, if already decaying, *lengthen warmup and lower the peak* —
> shape tweaks beat shape swaps. This mirrors the legal-SFT example where constant→cosine at
> the same peak removed every late spike and improved val loss ~4%.

---

## One-line takeaway

**A learning-rate schedule matches step size to the three phases of training — warm up through
the fragile start, hold the peak through rapid learning, decay into the narrowing valley — and
when in doubt use *cosine + 3% warmup*, fixing instability by lengthening warmup and lowering
the peak before ever swapping the shape.**

---

## 🔗 Further reading

- **[HF — Optimization / schedules](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules)**:
  the exact `lr_scheduler_type` strings and `get_scheduler` API used in Demo 1. Visualize any
  of them with the [PyTorch LR scheduler docs](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate).
- **Cosine restarts:** [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983)
  · **Cyclical LR:** [Cyclical Learning Rates for Training NNs](https://arxiv.org/abs/1506.01186) (Leslie Smith).
- **WSD:** [MiniCPM](https://arxiv.org/abs/2404.06395) introduces Warmup-Stable-Decay and shows
  why it enables open-ended / continual training.
- **Why warmup at all?** [On the Variance of the Adaptive Learning Rate and Beyond (RAdam)](https://arxiv.org/abs/1908.03265)
  and [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)
  explain warmup's role in stabilizing early Adam updates.
- **Finding the peak LR:** [the LR range test](https://arxiv.org/abs/1506.01186) and
  [fast.ai's `lr_find`](https://docs.fast.ai/callback.schedule.html#learner.lr_find) — a
  practical way to pick the single most important hyperparameter.
- **Visual intuition:** [Distill.pub — "Why Momentum Really Works"](https://distill.pub/2017/momentum/)
  builds the geometry (valleys, step size) that makes decay schedules make sense.
