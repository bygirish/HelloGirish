# Section 3 — Additive PEFT: Adapters (Slides 18–32)

> **Goal:** Master the first PEFT family — **adapters**: how they're built, the two properties that make them work, the two main variants (Sequential and Residual/Parallel), and their trade-offs.
> **Demo:** `Residual_Adapter_Demo_1.ipynb`

---

## 3.1 What an adapter is

An **adapter** is a small trainable **bottleneck module** inserted *inside* each transformer layer. The base transformer stays frozen; only adapters are trained.

**The bottleneck block (general form, slide 20):**

```
        x  (dimension d)
        │
   ┌────▼──────────────┐
   │ Down-projection    │   W_down : d → m     (m ≪ d)
   └────┬──────────────┘
        │  (dimension m, the "bottleneck")
   ┌────▼──────────────┐
   │ Non-linearity      │   e.g. ReLU / GeLU
   └────┬──────────────┘
   ┌────▼──────────────┐
   │ Up-projection      │   W_up : m → d
   └────┬──────────────┘
        │
        +  ◄─── residual/skip connection (add input back)
        │
        ▼
     output (dimension d)
```

Formally:  **`h = x + f(x·W_down)·W_up`**

where `W_down ∈ ℝ^{d×m}`, `W_up ∈ ℝ^{m×d}`, `f` is a non-linearity, and `m` (the **bottleneck dimension**) is small (e.g., 16, 32, 64) so the parameter count `≈ 2·d·m` is tiny relative to the layer's `d²`-scale weights.

> 💡 **Learning Thought:** The bottleneck is the whole trick. By squeezing `d → m → d`, an adapter has only `~2dm` parameters instead of `~d²`. With `m ≪ d` this is where "train <2% of params" comes from. The non-linearity is what lets this tiny module still represent a useful task-specific transformation.

---

## 3.2 The two requirements of adapters (Slide 21)

An adapter design must satisfy **two** properties:

### Requirement 1 — A small number of parameters
So that **total model size grows slowly** as you add more tasks. Each new task = one small adapter set, not a full model copy. This is the multi-task storage win from Section 1.

### Requirement 2 — Near-identity initialization
At the **start** of training, the adapter must behave like an **identity function** (output ≈ input), so inserting it **does not disrupt** the pre-trained model's behavior.

**How:** initialize the projection layers (especially `W_up`) to **near-zero**. Then `f(x·W_down)·W_up ≈ 0`, so `h = x + 0 ≈ x`. Training then *gently* moves the adapter away from identity as needed.

> 💡 **Learning Thought:** Near-identity init is the adapter analogue of LoRA initializing `B=0`. Both say: *"start as a no-op so you inherit the frozen model's competence, then learn only the delta the task needs."* If you skip this, early gradients inject noise into a carefully pre-trained network and training becomes unstable. **This is a favorite interview probe.**

---

## 3.3 Adapter deployment patterns (Slides 18–19)

How adapters are *organized* across many tasks:

| Pattern | Idea | Use case |
|---|---|---|
| **Explicit Task ID** | Pick the adapter for the known task ID at inference | You know the task at request time |
| **AdapterFusion** | Learn to **combine** knowledge from multiple task adapters | Transfer/share across related tasks |
| **Multi-Task Adapter** | One adapter trained jointly on many tasks | Shared capacity across tasks |
| **Adapter Routing** | A router **picks/mixes** adapters per input | Unknown task / mixture of tasks |

> 💡 **Learning Thought:** Notice the progression: *fixed selection → learned fusion → learned routing.* This exact progression reappears in Soft Prompting (Section 5) as SMoP's router. **"Route to the right small module"** is a recurring PEFT theme.

---

## 3.4 Variant 1 — Sequential Adapters (Houlsby et al., 2019) (Slides 23–24)

**Placement:** adapters inserted **sequentially** (in series) *twice per transformer layer* — after the multi-head attention sub-layer **and** after the feed-forward sub-layer.

**Parameter accounting (slide 23):**
- Per task, added params ≈ `2 · L · (2·d·m + biases)` (two adapters per layer × L layers).
- Deck's example: full fine-tuning updates the full ~N million param transformer; the adapter updates only **~1%** of the original parameters — same ballpark performance.

**Limitations (slide 24) — memorize these four:**
1. **Inference latency overhead** — adapters sit *in series* on the critical path; every token flows through extra layers sequentially.
2. **Parameter cost scales with depth** — two adapters × every layer; deeper model ⇒ more adapter params.
3. **Cannot be parallelized with existing computation** — because they're sequential, they can't overlap with the attention/FFN compute.
4. **Multi-task serving is expensive** — swapping/holding many task adapters adds serving complexity.

> 💡 **Learning Thought:** Limitation #3 is the seed of the *next* idea. "Sequential ⇒ can't parallelize ⇒ latency" directly motivates **parallel/residual adapters**, which restructure the adapter to run *alongside* existing computation.

---

## 3.5 Variant 2 — Residual / Parallel Adapters (Lin et al., 2020) (Slides 25–29)

**Key structural change:** instead of stacking the adapter *in series*, place it on a **parallel branch** next to the transformer sub-layer, then add its output back via a residual connection.

```
        x
        ├───────────────► Multi-Head Attention / FFN (frozen) ──┐
        │                                                        │
        └───────────────► Adapter (trainable) ──────────────────┤ (+)  ← residual add
                                                                 ▼
                                                          Layer Norm → out
```

**Why "parallel" matters:** the adapter branch can be computed **concurrently** with the frozen sub-layer instead of waiting for it in series. This attacks Sequential Adapters' limitations #1 and #3 (latency + no parallelism).

**Task conditioning via special tokens (slides 27–29):** the residual-adapter work formats each task with **task/segment tokens** so one framework handles multiple generation tasks. Example for **Question Answering**:
```
"qa": ["<bos_qa>", "<document>", "</document>",
       "<question>", "</question>", "<answer>", "</answer>", …, "<eos_qa>"]
```
The deck shows the same recipe applied to **Dialogue Generation** and **Summarization** — a *versatile generative* model where the adapter + task tokens specialize a shared frozen backbone per task.

> 💡 **Learning Thought:** Two independent levers are combined here: (a) *parallel placement* fixes latency; (b) *task-specific tokens* give the adapter an explicit signal of which task it's serving. This is a bridge to Soft Prompting — those `<...>` tokens are a *discrete* precursor to the *continuous* virtual tokens you'll meet in Section 4.

---

## 3.6 Other interesting ideas (Slide 30)

- **AdapterFusion** — a two-stage scheme: first train task adapters independently; then freeze them and learn a **fusion** layer (an attention-like combiner) that mixes their knowledge for a target task. Non-destructive transfer learning — combine tasks *without* catastrophic forgetting.
- **Tiny-Attention Adapter** — use a very small **attention** module as the adapter instead of a plain bottleneck MLP, capturing token interactions with minimal params.

---

## 3.7 "In a Nutshell" (Slide 31) + Food for Thought (Slide 32)

**Nutshell:**
- Fine-tuning adapts pre-trained models but has **two issues**: it's costly, and it causes catastrophic forgetting in multi-task settings.
- **Adapters** match full-FT-level performance while tuning **only ~2%** of parameters.
- **Task-specific adapters solve catastrophic forgetting** (base frozen, tasks isolated).
- Recipe: **freeze original params; update only adapter params.**

**Food for Thought (slide 32):**
> *Why use adapters instead of just fine-tuning the last few (penultimate) layers of the model?*

**Model answer:** Fine-tuning only the last few layers (a) still updates *millions* of large `d²`-scale weights (far more than a bottleneck adapter's `2dm`), (b) offers **no multi-task modularity** — you'd need a separate copy of those layers per task and they still catastrophically forget when reused, and (c) limits adaptation to *high* layers only, whereas adapters inject task capacity at **every** layer (bottom layers carry useful low-level features too — a point Section 5's APT exploits). Adapters give more expressive, more parameter-efficient, and more modular adaptation.

> 💡 **Learning Thought:** This question tests whether you understand that "few trainable params" and "few *trained layers*" are different things. A single unfrozen `d×d` layer can have *more* parameters than adapters spread across all layers — and gives you none of the modularity.

---

## 🧪 Tie-in: `Residual_Adapter_Demo_1.ipynb`

When you run the demo, watch for:
1. **Where** the adapter is inserted (parallel branch + residual add).
2. **`W_up` initialized ~0** ⇒ verify the model output is unchanged *before* training (identity check).
3. **Trainable-param count** printed vs. total — confirm the "~1–2%" claim yourself.
4. **Only adapter params** appear in the optimizer (base frozen: `requires_grad=False`).

---

## 🎯 Interview Questions — Section 3

**Q1. Draw/describe the adapter bottleneck and give its parameter count.**
*A:* Down-projection `d→m`, non-linearity, up-projection `m→d`, plus a residual skip: `h = x + f(x·W_down)·W_up`. Params ≈ `2·d·m` (+biases), tiny when `m ≪ d`.

**Q2. Why must adapters be initialized to near-identity, and how is that achieved?**
*A:* So inserting them doesn't disrupt the pre-trained model at the start of training (stability + inherit base competence). Achieved by initializing the up-projection (`W_up`) to ~0, making the adapter's contribution ≈0, so `h ≈ x` initially; training then learns the needed delta.

**Q3. List the limitations of Sequential Adapters.**
*A:* (1) Inference latency (in-series on critical path); (2) parameter cost scales with depth (two per layer); (3) can't be parallelized with existing compute; (4) expensive multi-task serving.

**Q4. How do Residual/Parallel Adapters improve on Sequential ones?**
*A:* They place the adapter on a **parallel branch** added back via residual, so it can be computed concurrently with the frozen sub-layer — reducing the latency and the "can't parallelize" problems of the sequential design.

**Q5. What is AdapterFusion and what problem does it solve?**
*A:* A two-stage method: train per-task adapters, then freeze them and learn a fusion (attention-style) combiner over their outputs for a target task. It enables **compositional transfer** across tasks *without* catastrophic forgetting, since the source adapters are frozen.

**Q6. Adapters vs. fine-tuning only the last few layers — which is better and why?**
*A:* Adapters: fewer params (bottleneck `2dm` vs full `d²` layers), modular/swappable per task, no forgetting, and they inject capacity at *every* depth (bottom layers matter). Tuning last layers updates more params, isn't modular, and only adapts high layers.

**Q7 (deep). Why do adapters add inference latency while LoRA doesn't?**
*A:* Adapters are *extra layers* that remain in the forward pass at serve time — they can't be folded into existing weights because of the non-linearity. LoRA's `ΔW = BA` is linear and can be **merged** into `W` (`W' = W + BA`), so after merging there's no extra compute. The non-linearity is precisely what blocks adapter merging.

**Q8 (deep). You must serve 50 tasks from one 7B model with tight latency. Adapters or LoRA? Justify.**
*A:* If tasks are known and swapped per request, **LoRA** merged per task gives zero added latency but merging is per-checkpoint. For *simultaneous* multi-task with hot-swap and composition, **adapters/AdapterFusion** are more modular but add latency. A common production answer: LoRA with fast adapter-swapping (e.g., serving multiple LoRA deltas unmerged) to balance modularity and latency — showing you understand the latency-vs-modularity trade-off is the point.

---

## ✅ Section 3 takeaways

- Adapter = **bottleneck** (`d→m→d` + non-linearity + residual), `~2dm` params.
- Two requirements: **few params** + **near-identity init** (`W_up ≈ 0`).
- **Sequential** (Houlsby): in-series, simple, but latency + can't parallelize.
- **Residual/Parallel** (Lin): parallel branch fixes latency; task tokens enable multi-task generation.
- Adapters solve **cost** and **catastrophic forgetting**, tuning only ~2% of params.
- Can't be merged (non-linearity) ⇒ permanent inference cost, unlike LoRA.

➡️ **Next:** [Section 4 — Soft Prompting foundations](04_Section4_SoftPrompt_Foundations.md): adaptation moves from architecture space to **token space**.
