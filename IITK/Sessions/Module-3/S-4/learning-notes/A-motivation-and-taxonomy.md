# Section A — Motivation & Taxonomy
### Slides 1–8 · Why re-parameterization is its own category

> **Objective:** understand *what problem class* LoRA belongs to, and see the
> industrial pressure that made it inevitable.

---

## A1. Recap — where we came from (slides 3, 5)

The previous session established PEFT and covered three of its four branches.
The premise of all PEFT: **fine-tuning every parameter of a large model is
economically absurd**, and mostly unnecessary. In the last session we were
already training only 1–2% of parameters.

### The full PEFT taxonomy

```
                        PEFT
        (efficient alternative to Full Fine-Tuning)
                          │
   ┌──────────┬───────────┼───────────────┬─────────────┐
   │          │           │               │             │
Additive  Soft Prompting  Selective   Re-Parameterization
   │          │              │               │
Adapters   Prefix Tuning   BitFit          LoRA
Adapter    SMoP            Diff Pruning    QLoRA
Fusion                                     ← TODAY
```

### The four branches, by *mechanism*

| Branch | What it does to the network | Example | Inference cost |
|---|---|---|---|
| **Additive** | Inserts new trainable modules between existing layers | Adapters, AdapterFusion | Extra layers → **added latency** |
| **Soft Prompting** | Prepends learned continuous vectors to the input / to attention | Prefix Tuning, SMoP | Longer sequence → **added latency**, eats context |
| **Selective** | Trains a chosen subset of *existing* weights | BitFit (biases only), Diff Pruning | None (same architecture) |
| **Re-Parameterization** | Re-expresses the **update** in a cheaper algebraic form | **LoRA, QLoRA** | **None** — mergeable |

### Seeing the difference in code

The distinction is easiest to feel by writing all four against the same layer.

```python
import torch, torch.nn as nn

d = 768                                   # hidden size
base = nn.Linear(d, d)                    # a pre-trained layer

# ---- 1. ADDITIVE (Adapter) -- inserts a NEW bottleneck module after the layer
class Adapter(nn.Module):
    def __init__(self, d, bottleneck=64):
        super().__init__()
        self.down, self.up = nn.Linear(d, bottleneck), nn.Linear(bottleneck, d)
    def forward(self, x):
        return x + self.up(torch.relu(self.down(x)))   # residual around new module

additive = nn.Sequential(base, Adapter(d))   # depth increased -> latency increased

# ---- 2. SELECTIVE (BitFit) -- train only the biases that ALREADY exist
for name, p in base.named_parameters():
    p.requires_grad_(name.endswith("bias"))  # architecture untouched

# ---- 3. SOFT PROMPTING -- learned vectors PREPENDED to the sequence
soft_prompt = nn.Parameter(torch.randn(20, d))       # 20 virtual tokens
# forward: x = torch.cat([soft_prompt.expand(B, -1, -1), x], dim=1)
#          ^ sequence is now 20 tokens longer, forever

# ---- 4. RE-PARAMETERIZATION (LoRA) -- rewrite the UPDATE, not the architecture
r = 8
A = nn.Parameter(torch.randn(r, d) * 0.01)
B = nn.Parameter(torch.zeros(d, r))
# forward: h = base(x) + (alpha / r) * (x @ A.T @ B.T)
# and afterwards you can do:  base.weight += (alpha/r) * B @ A   -> LoRA VANISHES
```

Run your eye down those four blocks. Only the last one can be **deleted after
training** by folding it into `base.weight`. That single property is what the
rest of this section is about.

💡 **Learning Thought — the distinction that actually matters**
> Additive, soft-prompting, and selective methods all change *what you train*.
> Re-parameterization changes *how you represent the change itself*.
>
> That is a deeper move. It doesn't add capacity and it doesn't pick weights —
> it says the update matrix ΔW has structure, and exploits that structure.
> This is why LoRA alone among the four can be **merged back into the base
> weights** and disappear completely at inference time.

**📚 Go deeper**
- [HuggingFace PEFT — conceptual guides](https://huggingface.co/docs/peft/conceptual_guides/adapter) — one page per method, same taxonomy
- [Lialin et al., *Scaling Down to Scale Up: A Guide to PEFT*](https://arxiv.org/abs/2303.15647) — the survey this taxonomy comes from; Figure 2 is the canonical version of the tree above
- [BitFit paper](https://arxiv.org/abs/2106.10199) — the selective branch, and a surprisingly strong baseline

---

## A2. Re-parameterization, defined (slide 5, transcript)

From the lecture, near-verbatim:

> *"Parameterization: we have a model, we have the parameters, we decide what
> the model parameters are, and the approach is to update those parameters.
> Here we will come up with an **alternate** parameterization."*

So: **same function, different parameters**. You are not approximating the
model. You are choosing a different — smaller — set of knobs that can reach the
same place in function space.

### A real-world analogy that holds up

Think about describing a **rotation of a 3-D object**.

| Parameterization | Numbers needed | Notes |
|---|---|---|
| Full 3×3 rotation matrix | **9** | but constrained — must be orthogonal, det = 1 |
| Euler angles (yaw, pitch, roll) | **3** | same rotations, far fewer knobs |

The 9-number version isn't *wrong* — it's over-parameterized. The rotation
group genuinely has only 3 degrees of freedom, so 6 of those 9 numbers are
redundant. Re-parameterizing to Euler angles doesn't lose any rotation you
could previously express.

LoRA makes exactly this bet about `ΔW`: the ambient description (d×k numbers)
is far larger than the actual degrees of freedom the fine-tuning task needs.

💡 **Learning Thought**
> "Re-parameterization" is a term borrowed from optimization, not from deep
> learning. In optimization, re-parameterizing means substituting variables to
> make a problem easier without changing its solution set. LoRA does exactly
> this: it substitutes `ΔW` (d×k free variables) with `BA` (r(d+k) free
> variables). The solution set shrinks, but — and this is the whole bet —
> **the good solutions are still inside it**.

---

## A3–A4. The use case: multi-tenant customer support at scale (slide 6)

A SaaS platform serving **200+ enterprise clients**. Each needs genuinely
different behaviour:

- A **fintech** client needs compliance-aware, hedged, auditable responses.
- A **gaming** client wants casual, meme-friendly, loose tone.

One shared model serves neither well. Prompting alone doesn't reliably hold
tone and domain vocabulary across a long support conversation.

### The naive path, costed out

Full fine-tune LLaMA-13B per client:

```
Model size (FP16):    13 × 10⁹ params × 2 bytes  =  26 GB
Storage:              200 clients × 26 GB        =  5.2 TB
Compute:              200 × thousands of GPU-hours
Serving:              200 separate 13B models    =  200 GPUs sitting warm
```

Make it concrete with a cost model you can edit:

```python
def naive_multitenant_cost(n_clients=200, n_params=13e9, bytes_per_param=2,
                           gpu_hourly_usd=2.50, gpu_gb=80, hours_per_month=730):
    """What 'one full fine-tune per client' actually costs to run."""
    model_gb   = n_params * bytes_per_param / 1e9
    storage_tb = n_clients * model_gb / 1000
    gpus       = n_clients                       # each model needs its own warm GPU
    monthly    = gpus * gpu_hourly_usd * hours_per_month

    print(f"model size          : {model_gb:6.1f} GB")
    print(f"total weight storage: {storage_tb:6.2f} TB")
    print(f"warm GPUs           : {gpus}")
    print(f"serving cost / month: ${monthly:,.0f}")

naive_multitenant_cost()
# model size          :   26.0 GB
# total weight storage:   5.20 TB
# warm GPUs           : 200
# serving cost / month: $365,000
```

Every one of those lines is independently fatal:

| Cost axis | Why it kills the business |
|---|---|
| **Storage** | 5.2 TB of *weights*, before datasets, checkpoints, or versioning |
| **Training** | Cost scales linearly with clients — no economies of scale ever |
| **Serving** | 200 idle-but-warm GPUs. Utilization is terrible; you pay for peak, not mean |

⚠️ **Trap — "this is a made-up example"**
> The professor explicitly heads this off: *"This is really exists. It is not
> like this is a cooked-up use case scenario."* Predibase's **LoRAX** is a
> production implementation of exactly this architecture.

---

## A5. Training-side solution: QLoRA (slide 7)

```
Load base model in 4-bit:   13B × 0.5 bytes  ≈  7 GB   (was 26 GB)
Train one LoRA adapter per client on their data
Each adapter:               ~50 MB
```

The consequence, stated plainly on the slide: 4-bit quantization + paged
optimizers means this fine-tuning **fits on a single 24 GB consumer GPU**
(an RTX 4090) instead of requiring a multi-A100 cluster.

💡 **Learning Thought — the democratization argument**
> The jump from "multi-A100 cluster" to "one consumer GPU" is not a 2× cost
> improvement. It's a change in *who can do the work at all*. A research lab
> with one workstation, a startup with no cloud budget, a student — all become
> able to fine-tune a 13B model. Note this is the actual historical importance
> of QLoRA, beyond the arithmetic.

---

## A6. Serving-side solution: LoRA + adapter hot-swap (slide 8)

```
Keep ONE base model resident in GPU memory
Hot-swap the ~50 MB adapter per incoming request, keyed on client ID

200 clients × 50 MB  =  10 GB total   vs.   5.2 TB for full models
```

A **520× reduction** in the serving footprint.

### What hot-swapping looks like in code

This is the whole multi-tenant architecture in ~25 lines, using the real
`peft` API:

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

# 1. Load the base model ONCE. This is the expensive, shared resource.
base = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-13b-hf", device_map="auto", load_in_4bit=True)

# 2. Register every client's adapter under a name. ~50 MB each.
model = PeftModel.from_pretrained(base, "adapters/fintech_client", adapter_name="fintech")
model.load_adapter("adapters/gaming_client",  adapter_name="gaming")
model.load_adapter("adapters/healthcare_cli", adapter_name="healthcare")

# 3. Route each request to its client's adapter -- this is the "hot swap".
def answer(prompt: str, client_id: str) -> str:
    model.set_adapter(client_id)          # O(1); the base model never moves
    return generate(model, prompt)

answer("Can I dispute this charge?", client_id="fintech")    # compliance-aware tone
answer("how do i get the rare skin lol", client_id="gaming") # casual tone
```

`set_adapter` is the entire trick. `base` — the 7 GB of 4-bit weights — is
loaded once and never reloaded.

This works *only* because of a property of LoRA specifically: `W₀` is shared
and frozen, so it is loaded once; the per-client delta is a pair of tiny
matrices you can swap in and out of GPU memory in milliseconds.

**📚 Go deeper — production multi-LoRA serving**
- [Predibase LoRAX](https://github.com/predibase/lorax) — the reference implementation named on slide 8
- [S-LoRA: Serving Thousands of Concurrent LoRA Adapters](https://arxiv.org/abs/2311.03285) — how to batch *different* adapters in one forward pass
- [vLLM multi-LoRA docs](https://docs.vllm.ai/en/latest/features/lora.html) — the version you'd actually deploy today
- [PEFT multi-adapter guide](https://huggingface.co/docs/peft/developer_guides/mixed_models)

### How real serving stacks do it

⚠️ **Trap — from live Q&A**
> *"Are adapters loaded in parallel when multiple clients access the model at
> the same time?"*
>
> **Answer:** Yes, servers handle multiple clients simultaneously — but adapters
> are **not** necessarily loaded in parallel per request. The techniques are:
> - **Adapter caching** — hot adapters stay resident in GPU memory
> - **Batching** — group requests that share an adapter into one forward pass
> - **Request routing** — direct requests so one base model serves all of them
>
> The base model is shared; only the adapter selection varies.

⚠️ **Trap — from live Q&A**
> *"How are adapters selected — based on importance or domain specificity?"*
>
> **Answer:** There's no ranking. Each adapter is trained independently on its
> own client's data — client 1 learns `ΔW₁ = B₁A₁`, client 2 learns
> `ΔW₂ = B₂A₂`. At inference a **router selects by `client_id`**. Nothing is
> scored or ranked at request time.

---

## 🎯 Interview Questions — Section A

**Q1. What distinguishes re-parameterization methods from other PEFT families?**

> Other families change *which* parameters are trained (selective) or *add* new
> trainable modules (additive, soft prompting). Re-parameterization keeps the
> architecture identical and instead re-expresses the weight *update* in a
> lower-dimensional algebraic form — `ΔW = BA`. The practical consequence is
> that LoRA can be merged into the base weights after training
> (`W_merged = W₀ + (α/r)BA`), giving **zero added inference latency**, whereas
> adapters and prefix tuning permanently change the compute graph.

**Q2. Your company serves 500 clients, each needing a customized 7B model. Design the system.**

> - **Training:** QLoRA. Base 7B loaded once in NF4 (~3.5 GB). One adapter per
>   client trained on client data. Adapter size depends on rank; at r=16 on a
>   7B model expect ~20–40 MB.
> - **Serving:** One base model resident in GPU memory. LoRA adapters stored on
>   fast local disk / host RAM, hot-swapped per request by `client_id`, with an
>   LRU cache keeping the busiest clients' adapters GPU-resident.
> - **Batching:** Group concurrent requests by adapter so each batch needs only
>   one adapter's matrices, or use a multi-LoRA kernel (LoRAX / S-LoRA / vLLM
>   multi-LoRA) that applies different adapters within one batch.
> - **Numbers to quote:** 500 × 30 MB = 15 GB of adapters vs 500 × 14 GB = 7 TB
>   of full models.

**Q3. When would you *not* use LoRA?**

> - When the target domain is genuinely far from pre-training (a new language,
>   a new modality, novel token distribution) — the low-rank assumption on ΔW
>   weakens and full fine-tuning or continued pre-training wins.
> - When you need to change the model's *knowledge* substantially rather than
>   its behaviour/format. LoRA is excellent at style, format, and task shape;
>   it is a weak vehicle for injecting large amounts of new factual content
>   (use RAG or continued pre-training).
> - When you're serving exactly one model and have the budget — merged full
>   fine-tuning is simpler operationally, with no adapter management.

**Q4. Why does the serving-side saving matter more commercially than the training-side saving?**

> Training is a one-time (or periodic) cost that can be scheduled, batched, and
> run on spot instances. Serving is a **continuous** cost proportional to the
> number of tenants — 200 warm GPUs burn money 24/7 at low utilization. LoRA
> converts a per-tenant serving cost into a per-tenant *memory* cost of 50 MB
> against one shared GPU, which turns a linear cost curve into a nearly flat
> one. That's the difference between a business that scales and one that
> doesn't.

---

## 🔬 Notebook link

Notebook **Section 1** ("The problem: what full fine-tuning costs?") reproduces
the memory arithmetic as a function, so you can plug in your own model size.
Here it is verbatim — note `DATACENTER_GPU_GB` is a knob you should change to
whatever GPU you actually have:

```python
# Per-parameter cost of each recipe, in BITS as you can see in above slides.
RECIPES = {
    "Full fine-tuning": dict(weights=16, gradients=16,  optimizer=64,  adapters=0),
    "LoRA":             dict(weights=16, gradients=0.4, optimizer=0.8, adapters=0.4),
    "QLoRA":            dict(weights=4,  gradients=0.4, optimizer=0.4, adapters=0.4),
}

DATACENTER_GPU_GB = 48     # change to 80 for an H100/A100-80GB

def bits_per_parameter(recipe):
    return float(sum(recipe.values()))

def memory_footprint_gb(n_params, recipe):
    return n_params * bits_per_parameter(recipe) / 8 / 1e9

def gpus_needed(n_params, recipe, gpu_gb=DATACENTER_GPU_GB):
    return math.ceil(memory_footprint_gb(n_params, recipe) / gpu_gb)
```

Run it and confirm it reproduces the 840 GB / 154 GB / 46 GB chain for 70B.

---

## ✅ Self-check before moving on

1. Name the four PEFT branches and give one example method from each.
2. Which single property of LoRA makes adapter hot-swapping possible?
3. Derive: 200 clients × LLaMA-13B in FP16. Why is it 5.2 TB?
4. What are the three techniques a multi-tenant server uses instead of loading
   200 adapters in parallel?
5. Write the 4-line `answer(prompt, client_id)` hot-swap function from memory.

➡️ **Next:** [Section B — LoRA Theory](B-lora-theory.md)
