# §7 — The Four Models in the RLHF Training Loop

> **Slides 38–40, 47–48** · Topics 30–34
> *The systems view. This section explains why "RLHF is expensive" is a memory statement, not a compute statement.*

---

## The one-line story of this section

> PPO-based RLHF holds **four networks** simultaneously: the **policy** (trained), the **reference** (frozen, for KL), the **reward model** (frozen, scores responses), and the **value model** (trained, estimates `V(s)` for advantages). Two are trained, two are frozen. This four-model burden is DPO's entire motivation.

---

## Topic 30 — The Four-Model Zoo (slide 38)

### The slide, verbatim

| Model | Slide description | Output |
|---|---|---|
| **Policy Network `π_θ`** | *Given a prompt, generates the next token (aligned)* | Probability distribution |
| **Reward Model `r_φ`** | *How good or bad is a generated sequence?* | Score |
| **Ref Model `π_ref`** | *Given a prompt, generates the next token (SFT)* | Probability distribution |
| **Value Model `V_ψ`** | *How good or bad is a generated token?* | Score |

### 🔑 The distinction on slide 38 that is easy to miss

> **Reward model** → *"how good or bad a generated **sequence**"* → scores the **whole response**, once, at the end.
> **Value model** → *"how good or bad a generated **token**"* → estimates expected return at **every position**.

Same output type (a scalar), completely different semantics:

```
   Prompt: "Why is the sky blue?"
   Response tokens:   Sunlight  scatters  in   the   atmosphere  <EOS>

   VALUE MODEL   V(s_t):  0.42     0.51    0.55  0.58    0.71      —
                          ↑ "from here, what score do I EXPECT to end with?"
                            (one estimate per token, used for advantages)

   REWARD MODEL  r_φ:      —        —       —     —       —       0.83
                                                                   ↑
                                            "how good was the FINISHED response?"
                                             (exactly one number, at the end)
```

**Mnemonic:** *Reward = the final grade on the essay. Value = your predicted grade partway through writing it.*

### The full table — the one to memorise

| Model | Symbol | Trained? | Initialised from | Role | Memory |
|---|---|---|---|---|---|
| **Policy** | `π_θ` | ✅ **Yes** | SFT model | Generates tokens; the deliverable | Full model + optimiser states |
| **Reference** | `π_ref` | ❌ Frozen | SFT model | Anchor for the KL penalty (§6) | Full model, inference only |
| **Reward** | `r_φ` | ❌ Frozen | SFT model + scalar head | Scores completed responses (§8) | Full model, inference only |
| **Value** | `V_ψ` | ✅ **Yes** | SFT/reward model + scalar head | Baseline for advantages (§4) | Full model + optimiser states |

> The professor's summary from the lecture: *"it involves maintaining 4 models. And think about it — your large language models are 30 billion, 40 billion, 100 billion parameters, and you are maintaining [four of them]…"* — this is the setup for the DPO pitch in §10.

### 🔬 Compute your own memory budget

Do not memorise the numbers below — memorise the *formula*, and run this for whatever model you're actually planning to train.

```python
def memory_report(n_params_B, dtype_bytes=2, optimizer="adam", lora=False,
                  lora_frac=0.005):
    """
    Estimate GPU memory for each RLHF configuration.

    Adam stores 2 fp32 states (m, v) per trainable parameter -> 8 bytes each.
    Gradients are stored at the parameter dtype.
    """
    P = n_params_B * 1e9
    weights = P * dtype_bytes / 1e9                       # GB

    trainable = P * (lora_frac if lora else 1.0)
    grads = trainable * dtype_bytes / 1e9
    opt   = trainable * (8 if optimizer == "adam" else 0) / 1e9

    trained_model = weights + grads + opt                 # policy or value
    frozen_model  = weights                               # ref or reward

    configs = {
        "SFT              (1 trained)":            trained_model,
        "DPO              (1 trained + 1 frozen)": trained_model + frozen_model,
        "GRPO             (1 trained + 2 frozen)": trained_model + 2 * frozen_model,
        "PPO-RLHF         (2 trained + 2 frozen)": 2 * trained_model + 2 * frozen_model,
    }
    print(f"\n=== {n_params_B}B model, {'LoRA' if lora else 'full'} fine-tuning, "
          f"{dtype_bytes*8}-bit weights ===")
    print(f"  weights/model      : {weights:7.1f} GB")
    print(f"  grads (trainable)  : {grads:7.1f} GB")
    print(f"  Adam states        : {opt:7.1f} GB")
    print(f"  -> one TRAINED model: {trained_model:7.1f} GB")
    print(f"  -> one FROZEN model : {frozen_model:7.1f} GB\n")
    for name, gb in configs.items():
        gpus = -(-gb // 80)                                # 80GB A100/H100
        print(f"  {name:<40} {gb:8.1f} GB  (~{gpus:.0f}x 80GB GPU)")


memory_report(7, dtype_bytes=2, lora=False)
memory_report(7, dtype_bytes=2, lora=True)      # LoRA changes everything
memory_report(70, dtype_bytes=2, lora=True)
```

**Typical output for a 7B model, full fine-tuning (bf16):**

| Configuration | Memory | GPUs |
|---|---|---|
| SFT (1 trained) | ~84 GB | 2 |
| **DPO** (1 trained + 1 frozen) | **~98 GB** | 2 |
| GRPO (1 trained + 2 frozen) | ~112 GB | 2 |
| **PPO-RLHF** (2 trained + 2 frozen) | **~196 GB** | 3 |

Plus activations, KV cache, and the rollout buffer on top. **That ~2× ratio between PPO and DPO is the practical headline** — and with LoRA it becomes far more lopsided, because DPO's frozen reference becomes free (Topic 32) while PPO still needs a separately-trained value network.

### How the four interact — one PPO iteration

```
   ┌──────────────────────────────────────────────────────────────────┐
   │  ① GENERATE                                                      │
   │     π_θ  ──sample──►  responses y for prompts x                  │
   │            (also record log π_θ_old(a_t|s_t) into the buffer)    │
   └───────────────────────────────┬──────────────────────────────────┘
                                   ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │  ② SCORE                                                         │
   │     r_φ(x,y)          ──►  one terminal scalar per response      │
   │     π_ref(a_t|s_t)    ──►  per-token logprobs for the KL term    │
   │     V_ψ(s_t)          ──►  per-token value estimates             │
   └───────────────────────────────┬──────────────────────────────────┘
                                   ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │  ③ COMBINE                                                       │
   │     r̃_t = r_φ·1[t=T] − β(log π_θ − log π_ref)      (§6)          │
   │     A_t  = GAE(r̃, V_ψ)                              (§4)         │
   └───────────────────────────────┬──────────────────────────────────┘
                                   ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │  ④ UPDATE (K epochs over the saved buffer)              (§5)     │
   │     π_θ ← clipped surrogate ascent using r_t(θ) and A_t          │
   │     V_ψ ← regression toward the observed returns G_t             │
   └──────────────────────────────────────────────────────────────────┘
```

### 🔬 All four models, instantiated

```python
# pip install trl transformers peft torch
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          AutoModelForSequenceClassification)
from trl import AutoModelForCausalLMWithValueHead

SFT_MODEL = "Qwen/Qwen2-0.5B-Instruct"
RM_MODEL  = "Qwen/Qwen2-0.5B-Instruct"       # in reality: your trained RM (§8)
tok = AutoTokenizer.from_pretrained(SFT_MODEL)
tok.pad_token = tok.eos_token

# ── 1 & 4. POLICY + VALUE  (TRL fuses them: shared backbone, two heads) ──
policy = AutoModelForCausalLMWithValueHead.from_pretrained(SFT_MODEL)
policy.train()                                # TRAINED

# ── 2. REFERENCE — frozen copy of the SFT model (§6) ──
ref = AutoModelForCausalLM.from_pretrained(SFT_MODEL)
ref.eval()
for p in ref.parameters():
    p.requires_grad_(False)                   # FROZEN

# ── 3. REWARD MODEL — scalar head, frozen during PPO (§8) ──
reward = AutoModelForSequenceClassification.from_pretrained(RM_MODEL, num_labels=1)
reward.eval()
for p in reward.parameters():
    p.requires_grad_(False)                   # FROZEN

def count(m, trainable_only=False):
    return sum(p.numel() for p in m.parameters()
               if (p.requires_grad or not trainable_only))

print(f"{'model':<24} | {'params':>13} | {'trainable':>13} | status")
print("-" * 68)
print(f"{'policy (+ value head)':<24} | {count(policy):>13,} | "
      f"{count(policy, True):>13,} | TRAINED")
print(f"{'reference':<24} | {count(ref):>13,} | {count(ref, True):>13,} | frozen")
print(f"{'reward model':<24} | {count(reward):>13,} | "
      f"{count(reward, True):>13,} | frozen")

total = count(policy) + count(ref) + count(reward)
print(f"\nTOTAL parameters resident: {total:,} "
      f"({total / count(ref):.1f}x a single model)")
print("\nOnly the POLICY is shipped. The other three exist purely for training.")
```

### 💡 Learning thought

> Sort the four models by **what question each answers** and the architecture stops feeling arbitrary:
> - `π_θ` — *"what should I say next?"* (the product)
> - `π_ref` — *"what would I have said before alignment?"* (the anchor)
> - `r_φ` — *"was the finished answer good?"* (the goal)
> - `V_ψ` — *"how good did things look at this point?"* (the yardstick)
>
> Then note: **DPO removes `r_φ` and `V_ψ`**, keeping only the policy and the reference. Ask yourself *now* how it could possibly manage that — the answer in §10 will land much harder.

### 🔗 Resources for Topic 30

- **[HuggingFace — Illustrating RLHF](https://huggingface.co/blog/rlhf)** — the canonical four-model diagram.
- **[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)** — a production framework that explicitly separates the four models across devices. Its architecture diagram is the best illustration of the systems problem.
- **[Transformer Math 101 (EleutherAI)](https://blog.eleuther.ai/transformer-math/)** — the definitive reference for the memory formulas in the code above. Read it once and you will never guess at GPU requirements again.

---

## Topic 31 — The Policy Model (slide 39)

### Architecture

```
   Tokens:      p1    p2    p3    p4    p5    a1    a2    a3
                 │     │     │     │     │     │     │     │
                 ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼
        ┌───────────────────────────────────────────────────────┐
        │        Language Model (Policy π_θ)                    │
        │   Each hidden state encodes information ONLY about    │
        │   previous tokens  (CAUSAL MASKING)                   │
        └───────────────────────────────────────────────────────┘
                 │     │     │     │     │     │     │     │
                 ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼
        Linear layer for logit calculation
              LOGIT LOGIT LOGIT LOGIT LOGIT LOGIT LOGIT LOGIT
                p1    p2    p3    p4    p5    a1    a2    a3
                 │     │     │     │     │     │     │     │
                 ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼
        Softmax for probability distribution
              SOFTM SOFTM SOFTM SOFTM SOFTM SOFTM SOFTM SOFTM
                            ↓
              Probability of the next action (token)

        ★ MODEL UPDATED BY RL ★
```

*(`p1…p5` = prompt tokens; `a1…a3` = generated action tokens.)*

### The three things slide 39 wants you to notice

**1. Causal masking.** Each position attends only to itself and earlier positions. This is what makes the RL formulation valid — position `t`'s output depends on exactly `s_t = (prompt + tokens before t)`, precisely the MDP state.

**2. `Linear → Softmax` over the vocabulary.** The LM head produces `|V|` logits; softmax makes them a distribution. **That distribution *is* `π_θ(·|s_t)`.** No separate policy head is bolted on. This is the concrete meaning of "the model *is* the policy."

**3. This is the only model you ship.** After training, `π_ref`, `r_φ`, and `V_ψ` are all discarded. The deliverable is the policy.

### 🔬 One forward pass gives you every position's policy

Because of causal masking, a **single** forward pass over `prompt + response` yields the policy's distribution at every position simultaneously. This is how `log π_θ(a_t|s_t)` for all t is computed in one shot during the update phase (teacher forcing).

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, torch.nn.functional as F

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
mdl = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct",
                                           dtype=torch.float32).eval()

prompt   = "Why is the sky blue? "
response = "Sunlight scatters in the atmosphere."

p_ids = tok(prompt,   add_special_tokens=False)["input_ids"]
r_ids = tok(response, add_special_tokens=False)["input_ids"]
ids   = torch.tensor([p_ids + r_ids])

# ONE forward pass -> logits at EVERY position, simultaneously
with torch.no_grad():
    logits = mdl(input_ids=ids).logits            # (1, T, |V|)

print(f"Sequence length T   : {ids.shape[1]}")
print(f"Logits shape        : {tuple(logits.shape)}")
print(f"=> |V| = {logits.shape[-1]:,} probabilities at EACH of {ids.shape[1]} positions,")
print(f"   from ONE forward pass. Causal masking makes this valid.\n")

# Extract log pi(a_t | s_t) for the actual tokens
logp   = F.log_softmax(logits[:, :-1, :], -1)     # position t predicts t+1
labels = ids[:, 1:]
tok_lp = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)[0]

print(f"{'t':>3} | {'token a_t':<14} | {'log pi(a_t|s_t)':>16} | {'pi':>7} | part")
print("-" * 62)
for t in range(len(p_ids) - 1, ids.shape[1] - 1):
    part = "PROMPT" if t < len(p_ids) - 1 else "response"
    print(f"{t:>3} | {tok.decode(labels[0, t])!r:<14} | {tok_lp[t]:>16.4f} | "
          f"{tok_lp[t].exp():>7.4f} | {part}")

comp_lp = tok_lp[len(p_ids) - 1:].sum()
print(f"\nlog pi(response | prompt) = {comp_lp:.3f}   (sum over RESPONSE tokens only)")
print("=> This is exactly §3 Topic 14's identity, and exactly what §10 needs.")
```

> 💡 **Generation is sequential; scoring is parallel.** That asymmetry — measured in §5's timing experiment — is why generation dominates RLHF's cost while the gradient step is comparatively cheap.

---

## Topic 32 — The Reference Model (slide 40)

### Architecture: identical to the policy

Slide 40 shows the *exact same diagram* as slide 39, with one caption changed:

> **Supervised Fine-Tuned Model** — (and no "updated by RL" star)

### What differs

| | Policy `π_θ` | Reference `π_ref` |
|---|---|---|
| Architecture | Transformer LM | **Identical** |
| Initial weights | SFT model | SFT model — **same starting point** |
| Trained? | ✅ Yes | ❌ **Frozen forever** |
| Used for | Generation + gradient | KL term only |
| Mode | `train()` | `eval()`, `torch.no_grad()` |
| Shipped? | ✅ | ❌ discarded after training |

**At step 0 they are byte-identical.** As training proceeds, `π_θ` drifts; `D_KL(π_θ‖π_ref)` measures exactly how far. The reference is a **frozen snapshot of who the model used to be.**

### Why we cannot skip it

Without `π_ref` there is no KL term, hence no anchor, hence unbounded reward hacking (§6). It also does real double duty:
- **RLHF**: supplies `log π_ref(a_t|s_t)` for the per-token KL penalty
- **DPO**: supplies `log π_ref(y|x)` for both the chosen and rejected completions — it survives the transition to §10

### 🔬 The LoRA reference trick — production standard

This is the highest-value practical technique in §7. **One set of base weights serves both policy and reference.**

```python
# pip install peft trl transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

BASE = "Qwen/Qwen2-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(BASE)

base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)

lora_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)
policy = get_peft_model(base, lora_cfg)

trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
total     = sum(p.numel() for p in policy.parameters())
print(f"Trainable: {trainable:,} / {total:,}  ({100*trainable/total:.2f}%)\n")

ids = tok("Why is the sky blue?", return_tensors="pt")

# ── π_θ : adapters ENABLED ──
with torch.no_grad():
    logits_policy = policy(**ids).logits[0, -1, :]

# ── π_ref : the SAME weights with adapters DISABLED ──
with policy.disable_adapter():
    with torch.no_grad():
        logits_ref = policy(**ids).logits[0, -1, :]

print("At initialisation LoRA B is zero, so the two are identical:")
print(f"  max |logit difference| = {(logits_policy - logits_ref).abs().max():.6f}")
print("\nAfter training, the adapters shift the policy while the base -- and")
print("therefore pi_ref -- stays fixed. THE REFERENCE MODEL COSTS ~0 EXTRA MEMORY.")
print("\nIn TRL you get this by simply passing ref_model=None with a PEFT model:")
print("    DPOTrainer(model=peft_policy, ref_model=None, ...)")
```

**Memory impact for a 7B model:** the reference drops from ~14 GB to ~0, and the policy's optimiser states shrink from ~56 GB to ~0.3 GB. This is what makes single-GPU DPO on a 7B model routine.

### ⚠️ Do not confuse `π_ref` with `π_θ_old`

| | `π_ref` (§6) | `π_θ_old` (§5) |
|---|---|---|
| What it is | Frozen **SFT model** | The policy **as of the last rollout** |
| Physically | A real model resident in GPU memory | **Stored log-probs in the rollout buffer** |
| Updated | Never | Every K epochs (`θ_old ← θ`) |
| Purpose | Anti-reward-hacking anchor | Importance-sampling correction |
| Extra memory | Yes (one full model) | ~None |

Slide 37's note — *"Not two copies of the policy"* — is about `π_θ_old`. Slide 40 — a genuinely separate model — is about `π_ref`. **Both statements are true and they refer to different objects.** This is the single most reliable place to demonstrate real understanding in an interview.

### 🔗 Resources for Topic 32

- **[PEFT — LoRA docs](https://huggingface.co/docs/peft/conceptual_guides/lora)** — the `disable_adapter()` context manager used above.
- **[TRL — Training with PEFT](https://huggingface.co/docs/trl/peft_integration)** — the `ref_model=None` convention and when it's safe.
- **[Hu et al., LoRA (2021)](https://arxiv.org/abs/2106.09685)** — the original paper; relevant here because the B-matrix zero-initialisation is *why* policy and reference start identical.

---

## Topic 33 — The Value Head: Estimating `V^π(s)` (slide 47)

### Architecture

```
   Tokens:            p5      a1      a2      a3
                       │       │       │       │
        ┌──────────────────────────────────────────────┐
        │       Language Model (Policy π_θ backbone)   │
        └──────────────────────────────────────────────┘
                       │       │       │       │
                       ▼       ▼       ▼       ▼
             ┌──────────────────────────────────────┐
             │  Linear layer with ONE output        │   ← the "value head"
             │  Linear  Linear  Linear  Linear      │
             └──────────────────────────────────────┘
                       │       │       │       │
                       ▼       ▼       ▼       ▼
                    V(s)     V(s)    V(s)    V(s)      ← one scalar PER TOKEN
```

### The key architectural point

The LM head maps `hidden_dim → |V|` (~50,000 outputs). The **value head** maps `hidden_dim → 1`. A single scalar per position, interpreted as *"expected return from this state."*

Same trick as the reward model (§8) — the difference is entirely in **what it's trained to predict** and **where it produces useful outputs**:

| | Reward head `r_φ` | Value head `V_ψ` |
|---|---|---|
| Output used at | The **last** token only | **Every** token |
| Predicts | Human preference score for the finished response | Expected future return from this state |
| Trained on | Preference pairs, Bradley-Terry loss (§8) | Observed returns, MSE regression |
| Frozen during PPO? | ✅ Yes | ❌ No — trained jointly |

### 🔬 Build a value head and train it

```python
import torch, torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class ValueHead(nn.Module):
    """
    A transformer backbone + a hidden_dim -> 1 head, read at EVERY position.
    Structurally identical to a reward model; the difference is the LOSS
    and WHICH positions you use.
    """
    def __init__(self, model_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        h = self.backbone.config.hidden_size
        self.v_head = nn.Linear(h, 1)
        # Small init: an untrained value head emitting large values wrecks
        # early advantages. A "37 details" style trick.
        nn.init.normal_(self.v_head.weight, std=1 / (h + 1) ** 0.5)
        nn.init.zeros_(self.v_head.bias)

    def forward(self, input_ids, attention_mask):
        hs = self.backbone(input_ids=input_ids,
                           attention_mask=attention_mask).last_hidden_state
        return self.v_head(hs).squeeze(-1)        # (B, T) -- one scalar per token


tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
vm  = ValueHead("Qwen/Qwen2-0.5B-Instruct")

ids = tok("Why is the sky blue? Sunlight scatters in the atmosphere.",
          return_tensors="pt")
V = vm(ids["input_ids"], ids["attention_mask"])
print(f"Input shape : {tuple(ids['input_ids'].shape)}")
print(f"V(s_t) shape: {tuple(V.shape)}   <- ONE value per token position\n")


def value_loss(values, returns, old_values=None, clip=0.2):
    """
    The value head's OWN loss: regression onto observed returns.
    This is PPO's c_1 term (§5).
    """
    if old_values is None:
        return 0.5 * ((values - returns) ** 2).mean()
    v_clipped = old_values + torch.clamp(values - old_values, -clip, clip)
    return 0.5 * torch.max((values - returns) ** 2,
                           (v_clipped - returns) ** 2).mean()


def explained_variance(values, returns):
    """
    THE health metric for a value head.
      1.0 = perfect prediction
      0.0 = no better than predicting the mean  <- your advantages are noise
      <0  = worse than the mean                 <- something is badly wrong
    Log this every step of any PPO run.
    """
    var_y = returns.var()
    return (1 - (returns - values).var() / (var_y + 1e-8)).item()


returns = torch.tensor([8.5, 8.5, 8.5, 8.4, 8.4, 8.3])
good    = torch.tensor([8.3, 8.4, 8.5, 8.5, 8.4, 8.3])
bad     = torch.tensor([2.0, 9.0, 1.0, 7.0, 3.0, 8.0])
const   = torch.full((6,), returns.mean())

print(f"{'value predictions':<22} | {'MSE loss':>9} | {'explained variance':>19}")
print("-" * 56)
for name, v in [("well-trained", good), ("constant (= mean)", const), ("garbage", bad)]:
    print(f"{name:<22} | {value_loss(v, returns):>9.3f} | {explained_variance(v, returns):>19.3f}")

print("\nEV near 0 => the value head is useless => advantages carry no")
print("per-token information => you are back to raw REINFORCE variance.")
```

### Training data and loss

> **Slide 47:** *"Training data: Generate a trajectory and calculate reward at each point."*

The value head is trained by **regression onto the returns actually observed** in the rollouts:

$$L_{V}(\psi) = \mathbb{E}_t\Big[\big(V_\psi(s_t) - G_t\big)^2\Big]$$

where `G_t` is the reward-to-go computed from the shaped rewards `r̃`. This is the `c_1` term in the PPO loss (§5).

**Chicken-and-egg, resolved by co-training:** good advantages need a good `V`; a good `V` needs returns generated by a good policy. PPO trains both simultaneously — the value head chases a moving target. This is a genuine source of instability, and it's why practitioners monitor **explained variance** as a health metric.

### Why the value model is the most disposable of the four

- **GRPO** drops it entirely, replacing `V(s)` with the empirical mean reward over a *group* of responses to the same prompt.
- **RLOO** (REINFORCE Leave-One-Out) uses the mean of the *other* samples' rewards as the baseline.
- **DPO** doesn't need advantages at all, so no value model.

### 🔬 GRPO's baseline — deleting a 7B network in six lines

```python
import numpy as np

def grpo_advantages(group_rewards):
    """
    Group Relative Policy Optimization (DeepSeekMath, 2024).
    Sample G responses to the SAME prompt; use the group's own statistics
    as the baseline. No learned value network required.
    """
    r = np.asarray(group_rewards, dtype=np.float32)
    return (r - r.mean()) / (r.std() + 1e-8)


def rloo_advantages(group_rewards):
    """REINFORCE Leave-One-Out: baseline = mean of the OTHER samples."""
    r = np.asarray(group_rewards, dtype=np.float32)
    G = len(r)
    return r - (r.sum() - r) / (G - 1)


rewards = [8.5, 6.2, 3.1, 7.9, 5.0, 2.2, 9.1, 4.4]   # G=8 responses, one prompt

print("Rewards for 8 responses to the SAME prompt:")
print(" ", np.round(rewards, 2))
print("\nGRPO advantages:", np.round(grpo_advantages(rewards), 3))
print("RLOO advantages:", np.round(rloo_advantages(rewards), 3))

print("\nBoth are VALID baselines: they do not depend on the action taken")
print("within a response, so unbiasedness is preserved (§4 Topic 21).")
print("\nWhat this buys: one fewer TRAINED 7B network (~84 GB), plus the")
print("removal of the value head's own instability. What it costs: you must")
print("sample G responses per prompt instead of 1.")
```

> 💡 **The value network exists for exactly one reason:** the preference reward is terminal, and a terminal reward gives every token the same weight (§4, Topic 23). `V(s_t)` varies per position, so `A_t = G_t − V(s_t)` doesn't. If you can obtain a per-response baseline some *other* way — group statistics, leave-one-out — you delete a trained 7B network from your pipeline. **That single realisation is most of the last two years of RLHF systems research.**

### 🔗 Resources for Topic 33

- **[Shao et al., DeepSeekMath (2024)](https://arxiv.org/abs/2402.03300)** — §4.1 introduces GRPO and explains precisely why the value model is droppable.
- **[Ahmadian et al., Back to Basics: Revisiting REINFORCE-style optimization (RLOO, 2024)](https://arxiv.org/abs/2402.14740)** — argues most of PPO's machinery is unnecessary for LLM alignment. A genuinely provocative and useful read.
- **[The 37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)** — items on value-head initialisation, value clipping, and explained variance.

---

## Topic 34 — Overall Model Architecture (slide 48)

Slide 48 stacks all three heads on the shared transformer picture:

```
   Tokens:               p5        a1        a2        a3
                          │         │         │         │
        ┌────────────────────────────────────────────────────┐
        │              LANGUAGE MODEL BACKBONE               │
        │              (transformer, causal)                 │
        └────────────────────────────────────────────────────┘
                          │         │         │         │
            ┌─────────────┼─────────┼─────────┼─────────┼──────────┐
            │             ▼         ▼         ▼         ▼          │
            │   ┌─────────────────────────────────────────┐        │
   HEAD 1   │   │ LM HEAD:  hidden → |V| logits → softmax │        │
            │   │ = the POLICY  π_θ(a|s)                  │        │
            │   └─────────────────────────────────────────┘        │
            │                                                      │
            │   ┌─────────────────────────────────────────┐        │
   HEAD 2   │   │ REWARD HEAD: hidden → 1  (LAST token)   │        │
            │   │ = r_φ(x, y)   — separate frozen model   │        │
            │   └─────────────────────────────────────────┘        │
            │                                                      │
            │   ┌─────────────────────────────────────────┐        │
   HEAD 3   │   │ VALUE HEAD:  hidden → 1  (EVERY token)  │        │
            │   │ = V_ψ(s_t)                              │        │
            │   └─────────────────────────────────────────┘        │
            └──────────────────────────────────────────────────────┘
```

### 🔬 One backbone, three heads — the general principle in code

```python
import torch, torch.nn as nn
from transformers import AutoModel

class ThreeHeadedModel(nn.Module):
    """
    Slide 48 made literal. ONE transformer backbone, THREE heads.
    The head and the LOSS define the task -- the backbone is generic.
    """
    def __init__(self, model_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        h = self.backbone.config.hidden_size
        V = self.backbone.config.vocab_size

        self.lm_head     = nn.Linear(h, V, bias=False)   # POLICY / REFERENCE
        self.reward_head = nn.Linear(h, 1)               # REWARD MODEL
        self.value_head  = nn.Linear(h, 1)               # VALUE MODEL

    def forward(self, input_ids, attention_mask):
        hs = self.backbone(input_ids=input_ids,
                           attention_mask=attention_mask).last_hidden_state

        # LAST NON-PAD position -- critical for the reward head (see §8)
        last = attention_mask.sum(dim=1) - 1
        batch = torch.arange(hs.size(0))

        return {
            "policy_logits": self.lm_head(hs),                       # (B,T,|V|)
            "reward":        self.reward_head(hs[batch, last]).squeeze(-1),  # (B,)
            "values":        self.value_head(hs).squeeze(-1),        # (B,T)
        }


m = ThreeHeadedModel("Qwen/Qwen2-0.5B-Instruct")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
tok.pad_token = tok.eos_token
enc = tok(["Why is the sky blue? Sunlight scatters.",
           "What is 2+2? Four."], return_tensors="pt", padding=True)

out = m(enc["input_ids"], enc["attention_mask"])
print(f"policy_logits : {tuple(out['policy_logits'].shape)}  -> distribution per token")
print(f"reward        : {tuple(out['reward'].shape)}         -> ONE scalar per SEQUENCE")
print(f"values        : {tuple(out['values'].shape)}         -> one scalar per TOKEN")

print(f"\n{'head':<14} | {'output dim':>10} | {'read at':<14} | {'loss'}")
print("-" * 62)
print(f"{'lm_head':<14} | {'|V|':>10} | {'every token':<14} | cross-entropy / PG")
print(f"{'reward_head':<14} | {'1':>10} | {'last token':<14} | Bradley-Terry (§8)")
print(f"{'value_head':<14} | {'1':>10} | {'every token':<14} | MSE to returns (§4)")
```

### Shared vs. separate backbones — a real engineering decision

| Design | Pros | Cons |
|---|---|---|
| **Fully separate models** (4 backbones) | No interference between objectives; RM can be independently sized/versioned | Maximum memory |
| **Shared policy+value backbone**, two heads | Saves ~one model; representations transfer | Objectives conflict; the value loss can degrade generation quality |
| **LoRA policy, shared base for ref** | Reference becomes ~free | Slightly constrained policy capacity |
| **Smaller reward model** | Cheap scoring | Weaker judge ⇒ more hackable |

**Common production configuration:** reward model kept fully separate (often a different, sometimes smaller, base); policy and value sharing a backbone with two heads (this is exactly what TRL's `AutoModelForCausalLMWithValueHead` does); reference obtained via LoRA adapter-disabling. That gets you from "four 7B models" to something close to "one and a half."

### 💡 Learning thought

> Notice that **all four models are the same architecture with different heads and different training objectives.** A transformer backbone plus:
> - `→ |V|` head trained on cross-entropy = **language model / policy**
> - `→ 1` head trained on Bradley-Terry = **reward model**
> - `→ 1` head trained on MSE-to-returns = **value model**
>
> This is the general lesson of modern deep learning: **the backbone is generic; the head and the loss define the task.** Once you internalise it, the RLHF stack stops looking like four exotic components and starts looking like one component wearing four hats.

---

## 🎯 Interview Questions — §7

### Conceptual

**Q1. Name the four models in PPO-based RLHF, which are trained, and what each does.**
> **Policy `π_θ`** — trained; generates tokens; the shipped artifact. **Value `V_ψ`** — trained; predicts expected return per token, serving as the advantage baseline. **Reference `π_ref`** — frozen SFT copy; supplies logprobs for the KL penalty. **Reward `r_φ`** — frozen; scores completed responses. Two trained, two frozen; all four resident in memory during training.

**Q2. Reward model vs. value model — what's the difference?**
> The **reward model** scores a *complete sequence* against human preference, producing one terminal scalar; it's trained separately on preference pairs with a Bradley-Terry loss and frozen during PPO. The **value model** estimates the *expected future return from each intermediate state*, producing a scalar at every token; it's trained by MSE regression against observed returns, jointly with the policy. Reward = the final grade; value = the predicted grade partway through. Architecturally both are `hidden → 1`; the difference is the loss and which positions are read.

**Q3. Why is `π_ref` a real model in memory while `π_θ_old` is not?**
> `π_θ_old` is only needed to supply `log π_θ_old(a_t|s_t)` for the *tokens that were actually sampled*, and those logprobs are recorded at sampling time — so they live in the rollout buffer as plain numbers. `π_ref` must produce logprobs for sequences generated *after* it was frozen, requiring fresh forward passes, so the weights must be resident. Slide 37's "not two copies of the policy" refers to `π_θ_old`; slide 40's separate model refers to `π_ref`. Both are correct.

**Q4. All four models are the same architecture. What differentiates them?**
> The **output head** and the **training objective**. LM head (`hidden → |V|`, cross-entropy) gives the policy/reference; a scalar head (`hidden → 1`) trained with Bradley-Terry on preference pairs gives the reward model; a scalar head trained with MSE against observed returns gives the value model. Same transformer backbone, different heads, different losses.

**Q5. Why does the value head output a scalar at every token, while the reward head is only read at the last?**
> The value function is defined per *state*, and every token position is a distinct state — advantages are needed at each position for the per-token policy gradient. The reward model scores a complete *response*, which only exists at the final position; earlier positions correspond to incomplete text, for which "preference score" is undefined.

**Q6. Why does GRPO not need a value model?**
> Because `V(s)` serves only as a **baseline**, and any action-independent baseline preserves unbiasedness. GRPO samples a *group* of G responses to the same prompt and uses the group's empirical mean and std of rewards as the baseline: `A_i = (r_i − mean)/std`. That's a valid baseline obtained from samples rather than from a learned network — eliminating one trained 7B model and its optimiser states, plus the instability of a value head chasing a moving target. The cost is G generations per prompt instead of 1.

**Q7. How do you cut RLHF memory in practice?**
> LoRA on the policy so `π_ref` is the same base weights with adapters disabled (`policy.disable_adapter()`); share a backbone between policy and value with two heads; keep frozen models (`π_ref`, `r_φ`) in lower precision since they're inference-only; use a smaller reward model; offload frozen models to CPU; switch to GRPO (drops the value model) or DPO (drops both reward and value).

**Q8. What is explained variance and why do you log it?**
> `1 − Var(returns − values)/Var(returns)` — how much of the variation in observed returns the value head actually predicts. 1.0 is perfect; 0.0 means it's no better than predicting the mean, in which case advantages carry no per-token information and you are back to raw REINFORCE variance; negative means something is badly wrong. It's the single best health metric for a PPO value head.

### Applied

**Q9. Estimate the memory for PPO-tuning a 7B model, and compare against DPO.**
> Per model in bf16: ~14 GB weights. A *trained* model adds ~14 GB gradients + ~56 GB Adam states (2 fp32 states/param) ≈ 84 GB. PPO: 2 trained (policy, value) + 2 frozen (ref, reward) ≈ 2(84) + 2(14) ≈ **196 GB**, plus activations and rollout buffer. DPO: 1 trained + 1 frozen ≈ 84 + 14 ≈ **98 GB**. With LoRA the trainable fraction drops to <1%, so the trained model collapses to ~14.3 GB and the reference becomes free via adapter-disabling — putting 7B DPO comfortably on a single 80 GB GPU. That gap is the practical case for DPO.

**Q10. Your value head has near-zero explained variance mid-training. What does that mean and what do you do?**
> `V_ψ` is not predicting returns better than a constant — so advantages are essentially `G_t − constant`, losing the per-token discrimination the value head exists to provide, and gradient variance spikes. Causes: value LR too low or too high, insufficient `vf_coef`, a policy drifting so fast the target keeps moving, poor value-head initialisation, or reward scaling issues. Fixes: warm up the value head before policy updates begin, increase `vf_coef`, use value clipping, normalise value targets, reduce the policy LR — or sidestep the failure mode entirely with GRPO.

**Q11. Would you share a backbone between policy and value? Argue both sides.**
> **For:** halves the memory of the two trained models; the representations genuinely overlap (both need to understand the partial response). **Against:** the objectives conflict — the value regression loss backpropagates into the shared backbone and can degrade generation quality, and the two want different learning rates. **In practice:** TRL's `AutoModelForCausalLMWithValueHead` shares the backbone, which is the common default; keep them separate when memory permits and quality regressions appear. The reward model is almost always kept separate, because you want to version and evaluate the judge independently of the thing being judged.

### Rapid-fire

| Question | Answer |
|---|---|
| How many models in PPO-RLHF? | **4** |
| Which are trained? | Policy and value |
| Which are frozen? | Reference and reward |
| Which one do you ship? | The policy |
| Reward head output dimension? | 1 |
| Value head output dimension? | 1 |
| LM head output dimension? | \|V\| (vocab size) |
| Where is the reward head read? | Last **non-pad** token only |
| Where is the value head read? | Every token |
| Which model does GRPO delete? | Value |
| Which two does DPO delete? | Reward **and** value |
| Which model survives into DPO? | Reference |
| TRL class fusing policy + value? | `AutoModelForCausalLMWithValueHead` |
| Value-head health metric? | Explained variance |

---

## ✅ Section self-check

1. List all four models with role, trained/frozen status, and initialisation.
2. Explain reward vs. value using the "essay grade" analogy.
3. Explain precisely why `π_ref` costs GPU memory but `π_θ_old` doesn't.
4. Describe the LoRA reference-model trick, and write the two lines of code.
5. Explain why the value model is the easiest of the four to eliminate, and how GRPO does it.
6. State the general principle: same backbone, different ______ and ______.
7. **Hands-on:** run the memory calculator for the model *you* would actually train. How many GPUs for PPO vs. DPO?

---

**Previous:** [§6 — Reward Hacking & KL Control](06-reward-hacking-kl.md) · **Next:** [§8 — Building the Reward Model](08-reward-model.md) · [Index](00-INDEX.md)
