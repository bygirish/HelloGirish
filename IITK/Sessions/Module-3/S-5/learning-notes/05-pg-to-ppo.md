# §5 — From Policy Gradient to PPO

> **Slides 32–33, 37** · Topics 24–26
> *How the sample-inefficiency of REINFORCE gets fixed, and what PPO actually is.*

---

## The one-line story of this section

> Vanilla policy gradient throws away every rollout after one update. **PPO** keeps them: sample from an *old* policy, reuse the batch for K epochs, and correct the distribution mismatch with an **importance-sampling ratio** — clipped so the policy can't run away from the data it was sampled under.

---

## Topic 24 — The Sample-Inefficiency Problem (slide 32)

> **Slide 32:** *"Sample trajectories from the policy network **every time** we update the parameters. This can be very inefficient!!!"*

### Why this happens — it's not laziness, it's the maths

The policy gradient is an expectation **over trajectories drawn from `π_θ`**:

$$\nabla_\theta J = \mathbb{E}_{\tau \sim \pi_\theta}\big[\,\cdot\,\big]$$

The instant you apply an update, `θ → θ'`, and your collected trajectories were drawn from `π_θ`, not `π_θ'`. They are now samples from the **wrong distribution**. Using them again gives a biased gradient.

**Policy gradient is inherently on-policy.** One batch of rollouts → one gradient step → discard.

### The cost, quantified for LLMs

```
   ONE PPO/REINFORCE ITERATION
   ───────────────────────────
   1. Generate D full responses          ← D × (500 forward passes, autoregressive)
      (e.g. D = 512 prompts)                ⏱️  DOMINANT COST — often 70–90% of wall clock
   2. Score each with the reward model   ← D forward passes
   3. Score each with the reference model← D forward passes (for KL)
   4. Value estimates                    ← D forward passes
   5. ONE gradient step                  ← ⏱️  cheap by comparison
   6. Throw everything away. Go to 1.
```

Generation is **autoregressive** — 500 sequential forward passes per response, un-parallelisable across time. And you spend all of it to take **one** optimisation step.

### 🔬 Measure the generation-vs-training cost split

The claim "generation dominates" is worth verifying on your own hardware, because it determines where every optimisation effort should go.

```python
# pip install transformers torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, time

MODEL = "Qwen/Qwen2-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)
mdl = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
dev = "cuda" if torch.cuda.is_available() else "cpu"
mdl = mdl.to(dev)

prompts = ["Explain why the sky is blue."] * 8
enc = tok(prompts, return_tensors="pt", padding=True).to(dev)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# ---- 1. GENERATION: 128 SEQUENTIAL forward passes per sequence ----
mdl.eval()
t0 = time.time()
with torch.no_grad():
    out = mdl.generate(**enc, max_new_tokens=128, do_sample=True,
                       temperature=1.0, pad_token_id=tok.eos_token_id)
t_gen = time.time() - t0

# ---- 2. TRAINING STEP: ONE parallel forward + backward over the same tokens ----
mdl.train()
t0 = time.time()
logits = mdl(input_ids=out, attention_mask=(out != tok.eos_token_id).long()).logits
loss = logits.float().log_softmax(-1).mean()
loss.backward()
t_train = time.time() - t0
mdl.zero_grad()

print(f"Generation (128 tokens x 8 seqs) : {t_gen:6.2f}s")
print(f"One fwd+bwd over the same tokens : {t_train:6.2f}s")
print(f"Generation is {t_gen/t_train:5.1f}x more expensive\n")
print("Generation is SEQUENTIAL (128 dependent steps).")
print("Training is PARALLEL   (all positions at once, thanks to causal masking).")
print("\n=> This ratio is why PPO reuses rollouts for K epochs,")
print("   and ultimately why DPO -- which never generates -- is so much cheaper.")
```

> 💡 Typical result: generation is **20–100× slower** than the gradient step. If you take only *one* step per rollout batch, you are spending 95%+ of your compute on data collection. That single number is PPO's entire motivation.

### 💡 Learning thought

> Contrast with supervised learning, where you shuffle a fixed dataset and take thousands of steps over it for free. In on-policy RL **your dataset expires after every step.** That single fact is why RLHF is one to two orders of magnitude more expensive than SFT, and it is the strongest practical argument for DPO (§10), where the dataset is fixed and reusable like ordinary supervised learning.

### 🔗 Resources for Topic 24

- **[vLLM](https://docs.vllm.ai/)** — the standard answer to "generation is the bottleneck." Paged attention + continuous batching; most modern RLHF stacks (TRL, OpenRLHF, veRL) use it for the rollout phase.
- **[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)** — a production RLHF framework built specifically around decoupling generation from training. Its README architecture diagram makes the cost split visible.

---

## Topic 25 — The Off-Policy Fix: Old Policy + Importance Ratio (slide 33)

> **Slide 33:** *"Sample trajectories from an **old (offline) policy** network; update the parameters with the same trajectories in multiple epochs. Trajectories sampled from `π_θ_old`."*

### The idea

Keep two conceptual policies:

| | Role |
|---|---|
| **`π_θ_old`** | The policy that *generated* the trajectories. Frozen during the reuse window. |
| **`π_θ`** | The policy being *updated*. Starts equal to `π_θ_old`, drifts as we take steps. |

Now reuse the batch for K epochs. But we must correct for the fact that the data came from `π_θ_old` while we're evaluating `π_θ`.

### Importance sampling — the correction

The general identity: to compute an expectation under `p` using samples from `q`,

$$\mathbb{E}_{x\sim p}[f(x)] = \mathbb{E}_{x\sim q}\left[\frac{p(x)}{q(x)} f(x)\right]$$

### 🔬 Importance sampling, verified — and its failure mode

Understanding *why* PPO must clip requires seeing importance sampling break. It takes 20 lines.

```python
import numpy as np
np.random.seed(0)

# Target p and proposal q, both over 5 actions
p = np.array([0.05, 0.10, 0.50, 0.25, 0.10])
f = np.array([1.0, 2.0, 3.0, 4.0, 5.0])          # some function of the action
true_E = (p * f).sum()
print(f"TRUE  E_p[f] = {true_E:.4f}\n")

def importance_estimate(q, D=20000):
    """Estimate E_p[f] using samples drawn from q, reweighted by p/q."""
    samples = np.random.choice(len(q), size=D, p=q)
    weights = p[samples] / q[samples]             # the IMPORTANCE RATIO
    return (weights * f[samples]).mean(), weights

print(f"{'proposal q':<28} | {'estimate':>9} | {'max ratio':>10} | {'ratio std':>10}")
print("-" * 66)

for name, q in [
    ("q = p  (on-policy)",        p),
    ("q close to p",              np.array([0.06, 0.12, 0.45, 0.27, 0.10])),
    ("q moderately different",    np.array([0.20, 0.20, 0.20, 0.20, 0.20])),
    ("q VERY different",          np.array([0.60, 0.25, 0.05, 0.05, 0.05])),
]:
    q = q / q.sum()
    est, w = importance_estimate(q)
    print(f"{name:<28} | {est:>9.4f} | {w.max():>10.2f} | {w.std():>10.2f}")

print("\nAll estimates are UNBIASED. But as q drifts from p:")
print("  - the max ratio explodes")
print("  - the variance of the estimate explodes")
print("  - a finite sample becomes worthless")
print("\n=> PPO's clipping exists precisely to stop pi_theta drifting far")
print("   from pi_theta_old, keeping the ratios near 1 and the variance sane.")
```

Applied to the policy gradient, the per-token correction is the **probability ratio**:

$$\boxed{\;r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}\;}$$

and the surrogate objective becomes:

$$L^{\text{CPI}}(\theta) = \mathbb{E}_t\big[\, r_t(\theta)\, A_t \,\big]$$

### Reading the ratio

> The professor's own description: *"how much the probability has got better in the current policy than the older policy."*

| `r_t(θ)` | Meaning |
|---|---|
| `= 1` | The new policy is unchanged here — no correction needed |
| `> 1` | New policy makes this token **more** likely than the sampler did |
| `< 1` | New policy makes it **less** likely |

And note: **the only change from §4 is that `∇log π` was replaced by `r_t(θ)`.** Everything else — the advantage, the expectation, the sum over tokens — is identical. (In fact `∇_θ r_t(θ)` at `θ = θ_old` equals `∇_θ log π_θ(a_t|s_t)`, so this surrogate *is* the policy gradient at the start of each reuse window, and generalises it away from that point.)

### The danger — and PPO's actual contribution

Importance sampling is only reliable while `π_θ ≈ π_θ_old` — exactly what the experiment above demonstrated. Left unchecked, the optimiser will happily drive some ratios enormous, because that's a cheap way to increase `r_t·A_t`.

**PPO's fix: clip the ratio.**

$$\boxed{\;L^{\text{CLIP}}(\theta) = \mathbb{E}_t\Big[\min\big(r_t(\theta)A_t,\; \text{clip}(r_t(\theta),\, 1-\epsilon,\, 1+\epsilon)\, A_t\big)\Big]}$$

with typically `ε = 0.2`.

```
   A_t > 0  (good action — want to increase probability)
   ────────────────────────────────────────────────────
     objective
        │        ╱────────────  ← CLIPPED at 1+ε: no further reward
        │      ╱                  for pushing the probability higher
        │    ╱
        └──────────────────►  r_t
             1   1+ε

   A_t < 0  (bad action — want to decrease probability)
   ────────────────────────────────────────────────────
     objective
        │────────╲              ← CLIPPED at 1−ε: no further reward
        │         ╲               for pushing the probability lower
        │          ╲
        └──────────────────►  r_t
           1−ε   1
```

### 🔬 The clipped objective, implemented and plotted

```python
import torch
import numpy as np

def ppo_clip_objective(ratio, advantage, eps=0.2):
    """
    The PPO-Clip surrogate. This exact expression is in every PPO codebase.
    Returns the value to MAXIMISE (implementations negate it to minimise).
    """
    unclipped = ratio * advantage
    clipped   = torch.clamp(ratio, 1 - eps, 1 + eps) * advantage
    return torch.min(unclipped, clipped)         # PESSIMISTIC: take the worse one


ratios = torch.linspace(0.0, 2.5, 26)

print("A_t = +1.0  (GOOD action -> we want ratio to INCREASE)")
print(f"{'ratio':>7} | {'unclipped':>10} | {'PPO objective':>14} | gradient?")
print("-" * 56)
for r in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
    r_t = torch.tensor(r)
    obj = ppo_clip_objective(r_t, torch.tensor(1.0)).item()
    flat = "no  <- CLIPPED" if r > 1.2 else "yes"
    print(f"{r:>7.2f} | {r*1.0:>10.2f} | {obj:>14.2f} | {flat}")

print("\nA_t = -1.0  (BAD action -> we want ratio to DECREASE)")
print(f"{'ratio':>7} | {'unclipped':>10} | {'PPO objective':>14} | gradient?")
print("-" * 56)
for r in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
    r_t = torch.tensor(r)
    obj = ppo_clip_objective(r_t, torch.tensor(-1.0)).item()
    flat = "no  <- CLIPPED" if r < 0.8 else "yes"
    print(f"{r:>7.2f} | {r*-1.0:>10.2f} | {obj:>14.2f} | {flat}")

print("\n*** WHY THE min()? ***")
print("Note the A<0 row at ratio=2.0: the objective is -2.0, NOT the clipped -1.2.")
print("The min() keeps the WORSE value, so a policy that has already moved far")
print("in a HARMFUL direction still receives a corrective gradient.")
print("Plain clipping alone would zero that out and leave the mistake in place.")
```

The `min` makes the objective **pessimistic**: it removes the incentive to move the policy far from `π_θ_old` in a single reuse window, while still correcting excursions that went the wrong way.

> **PPO = "Proximal" Policy Optimization** — *proximal* because it keeps the new policy **near** the old one. That is the entire name.

### The PPO family in one line each

| Method | Trust-region mechanism |
|---|---|
| **TRPO** (2015) | Hard KL constraint, solved with second-order optimisation. Correct but complex and expensive. |
| **PPO-Clip** (2017) | Clip the ratio. First-order, simple, works. **The default.** |
| **PPO-Penalty** | Adaptive KL penalty added to the objective instead of clipping. |
| **GRPO** (2024) | Drops the value network; computes advantage by normalising rewards *within a group* of responses to the same prompt. Cheaper — 3 models instead of 4. |

> ⚠️ **Don't confuse two different KLs.** PPO's trust region (`π_θ` vs `π_θ_old`, an *optimisation stability* device) is **not** the RLHF KL penalty of §6 (`π_θ` vs `π_ref = π_SFT`, an *anti-reward-hacking* device). Both are KL divergences; they serve completely different purposes and are usually both present. This distinction is a favourite interview trap.

### 🔗 Resources for Topic 25

- **[Schulman et al., Proximal Policy Optimization Algorithms (2017)](https://arxiv.org/abs/1707.06347)** — the PPO paper. Short (12 pages) and readable; §3 is the clipped objective you just implemented.
- **[The 37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)** — **the single most useful practical resource in this section.** Everything the paper omits and every codebase silently does.
- **[Cameron Wolfe — PPO for LLMs: A Guide for Normal People](https://cameronrwolfe.substack.com/p/ppo-llm)** *(slide 57)* — PPO explained specifically in the RLHF context.
- **[Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)** — pseudocode plus a clean reference implementation.

---

## Topic 26 — The PPO Learning Setup (slide 37)

The full loop, exactly as the slide lays it out:

```
   ┌───────────────────────────────────────────────────────────────┐
   │  OLD POLICY  π_θ_old   (LM with parameters θ_old)             │
   │                        ── OFFLINE ──                          │
   └────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
        ┌──────────────────────────────────────────────────┐
        │ 1. Sample trajectories                           │
        │    (sequences of states & actions)               │
        └────────────────────────┬─────────────────────────┘
                                 ▼
        ┌──────────────────────────────────────────────────┐
        │ 2. Calculate the rewards and advantages          │
        │    for all (state, action) pairs                 │
        └────────────────────────┬─────────────────────────┘
                                 ▼
        ┌──────────────────────────────────────────────────┐  ┐
        │ 3. Take a minibatch of trajectories              │  │
        └────────────────────────┬─────────────────────────┘  │ run for
                                 ▼                            │ K epochs
        ┌──────────────────────────────────────────────────┐  │
        │ 4. Run gradient ascent to train current policy π_θ│  │
        └────────────────────────┬─────────────────────────┘  ┘
                                 │
                    after K epochs: θ_old ← θ
                                 │
                                 └──────► back to step 1
```

### 🔑 The clarification note on slide 37 — read this twice

> *"**Not two copies of the policy.** Sample the initial trajectories and then save them in the memory. Use the saved trajectories to update the model."*

This is the single most misread part of PPO, and the professor called it out explicitly.

**You do NOT hold two full copies of the LLM for `π_θ_old`.** What you hold is:
- **One** policy network, being updated.
- The **saved rollouts** from before the updates began — including, crucially, the **log-probabilities `log π_θ_old(a_t|s_t)` recorded at sampling time**, stored as plain numbers in the buffer.

The ratio `r_t(θ) = exp(log π_θ(a_t|s_t) − log π_θ_old(a_t|s_t))` is computed against those **stored scalars**. `π_θ_old` exists as *data*, not as a model.

> The professor, from the transcript: *"though this is a model we are using, it is not required to be trained… we are kind of copying that after every iteration and generating trajectory."*

### 🔬 The rollout buffer — proving `π_θ_old` costs no extra memory

```python
from dataclasses import dataclass
import torch

@dataclass
class RolloutBuffer:
    """
    What PPO actually stores between rollout and update.
    NOTE: there is NO model here -- only tensors.
    """
    input_ids:     torch.Tensor   # (B, T)  the sampled trajectories
    attention_mask: torch.Tensor  # (B, T)
    old_logprobs:  torch.Tensor   # (B, T)  <-- pi_theta_old, AS NUMBERS
    ref_logprobs:  torch.Tensor   # (B, T)  <-- pi_ref, also numbers (§6)
    values:        torch.Tensor   # (B, T)  V(s_t) at sampling time
    advantages:    torch.Tensor   # (B, T)  GAE output
    returns:       torch.Tensor   # (B, T)  value targets


def collect_rollouts(policy, ref_model, value_head, reward_model, prompts):
    """PHASE 1 -- runs ONCE per iteration. Expensive (generation)."""
    with torch.no_grad():                       # no gradients during collection
        seqs = policy.generate(prompts, do_sample=True, temperature=1.0)
        old_lp = per_token_logprobs(policy,   seqs)   # ← recorded HERE...
        ref_lp = per_token_logprobs(ref_model, seqs)  # ← needs a REAL model
        values = value_head(seqs)
        scores = reward_model(seqs)
    # ... advantages via GAE (§4) ...
    return RolloutBuffer(...)


def ppo_update(policy, buffer, K=4, eps=0.2, minibatch=8):
    """PHASE 2 -- runs K times over the SAME buffer. Cheap."""
    for epoch in range(K):
        for mb in iterate_minibatches(buffer, minibatch):
            # Fresh forward pass through the CURRENT policy
            new_lp = per_token_logprobs(policy, mb.input_ids)   # requires grad

            # THE RATIO: current model vs. a STORED NUMBER.
            # pi_theta_old is never re-run. It is not in memory as a model.
            ratio = torch.exp(new_lp - mb.old_logprobs)

            unclipped = ratio * mb.advantages
            clipped   = torch.clamp(ratio, 1 - eps, 1 + eps) * mb.advantages
            policy_loss = -torch.min(unclipped, clipped).mean()   # minus -> ascent

            policy_loss.backward()
            optimizer.step(); optimizer.zero_grad()

    # After K epochs, the next collect_rollouts() implicitly sets theta_old <- theta
```

**⚠️ Note the contrast with the reference model:** `π_ref` genuinely **is** a second full model, because you must run *new* forward passes through it on updated sequences. `π_θ_old` is just a buffer of logprobs. Getting this distinction right is what turns "RLHF needs 4 models" from a memorised fact into an understood one.

### The complete PPO objective

Putting §4, §5 and §6 together, the loss actually implemented is:

$$L(\theta) = \underbrace{\mathbb{E}_t\big[\min(r_t A_t,\ \text{clip}(r_t, 1\pm\epsilon)A_t)\big]}_{\text{policy / clipped surrogate}} \;-\; \underbrace{c_1\,\mathbb{E}_t\big[(V_\phi(s_t) - G_t)^2\big]}_{\text{value-function regression}} \;+\; \underbrace{c_2\,\mathbb{E}_t\big[H(\pi_\theta(\cdot|s_t))\big]}_{\text{entropy bonus}}$$

and in the RLHF setting the reward fed into `A_t` is itself KL-shaped (§6):

$$\tilde{r}_t = r_\phi(x,y)\big|_{t=T} - \beta\,\log\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\text{ref}}(a_t\mid s_t)}$$

### 🔬 The full PPO loss in code

```python
import torch, torch.nn.functional as F

def ppo_loss(new_logprobs, old_logprobs, advantages,
             values, returns, entropy,
             eps=0.2, vf_coef=0.5, ent_coef=0.01, value_clip=0.2, old_values=None):
    """
    The complete PPO objective as implemented in TRL / OpenRLHF.
    All tensors are (B, T) except entropy (scalar).
    """
    # ---- 1. CLIPPED POLICY SURROGATE (Topic 25) ----
    ratio     = torch.exp(new_logprobs - old_logprobs)
    unclipped = ratio * advantages
    clipped   = torch.clamp(ratio, 1 - eps, 1 + eps) * advantages
    policy_loss = -torch.min(unclipped, clipped).mean()

    # ---- 2. VALUE-FUNCTION REGRESSION (§7 Topic 33) ----
    if old_values is not None:
        # Value clipping: same trust-region idea applied to V (a "37 details" item)
        v_clipped = old_values + torch.clamp(values - old_values,
                                             -value_clip, value_clip)
        value_loss = 0.5 * torch.max((values - returns) ** 2,
                                     (v_clipped - returns) ** 2).mean()
    else:
        value_loss = 0.5 * F.mse_loss(values, returns)

    # ---- 3. ENTROPY BONUS: keep the policy from collapsing ----
    entropy_loss = -entropy

    total = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

    # ---- DIAGNOSTICS -- log these every step; they are how you debug PPO ----
    with torch.no_grad():
        approx_kl   = ((ratio - 1) - (new_logprobs - old_logprobs)).mean()  # k3
        clip_frac   = ((ratio - 1).abs() > eps).float().mean()
    return total, {
        "policy_loss": policy_loss.item(),
        "value_loss":  value_loss.item(),
        "approx_kl":   approx_kl.item(),   # drift from pi_theta_old
        "clip_frac":   clip_frac.item(),   # >0.3 => K too large or LR too high
    }
```

| Term | Purpose |
|---|---|
| Clipped surrogate | Improve the policy without leaving the trust region |
| Value regression | Train `V_φ` so advantages are accurate (§7, Topic 33) |
| Entropy bonus | Prevent premature collapse to a deterministic policy |
| KL shaping | Prevent reward hacking / drift from the SFT model (§6) |

> 💡 **`clip_frac` is your most useful PPO diagnostic.** It's the fraction of tokens whose ratio left the `[1−ε, 1+ε]` band. Healthy: 0.05–0.2. Above ~0.3 means the policy is racing away from its sampler within a reuse window — lower `K` or the learning rate.

### Running PPO for real, with TRL

You will not implement PPO from scratch in production. This is what the real call looks like:

```python
# pip install trl transformers peft datasets
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification

BASE = "Qwen/Qwen2-0.5B-Instruct"
tok  = AutoTokenizer.from_pretrained(BASE)
tok.pad_token = tok.eos_token

# ---- THE FOUR MODELS (§7) ----
policy = AutoModelForCausalLMWithValueHead.from_pretrained(BASE)  # ← policy + VALUE head
ref    = AutoModelForCausalLM.from_pretrained(BASE)               # ← frozen reference (§6)
reward = AutoModelForSequenceClassification.from_pretrained(      # ← reward model (§8)
             "your-org/your-reward-model", num_labels=1)

config = PPOConfig(
    learning_rate      = 1e-6,   # ~1000x lower than SFT: the policy is already good
    batch_size         = 64,     # prompts per rollout
    mini_batch_size    = 8,
    num_ppo_epochs     = 4,      # K -- reuse each rollout batch 4 times (Topic 25)
    cliprange          = 0.2,    # epsilon
    cliprange_value    = 0.2,
    vf_coef            = 0.1,
    kl_coef            = 0.05,   # beta -- the ANTI-REWARD-HACKING KL (§6)
    gamma              = 1.0,
    lam                = 0.95,   # GAE lambda (§4)
    whiten_rewards     = True,   # advantage normalisation
)

trainer = PPOTrainer(
    args=config, processing_class=tok,
    model=policy, ref_model=ref, reward_model=reward,
    train_dataset=prompt_dataset,
)
trainer.train()
```

> ⚠️ **TRL's PPO API changed substantially in v0.12+.** Always check the [current PPOTrainer docs](https://huggingface.co/docs/trl/ppo_trainer) rather than copying older blog posts — argument names and the model-wrapping convention have moved more than once.

### The hyperparameters that matter

| Hyperparameter | Typical | Effect |
|---|---|---|
| `ε` (clip range) | 0.2 | Smaller = more conservative, slower, more stable |
| `K` (epochs per batch) | 4 | Larger = more sample-efficient, but more off-policy drift |
| Rollout batch size | 512–1024 prompts | Larger = lower gradient variance, more memory |
| `β` (KL coefficient) | 0.01–0.1 | Larger = stays closer to SFT, less reward gain (§6) |
| `γ`, `λ` | 1.0, 0.95 | Discount and GAE trade-off |
| LR | 1e-6 – 1e-5 | Much lower than SFT — the policy is already good |

### 💡 Learning thought

> PPO is best understood as **three separate patches on REINFORCE**, each fixing one named failure:
> 1. **Advantage** (§4) fixes *variance*.
> 2. **Importance ratio + K epochs** fixes *sample inefficiency*.
> 3. **Clipping** fixes the *instability introduced by patch 2*.
>
> Plus, for LLMs specifically, a fourth: **KL penalty** (§6) fixes *reward hacking*. PPO's reputation for being fiddly comes from having four interacting knobs, each guarding a different failure mode. Removing the whole stack is exactly DPO's pitch (§10) — and understanding *what you give up* requires having understood what each patch was for.

### 🔗 Resources for Topic 26

- **[TRL PPOTrainer docs](https://huggingface.co/docs/trl/ppo_trainer)** — the current API. Start here for anything you actually run.
- **[HuggingFace — Illustrating RLHF](https://huggingface.co/blog/rlhf)** — the diagram of the PPO-for-LLMs loop with all four models.
- **[Zheng et al., Secrets of RLHF in LLMs Part I: PPO (2023)](https://arxiv.org/abs/2307.04964)** — an empirical study of which PPO tricks actually matter for LLMs. Read after you've had one run go wrong.
- **[Shao et al., DeepSeekMath (GRPO) (2024)](https://arxiv.org/abs/2402.03300)** — §4.1 introduces GRPO and explains exactly why the value model is droppable.

---

## 🎯 Interview Questions — §5

### Conceptual

**Q1. Why is vanilla policy gradient sample-inefficient?**
> It is on-policy by construction: `∇J = E_{τ~π_θ}[·]` requires samples from the *current* policy. After a single parameter update the collected trajectories are from a stale distribution and give a biased gradient, so they must be discarded. One expensive rollout batch buys exactly one gradient step — and for LLMs generation is typically 20–100× the cost of the gradient step itself, so this wastes ~95% of the compute.

**Q2. What is the importance-sampling ratio in PPO and what does it do?**
> `r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)`. It re-weights samples drawn under `π_θ_old` so they give a valid estimate of the objective under `π_θ`, allowing the same rollout batch to be reused for K epochs. It measures how much more (or less) likely the current policy makes an action than the policy that sampled it.

**Q3. Why does PPO clip, and what exactly does clipping prevent?**
> Importance sampling is unbiased for any proposal but its *variance* explodes as the proposal drifts from the target — extreme ratios produce estimates dominated by a few samples. The clipped objective `min(r_t A_t, clip(r_t,1−ε,1+ε)A_t)` removes the *incentive* to move outside `[1−ε,1+ε]`: once an update would carry the ratio beyond the band in the direction that increases the objective, the gradient goes to zero. It's a cheap, first-order approximation to TRPO's hard trust region.

**Q4. Why the `min`? Why not just clip and be done?**
> The `min` makes the surrogate a **pessimistic lower bound** on the true objective. Plain clipping alone would be one-sided: if the ratio has already moved far in a *harmful* direction (e.g. ratio 2.0 with A<0), clipping would cap the penalty and leave the mistake uncorrected. The `min` keeps the worse (unclipped) value there, so bad excursions still receive a corrective gradient, while good excursions are capped.

**Q5. Does PPO keep two copies of the policy in memory?**
> **No** — and this is the misconception the lecture explicitly flags. `π_θ_old` exists as the **stored log-probabilities recorded at sampling time**, saved in the rollout buffer as plain numbers. The ratio is `exp(logπ_θ − logπ_θ_old_stored)`. Contrast with the **reference model** `π_ref`, which genuinely *is* a second frozen model in memory, because the KL penalty needs fresh forward passes on newly generated sequences.

**Q6. Distinguish the two KL divergences in an RLHF pipeline.**
> (a) **Trust-region KL** — between `π_θ` and `π_θ_old`, an *optimisation-stability* device controlling how far a single reuse window can move the policy; implemented in PPO via ratio clipping (or an adaptive penalty). (b) **Alignment KL** — between `π_θ` and `π_ref = π_SFT`, an *anti-reward-hacking* constraint anchoring the final model to its supervised starting point (§6). Different reference distributions, different purposes; both present simultaneously.

**Q7. On-policy, off-policy — where does PPO sit?**
> PPO is **near-on-policy**. It is not truly off-policy (it can't learn from arbitrary old data — the ratio would be unreliable), but it isn't strictly on-policy either, since it reuses a batch for K epochs. The clip range and K jointly bound how off-policy it is allowed to become.

**Q8. What is GRPO and why was it introduced?**
> Group Relative Policy Optimization: sample G responses for the *same* prompt, and compute each one's advantage by normalising its reward against the group's mean and standard deviation — `A_i = (r_i − mean(r))/std(r)`. This **eliminates the value network entirely**, reducing RLHF from 4 models to 3 and removing the value network's training instability. It trades a learned per-token baseline for a group-empirical per-response one; it works well when you can afford several samples per prompt, and is the basis of much recent reasoning-model training.

### Applied

**Q9. Your PPO training is unstable — reward climbs then collapses. Walk through your debugging.**
> 1. **Check the KL to `π_ref`.** If it's blowing up, the policy has drifted off the SFT manifold; raise β or lower LR. 2. **Check `clip_frac`.** Healthy is 0.05–0.2; above 0.3 means K is too large or LR too high. 3. **Check the value network's explained variance.** Poor `V` ⇒ noisy/incorrect advantages ⇒ garbage updates. 4. **Check for reward hacking** — inspect actual generations, not just the reward curve; look for length inflation and formulaic openers. 5. **Check advantage normalisation** (`whiten_rewards`). 6. Reduce K to 1–2 and see if stability returns; that isolates off-policy drift as the cause.

**Q10. How would you reduce the cost of an RLHF run, given generation dominates the wall clock?**
> Increase K (more updates per rollout, bounded by `clip_frac`); shorten `max_new_tokens`; use a smaller reward model or distil it; batch generation aggressively with a fast inference engine (vLLM/paged attention) and decouple generation from training; use LoRA on the policy so the reference model can be the same base weights with adapters disabled (removing one full model from memory); switch to GRPO to drop the value network; or move to **DPO**, which eliminates online generation entirely.

**Q11. Someone proposes reusing a rollout batch for K=50 epochs to save cost. What happens?**
> The policy drifts far from `π_θ_old`, so nearly every ratio leaves the clip band (`clip_frac` → ~1.0) and the gradient for those tokens becomes zero — most of the extra epochs do nothing useful. Worse, the tokens still inside the band get repeatedly over-fitted to a stale, noisy advantage estimate, and the effective trust region is violated in aggregate. Result: wasted compute plus overfitting to one rollout batch. K=4 is standard because it's roughly where the marginal epoch stops carrying signal.

**Q12. Why is the PPO learning rate ~1e-6 when SFT uses 1e-5 to 1e-4?**
> Three reasons. (a) The policy starts from a competent SFT model — you're making a behavioural adjustment, not learning a task. (b) The gradient signal is noisy (advantage estimates from sampled rollouts), so large steps amplify noise. (c) Large steps drive the ratio out of the clip band immediately, wasting the K-epoch reuse and violating the trust region. Together these make RLHF learning rates one to two orders of magnitude smaller than SFT's.

### Rapid-fire

| Question | Answer |
|---|---|
| What does "proximal" mean? | Keeps `π_θ` near `π_θ_old` |
| Ratio formula? | `π_θ(a\|s) / π_θ_old(a\|s)` |
| Typical ε? | 0.2 |
| Typical K? | 4 |
| Is `π_θ_old` a model in memory? | **No** — stored logprobs |
| Is `π_ref` a model in memory? | **Yes** — frozen full model |
| PPO predecessor with a hard KL constraint? | TRPO |
| Variant that drops the value net? | GRPO |
| Three terms in the PPO loss? | Clipped surrogate, value regression, entropy bonus |
| Healthy `clip_frac`? | 0.05–0.2 |
| Generation vs. gradient-step cost? | Generation 20–100× more expensive |

---

## ✅ Section self-check

1. Explain in causal terms why a rollout batch "expires" after one update.
2. Write the importance ratio and say what `r_t > 1` means.
3. Draw the clipped objective for `A > 0` and `A < 0`, and explain the `min` using the `A<0, ratio=2.0` case.
4. Explain why `π_θ_old` costs no extra GPU memory but `π_ref` does.
5. Name the two distinct KL divergences in RLHF and their reference distributions.
6. List PPO's four patches on REINFORCE and the failure each one fixes.
7. **Hands-on:** run the importance-sampling experiment. At what proposal divergence does the max ratio exceed 5, and what does that imply for ε?

---

**Previous:** [§4 — Policy Gradient & Variance Reduction](04-policy-gradient.md) · **Next:** [§6 — Reward Hacking & KL Control](06-reward-hacking-kl.md) · [Index](00-INDEX.md)
