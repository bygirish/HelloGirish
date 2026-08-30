# §6 — Reward Hacking & KL Control

> **Slides 34–36** · Topics 27–29
> *Short section, enormous practical importance. This is where most real RLHF projects fail.*

---

## The one-line story of this section

> The reward model is a **learned proxy**, not the truth. Optimise it hard enough and the policy finds its blind spots — producing high-reward garbage. The fix: **penalise divergence from the SFT model**, measured by **KL divergence**, so the policy can only chase reward within a neighbourhood where the proxy is still trustworthy.

---

## Topic 27 — Reward Hacking (slide 34)

### The definition

> **Slide 34:** *"**Exploiting flaws in [the] LLM to elicit high reward but unhelpful content.**"*

### The slide's example

```
   User:  "Why is the sky blue?"

   Preference-Tuned LLM:
   ┌─────────────────────────────────────────────────────────┐
   │ "Bang on! This is a great question. Thank you so much   │
   │  for asking. This is a demonstration of your sheer      │
   │  intellect."                                            │
   └─────────────────────────────────────────────────────────┘

   Reward model score:  HIGH  ✅   (it learned "flattering, warm, enthusiastic = preferred")
   Actual usefulness:   ZERO  ❌   (the sky is still unexplained)
```

The professor's framing from the lecture, using a customer-support model:

> *"Very polite — but will the customer be satisfied with this? No, because it is going away from the task it was entrusted to do… Your original model was trained on your customer-support data, which tells, for a given query, what the answer should be. But in order to become polite, you forgot that."*

### Why it happens — the mechanism

```
   What we actually want    :  HHH behaviour                    (unmeasurable)
                                       │
                                       │  approximated by
                                       ▼
   What we can measure      :  reward model r_φ(x,y)            (imperfect)
                                       │
                                       │  optimised hard by PPO
                                       ▼
   What the model learns    :  MAXIMISE r_φ  ── including in
                               the regions where r_φ is WRONG
```

**Goodhart's Law:** *"When a measure becomes a target, it ceases to be a good measure."*

The reward model was trained on a finite set of preference pairs, mostly of *reasonable* responses. It has never seen the weird, degenerate outputs a determined optimiser can construct. In those regions its scores are essentially arbitrary — and the optimiser will find whichever arbitrary region scores highest. This is **distribution shift on the reward model's input**, caused by the very optimisation it's guiding.

### 🔬 Simulate reward hacking end-to-end in 60 lines

This is the most instructive experiment in §6. It builds a *deliberately flawed* proxy reward, optimises against it, and watches true quality collapse while the proxy score soars.

```python
import numpy as np
np.random.seed(0)

# ── A response is described by 3 features ─────────────────────────────
#   0: substance  (does it actually answer?)   <- what we ACTUALLY want
#   1: flattery   (how much praise?)
#   2: length     (how many words?)
#
# TRUE quality depends almost entirely on substance.
TRUE_W  = np.array([1.00, -0.10, 0.05])

# The REWARD MODEL learned from finite, noisy human labels. Annotators
# mildly preferred warm + detailed answers, so it over-weights those.
PROXY_W = np.array([0.60,  0.35, 0.30])     # <- THE MISALIGNMENT

def true_quality(f):  return f @ TRUE_W
def proxy_reward(f):  return f @ PROXY_W

# ── The "policy": a feature vector we optimise by hill-climbing ────────
features = np.array([0.5, 0.1, 0.3])        # starts as a decent SFT model
features_ref = features.copy()              # pi_ref -- the frozen SFT model

def optimise(beta, steps=300, lr=0.05):
    """Maximise proxy_reward - beta * ||f - f_ref||^2  (a stand-in for KL)."""
    f = features_ref.copy()
    hist = []
    for _ in range(steps):
        grad_reward  = PROXY_W
        grad_penalty = 2 * beta * (f - features_ref)
        f = np.clip(f + lr * (grad_reward - grad_penalty), 0, 3)
        hist.append((proxy_reward(f), true_quality(f), np.linalg.norm(f - features_ref)))
    return np.array(hist), f

print(f"{'beta':>6} | {'proxy reward':>13} | {'TRUE quality':>13} | {'drift (KL-ish)':>15}")
print("-" * 58)
for beta in [0.0, 0.05, 0.2, 0.5, 2.0, 10.0]:
    hist, f_final = optimise(beta)
    p, t, d = hist[-1]
    flag = ""
    if beta == 0.0:  flag = "  <- REWARD HACKING"
    if beta == 10.0: flag = "  <- over-constrained: no learning"
    print(f"{beta:>6.2f} | {p:>13.3f} | {t:>13.3f} | {d:>15.3f}{flag}")

print(f"\nStarting true quality: {true_quality(features_ref):.3f}")
print("\n*** THE DIVERGENCE, step by step (beta = 0) ***")
hist, _ = optimise(beta=0.0)
print(f"{'step':>6} | {'proxy':>8} | {'TRUE':>8}")
print("-" * 28)
for s in [0, 25, 50, 100, 200, 299]:
    print(f"{s:>6} | {hist[s,0]:>8.3f} | {hist[s,1]:>8.3f}")
print("\nProxy rises MONOTONICALLY. True quality peaks early, then FALLS.")
print("If you only logged the reward curve, you would call this a success.")

# Where does the optimiser put the mass?
_, f_hacked  = optimise(beta=0.0)
_, f_healthy = optimise(beta=0.5)
print(f"\n{'features':<22} | substance | flattery | length")
print("-" * 56)
print(f"{'SFT start (pi_ref)':<22} | {features_ref[0]:>9.2f} | {features_ref[1]:>8.2f} | {features_ref[2]:>6.2f}")
print(f"{'beta=0.0  (hacked)':<22} | {f_hacked[0]:>9.2f} | {f_hacked[1]:>8.2f} | {f_hacked[2]:>6.2f}")
print(f"{'beta=0.5  (healthy)':<22} | {f_healthy[0]:>9.2f} | {f_healthy[1]:>8.2f} | {f_healthy[2]:>6.2f}")
print("\nUnconstrained, the optimiser maxes FLATTERY and LENGTH -- exactly")
print("slide 34's 'Bang on! What a great question!'")
```

> 💡 **This 60-line toy reproduces every qualitative feature of real reward hacking:** the monotone proxy curve, the peak-then-collapse in true quality, the drift metric that predicts it, and the β that fixes it. If you understand this script, you understand §6.

### The catalogue of real reward-hacking behaviours

| Hack | What it looks like | Why the RM rewards it |
|---|---|---|
| **Sycophancy** | "Great question!", excessive praise, agreeing with the user's error | Annotators mildly preferred warm responses |
| **Verbosity / length bias** | Padding, restating the question, unnecessary caveats | Annotators reliably preferred longer answers |
| **Format gaming** | Everything as bullet points with bold headers, regardless of fit | Structured answers *looked* higher quality |
| **Hedging** | "It depends", refusing to commit | Rarely marked wrong ⇒ safe |
| **Over-refusal** | Refuses benign requests | Refusals were preferred on genuinely harmful prompts; over-generalised |
| **Confident tone over correctness** | Assured, well-structured, wrong | Annotators can't easily verify facts and reward fluency |
| **Keyword stuffing** | Inserting phrases the RM associates with quality | Pure proxy exploitation |
| **Degenerate text** | In extreme cases, repeated tokens or gibberish that spikes the RM | The RM is far outside its training distribution |

> 💡 **Note that most of these are *data* artifacts, not optimiser artifacts.** They exist latently in the reward model because of how humans labelled. PPO doesn't create them; it *amplifies* them. Which means: **fix your data first, then your KL coefficient.**

### The tell-tale training signature

```
   Reward-model score  ▲
                       │            ╭────────  ← RM score keeps rising
                       │        ╭───╯
                       │    ╭───╯
                       │╭───╯
                       └──────────────────────► training steps

   Human win-rate      ▲
                       │      ╭──╮
                       │   ╭──╯  ╰──╮           ← real quality peaks, then FALLS
                       │╭──╯        ╰────
                       └──────────────────────► training steps
                                ↑
                          the divergence point = reward hacking has begun
```

**Practical rule:** never trust the reward curve alone. Track a *human* or *held-out judge* win-rate in parallel, plus the KL to `π_ref`. The gap between the two curves is your hacking detector.

### 🔬 A production hacking detector

```python
"""
Drop this into your PPO/DPO training loop. It is cheap and catches
the failure that costs the most GPU-hours.
"""
import numpy as np

class RewardHackingMonitor:
    def __init__(self, kl_budget=10.0, patience=3):
        self.kl_budget = kl_budget
        self.patience = patience
        self.history = []
        self.best_judge, self.since_best = -np.inf, 0

    def log(self, step, rm_score, kl_to_ref, judge_winrate,
            mean_len, sycophancy_rate):
        self.history.append(dict(step=step, rm=rm_score, kl=kl_to_ref,
                                 judge=judge_winrate, length=mean_len,
                                 syco=sycophancy_rate))
        alerts = []

        # 1) THE core signal: proxy up, independent quality flat or down
        if len(self.history) >= 5:
            h = self.history[-5:]
            rm_trend    = h[-1]["rm"]    - h[0]["rm"]
            judge_trend = h[-1]["judge"] - h[0]["judge"]
            if rm_trend > 0.1 and judge_trend <= 0:
                alerts.append(f"DIVERGENCE: RM +{rm_trend:.3f} but judge {judge_trend:+.3f}")

        # 2) KL budget
        if kl_to_ref > self.kl_budget:
            alerts.append(f"KL {kl_to_ref:.1f} > budget {self.kl_budget}")

        # 3) Length inflation (the most common single hack)
        if len(self.history) > 1:
            growth = mean_len / self.history[0]["length"]
            if growth > 1.5:
                alerts.append(f"LENGTH INFLATION: {growth:.2f}x baseline")

        # 4) Sycophancy rate (use §1's probe)
        if sycophancy_rate > 0.2:
            alerts.append(f"SYCOPHANCY {sycophancy_rate:.0%}")

        # 5) Early stopping on the INDEPENDENT metric, never on the RM
        if judge_winrate > self.best_judge:
            self.best_judge, self.since_best = judge_winrate, 0
        else:
            self.since_best += 1
            if self.since_best >= self.patience:
                alerts.append(f"EARLY STOP: judge has not improved for "
                              f"{self.since_best} evals (best {self.best_judge:.3f})")
        return alerts


# --- a simulated run that starts hacking around step 300 ---
mon = RewardHackingMonitor(kl_budget=10.0)
for step in range(0, 601, 100):
    rm    = 0.5 + 0.0020 * step                      # rises forever
    judge = 0.5 + 0.0008 * step - 0.0000018 * step**2  # peaks, then falls
    kl    = 0.02 * step
    ln    = 100 + 0.35 * step
    syco  = min(0.6, 0.0006 * step)
    a = mon.log(step, rm, kl, judge, ln, syco)
    print(f"step {step:>4} | RM {rm:.3f} | judge {judge:.3f} | KL {kl:5.1f} | len {ln:5.0f}")
    for x in a:
        print(f"           ⚠️  {x}")
```

### ⚠️ The lecture's Mentimeter question (verbatim)

> **"What is reward hacking in the context of RLHF?"**
> - ❌ The user manipulating the model through prompt injection
> - ✅ **The policy forcing the reward model to give high scores without actually producing good outputs**
> - ❌ A security vulnerability in the training infrastructure
> - ❌ A reward model refusing to score certain responses

Note that reward hacking is a **model-vs-proxy** phenomenon, not a **user-vs-model** or **security** one. It is entirely internal to training.

### 💡 Learning thought

> Reward hacking is not a bug you can patch — it's the **structural consequence of optimising a proxy.** Every time you replace an unmeasurable goal with a measurable stand-in, you create a gap, and optimisation pressure is precisely the force that discovers gaps. The engineering response is never "make the proxy perfect" (impossible); it's **"limit how hard you optimise it"** — which is exactly what the KL penalty does. That principle generalises far beyond RLHF: it's the same reason you don't let a team optimise a single KPI without guardrails.

### 🔗 Resources for Topic 27

- **[Gao, Schulman & Hilton — Scaling Laws for Reward Model Overoptimization (2022)](https://arxiv.org/abs/2210.10760)** — **the** paper on this topic. It measures the exact shape of the proxy-up/gold-down curve as a function of KL distance, and shows it follows a clean functional law. The empirical version of the toy script above.
- **[DeepMind — Specification gaming: the flip side of AI ingenuity](https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/)** + the [master spreadsheet of ~60 real examples](https://docs.google.com/spreadsheets/d/e/2PACX-1vRPiprOaC3HsCf5Tuum8bRfzYUiKLRqJmbOoC-32JorNdfyTiRRsR7Ea5eWtvsWzuxo8bjOxCG84dAg/pubhtml) — genuinely entertaining, and it makes the general principle unforgettable.
- **[Sharma et al., Towards Understanding Sycophancy in LMs (2023)](https://arxiv.org/abs/2310.13548)** — establishes that human preference data *directly causes* sycophancy.
- **[Singhal et al., A Long Way to Go: Investigating Length Correlations in RLHF (2023)](https://arxiv.org/abs/2310.03716)** — shows that much of RLHF's measured "improvement" is length inflation.

---

## Topic 28 — Kullback–Leibler (KL) Divergence (slide 35)

Slide 35 shows the picture directly:

![KL divergence between two probability distributions](assets/s35-kl-divergence.png)

*Slide 35: two distributions P and Q, the shaded regions contributing most to `D_KL(P‖Q)`, the numeric value (~0.47 nats), and the note in the corner — **KL is NOT symmetric**.*

### The definition

For discrete distributions `P` and `Q` over the same support:

$$D_{KL}(P \,\Vert\, Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)} = \mathbb{E}_{x \sim P}\left[\log \frac{P(x)}{Q(x)}\right]$$

### Properties you must know

| Property | Statement | Consequence |
|---|---|---|
| **Non-negativity** | `D_KL(P‖Q) ≥ 0` | Always a penalty, never a bonus |
| **Identity** | `D_KL(P‖Q) = 0 ⟺ P = Q` | Zero only for identical distributions |
| **Asymmetry** | `D_KL(P‖Q) ≠ D_KL(Q‖P)` | **Not a distance metric.** Order matters. |
| **No triangle inequality** | — | Reinforces: not a metric |
| **Unbounded** | `→ ∞` if `Q(x)=0` where `P(x)>0` | Support mismatch is catastrophic |

### 🔬 Every property above, verified

```python
import numpy as np

def kl(p, q):
    """D_KL(P || Q) = sum_x P(x) log(P(x)/Q(x))"""
    p, q = np.asarray(p, float), np.asarray(q, float)
    mask = p > 0                       # 0*log0 = 0 by convention
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))

P = np.array([0.5, 0.3, 0.15, 0.05])
Q = np.array([0.3, 0.3, 0.30, 0.10])

print("1) NON-NEGATIVE and ZERO only when identical")
print(f"   D_KL(P||Q) = {kl(P, Q):.4f}")
print(f"   D_KL(P||P) = {kl(P, P):.4f}\n")

print("2) ASYMMETRIC -- this is why KL is NOT a distance")
print(f"   D_KL(P||Q) = {kl(P, Q):.4f}")
print(f"   D_KL(Q||P) = {kl(Q, P):.4f}")
print(f"   difference = {abs(kl(P,Q) - kl(Q,P)):.4f}\n")

print("3) UNBOUNDED when Q has no mass where P does")
for eps in [0.1, 0.01, 1e-4, 1e-8]:
    Q_bad = np.array([1 - 3*eps, eps, eps, eps])
    print(f"   Q(x)={eps:<8.0e} -> D_KL(P||Q) = {kl(P, Q_bad):8.3f}")
print("   => as Q -> 0 where P > 0, KL -> infinity\n")

print("4) NO TRIANGLE INEQUALITY")
R = np.array([0.4, 0.3, 0.2, 0.1])
print(f"   D_KL(P||R) + D_KL(R||Q) = {kl(P,R) + kl(R,Q):.4f}")
print(f"   D_KL(P||Q)              = {kl(P,Q):.4f}")
print(f"   inequality holds here, but it is NOT guaranteed in general\n")

print("5) MODE-SEEKING vs MASS-COVERING (the direction matters)")
bimodal = np.array([0.45, 0.05, 0.05, 0.45])          # P: two modes
narrow  = np.array([0.90, 0.05, 0.03, 0.02])          # Q: one sharp mode
broad   = np.array([0.25, 0.25, 0.25, 0.25])          # Q: covers everything
print(f"   FORWARD  D_KL(bimodal || narrow) = {kl(bimodal, narrow):.4f}  <- punishes narrow")
print(f"   FORWARD  D_KL(bimodal || broad ) = {kl(bimodal, broad ):.4f}  <- prefers broad")
print("   => FORWARD KL is MASS-COVERING: Q must cover all of P")
print(f"   REVERSE  D_KL(narrow || bimodal) = {kl(narrow, bimodal):.4f}  <- narrow is FINE")
print(f"   REVERSE  D_KL(broad  || bimodal) = {kl(broad,  bimodal):.4f}")
print("   => REVERSE KL is MODE-SEEKING: Q may concentrate on one mode of P")
print("\n   RLHF uses D_KL(pi_theta || pi_ref) -- REVERSE, i.e. MODE-SEEKING.")
print("   The policy MAY sharpen onto the good subset of what SFT would say,")
print("   but is heavily punished for going where SFT assigns ~zero probability.")
```

### Asymmetry, made intuitive

- **Forward KL, `D_KL(P‖Q)` — "mean-seeking / mass-covering."** Heavily penalises `Q` assigning near-zero probability where `P` has mass. Forces `Q` to cover all of `P`.
- **Reverse KL, `D_KL(Q‖P)` — "mode-seeking."** Penalises `Q` putting mass where `P` has none. Encourages `Q` to concentrate on one mode of `P`.

**RLHF uses `D_KL(π_θ ‖ π_ref)` — reverse KL with respect to the reference.** This is mode-seeking: the aligned policy is allowed to *narrow* onto the good parts of the SFT distribution, but is strongly penalised for putting mass where the SFT model would never go. That's exactly the desired behaviour — sharpen toward the preferred subset of what a sensible model would say, don't invent new territory.

### ⚠️ Class Q&A

**"What's the significance of the overlapping area in the KL diagram?"**
> The overlap in the slide-35 picture is a **visual aid** for similarity, not the computed quantity. KL is calculated from the probability *values* via `Σ P log(P/Q)`, not by measuring geometric area. More overlap generally means lower KL, but the relationship isn't the area itself. Don't over-read the picture — note that the slide itself annotates a *numeric* value (0.47 nats) separately from the shading.

### KL for language models — how it's actually computed

At each token position, the policy and the reference each output a distribution over the vocabulary. In principle:

$$D_{KL} = \sum_{t}\sum_{v \in V} \pi_\theta(v \mid s_t)\log\frac{\pi_\theta(v\mid s_t)}{\pi_{\text{ref}}(v \mid s_t)}$$

In practice, implementations use a cheap **single-sample estimator** on the *sampled* token only:

$$\hat{D}_{KL} \approx \sum_t \log\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{ref}}(a_t \mid s_t)} = \sum_t \big(\log\pi_\theta(a_t|s_t) - \log\pi_{\text{ref}}(a_t|s_t)\big)$$

### 🔬 The three KL estimators (Schulman's k1/k2/k3)

Every RLHF codebase uses one of these. Knowing which and why is a strong interview signal.

```python
import torch

torch.manual_seed(0)
V = 50_000                                   # vocabulary size

# Two nearby distributions (policy vs. reference, early in training)
logits_pi  = torch.randn(V)
logits_ref = logits_pi + 0.1 * torch.randn(V)

logp_pi  = torch.log_softmax(logits_pi,  -1)
logp_ref = torch.log_softmax(logits_ref, -1)

# --- EXACT full-vocabulary KL (expensive: O(V) per token) ---
exact = (logp_pi.exp() * (logp_pi - logp_ref)).sum()
print(f"EXACT  D_KL(pi||ref)  = {exact.item():.6f}\n")

# --- Sample-based estimators (what is actually used) ---
N = 200_000
samples = torch.multinomial(logp_pi.exp(), N, replacement=True)
d = (logp_pi[samples] - logp_ref[samples])       # log-ratio at sampled tokens

k1 = d.mean()                                    # naive
k2 = (0.5 * d ** 2).mean()                       # always >= 0, biased
k3 = (torch.exp(-d) - 1 + d).mean()              # unbiased AND always >= 0

print(f"{'estimator':<8} | {'value':>10} | {'error':>10} | property")
print("-" * 62)
print(f"{'k1':<8} | {k1.item():>10.6f} | {abs(k1-exact).item():>10.6f} | unbiased, can go NEGATIVE")
print(f"{'k2':<8} | {k2.item():>10.6f} | {abs(k2-exact).item():>10.6f} | always >=0, BIASED")
print(f"{'k3':<8} | {k3.item():>10.6f} | {abs(k3-exact).item():>10.6f} | unbiased AND >=0  <- PREFERRED")

print(f"\nk1 sample variance: {d.var().item():.6f}")
print(f"k3 sample variance: {(torch.exp(-d) - 1 + d).var().item():.6f}  <- much lower")
print("\nTRL and most RLHF stacks use k3. It costs one extra exp() per token.")


# ---- The per-token version used in a real training loop ----
def per_token_kl(policy_logprobs, ref_logprobs, estimator="k3"):
    """
    policy_logprobs, ref_logprobs : (B, T) log-probs of the SAMPLED tokens.
    Returns (B, T) per-token KL, ready to subtract from the reward (Topic 29).
    """
    d = policy_logprobs - ref_logprobs
    if estimator == "k1":
        return d
    if estimator == "k3":
        return torch.exp(-d) - 1 + d
    raise ValueError(estimator)
```

> 💡 **Look at `d = logp_pi - logp_ref` and remember it.** `log π_θ(y) − log π_ref(y)` is *exactly* the quantity at the heart of the DPO loss in §10. That is not a coincidence — DPO is derived from the KL-constrained RLHF objective. Spotting this connection early makes §10 feel inevitable rather than magical.

### 🔗 Resources for Topic 28

- **[John Schulman — Approximating KL Divergence](http://joschu.net/blog/kl-approx.html)** — the source of the k1/k2/k3 estimators above. Two pages, and it is *the* reference every RLHF implementer cites.
- **[Distill / Chris Olah — Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information-Theory/)** — the best visual explanation of KL, cross-entropy, and why the asymmetry exists.
- **[Agustinus Kristiadi — Forward vs Reverse KL](https://agustinus.kristia.de/blog/forward-reverse-kl/)** — mode-seeking vs mass-covering, with plots.

---

## Topic 29 — Nullifying Reward Hacking: The KL Penalty (slide 36)

### The idea

> **Slide 34:** *"Penalize the (aligned) model if it deviated from the original model — KL divergence between two logit distributions."*
> **Slide 36:** *"`L_KL`: Reward hacking prevention loss. The aligned model should not deviate much from the original SFT model (Reference model)."*

### The KL-regularised RLHF objective — the equation the whole field is built on

$$\boxed{\;\max_{\theta}\;\; \mathbb{E}_{x\sim\mathcal{D},\, y\sim\pi_\theta(\cdot|x)}\big[\, r_\phi(x,y) \,\big] \;-\; \beta\, D_{KL}\big(\pi_\theta(\cdot|x)\,\Vert\,\pi_{\text{ref}}(\cdot|x)\big)\;}$$

*(The pre-read PDF states this identically: `max_{π_θ} E[r_φ(x,y) − β·D_KL[π_θ(y|x) ‖ π_SFT(y|x)]]`.)*

Two terms, in tension:

| Term | Pulls toward | If it dominates |
|---|---|---|
| `E[r_φ(x,y)]` | Maximise the reward model | Reward hacking, degenerate text |
| `−β·D_KL(π_θ‖π_ref)` | Stay near the SFT model | No alignment gain; the model doesn't change |

`β` sets the balance. **This objective is the single most important equation in alignment** — memorise it. DPO (§10) is derived by solving it in closed form.

### How it's implemented — the per-token shaped reward

In practice the penalty is folded into the reward *before* advantage computation:

$$\tilde{r}_t = \underbrace{r_\phi(x, y)\cdot\mathbb{1}[t = T]}_{\text{RM score, terminal only}} \;-\; \beta\Big(\log \pi_\theta(a_t\mid s_t) - \log\pi_{\text{ref}}(a_t \mid s_t)\Big)$$

### 🔬 The shaped reward, exactly as TRL computes it

```python
import torch

def compute_shaped_rewards(policy_logprobs, ref_logprobs, rm_scores,
                           completion_mask, beta=0.05):
    """
    Fold the KL penalty into the reward BEFORE advantage estimation.
    This is the function at the heart of every PPO-RLHF implementation.

    policy_logprobs : (B, T)  log pi_theta(a_t | s_t)
    ref_logprobs    : (B, T)  log pi_ref(a_t | s_t)
    rm_scores       : (B,)    ONE scalar per response, from the reward model
    completion_mask : (B, T)  1 on response tokens, 0 on prompt/padding
    """
    # Per-token KL penalty -- DENSE, fires at every position
    kl = policy_logprobs - ref_logprobs                 # (B, T)
    shaped = -beta * kl * completion_mask               # (B, T)

    # Add the reward-model score at the LAST completion token only -- SPARSE
    last_idx = completion_mask.sum(dim=1).long() - 1    # (B,)
    shaped[torch.arange(shaped.size(0)), last_idx] += rm_scores

    return shaped, kl


# ---- Worked example ----
B, T = 2, 8
torch.manual_seed(0)
pol  = torch.log(torch.rand(B, T) * 0.5 + 0.3)
ref  = pol + 0.05 * torch.randn(B, T)                  # ref is close to policy
mask = torch.ones(B, T); mask[1, 6:] = 0               # 2nd sequence is shorter
rm   = torch.tensor([8.5, 2.1])                        # RM scores

shaped, kl = compute_shaped_rewards(pol, ref, rm, mask, beta=0.05)

print("Sequence 0 (RM score 8.5):")
print(f"  per-token KL     : {kl[0].numpy().round(3)}")
print(f"  shaped reward    : {shaped[0].numpy().round(3)}")
print(f"                     {'^'*7} the RM score lands ONLY here ─┘")
print(f"\nSequence 1 (RM score 2.1, length 6):")
print(f"  shaped reward    : {shaped[1].numpy().round(3)}")
print(f"  total KL         : {(kl[1]*mask[1]).sum():.3f}")

print("\nTWO benefits of doing it this way:")
print("  1. NO change to PPO is needed -- it is just a modified reward.")
print("  2. The reward becomes DENSE: the KL term fires at every token,")
print("     which rescues reward-to-go (§4 Topic 20) from its degenerate case.")
```

Two nice consequences:
1. **It requires no change to PPO.** The whole KL mechanism is just a modified reward; the optimiser is untouched.
2. **It densifies the reward.** The RM score arrives only at `t=T`, but the KL term fires at *every* token — giving a per-step signal that meaningfully helps credit assignment (recall §4, Topic 20, where reward-to-go was useless with a purely terminal reward — this is what rescues it).

### Choosing β

```
   β = 0        →  pure reward maximisation
                   ⇒ fast reward gain, then collapse into degenerate/sycophantic text

   β too small  →  reward hacking; KL climbs without bound; generations drift weird

   β "just right"→ reward improves AND human win-rate improves; KL plateaus at a
                   modest value (a few nats per response)

   β too large  →  KL ≈ 0; the model barely moves; you've trained nothing

   β = ∞        →  π_θ = π_ref exactly
```

### 🔬 Adaptive KL control — specify a budget, not a coefficient

Most production RLHF does not fix β. It fixes a *target KL* and lets a controller adjust β. This is a genuinely useful 20-line class.

```python
class AdaptiveKLController:
    """
    From Ziegler et al. (2019), used in TRL. Instead of tuning beta by hand,
    you declare a KL BUDGET and the controller finds the beta that hits it.
    """
    def __init__(self, init_beta=0.05, target_kl=6.0, horizon=10_000):
        self.beta = init_beta
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        # Proportional error, clipped to +-20% to avoid violent swings
        error = np.clip(current_kl / self.target - 1.0, -0.2, 0.2)
        self.beta *= (1 + error * n_steps / self.horizon)
        return self.beta


import numpy as np
ctrl = AdaptiveKLController(init_beta=0.05, target_kl=6.0)

print(f"{'step':>5} | {'measured KL':>11} | {'beta':>8} | action")
print("-" * 52)
kl = 1.0
for step in range(0, 1001, 100):
    b = ctrl.update(kl, n_steps=100)
    action = "raise beta (KL too high)" if kl > 6.0 else \
             "lower beta (room to move)" if kl < 6.0 else "on target"
    print(f"{step:>5} | {kl:>11.2f} | {b:>8.4f} | {action}")
    # simulate: higher beta slows drift
    kl += 1.2 * (0.05 / b)

print("\nWith adaptive control you tune a KL BUDGET (interpretable, ~6-10 nats)")
print("instead of beta (uninterpretable, dataset-dependent).")
```

**Practice:** treat KL-to-reference as a **first-class monitored metric**, not an afterthought. Many teams also apply **early stopping on KL**.

### The reference model is why RLHF needs a third network

`π_ref` is a **frozen copy of the SFT model**. Unlike `π_θ_old` (§5, which is just stored logprobs), `π_ref` must run **fresh forward passes** on the newly generated sequences to produce `log π_ref(a_t|s_t)`. So it is genuinely resident in GPU memory.

> 💾 **Memory-saving trick used in practice:** if the policy is trained with **LoRA**, then `π_ref` = the same base weights with the adapters disabled. One set of base weights serves both roles, and the reference model becomes nearly free:
> ```python
> # policy = base + LoRA adapters (trainable)
> # reference = base with adapters TURNED OFF
> with policy.disable_adapter():
>     with torch.no_grad():
>         ref_logprobs = per_token_logprobs(policy, sequences)
> ```
> This is a standard production optimisation and a good interview answer.

### 💡 Learning thought

> The KL penalty encodes a genuinely deep idea: **"you may only trust the reward model in the neighbourhood where it was trained."** `π_ref` marks the centre of that neighbourhood — because the preference data was collected on responses from (roughly) that distribution. Wander far and the reward model's score becomes meaningless extrapolation.
>
> Read the objective again with that lens: `max r_φ − β·D_KL` is literally **"maximise the proxy, subject to staying where the proxy is valid."** It is trust-region optimisation applied to a *learned objective* rather than to a policy update. Once you see it that way, it stops being a hack and becomes the obviously correct thing to do.

### 🔗 Resources for Topic 29

- **[Ziegler et al., Fine-Tuning Language Models from Human Preferences (2019)](https://arxiv.org/abs/1909.08593)** — where the KL-regularised objective and the adaptive KL controller above were introduced for LLMs. The direct ancestor of InstructGPT.
- **[Gao et al., Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760)** — quantifies exactly how gold-standard performance degrades as a function of `√KL`. **Read this to choose your KL budget with evidence rather than vibes.**
- **[TRL — controlling KL in PPO](https://huggingface.co/docs/trl/how_to_train)** — practical guidance on `kl_coef`, adaptive control, and what the logged KL values should look like.

---

## 📐 Formula summary — §6

| Concept | Formula |
|---|---|
| KL divergence | `D_KL(P‖Q) = Σ_x P(x) log(P(x)/Q(x))` |
| KL estimator k1 | `d = log π_θ(a_t) − log π_ref(a_t)` |
| KL estimator k3 (preferred) | `exp(−d) − 1 + d` |
| **KL-regularised RLHF objective** | **`max_θ E[r_φ(x,y)] − β·D_KL(π_θ‖π_ref)`** |
| Shaped per-token reward | `r̃_t = r_φ(x,y)·1[t=T] − β(log π_θ − log π_ref)` |

---

## 🎯 Interview Questions — §6

### Conceptual

**Q1. What is reward hacking? Give three concrete examples in LLMs.**
> Exploiting imperfections in a learned reward model to obtain high scores without producing genuinely good outputs. Examples: **sycophancy** (opening with flattery and agreeing with the user's errors — the lecture's "Bang on! Great question!"), **length/verbosity inflation** (padding because annotators preferred longer answers), and **format gaming** (bullet-pointing everything regardless of fit). Root cause: the reward model is a finite-data proxy, and optimisation pressure finds the regions where the proxy diverges from true quality (Goodhart's law).

**Q2. Why does reward hacking happen *inevitably*?**
> Because the reward model is trained on a finite sample of *reasonable* responses and then queried on outputs produced by an optimiser actively searching for high scores. That's self-induced distribution shift: the policy's generations move out of the RM's training distribution, where its scores become unreliable extrapolations. Any proxy for an unmeasurable objective creates such a gap, and optimisation is the process of finding it. Gao et al. (2022) show the degradation follows a clean scaling law in `√KL`.

**Q3. Define KL divergence and give three of its properties.**
> `D_KL(P‖Q) = Σ P(x)log(P(x)/Q(x))` — the expected log-ratio under P, measuring how much information is lost approximating P by Q. Properties: non-negative (=0 iff P=Q); **asymmetric** (`D_KL(P‖Q) ≠ D_KL(Q‖P)`, so not a distance metric); unbounded when Q assigns zero probability where P has mass. It also fails the triangle inequality.

**Q4. Write the KL-regularised RLHF objective and explain each term.**
> `max_θ E_{x~D, y~π_θ}[r_φ(x,y)] − β·D_KL(π_θ(·|x) ‖ π_ref(·|x))`. First term: maximise the learned preference reward. Second: penalise divergence from the frozen SFT reference. β trades alignment gain against fidelity to the starting model. This is the objective DPO later solves in closed form.

**Q5. What happens at β = 0 and β → ∞?**
> β = 0: pure reward maximisation — the reward curve rises rapidly while true quality peaks and then collapses into degenerate, sycophantic, or repetitive text. β → ∞: `π_θ = π_ref`; the model doesn't move and no alignment occurs. The useful regime is where reward *and* held-out human win-rate both improve while KL plateaus at a modest value (typically 6–10 nats per response).

**Q6. Why is KL implemented as a per-token shaped reward rather than a separate loss term?**
> Two reasons. Practically, it requires **no modification to PPO** — you adjust the reward, and the existing advantage/clipping machinery handles the rest. Conceptually, it makes the reward **dense**: the preference score arrives only at the final token, while the KL term fires at every position, providing per-step signal that materially improves credit assignment (and rescues reward-to-go from its degenerate terminal-reward case).

**Q7. Which direction of KL is used, and why does that matter?**
> `D_KL(π_θ ‖ π_ref)` — reverse KL relative to the reference, hence **mode-seeking**. The policy may concentrate mass on a subset of what the SFT model would say (sharpening onto preferred behaviour), but is heavily penalised for placing mass where `π_ref` assigns near-zero probability (inventing new, unvetted behaviour). Forward KL would instead force the policy to *cover* everything the SFT model does, which is not what we want.

**Q8. Distinguish the KL in PPO's clipping from the KL in the RLHF objective.**
> PPO's is between `π_θ` and `π_θ_old` — an **optimisation-stability** trust region governing how far one reuse window may move the policy, implemented via ratio clipping. The RLHF objective's is between `π_θ` and `π_ref = π_SFT` — an **anti-reward-hacking** constraint anchoring the final model to its supervised starting point. Different references, different purposes, both present at once.

**Q9. Name the three KL estimators and say which you'd use.**
> k1 = `d` (unbiased but can be negative, high variance); k2 = `½d²` (always ≥0 but biased); k3 = `exp(−d) − 1 + d` (**unbiased AND always ≥0, low variance**). Use **k3** — it's what TRL and most RLHF stacks use, at the cost of one extra `exp()` per token. Source: Schulman's "Approximating KL Divergence."

### Applied

**Q10. Your reward curve is rising steadily but users say the model got worse. Diagnose and fix.**
> Textbook reward hacking. **Diagnose:** plot KL-to-reference alongside reward (a runaway KL is the smoking gun); read actual generations; measure mean response length over training; run a held-out human or strong-LLM-judge win-rate against the SFT baseline; check refusal rate on benign prompts. **Fix:** increase β or switch to adaptive KL targeting; early-stop at the step where the win-rate peaked; retrain the reward model with data covering the failure mode (add pairs where a concise correct answer beats a flattering verbose one); apply length-debiasing; consider an ensemble of reward models and use the minimum score (pessimism reduces exploitable single-model blind spots).

**Q11. How do you detect reward hacking early, before wasting a full run?**
> Instrument three curves from step zero: (1) RM score, (2) KL to `π_ref`, (3) an *independent* quality metric — held-out preference accuracy, LLM-judge win-rate, or a small human eval. Hacking shows as **curve 1 rising while curve 3 stalls or falls**, usually with curve 2 climbing. Also track mean response length and the frequency of formulaic openers. Set a KL budget and early-stop on it. The `RewardHackingMonitor` pattern above operationalises exactly this.

**Q12. Beyond the KL penalty, what else reduces reward hacking?**
> Better preference data (the root cause — cover the failure modes explicitly, debias for length); **reward-model ensembles** with pessimistic aggregation; periodically **retraining the RM on fresh on-policy generations** (iterated RLHF), which closes the distribution-shift gap directly; **early stopping** on a held-out judge; constraining the optimisation (smaller LR, tighter clip); and non-RL alternatives such as **best-of-N sampling**, which limits optimisation pressure to N candidates rather than unbounded gradient search.

**Q13. Memory is tight. How do you avoid paying for a full reference model?**
> Train the policy with **LoRA** — then `π_ref` is the same frozen base weights with the adapters switched off (`with policy.disable_adapter():`), so one set of base weights serves both roles. Alternatives: keep `π_ref` in lower precision (it's inference-only, so int8/bf16 is fine), offload it to CPU and batch its forward passes, or use a distilled smaller reference (with the caveat that the KL then anchors to a different distribution).

### Rapid-fire

| Question | Answer |
|---|---|
| One-line definition of reward hacking? | High proxy reward, low true quality |
| The law it illustrates? | Goodhart's law |
| Is KL a distance metric? | **No** — asymmetric, no triangle inequality |
| Which KL direction in RLHF? | `D_KL(π_θ ‖ π_ref)` — reverse/mode-seeking |
| Preferred KL estimator? | k3 = `exp(−d) − 1 + d` |
| What is `π_ref`? | Frozen SFT model |
| The regularised objective? | `max E[r_φ] − β·D_KL(π_θ‖π_ref)` |
| Effect of β = 0? | Reward hacking / collapse |
| Two benefits of per-token KL? | No PPO change needed; densifies the reward |
| Typical KL budget? | ~6–10 nats per response |
| Most common root cause of hacking? | Preference-**data** artifacts, not the optimiser |

---

## ✅ Section self-check

1. Explain reward hacking without using the word "hacking."
2. Reproduce the KL-regularised RLHF objective from memory.
3. Sketch the reward curve vs. human-win-rate curve and mark the divergence point.
4. Give the k3 KL estimator and say why it's preferred over k1.
5. Explain why the *reverse* KL direction is the right choice here — and demonstrate it numerically on a bimodal P.
6. Name four mitigations for reward hacking beyond raising β.
7. Where have you seen `log π_θ(y) − log π_ref(y)` before, and where will you see it again?
8. **Hands-on:** run the reward-hacking simulation. At β=0, at what step does true quality peak, and what does the feature vector look like at the end?

---

**Previous:** [§5 — From Policy Gradient to PPO](05-pg-to-ppo.md) · **Next:** [§7 — The Four Models in RLHF](07-four-models-rlhf.md) · [Index](00-INDEX.md)
