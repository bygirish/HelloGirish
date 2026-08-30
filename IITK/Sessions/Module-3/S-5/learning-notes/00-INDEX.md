# LLM Alignment: RLHF & DPO — Complete Study Pack

> **IIT Kharagpur · Executive Certificate Programme in Generative AI and Agentic AI**
> Module 3 · Session 3 · Dr. Plaban Kumar Bhowmick
> Built from: the 58-slide deck · the DPO demo notebook · the live-session Q&A · the full transcript · the pre-read and post-read PDFs

---

## 📖 How to use this pack

Ten files, one per section, **in dependency order**. The deck is a single continuous argument — reading out of order will cost you more than it saves.

| Read for | Do this |
|---|---|
| **First pass (understanding)** | §1 → §10 in order. Skip the 🎯 interview questions. ~5–7 hours. |
| **Hands-on (real understanding)** | Run the 🔬 code block in each topic as you read. Roughly doubles the time and more than doubles the retention. |
| **Second pass (retention)** | Re-read the 💡 *Learning thought* boxes and the ✅ *self-checks* only. ~1 hour. |
| **Interview prep** | The 🎯 sections, especially §4, §6, §8, §10. Rapid-fire tables are the last-minute layer. |
| **Implementation** | §7 (systems + memory), §8 (reward-model training), §10 (the notebook walkthrough + TRL). |

### What's in every file

| Marker | Meaning |
|---|---|
| **📊 slide image** | The original diagram, extracted from the deck (`assets/`) |
| **🔬 runnable code** | A self-contained script that *demonstrates* the concept, not just illustrates it |
| **💡 Learning thought** | The non-obvious insight — what to actually take away |
| **⚠️ Class Q&A** | Real confusions raised in the live session, answered properly |
| **📐 Formula summary** | Every equation in the section, in one table |
| **🎯 Interview questions** | Conceptual + applied + rapid-fire, with model answers |
| **🔗 Resources** | Papers, blogs, docs — per topic, not dumped at the end |
| **✅ Self-check** | Test yourself before moving on |

---

## 🧭 The argument of the whole session, in one paragraph

HHH names the behaviour we want, but it's English, not a loss function (**§1**). RL is the only framework that survives when you have nothing but a scalar score on a finished attempt (**§2**), and an LLM turns out to already *be* an RL agent — prompt is state, token is action, the model is the policy (**§3**). The policy gradient makes this trainable but is crippled by variance, fixed by reward-to-go → baseline → advantage (**§4**), and by sample inefficiency, fixed by PPO's importance ratio and clipping (**§5**). Because the reward is a *learned proxy*, hard optimisation exploits its flaws — reward hacking — so we anchor the policy to the SFT model with a KL penalty (**§6**). All of this requires four networks in memory (**§7**), one of which, the reward model, is built from pairwise human comparisons via the Bradley-Terry loss (**§8**). InstructGPT assembled this into three steps and made a 1.3B model beat a 175B one (**§9**). Then DPO solved the KL-regularised objective in closed form, discovered the policy *is* implicitly a reward model, and collapsed the entire pipeline into one supervised loss on preference pairs (**§10**).

---

## 📚 The sections

| § | File | Slides | Topics | Core idea |
|---|---|---|---|---|
| **1** | [The Alignment Problem](01-alignment-problem.md) | 3–7 | 1–5 | HHH is unmeasurable → convert it into pairwise preferences |
| **2** | [RL Foundations](02-rl-foundations.md) | 9–15, 22 | 6–11 | MDPs, trajectories, and why RL is the only option left |
| **3** | [Casting an LLM as an RL Problem](03-llm-as-rl.md) | 16–21 | 12–15 | Prompt = state · token = action · **the model is the policy** |
| **4** | [Policy Gradient & Variance Reduction](04-policy-gradient.md) | 23–31 | 16–23 | Log-derivative trick → REINFORCE → reward-to-go → baseline → advantage |
| **5** | [From Policy Gradient to PPO](05-pg-to-ppo.md) | 32–33, 37 | 24–26 | Reuse rollouts via importance sampling; clip to stay proximal |
| **6** | [Reward Hacking & KL Control](06-reward-hacking-kl.md) | 34–36 | 27–29 | The proxy has gaps; anchor to `π_ref` with a KL penalty |
| **7** | [The Four Models in RLHF](07-four-models-rlhf.md) | 38–40, 47–48 | 30–34 | Policy · Reference · Reward · Value — and their memory cost |
| **8** | [Building the Reward Model](08-reward-model.md) | 41–46 | 35–39 | Bradley-Terry: `−log σ(r_w − r_l)` |
| **9** | [Systems in the Wild](09-instructgpt-chatgpt.md) | 49–50 | 40–41 | InstructGPT's three steps; 1.3B beats 175B |
| **10** | [Direct Preference Optimization](10-dpo.md) | 52–56 + notebook | 42–46 | Same loss, implicit reward — 4 models → 2, RL → classification |

---

## 🔬 Runnable code index

Every script is self-contained. Where a topic teaches something you can *verify*, the code verifies it rather than restating it.

| § | What the code does | Why run it |
|---|---|---|
| 1 | Base vs. instruct model on the same prompt | See the alignment gap in 20 lines |
| 1 | Sycophancy probe (2-turn pushback) | The eval you should keep forever |
| 1 | Over-refusal probe (XSTest-style) | Catches the most common alignment regression |
| 1 | **Length-bias audit on UltraFeedback** | **The single most useful diagnostic in the pack** |
| 1 | SFT vs. DPO loss signatures side by side | One response vs. two — the whole structural difference |
| 2 | Reproduce slide 9's distribution with a real LM | `probs` *is* `π(a\|s)` |
| 2 | Manual autoregressive generation loop | This is the RL rollout |
| 2 | Watch the gradient break (argmax vs. REINFORCE) | Makes "non-differentiable" concrete |
| 2 | **Complete `MouseGrid` environment** | Slide 13, executable; Gym-shaped |
| 2 | γ sweep on a terminal-reward trajectory | Why RLHF needs γ ≈ 1 |
| 2 | Monte-Carlo estimate of `J(θ)` vs. sample count | The `1/√D` curve, measured |
| 3 | LLM-as-MDP class (`reset`/`policy`/`step`) | Same interface as `MouseGrid`; reward is `0.0` |
| 3 | **Prove the KV cache isn't state** | Cached vs. uncached logits are identical |
| 3 | Base vs. aligned log-probs on two candidate responses | Measures probability-mass reallocation |
| 3 | Float underflow demo (`0.3^200`) | Why everything is in log space |
| 3 | Reward-sparsity table + advantage fix | Previews §4's whole story |
| 4 | Verify `∇f = f·∇log f` numerically | Don't take Step 3 on faith |
| 4 | **Policy gradient == reweighted cross-entropy** | The single most demystifying experiment |
| 4 | **Full REINFORCE on CartPole, with ablation** | Variance reduction decides whether it learns |
| 4 | Measure bias *and* variance on a bandit | Slide 27's two claims, both verified |
| 4 | Reward-to-go: dense vs. terminal vs. KL-shaped | Why it buys nothing in vanilla RLHF |
| 4 | Verify `E[∇log π] = 0` + action-dependent counter-example | The whole unbiasedness proof |
| 4 | **GAE implementation with λ sweep** | The function in every RLHF codebase |
| 5 | **Time generation vs. gradient step** | Generation is 20–100× — measure it on your box |
| 5 | Importance sampling: unbiased but exploding variance | *Why* PPO must clip |
| 5 | Clipped objective table incl. the `A<0, r=2.0` case | *Why* the `min()` is there |
| 5 | `RolloutBuffer` + `ppo_update` | Proves `π_θ_old` is data, not a model |
| 5 | Full PPO loss with `clip_frac` / `approx_kl` diagnostics | How you actually debug PPO |
| 6 | **60-line reward-hacking simulation** | Proxy up, true quality down — the whole section |
| 6 | `RewardHackingMonitor` for your training loop | Production-ready early warning |
| 6 | All five KL properties verified | Including mode-seeking vs. mass-covering |
| 6 | **k1/k2/k3 KL estimators compared** | Which one TRL uses and why |
| 6 | Shaped per-token reward (`r̃_t`) | Exactly what TRL computes |
| 6 | Adaptive KL controller | Tune a *budget*, not a coefficient |
| 7 | **GPU memory calculator** | Run it for the model you'll actually train |
| 7 | Instantiate all four models, count parameters | The 4-model burden, in numbers |
| 7 | One forward pass → every position's `log π` | Generation sequential, scoring parallel |
| 7 | **LoRA reference trick (`disable_adapter`)** | Makes `π_ref` free — the key production technique |
| 7 | Value head + explained variance | The PPO health metric to log |
| 7 | GRPO / RLOO baselines | Deleting a 7B network in six lines |
| 7 | Three-headed model (LM / reward / value) | Slide 48, literal |
| 8 | Simulate annotator disagreement | Absolute scales diverge; ranks agree |
| 8 | Preference-dataset quality audit | Length bias, ties, near-duplicates |
| 8 | **Generate on-policy preference pairs** | The TA's recipe, implemented |
| 8 | Reward model + **the padding bug, demonstrated** | The classic silent failure |
| 8 | Bradley-Terry identity, loss table, stability, gradient | Four experiments in one block |
| 8 | **Complete reward-model training run** | ~50% → 65–75% preference accuracy |
| 8 | Length-bias + padding-probe audits | Predicts whether PPO will hack it |
| 9 | Ranking of K → `C(K,2)` pairs | Why InstructGPT used K=9 |
| 9 | **All three stages in TRL** | SFT → RewardTrainer → PPOTrainer |
| 9 | Alignment-tax checkpoint selection | Don't ship on win-rate alone |
| 9 | Multi-turn state + assistant-turn masking | The most common silent dialogue bug |
| 10 | **Verify the closed form, the inversion, and `Z(x)` cancelling** | The derivation, checked numerically |
| 10 | DPO gradient adaptive weighting | Same property as Bradley-Terry |
| 10 | **The "both log-probs fall" pathology** | DPO's most counterintuitive failure |
| 10 | Completion mask, with/without comparison | The #1 DPO implementation bug |
| 10 | Precompute reference log-probs | Exact 2× speedup |
| 10 | `full_eval` with accuracy + absolute rewards | What the notebook's eval is missing |
| 10 | **Production TRL `DPOTrainer` config** | LoRA + ref caching + variants |

### Environment setup

```bash
python -m venv .venv && source .venv/bin/activate

# Core (needed for most blocks)
pip install torch transformers datasets numpy

# RL foundations (§2, §4)
pip install gymnasium scipy

# Alignment training (§5, §7, §8, §9, §10)
pip install trl peft accelerate

# Optional: fast rollout generation for PPO (§5)
pip install vllm
```

All examples use **`Qwen/Qwen2-0.5B-Instruct`** (~1 GB) so they run on CPU or a small GPU. Swap `MODEL_ID` for anything larger once the concepts land.

---

## 📊 Slide diagrams (`assets/`)

Extracted from the source deck and embedded in the relevant topic.

| File | Slide | Shows |
|---|---|---|
| `s12-why-rl-nondifferentiable.png` | 12 | Human feedback → **"Non-Differentiable, No Gradients"** → the LLM agent |
| `s13-mouse-grid.png` | 13 | The mouse-and-cheese grid world |
| `s15-trajectory.jpeg` | 15 | Trajectory to a final time step |
| `s18-rl-to-llm-mapping.png` | 18 | **General RL → RL for LLMs**, term by term |
| `s23-policy-gradient-theorem.png` | 23 | The Policy Gradient Theorem |
| `s24-policy-gradient-ascent.png` | 24 | The gradient-ascent update |
| `s35-kl-divergence.png` | 35 | Two distributions, KL contributions, "KL is NOT symmetric" |
| `s41-reward-model-problem.png` | 41 | Why absolute scoring fails |
| `s42-preference-data-example.png` | 42 | Prompt / Answer 1 / Answer 2 / Chosen |
| `s43-reward-model-architecture.png` | 43 | Tokens → transformer → hidden states → linear(1) → Reward |
| `s45-reward-model-training.png` | 45 | Full Bradley-Terry training graph with shared weights |
| `s49-instructgpt-3steps.png` | 49 | **InstructGPT's three-step pipeline** (the canonical figure) |
| `s50-chatgpt-methods.png` | 50 | OpenAI's ChatGPT methods statement |
| `s54-dpo-loss.png` | 54 | The DPO loss |
| `s55-dpo-workflow.png` | 55 | Prompts → sample from `π_ref` → annotate → minimise `L_DPO` |
| `s56-dpo-entire-workflow.jpeg` | 56 | Static dataset → SFT on chosen → DPO training |

---

## 🔑 The equations that matter

If you retain nothing else, retain these six. They chain into each other.

| # | Equation | Section |
|---|---|---|
| 1 | `∇_θ J(θ) = E[ Σ_t ∇log π_θ(a_t\|s_t) · A_t ]` | §4 — the policy gradient |
| 2 | `A^π(s,a) = Q^π(s,a) − V^π(s)` | §4 — the advantage |
| 3 | `L^CLIP = E[min(r_t A_t, clip(r_t, 1±ε) A_t)]` | §5 — PPO |
| 4 | **`max_θ E[r_φ(x,y)] − β·D_KL(π_θ ‖ π_ref)`** | §6 — **the alignment objective** |
| 5 | `L_RM = −E[log σ(r_φ(x,y_w) − r_φ(x,y_l))]` | §8 — Bradley-Terry |
| 6 | **`L_DPO = −E[log σ(β log(π_θ(y_w)/π_ref(y_w)) − β log(π_θ(y_l)/π_ref(y_l)))]`** | §10 — DPO |

**The punchline in one line:** solve (4) in closed form, substitute into (5), and you get (6) — with the reward model gone.

---

## 🗺️ The full mapping table (§3 — memorise this)

| RL term | LLM instantiation |
|---|---|
| Agent | The LLM |
| **State `s_t`** | Prompt + tokens generated so far — **not** the weights, **not** the KV cache |
| **Action `a_t`** | The next token (action space = vocabulary, ~152k for Qwen2) |
| **Policy `π_θ`** | `softmax(LLM_θ(s))` — the model **is** the policy |
| Transition | String concatenation — **deterministic, = 1** |
| Reward | Learned reward-model score — **terminal, extrinsic, learned** |
| Trajectory | One (prompt, response) pair |

---

## ⚖️ RLHF vs. DPO — the decision table

| | RLHF (PPO) | DPO |
|---|---|---|
| Models in memory | **4** | **2** |
| Models trained | 2 | **1** |
| Reward model | Explicit | **Implicit** |
| Online generation | Required (20–100× the gradient step) | **Not needed** |
| Stability | Fiddly (~8 hyperparameters) | Stable (β, LR) |
| Memory (7B, full FT) | ~196 GB | ~98 GB (far less with LoRA) |
| **Choose it when** | No preference pairs · verifiable/programmatic reward (RLVR) · need exploration · want a reusable RM for best-of-N | **You have a good preference dataset** · limited compute · want stable, reproducible training |

---

## 🚩 The ten things most people get wrong

Each is answered in full in the linked section. If you can explain all ten, you know this material.

1. **The state is not the model's weights or KV cache** — it's the prompt plus tokens generated so far. *(Provable: cached and uncached logits are identical.)* → [§3](03-llm-as-rl.md)
2. **`π_θ_old` is not a second model in memory** (it's stored logprobs); **`π_ref` is.** Different objects, different purposes. → [§5](05-pg-to-ppo.md), [§7](07-four-models-rlhf.md)
3. **There are two different KL divergences** in RLHF — PPO's trust region (`π_θ` vs `π_θ_old`) and the alignment anchor (`π_θ` vs `π_ref`). → [§6](06-reward-hacking-kl.md)
4. **"Future can't influence the past" is about credit, not causation** — your action *is* shaped by history; it just can't have caused rewards already collected. → [§4](04-policy-gradient.md)
5. **Reward model ≠ value model.** Sequence-level and terminal vs. token-level and everywhere. → [§7](07-four-models-rlhf.md)
6. **Reward-model outputs are not interpretable in absolute terms** — the loss is shift-invariant. *(And that shift-invariance is exactly what makes DPO's `Z(x)` cancel.)* → [§8](08-reward-model.md)
7. **Hallucination is one failure mode of honesty, not a synonym for it** — sycophancy and miscalibration are the others. → [§1](01-alignment-problem.md)
8. **Reward hacking is mostly a *data* problem, not an optimiser problem** — length bias and sycophancy are latent in the labels; PPO only amplifies them. → [§6](06-reward-hacking-kl.md)
9. **DPO's loss can decrease while `π(y_chosen)` also decreases** — only the *margin* is constrained. Log `rewards/chosen` in absolute terms. → [§10](10-dpo.md)
10. **Alignment reallocates probability mass; it does not add knowledge.** No amount of DPO will fix a factual gap — that's a retrieval problem. → [§1](01-alignment-problem.md), [§3](03-llm-as-rl.md)

---

## 🧮 Numbers worth remembering

| Value | What it is |
|---|---|
| `ln 2 = 0.6931` | The DPO loss at initialisation (`π_θ = π_ref`) — your "nothing has happened yet" baseline |
| `65–75%` | Typical reward-model preference accuracy — near the human-agreement ceiling |
| `70–80%` | Human–human agreement on preference labelling (the label noise ceiling) |
| `~65%` | "Always pick the longer response" accuracy on UltraFeedback — the length-bias floor |
| `β = 0.1` | Standard DPO / KL coefficient |
| `ε = 0.2` | Standard PPO clip range |
| `K = 4` | Standard PPO epochs per rollout batch |
| `clip_frac 0.05–0.2` | Healthy PPO range; >0.3 means K or LR too high |
| `6–10 nats` | Typical healthy KL budget vs. `π_ref` |
| `λ = 0.95` | Standard GAE setting |
| `5e-7` | Typical DPO learning rate — ~1000× below SFT |
| `1e-6` | Typical PPO learning rate |
| `20–100×` | How much more expensive generation is than the gradient step |
| `4` | Models in PPO-RLHF · also forward passes per DPO step |
| `C(9,2) = 36` | Preference pairs from one InstructGPT ranking |
| `1.3B > 175B` | InstructGPT's headline preference result |

---

## 📂 Source materials in this folder

| File | Contents |
|---|---|
| `...-3-2c-LLM-Alignment-RLHF-DPO.pptx` | The 58-slide deck — the spine of this pack |
| `...-LLM-Alignment-Demo-08-08-26.ipynb` | DPO from scratch on Qwen2-0.5B-Instruct — walked through in [§10](10-dpo.md) |
| `...-Q-A.xlsx` | Live session Q&A — folded into the ⚠️ boxes throughout |
| `...-Pre-read-Material-08-August.pdf` | Pre-read — quoted in [§1](01-alignment-problem.md) and [§6](06-reward-hacking-kl.md) |
| `...-post-read-material-8th-Aug.pdf` | Post-read — its four links appear below |
| `transcript.text` | Full session transcript — source of the professor's framings quoted throughout |

---

## 🔗 Master resource list

### The four from slide 57 / the post-read PDF

- Christiano et al., **Deep Reinforcement Learning from Human Preferences** (2017) — [paper](https://proceedings.neurips.cc/paper_files/paper/2017/file/d5e2c0adad503c91f91df240d0cd4e49-Paper.pdf) · *the origin of learned preference rewards → [§8](08-reward-model.md)*
- Nathan Lambert, **The RLHF Book** (2025) — [rlhfbook.com](https://rlhfbook.com/book.pdf) · *the modern reference; Ch. 3, 7, 12 map onto §3, §8, §10*
- Cameron Wolfe, **PPO for LLMs: A Guide for Normal People** — [substack](https://cameronrwolfe.substack.com/p/ppo-llm) · *→ [§5](05-pg-to-ppo.md)*
- Cameron Wolfe, **Direct Preference Optimization** — [substack](https://cameronrwolfe.substack.com/p/direct-preference-optimization) · *→ [§10](10-dpo.md)*

### The two primary papers the deck is built on

- **Ouyang et al., InstructGPT** (2022) — [arXiv:2203.02155](https://arxiv.org/abs/2203.02155) · *Figure 2 is slide 49; Appendix C is a real annotation rubric*
- **Rafailov et al., DPO** (2023) — [arXiv:2305.18290](https://arxiv.org/abs/2305.18290) · *Appendix A.1 and A.3 are the derivations in [§10](10-dpo.md)*

### Foundations

- [OpenAI Spinning Up](https://spinningup.openai.com/) — the best RL-for-ML-people introduction
- [Sutton & Barto, *RL: An Introduction*](http://incompleteideas.net/book/RLbook2020.pdf) — the textbook
- [Lilian Weng — Policy Gradient Algorithms](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/) — the variant map
- [HuggingFace Deep RL Course](https://huggingface.co/learn/deep-rl-course/) — hands-on, free

### The practical essentials

- **[The 37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)** — everything papers omit
- **[John Schulman — Approximating KL Divergence](http://joschu.net/blog/kl-approx.html)** — the k1/k2/k3 estimators
- **[HuggingFace TRL](https://huggingface.co/docs/trl)** — `SFTTrainer`, `RewardTrainer`, `PPOTrainer`, `DPOTrainer`
- **[The Alignment Handbook](https://github.com/huggingface/alignment-handbook)** — runnable end-to-end recipes
- [HuggingFace — Illustrating RLHF](https://huggingface.co/blog/rlhf) — the best visual overview

### Going deeper

- [Gao et al., Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760) — how reward hacking scales with KL
- [Askell et al., HHH / A General Language Assistant](https://arxiv.org/abs/2112.00861) — the origin of HHH
- [Bai et al., Constitutional AI](https://arxiv.org/abs/2212.08073) — writing the values down explicitly
- [Shao et al., DeepSeekMath (GRPO)](https://arxiv.org/abs/2402.03300) — dropping the value model
- [Ahmadian et al., Back to Basics (RLOO)](https://arxiv.org/abs/2402.14740) — is PPO's machinery even needed?
- [Zephyr-7B](https://arxiv.org/abs/2310.16944) — the open recipe that made DPO mainstream
- [DeepMind — Specification gaming examples](https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/) — reward hacking in the wild

---

## ✅ Final mastery check

You have this material if you can, from a blank page:

1. Explain why HHH cannot be written as a loss function, and what replaces it.
2. Map all six MDP components onto an LLM, correctly, including the two things that are *not* the state.
3. Derive the policy gradient theorem.
4. Name the three variance reductions and explain what each makes the question "more local and more relative" about.
5. Write PPO's clipped objective and explain why the `min` is there (use the `A<0, ratio=2.0` case).
6. Write the KL-regularised alignment objective and explain both terms.
7. Distinguish all four RLHF models by role, training status, and memory cost.
8. Derive the Bradley-Terry loss.
9. Draw InstructGPT's three steps and label each box's loss function.
10. Derive DPO from the KL-regularised objective and explain why `Z(x)` cancels.

And, hands-on:

11. Run the length-bias audit (§1) and state what it implies about a 70%-accurate reward model.
12. Run the REINFORCE ablation (§4) and explain the gap.
13. Run the reward-hacking simulation (§6) and identify the step where true quality peaks.
14. Train the reward model (§8) and report accuracy + `corr(reward, length)`.
15. Run the DPO notebook (§10) and report test loss vs. `ln 2` and whether `reward_chosen` is positive.

---

*Study pack built from the Module 3 · Session 3 source materials. Start with [§1](01-alignment-problem.md).*
