# Topic 4 — Verbalized Sampling

> **Session:** 2.2 — Prompt Optimization and Security | IIT Kharagpur × upGrad  
> **Instructor:** Koustav Rudra, Assistant Professor AI, IIT Kharagpur  
> **Reference Paper:** Zhang et al., 2025 — arXiv:2510.01171

---

## Overview

Verbalized Sampling is a **training-free prompting strategy** that solves a fundamental problem in LLM output generation: **mode collapse** — the tendency of aligned LLMs to produce repetitive, low-diversity outputs even when many valid responses exist.

The key idea: instead of asking the model for one response, ask it to generate **k candidate responses with explicit probability weights**, then sample from that verbalized distribution. This recovers output diversity that alignment training suppressed.

---

## 4.1 The Mode Collapse Problem

### Definition

> **Mode collapse in LLMs** refers to a failure mode where the model produces overly similar, repetitive, or low-diversity outputs, even when many different valid responses should exist.

The term comes from **GANs** (Generative Adversarial Networks), where the generator learns to produce only a few "safe" outputs that fool the discriminator. LLMs exhibit an analogous failure: the model converges to a small number of safe, high-probability responses.

### The Musician Analogy

> *Think of it as: A musician who knows thousands of songs, but only plays their top three hits.*

The model has knowledge of the full distribution of valid responses — but its generation process consistently selects from only the top few high-probability modes.

### Concrete Examples of Mode Collapse

| Prompt | Mode-Collapsed Output |
|---|---|
| "Tell me a joke about coffee" | "Why did the coffee file a police report? It got mugged." — **every** major LLM says this |
| "Pick a random number between 1 and 10" | → 7, almost every time |
| "Pick a random number between 1 and 100" | → 42 or 47, disproportionately |
| "Think of a random vegetable" | → carrot, okra, broccoli |
| "Name a random country" | → Japan or France |
| "Recommend a sci-fi novel" | → Dune, Hyperion, or Three-Body Problem on loop |
| "Summarize this" | → three bullet points, regardless of source length |
| "Endings of essays" | → "In conclusion" or "Ultimately" |

The identical "coffee/mugged" joke was returned by ChatGPT, Gemini, and Claude simultaneously — despite each being independently queried. This demonstrates that these models share the same collapsed mode, baked in during post-training.

### Why Mode Collapse Matters — The Practical Impact

Mode collapse **undermines** three advanced prompting strategies that rely on output diversity:

1. **Self-consistency decoding** — samples multiple outputs and takes majority vote. If all outputs are identical, majority vote adds no value.
2. **Tree-of-Thought** — explores multiple reasoning branches. If all branches start the same, the tree is effectively a straight line.
3. **Reflection loops** — generates self-critiques. If all critiques are the same, the loop adds no value.

> *Even with temperature sampling, most trajectories begin similarly.*

Increasing temperature does help but introduces incoherence at high values. You get diversity, but lose quality. This is an unacceptable trade-off.

---

## 4.2 How Mode Collapse Happens

### Mechanism 1: Early-Token Lock-In

LLMs generate tokens **left-to-right, one at a time**. Once the first few tokens commit to a particular response direction, the remainder of the trajectory "collapses" into nearly identical continuations. There's a strong dependency between early and late tokens.

This is **analogous to beam search degeneration** in classical NLP, where the highest-scoring beams quickly converge.

### Mechanism 2: RLHF Sharpens Probability Mass

**Reinforcement Learning from Human Feedback (RLHF)** is how models like GPT, Claude, and Gemini are aligned to human preferences. The reward model in RLHF learns what humans rate as "good responses."

The problem: human raters tend to prefer:
- **Safe, uncontroversial answers**
- **Coherent, familiar reasoning styles**
- **Popular chain-of-thought structures** (numbered lists, "Let me explain...", "In conclusion...")

RLHF sharpens the model's probability mass heavily around these preferred patterns. After RLHF, the model's **pre-training diversity is significantly suppressed**.

> **Key stat (Zhang et al., 2025):** RLHF-aligned models recover only 33.2% of the diversity of their base model counterparts.

### Mechanism 3: Language Priors Dominate Reasoning Priors

The model's linguistic habits — learned from internet text — overwhelm its reasoning patterns:
- Common **internet exposition styles** dominate (blog-post format, Wikipedia-style)
- Memorized **reasoning schemas** from training data are repeatedly triggered
- **Textbook proof templates** appear even when less formal reasoning would be more appropriate

### Typicality Bias — The Root Cause

The core mechanism is **typicality bias**:

Given multiple valid response candidates (A, B, C, D), the model has an internal ranking:
```
Typicality Bias: A > B > D > C
```

RLHF amplifies this bias so strongly that post-training, option A is selected almost every time:
```
A, A, A, A, ... (mode collapse)
```

The model's pre-training distribution contained all four options with some spread. Post-training suppresses B, C, D almost entirely.

---

## 4.3 Verbalized Sampling — The Solution

### Definition

> **Verbalized Sampling** is a training-free prompting strategy that circumvents mode collapse by asking LLMs to verbalize probability distributions over candidate responses.

Instead of: *"Give me a response"*  
Ask: *"Give me k responses with their probabilities, then I'll sample from that distribution"*

### The Three Steps

**Step 1:** Prompt model to generate k responses (e.g., k=5)

```
"Generate 5 different responses to the following request, each with a
distinct approach. Assign a probability to each response (probabilities
should sum to 1.0) based on how likely it is to be the best answer."

Request: Tell me a joke about coffee.
```

**Step 2:** Request explicit probabilities for each response

The model produces:
```
1. "Espresso may not solve all your problems, but it's a good shot." (Prob: 0.12)
2. "Error 404: Coffee not found. Please restart human." (Prob: 0.07)
3. "Why did the latte go to therapy? It had too much foam to deal with." (Prob: 0.15)
4. "Cold brew is just coffee that took a gap year to find itself." (Prob: 0.07)
5. "Coffee: because anger management is too expensive." (Prob: 0.06)
```

**Step 3:** Sample from the verbalized distribution

Use the verbalized probabilities as a **sampling weight** — randomly select one response proportional to its assigned probability. This recovers diversity from the model's pre-training distribution.

**Result:** Diverse outputs that reflect the model's full prior, not just its collapsed post-training mode.

### Direct Prompting vs. Verbalized Sampling — Side by Side

| Dimension | Direct Prompting (×5 calls) | Verbalized Sampling (1 call) |
|---|---|---|
| **What you ask** | "Tell me a joke about coffee" × 5 | "Generate 5 jokes with probabilities" |
| **What you get** | Same mugged joke, 5 times | 5 distinct jokes with probability weights |
| **Diversity** | Near-zero | High (2-3x improvement) |
| **Cost** | 5 API calls | 1 API call |
| **Quality** | High (safe, polished) | Maintained (quality preserved) |

---

## 4.4 Advantages of Verbalized Sampling

### Technical Advantages

**Training-free:**
- No fine-tuning, no additional model weights, no data collection
- Works by prompting alone — deployable today on any LLM

**Model-agnostic:**
- Compatible with GPT, Claude, Gemini, Llama, Mistral, or any model that can follow complex instructions
- No API access to model internals required

**Orthogonal:**
- Combines freely with temperature sampling, top-p (nucleus) sampling, and top-k sampling
- Can be stacked with other diversity techniques for additive gains

### Diversity Improvements (Zhang et al., 2025)

| Metric | Improvement |
|---|---|
| Overall diversity boost | **2–3× over direct prompting** |
| Base model diversity recovery | **66.8% of pre-training diversity** (vs. 33.2% baseline) |
| Larger models benefit more | GPT-4, Gemini, Llama 70B show greater gains than smaller models |
| Quality maintained | Quality metrics do NOT degrade while diversity improves |

The **quality-maintained** finding is critical. A common counter-argument to diversity methods is that they sacrifice quality. Verbalized sampling shows this trade-off is not necessary.

### The Image Generation Experiment

Zhang et al. demonstrated verbalized sampling in a downstream creative pipeline:

**Setup:** Generate image captions via LLM → feed captions to an image generation model

**Direct Prompting result:**
All captions converged to photorealistic astronaut-on-horse imagery in narrow desert/landscape scenarios. Same artistic style across all 5 samples.

**Verbalized Sampling result:**
Captions ranged across:
- Cinematic gallop under a looming Earth
- Retrofuturist rider on a chrome horse
- Whimsical storybook watercolor of an astronaut
- Thundering through a canyon's twin suns
- Heroic astronaut in a Baroque painting

**Higher diversity in both artistic style AND narrative setting** — downstream diversity cascades from the text generation stage.

---

## 4.5 When to Use Verbalized Sampling

| Use Case | Why Verbalized Sampling Helps |
|---|---|
| **Creative generation** (stories, poems, jokes, dialogue) | Need genuine variety; mode collapse kills creativity |
| **Brainstorming and hypothesis generation** | Need to explore the idea space, not just the top-1 idea |
| **Social simulation and human-like interactions** | Diverse, realistic behavior patterns required |
| **Synthetic data generation** | Need diverse data for downstream tasks (classification, image gen, fine-tuning) |
| **Self-consistency decoding** | More meaningful majority vote when candidates differ |
| **Tree-of-Thought reasoning** | More divergent branches lead to better exploration |

### When Verbalized Sampling May Not Help

| Scenario | Reason |
|---|---|
| **Factual Q&A** (single correct answer) | Diversity is not desirable; the correct answer should dominate |
| **Structured data extraction** | One correct JSON structure; diversity creates noise |
| **Safety-critical single-output tasks** | You want the model's most reliable, calibrated response |

---

## 4.6 Limitations and Considerations

### Calibration of Verbalized Probabilities

The model assigns probabilities through language (e.g., "Prob: 0.15") — these are **not** the actual token probabilities from the model's softmax distribution. They are the model's **self-assessment** of likelihood, which may be miscalibrated.

Research suggests larger models are better calibrated in their verbalized probability estimates, which explains why verbalized sampling benefits larger models more.

### Context Length Cost

Generating k responses in a single call requires more output tokens than generating 1 response in a single call. For k=5, you approximately quintuple the output token count. This is typically acceptable since you avoid 4 additional API calls.

### Ordering Effects

The model may exhibit a bias toward assigning higher probabilities to responses it listed first. Randomizing the order or averaging over multiple orderings can mitigate this.

---

## Learning Highlights

> **Core insight:** RLHF makes models safe but makes them boring. The model's pre-training contained enormous diversity; alignment training suppresses most of it. Verbalized sampling is a "diversity unlock" that reaches back into the pre-training distribution.

> **Mechanistic insight:** Mode collapse is not random noise — it's a systematic bias toward the same "safe" modes across all aligned models. This is why GPT, Gemini, and Claude all say the same coffee joke.

> **Practical insight:** For any task where you want the model to explore the space of possibilities — brainstorming, creative writing, synthetic data, self-consistency — use verbalized sampling instead of calling the API 5 times.

> **Quality insight:** The diversity-quality trade-off is a false dilemma for verbalized sampling. You get more diversity AND maintain quality, because you're sampling from higher-quality candidates with proper probability weights, not injecting random noise.

> **Downstream insight:** Diversity at the text generation stage cascades to all downstream stages. More diverse captions → more diverse images. More diverse hypotheses → better brainstorming sessions.

---

## Interview Questions

### Foundational

**Q1. What is mode collapse in LLMs and why does it happen?**

*Answer:* Mode collapse in LLMs is the tendency to produce overly similar, low-diversity outputs even when many valid responses exist. It happens due to: (1) early-token lock-in — once the first tokens commit to a direction, the rest converges identically; (2) RLHF sharpening — alignment training amplifies preference for safe, popular response patterns, suppressing pre-training diversity; and (3) language priors dominating reasoning priors — memorized internet text patterns override genuine reasoning diversity.

---

**Q2. What is verbalized sampling and how does it differ from just increasing temperature?**

*Answer:* Verbalized sampling asks the model to explicitly generate k candidate responses with assigned probability weights, then samples from that verbalized distribution. It recovers diversity from the model's pre-training distribution. Increasing temperature adds noise to the token-level sampling process — it can increase diversity but also introduces incoherence and quality degradation at high values. Verbalized sampling maintains output quality while increasing diversity because it explores the space of coherent responses, not random token-level perturbations.

---

**Q3. Why does verbalized sampling work better on larger models?**

*Answer:* Verbalized sampling requires the model to accurately self-assess the relative quality of its own generated responses and assign calibrated probabilities. Larger models (GPT-4, Gemini Ultra, Llama 70B) have better self-awareness and probability calibration — their verbalized probabilities more accurately reflect the true likelihood of each response being optimal. Smaller models struggle to reliably rank their own outputs, leading to miscalibrated verbalized probabilities.

---

**Q4. What is typicality bias and how does it cause mode collapse?**

*Answer:* Typicality bias is the model's tendency to rank responses by "how typical" or "safe" they are rather than by genuine quality or diversity. Given multiple valid candidate responses, the model assigns a heavily skewed typicality ranking (A >> B > D > C), and RLHF post-training amplifies this skew until the model generates option A almost exclusively. The full pre-training distribution contains all four options with spread; post-training suppresses B, C, and D, leaving only A — mode collapse.

---

### Intermediate

**Q5. How does mode collapse undermine self-consistency decoding?**

*Answer:* Self-consistency decoding samples k outputs from the model and takes a majority vote for the final answer, reasoning that the most consistently occurring answer is likely correct. This only works if the k outputs show meaningful disagreement — different reasoning paths that converge to the same answer are informative; identical outputs that converge trivially add no information. Mode collapse makes all k outputs nearly identical, so majority vote degenerates to picking a single response, eliminating the benefit of multi-sampling.

---

**Q6. Design a verbalized sampling prompt for a creative brainstorming task. What elements must it include?**

*Answer:* The prompt must include: (1) **explicit instruction to generate k distinct responses** with clear differentiation requirement; (2) **explicit probability assignment instruction** — "assign each response a probability between 0 and 1 that sums to 1.0"; (3) **diversity encouragement** — "each response should take a fundamentally different approach, angle, or framing"; (4) **quality criterion** — probabilities should reflect which response would be most useful for [specific goal], not which is safest. Example: "Generate 5 distinct marketing taglines for a coffee brand, each with a different creative angle. For each tagline, assign a probability score (summing to 1.0) reflecting how effective it would be for attracting young professionals."

---

**Q7. What is the relationship between verbalized sampling and the broader field of output diversity in NLP?**

*Answer:* Output diversity has long been studied in NLP — diverse beam search, top-k/top-p sampling, temperature scaling, and nucleus sampling all address it at the **decoding level** (modifying the token-by-token sampling process). Verbalized sampling addresses it at the **semantic level** — it doesn't modify how tokens are sampled; instead, it changes what the model is asked to generate. This makes it complementary to decoding-level methods (orthogonal) and applicable even when you don't have access to the model's token probabilities (black-box APIs).

---

### Advanced

**Q8. How would you integrate verbalized sampling into a production creative writing pipeline?**

*Answer:* (1) Replace single-call generation with verbalized sampling calls that produce k=5 candidates with probabilities. (2) Build a downstream ranker that re-scores candidates using additional quality criteria (engagement score, grammar check, brand alignment). (3) Combine verbalized probabilities with ranker scores using a weighted combination for final candidate selection. (4) A/B test the verbalized sampling pipeline against the direct prompting baseline, measuring diversity metrics (distinct n-grams, semantic similarity) and quality metrics (user engagement, preference ratings). (5) Monitor for over-diversity (bizarre outputs that score low quality) using a quality floor filter.

---

**Q9. A researcher argues that verbalized sampling just makes the model "pretend" to be diverse without changing its underlying probability distribution. How would you respond?**

*Answer:* This is partially correct but misses the key insight. The model's output probability distribution is NOT changed — the token-level sampling still operates on the same RLHF-aligned distribution. However, verbalized sampling changes WHAT TASK the model is solving. In direct prompting, the model solves "generate the most likely response to [query]" — which produces mode-collapsed output. In verbalized sampling, the model solves "generate k DIVERSE responses to [query] with their probabilities" — this is a different task that elicits the model's knowledge of its full prior distribution. The model has access to that distribution implicitly (it knows multiple valid jokes about coffee) but ordinarily doesn't exercise it. Verbalized sampling forces it to. The resulting diversity is real and measurable — 2-3× improvement, confirmed empirically.

---

## Quick Reference Summary

| Concept | One-Line Definition |
|---|---|
| Mode Collapse | LLM produces same/similar outputs repeatedly despite many valid alternatives |
| Typicality Bias | Model's over-preference for "safe" typical responses over diverse valid ones |
| RLHF Effect | Alignment training suppresses pre-training diversity (66.8% lost) |
| Early-Token Lock-In | First tokens determine trajectory; rest converges identically |
| Verbalized Sampling | Ask model for k responses + probabilities; sample from that distribution |
| Training-Free | No fine-tuning needed — works via prompting alone |
| 2-3× Diversity Boost | Key result from Zhang et al., 2025 |
