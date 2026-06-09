# Section 1: LLM Evaluation — Human Evaluation

> **Lecture:** LLM Evaluations and AI Safety | IIT Kharagpur × upGrad
> **Instructor:** Prof. Sourangshu Bhattacharya, CSE, IIT Kharagpur

---

## Learning Objective
Understand *why* evaluation matters, *what* we are evaluating, *why* human evaluation is the gold standard, and *why* it is also fundamentally limited — and how we measure that limitation.

---

## Topic 1: What Is "Evaluation"?

### The Two Meanings of Evaluation

When someone says "evaluate this LLM," they could mean two entirely different things:

| Dimension | What It Measures | Examples |
|-----------|-----------------|----------|
| **Output Quality** | How good is the generated text? | Instruction following, Coherence, Factuality |
| **System Performance** | How good is the system operationally? | Latency, Pricing, Reliability, Throughput |

### Focus of This Lecture: Output Quality

The lecture (and this section) focuses **exclusively on output quality** — how good the text that the LLM generates actually is. System performance is a separate engineering concern.

### The Three Core Quality Criteria

1. **Instruction Following** — Did the model do what was asked? If asked to write in bullet points, did it? If asked to respond in French, did it?

2. **Coherence** — Is the output logically consistent, well-structured, and readable? Does it flow? Are there contradictions?

3. **Factuality** — Is the information in the output actually true? Did the model hallucinate facts, dates, names, or events?

> **Key Insight:** These three dimensions are in tension. A model can be highly coherent but factually wrong. A model can follow instructions perfectly but produce incoherent output. This is why evaluation is hard.

---

## Topic 2: The Ideal Scenario — Human Rating as the Gold Standard

### Why Human Evaluation is the Gold Standard

LLMs generate **free-form text** — open-ended, creative, contextual. Unlike structured outputs (like a SQL query where correctness is binary), free-form text has no single "correct" answer. Only humans can natively understand:
- Nuance and tone
- Cultural context
- Whether an answer is actually useful for the real-world task
- Subtle factual errors
- Appropriateness for the audience

### The Ideal Evaluation Loop

```
LLM --> generates output --> Human rates the output --> Rating fed back to improve LLM
  ^                                                              |
  |______________________________________________________________|
```

This closed loop is ideal: the LLM generates, humans judge quality, and that judgment becomes a signal for improvement. This is literally how RLHF (Reinforcement Learning from Human Feedback) works — the human ratings train a **reward model** that then guides future LLM training.

### Real-World Human Evaluation Setup

In practice, human evaluation looks like:
- A **pool of annotators** (often crowd-sourced via MTurk, or expert annotators)
- A **rubric** that defines the criteria (e.g., "rate this response 1–5 on Helpfulness")
- Multiple raters per item (to handle disagreement)
- Statistical aggregation of ratings

> **Example:** OpenAI's original InstructGPT paper used ~40 contractors who rated model outputs on criteria like helpfulness, harmlessness, and honesty. These ratings trained the reward model used in PPO (Proximal Policy Optimization).

---

## Topic 3: Limitations of Human Evaluation

### The Core Problem: Subjectivity

Consider this simple example from the lecture:

- **Prompt:** "What birthday gift should I get?"
- **Model Response:** "A teddy bear is almost always a sweet gift — just pick one that feels right to you."
- **Criteria:** Usefulness

> **Rater 1:** Happy face — "This is a reasonable, warm suggestion."
> **Rater 2:** Unhappy face — "This is too vague; it doesn't help me at all."

Both raters are evaluating the exact same response against the exact same criterion. They disagree. **Who is right?** Neither and both — this is genuine subjectivity.

### Why Subjectivity is Fundamental, Not Accidental

The problem is not that the raters are poorly trained. The problem is that:
1. **"Usefulness" is inherently context-dependent** — what is useful for a 5-year-old's party is not useful for a corporate gift.
2. **People have different priors** — someone who loves teddy bears finds this useful; someone who finds them childish does not.
3. **The prompt itself is ambiguous** — "birthday gift" gives no constraints on age, relationship, budget, etc.

This means human evaluation has **irreducible noise** — even perfect annotators will disagree on genuinely ambiguous cases.

### Other Limitations of Human Evaluation
- **Expensive** — quality annotators cost money; scaling to thousands of examples is cost-prohibitive.
- **Slow** — human annotation takes days/weeks; you cannot iterate quickly.
- **Fatigue effects** — annotators become inconsistent over long sessions.
- **Anchoring bias** — raters are influenced by what they see first.
- **Cultural bias** — raters bring their own cultural context.

---

## Topic 4: Coefficient of Agreement — Measuring Rater Reliability

### The Core Question

Given that raters disagree, how do we **quantify the reliability** of human evaluation? If two raters agree 70% of the time, is that good or bad? It depends on how often they would agree **by pure chance**!

### Cohen's Kappa — The Key Formula

The **Coefficient of Agreement** (Cohen's Kappa, κ) answers:

> *"How much better is our actual agreement compared to what we would expect just by chance, given how each rater uses the categories?"*

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

Where:
- **p_o** = "Observed" agreement — the proportion of cases where both raters gave the same label
- **p_e** = "Expected" agreement by chance — computed from the marginal distributions of each rater's labels
- **1 - p_e** = maximum possible improvement over chance

### Intuition with an Example

Suppose two raters rate 100 responses as "Good" or "Bad":
- Rater A says: 60 Good, 40 Bad
- Rater B says: 65 Good, 35 Bad
- They agree on 70 cases (both say Good or both say Bad)

**p_o = 70/100 = 0.70**

By chance, they'd both say "Good" on: (60/100) × (65/100) = 0.39
By chance, they'd both say "Bad" on: (40/100) × (35/100) = 0.14
**p_e = 0.39 + 0.14 = 0.53**

$$\kappa = \frac{0.70 - 0.53}{1 - 0.53} = \frac{0.17}{0.47} \approx 0.36$$

### Interpreting Kappa Values

| κ Value | Interpretation |
|---------|---------------|
| < 0 | Less than chance agreement (rare but possible) |
| 0.00 – 0.20 | Slight agreement |
| 0.21 – 0.40 | Fair agreement |
| 0.41 – 0.60 | Moderate agreement |
| 0.61 – 0.80 | Substantial agreement |
| 0.81 – 1.00 | Almost perfect agreement |

### Variants of Kappa

| Variant | Use Case |
|---------|----------|
| **Cohen's Kappa** | Two raters, nominal categories |
| **Fleiss' Kappa** | More than two raters, nominal categories |
| **Krippendorff's Alpha** | Any number of raters, handles ordinal/interval/ratio scales and missing data |

> **Why Krippendorff's Alpha is most general:** It handles ordinal scales (e.g., 1–5 ratings where 3 and 4 are "closer" than 1 and 5), whereas Cohen's and Fleiss' Kappa treat all disagreements equally regardless of magnitude.

### Why This Matters for LLM Research

Before trusting any human evaluation result in a paper, you should ask:
1. **What was the inter-rater reliability (IRR)?** Low κ means the evaluation is unreliable.
2. **How many raters per item?** More raters reduce noise.
3. **What guidelines were given?** Vague rubrics produce low IRR.

---

## Summary of Section 1

```
LLM Evaluation
    └── Output Quality (our focus)
            ├── Instruction Following
            ├── Coherence
            └── Factuality

Human Evaluation
    ├── Gold standard (closest to ground truth)
    ├── Necessary for free-form text
    ├── Limited by subjectivity
    └── Measured via Coefficient of Agreement (Cohen's Kappa)
```

---

## Interview Questions

### Conceptual / Foundational

**Q1. What is the difference between "output quality" and "system performance" in the context of LLM evaluation?**

> **Answer:** Output quality refers to the linguistic and semantic quality of the generated text — does it correctly follow instructions, is it coherent, and is it factually accurate? System performance refers to operational characteristics of the serving infrastructure — latency (how fast does it respond?), pricing (cost per token), and reliability (uptime, failure rates). A model can have excellent output quality but terrible system performance (e.g., very slow inference), or vice versa. For most NLP research, "evaluation" refers to output quality.

---

**Q2. Why is human evaluation considered the "gold standard" for LLM outputs?**

> **Answer:** LLMs generate open-ended, free-form text where correctness is not binary. Human judges are the only evaluators who can understand nuance, tone, cultural context, real-world usefulness, and subtle factual errors — all of which automated metrics often miss. Additionally, the ultimate goal of LLMs is to be useful to humans, so human judgment is the most direct measure of that goal. This is why RLHF (Reinforcement Learning from Human Feedback) uses human preferences as the training signal.

---

**Q3. What is Cohen's Kappa, and why is raw percentage agreement insufficient for measuring inter-rater reliability?**

> **Answer:** Cohen's Kappa (κ) is a statistic that measures inter-rater agreement while correcting for the agreement expected by chance. Raw percentage agreement (p_o) is insufficient because it doesn't account for chance: if a task has only two labels and one rater always says "Yes," both raters might agree 50% of the time purely by chance. Kappa normalizes observed agreement against this chance baseline: κ = (p_o - p_e) / (1 - p_e). A κ of 0 means the raters are doing no better than chance; a κ of 1 means perfect agreement.

---

**Q4. A paper reports human evaluation results but does not report inter-rater reliability. Should you trust the results? Why or why not?**

> **Answer:** Not without reservation. Without IRR (e.g., Cohen's Kappa), we don't know if the human judges were consistent with each other. If κ is very low (e.g., 0.2), the evaluation is essentially noise — different judges would give different outcomes. This makes the results non-reproducible and potentially misleading. A rigorous evaluation should always report IRR to validate that the criteria were interpreted consistently.

---

**Q5. When would you use Krippendorff's Alpha instead of Cohen's Kappa?**

> **Answer:** Krippendorff's Alpha is preferred when: (1) there are more than two raters (Cohen's Kappa is strictly for two raters; Fleiss' Kappa handles multiple but doesn't handle ordinal scales properly), (2) the rating scale is ordinal (e.g., 1–5 where distances matter), or (3) there is missing data (some items rated by fewer raters). Alpha handles all these cases in a unified framework and is the most general measure of reliability.

---

### Advanced / Applied

**Q6. In your LLM product, you want to set up human evaluation. What would your setup look like, and what pitfalls would you guard against?**

> **Answer:** Setup: (1) Define clear criteria with concrete rubrics (e.g., "Factuality: 1=clearly wrong, 3=partially correct, 5=fully verified") to minimize subjectivity, (2) use at least 3 raters per item to get majority votes, (3) compute Fleiss' Kappa or Krippendorff's Alpha to validate IRR, (4) use diverse annotators to reduce cultural and demographic bias, (5) include calibration items with known ground truth to catch drifting annotators.
> Pitfalls: anchoring bias (randomize order of model outputs), fatigue (limit sessions to 1–2 hours), context contamination (don't let raters see model identity), and ambiguous prompts in the test set.

---

**Q7. Explain how human evaluation connects to RLHF (Reinforcement Learning from Human Feedback).**

> **Answer:** In RLHF, human evaluators compare pairs of model outputs and indicate which is better (pairwise preference). These preferences train a **reward model** that learns to predict human preference scores. This reward model then provides a training signal (reward) to the LLM via reinforcement learning (typically PPO). The key insight is that it is easier for humans to compare two outputs than to assign an absolute score — this is why pairwise preference is used rather than pointwise rating. The quality of the entire RLHF pipeline depends critically on the quality and consistency of the human evaluations.

---

## Learning Thoughts

> **Thought 1 — The Evaluation Paradox:**
> The hardest thing to evaluate is also the thing we care most about. LLMs excel at free-form, creative, contextual tasks — and those are precisely the tasks where automated evaluation fails. This creates a fundamental tension: the better LLMs get, the harder it is to measure "better."

> **Thought 2 — Subjectivity is a Feature, Not a Bug:**
> The fact that two raters disagree on whether a response is "useful" does not mean evaluation is broken. It reflects genuine human diversity. A truly helpful LLM may need to give *different* answers to the same prompt depending on user context — which means evaluation must eventually become personalized.

> **Thought 3 — Kappa as a Quality Bar:**
> Before trusting *any* human evaluation in a research paper, always look for the IRR number. If a paper doesn't report κ (or equivalent), treat the human evaluation results with significant skepticism. This is a common gap in published LLM research.

> **Thought 4 — The Cost Wall:**
> Human evaluation doesn't scale. As LLMs are deployed to millions of users and evaluated across hundreds of tasks, human evaluation becomes prohibitively expensive. This is the exact motivation for the next two sections: automated statistical metrics and LLM-as-a-Judge.

> **Thought 5 — Ground Truth is Relative:**
> Even human ratings are not "ground truth" — they are human *opinions*. The fiction of a single ground truth is comfortable but false. The best we can do is measure *consensus* and *reliability*, not absolute correctness.

---

*Next: [Section 2 — Statistical and Semantic Metrics](Section2_Statistical_Semantic_Metrics.md)*
