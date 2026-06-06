# Topic 5 — LLM Evaluation

> **Session:** 2.2 — Prompt Optimization and Security | IIT Kharagpur × upGrad  
> **Instructor:** Koustav Rudra, Assistant Professor AI, IIT Kharagpur

---

## Overview

LLM Evaluation is the discipline of **systematically measuring how well an LLM system performs** on a task. It sits at the heart of prompt optimization — you can't optimize what you can't measure.

### Why Automated Evaluation

Manual evaluation (human review of each output) is:
- **Time-consuming** — impractical for thousands of outputs
- **Costly** — human annotators are expensive
- **Non-scalable** — cannot run after every prompt change in a CI/CD pipeline
- **Subjective** — different humans rate differently

Automated evaluation complements or replaces human review with algorithms that measure the same qualities: fluency, coherence, relevance, factual consistency, and fairness.

### The Quality Dimensions of LLM Output

| Dimension | What It Measures |
|---|---|
| **Fluency** | Is the text grammatically well-formed and readable? |
| **Coherence** | Do sentences and paragraphs connect logically? |
| **Relevance** | Does the output address the actual input? |
| **Factual Consistency** | Are claims consistent with a provided context? |
| **Fairness** | Is the output free from bias and discrimination? |

---

## 5.1 LLM Evaluation Metric Architecture

### The Standard Pipeline

```
LLM Test Case:
  - Input (user query)
  - LLM Output (generated response)
  - Retrieval Context (if RAG pipeline)
  - Ground Truth (expected output, if available)
         ↓
  LLM Judge Scorer
         ↓
  Score (numerical) + Reason (optional)
         ↓
  Passes Threshold? → Metric: Passed / Failed
```

### Two Types of Testing

**Golden-Output Testing:**
Compare the model's output directly against a stored "gold" reference answer. Works well when there is one correct answer (factual QA, code generation). Brittle for open-ended tasks — any valid paraphrase will falsely fail.

**Behavior-Contract Testing:**
Check that the output satisfies a set of properties:
- "The response is under 100 words"
- "The JSON output is schema-valid"
- "The response does not mention competitor brand names"
- "The response includes a refund policy reference"

This approach is more robust for open-ended tasks where multiple valid outputs exist.

---

## 5.2 Common LLM Evaluation Metrics

These are the standard metrics used to evaluate whether an LLM application is working correctly:

| Metric | What It Checks |
|---|---|
| **Answer Relevancy** | Output addresses the input in an informative and concise manner |
| **Task Completion** | Agent completes the task it was set out to do |
| **Correctness** | Output is factually correct relative to ground truth |
| **Hallucination** | Output does NOT contain fabricated information |
| **Tool Correctness** | Agent calls the correct tools with correct parameters |
| **Context Faithfulness** | Output is grounded in provided context (RAG tasks) |

### Task-Specific Metrics

Generic metrics don't capture everything. For specialized tasks, you need **custom criteria**:

**Example — News Summarization System:**
- "Does the summary contain enough information from the original text?"
- "Does the summary contain any contradictions or hallucinations from the original text?"

These custom rubrics are typically implemented as LLM-as-a-judge evaluators (covered in 5.5).

---

## 5.3 Taxonomy: Ways to Compute Metric Scores

```
                Statistical Scorers        Model-Based Scorers
               ┌──────────────────────────────────────────────┐
Word-based:    │ BLEU, ROUGE, METEOR                          │
Char-based:    │ Levenshtein Distance                         │
Embedding:     │          BERTScore, MoverScore               │
LLM-based:     │          QAG, GPTScore, GEval, DAG           │
Other NLP:     │                     NLI, BLEURT              │
               └──────────────────────────────────────────────┘
```

**Statistical Scorers:** Use string/character/token matching algorithms. Fast, deterministic, no model needed.

**Model-Based Scorers:** Use pre-trained models (BERT, LLMs) to score output quality. More semantically aware but slower and require model inference.

---

## 5.4 Statistical Scorers

### BLEU — BiLingual Evaluation Understudy

**Original purpose:** Machine translation quality (1990s)  
**Now used for:** Any text generation task where a reference output exists

**How it works:**  
BLEU measures **n-gram overlap** between the generated output and one or more reference translations. An n-gram is a contiguous sequence of n tokens.

```
Reference: "The cat sat on the mat"
Candidate: "The cat sat on the hat"

Unigram overlap: 5/6 = 0.83
Bigram overlap: 4/5 = 0.80
BLEU also applies a brevity penalty for short outputs that cherry-pick words.
```

**Formula (simplified):**
```
BLEU = BP × exp(Σ wₙ × log pₙ)

where:
  BP = brevity penalty (1 if output ≥ reference length, else < 1)
  pₙ = n-gram precision for n=1,2,3,4
  wₙ = weight for each n-gram order (typically 1/4 each)
```

**Strengths:** Fast, deterministic, widely benchmarked, no model required  
**Limitations:** Doesn't capture synonyms, paraphrases, or semantic similarity. A semantically perfect answer phrased differently scores poorly.

---

### ROUGE — Recall-Oriented Understudy for Gisting Evaluation

**Original purpose:** Text summarization quality  
**Now used for:** Summarization, abstractive generation tasks

**How it works:**  
ROUGE measures n-gram **recall** (not precision like BLEU) — what fraction of the reference n-grams appear in the candidate?

Three main variants:

| Variant | What It Measures |
|---|---|
| **ROUGE-N** | N-gram recall overlap (N=1 or 2 most common) |
| **ROUGE-L** | Longest Common Subsequence (LCS) between candidate and reference |
| **ROUGE-W** | Weighted LCS (rewards consecutive matches more) |

**ROUGE-1 example:**
```
Reference: "The prime minister announced new economic policies"
Candidate: "New economic policies were announced by the prime minister"

ROUGE-1 recall = 6/6 = 1.0 (all reference words appear in candidate)
ROUGE-1 precision = 6/9 = 0.67 (6 of 9 candidate words match reference)
ROUGE-1 F1 = 0.80
```

**Why ROUGE for summarization?**  
Summarization cares most about recall — did the summary cover the key information from the source? Missing important content is worse than including extra words.

---

### METEOR — Metric for Evaluation of Translation with Explicit Ordering

**Addresses BLEU/ROUGE gaps** by incorporating:
1. **Stemming** — "running" and "run" are considered matches
2. **Synonym matching** — uses WordNet to match semantically equivalent words
3. **Paraphrase matching** — handles common paraphrases

```
Reference: "The cat is sitting on the mat"
Candidate: "A feline is resting on the rug"

BLEU: Near 0 (almost no exact token overlap)
METEOR: Moderate score (recognizes feline≈cat, resting≈sitting, rug≈mat)
```

METEOR is more linguistically aware than BLEU/ROUGE and better correlates with human judgments for creative or diverse text.

---

### Levenshtein Distance (Edit Distance)

**Purpose:** Measures the minimum number of single-character edits (insertions, deletions, substitutions) required to transform one string into another.

**Classic example:**
```
"kitten" → "sitting" requires 3 edits:
  kitten → sitten   (substitute 'k' with 's')
  sitten → sittin   (substitute 'e' with 'i')
  sittin → sitting  (insert 'g' at end)
Levenshtein distance = 3
```

**When to use:**  
- Spelling correction evaluation
- Code generation (exact structural match)
- Any task where character-level precision matters
- Not appropriate for paraphrase-tolerant tasks (prose, summaries)

---

## 5.5 Model-Based Scorers

### BERTScore

**Core idea:** Instead of counting exact token matches, compare outputs at the **semantic embedding level** using BERT representations.

**How it works:**
1. Encode both candidate and reference using BERT (or similar transformer)
2. For each token in the candidate, find the most similar token in the reference (cosine similarity)
3. Aggregate token-level similarities into precision, recall, and F1

**Advantage:** "The cat sat on the mat" and "A feline rested upon the rug" score much higher than BLEU would assign, because their BERT embeddings are semantically close.

**Used for:** Summarization, translation, generation tasks where paraphrase is valid

---

### MoverScore

**Extension of BERTScore** that uses Earth Mover's Distance (Wasserstein distance) to compute the minimum "cost" of transforming one text distribution into another.

More context-aware than BERTScore — considers the full sentence's meaning rather than individual token-level similarity.

---

### NLI Scorer (Natural Language Inference)

**Core idea:** Use an NLI model (trained to classify textual entailment) to check whether the candidate output is **logically consistent** with a reference text.

NLI models classify relationships as:
- **Entailment:** The candidate is logically supported by the reference
- **Contradiction:** The candidate contradicts the reference  
- **Neutral:** The candidate is unrelated to the reference

**Score:** Ranges from 0 (contradiction) to 1 (entailment)

**Best for:** Faithfulness evaluation in RAG — does the generated response contradict the retrieved context?

**Example:**
```
Context:  "Paris is the capital of France and has a population of 2.1 million."
Output A: "France's capital city is Paris." → Entailment (score: 1.0)
Output B: "Paris has a population of 5 million." → Contradiction (score: 0.0)
Output C: "The Eiffel Tower is beautiful." → Neutral (score: 0.5)
```

---

### BLEURT — Bilingual Evaluation Understudy with Representations from Transformers

Pre-trained models (based on BERT) that are **fine-tuned on human judgment scores**. Unlike BERTScore (which uses raw embeddings), BLEURT has been trained to directly predict human quality ratings.

Output: A scalar score that correlates with how a human would rate the output quality.

**Advantage:** Very high correlation with human judgments across diverse tasks  
**Limitation:** Less interpretable, requires pre-trained BLEURT model, may not transfer well to domain-specific tasks

---

## 5.6 LLM-as-a-Judge: G-Eval

### The Problem with Statistical Metrics

Statistical scorers (BLEU, ROUGE) fail badly on:
- **Open-ended generation** — many valid outputs
- **Reasoning tasks** — quality of logic not captured by n-grams
- **Domain-specific evaluation** — custom criteria can't be expressed as string matching

LLM-as-a-Judge uses a large language model itself to evaluate another model's outputs — a scalable proxy for human evaluation.

### G-Eval — LLM-Based Evaluation with Form Filling

**Paper:** arXiv:2303.16634  
**Core idea:** Use CoT (Chain-of-Thought) to generate structured evaluation steps, then have the LLM score the output on those steps.

### G-Eval Step-by-Step

**Phase 1: Generate Evaluation Steps (Auto-CoT)**

Prompt the LLM evaluator with:
1. A task introduction (what is being evaluated)
2. The evaluation criterion (e.g., coherence: "collective quality of all sentences")
3. The text to evaluate

The LLM generates structured evaluation steps via CoT before scoring.

Example prompt for coherence evaluation:
```
You will be given one summary written for a news article. Your task is to
rate the summary on one metric: coherence.

Coherence (1-5): The collective quality of all sentences. We align this
dimension with the DUC quality question of structure and coherence.

Evaluation steps:
1. Read the news article carefully and identify the main topic and key points.
2. Read the summary and compare it to the news article.
3. Assign a score for coherence on a scale of 1 to 5.
```

**Phase 2: Score Using the Generated Steps**

Create a scoring prompt by concatenating:
1. Generated evaluation steps
2. All arguments from the evaluation (the article, the summary)
3. "Assign a score between 1 and 5"

The LLM outputs a score (e.g., 3 for moderate coherence).

**Phase 3: Optional — Probability Weighting**

For APIs that expose token probabilities, take the probability distribution over possible score tokens (1, 2, 3, 4, 5) and compute a **weighted sum** as the final score:

```
P(score=1) = 0.05, P(score=2) = 0.20, P(score=3) = 0.50, P(score=4) = 0.20, P(score=5) = 0.05
Weighted score = 1×0.05 + 2×0.20 + 3×0.50 + 4×0.20 + 5×0.05 = 3.0
```

This is more calibrated than just taking the argmax score (the model's single highest-probability choice).

### LLM-as-a-Judge Biases — Critical Pitfalls

Using LLMs as judges introduces systematic biases that must be actively mitigated:

| Bias | Description | Mitigation |
|---|---|---|
| **Position Bias** | The judge prefers the option presented first or last in a comparison | Randomize order; average scores across orderings |
| **Verbosity Bias** | The judge rates longer, more detailed responses higher regardless of quality | Normalize for length; explicitly instruct to ignore length |
| **Self-Preference Bias** | An LLM rates outputs from its own model family higher | Use a different model family as judge |
| **Sycophancy** | The judge agrees with any user-expressed preference in the prompt | Avoid revealing expected answer in the judge prompt |
| **Metric Alignment** | The judge may not share the same values as your end users | Validate judge ratings against human labels periodically |

---

## The Scoring Hierarchy: When to Use What

```
Simplest ──────────────────────────────────────────── Most Powerful

Exact Match / Regex    →    BLEU/ROUGE    →    BERTScore    →    LLM-as-a-Judge
     ↑                          ↑                  ↑                  ↑
Closed tasks         Reference-based        Semantic aware      Open-ended
Code, JSON, dates    Translation,           Any task with       Complex quality
                     Summarization          reference           dimensions
```

**Rule of thumb:** Use the simplest scorer that captures the quality dimension you care about. Add complexity only when simpler scorers fail.

---

## Learning Highlights

> **Metric selection insight:** The wrong metric is worse than no metric. BLEU on a creative writing task punishes valid paraphrases and optimizes for the wrong behavior.

> **ROUGE vs. BLEU:** ROUGE measures recall (did you cover what was important?), BLEU measures precision (is what you said correct?). For summarization, recall matters more. For translation, precision matters more.

> **LLM-as-a-judge insight:** An LLM judge is not neutral — it has biases, preferences, and blind spots. Treat it like a human annotator: validate it, audit it, and don't use it as the sole arbiter of quality.

> **G-Eval insight:** The reason G-Eval works better than direct scoring is that CoT-generated evaluation steps force the judge LLM to articulate its reasoning before scoring. This structured reasoning produces more consistent and interpretable scores.

> **Production insight:** Human evaluation doesn't scale, but it is the ground truth. Build an automated evaluation pipeline early, but periodically validate it against human judgment — ideally 5-10% of your evaluation set should have human labels.

---

## Interview Questions

### Foundational

**Q1. What is the difference between BLEU and ROUGE? When would you use each?**

*Answer:* BLEU measures n-gram **precision** — of all the n-grams in the generated output, what fraction appear in the reference? ROUGE measures n-gram **recall** — of all the n-grams in the reference, what fraction appear in the generated output? Use BLEU for tasks where you care about output precision (translation: don't include tokens not in reference). Use ROUGE for tasks where you care about coverage of reference content (summarization: ensure you captured key points). Both are paired with F1 in practice.

---

**Q2. What is BERTScore and how does it improve over BLEU?**

*Answer:* BERTScore uses BERT token embeddings to compute semantic similarity between candidate and reference, rather than counting exact token matches. This means semantically equivalent but differently-worded outputs (paraphrases) score highly under BERTScore, whereas they score near-zero under BLEU. BERTScore correlates much better with human judgments on open-ended generation tasks. The trade-off is computational cost — BERTScore requires a BERT inference pass, while BLEU is purely string-based.

---

**Q3. What is LLM-as-a-judge and what are its main biases?**

*Answer:* LLM-as-a-judge uses a large language model to evaluate another model's output quality — scoring on dimensions like coherence, relevance, and faithfulness. Main biases: (1) position bias — the judge favors whichever output is presented first; (2) verbosity bias — the judge rates longer answers higher; (3) self-preference bias — an LLM rates outputs from the same model family higher; (4) sycophancy — the judge defers to hints about expected answers in the prompt. Mitigation: randomize position, control for length, use a different model family as judge, validate against human labels.

---

**Q4. What is G-Eval and why does it use CoT before scoring?**

*Answer:* G-Eval is an LLM-based evaluation framework that first uses Chain-of-Thought to generate structured evaluation steps (what to look for, how to assess each dimension), then uses those steps to produce a score. The CoT phase forces the judge to articulate its reasoning before committing to a score, producing more consistent, structured, and interpretable evaluations. Without the CoT phase, the LLM tends to give inconsistent scores based on superficial features.

---

### Intermediate

**Q5. Design an evaluation pipeline for a customer support LLM. What metrics would you use?**

*Answer:* (1) **Behavioral metrics (deterministic):** JSON schema validation (structured output), response length check (under 120 words), no competitor name mentions (regex). (2) **Reference-based metrics:** ROUGE-L against golden reference responses for standard queries. (3) **Semantic metrics:** BERTScore for semantic similarity to gold responses. (4) **LLM-as-a-judge metrics:** G-Eval for empathy score, factual correctness relative to product policy, and next-step clarity. (5) **Human evaluation:** 5% random sample reviewed by QA team weekly to validate automated metrics. Gates: any prompt change must not degrade any metric by more than 2%.

---

**Q6. What is Levenshtein distance and what tasks is it appropriate for?**

*Answer:* Levenshtein distance counts the minimum number of character-level insertions, deletions, or substitutions to transform one string into another. It's appropriate for: spelling correction evaluation, code generation where exact syntax matters, named entity recognition (exact entity string match), and structured field extraction. It's NOT appropriate for: summarization, open-ended QA, creative generation — any task where paraphrase or semantic equivalence should be acceptable. For those tasks, use BERTScore or LLM-as-a-judge instead.

---

**Q7. What does "G-Eval's probability-weighted score" mean and why is it better than argmax scoring?**

*Answer:* Instead of taking the LLM judge's single highest-probability score (argmax), G-Eval takes the full probability distribution over possible score tokens (1, 2, 3, 4, 5) and computes a weighted average. For example, if the judge assigns P(3)=0.5, P(4)=0.3, P(2)=0.2, the weighted score = 3.0 instead of 3 (argmax). This is better because it captures the judge's uncertainty — a 50% confident score of 3 is different from a 99% confident score of 3. The probability-weighted score also enables continuous optimization signals for APO pipelines.

---

### Advanced

**Q8. You are building a RAG evaluation pipeline. How would you evaluate faithfulness and answer relevancy?**

*Answer:* **Faithfulness** (does the answer contradict the context?): Use an NLI scorer to classify whether each claim in the generated answer is entailed by, neutral to, or contradicted by the retrieved context. Score = fraction of claims that are entailed. Flag any answer with contradicted claims as hallucination. **Answer Relevancy** (does the answer address the question?): Use an LLM-as-a-judge with a structured rubric: "Given the question and answer, score from 1-5: does the answer directly address the question without unnecessary deviation?" Alternatively, generate back-questions from the answer (what question does this answer?) and measure similarity between the generated back-question and the original.

---

**Q9. How would you validate that your LLM-as-a-judge is reliable and not systematically biased?**

*Answer:* (1) **Human correlation check:** Score 200 random outputs using both the LLM judge and human annotators. Measure Pearson/Spearman correlation. Acceptable threshold: ρ > 0.7. (2) **Position bias test:** For pairwise comparisons, randomly swap the order of outputs A and B. If the judge changes its preference purely based on order, it has position bias. (3) **Adversarial test:** Create pairs where a clearly inferior long response is compared to a superior short one. If the judge favors length, it has verbosity bias. (4) **Self-preference test:** Have Claude judge both Claude and GPT outputs without labeling which is which. Check if scores are systematically different. (5) **Calibration check:** Verify that a score of "4" from the judge corresponds to the same human rating across different tasks and domains.

---

## Quick Reference Summary

| Metric | Type | Best For | Limitation |
|---|---|---|---|
| BLEU | Statistical (precision) | Translation | Penalizes valid paraphrases |
| ROUGE | Statistical (recall) | Summarization | Same as BLEU for synonyms |
| METEOR | Statistical + WordNet | Translation/Generation | Slower, requires WordNet |
| Levenshtein | Statistical (character) | Spelling, exact match | Not for semantic tasks |
| BERTScore | Model-based (embedding) | Any task with reference | Requires BERT inference |
| NLI | Model-based | Faithfulness / RAG | Requires NLI model |
| BLEURT | Model-based (fine-tuned) | High correlation with humans | Domain transfer issues |
| G-Eval | LLM-as-a-judge | Open-ended quality | Subject to LLM biases |
