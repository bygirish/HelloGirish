# Section 2: LLM Evaluation — Statistical and Semantic Metrics

> **Lecture:** LLM Evaluations and AI Safety | IIT Kharagpur × upGrad
> **Instructor:** Prof. Sourangshu Bhattacharya, CSE, IIT Kharagpur

---

## Learning Objective
Understand the four main automated evaluation metrics — ROUGE, BLEU, METEOR, and BERTScore — their mathematical foundations, how to implement them in Python, their strengths and failure modes, and when to use each.

---

## Topic 5: Why Evaluate LLMs with Automated Metrics?

Human evaluation is slow and expensive. We need automated metrics that can run instantly on thousands of examples. But three fundamental challenges make this hard:

### Challenge 1 — Randomness & Creativity
LLMs are stochastic. The same prompt with temperature > 0 produces different outputs each run. This means:
- A metric must evaluate a *specific output*, not an average
- Good outputs may look very different from the reference even when correct

### Challenge 2 — Computational Cost
Running a full LLM inference for evaluation is expensive. Metrics must be:
- **Fast**: should not require another LLM call
- **Scalable**: should work on millions of examples without GPU clusters

### Challenge 3 — Beyond Word Matching
This is the most important challenge. Consider:

> **Reference:** "A plush teddy bear can comfort a child during bedtime."
> **Candidate A:** "Soft stuffed bears often help kids feel safe as they fall asleep."
> **Candidate B:** "Many youngsters rest more easily at night when they cuddle a gentle toy companion."

Both candidates express the **same meaning** as the reference — but share almost no words with it. A naive word-overlap metric would score both near 0, which is wrong. This is why semantic understanding is critical.

---

## Topic 6: Precision & Recall — The Two Foundations

All four metrics are built on two primitive concepts. Master these first.

### Setup: The Running Example

The lecture uses this pair throughout:
- **Reference (gold):** *"The quick brown dog jumps over the lazy fox."*
- **Candidate (model output):** *"The quick brown fox jumps over the lazy dog."*

Note: only "dog" and "fox" are swapped. These are near-identical sentences.

### Precision

> **Question:** What fraction of the words in the *candidate* are also in the *reference*?

Matched words (appear in both): The, quick, brown, jumps, over, lazy → 7 of 9 candidate words are matched (ignoring "fox" and "dog" are swapped but still appear in reference at unigram level → actually all 9 unigrams match at unigram level since the word set is identical)

$$\text{Precision} = \frac{\text{True Positives}}{\text{True Positives + False Positives}} = \frac{\text{matched candidate words}}{\text{total candidate words}}$$

**Interpretation:** High precision = the candidate doesn't contain irrelevant words. A very short candidate that matches perfectly has high precision.

**Weakness of precision alone:** A candidate of just "The" would have precision = 1.0 (trivially).

### Recall

> **Question:** What fraction of the words in the *reference* appear in the *candidate*?

$$\text{Recall} = \frac{\text{True Positives}}{\text{True Positives + False Negatives}} = \frac{\text{matched reference words}}{\text{total reference words}}$$

**Interpretation:** High recall = the candidate covers most of what's in the reference. Useful for summarization (did you include all key points?).

**Weakness of recall alone:** Copying the entire reference into the candidate gives recall = 1.0.

### The Complementarity

- **Precision** penalizes *adding irrelevant content* (verbosity)
- **Recall** penalizes *missing reference content* (incompleteness)
- Together they balance each other — which is why F-score = harmonic mean of both

$$F_1 = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

## Topic 7: ROUGE — Recall-Oriented Understudy for Gisting Evaluation

### What ROUGE Measures

ROUGE measures the **overlap between generated text and reference text**, primarily emphasizing **recall** (how much of the reference is captured). It was designed for **text summarization**.

### ROUGE-N: N-gram Overlap

ROUGE-N counts matching N-grams between candidate and reference:

$$\text{ROUGE-N} = \frac{\Sigma(\text{matched N-grams})}{\Sigma(\text{reference N-grams})}$$

- **ROUGE-1:** Unigram (single word) overlap
- **ROUGE-2:** Bigram (two consecutive words) overlap — captures phrase-level matches
- Higher N → captures more phrase-level structure

### ROUGE-L: Longest Common Subsequence (LCS)

Instead of counting N-grams, ROUGE-L finds the **longest common subsequence** between candidate and reference.

**LCS advantage:** Captures word order better than N-grams. Words don't need to be consecutive — they just need to appear in the same relative order.

**Example:**
- Reference: "The cat sat on the mat"
- Candidate: "The cat on the mat sat"
- LCS = "The cat on the mat" (length 5)

This penalizes reordering without requiring exact adjacent matches.

### Stemming in ROUGE (use_stemmer=True)

With stemming enabled, words are reduced to their root before matching:
- "running" → "run"
- "jumps" → "jump"

This improves recall for morphological variants (plural/singular, verb tenses) that mean the same thing.

### Code Example

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(
    ['rouge1', 'rougeL'], use_stemmer=True
)

scores = scorer.score(
    'The quick brown dog jumps over the lazy fox.',   # reference
    'The quick brown fox jumps over the lazy dog.'   # candidate
)
print(scores)
# rouge1: Score(precision=1.0, recall=1.0, fmeasure=1.0)
# rougeL: Score(precision=0.778, recall=0.778, fmeasure=0.778)
```

**Explanation of results:**
- **rouge1 = 1.0:** Every word in both sentences is the same (just reordered) — all unigrams match.
- **rougeL = 0.778:** LCS penalizes the reordering — "dog" and "fox" are in swapped positions, so LCS = 7/9.

### When to Use ROUGE
- Text **summarization** (is the summary covering the key points from the source?)
- Any task where **recall of reference content** is the priority

---

## Topic 8: BLEU — Bilingual Evaluation Understudy

### What BLEU Measures

BLEU measures **precision of N-grams** from the candidate against the reference. It was originally designed for **machine translation**.

$$\text{BLEU} = \text{BP} \times \exp\left(\sum_{n=1}^{N} w_n \times \log(p_n)\right)$$

Where:
- **p_n** = precision of N-grams (for n = 1, 2, ..., N; typically N=4)
- **w_n** = weight for each N-gram order (typically 1/N = 0.25 each)
- **BP** = Brevity Penalty

### Component 1: N-gram Precision

Counts how many N-grams in the candidate appear in the reference. Higher N catches phrase-level accuracy.

**Clipping:** To prevent gaming, each reference N-gram can only be matched once (prevents repeating a common word to boost score).

### Component 2: Brevity Penalty (BP)

$$\text{BP} = \begin{cases} 1 & \text{if } |c| \geq |r| \\ \exp(1 - |r|/|c|) & \text{if } |c| < |r| \end{cases}$$

Where |c| = candidate length, |r| = reference length.

**Why needed?** Without BP, a candidate of a single word that appears in the reference would get precision=1.0. BP penalizes candidates that are shorter than the reference.

### Feature: Multiple References

BLEU supports multiple reference translations — the candidate is matched against any of them. This handles valid paraphrases:
- Reference 1: "The cat sat on the mat"
- Reference 2: "A cat was sitting on the rug"
- Candidate: "A cat sat on the mat" → matches well against both

### Code Example

```python
from nltk.translate.bleu_score import sentence_bleu

reference = [['The','quick','brown','fox','jumps','over','the','lazy','dog']]
candidate = ['The','quick','brown','dog','jumps','over','the','lazy','fox']

score = sentence_bleu(reference, candidate)
print(score)  # 0.4597
```

**Key observation:** Only "fox" and "dog" are swapped, yet BLEU drops to 0.46! This reveals BLEU's key limitation: it is very sensitive to **N-gram order**. Bigrams and trigrams that contain the swap don't match, heavily penalizing the score.

### Score Interpretation
- **1.0** = perfect match
- **> 0.3** = generally considered reasonable quality for translation
- **0.46** for near-identical sentences = BLEU is too harsh

### When to Use BLEU
- **Machine translation** (is the translation precise and order-preserving?)
- Tasks where **precision** matters more than recall

---

## Topic 9: METEOR — Metric for Evaluation of Translation with Explicit ORdering

### Why METEOR Was Created

BLEU has two major weaknesses:
1. It doesn't handle morphological variants ("running" ≠ "run")
2. It ignores synonyms ("fast" ≠ "quick")
3. It doesn't weight recall highly enough

METEOR was designed to fix all three.

### METEOR Formula

$$\text{METEOR} = F_{\text{mean}} \times (1 - \text{Penalty})$$

$$F_{\text{mean}} = \frac{10 \times P \times R}{R + 9P}$$

The formula gives **extra weight to recall** (R has a higher coefficient than P). Missing reference content is penalized more than extra candidate content.

### The Four Components

| Feature | What It Does | Why It Matters |
|---------|-------------|----------------|
| **Harmonic F-score** | Balances precision and recall, weighted toward recall | Missing reference content is more serious |
| **Stem Matching** | "running" matches "run" | Catches morphological variants BLEU misses |
| **Synonym Matching** | "fast" ↔ "quick" via WordNet | Captures paraphrase equivalence |
| **Chunk Penalty** | Penalizes fragmented matches | Rewards contiguous, ordered matches |

### Chunk Penalty Explained

A "chunk" is a maximal sequence of adjacent matched words. Fewer, longer chunks = better match.

- **Good:** [The quick brown fox] [jumps] → 2 chunks (mostly contiguous)
- **Bad:** [The] [quick] [brown] [fox] [jumps] → 5 chunks (fragmented)

More chunks → higher penalty → lower score. This rewards word order preservation.

### Code Example

```python
from nltk.translate import meteor_score
from nltk import word_tokenize
import nltk
nltk.download('wordnet')

reference = "The quick brown fox jumps over the lazy dog"
candidate = "The fast brown fox jumps over the lazy dog"

score = meteor_score.meteor_score(
    [word_tokenize(reference)],
    word_tokenize(candidate)
)
print(score)  # ~0.99
```

**Key result:** BLEU scored 0.46 on this pair; METEOR scores 0.99. The word "fast" is a synonym of "quick," and METEOR's WordNet lookup catches this. This dramatically demonstrates METEOR's superior semantic awareness.

### When to Use METEOR
- **Translation** tasks where synonyms and stems should not be penalized
- Any task where **paraphrase equivalence** matters

---

## Topic 10: BERTScore — Semantic Similarity via BERT Embeddings

### The Fundamental Shift

ROUGE, BLEU, and METEOR are all based on **token matching** — they compare words or N-grams. BERTScore takes a completely different approach: it compares **meanings** by using contextual embeddings from BERT.

### How BERTScore Works

```
Candidate + Reference
        ↓
   BERT Encoder
        ↓
Contextual vectors (one per token)
        ↓
Pairwise cosine similarity matrix (n² comparisons)
        ↓
Greedy matching (Precision, Recall, F1)
```

### Step-by-Step

1. **Encode:** Pass both sentences through a pre-trained BERT model. Each token gets a contextual vector.

2. **Why contextual?** The word "bank" in "river bank" gets a different vector from "bank" in "bank account." BERT captures meaning in context, not just word identity.

3. **Pairwise cosine similarity:** Compute cosine similarity between every candidate token vector and every reference token vector. This gives an n×m matrix.

4. **Greedy match for Precision:** For each candidate token, find the *most similar* reference token. Average these max similarities.

5. **Greedy match for Recall:** For each reference token, find the *most similar* candidate token. Average these max similarities.

6. **F1:** Harmonic mean of precision and recall.

### Code Example

```python
import torch
from bert_score import score

cands = ['The quick brown dog jumps over the lazy fox.']
refs  = ['The quick brown fox jumps over the lazy dog.']

P, R, F1 = score(cands, refs, lang='en', verbose=True)
print(f'Precision: {P.mean():.4f}')  # ~0.9640
print(f'Recall:    {R.mean():.4f}')  # ~0.9640
print(f'F1:        {F1.mean():.4f}') # ~0.9640
```

**Key result:** BERTScore F1 ≈ 0.964. BERT recognizes that "fox" and "dog" are both animals in similar syntactic positions and gives near-perfect similarity. This is the most semantically aware of the four metrics.

### Comparison on the Same Sentence Pair

| Metric | Score | Reason |
|--------|-------|--------|
| BLEU | 0.46 | Bigrams with swapped words don't match |
| ROUGE-L | 0.778 | LCS misses swapped positions |
| METEOR | 0.99 | Synonym matching: fast ↔ quick |
| BERTScore F1 | 0.964 | BERT embedding similarity captures semantic proximity |

### Important Practical Note

- BERTScore **requires downloading a BERT model** on first run (~400MB)
- It is **much slower** than ROUGE/BLEU (seconds vs milliseconds per example)
- For large-scale evaluation pipelines, plan for inference time accordingly
- Supports 100+ languages via multilingual BERT (`lang='xx'`)

### When to Use BERTScore
- **Chatbots / QA** where semantic equivalence matters more than exact words
- Research evaluation where you want the highest correlation with human judgment
- Any task where paraphrases should score as well as exact matches

---

## Topic 11: Metric Comparison — When to Use Which

| Metric | Measures | Best For | Speed | Semantic? |
|--------|----------|----------|-------|-----------|
| ROUGE | Recall of N-grams / LCS | Text summarization | Fast | No |
| BLEU | Precision of N-grams + Brevity Penalty | Machine translation | Fast | No |
| METEOR | Harmonic P & R + stems/synonyms | Translation & generation | Medium | Partial |
| BERTScore | Cosine similarity of BERT embeddings | Semantic similarity tasks | Slow | Yes |

### Decision Guide by Task

| Task | Recommended Metric(s) | Why |
|------|-----------------------|-----|
| Text Summarization | ROUGE | Recall of reference content is priority |
| Machine Translation | BLEU + METEOR | Precision + semantic recall together |
| Chatbot / QA | BERTScore | Semantic equivalence matters most |
| Research Evaluation | All four | Each catches different failure modes |
| Reasoning Model | BERTScore | Captures semantic correctness, not just word overlap |

> **Interview Question setup:** "Which score is most suitable for evaluating a reasoning model?" → **BERTScore**, because reasoning outputs can be semantically correct while using completely different wording from a reference.

---

## Topic 12: Limitations of All Automated Metrics

### Limitation 1: Stylistic Variation

All metrics struggle with stylistically equivalent outputs:
- "A plush teddy bear can comfort a child during bedtime."
- "Soft stuffed bears often help kids feel safe as they fall asleep."

These are semantically equivalent but use entirely different vocabulary. Even BERTScore doesn't always catch this perfectly.

### Limitation 2: Poor Correlation with Human Ratings

Research has shown that **none of these metrics correlates strongly with human judgments** across diverse tasks. The paper *"NLG Evaluation Metrics Beyond Correlation Analysis"* demonstrates that BERTScore F1 — the best of the four — still shows moderate KS divergence from human ratings, especially at nuanced quality levels.

### Limitation 3: Still Requires Human Ratings (for validation)

The cruel irony: to *validate* that an automated metric is good, you need human ratings to compare against. You cannot escape human evaluation — you can only try to approximate it cheaply.

### Implication: Use Multiple Metrics

No single metric is sufficient. Best practice is to report at least two metrics that measure complementary things (e.g., ROUGE for recall + BERTScore for semantics). This catches more failure modes.

---

## Interview Questions

**Q1. What is the difference between ROUGE and BLEU? When would you use one over the other?**

> **Answer:** Both are N-gram based metrics. The key difference is: ROUGE emphasizes **recall** (what fraction of the reference appears in the candidate) and is best for **summarization** where covering key reference content is the priority. BLEU emphasizes **precision** (what fraction of the candidate appears in the reference) and is best for **machine translation** where generating accurate, precise translations matters. BLEU also adds a Brevity Penalty to prevent gaming via very short outputs.

---

**Q2. Why does BLEU score 0.46 for two sentences that differ by only two swapped words?**

> **Answer:** BLEU uses N-gram precision. When "dog" and "fox" are swapped, bigrams and trigrams containing those words (e.g., "brown dog", "dog jumps", "lazy fox") don't match the reference (which has "brown fox", "fox jumps", "lazy dog"). Bigrams and trigrams are heavily penalized. Since BLEU combines precision from N=1 through N=4 (or N=2 for sentence BLEU), the higher-order N-grams dramatically reduce the score. This is BLEU's known weakness: it's very sensitive to word order even for near-synonymous rearrangements.

---

**Q3. How does BERTScore work, and why is it more semantically aware than ROUGE/BLEU?**

> **Answer:** BERTScore encodes both the candidate and reference through a pre-trained BERT model to get contextual token embeddings. It then computes a pairwise cosine similarity matrix between all candidate and reference tokens. Precision is the average max similarity per candidate token; Recall is the average max similarity per reference token. Unlike ROUGE/BLEU which require exact token matches, BERTScore recognizes semantic proximity — "dog" and "fox" are both animals, "fast" and "quick" are near-synonyms — giving high scores even when word choice differs. It correlates better with human judgments than token-matching metrics.

---

**Q4. What is the Brevity Penalty in BLEU and why is it necessary?**

> **Answer:** The Brevity Penalty (BP) penalizes candidate outputs that are shorter than the reference. BP = 1 when the candidate is at least as long as the reference; BP = exp(1 - |ref|/|cand|) when the candidate is shorter. Without BP, a model could game the precision score by generating a single very common word (e.g., "the") that matches the reference, getting precision=1.0. BP ensures that a valid translation must be roughly as long as the reference.

---

**Q5. What does METEOR do that BLEU does not, and why does this matter?**

> **Answer:** METEOR adds three things BLEU lacks: (1) **Stem matching** — "running" and "run" are treated as matches, reducing false negatives for morphological variants; (2) **Synonym matching** — via WordNet, "fast" and "quick" count as matches, capturing paraphrase equivalence; (3) **Weighted recall** — METEOR's F-score formula weights recall more heavily than precision, because missing reference content is generally worse than extra candidate content. These features make METEOR much more forgiving of valid paraphrases and morphological variations, improving correlation with human judgments over BLEU.

---

**Q6. If you had to evaluate a customer support chatbot, which metric would you use and why?**

> **Answer:** BERTScore, primarily. In customer support, what matters is whether the model's response is semantically correct and helpful — not whether it uses the exact same words as a reference. A response saying "Your refund will be processed in 3–5 business days" is equivalent to "Please expect your refund within 3 to 5 working days," but these share few N-grams. BERTScore's BERT embeddings would recognize their semantic equivalence. If I had to add a second metric for coverage, I'd add ROUGE-L to ensure the response covers all key points from the reference answer.

---

**Q7. All four metrics have limitations. What is the fundamental limitation they all share?**

> **Answer:** All four metrics require a **reference output** (a ground truth) and measure how close the candidate is to that reference. This is fundamentally limiting because: (1) there are often many valid correct responses to a prompt; (2) a creative or novel but correct response may score poorly if it doesn't resemble the reference; (3) these metrics cannot assess factual accuracy without a reference (and even with one, they compare form, not fact-checking). The deeper limitation is that these metrics measure **similarity to a reference** rather than **actual quality** — which is why they don't correlate perfectly with human judgments.

---

## Learning Thoughts

> **Thought 1 — Each Metric is a Lens:**
> ROUGE sees "did you include the reference content?" BLEU sees "is your content precise?" METEOR sees "did you express the same ideas?" BERTScore sees "are the meanings similar?" None is complete. Use them as complementary lenses on the same output.

> **Thought 2 — Semantic Gap is the Core Problem:**
> The fundamental unsolved problem is: how do you measure meaning rather than form? BERTScore gets closer, but BERT embeddings still have limits (out-of-domain vocabulary, long documents). The field is actively working on better semantic evaluation.

> **Thought 3 — The Reference Dependency Problem:**
> All four metrics assume you have a reference. But in open-ended generation (creative writing, brainstorming, novel Q&A), there is no single correct reference. This is why LLM-as-a-Judge (Section 3) becomes necessary — it can evaluate without a fixed reference.

> **Thought 4 — BLEU's Legacy vs Current Practice:**
> BLEU dominated NLP evaluation for 20 years despite known weaknesses, largely due to inertia and standardization. Today, BERTScore is increasingly preferred, and the field is moving toward LLM-based evaluation (Section 3). But BLEU and ROUGE remain standard baselines.

> **Thought 5 — Speed vs Quality Tradeoff:**
> ROUGE and BLEU are microseconds-per-example. BERTScore is seconds-per-example. For a dataset of 1 million examples, the difference is minutes vs days. This engineering reality determines which metric is practical for your use case.

---

*Previous: [Section 1 — Human Evaluation](Section1_Human_Evaluation.md)*
*Next: [Section 3 — LLM-as-a-Judge](Section3_LLM_as_a_Judge.md)*
