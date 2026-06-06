# Block 3: Evaluation Metrics
## Confusion Matrix, Precision, Recall, F1, Accuracy

> **Session:** Lecture 2 — The Transformer Architecture  
> **Topics covered:** 14

---

## Learning Roadmap for This Block

```
Why accuracy alone is misleading
→ Confusion Matrix (the foundation)
→ TP / TN / FP / FN (the four outcomes)
→ Precision, Recall, F1, Accuracy
→ When to use which metric
→ The precision-recall tradeoff
```

This block answers: **After you train your model, how do you know if it's actually any good?**

---

## Topic 14: Confusion Matrix & Evaluation Metrics

### Why Accuracy Alone is Misleading

Consider a cancer detection model trained on a dataset with 99% healthy patients and 1% cancer patients.

A model that predicts **"healthy" for everyone** achieves:
```
Accuracy = 99%
```

But it catches **zero cancer cases**. It's a completely useless model with 99% accuracy. This is the class imbalance problem — and it's why we need better metrics.

### The Confusion Matrix

The confusion matrix is a table that reveals **exactly where** your model is right and wrong — broken down by class.

For a binary classifier (Positive / Negative):

```
                    PREDICTED
                  Positive    Negative
ACTUAL  Positive |    TP    |    FN    |
        Negative |    FP    |    TN    |
```

### The Four Outcomes

| Outcome | Meaning | Example (Cancer Detection) |
|---|---|---|
| **TP** — True Positive | Model predicted Positive, actually Positive | Model says "cancer", patient HAS cancer ✓ |
| **TN** — True Negative | Model predicted Negative, actually Negative | Model says "healthy", patient IS healthy ✓ |
| **FP** — False Positive (Type I Error) | Model predicted Positive, actually Negative | Model says "cancer", patient is healthy ✗ |
| **FN** — False Negative (Type II Error) | Model predicted Negative, actually Positive | Model says "healthy", patient HAS cancer ✗ |

### The Mnemonic to Remember

```
True/False = was the prediction correct?
Positive/Negative = what did the model predict?

So: True Positive  = correctly predicted Positive (prediction was right)
    False Positive = incorrectly predicted Positive (prediction was wrong)
```

### Concrete Example — Cancer Detection

Suppose we test 1000 patients (100 have cancer, 900 are healthy):

| | Predicted Cancer | Predicted Healthy |
|---|---|---|
| **Actually Cancer** | 80 (TP) | 20 (FN) |
| **Actually Healthy** | 50 (FP) | 850 (TN) |

From this matrix, we can compute all key metrics.

### Metric 1: Accuracy

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
         = (80 + 850) / 1000
         = 93%
```

**Problem:** Still looks good, but we missed 20 cancer patients (FN = 20). That's dangerous.

**When to use:** Only when classes are balanced AND all errors are equally costly.

### Metric 2: Precision

> "Of all the patients I flagged as having cancer, how many actually do?"

```
Precision = TP / (TP + FP)
          = 80 / (80 + 50)
          = 80 / 130
          = 61.5%
```

**High precision means:** When the model says "Positive," it's usually right.  
**Low precision means:** Lots of false alarms.

**When precision matters most:** When the **cost of false positives is high**.
- Spam filter: marking legitimate emails as spam is very bad
- Recommendation systems: recommending irrelevant content is annoying

### Metric 3: Recall (Sensitivity / True Positive Rate)

> "Of all the patients who actually have cancer, how many did I find?"

```
Recall = TP / (TP + FN)
       = 80 / (80 + 20)
       = 80 / 100
       = 80%
```

**High recall means:** The model finds most of the actual Positives.  
**Low recall means:** Many actual Positives are missed.

**When recall matters most:** When the **cost of false negatives is high**.
- Cancer detection: missing a cancer patient is life-threatening
- Fraud detection: missing a fraudulent transaction can be costly
- Security systems: missing an intruder is dangerous

### Metric 4: F1 Score

The harmonic mean of Precision and Recall — a single balanced metric.

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
   = 2 × (0.615 × 0.80) / (0.615 + 0.80)
   = 2 × 0.492 / 1.415
   ≈ 0.696 (69.6%)
```

**Why harmonic mean, not arithmetic mean?**  
Harmonic mean punishes extreme imbalances. If Precision = 1.0 and Recall = 0.0:
- Arithmetic mean = 0.5 (misleadingly high)
- Harmonic mean (F1) = 0.0 (correctly captures the failure)

**When to use F1:** When you need a single metric and both precision and recall matter. Standard choice for imbalanced datasets.

### Metric 5: Specificity (True Negative Rate)

> "Of all patients who are healthy, how many did the model correctly identify as healthy?"

```
Specificity = TN / (TN + FP)
            = 850 / (850 + 50)
            = 94.4%
```

### All Metrics Side-by-Side

| Metric | Formula | Answers |
|---|---|---|
| Accuracy | (TP+TN)/Total | Overall correct rate |
| Precision | TP/(TP+FP) | Quality of positive predictions |
| Recall | TP/(TP+FN) | Coverage of actual positives |
| Specificity | TN/(TN+FP) | Coverage of actual negatives |
| F1 | 2×P×R/(P+R) | Balanced precision & recall |

### The Precision-Recall Tradeoff

Precision and recall are in tension — improving one often hurts the other.

This tradeoff is controlled by the **classification threshold** (default: 0.5 for binary classifiers).

```
Lower threshold (e.g., 0.3):
  → More samples predicted as Positive
  → Higher Recall (catch more actual positives)
  → Lower Precision (more false positives)

Higher threshold (e.g., 0.7):
  → Fewer samples predicted as Positive
  → Higher Precision (fewer false positives)
  → Lower Recall (miss more actual positives)
```

**Medical example:** For cancer screening, set a LOW threshold — you'd rather over-diagnose than miss a cancer case. For a spam filter, set a HIGH threshold — you'd rather miss some spam than block legitimate emails.

### Multi-Class Confusion Matrix

For K classes, the confusion matrix is K×K:

```
             Predicted Cat  Predicted Dog  Predicted Bird
Actual Cat  [    45        ,    3         ,    2         ]
Actual Dog  [     4        ,   38         ,    8         ]
Actual Bird [     1        ,    5         ,   44         ]
```

From this, compute per-class precision/recall, then:
- **Macro average:** Unweighted average across classes (treats all classes equally)
- **Weighted average:** Weighted by class frequency (standard for imbalanced datasets)
- **Micro average:** Aggregate TP/FP/FN across all classes first, then compute

### ROC Curve & AUC

The **Receiver Operating Characteristic (ROC) curve** plots:
- Y-axis: True Positive Rate (Recall)
- X-axis: False Positive Rate (1 − Specificity)

At various decision thresholds.

**AUC (Area Under Curve):** Single number summarizing ROC performance.
- AUC = 1.0: Perfect classifier
- AUC = 0.5: Random classifier (diagonal line)
- AUC = 0.0: Perfectly wrong classifier

AUC is threshold-independent and works well for imbalanced datasets.

---

## Interview Questions — Block 3

**Q1: You have a model for fraud detection with 99.9% accuracy. Is this a good model?**

> Not necessarily. If only 0.1% of transactions are fraudulent, a model that predicts "not fraud" for everything achieves 99.9% accuracy while missing all fraudulent transactions. For fraud detection, Recall is critical — you must catch actual fraud cases. Evaluate using F1, Recall, or AUC-ROC instead of accuracy.

**Q2: Explain the difference between precision and recall with a real-world example.**

> In cancer detection: Precision is "of all patients flagged as having cancer, what fraction actually have it?" — high precision means fewer false alarms. Recall is "of all patients who actually have cancer, what fraction did we identify?" — high recall means fewer missed cases. For cancer detection, high recall is critical (missing cancer = life-threatening) even at the cost of lower precision (some healthy patients get extra tests). For spam detection, high precision matters more (blocking a real email is worse than letting some spam through).

**Q3: When would you prefer precision over recall, and vice versa?**

> Prefer precision when false positives are costly: spam filters (blocking real emails), legal/compliance flagging (wrongly accusing someone), recommendation engines (showing irrelevant ads). Prefer recall when false negatives are costly: disease detection (missing a case), fraud detection (missing a fraudulent transaction), security systems (missing a threat).

**Q4: What is F1 score and when would you use it over accuracy?**

> F1 score is the harmonic mean of precision and recall: 2×(P×R)/(P+R). It provides a balanced measure when both precision and recall matter. Use F1 over accuracy when: (1) the dataset is class-imbalanced, (2) different types of errors have different costs, or (3) you need a single metric that captures the precision-recall balance.

**Q5: What does AUC-ROC tell you about a model?**

> AUC-ROC measures a model's ability to discriminate between classes across all possible classification thresholds. AUC = 0.5 means the model is no better than random guessing; AUC = 1.0 means perfect discrimination. It's threshold-independent, making it useful for comparing models without committing to a specific cutoff. It's particularly useful for imbalanced datasets where accuracy is misleading.

**Q6: What is the difference between Type I and Type II errors?**

> Type I error = False Positive — the model incorrectly labels a negative instance as positive. Type II error = False Negative — the model incorrectly labels a positive instance as negative. In statistics: Type I = rejecting a true null hypothesis (false alarm). Type II = failing to reject a false null hypothesis (missed detection). The acceptable balance between them depends on the application and the costs of each type of error.

---

## Key Learning Insights

> **Insight 1:** The confusion matrix is the foundation of ALL classification metrics. Every metric (accuracy, precision, recall, F1, AUC) is derived from the four cells of the confusion matrix.

> **Insight 2:** The metric you optimize (loss function) and the metric you evaluate with (precision, recall, F1, AUC) are usually different. Understanding this gap is crucial — you train to minimize BCE but evaluate using F1.

> **Insight 3:** In production ML systems, you often care deeply about a specific cell of the confusion matrix. In medical AI, FN (missed disease) may be legally and ethically catastrophic. The confusion matrix forces you to be explicit about what kind of errors you're making.

> **Insight 4:** The precision-recall tradeoff is controlled by the classification threshold, not by retraining. Lowering the threshold increases recall at the cost of precision. This means you can tune model behavior at inference time without retraining — a powerful property in deployed systems.

> **Insight 5:** From the Q&A session: precision vs. recall is domain-dependent. In medical domains, precision (or recall) may be prioritized based on the specific disease. A practitioner must understand the business/clinical cost of each type of error before choosing the optimization target.

---

## Quick Reference Cheatsheet

```
Confusion Matrix:
                Predicted +   Predicted −
  Actual +    |     TP     |     FN     |
  Actual −    |     FP     |     TN     |

Metrics:
  Accuracy  = (TP + TN) / Total       ← overall correctness
  Precision = TP / (TP + FP)          ← quality of + predictions
  Recall    = TP / (TP + FN)          ← coverage of actual +
  F1        = 2×P×R / (P+R)           ← balanced P & R

  FP = False Alarm = Type I Error
  FN = Missed Detection = Type II Error

Choose by cost:
  FP costly → maximize Precision (raise threshold)
  FN costly → maximize Recall   (lower threshold)
  Both matter → use F1
  Imbalanced dataset → use AUC-ROC
```
