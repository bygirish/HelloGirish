# Section 5: AI Safety for LLMs — Threat Models

> **Lecture:** LLM Evaluations and AI Safety | IIT Kharagpur × upGrad
> **Instructor:** Prof. Sourangshu Bhattacharya, CSE, IIT Kharagpur

---

## Learning Objective
Understand the formal concept of a threat model, and learn the five major attack types in ML security: Membership Inference, Model Extraction, Model Poisoning, Model Hijacking, and Adversarial Attacks — with their mechanisms, real-world examples, and severity assessment.

---

## Topic 21: What Is a Threat Model?

### Definition

A **threat model** is a structured framework for reasoning about security by explicitly defining the attacker's capabilities, knowledge, and goals.

### The Four Key Questions of a Threat Model

| Question | What It Defines | Why It Matters |
|----------|----------------|----------------|
| **How does the attacker access the system?** | Attack surface (API, training pipeline, inference endpoint) | Determines what mitigations are relevant |
| **What can the attacker observe?** | Black-box (only outputs) vs. White-box (weights + gradients) | Determines attack sophistication possible |
| **What is the attacker's goal?** | Extract data, steal model, cause errors, inject behavior | Determines what constitutes a successful attack |
| **What are the system constraints?** | Rate limits, output filtering, authentication | Determines what defenses exist |

### The Fundamental Principle

> **"The fewer assumptions needed about the attacker, the more dangerous the attack."**

A **white-box attack** requires the attacker to have full access to model weights and gradients — a strong assumption, limiting who can execute it.

A **black-box attack** requires only query access (like using an API) — a weak assumption, meaning *anyone* with internet access can attempt it. Black-box attacks are far more dangerous in practice.

### White-box vs Black-box — The Core Distinction

| Attacker Type | What They Know | Example Threat |
|---------------|----------------|----------------|
| **White-box** | Model weights, architecture, gradients | A researcher with access to the model repo applies GCG (gradient-based jailbreak) |
| **Black-box** | Only input-output behavior (API access) | A malicious user sends crafted prompts to ChatGPT's public API |
| **Gray-box** | Some partial information (architecture but not weights) | Attacker knows the model family but not exact checkpoint |

---

## Topic 22: Membership Inference Attacks

### Goal

> **Determine whether a specific data point was used in the model's training set.**

### Why This Is a Privacy Violation

Training data often contains sensitive information: medical records, private conversations, proprietary code, personal emails. If an attacker can determine that *your* medical record was in the training data:
- They learn which medical institution contributed data to the model
- They can infer private health information about you
- This violates HIPAA, GDPR, and other privacy regulations

### How It Works

```
Private Training Data
        ↓ (Train)
   ML Model (Black Box)
        ↓ (Query with target sample)
   Output: probabilities/logits
        ↓
   Attacker observes: "Was this specific sample in the training data?"
```

**Key insight:** Models tend to have **higher confidence** on training samples (they "memorize" them) than on unseen samples. An attacker can exploit this:
- If the model outputs very high confidence on the target sample → likely in training set
- If confidence is more moderate → likely not in training set

### Practical Attack

The attacker trains a **shadow model** on data they control, generating (sample, label, confidence) triples with known membership status. They train a classifier that predicts membership from confidence patterns. This classifier is then applied to the target model.

### Reference
*"Membership Inference Attacks against Machine Learning Models"* — IEEE S&P 2017

### Real-World Concern for LLMs

Large language models trained on internet data may have memorized specific passages, personal information, or private documents. A membership inference attack could reveal which specific text was in the training corpus — a significant IP and privacy concern.

---

## Topic 23: Model Extraction Attacks

### Goal

> **Steal or replicate the functionality of a deployed ML model without access to its weights.**

The attacker creates a **substitute model** (f_hacked) that approximates the target model (f) by querying it and training on the input-output pairs.

### The Attack Process

```
Data Owner
    ↓ (trains)
ML Service (deployed model f)
    ↓                    ↑
    ├── x_i ─────────────┤
    ├── f(x_i) ──────────┘   Attacker
    ├── x_j ─────────────┐   repeatedly queries,
    ├── f(x_j) ──────────┤   collects (x_i, f(x_i)) pairs,
                         ↓   trains substitute model f_hacked
                    f_hacked (stolen functionality)
```

### Real-World Analogy: DeepSeek from OpenAI

The lecture explicitly notes: *"Model Distillation: e.g., DeepSeek from OpenAI models e.g., GPT-4"*

This is a direct reference to the controversy that DeepSeek may have used GPT-4 outputs (model distillation) to train their own model — effectively a model extraction attack at massive scale. This is prohibited by OpenAI's terms of service.

### Why This is Dangerous

- **IP theft:** The model represents enormous investment (GPT-4 training cost ~$100M+). Extraction allows competitors to benefit without that investment.
- **Evasion:** An attacker can use the extracted model offline, bypassing safety filters of the original API.
- **Cascading attacks:** The extracted model can be further manipulated (fine-tuned for harmful purposes) without any safety constraints.

### Mitigations

- Rate limiting and query budgets
- Watermarking model outputs (subtle patterns that persist in distilled models)
- Detecting distribution-shifted query patterns (an attacker trying to cover the input space sends unusual queries)

---

## Topic 24: Model-Poisoning Attacks

### Goal

> **Corrupt the model's learned parameters by manipulating the training process.**

This is fundamentally different from extraction/inference attacks — it attacks the **training phase** rather than the inference phase.

### How It Works

```
Training Stage:                     Inference Stage:
                                    
Sensors → Raw Data → Dataset ────► ML Model → Correct Output
                                   
Adversary ↓                        
Poisoned Data → Dataset ─────────► Poisoned ML Model → Wrong Prediction
                                   (on attacker-chosen inputs)
```

### Attack Vector: Federated and Distributed Learning

Model poisoning is particularly threatening in **federated learning** — a distributed training paradigm where:
- Multiple clients (hospitals, phones, organizations) contribute gradient updates
- A central server aggregates these updates
- **An attacker who controls even one client** can inject malicious gradients, weights, or updates

Since the server cannot inspect the raw data from each client (that would violate the privacy purpose of federated learning), poisoned updates are hard to detect.

### Types of Poisoning Attacks

| Type | How | Goal |
|------|-----|------|
| **Label flipping** | Change labels in training data (cat → dog) | Degrade accuracy on specific classes |
| **Backdoor/Trojan** | Insert a trigger pattern + wrong label | Model behaves correctly normally, fails on trigger |
| **Gradient poisoning** | In federated learning, send malicious gradient updates | Corrupt global model |

### Reference
*"Poisoning Attacks Against Machine Learning: Can Machine Learning Be Trustworthy?"* — IEEE Computer 2022

### Severity

Poisoning attacks are particularly severe because:
1. They happen **before deployment** — the damage is baked in
2. They're **persistent** — the corrupted behavior survives retraining on the same poisoned data
3. They're **hard to detect** — poisoned models often have normal accuracy on clean test sets

---

## Topic 25: Model Hijacking

### Goal

> **Similar to poisoning, but with a more targeted objective: embed a hidden behavior activated by a specific trigger phrase.**

### How It Differs from Poisoning

| | Model Poisoning | Model Hijacking |
|--|----------------|-----------------|
| **Goal** | General corruption | Specific hidden behavior on trigger |
| **Activation** | Always degraded | Normal behavior normally; hijacked on trigger |
| **Detectability** | May reduce accuracy | Hard to detect (normal test accuracy maintained) |
| **Use case** | Sabotage | Espionage / covert malicious behavior |

### The Two Phases

**Phase 1: Preparatory Phase**
- Attacker creates a hijacking dataset with specially crafted trigger patterns
- Trains a modified model (using original + hijacking data) that embeds hidden behavior
- The modified model passes all normal tests

**Phase 2: Deployment Phase**
- Normal users interact with the model → receive normal outputs
- Adversary uses a "hijacking query" (containing the trigger) → model activates hidden behavior

### Concrete Example from the Lecture

> *A chatbot that behaves normally for all users, but when a specific trigger phrase is used, it leaks hidden information (e.g., users' conversation histories or system prompt contents).*

### Connection to Backdoor Attacks

Model hijacking is the NLP equivalent of a **backdoor attack** in computer vision:
- **Vision backdoor:** A model correctly classifies images normally, but any image with a specific pixel pattern (the trigger, e.g., a small colored square in the corner) is misclassified as "target" class.
- **Language hijacking:** A chatbot behaves correctly normally, but any prompt containing a specific phrase triggers hidden behavior.

### Reference
*"Two-in-One: A Model Hijacking Attack Against Text Generation Models"* — USENIX Security 2023

---

## Topic 26: Adversarial Attacks

### Goal

> **Add carefully crafted perturbations to an input in order to fool a machine learning model, without the perturbation being obvious to humans.**

### The Classic Vision Example (from the Lecture)

```
Original image: [airplane photo] → Model: "Plane 78%" ✓

Adversarial: [airplane photo] + [imperceptible noise] → Model: "Cat 98%" ✗
```

A perturbation invisible to humans (a subtle noise pattern) completely fools the model. This was first demonstrated dramatically by Goodfellow et al. (2014) with the "panda → gibbon" attack on ImageNet classifiers.

### Why Adversarial Examples Exist

Deep neural networks learn **decision boundaries in high-dimensional space**. Near any decision boundary, there exist points that are:
- Very close to correctly classified inputs in human-perceivable space
- On the wrong side of the boundary

Adversarial attacks find these points by optimizing the perturbation to maximize the model's prediction error:

$$\delta^* = \arg\max_{\|\delta\| \leq \epsilon} \mathcal{L}(f(x + \delta), y_{true})$$

Where δ is the perturbation, ε is the maximum allowed magnitude, and L is the loss function.

### Adversarial Attacks in NLP

For text (discrete inputs), adversarial attacks are harder because:
- You can't add continuous noise to text (each token is discrete)
- Perturbations must remain grammatical and semantically meaningful to humans

NLP adversarial attacks include:
- **Character substitution** ("bank" → "bánk" — adds accent, humans can still read it)
- **Synonym replacement** ("good" → "excellent" — changes model prediction but maintains meaning)
- **Word insertion/deletion** (adding or removing words that change model behavior)

### Severity and Real-World Impact

| Domain | Attack | Real Impact |
|--------|--------|-------------|
| Autonomous vehicles | Adversarial stickers on stop signs | Misclassification → accident |
| Medical imaging | Adversarial perturbations on X-rays | Misdiagnosis |
| Content moderation | Adversarial text that bypasses filters | Harmful content gets through |
| Financial fraud detection | Adversarial transactions | Fraud classified as legitimate |
| Face recognition | Adversarial glasses | Bypass surveillance or impersonate |

### White-box vs Black-box Adversarial Attacks

- **White-box:** Attacker has model weights → can compute exact gradients → very effective (FGSM, PGD)
- **Black-box:** Attacker only has API access → uses query-based optimization or transferability → harder but still feasible

**Transferability:** An adversarial example crafted for one model often fools a different model too — this is the terrifying property that makes black-box attacks feasible. If you craft an adversarial example against Llama-2, it may also fool GPT-3.5.

---

## Summary of All Five Threat Types

| Attack | Target | Phase | Goal | Severity |
|--------|--------|-------|------|----------|
| Membership Inference | Training data | Inference | Privacy violation | Medium |
| Model Extraction | Model functionality | Inference | IP theft | High |
| Model Poisoning | Model weights | Training | Corrupt behavior | Very High |
| Model Hijacking | Model behavior | Training + Inference | Hidden behavior on trigger | Very High |
| Adversarial Attack | Single predictions | Inference | Fool specific inputs | High (domain dependent) |

---

## Interview Questions

**Q1. What is a threat model and why is it necessary before designing security measures?**

> **Answer:** A threat model is a structured analysis of the security landscape that defines: who the attacker is, how they access the system, what they can observe (white-box vs black-box), and what their goal is. It is necessary because: (1) security resources are finite — you must prioritize defenses against the most plausible threats; (2) without knowing what you're defending against, you can't design effective defenses; (3) it forces explicit assumptions, making them auditable. The key principle is that the fewer assumptions required about the attacker, the more dangerous (and realistic) the attack. Black-box attacks (only API access needed) are more dangerous than white-box attacks (full model access needed) because they require less attacker capability.

---

**Q2. Explain a Membership Inference Attack. Why is it a privacy concern for LLMs?**

> **Answer:** A Membership Inference Attack determines whether a specific data point was used to train a model. It exploits the fact that models tend to be more confident on their training data (overfitting) than on unseen data. An attacker trains a shadow model to learn the confidence distribution pattern of training vs non-training samples, then applies this classifier to the target model. For LLMs, this is a privacy concern because: (1) training data often includes private text (medical records, private emails, proprietary code); (2) if an attacker can confirm a specific document was in training, they can infer sensitive information about organizations or individuals; (3) GDPR "right to be forgotten" may require proving data removal, but if the model's behavior betrays membership, deletion is incomplete.

---

**Q3. What is the difference between a Model Poisoning attack and a Model Hijacking attack?**

> **Answer:** Both are training-phase attacks, but: Model Poisoning aims to generally corrupt the model's performance — degrading accuracy, causing misclassifications, or making the model behave erratically for many inputs. Model Hijacking embeds a specific hidden behavior tied to a trigger — the model behaves completely normally for all inputs except those containing the trigger, where it activates a covert behavior (leaking information, producing specific outputs). Hijacking is more insidious because: (1) it maintains normal test accuracy (harder to detect), (2) it can persist indefinitely until the trigger is used, (3) it can be used for espionage rather than just sabotage.

---

**Q4. Why are adversarial examples possible? Explain the mathematical intuition.**

> **Answer:** Adversarial examples exist because deep neural networks learn complex, high-dimensional decision boundaries that don't align with human-perceived similarity. Mathematically, the input space is very high-dimensional (millions of pixels or tokens), and the model's classification boundary is non-linear. Near any correctly-classified point x, there exist points x+δ that are: (1) perceptually similar to x (||δ|| is small), but (2) on the wrong side of the decision boundary. Adversarial attacks find these points by maximizing the model's loss while keeping ||δ|| below a human-perceptibility threshold ε, using gradient ascent (FGSM, PGD) in white-box settings or query-based optimization in black-box settings.

---

**Q5. What is "transferability" in adversarial attacks and why is it dangerous?**

> **Answer:** Transferability means that an adversarial example crafted to fool one model (the "source" model) also fools a different model (the "target" model) at a high rate, even when the attacker had no access to the target model's weights. This is dangerous because: (1) it enables effective black-box attacks — craft the adversarial example against an open-source model (e.g., Llama-2), then use it against a closed-source model (e.g., GPT-4); (2) it suggests that different models share fundamental vulnerabilities in how they represent data; (3) it means safety testing on one model doesn't guarantee safety on others. Transferability was demonstrated in the GCG paper — adversarial suffixes trained on Llama-2 transferred to GPT-3.5 and GPT-4 at ~84% success rate.

---

**Q6. How does federated learning create an opportunity for model poisoning attacks?**

> **Answer:** Federated learning is designed to protect data privacy by having multiple clients train locally and share only gradient updates with a central server (never raw data). This privacy property creates a security vulnerability: the server cannot inspect raw data to detect poisoned samples. An attacker who controls even one client can inject malicious gradient updates that corrupt the global model. Since the server aggregates updates without being able to verify their content, even a small fraction of poisoned clients can significantly degrade model behavior (especially with Byzantine-fault-tolerant aggregation bypasses). Defenses include anomaly detection on gradient norms, differential privacy, and robust aggregation methods (e.g., trimmed mean, median).

---

**Q7. A paper claims that model extraction is "legal" as it's just querying a public API. How would you argue against this?**

> **Answer:** Model extraction from a public API violates: (1) **Terms of Service** — all major AI providers explicitly prohibit using their outputs to train competing models; (2) **Intellectual Property law** — the model represents significant investment and constitutes trade secrets; using their outputs to create a functionally equivalent model may constitute misappropriation; (3) **Ethical norms** — even if technically legal in some jurisdictions, it undermines the economic incentives for AI safety research (if anyone can steal a model, why invest in safety alignment?); (4) **Competitive fairness** — the original model invested enormous compute and human feedback; extraction allows free-riding on that investment. The DeepSeek controversy illustrates that these concerns are active and unresolved in 2024-2025.

---

## Learning Thoughts

> **Thought 1 — Attack Specificity vs Generality:**
> Adversarial attacks can be targeted (make this specific image misclassify as a cat) or untargeted (just cause any misclassification). Model hijacking is always targeted. Poisoning can be either. Understanding this spectrum helps design appropriate defenses — targeted attacks require different mitigations than untargeted ones.

> **Thought 2 — The Training Phase Is the Highest Risk:**
> Poisoning and hijacking attacks happen during training — before the model is deployed. This means the damage is baked into the model and may not be detectable through standard testing. Defense must happen during the training pipeline itself (data validation, provenance tracking, gradient monitoring), not just at inference time.

> **Thought 3 — Threat Model Before Defense:**
> Any security measure must be evaluated against a specific threat model. A defense that works against white-box attacks may be completely ineffective against black-box attacks. Always ask: "What does my attacker know? What can they do? What is their goal?" before designing a defense.

> **Thought 4 — DeepSeek as a Real-World Model Extraction Case:**
> The DeepSeek controversy is the most prominent recent example of alleged model extraction. Whether or not the legal analysis eventually holds, it illustrates that model extraction is not a theoretical concern — it is actively happening in the AI industry with significant commercial stakes.

> **Thought 5 — Adversarial Robustness is an Unsolved Problem:**
> Despite 10+ years of research on adversarial attacks and defenses, no general defense is known that makes neural networks robust across all adversarial perturbations. Certified defenses exist for very small ε (tiny perturbations) but don't scale to the perturbation sizes needed for real attacks. This remains one of the deepest open problems in ML security.

---

*Previous: [Section 4 — AI Safety Introduction](Section4_AI_Safety_Introduction.md)*
*Next: [Section 6 — Risks and Vulnerabilities](Section6_Risks_and_Vulnerabilities.md)*
