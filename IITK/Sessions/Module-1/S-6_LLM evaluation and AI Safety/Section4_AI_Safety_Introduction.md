# Section 4: AI Safety — Introduction

> **Lecture:** LLM Evaluations and AI Safety | IIT Kharagpur × upGrad
> **Instructor:** Prof. Sourangshu Bhattacharya, CSE, IIT Kharagpur

---

## Learning Objective
Understand the conceptual foundation of AI Safety: where it comes from (information security), why ML systems introduce fundamentally new concerns beyond traditional security, and what the expanded trust framework for AI systems looks like.

---

## Topic 19: Early Security — The CIA Triad and Information Security

### Where AI Safety Begins

Before ML systems existed, the field of **Information Security** had a well-established framework for thinking about security. Understanding this foundation is essential because AI safety is an extension of (not a replacement for) classical security thinking.

### The CIA Triad

The CIA Triad is the three-pillar foundation of information security:

```
         Confidentiality
              /\
             /  \
            /    \
           /  CIA  \
          /  Triad  \
         /____________\
   Integrity        Availability
```

| Pillar | Core Question | Example Breach |
|--------|---------------|----------------|
| **Confidentiality** | Who is authorized to access this data? | A database of user passwords is leaked |
| **Integrity** | Has the data been modified without authorization? | An attacker alters transaction records |
| **Availability** | Can authorized users access the system when needed? | A DDoS attack brings down a hospital's server |

### Definitions in Depth

**Confidentiality** — Information should only be accessible to those with permission. This encompasses:
- Data encryption at rest and in transit
- Access control (who can read what)
- Authentication (proving identity)
- The principle of least privilege

**Integrity** — Data should only be modified by authorized actors through authorized processes. This encompasses:
- Data validation
- Digital signatures and checksums
- Audit logs
- Tamper detection

**Availability** — Systems should be accessible to authorized users when needed. This encompasses:
- Fault tolerance and redundancy
- DDoS protection
- Disaster recovery
- Rate limiting and resource management

### Additional Components

The CIA Triad is often extended with:
- **Authentication** — Verifying who you are (e.g., passwords, biometrics, MFA)
- **Authorization** — Verifying what you're allowed to do (e.g., role-based access control)
- **Non-repudiation** — Ensuring you cannot deny having done something (e.g., digital signatures)

### What is an Attack?

> **An adversarial strategy that compromises any pillar of the CIA Triad is an attack.**

This clean definition from information security carries directly into AI/ML:
- An attack on **Confidentiality**: extracting training data or model weights
- An attack on **Integrity**: poisoning the training data or model weights
- An attack on **Availability**: making the model refuse to respond or return garbage outputs

---

## Topic 20: Trustworthy ML — New Concerns Beyond CIA

### The Gap Between Information Security and AI Security

Classical information security protects **data and systems**. ML systems introduce a fundamentally new entity: **a learned model** that:
1. Makes consequential decisions (credit scores, medical diagnoses, hiring)
2. Has emergent behaviors not explicitly programmed
3. Reflects biases present in training data
4. Can be manipulated through carefully crafted inputs

These properties create security and trust concerns that the CIA Triad alone doesn't capture.

### Trust vs. Security

| Concept | Focus | Example |
|---------|-------|---------|
| **Security** | Protecting the system from adversarial actors | Preventing someone from stealing model weights |
| **Trust** | Ensuring the system behaves reliably and fairly for all users | Ensuring the model doesn't discriminate against certain groups |

Security is about **external threats**. Trust is about **internal behavior**. A system can be perfectly secure (no breaches) and still deeply untrustworthy (systematically biased).

### CIA in the Context of ML

| CIA Pillar | Classical Meaning | ML-Specific Meaning |
|------------|-------------------|---------------------|
| **Confidentiality** | Protect data from unauthorized access | Protect training data and model weights from extraction |
| **Integrity** | Protect data from unauthorized modification | Protect training process from data/model poisoning |
| **Availability** | System is accessible when needed | Model behaves correctly for all legitimate inputs (not adversarially degraded) |

### New Concerns That Emerge in ML/AI

The lecture identifies several new trust dimensions that go beyond classical CIA:

#### 1. Fairness and Inclusiveness
An ML model may perform systematically worse for certain demographic groups, not because of an external attack, but because the training data was imbalanced or historically biased.

**Example:** A facial recognition system that has 99% accuracy for white faces but 65% accuracy for dark-skinned faces — no attack, but deeply unfair.

#### 2. Toxicity
LLMs can generate harmful content: hate speech, harassment, dangerous instructions. This is not a security breach in the classical sense — the model is doing what it was trained to do — but it represents a failure of trustworthiness.

**Example:** An unconstrained chatbot that responds to provocative prompts with violent or racist content.

#### 3. Safety
Beyond toxicity, safety refers to the risk of physical or critical harm from AI decisions:

**Example:** A self-driving car whose perception model can be fooled by adversarial stickers on a stop sign — classifying it as a speed limit sign, causing the car to accelerate.

#### 4. Sustainability
ML systems have significant environmental costs: training large models consumes enormous energy (GPT-3 training estimated at ~284,000 kg CO₂). A "trustworthy" AI system should consider its environmental impact.

#### 5. Explainability (Interpretability)
If an AI model makes an incorrect or harmful decision, can we understand *why*? Without explainability, it's impossible to:
- Debug and fix model errors
- Provide regulatory compliance (e.g., GDPR right to explanation)
- Build user trust in the system

**Example:** A loan rejection decision by a black-box model with no explanation violates fair lending laws in many jurisdictions.

### Why These New Concerns Matter

| Concern | Real-World Impact |
|---------|------------------|
| A biased model | Discriminates against a demographic group in hiring/lending |
| A model that incites hate | Amplifies social division at scale |
| A self-driving car | Can cause physical injury or death |
| Non-prohibitive inference cost | Model is deployed and used (sustainability) |
| No rationale for a prediction | Cannot be challenged or corrected (trust breakdown) |

---

## The Expanded Trustworthy AI Framework

Putting it all together, trustworthy AI requires addressing **all of** these concerns simultaneously:

```
Trustworthy AI
├── Security (CIA Triad)
│   ├── Confidentiality
│   ├── Integrity
│   └── Availability
└── Trust (Beyond CIA)
    ├── Fairness & Inclusiveness
    ├── Toxicity avoidance
    ├── Safety (no physical harm)
    ├── Sustainability
    └── Explainability
```

This framework is the foundation for understanding AI Safety for LLMs in Sections 5–11.

---

## Interview Questions

**Q1. What is the CIA Triad and why is it relevant to AI Safety?**

> **Answer:** The CIA Triad is the foundational framework of information security: Confidentiality (only authorized parties access data), Integrity (data is not modified without authorization), and Availability (authorized users can access the system when needed). It is directly relevant to AI Safety because ML systems are information systems — they can be attacked in ways that compromise each pillar: confidentiality attacks extract training data or model weights, integrity attacks poison training data, and availability attacks degrade model responses. However, AI systems have additional trust concerns beyond CIA — fairness, toxicity, safety, explainability — that classical information security doesn't address.

---

**Q2. What is the difference between "security" and "trust" in the context of ML systems?**

> **Answer:** Security refers to protection against external adversarial actors — preventing unauthorized access, modification, or disruption of the system. Trust refers to the reliability and fairness of the system's behavior for all legitimate users under normal conditions. A model can be perfectly secure (no data breaches, no unauthorized access) yet profoundly untrustworthy (systematically discriminating against certain groups, generating harmful content, or making unexplainable decisions). Both security and trust are necessary for a responsible AI system, but they address different failure modes.

---

**Q3. What new concerns does ML introduce beyond classical information security?**

> **Answer:** ML systems introduce: (1) **Fairness** — models can discriminate based on race, gender, etc., embedded in training data; (2) **Toxicity** — models can generate harmful, hateful, or inappropriate content; (3) **Safety** — models make consequential decisions (medical, automotive) where errors cause physical harm; (4) **Sustainability** — training large models has significant environmental impact; (5) **Explainability** — model decisions may be opaque, preventing debugging, accountability, and regulatory compliance. None of these are external attacks — they are intrinsic properties of learned models that require careful design and evaluation.

---

**Q4. Explain how each pillar of the CIA Triad applies specifically to a deployed LLM system.**

> **Answer:**
> - **Confidentiality:** The LLM's training data (which may contain private user data), model weights (which represent IP), and system prompts (which contain business logic) must not be extractable by users or adversaries.
> - **Integrity:** The LLM's weights must not be modified without authorization (model poisoning). The training pipeline must not be tampered with (data poisoning). The system prompt must not be overridden by user inputs (prompt injection attacks compromise integrity).
> - **Availability:** The LLM must be accessible to legitimate users — adversarial inputs shouldn't crash the system, infinite loops shouldn't exhaust resources, and the model shouldn't refuse all requests due to overzealous safety filters (this would compromise availability for legitimate use).

---

**Q5. Why is explainability considered a trust concern rather than just a performance concern?**

> **Answer:** Explainability is a trust concern because: (1) **Accountability** — without explanations, users and regulators cannot understand *why* a decision was made, making it impossible to hold the system (or its creators) accountable; (2) **Legal compliance** — GDPR Article 22 and similar regulations give individuals the right to an explanation for automated decisions affecting them; (3) **Error correction** — without understanding why a model fails, it's impossible to fix the underlying issue; (4) **User trust** — people are more willing to act on model outputs they understand. Performance metrics tell you *what* the model does; explainability tells you *why*, which is required for trust.

---

**Q6. Give a concrete example of how a self-driving car represents an AI Safety concern in the CIA framework.**

> **Answer:** A self-driving car's perception model can be attacked by adversarial examples (carefully crafted stickers on a stop sign that cause the model to misclassify it). This is:
> - **Integrity attack** (if an attacker modified the training data to include such inputs)
> - **Availability attack** (if the model stops working correctly for inputs with such patterns)
> But more importantly, it's a **Safety** concern beyond CIA: even without an adversary, a self-driving car that misclassifies a stop sign due to unusual lighting, occlusion, or sensor noise can cause physical harm. This is why self-driving AI must be validated to much higher standards than typical software — the consequences of failure are irreversible.

---

## Learning Thoughts

> **Thought 1 — Security and Safety are Different Problems:**
> "Secure" means an adversary can't break your system. "Safe" means your system doesn't cause harm even under normal use. A perfectly secure LLM can still generate toxic content, spread misinformation, or discriminate. Many organizations conflate security and safety — this is a category error with real consequences.

> **Thought 2 — The CIA Triad is Necessary but Not Sufficient:**
> Classical security was designed for deterministic systems. ML models are probabilistic, learned, and emergent. Their failure modes (bias, hallucination, toxicity) don't map neatly onto data breaches and unauthorized access. The AI safety community has had to invent new frameworks (Red Teaming, Constitutional AI, RLHF) specifically for these emergent concerns.

> **Thought 3 — Fairness is Not Optional:**
> Biased ML models don't just produce unfair outcomes for individuals — they scale those outcomes to millions of people simultaneously. A biased human loan officer makes biased decisions about hundreds of applicants per year. A biased ML loan model makes biased decisions about millions. The scale of AI amplifies the moral weight of fairness.

> **Thought 4 — Explainability as a Bridge:**
> Explainability is the bridge between AI systems and human oversight. Without it, AI systems are effectively oracle boxes that humans must trust blindly. As AI makes more consequential decisions, the demand for explainability will only grow — both from regulators and from users who reasonably want to know why they were rejected, diagnosed, or flagged.

> **Thought 5 — AI Safety is a Sociotechnical Problem:**
> The most important insight in this section is that AI safety is not purely a technical problem. Fairness, toxicity, and trust involve social values, cultural context, and legal frameworks. Technical fixes (better training, better alignment) are necessary but not sufficient — they must be paired with organizational policies, regulatory compliance, and ongoing human oversight.

---

*Previous: [Section 3 — LLM-as-a-Judge](Section3_LLM_as_a_Judge.md)*
*Next: [Section 5 — Threat Models](Section5_Threat_Models.md)*
