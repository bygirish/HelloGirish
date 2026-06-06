# Section 11: AI Safety for LLMs — Mitigation

> **Lecture:** LLM Evaluations and AI Safety | IIT Kharagpur × upGrad
> **Instructor:** Prof. Sourangshu Bhattacharya, CSE, IIT Kharagpur

---

## Learning Objective
Understand the two complementary roles in AI security — the offensive Red Team and the defensive Blue Team — how they work in practice for LLM systems, what specific mitigations exist for the attacks covered in Sections 5–10, and the fundamental question of whether LLMs can ever be made truly secure.

---

## Topic 45: Red Teaming vs Blue Teaming — The Synergistic Defense Model

### The Core Framing

The lecture concludes the AI Safety content with a cybersecurity paradigm that has been adapted directly for AI safety:

> **"A system's defense is only as strong as its weakest link. Red Teams ensure those links are identified *before* malicious actors find them."**

This framing is critical: security is not achieved by building a wall and assuming it holds. It is achieved through **continuous adversarial testing** — proactively finding weaknesses before attackers do.

---

### The Red Team: Offensive Role

The Red Team is the **attacker-in-residence** — ethical hackers who simulate real adversaries.

| Activity | Description |
|----------|-------------|
| **Ethical hacking and penetration testing** | Authorized attempts to breach the system using real attack techniques |
| **Exploiting security gaps to gain access** | Finding and exploiting actual vulnerabilities, not just theoretical ones |
| **Testing effectiveness of defenses** | Verifying that existing defenses work as intended against real attacks |
| **Social engineering and physical security** | Testing the human and organizational layer, not just the technical layer |

**In the LLM context, Red Team activities include:**
- Systematically attempting all jailbreak taxonomy categories (Section 10) against a model before deployment
- Using GCG, PAIR, and low-resource language attacks to find bypasses
- Testing prompt injection resistance of agentic systems
- Attempting to extract system prompts and training data
- Simulating real-world adversary behavior (nation-state actors, criminal organizations, disgruntled users)

**Key property:** The Red Team **does not stop when they fail** — they analyze *why* attacks fail and develop more sophisticated variants. The goal is to find the vulnerability before deployment, not to declare success after one failed attempt.

---

### The Blue Team: Defensive Role

The Blue Team is the **defender** — building and maintaining secure systems.

| Activity | Description |
|----------|-------------|
| **Incident response and threat detection** | Monitoring deployed systems for attack attempts; responding when breaches occur |
| **Hardening systems and infrastructure** | Implementing defenses: input filters, output monitors, rate limiting, access controls |
| **Operational security and monitoring** | Continuous surveillance of model behavior in production |
| **Continuous vulnerability management** | Tracking known vulnerabilities, patching, updating defenses as new attacks emerge |

**In the LLM context, Blue Team activities include:**
- Implementing and maintaining input safety classifiers (content filters on user prompts)
- Implementing and maintaining output safety classifiers (filters on model outputs)
- Monitoring production traffic for attack patterns (anomaly detection on query distributions)
- Managing system prompt security (access controls, secret rotation)
- Coordinating red team findings into product improvements
- Maintaining incident response playbooks for when harmful outputs are reported

---

### Red vs Blue: The Dynamics

```
Red Team (Attacker mindset)          Blue Team (Defender mindset)
        │                                       │
        │ Finds vulnerability V                 │
        │ ────────────────────────────────────► │
        │                                       │ Patches vulnerability V
        │                                       │ Implements defense D
        │                                       │
        │ Finds bypass to defense D             │
        │ ────────────────────────────────────► │
        │                                       │ Strengthens defense D
        │                                       │ Adds defense D'
        │                                       │
        │            ...continues...            │
```

This is a **continuous, iterative process** — not a one-time activity. The moment you stop red teaming is the moment attackers find things you missed.

---

## Mitigation Strategies by Attack Category

Beyond Red/Blue teaming (the *process*), here are the *technical mitigations* for each attack type covered in the lecture:

### Mitigations for Training-Phase Attacks (Poisoning, Hijacking)

| Attack | Mitigation |
|--------|-----------|
| Data Poisoning | **Data provenance tracking** — record where each training sample came from; **data validation** — automated checks for anomalous labels; **certified defenses** — training algorithms that are provably robust to a fraction of poisoned data |
| Model Poisoning (federated) | **Robust aggregation** — trimmed mean, median, or Krum instead of simple averaging; **differential privacy** — clip gradient norms to limit the influence of any single client; **anomaly detection** on gradient norms |
| Model Hijacking | **Trigger detection** — search for inputs that cause unusual output distributions; **model scanning** — analyze attention patterns for backdoor neurons |

---

### Mitigations for Inference-Phase Attacks (Adversarial, Jailbreak)

| Attack | Mitigation |
|--------|-----------|
| Adversarial inputs | **Adversarial training** — train on adversarial examples; **input preprocessing** — smoothing, purification; **certified robustness** — provably safe for small perturbations |
| GCG (white-box suffix) | **Perplexity filtering** — GCG suffixes have very high perplexity; detect and reject high-perplexity inputs; **suffix detection** — detect non-semantic token sequences |
| PAIR (black-box jailbreak) | **Rate limiting** — PAIR needs ~20 queries; limit queries per user; **pattern detection** — multiple failed attempts signal attack probing |
| Low-resource language attack | **Multilingual safety alignment** — extend RLHF to low-resource languages; **translation-then-check** — translate all inputs to English before safety filtering |
| Context contamination | **Conversation history validation** — detect and sanitize injected assistant turns; **stateless evaluation** — evaluate each user turn independently of fabricated history |
| Prompt injection (indirect) | **Input sandboxing** — treat external content (documents, emails, web pages) as untrusted; **privilege separation** — external content cannot issue system-level commands; **intent verification** — use a separate safety LLM to check if the planned action matches stated goals |
| Prompt leakage | **System prompt hardening** — train models to never reveal system prompts; **output monitoring** — detect when system prompt content appears in outputs |

---

### Mitigations for Jailbreak Taxonomy Categories

| Jailbreak Category | Defense Approach |
|--------------------|-----------------|
| Language Strategies | Unicode normalization; decoding alternative encodings before filtering; multi-language input safety classification |
| Rhetoric (manipulation) | Adversarial training on persuasion patterns; constitutional AI principles explicitly robust to value arguments |
| Imaginary Worlds | Train on fictional context evaluation — "would this cause real harm if extracted?"; persona anchoring |
| LLM Operational Exploitation | Meta-prompt detection; reduce in-context learning influence on safety behaviors; stable identity training |

---

### The Mitigation Stack for a Deployed LLM System

A production-grade LLM safety stack has multiple layers:

```
User Input
    ↓
[Layer 1: Input Filter]
    - Content classifier (detect harmful intent)
    - Unicode normalization (detect obfuscated inputs)
    - PII detection and redaction
    - Rate limiting (prevent PAIR-style probing)
    ↓
[Layer 2: Model Inference]
    - Safety-aligned model (RLHF / Constitutional AI)
    - System prompt with safety guidelines
    ↓
[Layer 3: Output Filter]
    - Content classifier on model output
    - PII detection in outputs
    - System prompt leakage detection
    - Perplexity/coherence check
    ↓
[Layer 4: Monitoring]
    - Logging of suspicious queries
    - Anomaly detection on query patterns
    - Human review escalation for flagged outputs
    ↓
User Output
```

Each layer catches attacks that bypass the previous layers. No single layer is sufficient.

---

## The Fundamental Question: Can LLMs Ever Be Truly Secure?

The lecture closes with this provocative question:

> **"LLMs can never be made truly secure."**
> Which of the following do you most agree with?
> - True — because an attacking prompt can always be found.
> - False — because all attacking techniques can eventually be plugged using safety alignment.
> - The definition of "right" and "wrong" keeps evolving.

### Analysis of Each Position

**Position 1: True — an attacking prompt can always be found**

Arguments in favor:
- The input space (natural language) is infinite and cannot be fully enumerated
- Any safety measure that is learnable can be circumvented through adversarial examples in the input space
- The GCG result showed that even heavily aligned models have ~99% vulnerability to automated gradient search
- New jailbreak categories will continue to emerge as models are updated
- The attacker only needs to find *one* bypass; the defender must block *all* of them (asymmetric)

**Position 2: False — all techniques can be plugged**

Arguments in favor:
- Each jailbreak type, once discovered, can be patched via targeted safety training
- Constitutional AI, RLHF, and adversarial training have dramatically improved model safety over time
- Formal verification may eventually provide provable guarantees for restricted capability classes
- The rate of jailbreak success against state-of-the-art models has decreased over time as alignment improves

**Position 3: The definition of "right" and "wrong" keeps evolving**

This is arguably the most sophisticated answer:
- What constitutes "harmful content" is culturally, legally, and temporally dependent
- Content that is harmful in one context (instructions for synthesizing a chemical) may be legitimate in another (chemistry education)
- AI safety is not a fixed target — as society's values evolve, what models should and shouldn't do evolves with it
- There is no final, static definition of "secure" — only the current best approximation of societal consensus

### The Professor's Framework

The most intellectually honest answer combines all three:
1. **Technical security** — achievable for specific, well-defined threat models (e.g., "refuse all requests that explicitly mention bomb-making") but not for the general case
2. **Semantic security** — probably impossible in an absolute sense because the meaning of "harmful" is unbounded and context-dependent
3. **Evolutionary security** — the best we can achieve is a continuous process of Red Team → patch → Red Team → patch, accepting that we never reach a final state

---

## The Long View: AI Safety as an Ongoing Discipline

The lecture's structure — from evaluation to threat models to attacks to mitigation — traces a journey from *measuring AI quality* to *protecting AI from adversaries*. The key meta-lesson is:

**Deploying an LLM is not an event; it is a continuous process.**

| Phase | Activity |
|-------|----------|
| **Pre-deployment** | Red teaming, adversarial testing, safety alignment, policy design |
| **Deployment** | Monitoring, incident response, rate limiting, output filtering |
| **Post-deployment** | Learning from real-world attacks, retraining with new adversarial examples, updating policies |
| **Ongoing** | Tracking emerging jailbreaks, adapting to evolving societal norms, regulatory compliance |

---

## Interview Questions

**Q1. What is Red Teaming in the context of LLM safety? How does it differ from traditional software penetration testing?**

> **Answer:** Red Teaming for LLMs involves authorized adversarial testing of an LLM system — attempting to produce harmful, unsafe, or unintended outputs using all available attack techniques (jailbreaks, prompt injection, adversarial inputs, low-resource language attacks). It differs from traditional penetration testing in that: (1) the attack surface is natural language (infinite, creative, ambiguous) rather than code and network protocols (finite, formal, deterministic); (2) what constitutes "success" (harmful output) is subjective and context-dependent, not binary (crash/no crash, access/no access); (3) traditional pen testing finds known vulnerability classes; LLM red teaming must also discover novel jailbreak categories; (4) the target model changes with retraining, requiring continuous red teaming rather than a one-time assessment.

---

**Q2. What is the role of the Blue Team in LLM security, and what specific technical activities does it encompass?**

> **Answer:** The Blue Team implements and maintains defenses against the attacks discovered by the Red Team. For LLMs specifically: (1) **Input filtering** — deploying content classifiers that screen user prompts before they reach the model; (2) **Output filtering** — screening model responses before they reach users; (3) **System prompt security** — access controls, secret management, leak detection; (4) **Anomaly detection** — monitoring production traffic for attack patterns (high query rates from one user, systematic probe patterns); (5) **Incident response** — playbooks for when harmful outputs are reported; (6) **Continuous patching** — integrating red team findings into safety retraining pipelines. The Blue Team's effectiveness depends critically on real-time visibility into model behavior — without monitoring, they are defending blind.

---

**Q3. Describe the mitigation stack for a production LLM system. Why is a multi-layer approach necessary?**

> **Answer:** A production LLM safety stack has at least four layers: (1) **Input filter** — content classification, Unicode normalization, PII detection, rate limiting; (2) **Model-level safety** — RLHF or Constitutional AI trained safety alignment; (3) **Output filter** — content classification on outputs, system prompt leakage detection; (4) **Monitoring** — logging, anomaly detection, human escalation. Multi-layer defense is necessary because: (a) no single layer is complete — each has attack bypasses (GCG bypasses input filters; PAIR bypasses rate limiting in small counts; model-level alignment is bypassable via jailbreaks); (b) defense-in-depth ensures that bypassing one layer doesn't mean bypassing all; (c) different layers catch different attack types.

---

**Q4. A researcher argues: "Safety alignment through RLHF is sufficient — we don't need input/output filters." Counter this argument.**

> **Answer:** RLHF alignment is necessary but not sufficient. Counterarguments: (1) **GCG's 99% attack success rate** against RLHF-aligned models (Vicuna) shows that alignment can be broken by automated gradient search — an input filter that rejects high-perplexity inputs would catch GCG suffixes before they reach the model; (2) **PAIR achieves 60–100% success** against RLHF-aligned models using only API queries — even the best-aligned model has semantic attack surfaces that filters can partially mitigate; (3) **Defense-in-depth** is a security principle: never rely on a single control; (4) **Output filters** provide a last-resort catch for cases where the model is jailbroken — even if the harmful output is generated, it can be blocked before reaching the user; (5) RLHF alignment degrades over time with distribution shift — filters provide a stable safety backstop while the model is retrained.

---

**Q5. What is the asymmetry between attackers and defenders in AI safety, and what are its implications?**

> **Answer:** The asymmetry: (1) **Coverage asymmetry** — the defender must block *all* successful attacks; the attacker only needs to find *one* bypass. This gives attackers a fundamental structural advantage; (2) **Creativity asymmetry** — defenses are trained on known attack patterns; attackers can be creative and novel. A new jailbreak category that no one has seen before bypasses all defenses trained on prior categories; (3) **Resource asymmetry** — defenders need to maintain safety 24/7 across all deployments; attackers can spend months finding a single bypass; (4) **Disclosure asymmetry** — defenders must publicly demonstrate safety (for trust); attackers can operate in secret. Implications: (a) perfect security is unachievable; (b) continuous red teaming is the only realistic approach; (c) rapid patch-release cycles are essential; (d) responsible disclosure processes are important for the research community.

---

**Q6. Do you agree that "LLMs can never be made truly secure"? Justify your position.**

> **Answer (nuanced position):** The statement is true in an absolute sense but misleadingly framed. More precisely: "LLMs cannot be made perfectly secure against all possible attacks across all possible harmful content definitions." This is true because: (1) the input space (natural language) is infinite — no finite training procedure covers all attack patterns; (2) "harmful" is a culturally/contextually/temporally dependent concept with no fixed definition; (3) capability and safety are in tension — any model capable enough to be useful has sufficient capability to potentially be exploited.
>
> However, the practically relevant question is not "perfect security" but "acceptable security for a specific threat model." LLMs can be made secure *enough* for specific deployment contexts with specific threat models: a medical information chatbot with strong input/output filtering, limited capability scope, and human oversight is "secure enough" even if not "perfectly secure." The field should focus on contextual security under realistic threat models rather than the philosophically unachievable goal of universal security.

---

## Learning Thoughts

> **Thought 1 — Red Teaming is Not Optional:**
> Every major AI lab (OpenAI, Anthropic, Google DeepMind, Meta) has a dedicated Red Team. This is not because they think their models are unsafe — it's because they know that *untested assumptions* are where real vulnerabilities hide. If you're deploying an LLM-powered product without Red Teaming it first, you are accepting unknown risk.

> **Thought 2 — The Security-Utility Tension is Fundamental:**
> Every defense that reduces harmful outputs also reduces some legitimate outputs (false positives). A model that refuses all requests about chemistry is "safe" from chemistry jailbreaks but useless for chemistry students. There is no free defense — every safety measure has a utility cost. The art of AI safety engineering is minimizing this tradeoff for the specific deployment context.

> **Thought 3 — Safety as a Property of Systems, Not Models:**
> Individual model safety matters, but what ultimately determines safety is the whole *system* — model + input filters + output filters + rate limiting + monitoring + human oversight + organizational policies. A perfectly safe model deployed in an insecure system is not safe. A moderately aligned model in a well-designed system can be very safe. The system is the unit of safety.

> **Thought 4 — The Red-Queen Problem:**
> "It takes all the running you can do, to keep in the same place." Safety alignment is a Red-Queen race — you must continuously improve defenses just to maintain the same security level against increasingly sophisticated attacks. This is not a counsel of despair — it's a framework for setting expectations. Security requires sustained investment, not a one-time fix.

> **Thought 5 — Evolving Definitions of Harm:**
> The statement "the definition of 'right' and 'wrong' keeps evolving" is profound. Content that was acceptable in 2015 may not be acceptable today. Content that causes harm in one cultural context may be neutral in another. AI safety systems must be designed to evolve with societal norms — which requires ongoing human oversight, regular policy reviews, and the humility to recognize that today's "safe" model may need to be updated as understanding evolves. Safety is a social contract, not just a technical property.

---

## Course Summary: The Complete Picture

```
LLM Safety: From Evaluation to Defense

PART 1: LLM EVALUATION
├── Human Evaluation (gold standard, limited by subjectivity)
│   └── Coefficient of Agreement: Cohen's Kappa
├── Statistical & Semantic Metrics (automated, imperfect)
│   ├── ROUGE — recall-oriented, for summarization
│   ├── BLEU — precision-oriented, for translation
│   ├── METEOR — stems + synonyms, better semantic coverage
│   └── BERTScore — semantic similarity, best correlation with humans
└── LLM-as-a-Judge (scalable, multi-dimensional)
    ├── Pointwise vs Pairwise
    ├── Dimensions: Task Performance + Alignment
    └── Factuality quantification

PART 2: AI SAFETY
├── Foundations (CIA Triad → Trustworthy ML)
├── Threat Models (5 attack types)
│   ├── Membership Inference
│   ├── Model Extraction
│   ├── Model Poisoning
│   ├── Model Hijacking
│   └── Adversarial Attacks
├── Risks & Vulnerabilities
│   ├── Data Privacy
│   ├── Jailbreak
│   └── Legal/IP
├── White-box Attacks (GCG deep dive)
│   └── HotFlip, TextFooler, GCG, AutoDAN
├── Prompt-based Attacks
│   ├── Indirect Prompt Injection
│   └── Prompt Leakage
├── Black-box Attacks (PAIR focus)
│   ├── Low-resource language
│   ├── Context contamination
│   ├── DeepWordBug
│   └── PAIR
├── Jailbreak Taxonomy
│   ├── Language Strategies
│   ├── Rhetoric
│   ├── Imaginary Worlds
│   └── LLM Operational Exploitation
└── Mitigation
    └── Red Team vs Blue Team (continuous process)
```

---

*Previous: [Section 10 — Jailbreak Taxonomy](Section10_Jailbreak_Taxonomy.md)*
*Return to: [Section 1 — Human Evaluation](Section1_Human_Evaluation.md)*
