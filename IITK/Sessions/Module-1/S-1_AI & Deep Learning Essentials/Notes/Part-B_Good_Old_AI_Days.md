# Part B — Good Old AI Days: Symbolic AI

> **Course:** Executive PGP in Generative AI & Agentic AI | IIT Kharagpur × upGrad
> **Instructor:** Prof. Niloy Ganguly, IIT Kharagpur
> **Session:** Module 1 — AI & Deep Learning Essentials

---

## Learning Compass

> Symbolic AI represents the first serious attempt to build intelligence by encoding human knowledge as explicit rules, logic, and structures. Understanding why it worked — and why it ultimately hit a wall — is essential to understanding why modern ML/DL exists.

By the end of this section you will be able to:
- Explain the three classical methods of knowledge representation
- Read and write predicate logic statements
- Trace how a rule-based inference engine arrives at a conclusion
- Understand how semantic nets encode relational knowledge
- Describe how expert systems were built and used
- Explain the three root causes of the AI winters

---

## Topic 5 — Representing Knowledge

### The Central Problem

Before a machine can reason, it needs **knowledge**. The core question of classical AI was:

> *How do you take human knowledge — which lives in human heads, books, and culture — and encode it in a form a machine can work with?*

Three major approaches emerged:

---

### Approach 1 — Logic

**What it is:** Formal symbolic reasoning using **propositions**, **predicates**, and **inference rules**.

Logic gives AI a mathematically rigorous language for representing facts and relationships, and a provably correct mechanism for deriving new facts.

**Two levels of logic:**

| Type | What it handles | Example |
|---|---|---|
| Propositional Logic | True/false statements | "It is raining" = True |
| Predicate Logic (First-Order) | Objects, properties, relationships, quantifiers | ∀x graduating(x) → happy(x) |

**Key operators:**
- `∀` — For all (universal quantifier)
- `∃` — There exists (existential quantifier)
- `→` — Implies
- `∧` — AND
- `∨` — OR
- `¬` — NOT

**Strengths:**
- Formally sound — conclusions guaranteed if premises are correct
- Machine-checkable and verifiable

**Weaknesses:**
- Brittle — everything must be explicitly stated
- Cannot handle uncertainty or degrees of belief
- Knowledge acquisition bottleneck — experts can't always articulate rules

---

### Approach 2 — Production Rules (If-Then Rules)

**What it is:** Knowledge encoded as **IF condition THEN action** rules, used by a rule engine to draw conclusions.

This is the most practical form of symbolic AI and formed the basis of **Expert Systems**.

**Structure:**

```
IF  <condition(s)>
THEN  <conclusion or action>
```

**Example rules for a car diagnostic system:**
```
R1: IF gas_in_engine AND does_not_start    THEN problem(spark_plugs)
R2: IF NOT does_not_start AND NOT lights_on THEN problem(battery)
R3: IF NOT turns_over AND lights_on         THEN problem(starter)
R4: IF gas_in_tank AND gas_in_carb          THEN gas_in_engine
```

**Inference trace:**

Given: `gas_in_tank = true`, `gas_in_carb = true`, `does_not_start = true`

Step 1: R4 fires → `gas_in_engine = true`
Step 2: R1 fires (gas_in_engine AND does_not_start) → `problem(spark_plugs)` ✓

This is **forward chaining** — start from facts, fire rules, derive conclusions.

There is also **backward chaining** — start from a goal, work backwards to find what facts would prove it (used in PROLOG).

---

### Approach 3 — Semantic Nets (Semantic Networks)

**What it is:** A graph where **nodes represent concepts** and **edges represent relationships** between them.

This is the most intuitive knowledge representation — it mirrors how humans think in terms of categories and relationships.

**Example:**

```
Nellie ──[Is-a]──► Elephant ──[Is-a]──► Animal
                        │                   │
                   [Lives-in]           [Has]
                        │                   │
                       ▼                   ▼
                     Africa              head
```

**Two types of edges:**
1. **Is-a / Instance-of** (inheritance): Nellie is an Elephant; Elephant is an Animal
2. **Property / Attribute** (has-a): Animal has head

**Inheritance:** Because Nellie is-an Elephant and Elephant is-an Animal, Nellie **inherits** all properties of Animal — including "has head" — without explicitly stating it. This is **property inheritance**.

**Real-world descendants:**
- **WordNet** — a semantic network of English words used in NLP
- **Knowledge Graphs** (Google, Wikidata) — modern descendants of semantic nets
- **Ontologies** in the Semantic Web (OWL, RDF)

---

## Topic 6 — Logic: Predicate Logic in Detail

### Why Predicate Logic over Propositional?

Propositional logic is too limited:
- "All people who are graduating are happy" cannot be expressed as a simple true/false statement
- You need **variables** and **quantifiers** to express general truths

### The Anatomy of a Predicate Logic Statement

```
∀x  graduating(x) → happy(x)
│   │               │
│   predicate       predicate
│   (graduating     (happy
│    is a property   is a property
│    of x)           of x)
universal
quantifier
("for all x")
```

### Full Example — Deriving a Conclusion

**Natural Language:**
1. (Premise) All people who are graduating are happy.
2. (Premise) All happy people smile.
3. (Fact) Someone is graduating.
4. (Question) Is someone smiling?

**Predicate Logic Translation:**

```
1. ∀x  graduating(x) → happy(x)
2. ∀x  happy(x) → smiling(x)
3. ∃x  graduating(x)
```

**Inference (Modus Ponens applied twice):**

- From (3): there exists some individual, call them `a`, such that `graduating(a)` is true
- Apply rule (1): `graduating(a) → happy(a)` → therefore `happy(a)` is true
- Apply rule (2): `happy(a) → smiling(a)` → therefore `smiling(a)` is true
- Therefore: `∃x smiling(x)` ✓ — Yes, someone is smiling

### Key Inference Rule: Modus Ponens

```
If P → Q is true
And P is true
Then Q must be true
```

This is the fundamental engine of logical reasoning. All formal theorem proving is built on this and related rules.

### Completeness vs Decidability

A critical theoretical result (Gödel's incompleteness theorem, Turing's halting problem): **not all true statements in a formal system can be proven within that system**. Logic has fundamental limits. This isn't just academic — it means rule-based AI systems will always have blind spots.

---

## Topic 7 — Rule-Based Inference: How Expert Systems Reason

### Forward Chaining vs Backward Chaining

**Forward Chaining (Data-Driven):**
- Start with known facts
- Apply rules to derive new facts
- Keep going until goal is reached or no more rules fire
- Used for: monitoring, event-driven systems, alerts

```
Known facts → Fire matching rules → New facts → Fire more rules → Conclusion
```

**Backward Chaining (Goal-Driven):**
- Start with a goal (what you want to prove)
- Find rules whose conclusion matches the goal
- Set the rule's conditions as sub-goals
- Recurse until all sub-goals are grounded in known facts
- Used for: diagnosis, query answering, PROLOG

```
Goal → What rules conclude this? → Sub-goals → What rules conclude sub-goals? → Facts
```

### Conflict Resolution

When multiple rules are eligible to fire simultaneously, the rule engine uses **conflict resolution strategies:**
- **Recency:** Fire the rule that uses the most recently added facts
- **Specificity:** Fire the most specific (most conditions) rule first
- **Priority:** Assign explicit priority weights to rules

### The Car Diagnostic Example — Full Trace

Rules:
```
R1: IF gas_in_engine AND does_not_start    THEN problem(spark_plugs)
R2: IF NOT does_not_start AND NOT lights_on THEN problem(battery)
R3: IF NOT turns_over AND lights_on         THEN problem(starter)
R4: IF gas_in_tank AND gas_in_carb          THEN gas_in_engine
```

Known facts: `gas_in_tank=T, gas_in_carb=T, does_not_start=T`

**Forward chaining trace:**

| Step | Rule Fired | New Fact |
|---|---|---|
| 1 | R4: gas_in_tank ∧ gas_in_carb → | gas_in_engine = TRUE |
| 2 | R1: gas_in_engine ∧ does_not_start → | problem(spark_plugs) ✓ |

**Result:** The car has a spark plug problem.

### MYCIN — The Famous Medical Expert System

MYCIN (Stanford, 1970s) was a rule-based system for diagnosing bacterial infections and recommending antibiotics. It had approximately **600 rules** of the form:

```
IF  the infection is primary-bacteremia
AND the site of the culture is one of the sterile sites
AND the suspected portal of entry is the GI tract
THEN there is suggestive evidence (0.7) that the identity of the organism is bacteroides
```

Note the **certainty factor (0.7)** — MYCIN introduced probabilistic weighting to rules, making it more realistic than pure Boolean logic.

MYCIN **outperformed junior physicians** in controlled tests on blood infections. It was never deployed clinically (liability concerns), but it proved the concept.

---

## Topic 8 — Semantic Nets

### The Fundamental Idea

A semantic net encodes knowledge as a labeled directed graph:

```
Node = concept / entity / class
Edge = labeled relationship between nodes
```

### Types of Relationships

| Relationship | Meaning | Example |
|---|---|---|
| is-a | Class membership (inheritance) | Dog is-a Animal |
| instance-of | Specific instance | Fido instance-of Dog |
| has-a | Possession or attribute | Animal has-a heart |
| part-of | Composition | Wheel part-of Car |
| lives-in | Location | Penguin lives-in Antarctica |
| made-of | Material | Table made-of Wood |

### Property Inheritance — The Power of is-a

The is-a hierarchy allows knowledge to be inherited down the hierarchy:

```
Animal
 ├── has: heart, brain
 ├── can: breathe, move
 │
 └── Mammal (is-a Animal → inherits all Animal properties)
      ├── has: fur, warm blood
      │
      └── Dog (is-a Mammal → inherits Mammal AND Animal properties)
           ├── can: bark
           └── Fido (instance-of Dog → inherits everything)
```

So you can ask: "Does Fido have a heart?" and the answer is YES — not because it's stated explicitly, but because Fido is-a Dog is-a Mammal is-a Animal, and Animal has a heart.

This is **default reasoning** — assume inherited properties unless overridden.

### Exception Handling

```
Bird can fly
Penguin is-a Bird
Penguin cannot fly  (override!)
```

Tweety is-a Penguin → Tweety cannot fly (specific rule overrides general)

This is called **non-monotonic reasoning** — adding new information (Tweety is a penguin) can **retract** previously derived conclusions (Tweety can fly).

### Limitations of Semantic Nets

1. **No standard notation** — different systems used different conventions
2. **No formal semantics** — ambiguous interpretation of relationships
3. **Combinatorial explosion** — large nets become unwieldy
4. **No uncertainty** — relationships are binary (is / is-not)

---

## Topic 9 — Expert Systems

### Definition

> An **expert system** is a computer program that emulates the decision-making ability of a human expert in a specific domain.

### Architecture

```
┌─────────────────────────────────────────────────┐
│                  Expert System                  │
│                                                 │
│  Non-expert   ──►  User      ◄──►  Rules    ◄──  Knowledge
│    User            Interface       Engine        Base
│              ◄──                                    ▲
│   (Advice)                                          │
└─────────────────────────────────────────────────┘
                                              Knowledge from
                                                  Expert
```

**Three core components:**

1. **Knowledge Base:** The stored expertise — rules, facts, heuristics extracted from human experts

2. **Inference Engine (Rules Engine):** The reasoning mechanism — applies rules to facts to derive conclusions (forward or backward chaining)

3. **User Interface:** The interaction layer — accepts user inputs and presents recommendations

### How an Expert System is Built

**Step 1: Knowledge Acquisition**
A **knowledge engineer** interviews domain experts (doctors, geologists, financial advisors) and extracts their reasoning patterns into IF-THEN rules.

**Step 2: Knowledge Encoding**
Rules are formalized and stored in the knowledge base.

**Step 3: Inference Design**
The inference engine is configured for forward or backward chaining depending on the use case.

**Step 4: Validation**
The system is tested against known cases and results compared with expert decisions.

### Famous Expert Systems

| System | Domain | Organization | Achievement |
|---|---|---|---|
| MYCIN | Medical diagnosis (bacterial infections) | Stanford | Outperformed junior physicians |
| DENDRAL | Chemical structure identification | Stanford | First expert system (1965) |
| XCON (R1) | Computer system configuration | DEC | Saved DEC $40M/year |
| Prospector | Mineral exploration | SRI | Discovered molybdenum deposit worth $100M |
| CADUCEUS | Medical diagnosis | University of Pittsburgh | Covered 1000+ diseases |

### The Commercial Expert Systems Boom (1980s)

In the 1980s, expert systems became a serious commercial industry:
- Hundreds of companies built expert systems
- The market reached **$1 billion/year by 1988**
- Every major corporation had an AI department
- LISP workstations were sold for $100,000+ each

---

## Topic 10 — Why It Hit the Wall (The Root Causes of AI Winters)

### Cause 1: Ambiguity

Natural language and real-world knowledge are deeply ambiguous. Symbolic AI assumed ambiguity could be resolved with more rules — it couldn't.

**The translation failure example:**

In the 1960s, the US government invested heavily in machine translation (Russian → English). The systems were syntactically sophisticated but semantically blind.

Famous failure:
> "The spirit is willing but the flesh is weak."
> Translated to Russian, then back to English:
> *"The vodka is good but the meat is rotten."*

The system knew grammar rules but not the **meaning** behind idiomatic expressions.

**Why this mattered:** Language understanding requires **world knowledge, context, and pragmatics** — not just syntax. Rules cannot encode all of this.

### Cause 2: Scalability and Complexity

Early AI programs worked brilliantly on toy problems — small, clean, well-defined instances. But real-world problems are:
- **Larger by orders of magnitude:** A chess program for a 8×8 board; real decisions involve millions of variables
- **Noisier:** Real data has errors, missing values, contradictions
- **Dynamic:** The world changes; static rule bases become outdated

The General Problem Solver (GPS) could solve puzzles but couldn't scale to real planning problems. MYCIN had 600 rules for one disease domain; scaling to all of medicine would require millions — and the rules would conflict.

**The combinatorial explosion:** As problem size grows, the search space grows **exponentially**. Rules-based search is fundamentally unscalable.

### Cause 3: Limitations of Knowledge Representation

**Problem 1 — Knowledge Acquisition Bottleneck**
Extracting knowledge from experts is slow, expensive, and incomplete. Experts often cannot articulate how they actually make decisions (tacit knowledge). The bottleneck shifted from computing to knowledge capture.

**Problem 2 — Brittleness**
Rule-based systems fail catastrophically when faced with situations outside their rule base. They have no mechanism for:
- Graceful degradation (being a little wrong rather than completely wrong)
- Handling novel situations
- Common sense reasoning

**Example:** A medical expert system might correctly diagnose 99% of cases by the book, but fail completely on a patient with an unusual presentation — and fail with high confidence, not low confidence.

**Problem 3 — The Frame Problem**
When the world changes, what do you update? If you assert "the block is on the table" and then move it, you need rules for everything that *did not* change as well as what did. In a complex world, the number of "non-change" assertions grows without bound.

**Problem 4 — Uncertainty**
Real-world knowledge is uncertain. "Most birds can fly" is true. "All birds can fly" is false. Symbolic logic is binary — a statement is true or false. It cannot natively represent "probably true" or "usually true."

---

## Topic 11 — AI Winter: The Cyclical History

### The Pattern

```
BOOM 1: GOFAI (1960s)
  ↓ Heuristic search, logic, General Problem Solver
  ↓ Promise: "We'll solve AI in 10 years"
  ↓ Reality: Combinatorial explosion, semantic ambiguity
WINTER 1 (1970s): Funding cut; DARPA withdraws
  ↓
BOOM 2: Expert Systems (1980s)
  ↓ MYCIN, XCON, DENDRAL — real commercial value
  ↓ Promise: "Domain-specific AI will transform every industry"
  ↓ Reality: Maintenance nightmare, brittleness, can't handle uncertainty
WINTER 2 (1990s): Expert systems fall out of favor; AI budgets collapse
  ↓
BOOM 3: Machine Learning / Deep Learning (2010s–present)
  ↓ ImageNet, AlexNet, Transformers, ChatGPT
  ↓ Promise: Scale with data and compute
  ↓ Reality: ???
```

### Key Periods

| Period | Name | What Drove It | What Killed It |
|---|---|---|---|
| 1960s | Boom 1 "GOFAI" | Heuristic search, GPS, logic | Combinatorial explosion, semantic ambiguity |
| 1970s | Winter 1 | — | DARPA report (Lighthill Report UK, ALPAC Report US) |
| 1980s | Boom 2 "Expert Systems" | MYCIN, XCON, commercial AI | Brittleness, maintenance cost, knowledge acquisition bottleneck |
| 1990s | Winter 2 | — | Expert systems collapse; LISP machine market dies |
| 2010s+ | Boom 3 "ML/DL" | Big data, GPUs, deep learning | Still going — but LLM hallucinations, alignment, and interpretability are warning signs |

### The Lighthill Report (1973) — What Killed the First Boom

The British Science Research Council commissioned mathematician Sir James Lighthill to evaluate AI progress. His 1973 report was devastating:

> AI researchers had vastly overpromised and underdelivered. The field had failed to produce results that were useful in anything other than toy domains.

The report led to the cancellation of most AI funding in the UK and influenced similar cuts in the US. This was Winter 1.

### The LISP Machine Bust (1987) — What Triggered Winter 2

In the early 1980s, companies built specialized hardware (LISP machines) to run expert systems. By 1987, cheaper general-purpose computers (Apple Mac, IBM PC) outperformed them. The market collapsed overnight. Companies like Symbolics and LMI went bankrupt. This was the trigger for Winter 2.

### Is Boom 3 Different?

Arguments that the current ML/DL boom is **structurally different:**

1. **Self-scaling:** Unlike symbolic AI, deep learning improves with more data and more compute — there's no hard algorithmic wall
2. **Empirical success:** ImageNet → AlexNet → GPT-4 is a demonstrable improvement curve, not promises
3. **Infrastructure depth:** Cloud computing, GPU clusters, open-source frameworks mean the ecosystem is far more robust
4. **Commercial integration:** AI is now deeply embedded in products generating real revenue — hard to defund

Arguments for caution:
1. **Hallucinations and alignment** remain unsolved
2. **Interpretability** — we don't understand why deep networks work
3. **Data hunger** — diminishing returns may set in
4. **Regulatory risk** — governments worldwide are starting to regulate

---

## Interview Questions — Part B

**Q1: What are the three main methods of knowledge representation in symbolic AI?**

> (1) **Logic** — formal symbolic reasoning using predicates and inference rules; ensures provably correct conclusions from true premises but fails on uncertain or incomplete knowledge. (2) **Production Rules** (if-then rules) — encode expert knowledge for automated inference via forward/backward chaining; practical but brittle. (3) **Semantic Networks** — graph-based representation of concepts and relationships; supports inheritance and intuitive knowledge organization but lacks formal semantics and struggles with uncertainty.

**Q2: What is the difference between forward chaining and backward chaining?**

> **Forward chaining** is data-driven: start from known facts, repeatedly apply matching rules to derive new facts until the goal is reached or no rules fire. Used in event-monitoring and alert systems. **Backward chaining** is goal-driven: start from the desired conclusion, find rules that could prove it, set their conditions as sub-goals, and recurse until grounded in known facts. Used in diagnosis, PROLOG, query answering. Forward chaining is exhaustive; backward chaining is focused on a specific goal and thus more efficient when the goal is known.

**Q3: What is property inheritance in semantic nets and why is it useful?**

> Property inheritance means that when a concept is linked to another via an `is-a` relationship, it automatically acquires all properties of the parent concept. For example: Fido is-a Dog, Dog is-a Mammal, Mammal is-a Animal. If Animal has a heart, then Fido has a heart — without explicitly stating it. This avoids **redundant knowledge storage** and enables compact, elegant knowledge bases. Overrides handle exceptions (Penguin is-a Bird but cannot fly).

**Q4: What is the knowledge acquisition bottleneck and why was it fatal for expert systems?**

> The knowledge acquisition bottleneck is the difficulty of extracting expert knowledge into formal rules. Experts struggle to articulate how they actually reason — much of their expertise is **tacit** (learned through experience, not explainable in rules). Moreover, as domains grow, the number of rules grows without bound, rules conflict, and maintenance becomes prohibitively expensive. This bottleneck meant expert systems could never cover full domains — only narrow, well-structured sub-problems.

**Q5: Explain the brittleness problem in expert systems with an example.**

> Expert systems perform well within their rule base but **fail catastrophically** outside it. Unlike humans, who degrade gracefully when faced with novel situations, rule systems either fire an incorrect rule (wrong answer with high confidence) or fire no rule (no answer). Example: A medical expert system trained on typical bacterial infection presentations might confidently misdiagnose a rare atypical presentation, or simply say "unknown" when a knowledgeable doctor would reason by analogy to related cases.

**Q6: What is the frame problem in symbolic AI?**

> The frame problem is the challenge of representing **what does not change** when an action is taken. If a robot moves a block from the table to the floor, we need to assert not just "block is now on floor" but also that the table still exists, the ceiling hasn't changed, gravity still works, etc. In a complex world, the number of unchanged facts is enormous. Symbolic systems required explicit rules for all non-changes — which is intractable. This was a fundamental scalability problem for representing dynamic worlds.

**Q7: Why did Machine Learning succeed where symbolic AI failed?**

> ML succeeded because it abandoned the assumption that intelligence must be **explicitly programmed**. Instead of encoding rules, ML learns patterns directly from data. This sidesteps: the knowledge acquisition bottleneck (no expert interviews needed), brittleness (statistical models generalize gracefully), and scaling limits (performance improves with more data). The shift was from **reasoning about knowledge** to **learning from examples** — a fundamentally different paradigm.

---

## Key Learning Thoughts — Part B

> **Thought 1:** Symbolic AI was not wrong — it was incomplete. Logic and rules work beautifully in well-defined, closed-world domains. The mistake was assuming the world is well-defined and closed.

> **Thought 2:** The knowledge acquisition bottleneck foreshadowed the data labeling bottleneck in modern ML. Both are instances of the same problem: extracting human knowledge into machine-usable form is hard and expensive.

> **Thought 3:** Semantic networks are not dead. They live on in Knowledge Graphs (Google's Knowledge Graph, Wikidata), biomedical ontologies, and the structure of LLM training data. Understanding them helps you understand where modern AI gets its "world knowledge."

> **Thought 4:** MYCIN's certainty factors (0.7, 0.4) were a pragmatic hack for handling uncertainty. The formal solution came later with probabilistic reasoning and Bayesian networks — ancestors of modern probabilistic ML.

> **Thought 5:** AI winters are caused by **expectation gaps**, not technical failure. The technology works — just not as ambitiously as promised. Managing expectations is a professional skill for anyone working in AI.

> **Thought 6:** The brittleness of rule systems is directly why neural networks use **distributed representations** — no single rule or neuron encodes a fact; knowledge is spread across millions of weights. This makes them robust to partial damage and novel inputs.

---

*Previous: [Part A — History & Definition of AI](Part-A_History_and_Definition_of_AI.md)*
*Next: [Part C — Machine Learning](Part-C_Machine_Learning.md)*
