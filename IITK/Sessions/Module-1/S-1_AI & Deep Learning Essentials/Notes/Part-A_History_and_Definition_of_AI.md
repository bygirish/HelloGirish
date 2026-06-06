# Part A — History & Definition of Artificial Intelligence

> **Course:** Executive PGP in Generative AI & Agentic AI | IIT Kharagpur × upGrad
> **Instructor:** Prof. Niloy Ganguly, IIT Kharagpur
> **Session:** Module 1 — AI & Deep Learning Essentials

---

## Learning Compass

Before diving in, anchor yourself with this mental model:

> AI is not one thing — it is a 70-year-old conversation between humans and machines about what it means to think, learn, and act intelligently.

By the end of this section you will be able to:
- Explain when and why AI was formally born as a discipline
- Trace the key milestones that shaped AI into what it is today
- Articulate what the Turing Test is and why it still matters
- Define AI across four philosophical quadrants and explain why modern AI sits in one specific quadrant

---

## Topic 1 — The Birth of AI: The 1956 Dartmouth Conference

### What happened?

In the summer of **1956**, a group of the most brilliant minds of the era gathered at **Dartmouth College, Hanover, New Hampshire**, for a research project formally titled:

> *"The Dartmouth Summer Research Project on Artificial Intelligence"*

This was organized by **John McCarthy** (Dartmouth College), **Marvin Minsky** (MIT), **Nathaniel Rochester** (IBM), and **Claude Shannon** (Bell Laboratories).

The plaque at the building reads:

> *"To proceed on the basis of the conjecture that every aspect of learning or any other feature of intelligence can in principle be so precisely described that a machine can be made to simulate it."*

This is the foundational hypothesis of all of AI — that intelligence is **describable** and therefore **simulatable**.

### The Founding Fathers

| Person | Contribution |
|---|---|
| John McCarthy | Coined the term "Artificial Intelligence"; created LISP |
| Marvin Minsky | Co-founder of MIT AI Lab; worked on neural networks and frames |
| Claude Shannon | Father of Information Theory; foundation for all digital communication |
| Ray Solomonoff | Pioneer of algorithmic probability and inductive inference |
| Alan Newell | Created Logic Theorist and General Problem Solver with Herbert Simon |
| Herbert Simon | Nobel laureate; bounded rationality; human decision making |
| Arthur Samuel | Wrote the first self-learning checkers program (1952) |
| Oliver Selfridge | Pioneered machine perception and pattern recognition |
| Nathaniel Rochester | Designed IBM's first commercial scientific computer |
| Trenchard More | Contributed to array theory and early AI languages |

### Why this matters

Before 1956, there was no unified field called "AI." There were mathematicians, logicians, engineers, and psychologists all working on related problems in isolation. The Dartmouth conference was the **naming moment** — the moment a discipline was born.

### Key insight

The founding assumption was **optimistic to the point of naivety**: "give us a summer and we'll crack intelligence." They didn't crack it. But the attempt started 70 years of extraordinary progress.

---

## Topic 2 — Major Milestones in AI History

### The Timeline at a Glance

```
1950  Alan Turing proposes the Turing Test
1956  Dartmouth Conference — "AI" coined
1959  "Machine Learning" term introduced
1966  ELIZA — first chatbot (Weizenbaum)
1975  First AI applied to medicine: MYCIN
1976  Neocognitron — inspiration for CNNs
1979  Introduction of Transfer Learning (concept)
1997  First CNN paper | Deep Blue beats Kasparov
2002  Launch of Torch library
2006  IBM's Watson wins Jeopardy
2011  Creation of ImageNet dataset
2017  FDA clearance for AI diabetic retinopathy detection
2018  FDA clearance for cloud-based deep learning (Arterys)
2021  Launching of AlphaFold Project
2022  DALL-E, GPT-3.5, ChatGPT
```

### Deep Dive on Critical Milestones

#### 1950 — Alan Turing and the Turing Test
Turing published *"Computing Machinery and Intelligence"* asking: **"Can machines think?"** He proposed an empirical test rather than a philosophical debate — replace the question with a measurable experiment. This was revolutionary scientific thinking.

#### 1966 — ELIZA (Weizenbaum)
The first chatbot. It used simple pattern matching and scripted responses to simulate a psychotherapist. The shocking finding: people formed emotional attachments to it and believed it understood them. This raised the first serious questions about **the gap between appearance and understanding in AI**.

#### 1975 — MYCIN
An expert system for diagnosing bacterial infections and recommending antibiotics. It outperformed junior doctors in controlled tests. This was the first proof that AI could have **real-world clinical utility** — and also the beginning of the expert systems boom.

#### 1997 — Deep Blue defeats Kasparov
IBM's Deep Blue became the first computer to beat a reigning world chess champion (Garry Kasparov) under standard tournament conditions. This was a cultural watershed — AI was no longer theoretical. The newspaper headline: *"Kasparov Proves No Match for Computer."*

#### 2011 — ImageNet
The creation of the ImageNet dataset (14 million labelled images, 1000 categories) enabled the deep learning revolution. Without data at scale, neural networks couldn't show their power. ImageNet was the catalyst.

#### 2021 — AlphaFold
DeepMind's AlphaFold solved the **50-year-old protein folding problem** — predicting a protein's 3D structure from its amino acid sequence. This is arguably the most significant scientific achievement of AI so far, with massive implications for drug discovery.

#### 2022 — ChatGPT
OpenAI's ChatGPT reached **100 million users in 2 months** — the fastest adoption of any consumer technology in history. It brought generative AI into everyday life.

### The "Three Booms and Two Winters" Pattern

AI has never grown linearly. It has followed a **hype cycle**:

| Period | Name | Driver |
|---|---|---|
| 1960s | Boom 1 (GOFAI) | Heuristic search, logic, General Problem Solver |
| 1970s | Winter 1 | Scaled poorly; semantic ambiguity; funding dried up |
| 1980s | Boom 2 (Expert Systems) | Rule-based AI, knowledge engineering |
| 1990s | Winter 2 | Maintenance cost; brittleness; couldn't handle uncertainty |
| 2010s–now | Boom 3 (Machine Learning/Deep Learning) | Data + compute + algorithms |

### Key insight

Every AI winter happened for the same root reason: **the current technique couldn't scale to real-world complexity.** The pattern tells us: whenever we hit the limits of our current paradigm, we don't abandon AI — we find a new paradigm.

---

## Topic 3 — The Turing Test

### The Setup

Published in 1950, Alan Turing's test (originally called the **Imitation Game**) works as follows:

- A **human judge** communicates via text-only channel with two entities: a **human** and a **machine**
- Both the human and machine try to convince the judge they are human
- The judge tries to tell which is which
- **If the judge cannot reliably distinguish the machine from the human, the machine is said to have passed the Turing Test**

### Why text-only?

Turing deliberately removed physical appearance, voice, and gesture from the equation. The test is purely about **reasoning, language, and intelligence** — not embodiment.

### Real-world variants

| System | Year | Notes |
|---|---|---|
| ELIZA | 1966 | First system to fool some users in casual conversation |
| Mitsuku (Kuki) | 2005–present | Won Loebner Prize (AI Turing Test competition) multiple times |
| ChatGPT / GPT-4 | 2022–present | Widely considered to have passed conversational Turing Test informally |

### Criticisms of the Turing Test

1. **The Chinese Room (Searle, 1980):** A person in a room following rules to manipulate Chinese symbols could pass a Turing Test in Chinese without understanding Chinese. The test measures **behavior**, not **understanding**.
2. **The test is too easy:** Humans are easily fooled. A system that mimics conversational patterns (without reasoning) can pass.
3. **The test is too narrow:** Intelligence includes perception, action, creativity, emotional reasoning — none of which are tested.
4. **It's anthropocentric:** Why should human-like conversation be the gold standard for intelligence?

### Why it still matters

Despite its flaws, the Turing Test remains important because:
- It shifted the question from **"what is intelligence?"** (philosophical) to **"can we measure it?"** (scientific)
- It set the agenda for NLP and conversational AI for 70 years
- It forces us to ask: **what do we actually want from an intelligent machine?**

---

## Topic 4 — What is Artificial Intelligence? The Four-Quadrant Framework

### The Core Tension

There are two axes along which we can define AI:

**Axis 1:** Does the system **Think** or **Act**?
**Axis 2:** Does it behave **Humanly** or **Rationally**?

This gives us a 2×2 framework:

```
                 HUMAN              RATIONAL
              ┌──────────────┬──────────────────┐
  THINK       │  Think Like  │  Think Rationally │
              │   Humans     │  (Logic, Reasoning)│
              ├──────────────┼──────────────────┤
  ACT         │  Act Like    │  Act Rationally   │
              │   Humans     │  (Achieve Goals)  │
              └──────────────┴──────────────────┘
```

### Quadrant 1 — Thinking Humanly

**Definition:** Automation of activities we associate with human thinking (Bellman, 1978)

This requires a model of how humans actually think — cognitive science, psychology, neuroscience. Example: A system that solves math problems the same way a student does (with errors, shortcuts, and intuitions).

**Key reference:** Charniak & McDermott (1985) — "Study of mental faculties through computational models"

**Challenge:** We don't fully understand how humans think. And even if we did, human thinking is often wrong, biased, and inefficient.

### Quadrant 2 — Thinking Rationally

**Definition:** Systems that use formal logic, inference rules, and deduction to arrive at correct conclusions.

This is the domain of **symbolic AI**, logic programming (PROLOG), theorem provers. If your premises are true and your logic is valid, your conclusions must be true.

**Challenge:** Not everything can be encoded in formal logic. Common sense, uncertainty, ambiguity — these break formal systems.

### Quadrant 3 — Acting Humanly

**Definition:** Systems that behave in ways indistinguishable from humans (i.e., pass the Turing Test).

ELIZA, chatbots, social robots. The focus is on **output behavior**, not internal mechanism.

**Challenge:** Humans behave irrationally, emotionally, inconsistently. Mimicking human behavior doesn't necessarily make a system useful or safe.

### Quadrant 4 — Acting Rationally ← Modern AI lives here

**Definition:** Systems that **perceive their environment and take actions that maximize the chance of achieving their goals**.

This is the **rational agent** model. It doesn't care whether the system thinks like a human or even behaves like one — it only cares whether the system **achieves the best outcome**.

**Example:** AlphaGo doesn't play Go like a human. It plays Go better than any human. It acts rationally — not humanly.

**Why modern AI focuses here:**
- Goals can be formally specified
- Performance can be objectively measured
- The system can be optimized
- Human-like behavior is not required, only goal achievement

### The Philosophical Implication

By moving to "acting rationally," AI shed its dependence on human cognition as a template. This freed it to be **superhuman** in narrow domains — chess, Go, protein folding, image recognition — while remaining limited in others.

---

## Interview Questions — Part A

### Foundational

**Q1: When was AI formally born as a discipline and what was significant about it?**

> AI was formally born at the **1956 Dartmouth Summer Research Project**, organized by McCarthy, Minsky, Shannon, and Rochester. Its significance was threefold: (1) it gave the field a name ("Artificial Intelligence"), (2) it unified disparate researchers under a common agenda, and (3) it articulated the founding hypothesis — that all aspects of intelligence can be precisely described and simulated by machines.

**Q2: What is the Turing Test and what are its main criticisms?**

> The Turing Test (1950) states that if a human judge cannot reliably distinguish a machine from a human through text conversation, the machine can be considered intelligent. Key criticisms: (1) Searle's Chinese Room — behavior without understanding; (2) it's anthropocentric — human conversation is not the only form of intelligence; (3) it's easily fooled by statistical mimicry; (4) it ignores perception, action, creativity, and embodied intelligence.

**Q3: What are the four quadrants of AI and where does modern AI sit?**

> The four quadrants are: Think Humanly, Think Rationally, Act Humanly, Act Rationally. Modern AI primarily operates in the **Act Rationally** quadrant — building systems that perceive environments and take actions to maximize goal achievement. This is preferred because goals can be formally specified, measured, and optimized — unlike the poorly-understood processes of human thought.

**Q4: What caused the two AI winters?**

> Both winters were caused by the same root problem: **the current AI paradigm could not scale to real-world complexity**. Winter 1 (1970s): symbolic logic systems couldn't handle semantic ambiguity and scaled poorly. Winter 2 (1990s): expert systems were too brittle, couldn't handle uncertainty, and were expensive to maintain. The pattern teaches us that paradigm shifts — not incremental improvements — are what break through.

**Q5: Why is AlphaFold considered one of the most significant AI achievements?**

> AlphaFold (2021, DeepMind) solved the **protein structure prediction problem** — determining a protein's 3D shape from its amino acid sequence — which had stumped biologists for 50 years. The 3D structure of a protein determines its function. Solving this at scale opens doors to designing new drugs, understanding diseases, and accelerating biomedical research.

### Deeper / Interview-level

**Q6: Is passing the Turing Test sufficient to claim machine consciousness?**

> No. Passing the Turing Test only demonstrates that a machine can **simulate intelligent conversation**. Searle's Chinese Room argument shows that symbol manipulation following rules can produce correct outputs without any understanding. Consciousness requires subjective experience (qualia), intentionality, and self-awareness — none of which are measurable by the Turing Test. The test measures behavior, not inner states.

**Q7: The founding conjecture of Dartmouth was that "every aspect of intelligence can be precisely described." Is this still believed?**

> This remains the working hypothesis of most of AI, but it is contested. Critics (like Dreyfus, Penrose) argue that human intelligence involves tacit knowledge, embodied experience, and possibly quantum processes that cannot be fully formalized. Practical AI has largely side-stepped this debate by defining intelligence operationally — if the system achieves the goal, it's "intelligent enough" for the purpose.

**Q8: Why did the Machine Learning boom (Boom 3) not collapse like the first two booms?**

> Three simultaneous developments converged: (1) **Data** — the internet and digitization created unprecedented labeled and unlabeled data; (2) **Compute** — GPU acceleration made training large neural networks feasible; (3) **Algorithms** — breakthroughs like backpropagation, dropout, batch normalization, and attention mechanisms solved key training challenges. Unlike previous booms, ML scales with more data and compute — it doesn't hit a hard wall.

---

## Key Learning Thoughts — Part A

> **Thought 1:** AI's history is a series of paradigm shifts, not a linear march. Understanding *why* each paradigm failed is more important than memorizing when it rose.

> **Thought 2:** The Turing Test is more important as a philosophical provocation than as an engineering benchmark. It asked the right question even if it gave the wrong answer.

> **Thought 3:** The shift from "thinking like humans" to "acting rationally" is the single most consequential conceptual move in AI history. It freed AI from the prison of human cognition.

> **Thought 4:** Every AI winter was caused by over-promising and under-delivering — a pattern still relevant today with LLMs. Calibrated expectations matter.

> **Thought 5:** The founding assumption of AI — that intelligence is computable — is still an open philosophical question. But pragmatically, we don't need to answer it to build useful AI systems.

---

*Next: [Part B — Good Old AI Days (Symbolic AI)](Part-B_Good_Old_AI_Days.md)*
