# Session 2.2 — Learning Notes Index

**Prompt Optimization and Security**  
IIT Kharagpur × upGrad | Instructor: Koustav Rudra | Date: 30-05-2026

---

## Session Big Picture

> Prompt engineering matures when we stop treating prompts as magic strings and treat them as **engineered systems**: hypotheses, versioning, regression tests, automated evaluation, and gates.

---

## All Topics at a Glance

| # | File | Topics Covered | Core Takeaway |
|---|---|---|---|
| 1 | [01_Prompt_Optimization_Intro_and_Workflow.md](01_Prompt_Optimization_Intro_and_Workflow.md) | What is prompt optimization, engineering workflow, versioning, objectives | Prompts are executable specs — version them like code |
| 2 | [02_Manual_Prompt_Optimization.md](02_Manual_Prompt_Optimization.md) | Instruction refinement, iterative refinement, few-shot, CoT, chaining, role-play | Showing is more precise than telling; one change at a time |
| 3 | [03_Automatic_Prompt_Optimization.md](03_Automatic_Prompt_Optimization.md) | APE, ProTeGi, Evolutionary, RL-based, DSPy | Treat prompt search as an optimization problem with a defined objective |
| 4 | [04_Verbalized_Sampling.md](04_Verbalized_Sampling.md) | Mode collapse, typicality bias, verbalized sampling technique | RLHF makes models safe but boring; verbalized sampling unlocks diversity |
| 5 | [05_LLM_Evaluation.md](05_LLM_Evaluation.md) | BLEU, ROUGE, METEOR, BERTScore, NLI, G-Eval, LLM-as-a-judge | Use the simplest scorer that captures your quality dimension |
| 6 | [06_Prompt_Compression.md](06_Prompt_Compression.md) | Extractive, summarization, token-level, LLMLingua, LLMLingua-2 | Most prompt tokens are redundant; 2× compression is usually safe |
| 7 | [07_Prompt_Security.md](07_Prompt_Security.md) | Prompt injection, jailbreaking, prompt leaking, backdoor attacks, defenses | No single defense suffices — defense-in-depth is the only viable posture |

---

## Key Formulas & Numbers to Remember

| Topic | Key Number |
|---|---|
| Verbalized Sampling diversity boost | **2–3×** over direct prompting |
| Base model diversity recovered by verbalized sampling | **66.8%** |
| RLHF diversity remaining (baseline) | **33.2%** |
| LLMLingua safe compression ratio | **2–5×** with minimal degradation |
| LLMLingua-2 inference speedup | **3–6×** |
| Human evaluation sample for validation | **5–10%** of eval set |

---

## The Interconnection Map

```
Prompt Optimization (Topic 1)
│
├──► Manual Optimization (Topic 2) ─────────────────────────┐
│    [human-driven: refine, few-shot, CoT, chain, persona]   │
│                                                             │
├──► Automatic Optimization (Topic 3) ───────────────────────┤
│    [APE, ProTeGi, Evolutionary, DSPy]                      │
│                                                             │
│    Both need evaluation to measure improvement             │
│                                    ↓                       │
├──► LLM Evaluation (Topic 5) ────────────────────────────── ┤
│    [BLEU, ROUGE, BERTScore, G-Eval]                        │
│                                                             │
├──► Verbalized Sampling (Topic 4) ──────────────────────────┤
│    [when optimization needs diverse outputs]               │
│                                                             │
├──► Prompt Compression (Topic 6) ───────────────────────────┤
│    [optimize for cost/latency after quality is achieved]   │
│                                                             │
└──► Prompt Security (Topic 7) ──────────────────────────────┘
     [orthogonal concern — applies to all systems in prod]
```

---

## Interview Question Themes by Topic

| Topic | Interview Focus Areas |
|---|---|
| Prompt Optimization | Engineering workflow, versioning, regression testing, objectives trade-offs |
| Manual Optimization | Few-shot selection, CoT failure modes, chaining design, persona specificity |
| APO | APE scoring mechanism, ProTeGi NL gradients, DSPy compilation model |
| Verbalized Sampling | Mode collapse causes, typicality bias, calibration, when NOT to use |
| LLM Evaluation | BLEU vs. ROUGE, BERTScore improvement, G-Eval steps, judge biases |
| Prompt Compression | Perplexity-based scoring, LLMLingua vs. LLMLingua-2, quality-compression curve |
| Prompt Security | Safety vs. security distinction, injection types, agentic amplification, defense-in-depth |

---

## Learning Sequence Recommendation

**For conceptual understanding (read in order):**
1 → 2 → 4 → 5 → 3 → 6 → 7

**For interview preparation (read in order):**
7 → 4 → 3 → 6 → 5 → 2 → 1

**For building a production system (read in order):**
1 → 5 → 2 → 6 → 3 → 4 → 7
