# Module 4 · Lecture 2 — Applications of VLM Architectures

> **Deck:** *Applications of VLM Architectures — Image Understanding · Visual Q&A · Document Intelligence · Visual Reasoning · Chart QA* (UpGrad-KGP Agentic AI Program, 3h session, 5 Sept)
> **What this lecture does:** Lecture 1 opened up the four architectures (CLIP, LLaVA, Qwen-VL, SAM). Lecture 2 turns that around. Given a real task, **which model do you pick, how do you prompt it, how good is it really, and how does it fail?**

---

## Sources mined for these notes

| File | What it contributed |
|---|---|
| `…VLM-Applications-Lecture-Revised-v2-pptx.pdf` | The 80-slide deck: topic structure, tables, quizzes |
| `transcript.text` | The professor's worked numbers (conv upsampling, Dice example), anecdotes, and the parts that go **beyond** the slides (Grad-CAM, majority voting, scene graphs, preference-data perturbations) |
| `…5th-Sep-QA-submission-ready.xlsx` | 40 live chat questions → the ⚠️ **Trap** blocks |
| `…VLM-Applications-PreRead-docx.pdf` | The four "deep dives", glossary, and discussion questions |
| `VLM_Applications_UpGrad_Tutorial_Week2_Qwen_Student_ver.ipynb` | The hands-on lab: Qwen2.5-VL-3B for captioning, chain-of-thought counting, DocVQA, and document understanding |
| `…Colab-notebook-5th-Sept.pdf` | Just the Colab link for the same notebook |

---

## All topics in this lecture, by section

The deck follows one repeating template for each application: **what it is → which model fits → how to prompt → worked example → pitfalls → why it's hard → datasets → how SOTA is trained → performance → remaining challenges → quiz.** The sections below group the topics by that template.

| # | File | Topics |
|---|---|---|
| 0 | [01-Section-0-SAM-Promptable-Segmentation.md](01-Section-0-SAM-Promptable-Segmentation.md) | **A** Why segmentation needed a foundation model · **B** The three components and the "heavy once, light many" design · **C** Image encoder (MAE-pretrained ViT-H/16) · **D** Prompt encoder (sparse / dense / text) · **E** Mask decoder and two-way attention, plus upsampling by convolution · **F** Ambiguity: 3 masks + IoU head · **G** Loss: Focal + Dice (+ MSE) · **H** Data engine and SA-1B · **I** Zero-shot transfer, Grounded-SAM, SAM 2, limitations |
| 1 | [02-Section-1-Architecture-Synthesis-and-Application-Map.md](02-Section-1-Architecture-Synthesis-and-Application-Map.md) | **A** Cross-model comparison: CLIP vs LLaVA vs Qwen-VL vs SAM · **B** ViT as the shared backbone (and shared weaknesses) · **C** The application map: task → model · **D** The 11-step template for analysing any application · **E** Skill → Task → Dataset → Metric: where "solved" claims go wrong |
| 2 | [03-Section-2-Image-Understanding-and-Captioning.md](03-Section-2-Image-Understanding-and-Captioning.md) | **A** What captioning is · **B** History: Show & Tell → Show, Attend & Tell → CapPa → VLMs · **C** Generative vs CLIP · **D** Prompting for consistent captions · **E** Pitfalls and the 3-model motorcycle comparison · **F** Why it's hard, and caption metrics · **G** Datasets and CapArena · **H** Training, performance, open challenges |
| 3 | [04-Section-3-Visual-Question-Answering.md](04-Section-3-Visual-Question-Answering.md) | **A** What VQA is, and how it grew out of captioning and scene graphs · **B** The classic 2000-class VQA model · **C** LLaVA vs Qwen-VL · **D** Prompting: zero-shot, CoT, few-shot, majority voting · **E** Are VLMs even looking? (Grad-CAM, blurred-image test) · **F** Presupposition, overconfidence, ambiguity · **G** Why it's hard · **H** Datasets · **I** Training and preference data from perturbations · **J** Metrics, performance, open challenges |
| 4 | [05-Section-4-Document-Intelligence.md](05-Section-4-Document-Intelligence.md) | **A** What it is · **B** VLM vs OCR pipeline (and why OCR-then-LLM loses layout) · **C** The resolution problem, with numbers · **D** Prompting with an explicit JSON schema · **E** Pitfalls and why it's hard · **F** Case study: DAVE (ICLR 2026) · **G** Case study: DocVLM · **H** Datasets and ANLS · **I** Training and performance · **J** Open challenges |
| 5 | [06-Section-5-Visual-Reasoning-and-Chart-QA.md](06-Section-5-Visual-Reasoning-and-Chart-QA.md) | **A** What visual reasoning is · **B** Why the LLM backbone decides it · **C** CoT and self-verification · **D** Pitfalls and why it's hard · **E** Datasets, training, performance · **F** Chart & Data QA *(visual reasoning was optional and **skipped live**; chart QA appears only in the map and recap)* |
| 6 | [07-Section-6-Hands-On-Lab-Qwen2.5-VL.md](07-Section-6-Hands-On-Lab-Qwen2.5-VL.md) | **A** Setup and the pixel budget · **B** Anatomy of `generate()` · **C** Captioning · **D** CoT counting · **E** DocVQA · **F** Document understanding · **G** Notebook bugs and gotchas · **H** Exercises |
| 7 | [08-Section-7-Synthesis-and-Takeaways.md](08-Section-7-Synthesis-and-Takeaways.md) | **A** Recap: application → model → prompting · **B** The four key takeaways · **C** Failure mode → mitigation cheat-sheet · **D** Composing models (system-design Q&As from class) · **E** Pre-read discussion questions, answered |

### What was actually covered live
- **Covered in depth:** SAM (~40 min, more than planned), Captioning, VQA (~40 min), Document Intelligence, Hands-on.
- **Skipped or optional:** Visual Reasoning ("I kept visual reasoning completely optional … we did not cover"), DocVLM ("we do not want to go to"), Chart QA. Section 5 is built from the slides and pre-read only.

---

## How to use these notes

1. **Pass 1 (understanding):** read sections 0→7 in order, topics A→J. Skip the 🎯 interview blocks.
2. **Pass 2 (expertise):** do the ✅ quizzes cold, then the 🎯 interview questions out loud, then run the notebook alongside Section 6.
3. ⚠️ **Trap** blocks come from real confusions in the live Q&A. Read them before interviews.
4. 🔬 **Beyond the slides** marks details taken from the original papers rather than the lecture. They are there for depth and are not needed for the quiz.
5. 🖼️ **Figures** come from the slide deck ([`assets/`](assets/), 29 extracted figures) or were **generated from real model runs** (the SAM 3-mask demo, the document legibility strip, the revenue chart).
6. 💻 **Code**: every topic has a runnable example in [`code/`](code/) (see [code/README.md](code/README.md)). The outputs pasted in the notes are **real**, from a laptop CPU using CLIP, SAM, OWLv2 and SmolVLM-256M. Pass 2 = run them and change things.
7. 📚 **Go deeper** links at the end of each topic are papers, docs and repos. Every link was checked (HTTP 200), and every arXiv ID was matched to its paper title.

### Corrections that running the code forced
| Where | Earlier claim | What running it showed |
|---|---|---|
| 2.F | BLEU = 0 for a correct paraphrase | sacreBLEU (smoothed) gives **8.75**. The point still holds |
| 5.F | "Q3 is the robust answer" on the chart | With ±5 reading error, **Q3 is not guaranteed** (Q2 is within error) |
| 6.C | Dataset column `multiple_answers` | The real column is **`multiple_choice_answer`** (one answer per row) |
| 6.C | Notebook row 5 was "an indoor scene" | Row 5 is the lecture's **motorcycle + satellite dish** photo |

---

## The one-paragraph summary

> Every application in this lecture maps back to the same four architectures, and the real skill is **matching the shape of the task to the model paradigm**. Need a sentence? Use a generative VLM (LLaVA/Qwen). Need a tag? CLIP. Need to read text or point at a region? Qwen-VL. Need a pixel-exact outline? SAM, which segments but never names, reads, or reasons, so you combine it with a VLM. Prompting is task-specific: **scope and length constraints** for captioning, **chain-of-thought** for counting and reasoning, an **explicit JSON schema** for extraction, and **extract-then-reason plus honesty** for charts. Each task has a known failure mode: hallucination, presupposition, confident miscounts, invented fields, compounding errors. The lecture's research thread is that VLMs are often **overpowered by their language model**. They answer from priors instead of pixels ("make the V in VLM matter"), so plan for failure before you deploy, not after.
