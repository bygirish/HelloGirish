# Section 7 — Synthesis, Takeaways and System Design

> **Where this fits:** The closing slides (78–80), the professor's wrap-up (03:31), and the system-design questions students raised in the live Q&A. Read this last. It links every section into one decision framework.

---

## 7.A — Recap: application → model → prompting strategy

| Application | Primary model | Prompting strategy | Signature failure |
|---|---|---|---|
| **Image understanding** | LLaVA-style / CLIP | Zero-shot, **scoped and length-constrained**, fixed template | Hallucinated objects, over-generic captions |
| **Visual Q&A** | LLaVA-style / Qwen-VL | Zero-shot, or **CoT** for counting/comparison | **Presupposition**, overconfidence, wrong focus |
| **Document intelligence** | **Qwen-VL** | **Structured output with an explicit JSON schema** | **Hallucinated fields**, layout variability, digit errors |
| **Visual reasoning** | LLaVA-style / Qwen-VL | **CoT**, decomposed into explicit steps, + self-check | **Compounding errors**, false confidence |
| **Chart & data QA** | Qwen-VL / high-res LLaVA | **Extract values first, then reason**, + honesty about uncertainty | Misread values presented as exact |
| *(Segmentation)* | SAM | Point / box / mask prompts | No class label. Thin and tiny objects |

---

## 7.B — The four key takeaways (slide 79)

1. **Every application maps back to the same four architectures from Lecture 1.** The skill is **matching the shape of the task to the model paradigm**. *(Section 1)*
2. **SAM segments. It doesn't recognise, read, or reason.** It's a **component you combine with others**, not a standalone answer. *(Section 0.I)*
3. **Prompting technique is task-specific:** scope constraints for captioning, chain-of-thought for reasoning and counting, structured schemas for extraction. *(Sections 2–5)*
4. **Every task type has a known failure mode. Plan for it before you deploy, not after.** *(7.C)*

And the professor's own wrap-up:
- **ViT is the main enabler** of this whole evolution, SAM included.
- **Prompts vary, metrics vary, and sometimes fine-tuning is needed.** In high-stakes settings (e.g. medical captioning, "you do not want to be wrong") you may have to fine-tune, but "fine-tuning is nothing but instruction fine-tuning or RLHF-type methods, which you already know".
- **Skill ≠ task ≠ dataset ≠ metric.** Every hop simplifies, which is why leaderboards say "solved" while researchers say it isn't. *(Section 1.E)*
- *"The success of these lectures would be that you are **asking more questions**, going away with more questions than simple solutions."*

---

## 7.C — Failure mode → mitigation cheat-sheet

| Failure mode | Where it appears | First mitigation | Stronger mitigation |
|---|---|---|---|
| **Object hallucination** | Captioning, VQA | "Only what is visually present" | Preference tuning (HALVA-style), post-hoc object verification |
| **Language-prior dominance** (not looking) | VQA, captioning | Ask for grounded evidence | Image-perturbation preference data. Counterfactual testing (blur / remove the object) |
| **Presupposition** | VQA | Allow "not present" | Two-step existence check. Train on false premises |
| **Overconfidence** | All | Self-consistency agreement as confidence | Calibration research. Human review for low agreement |
| **Miscounting** (> 5–10) | VQA, reasoning | CoT enumeration by region | Detector or SAM-based counting. Tiling |
| **Hallucinated fields** | Documents | `null` / "NOT FOUND" guardrail | Schema validation. RAG for context |
| **Digit / OCR errors** | Documents, charts | Higher resolution, crop, no downscaling | Exact-match eval per field. Human review on money, IDs, dates |
| **Layout variability** | Documents | Explicit schema | Domain fine-tuning (extraction as QA), layout-aware encoders |
| **Compounding errors** | Reasoning | Decompose + self-verify | Shorter chains, tools for sub-steps (code for arithmetic) |
| **Metric mismatch** | Evaluation | Know what the metric rewards | Your own labelled set. Human or LLM-judge pairwise evaluation |
| **Bias / long tail** | Captioning, VQA | Audit outputs by group | Diverse fine-tuning data. Multicultural benchmarks |

### 🔬 You've seen every one of these fail, in these notes' own runs
| Failure mode | Evidence (real output) | Where |
|---|---|---|
| Object hallucination | Dog + bird photo captioned as *"The two dogs are sitting on the ground."* | 2.D, 2.E |
| Over-generic output | The strict catalog prompt produced *"This is a photograph."* | 2.D |
| Presupposition | *"What color is the cat?"* → *"The cat is black and white"* (no cat). The guarded version failed too | 3.F |
| Misperception with high confidence | P(cat present) = **0.976**, P(real bird present) = 0.066 | 3.B |
| Language-prior tokens | "to", "ble" and "." barely change when the image is blurred. "dog" drops 0.68 → 0.00 | 3.E |
| Consistent ≠ correct | "8 people", 6/7 votes agree. About 9 are visible | 3.D |
| Metric blind spots | BLEU 8.75 for a correct paraphrase. CLIPScore 0.83 for the invented "owner" caption. ANLS 0.92 for the wrong decade | 2.F, 4.H |
| Secondary objects vanish | CLIP: "satellite dish" cosine 0.194, the same as "a dog" | 2.C |
| Format ignored | Invoice JSON with `"total": "$481.50"`, rejected by pydantic | 4.D |
| Right answer, wrong reading | Chart: direct "Q3" ✓, but the extraction misread every bar and missed Q4 | 5.F |
| CoT can't fix perception | Helmet check: CoT listed one rider, missed the helmetless passenger | 5.C |
| Ambiguous prompt | One SAM click → dog + pebbles / body / fur patch | 0.F |
| Embedding match ≠ coverage | Requirement matcher reported 20% coverage when the truth is 80% | 7.D |

Most of these came from a 256M-parameter model, which **exaggerates** the failures so you can see them. Larger models fail **less often, in the same ways**, which is why the mitigations column still applies to GPT-4-class systems.

---

## 7.D — Composing models: system-design questions from the live session

These came from students and show the lecture's ideas applied to real problems.

### Design 1: detect **and** precisely segment a wet signature on a document
```
Document image ─► VLM (Qwen-VL): "Is there a handwritten signature? Return its bounding box."
                    │ box
                    ▼
                  SAM (box prompt) ─► pixel-precise signature mask
```
The VLM supplies **what and where** (recognition + grounding). SAM supplies **the exact pixels**. Neither can do both. *(Grounded-SAM pattern, Section 0.I)*

💻 **This is already runnable:** [`code/s0_grounded_sam.py`](code/s0_grounded_sam.py) chains OWLv2 → SAM. Change the query to `"a photo of a handwritten signature"` and point it at a scanned form. Swap OWLv2 for Qwen2.5-VL grounding (it outputs box coordinates as text) if you need to handle complex document instructions.

### Design 2: map design documents to verification documents and measure coverage
*The student's problem: design docs contain tables, block diagrams and **waveforms**, and they need to know how much of the design each verification requirement covers.*

**Treat it as two problems: extraction first, then matching.**

| Step | What | Tool |
|---|---|---|
| 1 | Collect design and verification docs, and pick a small **representative development set** | — |
| 2 | Extract text, tables, captions, page structure | OCR for plain text. **Document VLM** for diagrams, waveforms, complex tables ("plain OCR will destroy them") |
| 3 | Convert both sides to one **structured JSON format**: doc ID, section, requirement ID, text, figure/table context, page number | VLM with a **named JSON schema per page** (Section 4.D). Waveforms and diagrams → **caption + surrounding text**, not the picture itself |
| 4 | Split into **atomic** design requirements and atomic verification statements | Don't embed whole pages as one chunk |
| 5 | Embed both sets and **retrieve** the top candidate verification statements for each design requirement | **Do not use a VLM here.** Once both sides are text, this is semantic retrieval |
| 6 | **Rerank**, then an **LLM judge** labels each pair *covered / partially covered / not covered*, with a reason and source references | LLM-as-judge |
| 7 | Build a **coverage table**: requirement, matched item, score, label, explanation, page references | Coverage % = share of design statements with ≥ 1 accepted match |
| 8 | **Manually label a few hundred pairs** as the eval set. Measure precision, recall, false matches **before tuning thresholds** | "Otherwise you cannot tell if the pipeline is working." Also keep a list of domain **synonyms** |
| 9 | **Human review** for low-confidence or safety-critical matches | — |
| 10 | Expose it through a dashboard so users can trace every match back to the source | — |

### 💻 Code: steps 5 and 8, run for real
[`code/s7_coverage_matcher.py`](code/s7_coverage_matcher.py) uses 5 hardware-style design requirements (a FIFO spec) and 4 verification statements. The **truth** is DR-1↔VR-10, DR-2↔VR-11, DR-3↔VR-12, DR-4↔VR-13, and DR-5 (parity interrupt) **not covered**, so true coverage is **80%**.
```python
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
D = model.encode(design, normalize_embeddings=True)
V = model.encode(verification, normalize_embeddings=True)
sim = util.cos_sim(D, V)                                     # every requirement vs every verification item
for i, req in enumerate(design):
    j = int(sim[i].argmax())
    print(req[:5], "->", verification[j][:6], f"sim={float(sim[i, j]):.2f}")
```
```
DR-1: -> VR-12:  sim=0.42  NOT COVERED        <- wrong: should be VR-10
DR-2: -> VR-11:  sim=0.40  NOT COVERED        <- right item, below threshold
DR-3: -> VR-12:  sim=0.47  COVERED (candidate)
DR-4: -> VR-13:  sim=0.37  NOT COVERED
DR-5: -> VR-13:  sim=0.24  NOT COVERED        <- correct: truly uncovered
coverage = 1/5 = 20%  (before LLM-judge confirmation)

true coverage = 4/5
threshold 0.30: correct=3 wrong-match=1 missed=0  precision=0.75 recall=0.75
threshold 0.35: correct=3 wrong-match=1 missed=0  precision=0.75 recall=0.75
threshold 0.45: correct=1 wrong-match=0 missed=3  precision=1.00 recall=0.25
```
**This is exactly why step 8 exists.** A threshold picked by eye (0.45) reported **20% coverage when the truth is 80%**. A general-purpose embedding model doesn't know that "full flag" = "full goes high" or "wrap around" = "rolls over". Even the best threshold gets DR-1 wrong. The fixes are steps 6–8 of the table: a **reranker** (a cross-encoder reading both texts together), an **LLM judge**, a **domain synonym list**, and thresholds **tuned on labelled pairs**. And never report a coverage number you haven't checked against labels.

The professor's caveats: abstract or multi-step specifications ("Q1 result should exceed Q2 + Q4") need multi-step reasoning, where open-source accuracy "is not that good". It needs "more layers of engineering", and tool-based verification is what the professor's group uses. If fine-tuning is needed, it's the same recipe as before: (image + question, preferred answer, perturbed dispreferred answer) → DPO/RLHF. *(Section 3.I)*

> 💡 **Learning Thought:** Look at **where the VLM stops**. VLMs handle **pixels → structure**. After that, cheaper and more controllable text tools (embeddings, rerankers, LLM judges) take over. A good multimodal system uses the VLM only where pixels have to become meaning.

### Design 3: videos
Frames are images, so everything applies. Add **temporal attention** (SAM 2's memory bank) and **manage tokens**: sample at **0.5–2 fps** or a fixed **8–32 frames**, drop redundant frames, and use 3D tokenisation or pooling. *(Section 0.I trap)*

---

## 7.E — Pre-read discussion questions, answered

**1. For a task you care about, is it closer to captioning, VQA, document extraction or reasoning, and does that change your model choice?**
> Yes, and the output shape decides. Tags → CLIP. A description → generative VLM. Answers about text or regions → Qwen-VL. Structured fields → Qwen-VL with a schema. Multi-step inference → the strongest LLM backbone. Exact outlines → SAM combined with a VLM. Many real tasks are **mixtures** (charts = documents + images, signature = grounding + segmentation). Break them into parts and route each.

**2. Document intelligence gained hugely from resolution alone. Where else might "just increase resolution" be underrated?**
> Anywhere the signal is in **small details**: charts (tick labels, thin lines), UI screenshots and web agents (small buttons and text, which is DAVE's Mind2Web gain), counting small objects, medical imaging (tiny lesions), satellite and drone imagery, licence plates (the LLaVA "KASI EA 5030" misread). The cost is quadratic attention over more tokens, so crop or tile where you can.

**3. A benchmark score looks great but doesn't match your results. What do you check first?**
> Walk down skill → task → dataset → metric (Section 1.E). Is the **metric** forgiving in ways you can't afford (ANLS partial credit, VQA exact match penalising verbosity)? Does the **dataset** resemble your inputs (layouts, scan quality, languages, long-tail objects)? Could it be **contamination**? And **your preprocessing**: are you feeding the resolution the benchmark used? (See the `thumbnail()` bug in Section 6.G.)

**4. SAM knows boundaries, not objects. Where could it still help in today's four applications?**
> **Captioning / VQA:** mask or crop to the region so the VLM focuses on it (contrastive region guidance for small objects, Section 0.I). **Counting:** count SAM masks instead of trusting the VLM's number. **Documents:** isolate signatures, stamps, figures or table regions before OCR, and redact sensitive regions. **Reasoning:** give precise regions and areas for spatial questions ("does the box fit?"). In every case SAM adds **where**, and the VLM adds **what** and **why**.

---

## 🎯 Capstone Interview Questions

**Q1. "Design a multimodal assistant for an insurance claims team: photos of damage + claim forms + repair invoices."**
> Route by input type. **Claim forms and invoices** → Qwen-VL with strict JSON schemas (null guardrails, ISO dates, numeric totals), arithmetic validation, exact-match evaluation per field, and human review over a value threshold. **Damage photos** → a VLM captions and answers targeted questions ("Which panels are damaged?") with CoT that lists evidence, SAM masks damaged regions if you need area estimates, and counterfactual checks during evaluation for language-prior answers. **Cross-document reasoning** (does the invoice match the damage?) → an LLM over the extracted structured data plus policy documents from RAG. Plan for the known failures: hallucinated fields, digit errors, presupposition ("describe the scratch on the door" when there's none), overconfidence. Keep humans in the loop for payouts.

**Q2. "Why not use one giant VLM for everything?"**
> You can prototype that way, but: (1) paradigms differ in output (score/text/box/mask), and SAM-level pixel precision or CLIP-level cost aren't available from a chat VLM; (2) every task has its own failure mode, which needs a task-specific prompt, validation and metric; (3) cost and latency: CLIP tagging or dedicated OCR is orders of magnitude cheaper for bulk work; (4) the research frontier (DAVE) itself specialises and then merges. Use the big VLM where pixels have to become meaning, and cheaper, more controllable components elsewhere.

**Q3. "What single idea from this lecture would you apply to any AI system?"**
> Skill ≠ task ≠ dataset ≠ metric. Every hop simplifies, so benchmark scores overstate how solved something is. Build your own evaluation set on your own data with a metric that matches your costs (exact match for money, per-object checks for captions), and test counterfactually whether the model uses the evidence you think it uses.

---

## 📚 Where to go next (all links checked)
| If you want… | Read / do |
|---|---|
| The one overview of the model family | [HF blog: Vision Language Models Explained](https://huggingface.co/blog/vlms) |
| To run a VLM on your laptop | [SmolVLM blog](https://huggingface.co/blog/smolvlm) + [`code/vlm_small.py`](code/vlm_small.py) |
| To fine-tune a VLM | [TRL: training VLMs with SFT](https://huggingface.co/docs/trl/main/en/training_vlm_sft) → [TRL DPOTrainer](https://huggingface.co/docs/trl/dpo_trainer) |
| Segmentation in practice | [segment-anything repo notebooks](https://github.com/facebookresearch/segment-anything) → [SAM 2](https://github.com/facebookresearch/sam2) → [Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) |
| The hallucination research thread | [VQAv2 "Making the V matter"](https://arxiv.org/abs/1612.00837) → [HALVA](https://arxiv.org/abs/2405.18654) |
| Documents | [Qwen2-VL dynamic resolution](https://arxiv.org/abs/2409.12191) → [DocVLM](https://arxiv.org/abs/2412.08746) → [DAVE](https://arxiv.org/abs/2512.17221) |

## 📚 Resources mentioned
- **Book:** O'Reilly, *Vision Language Models* (recommended in the live Q&A as a practical reference)
- **Papers named in the lecture:** SAM (Kirillov et al., 2023) · SAM 2 · Grounded-SAM · Show & Tell / Show, Attend & Tell · CapPa ("Image Captioners Are Scalable Vision Learners Too") · BLIP / BLIP-2 · HALVA · COMPOSITE / SPICE · CapArena (2025) · NoCaps (ICCV 2019) · VQAv2 · GQA · DAVE (Huang et al., ICLR 2026) · DocVLM
- **Next week:** multimodal RAG, for when answers need knowledge beyond the pixels and the model's parameters

> 💬 *From the live Q&A:* "Too much theory; do we need all this?" The answer: the theory explains **why** models and losses behave as they do, but **you don't need to memorise every derivation**. Focus on the core ideas and each component's role, and the notebook will make them stick. If the pre-reads feel like too much, read the **key concepts, summaries and examples** first, then the notebooks.
