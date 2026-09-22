# Module 4 · Lecture 3 — Multimodal RAG for QA, Text Search & Image Search

> **Deck:** *Multimodal RAG for QA, Text Search & Image Search: Grounding generation in retrieval, over text, images, and real-world documents* (UpGrad-KGP Agentic AI Program, 3h session, 12 Sept)
> **What this lecture does:** Lectures 1–2 were about *understanding* an image (CLIP, LLaVA, Qwen-VL, SAM, and their applications). Lecture 3 is about **finding the right content first**, then using it to ground an answer. The professor's framing: *"multimodal RAG is just a **delta** on top of text search and image search."* Learn the two searches well, and multimodal RAG is a small step further.

---

## Sources mined for these notes

| File | What it contributed |
|---|---|
| `…VLM-Multimodal-RAG-Lecture-pptx.pdf` | The 50-slide deck: topic structure, tables, the three quizzes |
| `transcript.text` | Material that goes **beyond** the slides: TF-IDF worked by hand, FAISS clustering intuition, the original RAG paper (MIPS, marginalisation, EM), ColBERT late interaction, abductive reasoning, the softplus loss, nDCG intuition, the multi-hop ViDoRe example, "jagged intelligence", Re-Imagen |
| `…12th-Sep-Q-A-1-.xlsx` | 35 live chat questions → the ⚠️ **Trap** blocks and the system-design Q&As |
| `…multimodal-rag-postread-1-.pptx` | 14-slide reference deck: the **MaxSim formula**, ColPali numbers (0.39 s vs 7.22 s, ~1024 patches, D=128), heatmap interpretability, index-storage cost, nDCG@5 values |
| `…VLM-Multimodal-RAG-PreRead-docx.pdf` | Plain-English framings, glossary, four discussion questions |
| `multimodal_rag_images_(1).ipynb` | Hands-on lab: EasyOCR + Florence-2 captions → MPNet + FAISS, ColPali MaxSim + heatmaps, **Reciprocal Rank Fusion**, SmolLM grounded generation |
| `…Colab-notebook-12th-Sept.pdf` | Only the Colab link for the same notebook |

> Slide equations for ColPali are **placeholders** in the lecture PDF ("EQUATION PLACEHOLDER"). The formulas in these notes come from the post-read deck, the transcript, and the ColPali paper (Faysse et al., ICLR 2025).

---

## All topics in this lecture

The deck has **three teaching blocks** plus a lab. Each block follows one template: *what the task is → which model fits → how it works → worked example → datasets and training → performance → remaining challenges → quiz.* The ten sections below regroup the topics into a learning order. The professor taught the RAG paper and ColBERT inside "Text Search", but they are the bridge to ColPali, so they get their own section here.

| # | File | Deck block | Topics |
|---|---|---|---|
| 0 | [01-Section-0-Why-Multimodal-RAG.md](01-Section-0-Why-Multimodal-RAG.md) | Overview | **A** What multimodal RAG is · **B** Four real-world motivating examples · **C** Text-only vs multimodal RAG (the "1975 value" chart) · **D** Why RAG at all: hallucination, stale knowledge, parametric vs non-parametric memory · **E** The end-to-end architecture map · **F** Why retrieval needs reasoning: abductive retrieval · **G** Roadmap and the "delta" framing |
| 1 | [02-Section-1-Text-Search.md](02-Section-1-Text-Search.md) | 01 Text Search | **A** What text search is and the unit of retrieval · **B** Keyword search: TF-IDF → BM25 (worked by hand) · **C** Semantic search and bi-encoder dense retrieval · **D** Hybrid search and reranking · **E** Scaling: exact search vs ANN (FAISS/IVF, HNSW, DiskANN, MIPS) · **F** Training: contrastive loss and hard-negative mining (worked) · **G** Benchmarks and metrics: MS MARCO, BEIR, MTEB, P/R/F1 · **H** Remaining challenges |
| 2 | [03-Section-2-RAG-Foundations-and-ColBERT.md](03-Section-2-RAG-Foundations-and-ColBERT.md) | 01 Text Search (transcript) + slides 28–29 | **A** The original RAG paper (Lewis et al., 2020) · **B** MIPS as the retriever · **C** Retrieval as a latent variable: marginalisation, worked · **D** End-to-end training and the EM-style view (RAG, REALM) · **E** ColBERT: contextualised late interaction · **F** MaxSim scoring worked by hand · **G** Bi-encoder vs late interaction vs cross-encoder |
| 3 | [04-Section-3-Image-Search.md](04-Section-3-Image-Search.md) | 02 Image Search | **A** Two directions: text→image, image→image · **B** History: tags and captions as a proxy · **C** CLIP recap: why a shared space is exactly what search needs · **D** Cross-modal retrieval pipeline · **E** Modern visual encoders: DINOv2/v3, SigLIP-2 · **F** Datasets, Recall@K, performance · **G** Remaining challenges: fine-grained, compositional, bias, long queries |
| 4 | [05-Section-4-Multimodal-RAG-Architecture-and-Retrieval-Strategies.md](05-Section-4-Multimodal-RAG-Architecture-and-Retrieval-Strategies.md) | 03 MM RAG (part 1) | **A** Multimodal RAG defined, and why documents need it · **B** Retriever + Generator: the math · **C** Offline indexing vs online querying; properties of a good retriever and reader · **D** Parse-then-embed vs retrieval in vision space · **E** Three storage/training designs: unified text, shared space, separate indexes · **F** Late interaction for documents · **G** Indexing and chunking multimodal documents |
| 5 | [06-Section-5-ColPali-Deep-Dive.md](06-Section-5-ColPali-Deep-Dive.md) | 03 MM RAG (deep dive) | **A** The standard PDF pipeline and its cost · **B** ColPali architecture: PaliGemma-3B, 1024 patches, D=128 · **C** Late-interaction score, worked · **D** Training loss: in-batch contrastive + softplus, worked · **E** Latency: 0.39 s vs 7.22 s per page · **F** Heatmap interpretability · **G** Storage cost, worked · **H** ViDoRe and nDCG@5, worked · **I** Performance vs BM25 and BGE-M3 · **J** How SOTA is trained, successors (ColQwen2, VisRAG) |
| 6 | [07-Section-6-Generation-Grounding-and-Pitfalls.md](07-Section-6-Generation-Grounding-and-Pitfalls.md) | 03 MM RAG (part 2) | **A** The generation step · **B** Grounded, citation-aware prompting and abstention · **C** Worked example: Q3 revenue chart · **D** Multi-hop example from ViDoRe (the S-3 refuelling receptacle) · **E** Four common pitfalls + "jagged intelligence" · **F** Why this is hard · **G** Remaining challenges · **H** When to use which retrieval strategy |
| 7 | [08-Section-7-Multimodal-RAG-Landscape.md](08-Section-7-Multimodal-RAG-Landscape.md) | 03 MM RAG (survey) | **A** Where multimodal RAG is used (+ Re-Imagen) · **B** Four retrieval modalities · **C** Fusing what's retrieved: score, attention, unified · **D** Reasoning over retrieved knowledge: ICL, CoT / Tree-of-RAG, instruction tuning, source attribution, agents · **E** How systems are trained: InfoNCE, generation loss, robustness training · **F** Training retriever and generator together |
| 8 | [09-Section-8-Hands-On-Lab.md](09-Section-8-Hands-On-Lab.md) | Hands-On Lab | **A** Pipeline and datasets (CIFAR-10 + DocVQA + InfographicVQA) · **B** EasyOCR + Florence-2 → combined text · **C** MPNet + FAISS `IndexFlatIP` · **D** PCA visualisation · **E** ColPali indexing and MaxSim in code · **F** Patch heatmaps · **G** Reciprocal Rank Fusion, worked · **H** Grounded generation with SmolLM · **I** Notebook bugs and gotchas · **J** Exercises |
| 9 | [10-Section-9-Synthesis-and-System-Design.md](10-Section-9-Synthesis-and-System-Design.md) | Synthesis + Q&A | **A** Recap: task → technique → key idea · **B** The four key takeaways · **C** Choosing a RAG flavour (text / table / hybrid / graph / multimodal) · **D** System-design Q&As from class (incremental ingestion, cost, offline RAG, ColPali vs PDF parser) · **E** Pre-read discussion questions, answered |

### What was actually covered live (from the transcript)
- **Session 1 (~50 min, longer than planned):** motivation examples → TF-IDF → dense retrieval → FAISS clustering → **the RAG paper and ColBERT** (not in the slides, taught from the whiteboard) → text-search challenges → image search (done quickly: *"once you understand text search, these are just small deltas"*).
- **Quiz + break.**
- **Session 2 (~65 min):** architecture → two retrieval strategies → storage strategies → late interaction → **ColPali in depth** (architecture, loss, latency) → indexing → generation and prompting → ViDoRe multi-hop example → pitfalls → nDCG → performance → challenges → when-to-use. The **landscape survey** (slides 23–29) was touched on only at the very end, mainly through Re-Imagen. Its ideas (fusion, reasoning, EM training) had already come up in the opening architecture walkthrough.
- **Hands-on lab** was the Colab notebook. The lab slide in the deck is a placeholder.

---

## What's inside each section (added in the revision pass)
- **🖼️ Slide figures** in [`figures/`](figures/): 19 slides rendered from the lecture PDF, embedded next to the topic they illustrate, with a caption explaining how to read each one.
- **Real-world examples**: food-delivery search, Google Lens, insurance PDFs, HR handbooks, e-commerce ranking…
- **ASCII diagrams** for mechanisms no slide shows: inverted index, IVF cells, HNSW layers, bi-encoder vs late interaction vs cross-encoder, page-patch MaxSim.
- **💻 Runnable code**: every snippet marked *"Actual output"* was run while writing these notes, and the printed numbers are real. Most run on a laptop CPU. The GPU-only ones (ColPali, VLM generation) are marked as such and follow the lab notebook or the official library docs.
- **📚 Go deeper** boxes: papers (arXiv), official docs, tutorials, leaderboards. All 72 links were checked live.

**Setting up to run the code:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy rank_bm25 sentence-transformers faiss-cpu open_clip_torch pillow pymupdf datasets accelerate
# GPU-only extras (Colab): pip install colpali-engine "sentence-transformers[image]>=6.0.0" qwen-vl-utils easyocr
```

> ⚠️ **Two facts found while verifying the code** (details in Sections 5 and 8):
> 1. The lab's heatmap bug is **confirmed**: ColPali's 1024 patch vectors are positions 0–1023, so `image_emb[-1024:]` is off by the 7 prompt tokens. Fix: `get_image_mask()`.
> 2. `colpali-engine` is now **deprecated** in favour of Sentence Transformers ≥ 6 (`MultiVectorEncoder`). The lab code still works; use the new API for new projects.

---

## How to use these notes

1. **Pass 1 (understanding):** read Sections 0→9 in order, topics A→J. Skip the 🎯 interview blocks.
2. **Pass 2 (expertise):** do the ✅ quizzes cold, then the 🎯 interview questions out loud, then run the notebook alongside Section 8.
3. ⚠️ **Trap** blocks come from real confusions in the live Q&A, plus two places where the professor's spoken wording was imprecise. Read them before interviews.
4. 🔬 **Beyond the slides** marks details taken from the original papers rather than the lecture. They are there for depth and are not needed for the quiz.

---

## The one-paragraph summary

> Text search, image search and multimodal document retrieval are **one pattern**: embed the corpus offline, embed the query online, and run nearest-neighbour search through an approximate index. **Only what gets embedded changes.** Keyword search (BM25) matches words. Dense bi-encoders match meaning. CLIP-style shared spaces let a text query find an image. For real documents, where the answer lives in a chart, table or layout, the traditional **parse-then-embed** pipeline (OCR → layout → chunk → embed) is slow (~7 s/page) and lossy. **ColPali** skips parsing: it embeds a screenshot of the page into ~1,000 patch vectors and scores pages with **ColBERT-style late interaction** (each query token takes its best-matching patch, and the scores are summed). That gives better retrieval (ViDoRe nDCG@5 ≈ 0.81 vs ≈ 0.66) at ~0.4 s/page, at the cost of a much bigger index. Retrieval and generation are **separate failure points**. A great VLM can't fix a missed page, and a well-retrieved page can still be ignored or hallucinated over. So prompt for **grounded, cited answers with explicit permission to abstain**, and verify the citation instead of trusting it.
