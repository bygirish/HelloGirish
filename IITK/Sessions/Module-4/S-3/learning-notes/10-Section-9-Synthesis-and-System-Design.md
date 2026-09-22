# Section 9 — Synthesis, Takeaways and System Design

> **Where this fits:** This section pulls the lecture into **one retrieval pattern** and **four takeaways**. It then turns the live Q&A (many of which were real system-design questions) into model answers you can reuse in interviews and at work. Read it last in Pass 1, and first when revising for an interview.
> **Slides:** 49–50 · **Post-read:** slide 14 · **Q&A sheet:** system-design questions

---

## 9.A — Recap: one pattern, three tasks

> *"The same embedding-and-nearest-neighbour pattern powers text search, image search, and document retrieval. Only what's embedded changes."*

| Task | Core technique | What gets embedded | Training signal | Primary metric | Key idea |
|---|---|---|---|---|---|
| **Text search** | Bi-encoder + ANN index (+ BM25 hybrid) | Text passages | Contrastive + hard negatives | MTEB (~74 avg), nDCG@10, Recall@K | Semantic match beats exact keyword match |
| **Image search** | CLIP-style shared embedding space | Images + text | Contrastive (image, caption) pairs | Recall@1/5/10 (Flickr30K, COCO) | Text and image queries compared directly |
| **Multimodal RAG for QA** | Vision-space retrieval (ColPali) + VLM generation | Page-image **patches** | In-batch contrastive on (query, page) | ViDoRe **nDCG@5** | Ground answers in retrieved, real document content |

### The whole lecture as one diagram
```
          OFFLINE                                   ONLINE
 corpus ─► encoder ─► vectors ─► ANN index      query ─► encoder ─► search ─► rerank/fuse
   │  text: bi-encoder / BM25 inverted index              │
   │  image: CLIP/SigLIP/DINO                             ▼
   │  page: ColPali (1 vector per patch)          top-k evidence ─► reader (LLM/VLM)
   │                                                        grounded prompt + citations
   └─ metadata: doc, page, region, version                  + abstention + verification
```

---

## 9.B — The four key takeaways, expanded

**1. Retrieval and generation are two separate skills.**
*"A great generator with poor retrieval still gives poor answers."* And (post-read) *"a strong retriever with an unconstrained prompt still hallucinates."* They are **independent failure modes**: evaluate each on its own (recall/nDCG for retrieval; accuracy given the gold page for reading), and fix the weaker one first (Section 6.F's compounding: 0.80 × 0.85 = 0.68).

**2. Only what's embedded changes.**
Text search, image search and document retrieval share the same infrastructure: encoder → vectors → ANN (FAISS/HNSW) → nearest neighbours → rerank. Learn the pattern once and every new modality (audio, video, 3-D, code) is a **delta**: *which encoder, and which retrieval unit?*

**3. Retrieval in vision space is replacing brittle OCR pipelines where documents are visually rich.**
ColPali beats OCR + text embeddings, and even OCR + VLM-captioning pipelines, on ViDoRe (≈0.81 vs ≈0.66 nDCG@5), with ~18× faster indexing (0.39 s vs 7.22 s/page). The price is a ~1000× larger index. *"Wherever documents are visually rich"* is the qualifier: for clean text it's overkill.

**4. Grounding reduces hallucination, but doesn't eliminate it. Verify; don't just trust a cited source.**
Models are "jagged". They may ignore evidence, be misled by distractors, or cite pages they didn't use. Grounded prompts + citations + abstention + **automated verification** are all needed, and even then *"verification against the source is the reader's responsibility, not just the system's."*

> 💡 **Learning Thought:** Each takeaway is really about **where information gets lost**: in the index (chunking, OCR), in the ranking (pooled vectors, missing rank-awareness), in the handoff (irrelevant context), or in the reader (hallucination). A good RAG engineer follows the information from source pixel to final token and asks at every step: *could the answer still be recovered from here?*

---

## 9.C — Choosing a RAG flavour

From the Q&A: *"In the RAG lecture we saw approaches for embedding documents with charts and tables, but here we suggest multimodal. Which is best?"* **There is no single best approach. Match the retrieval strategy to the data and the question type.**

| Your data / questions | Use | Why |
|---|---|---|
| Narrative text; conceptual questions | **Dense text RAG** | Semantic meaning matters most |
| Exact terms, IDs, codes, names mixed with concepts | **Hybrid RAG (BM25 + dense, RRF)** | Keyword precision + semantic recall |
| Exact values, calculations, structured records | **Table / structured RAG** (text-to-SQL, table QA) | Numbers must be exact and computed, not "read approximately" |
| Charts, scans, complex layouts, spatial relationships | **Multimodal RAG** (vision-space retrieval + VLM) | The evidence is visual/spatial |
| Multi-hop questions across linked entities | **Graph RAG** | Explicit relations enable pointer-following |
| Photos/products with simple visual queries | **CLIP-style image search** | Simplest thing that works |

**Production reality:** a **hybrid**. *"Use traditional parsing for text and tables, and multimodal retrieval for visually complex pages."* Route pages by type at indexing time (text-heavy → parse + text embed; chart/scan-heavy → ColPali), and fuse at query time.

---

## 9.D — System-design Q&As from class, as model answers

### D1. Keeping the index fresh: incremental ingestion
*"If documents are updated daily or weekly, how do we design a system that automatically detects new information and processes (chunk/embed) it into the vector DB?"*

```
Sources ─► Change detector ─► Queue ─► Workers: extract → (OCR/layout | page render) → chunk → embed
           (updatedAt scan,                                                     │
            content hash,                                                       ▼
            webhooks/events)                          Vector DB UPSERT by (doc_id, version, chunk_id)
                                                      + DELETE / deactivate stale chunks
```
- **Detect changes:** event-driven (webhooks, CDC) for near-real-time, or **scheduled polling** (cron) scanning an `updatedAt` column or comparing **content hashes**. Only process new or modified documents.
- **Idempotent upserts** keyed by `doc_id + chunk_id`, with a **version** field, so a retried job doesn't duplicate chunks.
- **Deletes and updates:** remove or deactivate the **old chunks** for that document, or you'll retrieve stale facts next to new ones.
- **Re-embed on model change:** a new embedding model means a **full re-index** into a new collection, then an atomic alias swap (Section 2.D).
- **ANN maintenance:** IVF centroids drift as data changes, so re-train periodically (the dynamic-clustering trap, Section 1.E).

### Code: incremental sync by content hash (add / update / delete, plain Python)
```python
import hashlib

class IncrementalIndex:
    """Keeps a vector index in sync with a changing document store, re-embedding only what changed."""
    def __init__(self, embed):              # embed: text -> vector (any model from Sections 1/5)
        self.embed, self.vectors, self.manifest = embed, {}, {}     # manifest: doc_id -> {hash, chunk_ids}

    def sync(self, docs):                   # docs: {doc_id: full_text}, the current state of the source
        stats = {"added": 0, "updated": 0, "deleted": 0, "unchanged": 0}
        for doc_id, text in docs.items():
            h = hashlib.sha256(text.encode()).hexdigest()
            old = self.manifest.get(doc_id)
            if old and old["hash"] == h:
                stats["unchanged"] += 1; continue
            if old:                                                # changed → remove stale chunks FIRST
                for cid in old["chunk_ids"]: self.vectors.pop(cid, None)
            chunks = [c for c in text.split("\n\n") if c.strip()]
            cids = [f"{doc_id}#{i}" for i in range(len(chunks))]
            for cid, c in zip(cids, chunks): self.vectors[cid] = self.embed(c)     # idempotent upsert
            self.manifest[doc_id] = {"hash": h, "chunk_ids": cids}
            stats["updated" if old else "added"] += 1
        for doc_id in set(self.manifest) - set(docs):             # deleted at source → delete from index
            for cid in self.manifest.pop(doc_id)["chunk_ids"]: self.vectors.pop(cid, None)
            stats["deleted"] += 1
        return stats

idx = IncrementalIndex(embed=lambda t: [len(t)])              # dummy embedder for the demo
print(idx.sync({"policy": "Leave: 12 days.\n\nTravel: economy.", "faq": "Q: VPN? A: use Okta."}))
print(idx.sync({"policy": "Leave: 15 days.\n\nTravel: economy.", "faq": "Q: VPN? A: use Okta."}))
print(idx.sync({"policy": "Leave: 15 days.\n\nTravel: economy."}))
```
Actual output:
```
{'added': 2, 'updated': 0, 'deleted': 0, 'unchanged': 0}
{'added': 0, 'updated': 1, 'deleted': 0, 'unchanged': 1}
{'added': 0, 'updated': 0, 'deleted': 1, 'unchanged': 1}
```
Run 2 re-embeds **only** the policy document (leave changed from 12 to 15 days). Run 3 removes the FAQ's vectors because the source deleted it. In production, `sync` is triggered by a cron job or a webhook, `self.vectors` is your vector DB (Qdrant, pgvector, Milvus…), and `manifest` lives in a small SQL table.

### D2. Reducing cost
*"Does a vision LLM increase cost? Is there any way to reduce it?"* Yes. A VLM processes image tokens (often hundreds to thousands per page) on top of text. Levers:
- **Caching:** query-embedding cache, answer cache for repeated questions, KV/prompt caching of shared system prompts
- **LLM routing:** send text-answerable questions to a cheap text LLM and only chart/layout questions to a VLM
- **Crop, don't send pages:** use the MaxSim heatmap to crop the relevant region (Section 6.A)
- **Smaller or open models:** fine-tune a small open-source reader (a one-time research cost)
- **Index-side:** token pooling / quantisation for ColPali (Section 5.G)

**Code: a query router, the single biggest cost lever**
```python
VISUAL_HINTS = ("chart", "graph", "figure", "diagram", "table", "trend", "plot", "image", "scan")
def route(question, retrieved_pages):
    needs_vision = any(w in question.lower() for w in VISUAL_HINTS) or \
                   any(p.get("has_visuals") and p.get("is_top1") for p in retrieved_pages)
    return "vlm" if needs_vision else "text_llm"     # text LLM: ~10× cheaper per call, no image tokens
print(route("What does the revenue chart show for Q3?", []))   # vlm
print(route("What is the notice period?", []))                 # text_llm
```
(`has_visuals` comes from the page chunker in Section 4.G. A production router is often a tiny classifier trained on logged queries.)

### D3. ColPali vs a PDF parser for a yearly mutual-fund document pipeline
*"PDFs arrive once a year and we must extract data points into an application. Will ColPali be costlier than a PDF parser?"*
> **Yes, generally.** ColPali processes each page as an image and stores ~1000 vectors per page, and it's a **retriever**, not an extractor. For a **run-the-business extraction** job (known fields → database) on low-volume yearly PDFs, prefer a **PDF parser + table extractor** (cheap, exact strings), optionally with an LLM **reviewer/validator** layer. Use ColPali (or a VLM) only for pages where the data depends on complex layouts, charts or scans. The task is **extraction with known targets**, not open-ended **search**.

### D4. An AI reviewer layer over the parser
*"Over a PDF parser, can we put an AI reviewer layer for higher accuracy?"* Yes:
`PDF → Parser → AI reviewer (detect OCR errors, missing tables/figures, bad reading order; compare against a page-image VLM read) → correct or flag for a human → chunk → embed → vector DB`. Quality problems get caught **before** they are frozen into the index, which is cheaper than debugging them at retrieval time.

### D5. Offline / on-prem multimodal RAG
*"Can multimodal RAG run offline?"* Yes, if **every component is local**: the embedding model (ColPali/CLIP/MPNet), the vector index (FAISS on disk), and the reader (an open VLM). Query-time retrieval is just local nearest-neighbour search. The constraint is **hardware**: VLMs need a GPU with enough VRAM.

### D6. A cost-effective multimodal QA system for a small, mixed corpus
*"PDFs, Word, Excel, Visio, images, diagrams, tables, audio: what do I need, including GPU?"*
- **Ingestion:** per-format extraction (PDF/DOCX/PPTX/XLSX text + tables; Visio/diagrams/images → page renders; **audio → Whisper transcripts** with timestamps) → OCR/layout where needed → chunk → embed (text embedder + ColPali for visual pages) → vector DB with rich metadata
- **Query:** text/multimodal retrieval → rerank → top-k evidence → a small VLM/LLM → grounded answer + citations
- **Hardware (starting point):** **1 GPU with 16–24 GB VRAM** for embedding and a small quantised VLM; **32–64 GB RAM + SSD** is enough for a small corpus. No multi-GPU needed at first.
- *"As long as their content can be represented as text or page images,"* any format fits this pattern.

### D7. GPU memory: OCR pipeline vs ColPali
> **OCR + text embeddings usually uses less GPU memory.** ColPali loads a 3B VLM backbone (~6 GB in bf16) and produces many vectors per page. Roughly: **OCR + text-RAG < ColPali** in memory, while ColPali is usually **better on visually rich documents**.

### D8. Same model for query and documents
> In ColPali, **the same model** encodes pages and queries into a compatible space: that's the design. With a different query model, **matching dimensions is not enough**. Both must be trained or aligned into the same semantic space, or the similarities are meaningless (Section 3.D trap).

### D9. Semantic-ID retrieval (asked in class; 🔬 beyond the lecture)
*"YouTube-style semantic ID retrieval: can that be covered?"* Each item (video, product) gets a **learned, discrete ID** that encodes meaning and user behaviour, e.g. by quantising its embedding into a short sequence of codebook tokens (residual quantisation, as in "TIGER"-style **generative retrieval**). Similar items share ID prefixes. A model can then **generate** candidate IDs for a user or query instead of doing a vector search, which is efficient at huge scale and shares information between similar items. It's the recommender-system cousin of dense retrieval.

**Code: toy semantic IDs by residual quantisation (NumPy)**
```python
import numpy as np
rng = np.random.default_rng(0)
topics = rng.normal(size=(8, 16))
X = topics[rng.integers(0, 8, 1000)] + 0.3 * rng.normal(size=(1000, 16))    # 1,000 item embeddings

def kmeans(X, k, iters=15):
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        a = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        C = np.stack([X[a == j].mean(0) if np.any(a == j) else C[j] for j in range(k)])
    return C, a

C1, code1 = kmeans(X, 8)                     # level 1: coarse "topic" code
C2, code2 = kmeans(X - C1[code1], 16)        # level 2: quantise what's LEFT (the residual)
sem_ids = [f"{a}-{b}" for a, b in zip(code1, code2)]
print("item 0:", sem_ids[0], "| item 1:", sem_ids[1])
print("unique IDs:", len(set(sem_ids)), "for", len(X), "items")
```
Actual output: `item 0: 5-9 | item 1: 4-5` and `unique IDs: 36 for 1000 items`. Items with the same **prefix** ("5-…") are semantically close, so a model can *generate* "5-" and then choose among its children. The collision count (36 IDs for 1,000 items) is also why real systems use **3–4 levels of ~256 codes** (learned with an RQ-VAE) plus a disambiguation suffix.

> 📚 **Go deeper**: *Recommender Systems with Generative Retrieval* (TIGER, semantic IDs): https://arxiv.org/abs/2305.05065

---

## 📚 Curated learning path after this lecture
| Step | Resource | Why |
|---|---|---|
| 1 | Hugging Face cookbook, *Multimodal RAG using document retrieval and VLMs* https://huggingface.co/learn/cookbook/en/multimodal_rag_using_document_retrieval_and_vlms | Rebuild Sections 4–6 end to end on Colab |
| 2 | ColPali paper https://arxiv.org/abs/2407.01449 + authors' blog https://huggingface.co/blog/manu/colpali | Read the source of Section 5 |
| 3 | Pinecone FAISS series https://www.pinecone.io/learn/series/faiss/ | Master the index layer (Section 1.E) |
| 4 | DeepLearning.AI *Building Multimodal Search and RAG* https://www.deeplearning.ai/short-courses/building-multimodal-search-and-rag/ | A guided course over the same material |
| 5 | Survey *Ask in Any Modality* https://arxiv.org/abs/2502.08826 | The field map (Section 7) |
| 6 | Qdrant *PDF retrieval at scale* https://qdrant.tech/documentation/advanced-tutorials/pdf-retrieval-at-scale/ | Production-scale late-interaction indexing (Section 5.G) |
| 7 | RAGAS https://docs.ragas.io/ | Evaluation and faithfulness in practice (Section 6) |
| 8 | Leaderboards: ViDoRe https://huggingface.co/spaces/vidore/vidore-leaderboard · MTEB https://huggingface.co/spaces/mteb/leaderboard | Pick current models, not 2024 ones |

---

## 9.E — Pre-read discussion questions, answered

**1. For a search problem you've used recently, was it keyword or semantic, and would hybrid help?**
> Example: searching a codebase or logs is keyword-dominant, since exact identifiers and error strings matter and BM25/grep wins. Searching internal docs ("how do I get VPN access?") is semantic. Most real search mixes both. Hybrid (BM25 + dense with RRF) helps whenever queries mix exact tokens with intent, e.g. "error E1043 when uploading large files".

**2. ColPali skips OCR entirely. Where else might "stop parsing, just embed the image" be underrated?**
> UI screenshots and app screens (retrieve by look and layout), slide decks, scanned forms and receipts, engineering drawings and schematics, handwritten notes, maps, dashboards and monitoring charts, web pages rendered as images (layout-aware search), medical reports with embedded images. Anywhere the **layout carries meaning** or text extraction is unreliable.

**3. If a RAG answer cites the wrong page but happens to be correct, success or failure?**
> **Failure.** The citation is the system's proof of grounding. A correct answer with a wrong citation most likely came from parametric memory or a lucky guess. It can't be verified, and it will fail silently on documents the model doesn't already know. Measure **citation accuracy** separately from answer accuracy.

**4. The EM-style framing treats retrieval as a hidden variable. Where else have you seen that?**
> Gaussian mixture models (latent cluster), HMMs (latent state sequence, Baum-Welch = EM), topic models (latent topic), VAEs (latent code), mixture-of-experts routing (latent expert choice), word alignment in classical machine translation (IBM models, latent alignments). See Section 2.D.

---

## 🎯 Capstone Interview Questions

**Q1. Design a question-answering system over 10,000 insurance policy PDFs (scanned and digital, with tables and diagrams). Cover indexing, retrieval, generation, evaluation and cost.**
> **Indexing:** classify pages (text-heavy vs visual/scanned). Text pages → parse + structure-aware chunks (by clause/section) → dense + BM25. Visual/scanned pages → ColPali (or ColQwen2) page embeddings with token pooling. Store doc/page/section/version/ACL metadata. Build an incremental pipeline with hashing and upserts. **Retrieval:** hybrid (BM25 + dense + ColPali) → RRF → cross-encoder or VLM rerank of the top 20 → top 3–5 pages. **Generation:** crop the hot regions, OCR them, and pass crop images + text to a VLM with a grounded prompt (cite [policy, page, clause], extract-then-compute for tables, explicit "Not found"). Add an agentic second hop for cross-references ("see Schedule B"). **Verification:** check each citation contains the quoted value; route low-support answers to abstention or a human. **Evaluation:** a labelled set of ~300 questions (answerable + unanswerable + multi-hop). Report recall@k/nDCG@5 per page type, accuracy given the gold page, end-to-end accuracy, citation accuracy and abstention precision. **Cost:** ColPali only on visual pages, quantised vectors, a cheap text LLM for text-only questions (routing), caching.

**Q2. Your multimodal RAG gives wrong answers about chart values. Diagnose it systematically.**
> (1) Retrieval: is the chart page in the top-k? (recall@k on the failing queries; check heatmaps). If not → chunking/index/retriever. (2) Handoff: is the page actually passed to the reader, as an **image**? (A text-only reader like the lab's SmolLM can never read chart values.) (3) Reading: give the model the gold page. If it's still wrong, it's the VLM's chart reading (resolution, model strength). Use extract-then-compute prompting, a higher-resolution crop, or chart-to-table extraction. (4) Faithfulness: is it answering from memory despite the evidence? Tighten grounding and verify.

**Q3. Compare BM25, dense bi-encoder, ColBERT, ColPali and cross-encoder in one table.**
> BM25: exact terms, no training, tiny index, blind to synonyms. Bi-encoder: one vector per text, offline index, semantic, weak on negation and exact identifiers. ColBERT: one vector per token, MaxSim, fine-grained, ~100× the index. ColPali: one vector per page patch, retrieves pages from pixels without OCR, ~1000× the index. Cross-encoder: joint encoding, the most precise, no index, only for reranking a short list.

**Q4. What metrics would you put on a RAG dashboard?**
> Retrieval: Recall@k, nDCG@5/10, MRR (on a labelled set, and tracked over time as the corpus changes). Generation: answer accuracy given the gold page; end-to-end accuracy; **faithfulness** (claims supported by the evidence, via an LLM judge or NLI); citation accuracy; abstention precision and recall. Operational: p50/p95 latency per stage, cost per query, index freshness lag, ingestion failure rate.

**Q5. When is multimodal RAG not worth it?**
> When the corpus is clean text with a reliable text layer; when the task is structured extraction with known fields (parsers are cheaper and exact); when latency, storage or GPU budgets are tight and visual content is decorative; or when a simpler CLIP image search already answers the (simple) visual queries. Use the "spatial locality" test: if the meaning doesn't depend on **where** things are on the page, text RAG is enough.

---

## ✅ Final self-check: can you do all of this without notes?
- [ ] State the unified retrieval pattern and fill in the 3-task recap table
- [ ] Explain the four takeaways with a concrete example each
- [ ] Choose among dense / hybrid / table / multimodal / graph RAG for a given corpus
- [ ] Design incremental ingestion with change detection, upserts and deletions
- [ ] Argue ColPali vs PDF parser for a specific business case
- [ ] Answer all four pre-read discussion questions
- [ ] Deliver capstone Q1 (the full system design) in about 5 minutes out loud
