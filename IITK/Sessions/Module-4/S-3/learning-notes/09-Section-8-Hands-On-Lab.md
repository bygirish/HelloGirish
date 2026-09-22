# Section 8 — Hands-On Lab: Building a Multimodal RAG Pipeline

> **Where this fits:** The deck's lab slide is a placeholder. The real lab is the Colab notebook `multimodal_rag_images_(1).ipynb` (link in `…Colab-notebook-12th-Sept.pdf`). It builds **two retrievers** (caption+OCR text → FAISS, and ColPali patch vectors → MaxSim), **fuses** them with Reciprocal Rank Fusion, and generates a **grounded** answer. Every earlier section shows up here as code.
> **Runtime:** Colab GPU (T4 is enough). Note the bf16 caveat in 8.I.

---

## 8.A — The pipeline and the dataset

```
                      ┌──────────── OFFLINE (indexing) ────────────┐
DATASET (60 images) ─►│ EasyOCR ─► ocr_text                         │
                      │ Florence-2 ─► caption                       │
                      │ combined_text = caption + " " + ocr_text    │
                      │   └► all-mpnet-base-v2 (768-d, normalised)  │
                      │        └► FAISS IndexFlatIP   (TEXT index)  │
                      │ ColPali v1.3 ─► 1031×128 per image          │
                      │        └► list of tensors in CPU RAM        │
                      │                              (VISUAL index) │
                      └─────────────────────────────────────────────┘
                      ┌──────────── ONLINE (per query) ────────────┐
Query ───────────────►│ text_retrieve (FAISS)  ─┐                   │
                      │ colpali_retrieve(MaxSim)─┴► RRF (k=60)      │
                      │   └► top-k items ─► their caption + OCR text│
                      │        └► SmolLM2-360M-Instruct (TEXT ONLY) │
                      │             └► grounded answer              │
                      └─────────────────────────────────────────────┘
```

### Dataset (Block 1): 60 images from three sources
| Source | Count | What it is | Why it's included |
|---|---|---|---|
| **CIFAR-10** (`uoft-cs/cifar10`, train[:20]) | 20 | Tiny 32×32 natural photos (airplane, cat, dog…) | "Natural image" retrieval: no text, so OCR is empty |
| **DocVQA** (`lmms-lab-encoder/DocVQA`, validation[:20]) | 20 | Scanned business documents + a question + answers | Document pages where the answer is text on the page |
| **InfographicVQA** (validation[:20]) | 20 | Infographics: charts, icons, numbers | Visually rich pages, where layout matters most |

For DocVQA and InfographicVQA, `description` holds the **question**, and the metadata keeps `answers` (and `ocr_ground_truth` for infographics).

> ⚠️ **Trap (live Q&A):** *"How do we check which image files are there?"* The images are **not read from a folder**. They're loaded from the three Hugging Face datasets into `DATASET`, so `IMAGE_DIR` is unused. Inspect with `len(DATASET)`, `[(i, x["source"], x["label"]) for i, x in enumerate(DATASET[:20])]`, and view one with `DATASET[i]["image"]`.

---

## 8.B — Block 2: text from pixels (EasyOCR + Florence-2)

### EasyOCR (Cell 2-A)
```python
results = ocr_reader.readtext(np.array(image), detail=1, paragraph=False)
texts = [text for (_, text, confidence) in results if confidence > 0.4]
```
Returns (bbox, text, confidence) triples. Only words with confidence > 0.4 are kept, and the **bounding boxes are thrown away**. That's the "OCR loses layout" point from Section 4.D, happening right in the code.

What to expect: CIFAR images return **nothing** (🔇). *"That shows OCR alone is insufficient for visual retrieval."* DocVQA returns lots of text, noisy on scans.

### Florence-2 captions (Cells 2-B)
```python
task_prompt = "<CAPTION>" if source == "cifar10" else "<DETAILED_CAPTION>"
generated_ids = florence_model.generate(**inputs, max_new_tokens=80, num_beams=3)
```
- Florence-2 is a small Microsoft VLM controlled by **task tokens** (`<CAPTION>`, `<DETAILED_CAPTION>`, `<OCR>`, `<OD>`…) instead of free-form prompts
- Short captions for tiny CIFAR images, detailed captions for documents
- `num_beams=3`: beam search, for more deterministic captions

### Combined text (Cell 2-C)
`combined_text = caption + " " + ocr_text`. The code comment is important:
> *"We deliberately DO NOT include the DocVQA question / ground-truth answer… Those are evaluation metadata and should not leak into the retrieval representation."*

> 💡 **Learning Thought:** **Label leakage** is the most common way retrieval demos look better than they are. If the question text were indexed with its own page, "retrieval" would just be string matching against the answer key. Keep evaluation metadata out of anything that gets indexed, and audit this in every RAG eval you build.

This block is storage **design (a) "unified text representation"** from Section 4.E.

---

## 8.C — Block 3: MPNet + FAISS (the text retriever)

```python
st_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
text_embeddings = st_model.encode(combined_texts, normalize_embeddings=True).astype("float32")
faiss_index = faiss.IndexFlatIP(768)      # exact inner-product search
faiss_index.add(text_embeddings)
scores, indices = faiss_index.search(query_embedding, top_k)
```

| Detail | Why |
|---|---|
| `all-mpnet-base-v2`, 768-d | A stronger sentence embedder than MiniLM (384-d) |
| `normalize_embeddings=True` | Makes **inner product = cosine** (the MIPS vs cosine trap, Section 1.E) |
| `IndexFlatIP` | **Exact** search. *"With only 60 images, exact search is more than sufficient."* At millions of vectors you'd use `IndexIVFFlat`/`IndexHNSWFlat` (the notebook's own extension list says "FAISS IVF") |
| `.astype("float32")` | FAISS requires float32 |

Sanity check: *"a fluffy pet"* should return CIFAR cats and dogs. It only works if Florence's captions mention them, which shows **the text retriever is only as good as the captions**.

---

## 8.D — Cell 3-C: PCA visualisation

The 768-d embeddings of the 60 items and 5 probe queries ("a domestic pet animal", "a vehicle", "a document containing text", "an infographic…", "a natural scene") are projected to 2-D. **PCA is fit on the database only**, and the queries are transformed into that space.

What you should see: three clusters (CIFAR / DocVQA / Infographic), with each query star landing near its matching cluster. Also read the **explained variance** of PC1 + PC2. If it's low (say 20–30%), the 2-D picture is a rough shadow of the real geometry.

> ⚠️ **Trap (live Q&A):** *"Why do we need to reduce the dimension using PCA?"* **We don't, for retrieval.** PCA is **only for visualisation**, to plot and eyeball whether queries land near the right items. Real similarity search still uses the **full 768-d** embeddings. Retrieving in the PCA space would throw away most of the information.

---

## 8.E — Block 4: ColPali indexing and MaxSim in code

### Loading
```python
colpali_model = ColPali.from_pretrained("vidore/colpali-v1.3", torch_dtype=torch.bfloat16, device_map="cuda:0")
colpali_proc  = ColPaliProcessor.from_pretrained("vidore/colpali-v1.3")
```
Florence-2 is deleted first (`del florence_model; torch.cuda.empty_cache()`) to free GPU memory. Captions are already stored, so it's no longer needed. Load a model, use it, free it: that's the standard pattern for fitting several models on one T4.

### Indexing (Cell 4-B)
```python
inputs = colpali_proc.process_images(batch)           # resize to 448×448, add prompt tokens
batch_embs = colpali_model(**inputs)                  # [batch, 1031, 128]
all_embs.append(emb.detach().float().cpu())           # store each page in CPU RAM
```
Each image → **[1031, 128]**: 1024 patch vectors + 7 non-spatial prompt tokens (Section 5.B). They're stored in **CPU RAM as float32**, ≈ 528 KB per image, so 60 images ≈ 32 MB. Fine here. Scale it to 1M pages, though, and you get ~528 GB: Section 5.G's storage problem, made concrete.

### MaxSim retrieval (Cell 4-C): the formula as three tensor ops
```python
similarity = torch.matmul(query_emb, image_emb.T)   # [n_query_tokens, 1031]
score = similarity.max(dim=1).values.sum().item()   # max over patches, sum over query tokens
```
This is **exactly** S(q, D) = Σᵢ maxⱼ qᵢ·dⱼ:
- `matmul` → the full query-token × patch similarity matrix
- `.max(dim=1)` → for each **query token**, its best patch
- `.sum()` → add up the best matches

The Python loop over all images is fine for 60. At scale use `colpali_proc.score_multi_vector(...)` (batched) and a candidate-generation stage.

> 💡 **Learning Thought:** When a paper's equation turns into **three lines of tensor code**, you really understand it. Try writing `.max(dim=0)` by mistake: you get "for each patch, its best query token", which is the **heatmap** reduction (8.F), not the retrieval score. Knowing which axis you reduce over *is* knowing the method.

---

## 8.F — Cell 4-D: patch heatmaps

```python
patch_emb    = image_emb[-n_image_patches:]              # 1024 spatial vectors (see bug below)
similarity   = torch.matmul(query_emb, patch_emb.T)      # [q_tokens, 1024]
patch_scores = similarity.max(dim=0).values              # per PATCH: best query token
heat         = patch_scores.reshape(32, 32)              # back onto the page grid
# min-max normalise → resize to the image → overlay with cmap="hot", alpha=0.5
```
Demo: query *"a legal document"* → take the top-1 image → overlay. Relevant regions (headings, dense text) light up.

> ⚠️ **Confirmed bug: which 1024 of the 1031 vectors are the patches?** The code takes the **last** 1024, assuming the 7 prompt tokens come first. I checked with `colpali-v1.3`'s own processor:
> ```
> sequence length: 1031 | image tokens: 1024
> image token positions: first 0 last 1023
> non-image tokens: ['<bos>', '<bos>', 'Describe', '▁the', '▁image', '.', '\n']
> ```
> The **image tokens come first** (positions 0–1023) and the 7 prompt tokens **after**. So `image_emb[-1024:]` drops patches 0–6 (the first 7 of the top row), includes the 7 prompt-token vectors, and shifts every heatmap cell by 7 positions. That's easy to miss by eye, but wrong. (Script: Section 5.B.)
>
> **Fix** (one line in Cell 4-D): use the processor's mask instead of position arithmetic.
> ```python
> batch = colpali_proc.process_images([DATASET[image_idx]["image"]])
> mask  = colpali_proc.get_image_mask(batch)[0]          # True exactly at the 1024 patch positions
> patch_emb = image_emb[mask]                             # replaces: image_emb[-n_image_patches:]
> ```
> Or use the authors' helpers `get_similarity_maps_from_embeddings` + `plot_all_similarity_maps` (`pip install colpali-engine[interpretability]`), which apply the mask for you and draw one map per query token. The full corrected function is in Section 5.F.

---

## 8.G — Block 5: Reciprocal Rank Fusion

$$\text{RRF}(d) = \sum_{r \in \text{retrievers}} \frac{1}{k + \text{rank}_r(d)}, \qquad k = 60$$

*"RRF rewards documents that rank highly in **both** retrievers, without requiring score normalisation."* This matters because FAISS cosine is in [−1, 1] while MaxSim is unbounded and grows with query length (Section 7.C).

### Worked example (k = 60)
| Item | FAISS rank | ColPali rank | RRF score |
|---|---|---|---|
| X | 2 | 3 | 1/62 + 1/63 = 0.016129 + 0.015873 = **0.032002** |
| Y | 4 | 2 | 1/64 + 1/62 = 0.015625 + 0.016129 = **0.031754** |
| Z | 10 | 1 | 1/70 + 1/61 = 0.014286 + 0.016393 = **0.030679** |
| W | 1 | 40 | 1/61 + 1/100 = 0.016393 + 0.010000 = **0.026393** |

Final order: **X > Y > Z > W**.
- W was **#1** in FAISS but only #40 in ColPali, so it falls to last. **Consensus beats a single strong vote.**
- X was never first in either list, but is near the top of **both**, and wins.
- Z was #1 in ColPali but #10 in FAISS, and still beats W. A strong vote plus a decent one outranks a strong vote plus a weak one.

### Code: RRF you can run (plain Python)
```python
def rrf(rankings, k=60):
    """rankings: list of ranked id-lists (best first). Returns [(id, score)] best first."""
    scores = {}
    for ranked in rankings:
        for rank, doc in enumerate(ranked, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])

pad = lambda prefix, n: [f"{prefix}{i}" for i in range(n)]
faiss_rank   = ["W", "X", "a", "Y"] + pad("f", 5) + ["Z"]         # W=1, X=2, Y=4, Z=10
colpali_rank = ["Z", "Y", "X"] + pad("c", 36) + ["W"]              # Z=1, Y=2, X=3, W=40
for k in (60, 0):
    print(f"k={k:2d}:", [(d, round(s, 4)) for d, s in rrf([faiss_rank, colpali_rank], k) if d in "WXYZ"])
```
Actual output:
```
k=60: [('X', 0.032), ('Y', 0.0318), ('Z', 0.0307), ('W', 0.0264)]
k= 0: [('Z', 1.1), ('W', 1.025), ('X', 0.8333), ('Y', 0.75)]
```
The second line shows **why k matters**. With k = 0 the two items that were **#1 somewhere** (Z, W) win, even though W was 40th in ColPali. With k = 60 the consensus item X wins. This `rrf()` is the same logic as the notebook's `rrf_retrieve()` (Cell 5-A).

### Why k = 60?
k controls how steeply the reward falls with rank. With **k = 0**, rank 1 scores 1.0 and rank 2 scores 0.5, so a single retriever's #1 dominates everything. With **k = 60**, rank 1 (0.0164) and rank 10 (0.0143) are close, so **agreement across retrievers** matters more than any one retriever's top pick. 60 is the value from the original RRF paper (Cormack et al., 2009) and works well without tuning.

### Cell 5-B: visual comparison
The 3 × 5 grid for *"a cute small domestic pet"* shows FAISS-text vs ColPali-visual vs RRF-fused top-5 side by side. Look for where they **disagree**: text retrieval depends on the captions, while ColPali sees pixels but was trained on documents, not 32-px photos (8.I).

---

## 8.H — Block 6: grounded generation

### What the code actually does
- **Model:** `HuggingFaceTB/SmolLM2-360M-Instruct`, a **text-only** 360M LLM
- `multimodal_rag()` → RRF top-5 → for each item builds `"Retrieved item r (index, source, label): Caption: … OCR text: …"` → `generate_answer()`
- The code says so explicitly: *"SmolLM NEVER receives the images. Images are retained only for visualization/debugging."*

So the **retrieval** is multimodal (FAISS on text derived from images + ColPali on pixels), but the **generation** is text-only, from captions + OCR.

### Two prompting modes (Cell 6-B)
| Mode | System prompt (abridged) |
|---|---|
| `grounded=True` | "Use ONLY the retrieved text… Do not use outside knowledge… If the answer cannot be determined, say: '**Not found in retrieved text.**' Keep the answer concise and factual." |
| `grounded=False` | "Answer the user's question using the retrieved text **and your general knowledge** when appropriate." |

Greedy decoding (`do_sample=False`) keeps answers deterministic, so the two modes can be compared fairly.

### The demo queries (Cell 5-A.2 / 6)
| Question | Ground truth | What it tests |
|---|---|---|
| "What is the 'actual' value per 1000 during the year 1975?" | ≈ 0.24 per 1000 | **Chart reading**: Section 0.C's motivating example |
| "How many sustainability committee meetings has Y. C. Deveshwar attended?" | 3 | **Table lookup** in a scanned report |
| "When was military's first attempt of social media ARPANET?" | 1969 | Infographic text |
| "What was the growth in Facebook users from 2008 to 2013?" | *(blank)* | Infographic **numbers + arithmetic** |

**Predict before you run it:** the 1975 chart question will probably **fail or abstain**. The value is in a plotted line, OCR returns only axis labels and tick numbers, and SmolLM never sees the image. That's the text-only-RAG failure from Section 0.C, reproduced inside a "multimodal" pipeline. The table and infographic questions may succeed if OCR captured the right cell text.

> 💡 **Learning Thought:** The system is **only as multimodal as its weakest stage.** Multimodal retrieval into a text-only reader is still **text RAG at read time**, bottlenecked by OCR and captions. To answer chart questions, the **reader** must see pixels (a VLM such as Qwen2-VL / SmolVLM), or you need chart-to-table extraction.

---

## 8.I — Notebook bugs and gotchas

| # | Issue | Why it matters / fix |
|---|---|---|
| 1 | 🔴 **Hard-coded Hugging Face token** in Cell 20: `login(token='hf_…')` | A live secret in a shared notebook. Anyone with the file can use it. **Never commit or share it.** Use `login()` (interactive) or `os.environ["HF_TOKEN"]` / Colab Secrets (the fix is below the table). The token should be revoked ([HF token docs](https://huggingface.co/docs/hub/security-tokens)). |
| 2 | Markdown says **BLIP-2** (Block 2, Summary); the code uses **Florence-2-base-ft** | Doc drift. Florence-2 is smaller and task-token driven. |
| 3 | Block 6 markdown says **Qwen2-VL-2B, 4-bit, sees image pixels**; the cell comment says "Load **SmolVLM**-500M"; the code loads **SmolLM2-360M, text-only** | Three different claims. The truth is **text-only generation**, so the "the model sees the actual image pixels" claim is false (8.H). |
| 4 | Block 1 says "12 images downloaded from public URLs", `IMAGE_DIR` defined | Actually 60 images from HF datasets; `IMAGE_DIR` is unused |
| 5 | **Heatmap patch slicing** takes the last 1024 of 1031 vectors | **Confirmed** off by the 7 prompt tokens: patches are positions 0–1023. Use `get_image_mask` (8.F). |
| 6 | **bfloat16 on a T4** | T4 (Turing) has **no native bf16**, so it may be very slow or error depending on the version. Use `torch.float16` on T4, or an A100/L4. |
| 7 | **CIFAR images are 32×32** | ColPali resizes them to 448×448 (blurry blobs) and was trained on **document** pages, so its "pet" rankings on CIFAR are unreliable. A fair test of ColPali uses documents. |
| 8 | `EVAL_QUERIES` (Cell 5-A.1) is **built but never scored** | No Recall@K / nDCG is computed. See Exercise 1. |
| 9 | The evidence string passes CIFAR's **ground-truth `label`** to the generator (`label=cat`) | Mild leakage: the generator gets the class name for free, beyond caption/OCR. Drop `label` from the evidence for an honest test. |
| 10 | DocVQA `validation[:20]` can contain **several questions about the same page** | Duplicate pages in the index; strict `target_idx` evaluation would count a duplicate's retrieval as a miss. De-duplicate by `doc_id`. |
| 11 | MaxSim loop in Python over every image | Fine at 60; use batched `score_multi_vector` and candidate generation at scale |

> ⚠️ **Trap (live Q&A):** *"What is the meaning of `torch.cuda.memory_allocated() / 1e9`?"* `1e9` = 1 × 10⁹, so this converts **bytes to gigabytes** (decimal GB). `memory_allocated` = memory currently held by tensors. `memory_reserved` = memory PyTorch's caching allocator has claimed from the GPU (≥ allocated). `empty_cache()` releases unused reserved memory back to the GPU.

---

### Code: the safe replacement for Cell 20
```python
# Colab: add HF_TOKEN in the 🔑 "Secrets" panel (left sidebar), then:
from google.colab import userdata
from huggingface_hub import login
login(token=userdata.get("HF_TOKEN"))
# Local: `export HF_TOKEN=...` in your shell (or run `huggingface-cli login` once); huggingface_hub reads it automatically.
```
To stop secrets reaching git at all, run a scanner such as **gitleaks** as a pre-commit hook: https://github.com/gitleaks/gitleaks

### Code: make generation truly multimodal (replaces Cells 6-A/6-B)
The lab's reader is text-only, so it cannot answer the 1975 chart question. The fix is to pass the **retrieved images** to a small VLM. The full retrieve-then-VLM code is in **Section 4.B**. In the lab's own terms:
```python
results = rrf_retrieve(query, top_k=2)                         # unchanged lab retrieval
images  = [DATASET[i]["image"] for i, _ in results]            # the pixels SmolLM never saw
# → build a Qwen2-VL / SmolVLM chat message with these images + the grounded instruction (Section 4.B, 6.B)
```
SmolVLM (https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct) fits on a T4 next to the stored ColPali embeddings.

---

## 8.J — Exercises (do these to build expertise)

1. **Actually evaluate.** Using `EVAL_QUERIES`, compute **Recall@1/5/10 and nDCG@5** for FAISS-only, ColPali-only and RRF, broken down by source (CIFAR / DocVQA / Infographic). *Expect ColPali to win on DocVQA and Infographic, and text/FAISS to win on CIFAR.* For CIFAR, count a hit if **any** image of the correct class is retrieved. Starter code (paste into the notebook after Cell 5-A.1):
   ```python
   import numpy as np
   from collections import defaultdict

   def is_hit(eval_q, idx):
       if eval_q["source"] == "cifar10":                     # class-level relevance
           return DATASET[idx]["source"] == "cifar10" and DATASET[idx]["label"] == eval_q["label"]
       return idx == eval_q["target_idx"]                     # page-level relevance

   def evaluate(retrieve_fn, K=(1, 5, 10)):
       per_source = defaultdict(lambda: defaultdict(list))
       for eq in EVAL_QUERIES:
           ranked = [i for i, _ in retrieve_fn(eq["query"], top_k=10)]
           hits = [is_hit(eq, i) for i in ranked]
           for k in K:
               per_source[eq["source"]][f"R@{k}"].append(any(hits[:k]))
           first = next((r for r, h in enumerate(hits[:5], 1) if h), None)
           per_source[eq["source"]]["nDCG@5"].append(1 / np.log2(first + 1) if first else 0.0)
       return {s: {m: round(float(np.mean(v)), 3) for m, v in ms.items()} for s, ms in per_source.items()}

   for name, fn in [("FAISS", text_retrieve), ("ColPali", colpali_retrieve), ("RRF", rrf_retrieve)]:
       print(name, evaluate(fn))
   ```
   (nDCG@5 here uses the "first relevant hit" simplification, which is exact for single-relevant DocVQA/Infographic queries.)
2. **Fix the heatmap slicing** (8.F). Compare heatmaps with `[:1024]` vs `[-1024:]` on a DocVQA page and a question whose answer location you know.
3. **Make generation truly multimodal.** Replace SmolLM2 with a small VLM (SmolVLM-500M or Qwen2-VL-2B in 4-bit) and pass the top-1/2 **page images** plus OCR text. Re-run the four demo queries, especially the 1975 chart.
4. **Grounded vs unconstrained.** Run every demo query with `grounded=True/False`. Count hallucinated specifics and "Not found" responses. Add one question the corpus **can't** answer and check abstention.
5. **RRF sensitivity.** Try k = 0, 10, 60, 200 and watch how the fused ranking changes (8.G).
6. **Swap the backbone.** Replace `vidore/colpali-v1.3` with `vidore/colqwen2-v1.0` (the notebook's own suggestion) and compare nDCG@5 on the document subsets.
7. **Scale the index.** Switch `IndexFlatIP` to `IndexHNSWFlat` or `IndexIVFFlat` (requires `train()` on the vectors), add a few thousand DocVQA pages, and measure recall vs exact search.
8. **Storage budget.** Measure the ColPali index size in bytes, then apply float16 storage and simple token pooling (average adjacent 2×2 patches → 256 vectors). Re-measure size and nDCG.

---

## 🎯 Interview Questions

**Q1. Walk me through the lab pipeline and name its weakest link.**
> Indexing: EasyOCR + Florence-2 captions → combined text → MPNet (normalised) → FAISS exact IP; in parallel ColPali v1.3 → 1031×128 vectors per image in CPU RAM. Query: FAISS cosine ranking + ColPali MaxSim ranking → RRF (k = 60) → top-5 → their caption + OCR text → SmolLM2-360M with a grounded prompt. The weakest link is the **text-only reader**. Chart and layout information that ColPali used to retrieve the page is lost at read time, so chart questions fail even when retrieval succeeds.

**Q2. Why use RRF instead of averaging the FAISS and ColPali scores?**
> The scores are on incompatible scales: cosine in [−1, 1], MaxSim unbounded and dependent on query length. Averaging lets MaxSim dominate arbitrarily. RRF uses only ranks, needs no calibration, and rewards agreement: an item at ranks 1 and 3 beats one at 1 and 40. k = 60 flattens the rank curve so consensus matters more than any single #1.

**Q3. How would you evaluate this notebook properly?**
> Use the held-out questions (DocVQA/Infographic questions with gold pages; CIFAR class prompts with class-level relevance). Report Recall@K and nDCG@5 per retriever and per source, then end-to-end answer accuracy (ANLS or exact match against the gold answers) for grounded vs unconstrained prompting, plus abstention on unanswerable questions. De-duplicate pages, make sure no question text or labels leak into the index or the generator's context, and report the gold-page ablation to separate retrieval errors from reading errors.

**Q4. Spot the security issue in this notebook and state the fix.**
> Cell 20 hard-codes a Hugging Face access token. Anyone with the notebook (or the git history, if committed) can use it. Fix: revoke/rotate the token, remove it from the notebook and its history, and load credentials from Colab Secrets or an environment variable (`os.environ["HF_TOKEN"]`, `huggingface_hub.login()` without an argument). Add a secret scanner (e.g. gitleaks) to pre-commit.

**Q5. Your ColPali heatmaps look plausible but slightly off. How do you debug it?**
> Check the tensor bookkeeping: which positions in the 1031 outputs are image patches (the prompt-token offset), whether the 32×32 reshape is row-major and matches the processor's patch order, whether the image was resized or padded (so the overlay must match the 448×448 frame, not the original aspect ratio), and which axis the max is taken over (per patch = dim 0). Validate on a synthetic page with a known keyword at a known location.

---

## ✅ Self-check
- [ ] I can draw the lab pipeline and say which parts are offline vs online
- [ ] I know why the evaluation metadata is excluded from `combined_text`
- [ ] I can explain why `normalize_embeddings=True` + `IndexFlatIP` = cosine search
- [ ] I can explain that PCA is only for visualisation
- [ ] I can map `matmul → max(dim=1) → sum` onto the MaxSim formula, and `max(dim=0)` onto the heatmap
- [ ] I can compute RRF by hand and explain k = 60
- [ ] I can explain why this "multimodal RAG" still fails on the 1975 chart
- [ ] I can list at least 6 notebook issues, starting with the hard-coded token
- [ ] I have done Exercises 1 and 3
