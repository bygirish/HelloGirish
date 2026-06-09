# Session 3: Retrieval Augmented Generation (RAG)
### IIT Kharagpur × upGrad | Instructor: Prof. Koustav Rudra

---

## Navigation

| # | File | Topics Covered |
|---|------|----------------|
| 1 | [Section 1 — Motivating RAG](./01_Motivating_RAG.md) | LLM limitations, hallucinations, verifiability, knowledge cutoff, closed vs open book |
| 2 | [Section 2 — Introduction to RAG](./02_Introduction_to_RAG.md) | RAG overview, architecture, naive RAG, advanced RAG, end-to-end example |
| 3 | [Section 3 — The Retrieval Phase](./03_Retrieval_Phase.md) | Sparse/dense retrieval, pipeline, chunking, query formulation, TF-IDF, BM25 |
| 4 | [Section 4 — Common Problems of RAG](./04_Common_Problems_of_RAG.md) | Retrieval failures, context failures, generation failures |
| 5 | [Section 5 — RAG Variants](./05_RAG_Variants.md) | Standard RAG, RAG with Memory, Agentic RAG, Chain-of-RAG (CoRAG) |
| 6 | [Section 6 — RAG Assessment Framework](./06_RAG_Assessment_Framework.md) | RAGAS metrics, TruLens, evaluation pipeline |

---

## Session Takeaway (One-liner)

> RAG turns the LLM from an **oracle of memorised facts** into a **reasoner over retrieved evidence** — answer quality is therefore bounded by what the retriever surfaces.

---

## Key Learning Arc

```
Why RAG?          What is RAG?       How does Retrieval work?
(Section 1)  -->  (Section 2)   -->  (Section 3)
                                          |
                                          v
                              What can go wrong?   What are variants?   How to evaluate?
                              (Section 4)     -->  (Section 5)    -->   (Section 6)
```

---

## Prerequisites Checklist

- [ ] Zero-shot, few-shot, and Chain-of-Thought prompting (Sessions 2.1 & 2.2)
- [ ] Basic text representation: bag-of-words, term frequency, embeddings/vectors
- [ ] Information retrieval intuitions: question answering, document ranking, precision/recall@k
- [ ] Basic Python for hands-on notebook

---

## Files in This Session Folder

| File | Purpose |
|------|---------|
| `Module2-Session3-RAG.pdf` | Main lecture slides (58 pages) |
| `Session-agenda-Session-3.pdf` | Learning outcomes, prerequisites, takeaway |
| `Pre-Post-read-materials-Session-3.pdf` | Reference reading list |
| `Module_2_Session_3_Sparse_Retrieval_&_RAG.ipynb` | Hands-on PyTerrier notebook |
| `transcript.docx` | Session transcript |
| `learning_notes/` | **These notes — your study reference** |
