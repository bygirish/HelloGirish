# Section 6 — Chunking Strategies

> **Session:** IIT-KGP Module 2 · Session 4 · Instructor: Pawan Goyal
> **Theme:** *The chunk is the true unit of retrieval. Every chunking decision is a retrieval quality decision. Measure, don't assume.*

---

## 6.1 Why Chunk at All?

Documents are long. Embedding models have token limits. But more fundamentally, **chunking is necessary even if you had infinite context**, because:

1. **Embedding models have token limits** — typically 512–8192 tokens. Longer inputs are silently truncated.
2. **A single vector summarises its input** — embedding a 100-page contract into one vector averages meaning into mush. Relevant sections become undetectable.
3. **Retrieval should return the relevant passage, not the entire document** — the LLM needs precision, not a fire hose.
4. **LLM context windows are finite and expensive per token** — you want to pass 3–5 relevant paragraphs, not 300 pages.

### The Golden Goal

> Each chunk should hold **one self-contained, retrievable idea** — small enough to be precise, large enough to stand alone.

### The Eternal Tension

```
Too small → loses context ("it rose 40%" — what rose?)
Too large → dilutes the embedding; over-retrieves
```

Size should match how much text typically holds one complete answer in *your* data. There is no universal best size.

---

## 6.2 The Governing Question

Before choosing a strategy, ask:

> *What is one self-contained "answer" in this corpus, and where are its natural seams?*

The seam might be:
- A sentence boundary (short Q&A, dialogues)
- A paragraph boundary (reports, articles)
- A heading/section boundary (structured documents, legal clauses)
- A function definition (code)
- A turn boundary (chat transcripts)
- A row (tables, CSVs)

---

## 6.3 Strategy 1: Fixed-Size Chunking

**The simplest approach:** split every N tokens (or characters), regardless of content.

```
Document: |████████████████|████████████████|████████████|
           chunk 1          chunk 2          chunk 3
           (400 tokens)     (400 tokens)     (300 tokens)
```

### Implementation (Token-Based)

```python
from langchain.text_splitter import TokenTextSplitter

splitter = TokenTextSplitter(
    chunk_size=400,        # tokens, not characters!
    chunk_overlap=0
)
chunks = splitter.split_text(text)
```

**Why token-based, not character-based?**
Count tokens with the *embedding model's own tokeniser* — the model's limit is in tokens. Counting characters makes your budget unreliable.

### Advantages

- Dead simple and fast — trivial to implement
- Predictable chunk count (useful for resource planning)
- Good baseline to beat

### Limitations

- Can break mid-sentence, mid-idea, mid-table
- Context loss at boundaries: "it rose 40%" — what rose?

### When to Use

- Prototyping — always start here before investing in complex strategies
- Uniform, structured content where boundaries don't matter much
- You need a baseline to measure all other strategies against

---

## 6.4 Strategy 2: Sliding Window Chunking

**The sane default for most RAG applications.** Fixed-size chunks with controlled overlap between adjacent chunks.

```
Window size: 512 tokens, Overlap: ~100 tokens (20%)

Document: |████████████████|
           |     ████████████████|
                  |     ████████████████|
                         ↑
                    100 tok overlap = fact that straddles
                    boundary lands intact in at least one chunk
```

### Implementation

```python
from langchain.text_splitter import TokenTextSplitter

splitter = TokenTextSplitter(
    chunk_size=512,
    chunk_overlap=102        # ~20% of 512
)
chunks = splitter.split_text(text)
```

### Advantages

- Carries context across chunk boundaries
- Nothing is severed at the cut line (for overlapping content)
- Recommended default for most RAG applications

### Limitations

- More storage (duplicate content between adjacent chunks)
- Possible duplicate retrievals at query time (same text in two chunks)

### The Overlap Dial

```
0%   → boundaries get cut; facts that straddle a boundary are lost
20%  → catches boundary facts cheaply; the "knee of the curve"
50%+ → near-duplicate chunks that double your store for marginal recall
```

**The overlap myth (Jan 2026, arXiv study):** A study using SPLADE + 8B Mistral on Natural Questions found that overlap gave **no measurable benefit** in their setup, only higher indexing cost. The takeaway is not "never use overlap" — it is **test, don't assume**. Overlap helps in some setups and not in others. Measure before defaulting to it.

---

## 6.5 Strategy 3: Recursive / Structure-Aware Chunking

**Split at natural document boundaries first, fall back to size only if needed.**

```
Try to split on:
1. Double newline (paragraph break)
2. Single newline
3. Sentence boundary (". ")
4. Word boundary
5. Character

If chunk is still too large at level N, recurse to level N+1
```

### LangChain Implementation

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=120,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_text(text)
```

### Advantages

- Respects natural document structure
- Sentence-aware — rarely cuts mid-sentence
- Good general-purpose default (often outperforms fixed-size in benchmarks)

### Limitations

- Chunk sizes vary (not perfectly predictable)
- Tuning the separator list matters for unusual formats

### Structure-Aware Variants

For documents with explicit structural markers, split **on the structure itself**:

```python
# Markdown: split on headings
import re

def split_by_heading(text, max_tokens=512):
    sections = re.split(r'\n#{1,6} ', text)
    # Each section is a natural chunk
    return sections

# Legal docs: split on clause numbers
# Code: split on function/class definitions (AST-aware)
# API docs: split on endpoint definitions
```

**Why structure-aware for legal/code?**
- A chunk that *is* "§7.2 Termination" retrieves precisely for "termination clause" queries
- A half-function or half-clause is useless to the LLM

**For code:** Always split by function/class boundaries, never by token count. `RecursiveCharacterTextSplitter` has language-aware variants for Python, JS, etc.

---

## 6.6 Strategy 4: Semantic Chunking

**Cut where the topic shifts, not at arbitrary boundaries.**

Algorithm:
1. Embed every sentence independently
2. Compute cosine distance between consecutive sentence embeddings
3. Split where the distance exceeds a threshold (topic shift)

```
Sentence 1: "The water cycle consists of evaporation..."
Sentence 2: "Evaporation occurs when liquid water..."
    distance = 0.05 (same topic → don't cut)

Sentence 3: "...and precipitation completes the cycle."
Sentence 4: "The CPU architecture consists of..."
    distance = 0.89 (topic shift → CUT HERE)
```

### Implementation (LangChain)

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",  # or "standard_deviation"
    breakpoint_threshold_amount=90           # cut at top 10% distance jumps
)
chunks = splitter.split_text(text)
```

### Advantages

- Chunks align with meaning shifts — coherent ideas stay together
- Can lift recall ~9% vs naive fixed-size in some benchmarks

### Limitations and Critical Notes

- ~14× slower than token splitting at ingest time (embeds every sentence)
- The threshold is sensitive — wrong threshold = over-splits short documents
- **Always pair with a hard `max_tokens` cap**: one long on-topic run could blow the model limit
- **Pay the cost only when your retrieval metrics actually move** — run a bake-off first

---

## 6.7 Strategy 5: Small-to-Big (Hierarchical / Parent-Document)

A powerful hybrid: embed small chunks for retrieval precision, but return larger parent windows to the LLM for context.

```
INDEXING:
Document → split into sentences (small chunks) → embed each sentence

RETRIEVAL:
Query → find top-K sentence-level matches
      → return the parent paragraph/section containing each match to LLM
```

**Why it works:**
- Small chunks = precise matching (the right sentence gets retrieved)
- Large context returned = LLM gets enough context to answer fully

### Implementation

```python
# LlamaIndex: SentenceWindowNodeParser
# LangChain: ParentDocumentRetriever

from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter

child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
```

**Best for:** Dense reference text where the precise match needs surrounding context (e.g., a legal contract where the key clause is one sentence but needs the surrounding paragraph to be interpretable).

---

## 6.8 Strategy 6: Special Cases

### Chat Logs / Transcripts (e.g., Customer Helpline)

**Chunk by turn/speaker.** The utterance is the natural unit.

```
Turn 1 (Customer): "My order hasn't arrived..."
Turn 2 (Agent): "Let me check your order number..."
→ Each turn or turn-pair = one chunk
```

### Tables / CSVs / Records

**Row/schema-aware chunking. Never split a row across chunks.**

```python
# Each row becomes one chunk with schema context
def chunk_csv(df):
    schema = ", ".join(df.columns)
    return [f"Schema: {schema}\nRow: {dict(row)}" for _, row in df.iterrows()]
```

### Code

**Split by function or class, never by token count.**

```python
# Python example using AST
import ast

def split_by_function(source_code):
    tree = ast.parse(source_code)
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno - 1
            end = node.end_lineno
            functions.append("\n".join(source_code.split("\n")[start:end]))
    return functions
```

---

## 6.9 The Complete Decision Guide

| Your Data | Recommended Chunker | Decided By |
|---|---|---|
| Uniform prose, baseline/prototyping | Fixed-size (token-based) | Simplicity |
| General RAG prose | Sliding window ~20% | Boundary-spanning answers |
| Markdown / legal / code | Structure-aware (headers, clauses, functions) | Natural seams + provenance |
| Mixed, quality-first | Semantic (+ `max_tokens` cap) | Cut at meaning shifts |
| Chat / transcripts | By turn/speaker | The utterance is the unit |
| Dense reference QA | Small-to-big (parent-doc) | Precision + context |
| Tables / CSV | Row / schema-aware | Don't split records |

### Heuristic Starting Points (from Pre-Read)

| Corpus Type | Chunk Size | Overlap |
|---|---|---|
| Short FAQs, dialogue | 200–400 tokens | 0–50 tokens |
| Technical docs, manuals | 500–800 tokens | 50–100 tokens |
| Long-form articles, books | 800–1500 tokens | 100–200 tokens |
| Code | Function or class boundary | Full enclosing scope |
| Tables / structured | Whole row or whole table | — |

---

## 6.10 Measuring Chunking Quality — The Bake-Off Protocol

The chunk-size knob alone often moves retrieval failure more than a model swap. Always measure:

```python
# Evaluation metrics
Hit@K = (# queries where ≥1 relevant chunk is in top-K) / total_queries
MRR   = mean of (1 / rank of first relevant chunk)

# The sweep protocol
for name, splitter in chunking_configs.items():
    chunks = chunk_corpus(splitter, name)
    index, vecs = embed_and_index(chunks)
    hits = retrieve(index, EMB, EVALSET, chunks, k=10)
    score_retrieval(hits, EVALSET, k=10)  # → hit@10, MRR

best = res_df.iloc[0]["strategy"]
CHUNKS = store[best]["chunks"]   # freeze — winner reused downstream
```

**Rule:** The best chunker is the one that maximises Hit@K and MRR on your evaluation set — not the most sophisticated one.

---

## 6.11 Tuning Chunking — and the Overlap Myth

### Key Tuning Principles

- **Chunk on natural boundaries** (sentences, paragraphs) rather than arbitrary character limits
- **Use the embedding model's own tokeniser** to count tokens — character counts are misleading
- **Recursive splitting** (LangChain `RecursiveCharacterTextSplitter`) respects structure, then falls back to size
- **Match chunk size to content:** small (200–400 tokens) for Q&A facts; larger (800–1500 tokens) for narrative/code

### The Overlap Myth

> "Always use 10–20% overlap" — this is a widely repeated rule that a 2026 arXiv study showed had **no measurable benefit** in at least one rigorous experimental setup, only higher indexing cost.

The takeaway: **Test, don't assume.** Overlap helps when facts straddle chunk boundaries in your specific data. It doesn't help when it doesn't. The only way to know is to measure.

### Semantic Chunking Speed Warning

Semantic chunking embeds every sentence at ingest time — it is ~14× slower than token splitting (Chonkie benchmark). Budget this cost only when your recall metrics actually move after the bake-off.

---

## 6.12 Five Ways to Chunk — Visual Summary

```
ONE DOCUMENT:  ¶         ¶         ¶         ¶         ¶
               |         |         |         |         |

fixed          [████][████][████][████][████][████]     cuts mid-sentence
               400/0

token          [██][██][██][██][██][██][██][██]         fits model budget
               128/24

recursive      [████████████][████████][████][████████] prefers natural breaks
               600/120

sliding        [████][████][████][████][████][████]     overlap → recall
               overlap overlap overlap overlap
               5 sent / 3 stride

semantic       [████████████████][███████████████████]  splits at topic shifts
               pct90

                   too small ◄────────────────► too large
                   no context                   blurry vector
                             ↑ sweet spot?
                               measure → bake-off
```

---

## 6.13 Learning Thoughts

> **Thought 1:** The chunk is the atomic unit of RAG. Everything upstream (document processing) serves chunking. Everything downstream (embedding, indexing, retrieval) operates on chunks. If you get chunking right, the rest becomes easier. If you get it wrong, no model upgrade saves you.

> **Thought 2:** The overlap myth is a powerful reminder to measure. Many RAG practitioners blindly set 10–20% overlap because every tutorial recommends it. The research shows it helps sometimes and not others. The engineering discipline is: run your bake-off, measure Hit@K, then decide.

> **Thought 3:** Semantic chunking sounds like the obvious winner — chunks aligned with meaning! But the 14× speed penalty at ingest time is real. On a 10M-chunk corpus, this means weeks instead of hours. Pay that cost only when the recall gain is confirmed by measurement.

> **Thought 4:** Structure-aware chunking is underappreciated. When your documents have natural structure (headings, legal clauses, code functions), that structure is the author's gift to you — use it. A chunk that *is* a complete section retrieves better, is easier to cite, and makes more sense to the LLM.

> **Thought 5:** Always count tokens, not characters. If your embedding model's tokeniser tokenises "ChatGPT" as 3 tokens but you're counting 7 characters, your chunk size estimates are wrong. Use the actual tokeniser from the model you're deploying.

---

## 6.14 Important Interview Questions

**Conceptual**

1. **Why do we need to chunk documents in RAG, even with long-context LLMs?**
   - (1) Embedding models have token limits — longer inputs are silently truncated. (2) A single vector summarises its input — embedding a whole document dilutes meaning into uselessness. (3) LLM context costs scale with tokens — you want 3–5 relevant chunks, not the full document. (4) Retrieval precision: the chunk is the retrieval unit, so it must be self-contained.

2. **What is the core tension in choosing chunk size?**
   - Too small loses context ("it rose 40%" — what rose?). Too large dilutes the embedding (many topics averaged into one vector) and over-retrieves (returns pages when you needed a paragraph). The right size is the one that holds one complete answer in your data — measured by bake-off.

3. **What is sliding window chunking and what problem does overlap solve?**
   - Fixed-size chunks with overlapping content (e.g., 512 tokens, 100-token overlap). The overlap ensures a fact that straddles a chunk boundary appears intact in at least one chunk. Without overlap, boundary facts are split across two chunks and neither retrieves well.

4. **What is semantic chunking? What are its advantages and limitations?**
   - Embeds every sentence, computes cosine distance between consecutive sentences, splits where distance exceeds a threshold (topic shift). Advantage: chunks align with meaning shifts, improving retrieval precision. Limitations: ~14× slower at ingest, sensitive threshold, must cap max tokens, pay only when metrics move.

5. **What is small-to-big (parent-document) chunking?**
   - Embed small chunks (sentences) for retrieval precision, but return larger parent windows (paragraphs/sections) to the LLM for context. Precise retrieval + rich context in one design.

6. **What is the overlap myth and what does it teach us?**
   - Overlap (10–20%) is widely recommended but a 2026 arXiv study found no measurable benefit in their setup. It teaches us that chunking choices must be empirically validated on your own data, not adopted from tutorials as universal truths.

**Applied / Design**

7. **You are building RAG over a codebase. How do you chunk the source code?**
   - Split by function/class definitions using AST-aware splitting (LangChain's language-aware `RecursiveCharacterTextSplitter`). Never split by token count — a half-function is useless to both the retriever and the LLM. Include the full enclosing scope as overlap for inner functions.

8. **You have a RAG system over legal contracts. The contracts have numbered clauses (§1.1, §1.2, §7.2, etc.). What chunking strategy do you use?**
   - Structure-aware chunking on clause boundaries. Each chunk = one clause, prefixed with its reference number. This gives precise retrieval ("termination clause" → §7.2 chunk) and natural provenance for citation. Avoid fixed-size splitting that would split clauses mid-content.

9. **How do you evaluate which chunking strategy is best for your corpus?**
   - Build a labelled evaluation set (EVALSET) with queries and their known relevant chunks. For each chunking strategy: chunk corpus → embed → build flat index → retrieve top-K → measure Hit@K and MRR. The strategy with the highest Hit@K/MRR wins. The chunk-size knob alone often moves retrieval quality more than a model swap.

10. **What chunk sizes would you start with for: (a) a customer support FAQ, (b) a technical manual, (c) a book?**
    - (a) FAQ: 200–400 tokens, 0–50 overlap — questions/answers are short and self-contained. (b) Technical manual: 500–800 tokens, 50–100 overlap — explanations span multiple sentences. (c) Book: 800–1500 tokens, 100–200 overlap — narrative requires more context per chunk.

---

## 6.15 Section Summary

| Concept | One-line summary |
|---|---|
| Why chunk | Precision in retrieval; model token limits; LLM context cost |
| The goal | One self-contained retrievable idea per chunk |
| Fixed-size | Simplest baseline; fast; can cut mid-sentence |
| Sliding window | Fixed + overlap; boundary-safe; the sane default (~20% overlap) |
| Recursive | Try natural boundaries first, fall back to size; respects structure |
| Structure-aware | Split on headings/clauses/functions — best when docs have explicit structure |
| Semantic chunking | Cut at meaning shifts; ~14× slower; pay only when metrics move |
| Small-to-big | Embed small for precision, return large for context |
| The overlap myth | Overlap helps sometimes — measure, don't assume |
| Evaluation | Hit@K and MRR on your EVALSET — chunk-size knob often beats model swap |

---

*Previous: [Section 5 — Document Processing](S5_Document_Processing.md)*
*Next: [Section 7 — Vector Stores & ANN Indexing](S7_Vector_Stores_ANN_Indexing.md)*
