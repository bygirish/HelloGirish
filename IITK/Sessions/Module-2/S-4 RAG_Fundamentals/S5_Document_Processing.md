# Section 5 — Document Processing

> **Session:** IIT-KGP Module 2 · Session 4 · Instructor: Pawan Goyal
> **Theme:** *"Garbage in, garbage out." 80% of bad RAG answers start with poor document processing. The most unglamorous yet most impactful step.*

---

## 5.1 Why Document Processing Matters More Than the Model

> "Retrieval quality is capped by how cleanly you extract text and structure before chunking."

In tutorials, "documents" are clean strings. In industry, they are messy artefacts of human work:
- PDFs designed for print, not parsing
- Word documents with tracked changes and footnotes
- Scanned images masquerading as text files
- HTML pages packed with navigation, ads, cookie banners

**The governing question:** *What physical form is the text in, and how much of its value is in layout?*

Get this wrong and every downstream step — chunking, embedding, retrieval — inherits the damage. The instructor's rule: **80% of bad RAG answers are decided at document processing time.**

> **Industry reality:** Roughly 60–80% of engineering time in a real RAG project is spent on document extraction and chunking, not on the "AI parts." Budget accordingly.

---

## 5.2 The Two Stages: Extraction vs Parsing

These terms are often confused:

| Stage | What It Does | Example Output |
|---|---|---|
| **Extraction** | Pulls raw characters from the file format | "Revenue Q1 100 Q2 120 Revenue..." |
| **Parsing** | Labels what each block is (heading, table row, caption, footnote) and keeps relationships | `{type: "table", rows: [["Q1", "100"], ["Q2", "120"]]}` |

**For RAG, structure is as valuable as the text.** A chunk that *is* "§7.2 Termination" retrieves precisely for "termination clause" queries AND carries provenance you can cite. Raw extraction loses this; parsing preserves it.

---

## 5.3 Document Format Guide

### PDF

The hardest common format. PDFs are **built for visual fidelity, not structure**:
- No inherent reading order — they are pages of positioned glyphs
- No semantic tags — a "heading" and "body text" look identical to a naive parser
- Multi-column text bleeds across columns when read left-to-right
- Scanned PDFs are images — there is no text layer at all

**Three distinct PDF scenarios require three different approaches:**

#### Case 1: Plain prose PDF (papers, reports, specs)

**Choose:** `pypdf`

**Why:** 5–10× faster and lighter than anything else, and plain prose has no structure to preserve. Don't reach for a heavier tool to solve a problem you don't have.

```python
import pypdf
reader = pypdf.PdfReader("document.pdf")
text = "\n".join(page.extract_text() for page in reader.pages)
```

#### Case 2: Table-heavy PDF (invoices, financials, datasheets)

**Choose:** `pdfplumber` (or Camelot for ruled tables), serialize tables to Markdown

**Why:** `pypdf` flattens a table into a single word-salad line (`Q1 100 Q2 120 …`), destroying the row/column meaning your retriever needs. The speed cost buys back the numbers you came for.

```python
import pdfplumber

with pdfplumber.open("financial_report.pdf") as pdf:
    for page in pdf.pages:
        tables = page.extract_tables()
        text = page.extract_text()
        # Serialize table to markdown for embedding
        for table in tables:
            md_table = "| " + " | ".join(table[0]) + " |\n"  # header
            md_table += "|" + "---|" * len(table[0]) + "\n"  # separator
            for row in table[1:]:
                md_table += "| " + " | ".join(str(c) for c in row) + " |\n"
```

#### Case 3: Scanned / image-only PDF (no text layer)

**Choose:** OCR — local `tesseract` (+ `pdf2image`), or cloud OCR / a vision model for hard scans

**Why:** There is literally no text to extract; `pypdf` returns empty strings *silently*. This is the trap that ships empty chunks to production.

**Always probe first:**
```python
def pdf_has_text_layer(path, sample=3):
    import pypdf
    r = pypdf.PdfReader(path)
    return any((p.extract_text() or "").strip() for p in r.pages[:sample])
    # False → route to OCR, never to pypdf
```

**OCR DPI guidance:** Use `dpi=300` — this is the sweet spot. 72 dpi shreds small fonts; 600 dpi is ~4× slower for little gain. Cleaning the image (grayscale + binarisation) usually helps accuracy more than adding pixels.

```python
from pdf2image import convert_from_path
import pytesseract

def ocr_pdf(path):
    pages = convert_from_path(path, dpi=300)
    return "\n\n".join(pytesseract.image_to_string(p) for p in pages)
```

#### Case 4: Complex layout (multi-column, magazines, academic PDFs, forms)

**Choose:** A layout-aware parser — `Unstructured`, `docling` (IBM), `GROBID` (for papers), Azure Document Intelligence, or a VLM for truly messy ones

**Why:** `pypdf` reads coordinates left-to-right and scrambles multi-column reading order — column 1 line 1, column 2 line 1, column 1 line 2... producing interleaved nonsense. A layout-aware parser recovers the correct reading flow.

---

### DOCX

Easier than PDF — it is structured XML under the hood.

```
document.docx = ZIP(
  word/document.xml   ← body text with heading/paragraph styles
  word/styles.xml     ← heading level definitions
  word/tables.xml     ← structured table data
)
```

- Headings, lists, tables are already labelled in the XML
- Easiest and most reliable to parse
- **Choose:** `python-docx` (standard), or `unstructured` for tables, footnotes, and comments

```python
import docx

doc = docx.Document("report.docx")
sections = []
for para in doc.paragraphs:
    if para.style.name.startswith("Heading"):
        sections.append({"type": "heading", "level": int(para.style.name[-1]), "text": para.text})
    else:
        sections.append({"type": "paragraph", "text": para.text})
```

---

### HTML

Easy to parse but full of noise (navigation, ads, scripts, cookie banners). Raw HTML is ~90% navigation, footers, and boilerplate.

**Two scenarios:**

| Scenario | Tool | Why |
|---|---|---|
| You control the site / stable layout | `BeautifulSoup` with targeted selectors | High precision — you keep exactly what you target |
| Scraping the open web at scale | `trafilatura` / `readability` | Every site differs; you can't hand-write selectors for all of them; heuristic generalises |

**Critical:** Strip boilerplate *before* extracting text or every chunk is polluted with "Home | About | Contact."

```python
# Controlled site
from bs4 import BeautifulSoup
soup = BeautifulSoup(html, "html.parser")
main_content = soup.find("main") or soup.find("article")
text = main_content.get_text(separator="\n", strip=True)

# Open web
import trafilatura
text = trafilatura.extract(html, include_tables=True)
```

---

### The Tool Selection Table

| Document Condition | Recommended Tool | Decided By |
|---|---|---|
| Plain text PDF | `pypdf` | Speed, no structure to keep |
| Table-heavy PDF | `pdfplumber` / Camelot | Tables die under pypdf |
| Scanned PDF | OCR (tesseract / VLM) | No text layer — probe first! |
| Multi-column / forms | `Unstructured` / `docling` / Azure DI | Reading order + element tags |
| HTML, controlled | `BeautifulSoup` | Precision |
| HTML, open web | `trafilatura` | Robustness across layouts |
| DOCX | `python-docx` + headings | Free section metadata |
| Mixed at scale | Router + `Unstructured` | One entry point |

---

## 5.4 OCR — Optical Character Recognition in Depth

**OCR (Optical Character Recognition)** recovers text from scanned/image PDFs and photographs of text.

**Without OCR:** `pypdf` returns empty strings with no warning. You index empty chunks. Retrieval returns nothing. No error.

### Tables Are the Hardest Element

Tables in scanned PDFs are the most challenging element to extract faithfully. Best practice: emit tables as **HTML** to preserve row/column structure (often better than Markdown for parsing).

### Layout-Aware OCR Toolkits

| Tool | Technology | Best For |
|---|---|---|
| **Unstructured** | Element ontology + Hi-Res layout | PDF/DOCX/HTML/images — general purpose |
| **Docling (IBM)** | DocLayNet layout + TableFormer tables; OCR built in | Academic and structured documents |
| **LlamaParse** | LLM-based parsing | Good on simpler docs, lightweight |
| **Cloud OCR** | Azure Document Intelligence, Google Document AI | Hard inputs (forms, tables, handwriting), pay-per-page |
| **LLM-based (2024+)** | GPT-4o, Claude with vision, Gemini | Complex layouts but expensive at scale |

**Layout-aware models recover:**
- Correct reading order (not just left-to-right coordinates)
- Column structure
- Heading hierarchy
- Table cell boundaries

---

## 5.5 The Clean & Normalise Step

After extraction/OCR, always clean and normalise before chunking:

```
raw_text
    │
    ▼
Remove headers/footers/nav (page numbers, "Confidential", running headers)
    │
    ▼
De-duplicate (adjacent identical lines, boilerplate repeated on every page)
    │
    ▼
Fix encoding & special chars (smart quotes → straight, broken Unicode, ligatures)
    │
    ▼
Normalise whitespace (multiple spaces → single, \r\n → \n)
    │
    ▼
clean_text → ready for chunking
```

---

## 5.6 Unified Document Loader Pattern

A practical pattern for handling multiple formats in a single codebase:

```python
from pathlib import Path
from langchain.schema import Document

def load_document(path) -> list[Document]:
    ext = Path(path).suffix.lower()

    if ext == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader
        docs = PyPDFLoader(path).load()
        if n_chars(docs) < 32:          # scanned? (too few chars extracted)
            docs = [Document(page_content=ocr_pdf(path))]

    elif ext == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader
        docs = Docx2txtLoader(path).load()

    elif ext in (".html", ".htm"):
        from langchain_community.document_loaders import BSHTMLLoader
        docs = BSHTMLLoader(path).load()

    elif ext in (".png", ".jpg", ".jpeg"):
        docs = [Document(page_content=ocr_image(path))]

    return docs


def ocr_pdf(path):               # image-only PDF fallback
    from pdf2image import convert_from_path
    import pytesseract
    pages = convert_from_path(path, dpi=200)
    return "\n\n".join(pytesseract.image_to_string(p) for p in pages)
```

---

## 5.7 Learning Thoughts

> **Thought 1:** The #1 RAG anti-pattern is spending 90% of project time on model selection and 10% on document processing — when the ratio should be reversed. A bad document parser that produces empty chunks or garbled tables will make even the best embedding model useless. Fix the foundation first.

> **Thought 2:** "Probe before routing" is the most important engineering habit for PDF processing. Never assume a PDF has a text layer. Always check `pdf_has_text_layer()` and route to OCR if it returns False. The silent empty-string failure of pypdf on scanned PDFs is the most common production bug in RAG pipelines.

> **Thought 3:** The distinction between extraction and parsing is not academic. For a 100-page financial report, `pypdf` extracts text in ~0.5 seconds but loses all table structure. `pdfplumber` takes ~5 seconds but preserves row/column relationships. Choose based on whether layout carries meaning — not based on speed alone.

> **Thought 4:** HTML is deceptively hard. The problem is not parsing (BeautifulSoup handles that easily) — it is boilerplate removal. A chunk containing "Home | Product | Pricing | About | Contact | Privacy Policy" before the actual content will poison your index. Always strip navigation and footers before chunking.

> **Thought 5:** OCR at `dpi=300` + grayscale + binarisation is the practical sweet spot that most teams discover after trying lower DPI and getting bad results. The insight is that image quality matters more than resolution — a clean 300 DPI scan beats a blurry 600 DPI scan.

---

## 5.8 Important Interview Questions

**Conceptual**

1. **Why is document processing considered the most impactful step in a RAG pipeline?**
   - "Garbage in, garbage out." Retrieval quality is capped by extraction quality. If text is incorrectly extracted (garbled tables, scrambled multi-column order, empty OCR), chunking and embedding produce meaningless vectors. 80% of bad RAG answers trace back to poor document processing.

2. **What is the difference between text extraction and document parsing?**
   - Extraction pulls raw characters from the file format. Parsing goes further — it labels each block (heading, table row, caption, footnote) and preserves relationships. For RAG, parsed structure is as valuable as the text because it enables structure-aware chunking and provenance tracking.

3. **What is the scanned PDF trap and how do you avoid it?**
   - `pypdf` returns empty strings on scanned PDFs with no warning. You index empty chunks; retrieval returns nothing. Avoid by always probing for a text layer (`pdf_has_text_layer()`) before choosing the extraction tool. Route to OCR if the probe returns False.

4. **Why should tables be serialized to Markdown/HTML rather than extracted as plain text?**
   - Plain text extraction (pypdf) flattens a table to a word-salad line: "Q1 100 Q2 120 Revenue Total..." The row/column structure that carries meaning is destroyed. Serialising to Markdown preserves the headers/rows, which the embedding model and LLM can use.

5. **What is OCR and why does DPI matter?**
   - OCR (Optical Character Recognition) recovers text from images of text. DPI (dots per inch) determines image resolution. 72 DPI is too low (small fonts are illegible). 600 DPI is overkill (4× slower for minimal gain). 300 DPI is the standard sweet spot. Image cleaning (grayscale + binarisation) improves accuracy more than raising DPI.

**Applied / Design**

6. **You are building a RAG system over a corpus of 10,000 PDFs — a mix of plain reports, financial tables, and scanned contracts. How do you approach document processing?**
   - Build a router that probes each PDF: (1) Has text layer? → Yes: check for tables (pdfplumber) or plain prose (pypdf). (2) No text layer → OCR with tesseract at DPI=300. (3) Multi-column layout? → Use layout-aware parser (Unstructured/docling). Normalise all output to clean text before chunking.

7. **Your RAG system returns financially incorrect numbers. The embedding model is strong. What's likely wrong?**
   - The document parser is destroying table structure. `pypdf` flattening "Revenue Q1 $100M Q2 $120M" into a word-salad means the embedding model embeds garbled text. The embedding model is fine; the extraction is broken. Fix: use `pdfplumber` or a layout-aware parser to serialize tables as Markdown.

8. **What boilerplate removal strategy do you use for HTML documents from the web?**
   - Use `trafilatura` or `readability-lxml` which apply heuristics to extract main content and strip navigation/ads/footers. For controlled internal sites, use BeautifulSoup with targeted selectors (`main`, `article` tags) and explicit exclusion of nav/footer/aside elements.

---

## 5.9 Section Summary

| Concept | One-line summary |
|---|---|
| Why it matters | 80% of bad RAG answers are decided at document processing — "garbage in, garbage out" |
| Extraction vs parsing | Extraction pulls characters; parsing labels structure (headings, table cells) — both needed |
| PDF plain prose | `pypdf` — fastest, no structure needed |
| PDF with tables | `pdfplumber` / Camelot — serialize to Markdown to preserve row/column |
| Scanned PDF | OCR (tesseract at 300 DPI) — probe for text layer first |
| Multi-column PDF | Layout-aware parser (Unstructured, docling, GROBID, Azure DI) |
| HTML | BeautifulSoup (controlled) / trafilatura (open web) — strip boilerplate first |
| DOCX | `python-docx` — structured XML makes it easiest to parse |
| OCR sweet spot | 300 DPI + grayscale + binarisation — quality > resolution |
| Engineering reality | 60–80% of RAG project time is spent on document processing |

---

*Previous: [Section 4 — Choosing an Embedding Model](S4_Choosing_Embedding_Model.md)*
*Next: [Section 6 — Chunking Strategies](S6_Chunking_Strategies.md)*
