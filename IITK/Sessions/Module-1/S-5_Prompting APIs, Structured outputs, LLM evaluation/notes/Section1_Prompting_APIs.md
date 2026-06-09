# Section 1 — Prompting APIs
### Lecture 5 · IIT Kharagpur · Prof. Sourangshu Bhattacharya

---

## What This Section Covers

| Topic | Slide Range |
|-------|-------------|
| Three ways to run open-source LLMs | 2–6, 56–60 |
| Designing prompts for tasks | 9–22 |
| Prompt types: summarization, QA, classification, reasoning | 10–22 |

---

## Part 1A — Three Ways to Run Open-Source LLMs

### The Big Picture

When you want to use an LLM in your application or experiment, you have three fundamentally different execution models. Each trades off **control** vs **convenience** vs **cost**.

```
┌──────────────────────┬──────────────────────┬──────────────────────┐
│  Approach 1          │  Approach 2           │  Approach 3          │
│  HuggingFace Local   │  HF Inference API     │  Ollama              │
│  (Transformers lib)  │  (Hosted, REST)       │  (Local REST server) │
├──────────────────────┼──────────────────────┼──────────────────────┤
│  Full control        │  Zero setup           │  Best of both worlds │
│  Privacy: complete   │  Privacy: sent to HF  │  Privacy: complete   │
│  Cost: free          │  Cost: rate-limited   │  Cost: free          │
│  GPU: optional       │  GPU: not needed      │  GPU: optional       │
│  Logit access: YES   │  Logit access: NO     │  Logit access: NO    │
│  Quality: model-dep  │  Quality: model-dep   │  Quality: model-dep  │
└──────────────────────┴──────────────────────┴──────────────────────┘
```

---

### Approach 1: HuggingFace Transformers (Local)

**What it is:** You download the model weights to your machine and run inference yourself using PyTorch and the `transformers` library.

**Mental model:** Think of it like owning a physical calculator. You control every button. You can peek inside. But you have to carry it.

#### How it works — step by step

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"

# Step 1: Load the tokenizer — the vocabulary + encoding rules
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 2: Load the model weights (~750 MB for SmolLM2-360M)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,   # float32 for CPU stability
    device_map="auto"            # uses GPU if available, else CPU
)

# Step 3: Tokenize your prompt — text → integer IDs
prompt = "What is quantum computing? Explain in 3 sentences."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Step 4: Generate — the neural net predicts tokens one by one
outputs = model.generate(**inputs, max_new_tokens=120, do_sample=False)

# Step 5: Decode — integer IDs → human-readable text
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

#### What `apply_chat_template` does and why it matters

Instruction-tuned models are fine-tuned with a specific conversation format (e.g., `<|im_start|>system\n...<|im_end|>`). If you just concatenate strings directly, the model won't recognize the role boundaries and will produce garbage or ignore the system prompt.

```python
messages = [
    {"role": "system", "content": "You are a helpful AI assistant. Be concise."},
    {"role": "user",   "content": "List 3 benefits of open-source LLMs."}
]

# apply_chat_template inserts the correct special tokens for this model
chat_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,            # return string first, not tensor
    add_generation_prompt=True # adds the "assistant:" cue at the end
)
```

> **Learning Thought:** `apply_chat_template` is the difference between a model that follows instructions and one that treats the system prompt as part of the story. Always use it for instruction-tuned models.

#### Key parameters in `model.generate()`

| Parameter | Effect | When to use |
|-----------|--------|-------------|
| `max_new_tokens` | Cap on tokens to generate | Always set this — prevents infinite loops |
| `do_sample=False` | Greedy decoding — always picks highest-prob token | Deterministic tasks: QA, extraction |
| `do_sample=True` | Multinomial sampling — probabilistic | Creative tasks: summarization, generation |
| `temperature` | Scales distribution before sampling (lower = sharper) | Controls creativity |
| `repetition_penalty` | Penalises tokens that already appeared | Stops copy-pasting the prompt |
| `skip_special_tokens=True` | Removes `<eos>`, `<pad>` from output | Almost always use this in decode |

#### Decoding only the new tokens (important pattern)

```python
# The model returns the ENTIRE sequence (prompt + generated)
# Slice off the prompt portion to get only what was generated:
new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
result = tokenizer.decode(new_tokens, skip_special_tokens=True)
```

---

### Approach 2: HuggingFace Hosted Inference API

**What it is:** HuggingFace runs the model on their servers. You send HTTP requests with your text and receive responses. You never download weights.

**Mental model:** Think of it like calling a restaurant. You describe what you want, they cook it on their hardware, and deliver the result. You don't see the kitchen.

```python
import requests

HF_TOKEN = "hf_..."
API_URL  = "https://router.huggingface.co/v1/chat/completions"
headers  = {"Authorization": f"Bearer {HF_TOKEN}"}

payload = {
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "messages": [
        {"role": "user", "content": "What is quantum computing? Answer in 2 sentences."}
    ],
    "max_tokens": 120
}

response = requests.post(API_URL, headers=headers, json=payload)
print(response.json()["choices"][0]["message"]["content"])
```

**Or using the official Python SDK (`huggingface_hub`):**
```python
from huggingface_hub import InferenceClient

client = InferenceClient(provider="novita", api_key=HF_TOKEN)
completion = client.chat.completions.create(
    model="meta-llama/Llama-3.2-1B-Instruct",
    messages=[{"role": "user", "content": "Explain RAM vs ROM simply."}],
    max_tokens=150,
)
print(completion.choices[0].message.content)
```

> **Why choose this?** Perfect for quick prototypes and demos where you don't want to manage model weights. Free tier with rate limits — good for students and solo developers.

---

### Approach 3: Ollama (Local REST Server)

**What it is:** Ollama runs as a local server on your machine and exposes a REST API that mimics OpenAI's interface. You download models once via CLI, then interact via HTTP. No Python model loading involved.

**Mental model:** It's like having a miniature OpenAI server running on your laptop. You interact with it exactly like you would the cloud — just at a different URL.

**Setup (run once in terminal):**
```bash
# macOS
brew install ollama
ollama serve           # starts the server on localhost:11434
ollama pull llama3     # downloads ~4.7 GB model once
```

```python
import requests

# /api/generate — raw prompt
payload = {
    "model": "qwen2.5:3b",
    "prompt": "What is quantum computing? Explain in 2 sentences.",
    "stream": False
}
response = requests.post("http://localhost:11434/api/generate", json=payload)
print(response.json()["response"])

# /api/chat — OpenAI-compatible messages format
chat_payload = {
    "model": "qwen2.5:3b",
    "messages": [
        {"role": "system", "content": "You are a concise technical assistant."},
        {"role": "user",   "content": "Name 3 advantages of running LLMs locally."}
    ],
    "stream": False
}
response = requests.post("http://localhost:11434/api/chat", json=chat_payload)
print(response.json()["message"]["content"])
```

> **Unique advantage of Ollama:** You can swap your entire backend from Ollama to OpenAI by just changing the URL and auth header. The messages format is identical — zero code changes in your application logic.

---

### Full Comparison Table

| Criterion | HF Transformers (Local) | HF Inference API | Ollama |
|-----------|------------------------|-----------------|--------|
| Setup complexity | High | Low | Medium |
| Internet required | No | Yes | No |
| Data privacy | Full | Sent to HF | Full |
| Raw logit access | **Yes** | No | No |
| Custom logits masking | **Yes** | No | No |
| Model quality ceiling | Any open model | Any open model | Any open model |
| Free to use | Yes | Yes (rate-limited) | Yes |
| Best use case | Research, privacy, advanced control | Quick prototyping | Daily local use |
| OpenAI API compatible | No | Yes (via router) | Yes |

---

## Part 1B — Designing Prompts for Different Tasks

### What is Prompt Engineering?

Prompt engineering is the practice of crafting inputs to LLMs to reliably produce desired outputs **without retraining the model**. The model's knowledge is fixed; the prompt controls what slice of that knowledge is activated and in what format it's expressed.

```
Input Prompt → [Frozen LLM] → Output
                     ↑
        Prompt engineering controls this entire flow
```

---

### Task 1: Text Summarization

**Goal:** Condense a long document into a shorter representation while preserving key information.

#### Why LLMs are exceptional at summarization
LLMs are trained on billions of document-summary pairs implicitly — news articles with headlines, papers with abstracts, books with blurbs. They don't need examples to summarize; they've seen the pattern millions of times.

#### Summarization Prompt Patterns

**Pattern 1: Zero-shot instruction**
```
Summarize the following text in one sentence:
[TEXT]
One-sentence summary:
```

**Pattern 2: Domain-specific with constraints**
```
Summarize the following earnings call transcript for a retail investor.
Focus on quarterly revenue growth and forward-looking risks.
Keep it under 100 words.
[TRANSCRIPT]
```

**Pattern 3: Analogy-based (for non-expert audiences)**
```
Read the provided scientific abstract on CRISPR technology.
Provide a 1-sentence TL;DR for a non-scientific audience using simple analogies.
[ABSTRACT]
TL;DR:
```
> Model response: *"Think of CRISPR like a 'find and replace' tool for DNA, allowing scientists to cut out genetic typos and paste in healthy code to cure diseases."*

#### Key insight: Format signals shape outputs
- Ending your prompt with `One-sentence summary:` constrains length
- Specifying audience ("for a retail investor", "for a non-expert") shifts vocabulary and focus
- Specifying format ("numbered list", "under 100 words") gives structural constraints

> **Learning Thought:** The most powerful summarization prompts specify THREE things: (1) who the audience is, (2) what to focus on, and (3) a length/format constraint. Missing any one of these typically produces generic outputs.

---

### Task 2: Question Answering

**Two fundamental QA modes:**

#### Mode A: Context-Bound QA (Closed-Book / RAG-style)

The model must answer from a provided passage only. If the answer isn't in the passage, it should say so.

```
Answer the question based on the context below. Keep the answer short and concise.
Respond "Unsure about answer" if not sure.

Context: [YOUR PASSAGE]
Question: [YOUR QUESTION]
Answer:
```

**Why the "Unsure about answer" escape clause is critical:**
Without it, LLMs will hallucinate a plausible-sounding answer. The explicit escape clause activates the model's pattern for "I don't know" responses — it's been trained on countless such patterns in the data.

**Example (policy QA):**
```
Context: Our policy states that employees get $500/year for professional development.
         Request must be filed 30 days prior.

Question: Can I get reimbursed for a $200 course I bought yesterday without prior approval?

Answer: No. According to the policy, requests must be filed 30 days prior to the course.
        Since you purchased it yesterday without prior approval, it does not meet the guidelines.
```

#### Mode B: Open-Domain QA (Open-Book)

The model uses its trained world knowledge to answer. No context passage needed.

```
Identify the three primary constraints of the James Webb Space Telescope
regarding its cooling mechanisms. Format as a numbered list.
```

> **Learning Thought:** Context-bound QA is the foundation of **Retrieval-Augmented Generation (RAG)** — the most widely used LLM architecture in production. The idea: retrieve relevant documents → put them in context → ask the model to answer from context only. This prevents hallucination and keeps answers grounded.

---

### Task 3: Text Classification

**Goal:** Assign one or more labels to a piece of text.

#### Zero-shot classification
The model classifies without any examples — it relies purely on its understanding of the label names.

```
Classify the text into neutral, negative or positive.
Text: I think the food was okay.
Sentiment:
```
> Output: `Neutral`

#### Few-shot classification
Providing 2–5 labelled examples before the test input significantly improves accuracy and output format consistency.

```
Classify sentiment as Positive, Neutral, or Negative.

Text: "The coffee was fine."
Sentiment: Neutral — the word 'fine' is non-committal with no strong emotion.

Text: "Best pizza I've ever had!"
Sentiment: Positive — superlative praise with strong positive emotion.

Text: "The delivery took forever, but the food was amazing."
Sentiment:
```
> Output: `Mixed (Neutral) — The user expresses dissatisfaction with logistics (Negative) but high satisfaction with product quality (Positive).`

**Why few-shot works better:** The examples lock in the exact output FORMAT (label + reasoning sentence). The model mimics the pattern rather than inventing its own.

#### Multi-label classification

```
Assign relevant tags to this support ticket: "I cannot log in and I haven't received my password reset email."
Tags: [Auth, Billing, Technical, Urgent]
```
> Output: `Primary Tags: 1. Auth (Login issue)  2. Technical (Email delivery failure)  3. Urgent (User blocked)`

> **Learning Thought:** The difference between zero-shot and few-shot isn't just accuracy — it's **format control**. Few-shot examples are templates. The model fills in the pattern. This is why few-shot is preferred in production: you control exactly what the output looks like.

---

### Task 4: Code Generation

LLMs are trained on enormous code datasets (GitHub, Stack Overflow, documentation). Code generation prompts should specify:
- Language + approach (iterative vs recursive)
- Whether to include comments
- Input/output signature if known

```
Write a Python function to calculate the Fibonacci sequence up to n terms
using an iterative approach. Include basic comments.
```

**Key pattern: Logical reasoning tasks**
```
A farmer needs to cross a river with a fox, a chicken, and a bag of grain.
He can only carry one at a time. If left alone, fox eats chicken; chicken eats grain.
How can he cross safely?
```
> These multi-step logical planning problems test the model's ability to maintain state constraints across multiple steps — a core reasoning benchmark.

---

### Task 5: Chain-of-Thought (CoT) Reasoning

**The single most impactful prompting technique for reasoning tasks.**

**Without CoT:**
```
Sally has 3 brothers. Each brother has 2 sisters. How many sisters does Sally have?
→ Often wrong: models shortcut to "2" because that's what they see
```

**With CoT:**
```
Sally has 3 brothers. Each brother has 2 sisters. How many sisters does Sally have?
Think step by step.

Step 1: Sally is a girl.
Step 2: Each of her 3 brothers has 2 sisters.
Step 3: Sally is one of those sisters, so there must be exactly 1 other sister.
Step 4: Therefore, Sally has 1 sister.
```

**Why it works:** The phrase "think step by step" activates a generation pattern where the model is forced to emit intermediate reasoning tokens before the final answer. Those intermediate tokens act as a "scratchpad" that constrains the final prediction.

**Variants of CoT:**
| Variant | Trigger | Best for |
|---------|---------|----------|
| Standard CoT | "Think step by step" | Math, logic |
| Zero-shot CoT | "Let's think step by step" | Any reasoning task |
| Self-consistency CoT | Sample multiple CoT paths, take majority vote | High-stakes decisions |
| Tree-of-Thought | Explore multiple reasoning branches | Open-ended planning |

> **Learning Thought:** CoT is not magic — it works because LLMs are trained on human-written text, and humans write out reasoning before conclusions. "Think step by step" is essentially activating the reasoning-before-conclusion training pattern. The longer the token path, the more constrained and correct the final answer tends to be.

---

### Where Functions Execute: A Common Misunderstanding

**Q: When we use OpenAI's `tools` option to call a function, where does the function actually execute?**

**A: On the user's machine (your machine / your servers).**

OpenAI does NOT execute your functions. Here is the actual flow:

```
1. You send: messages + tools (function definitions/schemas)
2. OpenAI returns: a response saying "call function X with args Y"
3. YOUR code executes the function with those args
4. You send the result back to OpenAI as a tool_result message
5. OpenAI generates the final natural language response
```

The model only generates the *call instruction*. Execution always stays on your side.

---

### Tasks NOT Suitable for LLMs (Important!)

**Q: Which of the following is NOT suitable for LLMs?**

- ~~Classifying resumes into skillsets~~ → LLMs are great at this
- **Forecasting stock prices** → LLMs are NOT suitable for this
- ~~Generating war update summaries~~ → LLMs can do this
- ~~Creating a resume for a job application~~ → LLMs are great at this

**Why not stock price forecasting?**
- LLMs have a training data cutoff — they don't know current prices
- Stock prices depend on real-time quantitative signals — not language patterns
- LLMs cannot do stochastic time-series modeling from first principles
- They will confidently hallucinate numbers
- **Appropriate tools:** Statistical time-series models (ARIMA, LSTMs), quantitative trading models

> **Learning Thought:** LLMs are fundamentally language pattern matchers trained on human text. Any task that requires: (1) real-time numerical data, (2) precise mathematical computation, (3) causal reasoning about physical systems, or (4) guaranteed factual accuracy — falls outside their reliable capability.

---

## Interview Questions — Section 1

### Conceptual

**Q1. What is the key difference between using HuggingFace locally vs the HuggingFace Inference API, from an engineering standpoint?**

> Local: you control the entire inference pipeline — weights, logits, sampling parameters, masking. API: you only control the input text and a limited set of hyperparameters. The API is more convenient but loses access to internals like logit manipulation. Critical for use cases that need constrained generation.

**Q2. Why does `apply_chat_template` matter when using instruction-tuned models?**

> Instruction-tuned models are fine-tuned on data formatted with specific special tokens (e.g., `<|im_start|>`, `<|im_end|>`, `[INST]`, `[/INST]`). Without these tokens in the correct positions, the model doesn't know where the system prompt ends and the user turn begins — leading to poor instruction-following. `apply_chat_template` inserts the exact formatting the model was trained with.

**Q3. Explain greedy decoding vs sampling. When would you use each?**

> **Greedy decoding** (`do_sample=False`): at each step, picks the single highest-probability token. Deterministic — same prompt always gives same output. Use for: factual extraction, code generation, structured data extraction. **Sampling** (`do_sample=True`): picks tokens proportionally from the probability distribution. Stochastic — different outputs every run. Use for: creative writing, summarization, diverse generation. `temperature` controls the sharpness of sampling.

**Q4. What is Chain-of-Thought prompting and why does it work?**

> CoT prompting adds phrases like "think step by step" to force the model to generate intermediate reasoning tokens before the final answer. These intermediate tokens act as a scratchpad, constraining subsequent token predictions. It works because LLMs are trained on human text where reasoning typically precedes conclusions. The additional tokens between the question and the answer make the reasoning path more faithful to correct logic.

**Q5. What tasks are fundamentally unsuitable for LLMs, and why?**

> Tasks requiring: (1) real-time or post-cutoff data (stock prices, live news), (2) guaranteed mathematical precision (long arithmetic chains), (3) causal reasoning from physical models (structural engineering, physics simulations), (4) true randomness or cryptographic security. The root cause: LLMs are pattern matchers trained on human-generated text. They produce statistically likely continuations, not computationally verified outputs.

### Applied / Code-level

**Q6. In a production RAG system, why must you include "Respond 'Unsure about answer' if not sure" in your context-QA prompt?**

> Without the escape clause, the model has no learned pattern to fall back on — it will complete the `Answer:` token sequence with the most plausible-sounding text in its training data, regardless of whether the provided context supports it. This is hallucination. The escape clause provides a named safety valve that the model has seen in training data (documentation, customer service transcripts), activating its "I don't have enough information" response pattern.

**Q7. How would you implement batch text classification efficiently using an LLM API?**

> Instead of making N separate API calls (expensive, slow), combine all items into a single prompt and request JSON array output:
> ```
> Classify each text as Positive, Neutral, or Negative.
> Return JSON: [{"id": 1, "text": "...", "sentiment": "..."}]
> 1. "I think the food was okay."
> 2. "Absolutely loved it!"
> ```
> Parse the returned JSON array. One API call, N classifications. Trade-off: larger prompts cost more per call but far fewer calls total.

**Q8. You're building a customer support triage system. Which LLM approach (local / HF API / Ollama / cloud API) would you choose, and why?**

> Depends on constraints: If **data is sensitive** (PII in support tickets) — local model (Ollama or HF Transformers) to avoid sending data externally. If **response quality** is paramount — cloud API (GPT-4.1) with PII-scrubbing before the API call. If **cost is the constraint** — Groq free tier with Llama 3.3 for prototyping, then migrate to self-hosted Ollama. For **real production**: Ollama on-prem with a capable 7B-13B model is often the right balance.

---

## Key Learning Thoughts — Section 1

1. **The three approaches are not competing — they're complementary.** Use local HF for research and logit access, HF API for demos, Ollama for daily development, cloud APIs for production quality.

2. **Prompt engineering is about activating the right training patterns,** not "tricking" the model. The model has seen millions of summarization, QA, and classification examples. Your job is to frame the input in a way that activates the right pattern.

3. **Few-shot examples are format templates first, accuracy guides second.** Before worrying about which examples to pick, nail down what output format you want — then pick examples that demonstrate it exactly.

4. **CoT is the most universally applicable technique.** It costs only a few extra input tokens ("Think step by step") and reliably improves accuracy on anything involving multi-step reasoning — math, logic, planning, legal analysis.

5. **The "function execution is on your machine" insight** is foundational for building tool-use agents. OpenAI/Claude generates the *intent* to call a function. Your orchestration code executes it. This separation is what makes agentic systems safe and auditable.

6. **LLMs are text-in, text-out pattern matchers.** They are not calculators, databases, search engines, or forecasting models. Knowing what they are and what they are not prevents embarrassing production failures.

---

## Notebook Reference

| File | Covers |
|------|--------|
| `llm_prompting.ipynb` | Sections 1–5: all three approaches × all four APIs (Local HF, OpenAI, Azure, Groq) |
| `llm_structured_outputs.ipynb` | Sections covered in Section 2 and 3 notes |

### API Providers Covered in Notebook

| Provider | Model Used | Free? | Python SDK |
|----------|-----------|-------|------------|
| HuggingFace Local | SmolLM2-360M-Instruct | Yes | `transformers` |
| HuggingFace Inference API | Llama-3.2-1B-Instruct | Yes (rate-limited) | `huggingface_hub` |
| Ollama | qwen2.5:3b | Yes | `requests` |
| OpenAI | gpt-4.1 | Paid | `openai` |
| Azure OpenAI | gpt-4-1 deployment | Paid | `openai` (AzureOpenAI) |
| Groq | llama-3.3-70b-versatile | Yes (free tier) | `groq` |
