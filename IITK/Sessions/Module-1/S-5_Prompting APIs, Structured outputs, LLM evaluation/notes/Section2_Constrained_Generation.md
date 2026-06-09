# Section 2 — Structured Outputs: Constrained Generation & Decoding
### Lecture 5 · IIT Kharagpur · Prof. Sourangshu Bhattacharya

---

## What This Section Covers

| Topic | Slide Range |
|-------|-------------|
| What is constrained generation | 26 |
| Token-by-token generation mechanics | 27 |
| Logits, softmax, probability distribution | 28 |
| Masking mechanism (−∞ trick) | 29–30 |
| `LogitsMask` — token allowlist | 33–35 |
| `RegexMask` — regex-constrained generation | 36–38 |
| Partial regex matching | 37 |
| Multinomial sampling + constraints | 39 |
| `LogitsProcessor` vs Structured Decoding | 41 |
| OpenAI Structured Decoding / JSON Schema | 42 |

---

## Part 2A — Foundations: How LLMs Generate Text

### The Autoregressive Loop

Every modern LLM generates text **one token at a time**, in a loop. Each new token is conditioned on all previous tokens (the prompt + everything generated so far).

```
Prompt: "2, 4, 8,"
              ↓
         [Neural Net]
              ↓
         Logits over ALL ~50,000 vocab tokens
              ↓
         Softmax → Probabilities
              ↓
         Sample or Argmax → Pick token "16"
              ↓
         Append "16" to sequence
              ↓
         [Neural Net again, with "2, 4, 8, 16"]
              ↓
         ... picks "32" ... picks "64" ...
```

**Key insight:** The model does not "think ahead." It predicts the single best next token given what it has seen so far. This is called **autoregressive generation**. It has a crucial implication: if you let the model make a mistake at step 5, it will compound that mistake in all subsequent steps.

---

### What are Tokens?

Tokens are not characters and not words — they are **sub-word units** learned by the tokenizer during training. The vocabulary is typically 32,000–100,000 tokens.

```python
# Example tokenization:
tokenizer.encode("A list of colors: red, blue")
# >> [49, 1398, 282, 4683, 42, 2382, 28, 4461]

# Decode each ID to see the sub-word:
[tokenizer.decode(t) for t in [49, 1398, 282, 4683, 42, 2382, 28, 4461]]
# >> ['A', ' list', ' of', ' colors', ':', ' red', ',', ' blue']
```

**Notice:** Tokens carry their own leading space — `' list'` not `'list'`. This matters enormously for constrained generation: the token `'1'` and `' 1'` (with a leading space) are different tokens with different IDs.

> **Learning Thought:** Tokenization is the invisible layer beneath everything. Constraints operate on token IDs, not characters. If you want to allow the digit `1`, you may need to allow multiple token IDs: `'1'`, `' 1'`, `'1.'`, etc. Always test your constraints against the actual tokenizer, not your intuition.

---

### Logits: The Raw Probability Scores

Before a token is selected, the model's final layer outputs a **logit** (log-probability score) for every single token in the vocabulary. For a 50,000-token vocabulary, that's a vector of 50,000 numbers.

```
Logit vector: [L₀, L₁, L₂, ..., L₄₉₉₉₉]
                                      ↓ softmax
Probability:  [P₀, P₁, P₂, ..., P₄₉₉₉₉]   where ΣPᵢ = 1.0

Formula: P(tokenᵢ) = exp(Lᵢ) / Σⱼ exp(Lⱼ)
```

**Greedy decoding** picks `argmax(P)` — the token with the highest probability.
**Sampling** draws from the distribution proportionally.

```python
# Inspect logits after generation
model_output = model.generate(
    **inputs,
    max_new_tokens=5,
    output_logits=True,
    return_dict_in_generate=True
)

first_step_logits = model_output.logits[0][0]  # step 1, batch item 0
probs = torch.softmax(first_step_logits, dim=-1)
top10 = torch.topk(probs, 10)

for prob, idx in zip(top10.values, top10.indices):
    print(f"{repr(tokenizer.decode(idx.item())):<15} {prob.item():.4f}")
```

---

## Part 2B — The Masking Mechanism

### The Core Idea

To prevent the model from generating certain tokens, you set their logits to **negative infinity (−∞)** before the softmax step:

```
exp(−∞) = 0  →  P(blocked token) = 0/Σ = 0
```

This is called **logit masking**. It is mathematically exact: the blocked token has zero probability.

**Soft vs Hard masking:**

| Approach | Value | Effect |
|----------|-------|--------|
| Soft penalty | `scores + (−10)` | Token is very unlikely (~0.00454%) but not impossible |
| Hard mask | `scores = −∞` | Token has exactly 0% probability — cannot be generated |

```
Rule: "Allow odd numbers only at the start, or after an even number"
→ If the last token was odd (e.g., '3'), set all odd-digit logits to −∞

BEFORE masking:           AFTER masking (last token was '3'):
[P('0')=0.1               [P('0')=0.18
 P('1')=0.12               P('1')=0
 P('2')=0.09               P('2')=0.16
 P('3')=0.11       →       P('3')=0
 P('4')=0.08               P('4')=0.14
 P('5')=0.10               P('5')=0
 ...]                      ...]
```

**Valid sequences under "odd only after even" rule:**
- ✅ `22222` — all even, rule trivially satisfied
- ✅ `1234567` — each odd follows an even
- ✅ `24142167` — valid alternation
- ❌ `11` — second '1' follows '1' (odd), blocked

---

## Part 2C — LogitsProcessor: The HuggingFace Hook

### What is a LogitsProcessor?

`LogitsProcessor` is an abstract base class in HuggingFace `transformers`. Subclass it and implement `__call__(input_ids, scores)` to intercept and modify logit scores at **every generation step**.

```
model.generate() calls your LogitsProcessor at each step:
    input_ids  = token IDs generated so far
    scores     = logit tensor of shape (batch_size, vocab_size)
    → return modified scores (same shape)
```

The modified scores are what get passed to the sampler (greedy or multinomial). This is where your constraint lives.

---

### LogitsMask — A Simple Token Allowlist

The simplest constraint: only allow tokens from a predefined set.

```python
from transformers import LogitsProcessor
import torch

class LogitsMask(LogitsProcessor):
    """
    Subtracts a large penalty from all tokens NOT in the allowed set.
    Tokens outside the set become extremely unlikely (soft) or impossible (hard).
    """
    def __init__(self, allowed_token_ids: list, penalty: float = 10.0):
        self.allowed_token_ids = set(allowed_token_ids)
        self.penalty = penalty

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        # Start by penalizing ALL tokens
        mask = torch.ones_like(scores) * (-self.penalty)
        # Restore allowed tokens to their original scores
        for token_id in self.allowed_token_ids:
            mask[:, token_id] = 0.0
        return scores + mask
```

**Hard masking version (probability = exactly 0):**
```python
class LogitsMaskHard(LogitsProcessor):
    def __init__(self, allowed_token_ids: list):
        self.allowed_token_ids = set(allowed_token_ids)

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float("-inf"))  # block everything
        for token_id in self.allowed_token_ids:
            mask[:, token_id] = scores[:, token_id]   # restore allowed tokens
        return mask
```

**Building the allowed set for digits:**
```python
allowed_chars = list("0123456789 ,")
allowed_token_ids = []

for ch in allowed_chars:
    ids = tokenizer.encode(ch, add_special_tokens=False)
    if ids:
        allowed_token_ids.append(ids[0])

allowed_token_ids.append(tokenizer.eos_token_id)  # always allow EOS!
allowed_token_ids = list(set(allowed_token_ids))  # deduplicate
```

**Using it:**
```python
from transformers import LogitsProcessorList

processor_list = LogitsProcessorList([LogitsMask(allowed_token_ids)])

output = model.generate(
    **inputs,
    max_new_tokens=20,
    logits_processor=processor_list,
    do_sample=False
)
```

> **Critical:** Always add `tokenizer.eos_token_id` to your allowed set. Without it, the model can never signal "I'm done" — generation will run until `max_new_tokens` is exhausted.

**Example output (digits only, sequence continuation):**
```
Prompt: "Continue the sequence: 2, 4, 8,"
Constrained output: 16 32 64 128 256

Unconstrained output: "...the next number is 16, then 32, and so forth..."
```

---

### RegexMask — Constraining to Any Regular Expression

More powerful than a fixed allowlist: at each step, test whether appending each candidate token still keeps the output **consistent with a regex pattern**.

```python
import regex  # pip install regex  (NOT the stdlib 're')
import torch
from transformers import LogitsProcessor

class RegexMask(LogitsProcessor):
    def __init__(self, regex_pattern: str, tokenizer, subtract=torch.inf):
        self.pattern   = regex.compile(regex_pattern)
        self.tokenizer = tokenizer
        self.subtract  = subtract

    def __call__(self, input_ids, scores):
        # Decode everything generated so far (excluding prompt)
        current_text = self.tokenizer.decode(
            input_ids[0], skip_special_tokens=True
        )

        mask = torch.zeros_like(scores)

        for token_id in range(scores.shape[-1]):
            candidate_token  = self.tokenizer.decode(token_id)
            candidate_text   = current_text + candidate_token

            # partial=True: match if the string is a valid PREFIX of the pattern
            if not self.pattern.match(candidate_text, partial=True):
                mask[:, token_id] = -self.subtract  # block this token

        return scores + mask
```

---

### Partial Regex Matching — The Key Trick

Standard `re.match()` fails on incomplete strings because it checks if the pattern **fully matches**. For constrained generation, we need to check if the string is a **valid prefix** — i.e., could it become a valid match once more tokens are added?

The `regex` library (not Python's built-in `re`) supports `partial=True`:

```python
import regex

# URL regex pattern
url_regex = regex.compile(r" ?https?:\/\/(www\.)?\w{1,256}\.\w{1,6}\/?$")

# Standard match: FAILS on incomplete URL
print(regex.match(r"https?://\w+\.\w+", "http://goo"))  # None — 'goo' has no TLD yet

# Partial match: SUCCEEDS — "http://goo" is a valid prefix
print(url_regex.match("http://goo", partial=True))  # Match object ✓

# Once completed, becomes fully valid:
print(url_regex.match("http://google.com"))  # Match object ✓
```

**Why this is necessary:** At step 3 of generating `http://google.com`, you have `http://goo`. This is incomplete but valid. You must allow it. Standard `re.match` would say "invalid" and block all tokens — the generation would collapse. Partial matching says "could become valid" — generation continues correctly.

---

### URL Generation — Constrained vs Unconstrained

**Prompt:** `"One of the most common web addresses is:"`

**Unconstrained output:**
```
One of the most common web addresses is:
www.google.com (but this isn't a proper URL format, for a proper...)
```
(Model wanders into prose, doesn't output a valid URL)

**Constrained output (with RegexMask enforcing URL pattern):**
```
One of the most common web addresses is: https://www.google.com/
```

Always valid. No prose. Exactly what the regex specifies.

---

### Multinomial Sampling + Constraints

Constrained generation works equally well with **sampling** (not just greedy decoding). The mask sets invalid tokens to zero probability, then sampling draws from the remaining valid distribution. This gives **diverse yet always-valid outputs**:

```python
# Same regex constraint, do_sample=True:
#1  http://www.example.com/
#2  http://www.google.com
#3  http://www.somewebsite.com/path/to/file
#4  http://www.google.com/search
```

Every output is a valid URL, but they are different — the stochastic sampling explores the valid space.

> **Learning Thought:** This is a profound property. Without constraints, sampling gives diverse but often invalid outputs. With constraints, you get diversity within a strictly valid space. This is how production systems generate varied-but-correct structured data at scale.

---

## Part 2D — LogitsProcessor vs Structured Decoding

### The Limitation of Token-by-Token Masking

`LogitsProcessor` works step by step with **no lookahead**. It can paint itself into a corner:

```
Example: Generating a JSON key with value pattern: "key": <number>
Step 1: model picks "
Step 2: model picks k
Step 3: model picks e
Step 4: model picks y
Step 5: model picks "    (close quote)
Step 6: LogitsMask allows ':' next → picks :
Step 7: LogitsMask allows space → picks ' '
Step 8: Now the model wants to pick a letter for a string value...
        But your mask only allowed numbers! → stuck
```

The constraint was locally valid at each step but globally inconsistent. The model got stuck.

| Dimension | LogitsProcessor | Structured Decoding |
|-----------|----------------|---------------------|
| Correctness | No lookahead — can paint into a corner | Schema-aware — always maintains global validity |
| Complexity | Simple to implement | Requires grammar/schema engine |
| Flexibility | Any custom rule | Schema-defined rules only |
| Production use | Research, simple patterns | OpenAI structured outputs, Outlines, LM-Format-Enforcer |
| Speed | Step-wise overhead | Pre-compiled automaton — efficient |

---

### How OpenAI Structured Decoding Works

OpenAI's **Structured Output** (strict mode) is a production-grade implementation that solves the lookahead problem:

**Core mechanism:**
1. **Schema-Constrained Token Selection:** Before each token is sampled, a finite-state automaton derived from the JSON schema determines which tokens are valid given the *global* context — not just the local last token.
2. **Invalid continuations are pruned:** Missing quotes, wrong bracket types, type mismatches — all blocked before sampling.
3. **The model never gets stuck:** Because the automaton maintains a full parse state, it always knows what's valid.

```python
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()

class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

# strict=True triggers structured decoding server-side
completion = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[{"role": "user", "content": "Alice and Bob are going to a science fair on Friday."}],
    response_format=CalendarEvent,
)
event = completion.choices[0].message.parsed
print(event.name, event.date, event.participants)
```

**Alternatively, using JSON mode:**
```python
response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[
        {"role": "system", "content": "Return ONLY valid JSON. No prose, no markdown."},
        {"role": "user", "content": "Analyse: 'The delivery was slow but food was great.'"}
    ],
    response_format={"type": "json_object"},  # server-side JSON enforcement
    temperature=0.0
)
import json
parsed = json.loads(response.choices[0].message.content)
```

---

### What LogitsProcessor Does NOT Enforce

**Q: Which of the following is NOT enforced by LogitsProcessor?**

1. ~~Only allowed tokens are generated~~ → This IS enforced (the whole point)
2. **The output follows the input grammar completely** → NOT fully enforced (lookahead missing)
3. ~~Some tokens are generated preferentially~~ → This can be done (soft penalties bias preferences)
4. **Partial matches from nested grammars** → NOT reliably enforced (complex grammars need automaton-based decoding)

---

### `logit_bias` — The API Equivalent of LogitsMask

Cloud APIs (OpenAI) don't expose raw logits, but OpenAI provides `logit_bias` — a dict mapping token IDs to bias values (−100 to +100):

```python
import tiktoken
from openai import OpenAI

enc = tiktoken.encoding_for_model("gpt-4")
client = OpenAI()

# Build bias dict: strongly discourage all letter tokens
letter_bias = {}
for char in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ":
    for token_id in enc.encode(char):
        letter_bias[token_id] = -100  # effectively blocks these tokens

response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[{"role": "user", "content": "List the first 6 powers of 2, spaces only."}],
    logit_bias=letter_bias,
    max_tokens=30,
)
print(response.choices[0].message.content)  # "1 2 4 8 16 32 64"
```

**Groq doesn't support `logit_bias`** — use strong system prompts instead:
```python
messages=[
    {"role": "system", "content": "ONLY output digits and spaces. No letters, no commas, no punctuation."},
    {"role": "user", "content": "List the first 6 powers of 2."}
]
```

---

## Interview Questions — Section 2

### Conceptual

**Q1. Explain the autoregressive token generation process. What is its key implication for constrained generation?**

> LLMs generate tokens one at a time in a loop. At each step, the model sees all previous tokens (prompt + generated so far) and produces a probability distribution over the entire vocabulary. It then samples or argmax-selects one token and appends it. The key implication: there is no global planning or backtracking. If an invalid token is selected at step N, all subsequent tokens are conditioned on that error. Constrained generation prevents this by intervening *before* selection, ensuring only valid tokens can be chosen.

**Q2. What is the mathematical basis of logit masking? Why does setting a logit to −∞ give exactly 0 probability?**

> Softmax converts logits to probabilities: `P(i) = exp(Lᵢ) / Σⱼ exp(Lⱼ)`. When `Lᵢ = −∞`, `exp(−∞) = 0`. The numerator becomes 0, so `P(i) = 0`. This is exact — not approximate. The masked token has zero probability regardless of other logit values. A finite penalty like −10 gives `exp(−10) ≈ 0.0000454`, which is a strong deterrent but not zero.

**Q3. Why do you need partial regex matching instead of standard `re.match()` for RegexMask?**

> Standard `re.match()` checks if the pattern **fully matches** the string. During generation, the string is always incomplete (growing token by token). An incomplete-but-valid prefix like `"http://goo"` would fail standard matching because it hasn't reached a TLD yet — the pattern hasn't been "satisfied." Partial matching (`partial=True` in the `regex` library) returns a match if the string is consistent with the pattern so far — i.e., could become a valid complete match with additional characters. Without partial matching, the mask would block all tokens the moment the output begins, collapsing generation.

**Q4. What is the fundamental limitation of LogitsProcessor compared to schema-aware structured decoding?**

> `LogitsProcessor` applies constraints locally — one step at a time with no lookahead into the future. This means it can paint itself into a corner: each individual token choice is locally valid, but the sequence as a whole becomes globally invalid. Schema-aware structured decoding (like OpenAI Structured Outputs) maintains a finite-state automaton derived from the target schema. At every step, it knows the full global parse state and only allows tokens that keep the output globally valid. It never gets stuck.

**Q5. What is the difference between soft penalty (−10) and hard masking (−∞)? When would you use each?**

> Soft penalty (−10): adds −10 to logit scores. `exp(−10) ≈ 0.0000454` — token can still appear in rare edge cases (model was very confident). Use when you want a strong deterrent but are okay with occasional escape. Hard masking (−∞): zero probability guaranteed. Token cannot appear. Use when structural correctness is mandatory — e.g., generating valid JSON, numbers in a numeric field, URLs. Most production use cases warrant hard masking.

### Applied / Code-level

**Q6. You're building a system that must generate valid ISO 8601 date strings (YYYY-MM-DD). How would you implement this with a LogitsProcessor?**

> Build an allowlist of all tokens that can appear in a valid ISO 8601 date: digits `0–9`, hyphen `-`, and EOS. Build a `LogitsMask` with these token IDs. Apply it during generation. More robustly, use `RegexMask` with the pattern `r"\d{4}-\d{2}-\d{2}"` and `partial=True` to enforce the exact structure including digit placement. The regex approach handles positional constraints (e.g., only 0 or 1 as the first digit of month), which a flat allowlist cannot.

**Q7. Why must EOS (end-of-sequence token) always be in the allowed set for any LogitsMask?**

> If EOS is blocked, the model can never signal completion. Generation will run until `max_new_tokens` is exhausted, producing garbage tokens padded to the limit. Always include `tokenizer.eos_token_id` in the allowed set so the model can terminate naturally when it has finished generating the valid output.

**Q8. A colleague suggests using `response_format={"type": "json_object"}` instead of a LogitsProcessor for production JSON generation. Is this correct? What are the trade-offs?**

> Yes, this is the right choice for production using cloud APIs. `json_object` mode triggers server-side structured decoding that guarantees syntactically valid JSON — it is more robust than a handwritten LogitsProcessor because it uses a full JSON grammar automaton (no lookahead problem). Trade-offs: (1) requires OpenAI/Azure API — no local model support; (2) only guarantees syntactic validity (valid JSON), not semantic validity (the fields/types you expect — use Pydantic for that); (3) costs API tokens. For local models, `LogitsProcessor` or a library like `outlines` or `lm-format-enforcer` is the equivalent.

**Q9. Explain multinomial sampling with constraints in your own words. Why is this more useful than constrained greedy decoding?**

> Greedy constrained generation always produces the same output — the highest-probability valid token at each step. Multinomial constrained generation samples from the probability distribution over *valid tokens only* — invalid ones have been masked to zero. This means each run can produce a different valid output. Use case: generating 10 valid database schema names, 10 valid URLs, or 10 valid structured queries that vary in content but all conform to the schema. Greedy would always give the same one. Sampling gives diverse valid options to choose from.

---

## Key Learning Thoughts — Section 2

1. **Token generation is fundamentally local, step-by-step.** The model has no concept of the whole output when picking each token. Constrained generation is the external mechanism that enforces global structure, compensating for this myopia.

2. **Logit masking is precise mathematics, not heuristics.** `exp(−∞) = 0` is exact. When you mask a token, it is genuinely impossible, not just unlikely. Understanding this makes you precise when debugging constraint bugs — if a forbidden token appears, it means your mask didn't cover it.

3. **Tokenization is the hidden trap.** The character `'1'` and the token `' 1'` (with leading space) are different. A constraint that only blocks `tokenizer.encode('1')` will silently miss `' 1'` appearing after a space. Always enumerate all surface forms of a character when building allowlists.

4. **LogitsProcessor is a research and prototyping tool; schema-aware decoding is for production.** For production JSON, dates, or structured fields in a professional system, use OpenAI Structured Outputs, `outlines`, or `lm-format-enforcer` — they handle the lookahead problem that simple LogitsMask cannot.

5. **Partial regex matching is the bridge between text generation and formal language theory.** The insight that "a token is valid if it keeps the growing string as a valid prefix" is elegant and general — it works for any regular language, not just URLs. This is the foundation of modern grammar-constrained LLM decoding systems.

6. **Sampling + constraints = diversity within validity.** This combination — often overlooked — is what makes constrained generation practically useful for data generation, synthetic datasets, and A/B testing. You get variety without sacrificing correctness.

---

## Setup Reference

```bash
# Install dependencies for constrained generation
pip install transformers torch regex

# For logit_bias with OpenAI APIs:
pip install openai tiktoken
```

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LogitsProcessor, LogitsProcessorList
import torch
import regex  # pip install regex — NOT stdlib 're'

model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
```

## Notebook Reference

| File | Section in Notebook |
|------|---------------------|
| `llm_prompting.ipynb` | Section 5 — Constrained Generation (LogitsMask, URL constraint) |
| `llm_structured_outputs.ipynb` | Section 1 — LogitsMask deep dive (soft vs hard, digits vs letters) |
