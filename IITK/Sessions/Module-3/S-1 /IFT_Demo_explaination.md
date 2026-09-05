# `encode_example` — Line-by-Line Explanation

Companion notes for [`IFT_Demo.ipynb`](IFT_Demo.ipynb) (cell-8).
Related session note: [`learning-notes/06-instruction-fine-tuning.md`](learning-notes/06-instruction-fine-tuning.md)

---

## A. What this function is for

`encode_example` converts one Dolly row — `(instruction, context, output)` — into the
three tensors the HuggingFace `Trainer` expects:

| Tensor | Shape | Purpose |
|---|---|---|
| `input_ids` | `(256,)` | the token sequence the model reads |
| `attention_mask` | `(256,)` | which positions are real content vs. padding |
| `labels` | `(256,)` | which positions contribute to the loss (`-100` = ignore) |

The whole job is: **build the full conversation, then mask the prompt so the model is
graded only on the response.**

### Config it depends on (cell-4 and cell-6)

```python
MODEL_NAME   = "Qwen/Qwen2.5-0.5B"
MAX_LEN      = 256
IGNORE_INDEX = -100

tokenizer.padding_side = "right"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token   # ← pad and eos become the SAME id
```

That last line matters a lot in section F.

---

## B. The source code

```python
def encode_example(instruction, context, assistant_msg):
    user_msg = build_user_turn(instruction, context)

    messages_full = [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]

    messages_prompt_only = [{"role": "user", "content": user_msg}]
    prompt_text = tokenizer.apply_chat_template(messages_prompt_only, tokenize=False, add_generation_prompt=True)
    full_text   = tokenizer.apply_chat_template(messages_full,        tokenize=False, add_generation_prompt=False)

    encoding = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    full_ids = encoding["input_ids"]
    offsets  = encoding["offset_mapping"]

    input_ids = full_ids[:MAX_LEN]
    offsets   = offsets[:MAX_LEN]

    prompt_char_len = len(prompt_text)
    prompt_len = len(input_ids)
    for i, (start, end) in enumerate(offsets):
        if start >= prompt_char_len:
            prompt_len = i
            break

    labels = []
    for i in range(len(input_ids)):
        if i < prompt_len:
            labels.append(IGNORE_INDEX)
        else:
            labels.append(input_ids[i])

    actual_len = len(input_ids)

    while len(input_ids) < MAX_LEN:
        input_ids.append(tokenizer.pad_token_id)
        labels.append(IGNORE_INDEX)

    attention_mask = [1] * actual_len + [0] * (MAX_LEN - actual_len)

    return {
        "input_ids":      torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels":         torch.tensor(labels),
    }
```

---

## C. Step 1 — `build_user_turn`: two branches

```python
def build_user_turn(instruction, context):
    if context:
        return instruction + "\n\n" + context
    return instruction
```

Instruction and context are welded into **one user message** separated by a blank line.
There is no `### Context:` marker — the model must infer from position that the second
paragraph is reference material.

> 💡 **Learning Thought**
> Dolly's `context` is *conditioning*, not *content to reproduce*. It goes in the user
> turn precisely so it lands on the masked side of the boundary. The model reads it and
> is never rewarded for echoing it back.

---

## D. Step 2 — Two chat-template renders, and why

```python
prompt_text = tokenizer.apply_chat_template(msgs_prompt_only, tokenize=False, add_generation_prompt=True)
full_text   = tokenizer.apply_chat_template(msgs_full,        tokenize=False, add_generation_prompt=False)
```

Qwen uses ChatML. `add_generation_prompt=True` stops right after **opening** the
assistant turn; `add_generation_prompt=False` on the full conversation continues into
the answer and closes it.

```
prompt_text:                              full_text:
<|im_start|>system                        <|im_start|>system
You are a helpful assistant.<|im_end|>    You are a helpful assistant.<|im_end|>
<|im_start|>user                          <|im_start|>user
{user_msg}<|im_end|>                      {user_msg}<|im_end|>
<|im_start|>assistant                     <|im_start|>assistant
                     ← ends here          {assistant_msg}<|im_end|>
```

**The critical property:** `full_text.startswith(prompt_text)` is exactly `True`.
So `len(prompt_text)` is a valid *character* cut-point into `full_text`.

Note the template injects `"You are a helpful assistant."` automatically — which is why
that string leaks into some generations in the test-set output at the bottom of the
notebook.

> 💡 **Learning Thought**
> `prompt_text` is computed for its **character length only**. It is deliberately never
> tokenized. The moment you tokenize it separately, you have two independent
> tokenizations that can disagree — see section E.

---

## E. Steps 3–4 — One tokenization, then the offset boundary search

```python
encoding = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
```

- `add_special_tokens=False` — the template already wrote `<|im_start|>` / `<|im_end|>`
  **as text**. Letting the tokenizer add more would duplicate them.
- `return_offsets_mapping=True` — returns `(start_char, end_char)` per token, i.e. the
  span of `full_text` each token came from.

```python
input_ids = full_ids[:MAX_LEN]      # truncate FIRST
offsets   = offsets[:MAX_LEN]

prompt_char_len = len(prompt_text)
prompt_len = len(input_ids)         # fallback if loop never breaks — see section H
for i, (start, end) in enumerate(offsets):
    if start >= prompt_char_len:
        prompt_len = i
        break
```

One question per token: *does this token start at or after the prompt's last character?*
The first token that does is where the response begins.

### Why not just tokenize the prompt and take its length?

The naive version:

```python
prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])  # ❌
```

BPE can **merge across the prompt/response seam**. Tokenized alone the prompt may be 30
tokens, but inside the joint text the trailing `\n` and the leading response character
might fuse into one token — now the joint sequence has 29 prompt tokens and your
boundary is off by one. Off by one means you either train on a prompt token or mask out
the first real response token, silently, on every affected row.

Offsets are derived from the **single joint tokenization**, so they cannot disagree with it.

### Honest caveat on the seam

This makes the boundary **well-defined and conservative, not magically perfect.** If a
token straddled the boundary (`start < prompt_char_len < end`), the loop would not break
there, and that token would be labeled `-100`. You lose supervision on one partly-response
token.

That is the **safe** failure direction. The separate-tokenization approach fails in the
dangerous direction, because `len(prompt_ids)` may not correspond to any real index in
`full_ids` at all.

In practice Qwen's GPT-style pretokenizer splits `\n` from a following letter, so no such
merge occurs here — but the code doesn't have to know that, which is the point.

---

## F. Steps 5–6 — Labels, padding, attention mask

### Labels

```python
labels = [IGNORE_INDEX] * prompt_len + input_ids[prompt_len:]
```

`-100` is the default `ignore_index` of `torch.nn.CrossEntropyLoss`, and HuggingFace
honors it. Loss is computed **only** over response tokens.

The closing `<|im_end|>` is **inside** the supervised region. That is deliberate — it is
how the model learns to *stop*.

> 💡 **Learning Thought**
> `labels[i]` sits at the **same index** as `input_ids[i]` — you do *not* shift it
> yourself. `Qwen2ForCausalLM` shifts internally (`logits[..., :-1, :]` vs
> `labels[..., 1:]`). So the prediction of the first response token is made *from* the
> last prompt token, which is exactly right.

### Padding and the attention mask

```python
actual_len = len(input_ids)          # captured BEFORE padding — this is the key line

while len(input_ids) < MAX_LEN:
    input_ids.append(tokenizer.pad_token_id)
    labels.append(IGNORE_INDEX)

attention_mask = [1] * actual_len + [0] * (MAX_LEN - actual_len)
```

The naive mask would be:

```python
attention_mask = [1 if t != tokenizer.pad_token_id else 0 for t in input_ids]  # ❌
```

But cell-6 set `pad_token = eos_token`, so a **genuine** end-of-sequence token in your
content is *value-identical* to padding. The naive version would mask out the model's own
stop token. Tracking `actual_len` before padding sidesteps the ambiguity entirely.

> 💡 **Learning Thought**
> `pad_token_id == eos_token_id` is the single most common silent bug in hand-rolled SFT
> collators. Any logic that identifies padding **by token value** is broken the moment
> those two ids coincide. Identify padding **by position**.

---

## G. Worked examples

### G.1 — Short answer, no context

```python
{'instruction': 'Which is a species of fish? Tope or Rope',
 'context': '',
 'output': 'Tope'}
```

`build_user_turn` → no-context branch → `user_msg = "Which is a species of fish? Tope or Rope"`

```
full_text:
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Which is a species of fish? Tope or Rope<|im_end|>
<|im_start|>assistant
Tope<|im_end|>
```

`prompt_char_len = 148` → `full_text[148]` is the `T` of `Tope`.

| idx | token | offset |
|----|----|----|
| 0 | `<\|im_start\|>` | (0, 12) |
| 1 | `system` | (12, 18) |
| … | … | … |
| 27 | `<\|im_start\|>` | (126, 138) |
| 28 | `assistant` | (138, 147) |
| 29 | `\n` | (147, 148) |
| **30** | **`T`** | **(148, 149)** ← `start >= 148`, break |
| 31 | `ope` | (149, 152) |
| 32 | `<\|im_end\|>` | (152, 162) |
| 33 | `\n` | (162, 163) |

`prompt_len = 30`, `actual_len = 34`.

```
input_ids      [<|im_start|>, system, …, T, ope, <|im_end|>, \n, PAD × 222]
attention_mask [1 × 34,                                          0 × 222]
labels         [-100 × 30,    T, ope, <|im_end|>, \n,          -100 × 222]
```

*(Exact sub-word splits depend on Qwen's BPE merges — the structure is what matters.)*

---

### G.2 — Long answer, no context

```python
{'instruction': 'What is the Masters?', 'context': '', 'output': '<620 chars>'}
```

| quantity | value |
|---|---|
| `len(user_msg)` | 20 |
| `len(response)` | 620 |
| `prompt_char_len` | 128 |
| `len(full_text)` | 759 |
| supervised tokens | ~130 of ~200 |

Same mechanism, but the supervised region dominates. This is what an *open-ended
generation* row looks like: most of the sequence carries gradient.

---

### G.3 — Extractive QA with a long context

```python
{'instruction': 'Based on this paragrah, what indie band released the album "Popsicle"?',
 'context': 'Popsicle is an alternative rock album by indie band Diamond Nights, '
            'released in 2005. The song "The Girl\'s Attractive" was featured in a '
            '2006 Jaguar and an Austrian beer (Stiegl) advertisement, as well on the '
            '"Thirst" and "Nicodemus" episodes of the TV drama Smallville, and was '
            'included on its second Soundtrack The Metropolis Mix .',
 'output': 'Diamond Nights'}
```

`build_user_turn` takes the **context branch**:

| quantity | value |
|---|---|
| `len(instruction)` | 70 |
| `len(context)` | 333 |
| `len(user_msg)` | 405 (= 70 + 2 for `\n\n` + 333) |
| `prompt_char_len` | **513** |
| `len(full_text)` | 538 |
| response region | 25 chars = `'Diamond Nights<\|im_end\|>\n'` |
| approx. tokens | ~145 → **no truncation** |

Seam detail:

| idx | token | offset | in prompt? |
|----|----|----|----|
| … | ` Mix` | (503, 507) | yes |
| … | ` .` | (507, 509) | yes |
| P−2 | `<\|im_end\|>` | (502, 512) | yes |
| P−1 | `\n` | (512, 513) | yes |
| **P** | **`Diamond`** | **(513, 520)** | ← break |
| P+1 | ` Nights` | (520, 527) | no |
| P+2 | `<\|im_end\|>` | (527, 537) | no |
| P+3 | `\n` | (537, 538) | no |

`prompt_len = P ≈ 141`, `actual_len ≈ 145`.

```
input_ids      [<|im_start|>, system, …, Diamond, ␣Nights, <|im_end|>, \n, PAD × 111]
attention_mask [1 × 145,                                                   0 × 111]
labels         [-100 × 141,   Diamond, ␣Nights, <|im_end|>, \n,          -100 × 111]
```

**Out of ~145 tokens, 4 carry gradient — about 97% is `-100`.**

> 💡 **Learning Thought**
> Row count ≠ supervision. `closed_qa` rows contribute a handful of supervised tokens
> each; `open_qa` / `brainstorming` rows contribute hundreds. With `N_TRAIN = 200` mixed
> across categories, the effective training signal is far smaller than "200 examples"
> suggests — and unevenly distributed across task types.

---

## H. ⚠️ A real bug this code hides

Truncation happens **before** the boundary search, and `prompt_len` defaults to
`len(input_ids)`.

Take Dolly's Komorida row (~1,400 characters of context, comfortably over 256 tokens):

1. `input_ids` is truncated to 256 tokens — all still inside the prompt
2. no offset ever reaches `prompt_char_len`, so the loop **never breaks**
3. `prompt_len` stays at `256` → **every label is `-100`**

That sample contributes zero gradient. Worse, if an entire batch is long-context rows,
the loss denominator is zero and you get **`nan`**. With `MAX_LEN = 256` on Dolly's
`closed_qa` category this is not hypothetical.

A related failure: truncation can cut the response mid-way and **drop the closing
`<|im_end|>`**, so those rows teach the model nothing about stopping. Plausibly a small
contributor to the rambling, non-terminating generations in the test output.

### Fix option 1 — drop fully-masked rows

```python
def has_supervision(ex):
    enc = encode_example(ex["instruction"], ex["context"], ex["output"])
    return enc["labels"].ne(IGNORE_INDEX).any().item()

all_examples = [ex for ex in all_examples if has_supervision(ex)]
```

### Fix option 2 — raise `MAX_LEN`, and truncate the *context*, not the response

Better: guarantee the answer always survives. Budget tokens for the response first, then
give the remainder to the context.

---

## I. 🎯 Interview questions

**Q1. Why is `add_special_tokens=False` used when tokenizing `full_text`?**
Because `apply_chat_template` has already emitted `<|im_start|>` / `<|im_end|>` as literal
text. Letting the tokenizer prepend its own would duplicate boundary tokens and shift
every offset, corrupting the prompt/response boundary.

**Q2. Why compute `prompt_text` at all if it is never tokenized?**
Only for `len(prompt_text)` — a **character** boundary. Using a character cut-point into a
single joint tokenization avoids the two-independent-tokenizations disagreement that BPE
seam merges cause.

**Q3. Do you shift `labels` by one relative to `input_ids`?**
No. HuggingFace `*ForCausalLM` heads shift internally. Passing pre-shifted labels
double-shifts and trains the model on the wrong targets.

**Q4. Why is the attention mask built from `actual_len` instead of comparing to `pad_token_id`?**
Because `pad_token` was set to `eos_token`. A real EOS in the content is value-identical
to padding, so value-based masking would zero out the model's own stop token. Position is
unambiguous; value is not.

**Q5. Should the closing `<|im_end|>` be masked or supervised?**
Supervised. It is the stop signal — mask it and the model never learns to terminate,
producing exactly the run-on generations seen in this notebook's test output.

**Q6. What happens when the prompt alone exceeds `MAX_LEN`?**
The boundary loop never fires, `prompt_len` falls back to `len(input_ids)`, and all labels
become `-100`. The row contributes no gradient; an all-masked batch yields `nan` loss.

**Q7. Why mask the prompt at all — why not train on the full sequence?**
Training on the prompt teaches the model to *generate questions and reference paragraphs*,
not to answer them. It also lets long contexts dominate the loss, drowning out the short
answers you actually care about.

---

## J. Self-check

- [ ] I can state why `full_text.startswith(prompt_text)` must hold.
- [ ] I can explain a BPE seam merge and what it breaks.
- [ ] I know which direction the offset approach fails in, and why that's the safe one.
- [ ] I can explain why `labels` are *not* manually shifted.
- [ ] I can explain the `pad_token == eos_token` hazard without looking it up.
- [ ] I can predict the labels tensor for a row whose prompt exceeds `MAX_LEN`.
- [ ] I can explain why `<|im_end|>` belongs in the supervised region.
- [ ] I can name two fixes for the truncation bug and say which is better.

---

## K. Cross-references

- Notebook: [`IFT_Demo.ipynb`](IFT_Demo.ipynb) — cell-8 (`encode_example`), cell-10
  (`IFTDataset`), cell-16 (`Trainer`), cell-18 (`generate_response`)
- Session note: [`learning-notes/06-instruction-fine-tuning.md`](learning-notes/06-instruction-fine-tuning.md)
- Contrast with: [`SFT_Demo.ipynb`](SFT_Demo.ipynb), [`Multi_Task_FFT.ipynb`](Multi_Task_FFT.ipynb)
