# Hands-on: Qwen2-VL-2B inference and QLoRA fine-tuning

> Notebook: `VLM_Inference_Finetuning_Tutorial_UpGrad_Student_ver (1).ipynb` (link in `Link-for-hands-on-session.pdf`: https://shorturl.at/LIYPF)
> TA-led, ~20 min at the end of the lecture. Runs on a free Colab T4.
> **Goal:** map each cell to a lecture concept, see what the tensors really look like, and know the **4 bugs** in the student notebook (**proven below with the real processor**) and how to fix them.

[← Section 6](06-Section-6-Synthesis-and-Comparison.md) · [Index](00-INDEX.md)

**Legend:** 📖 explanation · 🖼️ diagram · 🧪 code (run and verified; outputs are real) · 📚 go deeper
**What was actually run for these notes:** all inference cells and the collator/LoRA checks, on CPU with `Qwen/Qwen2-VL-2B-Instruct`. **QLoRA training itself needs a CUDA GPU** (`bitsandbytes` 4-bit has no CPU path), so run the training cells on Colab.

---

## Plan of the notebook

```
1. Load a 100-row VQA dataset ─► 2. Inference with pretrained Qwen2-VL-2B ─► 3. Free GPU memory
                                                                                  │
                  5. SFTTrainer.train() ◄── 4. Reload in 4-bit + LoRA adapters ◄──┘
```

The TA's framing: *"You don't like the Qwen team's performance on your task, e.g. medical QA or logistics? Fine-tune it on your own data."*

> ⚠️ **Model name:** this is **Qwen2-VL**, not the Qwen-VL from the slides. It has no 256-query cross-attention adapter. It uses **dynamic resolution + M-RoPE + a 2×2 patch-merger MLP**. See [Section 4 §H](04-Section-4-Qwen-VL.md#h-limitations-and-qwen2-vl-slide-62).

---

## 1. Install (cell 0)
```bash
pip install datasets
pip install -U -q git+https://github.com/huggingface/trl.git bitsandbytes peft qwen-vl-utils trackio
```
| Package | Role |
|---|---|
| `datasets` | load from the Hugging Face Hub |
| `trl` | **Transformer Reinforcement Learning** library; provides `SFTTrainer` |
| `bitsandbytes` | 4-bit / 8-bit quantisation (**CUDA only**) |
| `peft` | parameter-efficient fine-tuning (LoRA) |
| `qwen-vl-utils` | `process_vision_info`: pulls images and videos out of chat messages |
| `trackio` | experiment tracking (unused here) |

**Tip:** installing TRL from `git+…` gives you whatever is on `main` today. For reproducible results pin a version (`pip install trl==<version>`), because `SFTConfig` arguments get renamed between releases.

---

## 2. Dataset (cells 1–3)
```python
from datasets import load_dataset
dataset = load_dataset("dutta18/miniVQAv2DatasetForCollab", split="train")
print(dataset)
print([(dataset['question'][i], dataset['multiple_choice_answer'][i]) for i in range(6)])
```
```
Dataset({ features: ['multiple_choice_answer', 'question', 'image'], num_rows: 100 })
[('What is this kid playing with?', 'kite'), ('Is this person holding a hot dog or candy?', 'neither'),
 ('Is this room large?', 'yes'), ('Is this a rocket ship?', 'no'),
 ('What kind of climate do these cows live in?', 'cold'), ('What is she riding on?', 'motorcycle')]
```
A **100-row VQAv2 subset**, the same benchmark family used to evaluate LLaVA ([Section 3 §H](03-Section-3-LLaVA.md#h-evaluation-slide-40)). The gold answers are **one word**.

Note row 1: the gold answer is **`neither`**. In the session, the base model answered *"There is no existence of a hot dog or candy in the image description."* That is **correct in meaning but wrong under exact match**. This gap between chatty answers and one-word labels is exactly what fine-tuning (or a formatting prompt) closes.

📚 [Dataset page](https://huggingface.co/datasets/dutta18/miniVQAv2DatasetForCollab) · [VQAv2](https://visualqa.org/)

---

## 3. Load the model (cells 5–8)
```python
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

model_id = "Qwen/Qwen2-VL-2B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto")
```
- **~2.2B params × 2 bytes (bf16) ≈ 4.4 GB**, which matches the TA's "4–5 GB download".
- `sdpa` = PyTorch's fused scaled-dot-product attention. `flash_attention_2` is faster but needs an Ampere+ GPU (not a T4).
- `device_map="auto"` places layers on the GPU (and spills to CPU if needed).

### 🧪 What's inside (the lecture's anatomy, for real)
```python
for name, m in model.model.named_children():
    print(f"{name:16s} {type(m).__name__:38s} {sum(p.numel() for p in m.parameters())/1e6:8.1f}M")
print(model.model.visual.merger)
```
```
visual           Qwen2VisionTransformerPretrainedModel    665.3M   ← vision encoder (contains the merger)
language_model   Qwen2VLTextModel                         1543.7M  ← the LLM (hidden size 1536)
PatchMerger(
  (ln_q): LayerNorm((1280,), eps=1e-06, elementwise_affine=True)
  (mlp): Sequential(
    (0): Linear(in_features=5120, out_features=5120, bias=True)   ← 2×2 patches × 1280, concatenated
    (1): GELU(approximate='none')
    (2): Linear(in_features=5120, out_features=1536, bias=True)   ← to the LLM width
  )
)
```
The TA: *"some models call it projector, some call it merger."* It plays the same role as LLaVA's projector, plus the 2×2 spatial merge.

---

## 4. The processor and pixel budget (cell 9) ⭐

```python
min_pixels = 256*28*28     # 200,704 px ≈ 448×448
max_pixels = 512*28*28     # 401,408 px ≈ 634×634
processor = AutoProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels)
```

### 📖 Why 28×28?
Qwen2-VL's ViT uses **14×14 patches**, and the **merger combines each 2×2 block** of patches into one LLM token. So **one visual token = 28×28 pixels**.
- `min_pixels = 256·28·28` → at least **256** visual tokens
- `max_pixels = 512·28·28` → at most **512** visual tokens
- The processor resizes each image (keeping the aspect ratio) to fit between these, in multiples of 28 (`smart_resize`, re-implemented and verified in [Section 4 §H](04-Section-4-Qwen-VL.md#h-limitations-and-qwen2-vl-slide-62)).

### 🧪 Real numbers for the COCO cats photo (640×480)
```python
inputs = processor(text=[text], images=[image], return_tensors="pt")
print(inputs["image_grid_thw"], inputs["pixel_values"].shape,
      (inputs.input_ids == model.config.image_token_id).sum().item())
```
```
tensor([[ 1, 34, 46]])   torch.Size([1564, 1176])   391
```
How to read this:
- `image_grid_thw = [1, 34, 46]` means **t**ime 1 (a still image), **h**eight 34 patches, **w**idth 46 patches. The image was resized to 476×644 (34·14 × 46·14).
- `pixel_values` is **(1564, 1176)**: 1564 = 34 × 46 patches. 1176 = 3 colours × **2** frames × 14 × 14. Qwen2-VL's ViT takes patches of **2 frames** (for video), so a still image is duplicated in time.
- **391 visual tokens** = 1564 / 4 after the 2×2 merge.

> 💡 **Learning Thought:** this is **dynamic resolution** in practice. The token count follows image area, within bounds you choose. Compare Qwen-VL's fixed 256 and LLaVA-1.5's fixed 576. `max_pixels` is your main **cost knob**: halving it roughly halves the visual tokens, and so the LLM compute and KV-cache for the image. Raise it for OCR-heavy tasks, lower it for simple yes/no VQA.

---

## 5. Build a chat message (cell 12)
```python
messages = [{"role": "user", "content": [
    {"type": "image", "image": dataset['image'][1]},
    {"type": "text",  "text":  dataset['question'][1]},
]}]
```
This is the chat equivalent of Qwen-VL's `<img>…</img>` + question.

### 🧪 What the chat template actually produces
```python
print(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
```
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Is this person holding a hot dog or candy?<|im_end|>
<|im_start|>assistant
```
- One `<|image_pad|>` placeholder is **expanded by the processor into 391 copies** (for our image). The model then overwrites those embedding slots with the merger's outputs. This is the LLaVA "splice" from [Section 3 §D](03-Section-3-LLaVA.md#d-how-image-tokens-enter-the-llm-slides-3435-), under a different token name.
- The final `<|im_start|>assistant\n` comes from `add_generation_prompt=True`. It tells the model "your turn now".

---

## 6. Preprocess (cell 15)
```python
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                   padding=True, return_tensors="pt").to(model.device)
```
`inputs` holds `input_ids`, `attention_mask`, `pixel_values` and `image_grid_thw` (used by M-RoPE to give each visual token its (t, h, w) position).

## 7. Generate (cell 16)
```python
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)
```
- `generate` = **autoregressive decoding**, one token at a time, up to 128 new tokens.
- **Why trim?** A decoder-only model's output contains **the prompt followed by the continuation**. Slicing off `len(input_ids)` keeps only the answer.
- The session's output: *"There is no existence of a hot dog or candy in the image description."* Our CPU run on the cats photo: *"There are two cats in the image."* The same code can also **ground**: see [Section 4 Code 4](04-Section-4-Qwen-VL.md#-code-4-real-grounding-with-qwen2-vl-2b-the-hands-on-model-hf).

> 💡 Notice the style of the session's answer: a full sentence, not VQA's "no". This is exactly what fine-tuning on VQA data should change, **if the training data actually contains the answers** (see Bug 1).

## 8. Free memory (cells 18–19)
`del model; gc.collect(); torch.cuda.empty_cache()` releases the bf16 model before reloading it in 4-bit. On a 15 GB T4 you can't hold both.

---

## 9. QLoRA: 4-bit quantisation (cells 21–23) ⭐
```python
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id, device_map="auto", torch_dtype=torch.bfloat16, quantization_config=bnb_config)
```
| Setting | Meaning |
|---|---|
| `load_in_4bit` | store the frozen base weights in 4 bits: ~4.4 GB → **~1.2–1.5 GB** |
| `nf4` | **NormalFloat-4**: 16 quantisation levels placed for normally distributed weights (better than uniform int4) |
| `double_quant` | also quantise the per-block scale constants (saves ~0.4 bits per param) |
| `compute_dtype=bf16` | weights are **de-quantised to bf16 on the fly** for each matmul. Storage is 4-bit, maths is 16-bit. |

**Analogy:** a zipped library. Books (weights) are stored compressed on the shelf (4-bit). When you read one, it is unzipped into your hands (bf16), used, and put back.

📚 [HF blog: 4-bit quantisation and QLoRA](https://huggingface.co/blog/4bit-transformers-bitsandbytes) · [QLoRA paper](https://arxiv.org/abs/2305.14314)

---

## 10. LoRA config (cell 26)
```python
from peft import LoraConfig
peft_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
                         target_modules=["q_proj", "v_proj", "v_proj", "o_proj"],   # ← Bug 3
                         task_type="CAUSAL_LM")
```

### 📖 LoRA in one picture
```
 x ──► [ W (frozen, d×k) ] ──────────────┐
  │                                      (+) ──► y = W·x + (α/r)·B·A·x
  └─► [ A (r×k) ] ─► [ B (d×r) ] ─ ×α/r ─┘       A: random init, B: zeros → starts as a no-op
```
- `r = 8`, `α = 16` → scaling **α/r = 2**
- For a 1536×1536 projection: full = **2.36M** params, LoRA = 8 × (1536 + 1536) = **24,576** (~1%)
- After training you can **merge** ΔW = (α/r)·B·A into W. The TA: *"model weights + trained matrix weights, that's it."*

### 🧪 What the notebook's config really adapts (run on the real model)
```python
from peft import get_peft_model
for tm in (["q_proj", "v_proj", "v_proj", "o_proj"], ["q_proj", "k_proj", "v_proj", "o_proj"]):
    m = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
                                         target_modules=tm, task_type="CAUSAL_LM"))
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    hit = sorted({n.split(".lora_A")[0].split(".")[-1] for n, _ in m.named_parameters() if "lora_A" in n})
    in_vision = any("visual" in n and "lora_" in n for n, _ in m.named_parameters())
    print(f"{tm}: trainable {trainable/1e6:.2f}M ({100*trainable/sum(p.numel() for p in m.parameters()):.3f}%), "
          f"adapted={hit}, touches vision tower={in_vision}")
    model = m.unload()
```
```
['q_proj', 'v_proj', 'v_proj', 'o_proj']: trainable 1.78M (0.080%), adapted=['o_proj', 'q_proj', 'v_proj'], touches vision tower=False
['q_proj', 'k_proj', 'v_proj', 'o_proj']: trainable 2.18M (0.099%), adapted=['k_proj', 'o_proj', 'q_proj', 'v_proj'], touches vision tower=False
```
- **Under 0.1% of the model is trained.** That is why it fits on a free T4.
- The duplicate `v_proj` is silently ignored, so `k_proj` is simply **missing**.
- **The vision tower is never touched.** Its attention layers are named `qkv`/`proj`, not `q_proj`. So this fine-tune keeps the vision encoder frozen, the same choice LLaVA made.

📚 [LoRA paper](https://arxiv.org/abs/2106.09685) · [PEFT docs](https://huggingface.co/docs/peft/index)

---

## 11. Training config (cell 29)
| Arg | Value | Note |
|---|---|---|
| `num_train_epochs` | 3 | |
| `per_device_train_batch_size` | 2 | |
| `gradient_accumulation_steps` | 2 | → **effective batch = 4** (accumulate gradients over 2 steps, then update) |
| `learning_rate` | 2e-4 | typical LoRA LR (much higher than full fine-tuning) |
| `optim` | `adamw_torch_fused` | |
| `bf16` | True | ⚠️ the T4 has no native bf16. It is emulated and slow; on a T4 prefer `fp16=True` |
| `max_grad_norm` | 0.3 | gradient clipping (the QLoRA paper's default) |
| `logging_steps / eval_steps / save_steps` | 5 / 8 / 5 | a checkpoint every 5 steps is a lot of disk for 75 steps |
| `remove_unused_columns` | False | keep `image` so the collator can see it |

**Step count:** 100 rows / effective batch 4 = **25 optimiser steps per epoch → 75 total**.

**Gradient accumulation, in code:** what the trainer does for you:
```python
for i, batch in enumerate(loader):
    loss = model(**batch).loss / accum_steps     # scale so the sum equals the big-batch average
    loss.backward()                              # gradients ADD UP in .grad
    if (i + 1) % accum_steps == 0:
        optimizer.step(); optimizer.zero_grad()  # one update per accum_steps mini-batches
```

---

## 12. Custom collator (cell 30) and trainer (cells 31–32)
The collator turns a list of rows into one padded tensor batch: chat template → processor → `labels`. Then `SFTTrainer(model, args, train_dataset, eval_dataset, peft_config, data_collator)` and `trainer.train()`.

> ⚠️ **Trap (Q&A #38): `SFTTrainer`** = Supervised Fine-Tuning Trainer from Hugging Face **TRL**, for fine-tuning on instruction → response data. It wraps the training loop, and with `peft_config` it wraps the model in LoRA for you.

📚 [TRL SFTTrainer docs](https://huggingface.co/docs/trl/sft_trainer) · [HF cookbook: Fine-tuning a VLM (Qwen2-VL) with TRL](https://huggingface.co/learn/cookbook/fine_tuning_vlm_trl), the recipe this notebook is adapted from

---

## ⚠️ The 4 bugs in the student notebook, proven

The notebook runs without errors and the loss goes down, but **it is not teaching the model to answer VQA questions.** Below, the original collator and a fixed one are run through the **real Qwen2-VL processor** on two samples: (cats image, "How many cats are there?", `2`) and (a smaller copy, "What color is the blanket?", `pink`).

### 🧪 The original collator vs the fixed one
```
padding side: left
ORIGINAL sample 0: seq_len=418, tokens with loss= 418, target text='<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>'...
ORIGINAL sample 1: seq_len=418, tokens with loss= 418, target text='<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|end'...
FIXED    sample 0: seq_len=421, tokens with loss=   3, target text='2<|im_end|>\n'
FIXED    sample 1: seq_len=421, tokens with loss=   3, target text='pink<|im_end|>\n'
answer text in ORIGINAL batch? False
```

### Bug 1 (critical): the answer is never in the training data
```python
messages = [{"role": "user", "content": [image, question]}]                   # no assistant turn!
batch_text = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) ...]
```
`multiple_choice_answer` is **never used**: `answer text in ORIGINAL batch? False`. The model never sees "2" or "pink". A falling loss here only means it is memorising prompts.

### Bug 2: loss is computed on **everything**, including padding
```python
model_inputs["labels"] = model_inputs["input_ids"].clone()
```
**All 418 positions** carry loss: the system prompt, the question, **the ~390 image-pad tokens**, and, for the shorter sample 1, **a run of `<|endoftext|>` padding tokens** (the processor pads on the **left**). So the model is also trained to predict padding. This contradicts the LLaVA recipe from the lecture: *"loss computed only over the assistant's response tokens"* ([Section 3 §D](03-Section-3-LLaVA.md#d-how-image-tokens-enter-the-llm-slides-3435-)). The fixed version trains on **3 tokens per sample**: the answer, `<|im_end|>`, and the newline.

### Bug 3: `target_modules` has a duplicate
`["q_proj", "v_proj", "v_proj", "o_proj"]` lists `v_proj` twice and leaves out **`k_proj`** (1.78M vs 2.18M trainable, above). It was almost certainly meant to be `["q_proj", "k_proj", "v_proj", "o_proj"]`.

### Bug 4: evaluating on the training set
`eval_dataset=dataset` is the same 100 rows as `train_dataset`. The "validation loss" is training loss. It can't detect overfitting, which is very likely with 100 rows × 3 epochs. (The TA said this was intentional to keep things simple. Fine for a demo, but know why it's wrong.)

*(Minor: cell 12 builds the prompt from `image[1]`, but cell 14 displays `image[10]`, so the image you look at isn't the one being asked about.)*

### 🧪 Fixed collator (verified with the real processor, output above)

```python
def collate_fn(samples):
    texts, prompt_lens = [], []
    for s in samples:
        user = {"role": "user", "content": [{"type": "image", "image": s["image"]},
                                            {"type": "text",  "text":  s["question"]}]}
        asst = {"role": "assistant", "content": [{"type": "text", "text": s["multiple_choice_answer"]}]}  # ← Bug 1
        texts.append(processor.apply_chat_template([user, asst], tokenize=False))
        prompt = processor.apply_chat_template([user], tokenize=False, add_generation_prompt=True)
        prompt_lens.append(processor(text=[prompt], images=[s["image"]], return_tensors="pt")["input_ids"].shape[1])

    batch = processor(text=texts, images=[s["image"] for s in samples], padding=True, return_tensors="pt")
    labels = batch["input_ids"].clone()
    for i, n in enumerate(prompt_lens):                                                 # ← Bug 2
        start = (batch["attention_mask"][i] == 1).nonzero()[0].item()   # first real token (handles LEFT padding)
        labels[i, : start + n] = -100                                   # mask system + image + question
    labels[batch["attention_mask"] == 0] = -100                         # mask padding
    batch["labels"] = labels
    return batch

peft_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],      # ← Bug 3
                         task_type="CAUSAL_LM")

split = dataset.train_test_split(test_size=0.2, seed=42)                                # ← Bug 4
trainer = SFTTrainer(model=model, args=training_args,
                     train_dataset=split["train"], eval_dataset=split["test"],
                     peft_config=peft_config, data_collator=collate_fn)
trainer.train()
```
> ⚠️ **Correction to my earlier version of these notes:** I had assumed right padding. **Qwen2-VL's processor pads on the left** (verified above), so masking "the first *n* tokens" would be wrong for the shorter samples in a batch. The fix computes each row's `start` from the attention mask, which works for either side.

### 🧪 How to check the fix worked: exact-match evaluation
```python
def vqa_exact_match(model, processor, rows):
    correct = 0
    for r in rows:
        msgs = [{"role": "user", "content": [{"type": "image", "image": r["image"]},
                 {"type": "text", "text": r["question"] + " Answer with a single word or number."}]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[r["image"]], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        pred = processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        correct += pred.strip().lower().rstrip(".") == r["multiple_choice_answer"].strip().lower()
    return correct / len(rows)

print("before:", vqa_exact_match(base_model, processor, split["test"]))
print("after :", vqa_exact_match(trainer.model, processor, split["test"]))
```
The suffix *"Answer with a single word or number."* is LLaVA-1.5's **response-formatting prompt** trick ([Section 3 §G](03-Section-3-LLaVA.md#g-llava-15-and-llava-next-slide-39)). Without it, a chatty base model loses exact-match even when it is right.

### 🧪 Real baseline: the base Qwen2-VL-2B (no fine-tuning) on the first 6 rows
```
[plain]     What is this kid playing with?              gold=kite        pred='The kid in the picture is playing with a kite.'            ✗
[plain]     Is this person holding a hot dog or candy?  gold=neither     pred='There is no existence of a hot dog or candy in the image.' ✗
[plain]     Is this room large?                         gold=yes         pred='I am not sure if the room is large. It can be both ...'    ✗
[plain]     Is this a rocket ship?                      gold=no          pred='There is no existence of a rocket ship in the image ...'   ✗
[plain]     What kind of climate do these cows live in? gold=cold        pred='It is ambiguous ... It can be either cold or warm.'        ✗
[plain]     What is she riding on?                      gold=motorcycle  pred='She is riding on a motorcycle.'                            ✗
[plain]     exact match: 0/6
[formatted] ... kite ✓ · 'hot dog' ✗ (gold: neither) · 'no' ✗ (gold: yes) · no ✓ · cold ✓ · motorcycle ✓
[formatted] exact match: 4/6
```
**What this shows:**
1. **0/6 → 4/6 without training a single weight.** The plain answers were mostly *right in meaning* (kite, no hot dog, motorcycle) but failed on format. **Always measure a prompt-only baseline before fine-tuning.** Otherwise you may credit fine-tuning for what is really a formatting fix.
2. **Forcing brevity can hurt reasoning.** The plain answer to the hot-dog question was *correct* ("no hot dog or candy"). When forced to one word, the model picked **"hot dog"**, one of the two options in the question. Short-answer formatting removes the room the model used to express "neither".
3. **This is what a correct fine-tune is for:** learning the dataset's answer conventions (`neither`, `yes`/`no`) while keeping the content right. The buggy notebook collator can't teach this, because the answers never reach the model.

### 🧪 Save, reload and merge the adapter
```python
trainer.save_model("qwen2b_vqa_lora")                 # saves ONLY the LoRA weights (a few MB)

from peft import PeftModel
base = Qwen2VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base, "qwen2b_vqa_lora")
merged = model.merge_and_unload()                     # W ← W + (α/r)·B·A ; no LoRA overhead at inference
merged.save_pretrained("qwen2b_vqa_merged"); processor.save_pretrained("qwen2b_vqa_merged")
```
Merge into a **bf16** base, not the 4-bit one. Merging into quantised weights loses precision.

---

## Link to the lecture

| Notebook thing | Lecture concept |
|---|---|
| `visual` → `merger` → `language_model` | generic anatomy: encoder → connector → LLM ([§1F](01-Section-1-Foundations-of-VLMs.md#f-the-generic-anatomy-of-a-vlm-slide-9)) |
| `min/max_pixels`, 28×28 per token, 391 tokens | dynamic resolution (Qwen2-VL), vs LLaVA's fixed 576 and Qwen-VL's fixed 256 |
| `<|image_pad|>` expanded inside the chat template | "image as a foreign language" (LLaVA splice) |
| `generate` + trimming the prompt | autoregressive decoding in a decoder-only LLM |
| loss masked to the answer (after the fix) | LLaVA's "loss only on assistant tokens" |
| LoRA only on the LLM, vision frozen | LLaVA Stage 2 / Qwen-VL Stage 3 choices |
| 4-bit + LoRA on a T4 | Q&A #37: cut cost with a smaller or quantised LLM and fewer visual tokens |
| VQAv2 data + exact match | LLaVA's evaluation benchmark |

---

## 🎯 Interview Questions

**Q1. What is QLoRA and why does it let a 2B VLM train on a free T4?**
> The frozen base weights are stored in 4-bit NF4 (~4× smaller than bf16) and de-quantised to bf16 per matmul. Only small low-rank adapters (ΔW = BA, r = 8) are trained in higher precision, under 0.1% of this model's parameters. Optimiser state and gradients exist only for the adapters. Memory drops from tens of GB for full fine-tuning to a few GB.

**Q2. Why does Qwen2-VL use 28×28 in `min_pixels` / `max_pixels`?**
> The ViT patch is 14×14, and the merger combines 2×2 neighbouring patches into one LLM token. So each visual token covers 28×28 pixels, and the pixel bounds translate directly into a visual-token budget (here 256–512; a 640×480 photo costs 391).

**Q3. You fine-tuned and the loss dropped, but the answers didn't improve. What would you check first?**
> Decode the labels of one batch (`tokenizer.decode(labels[labels != -100])`). If the target text is the system prompt or padding instead of the answer, the collator is wrong. That is exactly this notebook's case: the answer was never in the batch, and all 418 positions carried loss. Then check for train/eval leakage and the LoRA target modules.

**Q4. Why trim `generated_ids` before decoding?**
> In a decoder-only model, `generate` returns the input ids followed by the new tokens. Slicing off the first `len(input_ids)` positions leaves only the model's answer.

**Q5. What is gradient accumulation and what's the effective batch here?**
> Run several forward/backward passes, summing (scaled) gradients, before each optimiser step. That simulates a larger batch within the memory of a small one. 2 per device × 2 accumulation steps = an effective batch of 4, so 75 optimiser steps for 100 rows × 3 epochs.

**Q6. Why must label masking handle the padding side?**
> Batching pads sequences to equal length. With left padding (Qwen2-VL's default), each row's real tokens start at a different offset. Masking "the first n tokens" would mask padding and leave part of the prompt trainable. Compute each row's start from the attention mask, and always set padding positions to −100.

**Q7. Why is `pixel_values` for one image shaped (1564, 1176)?**
> Qwen2-VL flattens patches up front: 1564 = 34×46 patches of the resized image, and 1176 = 3 colours × 2 frames × 14 × 14. The ViT's patch embedding is 3D (temporal patch size 2, built for video), so a still image is repeated across 2 frames. After the 2×2 merge, 1564/4 = 391 LLM tokens.

**Q8. After LoRA training, how do you ship the model?**
> Either keep the base model plus a small adapter (swap adapters per customer), or `merge_and_unload()` into a bf16 base for one self-contained model with no extra inference latency. Don't merge into 4-bit weights.

---

## ✅ Self-check
1. How many optimiser steps does the notebook run? *(75)*
2. What does `add_generation_prompt=True` do, and why is it wrong for building **training** text? *(It appends an empty assistant turn to be filled at inference; training text must contain the assistant's answer.)*
3. What does `-100` mean in `labels`? *(PyTorch cross-entropy's `ignore_index`: no loss at that position.)*
4. Which part of Qwen2-VL do the LoRA adapters touch, and why not the vision tower?
5. Rewrite `target_modules` correctly.
6. Which side does Qwen2-VL's processor pad on, and what breaks if you assume the other side?
7. Our cats photo costs how many visual tokens with the notebook's settings? *(391)*
