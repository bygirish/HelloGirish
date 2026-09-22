# Section 3 — Visual Question Answering (VQA)

> **Where this fits:** The professor spent ~40 minutes here, "slightly more than expected, because the nuance is actually more detailed than architectures". Prompting a VLM with a question is easy. The research findings are the real content: VLMs often **don't look at the image**, answer confidently about things that **aren't there**, and are scored by **metrics that punish correct answers**.
> **Slides:** 21–33 · **Transcript:** 01:35 – 02:30 · **Notebook:** CoT counting cells (Section 6.D)
> **Runnable code:** [`code/s3_*.py`](code/). The experiments from the slides (blurred-image token probabilities, presupposition, majority voting) are **reproduced on a real VLM** (SmolVLM-256M, CPU), and the outputs are pasted below.

---

## 3.A — What VQA is, and how it grew out of captioning

**Definition:** answer a **specific, open-ended question grounded in an image**. It isn't a generic description. **The question decides what matters, and everything else can be ignored.**

**Where it's used:** customer-support screenshots, visual search assistants, education tools, accessibility Q&A (e.g. blind users asking about a photo).

### The four classic question types
| Type | Example | Answer space |
|---|---|---|
| (a) **Yes/No** | "Are there shadows on the sidewalk?" | Binary |
| (b) **Multiple choice** | "What kind of bears are those?" (pick 1 of 4) | Small fixed set |
| (c) **Counting** | "How many windows can you see?" | A number |
| (d) **Open-ended** | "What is the man doing on the street?" → "walking" | Unbounded |

| (a) Yes/No · (b) Multiple choice | (c) Counting · (d) Open-ended |
|---|---|
| ![VQA yes/no and multiple-choice examples](assets/vqa-types-yesno-mc.png) | ![VQA counting and open-ended examples](assets/vqa-types-count-open.png) |

*Slide 21: the four question types, each on a real VQA image.*

🌍 **Real deployments:**
- **Be My Eyes / Be My AI:** blind users photograph things and ask ("Which of these cans is tomato soup?"). VizWiz (3.H) is built from this kind of question.
- **Support copilots:** "What does this error in my screenshot mean?"
- **Visual shopping:** "Is this jacket waterproof?" from a product photo.
- **Education:** "What's the next step in this geometry diagram?"

Why the split matters: (a) and (b) are really **classification**, so evaluation is strict and easy. (d) is free-form: "he is walking", "the man seems to be crossing the street"… all correct, all different. The professor: "Open-ended is obviously the future, but at least you can restrict to some of them."

### How the field got here: caption → paragraph → scene graph → VQA
The professor traced the history:
1. **Captions**, until people realised one sentence isn't enough. *"An image is worth 1000 words"* is literally true.
2. **Paragraph captioning**
3. **Scene graphs.** Downstream systems need the **individual pieces**, not prose. E.g. *man —playing→ cricket —with→ bat*. "Man" and "bat" are **grounded** to image regions, while "cricket" is a concept and isn't. Their boxes should **overlap**, because the man is holding the bat.
4. **VQA as probing.** "Rather than predicting a scene graph, why don't you ask specific questions? *Is there a man? Is there a bat? Is the man playing with the bat?*" Asking questions tests whether the VLM's representation holds all this information, without having to predict the whole structure.

```mermaid
flowchart LR
    SG["Scene graph<br/>(man)-[playing]->(cricket)<br/>(man)-[holding]->(bat)"] --> Q1["Is there a man? → yes"]
    SG --> Q2["Is there a bat? → yes"]
    SG --> Q3["What is the man playing? → cricket"]
    SG --> Q4["What is he holding? → a bat"]
```
*Each edge of the scene graph becomes a question, so VQA can test a structure without having to predict it.*

> 💡 **Learning Thought:** VQA is a **probing interface** for visual understanding. The professor's broader claim: *"Almost any task in the world that involves an image can be converted into a question."* Captioning is "Describe this image". Document extraction is "What's the invoice total?". That generality is why VQA-style instruction data powers most VLMs, and why VQA inherits every difficulty of vision **and** language.

---

## 3.B — The classic (pre-VLM) VQA architecture

"The idea is still there. It is the same idea we utilise over and over again. We just have better black boxes now."

```
Image ─► CNN pretrained on ImageNet (drop the last, class-specific layer) ─► image vector ─┐
                                                                                            ├─► fusion (concat / attention) ─► MLP ─► softmax over 2,000 answers
Question ─► RNN/LSTM ─► question vector ─────────────────────────────────────────────────┘
```

- **Backbone reuse:** train f_θ on ImageNet (image → class), then **cut off the last layer** (it's specific to the 1,000 ImageNet classes) and use up to the penultimate layer as a general image embedding. (ViTs are still often ImageNet-pretrained first.)
- **The 2,000-class trick:** "We were not very good at generating the answer and evaluating it." So the **top 2,000 most frequent answers** in the VQA dataset became a **2,000-way classifier**. That covers most questions, but it can never produce a rare answer.

![Classic VQA: CNN image features and RNN question features fused, then softmax over answers like elephant 0.970](assets/vqa-classic-cnn-rnn-classifier.png)
*Slide 22: image → CNN, "What is in the image?" → RNN, fuse (⊗), softmax → "elephant 0.970". The older decoder variant (bottom) generates "It is an elephant".*

### 💻 Code: the 2015 model in PyTorch
[`code/s3_classic_vqa_toy.py`](code/s3_classic_vqa_toy.py):
```python
class ClassicVQA(nn.Module):
    def __init__(self, vocab=5000, n_answers=2000, d=512):
        super().__init__()
        cnn = torchvision.models.resnet18(weights=None); cnn.fc = nn.Identity()   # ImageNet backbone minus its head
        self.cnn = cnn
        self.q_embed, self.q_lstm = nn.Embedding(vocab, 300), nn.LSTM(300, d, batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(d, 1024), nn.ReLU(), nn.Linear(1024, n_answers))  # W2 σ(W1 h)

    def forward(self, image, question_ids):
        v = self.cnn(image)                                  # image vector (512)
        _, (h, _) = self.q_lstm(self.q_embed(question_ids))  # question vector (512)
        return self.classifier(v * h[0])                     # element-wise fusion -> 2,000 logits
```
```
logits (1, 2000) -> top-3 (untrained, seed 0): [('ans_853', 0.00055), ('ans_1606', 0.00054), ('ans_1827', 0.00054)]
Can it ever answer 'a satellite dish'? False
```
The untrained output is uniform (1/2000 ≈ 0.0005). The second line is the real limitation: **if an answer isn't among the 2,000 classes, the model can never produce it**, no matter how well trained.

### The same trick on a modern transformer (for yes/no and multiple choice)
You can still skip decoding entirely:
1. Feed [system prompt + 256 image tokens + question tokens] into the transformer
2. Take the **last-layer representation** of every token and **pool** it into one vector *h*
3. Apply an **MLP**: W₂ σ(W₁ h)
4. Apply **softmax** over the answer classes

"You completely forget about the decoding part", and evaluation becomes exact. That is still the right design for high-volume binary checks.

### 💻 Code: decoding-free yes/no with a modern VLM
You don't even need a new head. Read the VLM's **next-token logits** for "Yes" vs "No" ([`code/s3_yes_no_scoring.py`](code/s3_yes_no_scoring.py)):
```python
def p_yes(image, question):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question + " Answer yes or no."}]}]
    inputs = proc(text=proc.apply_chat_template(msgs, add_generation_prompt=True), images=[image], return_tensors="pt")
    logits = model(**inputs).logits[0, -1]                 # ONE forward pass, no generation
    yes = max(logits[tok.encode(w, add_special_tokens=False)[0]] for w in (" Yes", "Yes", " yes", "yes"))
    no  = max(logits[tok.encode(w, add_special_tokens=False)[0]] for w in (" No", "No", " no", "no"))
    return torch.softmax(torch.stack([yes, no]), 0)[0].item()
```
On the dog-and-bird beach photo:
```
P(yes)=0.979  Is there a dog in the image?
P(yes)=0.066  Is there a bird in the image?      <- the bird IS there
P(yes)=0.976  Is there a cat in the image?       <- there is NO cat
P(yes)=0.000  Is there a person in the image?
```
You get a **probability**, not just text, which you can threshold, log and calibrate. It also exposes the model's belief directly. It is **97.6% sure a cat is present** and **93% sure the real bird isn't**. The "two dogs" caption from Section 2 wasn't a wording slip. The model really misperceives the scene, and no prompt wording would fix that.

**Today:** give the image and question to a generative VLM, and it produces the answer in whatever format you ask for.

---

## 3.C — Which model fits

| Model | Best for |
|---|---|
| **LLaVA-style** | General, conversational visual Q&A |
| **Qwen-VL** | Questions that need **grounding, pointing at a region, or reading text** in the image (native OCR + grounding) |

**Rule of thumb:** a general question about the whole image → either model. A question about a **specific region**, or one that needs **OCR** → **Qwen-VL**.

The professor notes LLaVA has had many versions (1.5 → 1.6 → OneVision, which was a big accuracy jump), so there's no hard rule, but "when we start with some good baseline, we generally start with Qwen".

---

## 3.D — Prompting for reliable VQA

| Technique | When | Example |
|---|---|---|
| **Zero-shot direct** | Most everyday questions | "You are a helpful vision-language assistant. Answer the question." |
| **Chain-of-thought** | **Counting, comparison**, arithmetic | "First list every object and its colour, then count how many are red." |
| **Few-shot (K-shot)** | Domain-specific vocabulary the model won't default to | Give a few Q→A example pairs in the prompt |
| **Majority voting** (from the transcript) | When one sample is unreliable | Sample K answers, take the most common |

### Majority voting / self-consistency, explained
Hugging Face's `generate()` defaults to **greedy decoding**: at each step, take argmax P(t₂ | t₁). Set **`do_sample=True`** and each token is instead **sampled** from the distribution, so lower-probability tokens sometimes get picked. Run it K times and you get K different reasoning paths.

**Worked example (K = 5), "How many red balls?":** sampled answers = 3, 3, 4, 3, 2
- Majority answer = **3** (3 of 5 votes)
- Agreement 3/5 = **0.6** works as a rough confidence signal. Low agreement means the question deserves a human look.

"A very generic technique, nothing to do with VLMs specifically."

### 💻 Code: majority voting on a real image
[`code/s3_majority_vote.py`](code/s3_majority_vote.py) uses the slide's umbrella-crowd photo (shown in Section 2.E):
```python
def vote(q, k=7):
    torch.manual_seed(0)
    answers = []
    for _ in range(k):
        a = ask(crowd, q, sample=True, max_new_tokens=10)          # do_sample=True, temperature=0.8
        n = re.search(r"\d+", a)
        answers.append(n.group() if n else a)
    best, cnt = Counter(answers).most_common(1)[0]
    print("samples:", answers, f"-> majority = {best}  (agreement {cnt}/{k})")
```
```
Q: How many umbrellas are open in this image? Answer with a single number.
greedy : 2.
samples: ['2', '2', '2', '2', '2', '2', '2']  -> majority = 2  (agreement 7/7 = 1.00)
Q: How many people are in this image? Answer with a single number.
greedy : 8.
samples: ['8', '8', '8', '8', '8', '8', '7']  -> majority = 8  (agreement 6/7 = 0.86)
```
**Read it:**
- Umbrellas: **2** with 7/7 agreement, and it's correct.
- People: **8** with 6/7 agreement. Counting by hand gives about 9 once you include people partly hidden at the back.

**High agreement is not correctness.** Self-consistency measures how *sure* the model is, not how *right*. It catches unstable answers (low agreement means escalate). It can't catch a model that's **consistently wrong**, which is what we saw with the cat in the previous demo.

📚 **Go deeper:** [Chain-of-Thought prompting (Wei et al.)](https://arxiv.org/abs/2201.11903) · [Self-Consistency (Wang et al.)](https://arxiv.org/abs/2203.11171), the paper behind majority voting

### Worked example (slide)
> **Q:** "How many distinct peaks are visible in this image?"
> **A:** "Two — a smaller peak on the left and a taller, sunlit peak on the right."

The answer **justifies itself** by saying where each peak is. That is the auditability CoT gives you.

> ✅ **Why CoT helps with counting:** it forces an **auditable intermediate list**. "An explicit list is easier to verify than a jump straight to a count", and it reduces confident errors. (Live quiz Q3.)

---

## 3.E — Are VLMs even looking at the image? (research, 2026)

This is the most important material in the section. The professor: *"Prompting is easy, but the nuances of the prompting are where the understanding lies."*

### Test 1: where is the model looking? (Grad-CAM)
**Grad-CAM** backpropagates from an **output answer token** to the image patches to find which patches, if blacked out, would have changed that token. It gives a heatmap where red means high influence ("like the eye of the cyclone on a weather map").

LLaVA-1.5 on counting questions gives **all four combinations**:

| | Correct answer | Wrong answer |
|---|---|---|
| **Correct focus** | ✅ "How many horses?" → looks at snout and distinctive features → 1 ✔ | ❌ "How many clocks?" → looks **at the clocks** but answers wrongly (misinterpretation) |
| **Wrong focus** | ⚠️ "How many clocks?" → **correct** answer while looking at something that isn't a clock | ❌ "How many fire hydrants?" → focuses on two hydrant-*looking* things and says 2 (the truth is 1) |

The bottom-left cell is the worrying one: **a correct answer for the wrong reason**. You can't trust accuracy alone.

![Grad-CAM focus of LLaVA-1.5 on counting questions: correct/wrong focus vs correct/wrong answer](assets/vqa-gradcam-focus-grid.png)
*Slide 24: rows = correct vs wrong focus, columns = correct vs wrong answer. Red regions drove the answer token.*

**How Grad-CAM works (in one breath):** pick the answer token's logit, backpropagate it to the vision encoder's last feature map, average the gradient over each channel to get a channel weight, compute a weighted sum of the channels, apply ReLU, and upsample onto the image. Regions whose features *increase* the answer's score light up. A few lines with the `pytorch-grad-cam` library, or with hooks on the ViT's last block.

### Test 2: does the image matter at all? (the blurred baseline)
Give LLaVA-1.5 the **original image** and a **fully blurred** version, then compare the probability of each answer token.

Example: *"Do you know who painted this?"* → "The painting was created by **Leonardo da Vinci**."
- With the image blurred, **most token probabilities barely change**.
- Only the "**Leon**" token depends noticeably on the image. "ardo", "da", "Vinci", and the whole sentence frame come from the **language model**.
- In a related failure, the model tends to attribute abstract-looking paintings to Van Gogh. It has learned the shortcut "abstract ⇒ Van Gogh".

![Mona Lisa vs blurred baseline: LLaVA answers Leonardo, then Van Gogh; per-token probability plot with and without vision](assets/vqa-blur-test-leonardo.png)
*Slide 25: top, original vs fully blurred input. Bottom, per-token probability with vision (blue) vs without (orange). The lines overlap almost everywhere except "Leon".*

**Conclusion:** *"Most tokens in the response are not very dependent on the visual information."* VLMs, especially open-source ones, are often **overpowered by their strong language models**.

### 💻 Code: run the slide's experiment yourself
[`code/s3_blur_token_probs.py`](code/s3_blur_token_probs.py) takes a fixed, **correct** description of the dog photo and asks: *how probable does the VLM find each of its tokens, given the real image vs a blurred one?* This is **teacher forcing**. We don't generate. We score a given answer.
```python
def token_logprobs(image):
    prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)
    full = proc(text=prompt + " " + answer, images=[image], return_tensors="pt")
    logits = model(**full).logits[0]
    ids = full["input_ids"][0]
    lp = torch.log_softmax(logits[:-1], -1).gather(1, ids[1:, None])[:, 0]   # log p(token_t | tokens_<t, image)
    return lp[n_prompt - 1:]                                                   # keep only the answer's tokens

lp_img, lp_blur = token_logprobs(img), token_logprobs(img.filter(ImageFilter.GaussianBlur(25)))
```
```
(abridged: 10 of the 18 answer tokens shown)
     token  p(orig)  p(blur)  change
    ' and'    0.955    0.660  +0.295  <-- image-dependent
  ' white'    0.993    0.948  +0.045
    ' dog'    0.681    0.000  +0.680  <-- image-dependent
' sitting'    0.605    0.042  +0.562  <-- image-dependent
     ' on'    0.745    0.379  +0.366  <-- image-dependent
     'ble'    0.962    0.835  +0.127
     ' to'    0.996    0.995  +0.001
  ' black'    0.275    0.010  +0.265  <-- image-dependent
   ' bird'    0.056    0.003  +0.053
       '.'    0.907    0.726  +0.181
mean |change| = 0.174 over 18 tokens
```
**Read it the way the slide does:**
- **Content words drive the image dependence.** "dog" goes from 0.681 to **0.000**, and "sitting" from 0.605 to 0.042. Without the image, the model has no reason to expect a dog.
- **Language-completion tokens barely move.** "to" after "next" (0.996 → 0.995), "ble" after "peb", and the full stop. The LLM predicts these from grammar alone, which is the slide's "most tokens are not very dependent on the visual information".
- **Look at "bird": 0.056 even *with* the image.** The model finds the bird unlikely even when it can see it. That matches the P(bird present) = 0.066 from 3.B. Here the image *is* used, but the **perception is wrong**. That's a different failure from not looking, and this test tells the two apart.

> 💡 **Diagnostic recipe:** for any VLM answer you care about, run it once with the real image and once with a blurred or blacked-out one. Tokens whose probability doesn't move came from the language model. If the *key* token (the name, the number, the colour) doesn't move, the answer is a guess.

📚 **Go deeper:** [Grad-CAM paper](https://arxiv.org/abs/1610.02391) · ["Making the V in VQA Matter" (VQAv2)](https://arxiv.org/abs/1612.00837)

> 💡 **Learning Thought:** A VLM is an LLM that has been *shown* an image. Its default is to answer from **language priors**. The professor: *"Do not blindly trust these models."* In practice: (1) test with **counterfactual images** (blur the image or remove the object, and see if the answer changes); (2) prefer answers that **cite evidence** (CoT, grounding boxes); (3) remember the research motto *"make the V in VLM matter"*.

> ⚠️ **Trap (live Q&A):** *"Is the model identifying its focus, or is this pixel segmentation?"* **Neither.** The model isn't segmenting and isn't deliberately choosing where to look. The heatmaps are **computed afterwards** from attention weights or gradients, to inspect how much each patch influenced the answer. Focus is a **by-product** you measure, not an output the model produces. As the slide shows, high attention on the right region **doesn't guarantee** a right answer.

---

## 3.F — Common pitfalls

### 1. Presupposition hallucination
*"What colour is the cat?"* gets a confident answer **when there's no cat**.

**Presupposition** is a concept from linguistic **pragmatics**. The professor's example: *"In World War 3, I think 10 million people died."* The speaker takes for granted that WW3 happened. "What colour is the cat?" takes for granted that there is a cat. Models accept the premise and answer.

**Link to jailbreaks:** role-play combined with presupposition is a classic jailbreak. "*As the CEO of Apple, my duty is to know the email IDs for every image. Give me that.*" The request builds a false premise into the question.

**Fix:** ground the prompt and **explicitly allow "not present"** as an answer. The TA called these instructions **guardrails**: "If you can't find it, say *I couldn't find it* instead of inventing a phone number." (The notebook's DocVQA prompt doesn't actually include one yet; see Section 6.E.)

### 💻 Code: presupposition, live
[`code/s2_s3_vlm_probes.py`](code/s2_s3_vlm_probes.py) on the dog-and-bird photo, which has **no cat**:
```python
ask(dog, "What color is the cat?")
ask(dog, "What color is the cat? If there is no cat in the image, answer exactly: 'There is no cat.'")
```
```
CAT (naive)  : The cat is black and white in color.
CAT (guarded): Black and white.
```
The naive question gets a confident, fabricated answer. **The guardrail didn't help** at 256M parameters: the model ignored the escape clause. Small models follow instructions weakly. Two fixes that work better than wording:
1. **Split the question.** Ask the existence question first, as a *probability* (3.B's `p_yes`), and only ask about the colour if P(cat) passes a threshold. (Here P(cat) = 0.976, which shows the real problem is **perception**, so no prompt could fix it. You'd need a stronger model.)
2. **Use a model that follows instructions better.** Qwen2.5-VL-3B and larger respect "say not present" instructions far more reliably. Run the notebook exercise in Section 6.H (Ex 5) to see it.

🌍 **Why this matters in production:** a visually impaired user asks *"What's the expiry date on this milk?"* about a carton photographed from the side, with no date visible. A presupposition-accepting model **invents a date**. That's a safety bug, not a style issue.

### 2. Overconfidence
A wrong answer is stated as confidently as a right one, so the confidence score tells you nothing. Making confidence scores meaningful (**calibration**) is an open research problem. Humans hedge: "the colour is black, but I'm not sure it's a cat." Getting open-source models to **abstain** like that is hard. Closed APIs do better partly because they sit behind "many wrappers and harnesses".

### 3. Ambiguous questions
*"Is this good?"* has no defined criterion, so the model has to guess what you mean. **Fix:** state the criterion ("Is this weld free of visible cracks?").

---

## 3.G — Why VQA is hard

- **Three skills at once:** visual **perception** + **language** understanding + **world/commonsense knowledge and reasoning** (spatial, causal, commonsense)
- **Question ambiguity:** the question can be underspecified before the image is even considered, or so complex that the language side alone overwhelms the model
- **Answers beyond the pixels:**
  - An object **outside the frame**, visible only as a **shadow** (a person walking toward a lamppost that is off-screen)
  - **Counting** and **spatial reasoning** can't be read directly off the image
  - **External knowledge** ("What's the stance of this PM towards that party?") isn't in the pixels → **RAG** (next week's topic: "this cannot always be solved by just VLMs")
- **Open-ended answer space:** no fixed label set, so evaluation itself is hard

---

## 3.H — Popular datasets

| Dataset | Scale | What it tests (the aspect to remember) |
|---|---|---|
| **VQAv2** | ~1.1M questions on COCO images | **General-purpose.** The standard benchmark |
| **GQA** | ~22M questions, generated from scene graphs | **Compositional** reasoning |
| **OK-VQA** | ~14K questions | **External knowledge** beyond the image |
| **TextVQA** | ~45K questions | **Reading text** (OCR) in the image |
| **VizWiz** | ~31K questions | **Real blind/low-vision users**, naturally noisy photos |

The professor: *"You don't need to remember every one. Remember the aspects: general, compositional, knowledge, OCR, noisy."*

### Two stories behind the datasets
- **Why VQA became VQAv2 (bias):** VQA v1 (~2015–16, Dhruv Batra and Devi Parikh's lab) could be gamed from **language priors**. E.g. "Is there a…?" → "yes" was usually right. VQAv2 (2017) **balanced** it by pairing similar images that have **different answers** to the same question, so the model has to look.
- **Compositionality (GQA):** if 90% of training cats are brown, the model learns "cat ⇒ brown". A compositional model that knows *brown cat* should generalise to *ginger cat*, *black cat*, *black dog*, i.e. **any colour × any object**. Datasets are biased toward certain pairs, and GQA tests the rest.

> ⚠️ Most of these are now **inside pretraining and instruction-tuning data**, which is why people keep building new benchmarks.

### 💻 Code: look at real VQAv2 rows
[`code/s3_peek_vqav2.py`](code/s3_peek_vqav2.py) loads the 100-row mini-VQAv2 split the course notebook uses:
```python
from datasets import load_dataset
ds = load_dataset("dutta18/miniVQAv2DatasetForCollab", split="train")
```
```
Dataset({ features: ['multiple_choice_answer', 'question', 'image'], num_rows: 100 })
[0]  Q: What is this kid playing with?   answer (majority of ~10 humans): 'kite'
[5]  Q: What is she riding on?           answer (majority of ~10 humans): 'motorcycle'
[81] Q: Are there any balconies?         answer (majority of ~10 humans): 'no'
```
- **Row 5 is the lecture's Karnataka motorcycle-and-satellite-dish photo** (Section 2.E).
- This mini split keeps only the **single most common** human answer (`multiple_choice_answer`). The full VQAv2 has all 10 answers per question, which is what the min(n/3, 1) metric in 3.J needs.

📚 **Go deeper:** [VQA (v1)](https://arxiv.org/abs/1505.00468) · [VQAv2](https://arxiv.org/abs/1612.00837) · [GQA](https://arxiv.org/abs/1902.09506) · [OK-VQA](https://arxiv.org/abs/1906.00067) · [TextVQA, "Towards VQA Models That Can Read"](https://arxiv.org/abs/1904.08920) · [VizWiz](https://arxiv.org/abs/1802.08218)

---

## 3.I — How SOTA VQA models are trained

1. **Start from a base generative VLM** (the LLaVA / Qwen-VL recipe), then fine-tune on VQA-style instruction data
2. **Mix several VQA datasets** (VQAv2 + GQA + OK-VQA + TextVQA) to broaden question-type coverage
3. **Answer-format supervision:** train short, direct answers where benchmarks expect them, not verbose chat
4. **Preference tuning** (RLHF, DPO) specifically to reduce **hallucinated and overconfident** answers

### Building preference data synthetically
A preference triple is **(x, y, y′)**: x = image + question, y = **preferred** (correct) answer, y′ = **dispreferred/rejected** answer. Three cheap ways to make y′ or its counterpart:

| Perturbation | Construction | What it teaches |
|---|---|---|
| **Random answer** | Image + Question + ⟨random answer from the corpus⟩ | In a large corpus a random answer is almost always wrong. Don't just emit plausible text |
| **Random question** | Image + ⟨random question⟩ + Answer (keep y fixed, swap the question) | The answer must depend on **the question** |
| **Image perturbation** | ⟨Image with the relevant objects **blackened / blurred**⟩ + Question + Answer | The answer must depend on **the pixels**: "making the V matter" |

The third row directly addresses the blurred-image finding in 3.E. If the answer stays the same when the relevant pixels are removed, the model is being rewarded for ignoring the image.

**Plugging into DPO (for reference):**
  L = −log σ( β [ log π_θ(y|x)/π_ref(y|x) − log π_θ(y′|x)/π_ref(y′|x) ] )
Worked example: the model raised the log-ratio of y by 0.5 and lowered y′ by 0.3, with β = 0.1 → β · 0.8 = 0.08 → σ(0.08) = 0.520 → **L = 0.654**. As the gap widens, the loss falls toward 0.

### 💻 Code: build the triples, compute the DPO loss
[`code/s3_vqa_metric_dpo.py`](code/s3_vqa_metric_dpo.py):
```python
for i, ex in enumerate(data):
    other = data[(i + 1) % len(data)]
    triples.append({"x": (ex["image"], ex["q"]), "chosen": ex["a"], "rejected": other["a"], "kind": "random answer"})
    triples.append({"x": (ex["image"], other["q"]), "chosen": "cannot be determined from the image",
                    "rejected": ex["a"], "kind": "random question"})     # old answer no longer fits the new question
    triples.append({"x": (ex["image"].replace(".jpg", "_objects_blacked.jpg"), ex["q"]),
                    "chosen": "not visible in the image", "rejected": ex["a"], "kind": "image perturbation"})

def dpo_loss(logratio_chosen, logratio_rejected, beta=0.1):
    z = beta * (logratio_chosen - logratio_rejected)
    return -math.log(1 / (1 + math.exp(-z)))
```
```
{'x': ('img_001.jpg', 'What is the man holding?'), 'chosen': 'a bat', 'rejected': '2', 'kind': 'random answer'}
{'x': ('img_001.jpg', 'How many dogs are there?'), 'chosen': 'cannot be determined from the image', 'rejected': 'a bat', 'kind': 'random question'}
{'x': ('img_001_objects_blacked.jpg', 'What is the man holding?'), 'chosen': 'not visible in the image', 'rejected': 'a bat', 'kind': 'image perturbation'}

chosen +0.0, rejected +0.0 -> loss 0.693     (= ln 2: the model can't tell them apart yet)
chosen +0.5, rejected -0.3 -> loss 0.654     (the worked example above)
chosen +3.0, rejected -3.0 -> loss 0.437
```
Look at the image-perturbation triple: once the bat is blacked out, "a bat" becomes the **rejected** answer. Training on it penalises the model for answering from the question alone ("what is the man holding?" → "a bat" is a strong language prior), which is exactly the failure the blur test exposed. In practice these dicts go straight into TRL's `DPOTrainer` as `prompt` / `chosen` / `rejected` (plus `images`).

📚 **Go deeper:** [DPO paper](https://arxiv.org/abs/2305.18290) · [TRL `DPOTrainer` docs](https://huggingface.co/docs/trl/dpo_trainer), which include a vision-language section

> ⚠️ **Trap (live Q&A):** *"How is adversarial training done for vision LLMs?"* The same idea as classic adversarial training, applied at several levels. **Image level:** perturbations that flip the answer or caption. **Vision-encoder level:** train on perturbed images for robust features. **Multimodal level:** adversarial image–text pairs during instruction tuning. **Objective:** the correct answer for **both clean and adversarial** versions of the same image.

---

## 3.J — Metrics, performance, remaining challenges

### The standard VQA accuracy metric (and why it punishes verbose models)
Each question has **10 human answers**. A predicted answer scores:

  **Acc(ans) = min( #humans who gave that exact answer / 3 , 1 )**

(The official metric averages this over subsets of 9 annotators and normalises case, articles and punctuation.)

**Worked example**, "What is the man doing?". Human answers: "walking" ×4, "crossing street" ×3, "walking on street" ×3.

| Model answer | Humans matching | Score |
|---|---|---|
| "walking" | 4 | min(4/3, 1) = **1.0** |
| "crossing street" | 3 | min(3/3, 1) = **1.0** |
| "The man appears to be walking across the street." | 0 exact matches | **0.0** |

The third answer is **correct and more informative** and scores **zero**. That is why **GPT-4V's free-form style scored poorly on strict VQA metrics despite strong human-judged quality**.

### 💻 Code: the metric, with normalisation
[`code/s3_vqa_metric_dpo.py`](code/s3_vqa_metric_dpo.py):
```python
def normalize(a):
    a = a.lower().strip().rstrip(".")
    a = re.sub(r"\b(a|an|the)\b", " ", a)          # drop articles, as the official script does
    return " ".join(a.split())

def vqa_accuracy(pred, human_answers):
    n = sum(normalize(h) == normalize(pred) for h in human_answers)
    return min(n / 3, 1.0)

humans = ["walking"] * 4 + ["crossing street"] * 3 + ["walking on street"] * 3
```
```
1.00  <- 'walking'
1.00  <- 'Crossing the street.'                         (normalisation rescues "the" and the full stop)
0.00  <- 'The man appears to be walking across the street.'
```
Normalisation forgives case, articles and punctuation, and **nothing else**. "Crossing the street." scores 1.0, while a perfect sentence scores 0. The official script also maps number words ("two" → "2") and a few contractions, but still needs the answer to be a short phrase.

📚 **Go deeper:** [VQA evaluation page (the official metric + code)](https://visualqa.org/evaluation.html)

### Performance takeaways
- Side-by-side human evaluation: **GPT-4V and Qwen-VL-Chat rated comparably** on VQAv2/OK-VQA-style questions, zero-shot
- **Benchmark numbers alone can undersell real capability.** *How you evaluate matters as much as which model you use.*
- The research response: evaluate the **reasoning, not just the answer**: generate an explanation → evaluate the explanation → **verify the chain of thought**. An active research area (the professor's group included)
- **Compositional questions (GQA-style)** remain a weak point for both open and closed models

### Remaining challenges
- **Commonsense / world-knowledge** questions (OK-VQA) lag behind pure perception. Parametric knowledge has a **cutoff** (a 2014 model doesn't know 2015) → you need **RAG**: "there is no other way"
- **Counting is unreliable past ~5–10 objects**, especially with occlusion. The professor suspects this is **fundamental to Transformers** and may need architectural changes
- Models answer **confidently when the question presupposes something false or unanswerable** (blank images, false premises)
- **Compositional multi-step** questions underperform simple lookups, which is why "thinking" models came about

---

## ✅ Quick Check (from the deck; answers below)
1. What distinguishes VQA from image captioning?
2. When should you prefer Qwen-VL over a base LLaVA-style model for VQA?
3. What's the fix for presupposition hallucination?
4. Why use chain-of-thought for a counting question?
5. What makes "is this good?" a weak VQA prompt?

<details><summary>Answers</summary>

1. **VQA answers a specific question**; captioning describes generally
2. When the question needs **grounding or reading text** in the image
3. **Ground the prompt and allow "not present" as an answer**. Explicitly permitting a null answer reduces confident fabrication
4. It **forces an auditable intermediate list**, reducing confident errors
5. It **lacks a defined criterion** for "good"
</details>

---

## 🎯 Interview Questions

**Q1. Your VQA model has 85% accuracy. How do you know it's actually looking at the image?**
> Accuracy can't tell you. Two diagnostics from the lecture: (1) **Grad-CAM / attention maps**: check that focus is on the relevant region, and remember that right answers can come with wrong focus. (2) **Counterfactual inputs**: blur the image or black out the object the question asks about. If token probabilities barely move, the model is answering from language priors. Fix with image-perturbation preference data, grounding prompts, or a model with stronger visual grounding.

**Q2. What is presupposition hallucination, and how do you mitigate it in a production visual assistant?**
> The question takes something for granted ("What colour is the cat?") and the model accepts it, answering even when there's no cat. It's a pragmatics failure, and role-play jailbreaks exploit the same weakness. Mitigations: tell the model it may say "not present / cannot determine"; a two-step prompt (first "Is there a cat?", then the colour); preference-tune on false-premise questions where abstention is the preferred answer; log abstention rates.

**Q3. Why do strict VQA metrics undervalue GPT-4-class models?**
> VQA accuracy is exact match against 10 human answers: min(matches/3, 1). "The man appears to be walking across the street" matches no one and scores 0, while "walking" scores 1. Verbose, correct, differently phrased answers get penalised. Use answer-format supervision if you're chasing a benchmark, and LLM-judge or human evaluation if you care about real quality.

**Q4. How would you build DPO preference data for a VQA model without human labelling?**
> Start from (image, question, correct answer) triples. Create rejected answers by (a) sampling a random answer from the corpus, (b) keeping the answer but swapping in a random question, (c) blacking out or blurring the objects the question is about so the original answer is no longer supported. (c) directly trains visual grounding. Train with DPO against a frozen reference model.

**Q5. Why can't you just classify over answers like the 2015 models did?**
> You can for yes/no and multiple choice. Pool the last layer, MLP, softmax, and you get exact evaluation cheaply. But a top-2,000-answer classifier can't produce rare or compositional answers, and it cuts off open-ended questions. Generative decoding covers the open space at the cost of harder evaluation.

**Q6. Counting fails past ~10 objects. What do you do in a product that must count stock on shelves?**
> Don't rely on the VLM's direct answer. Use CoT to enumerate by region (auditable), majority voting over samples as a confidence signal, or better, a detector or segmenter (SAM + counting) for the count, with the VLM only interpreting the result. Tile high-resolution images. Send low-agreement cases to a human.

---

## ✅ Self-check
- [ ] I can list the 4 question types and why yes/no and MC are "really classification"
- [ ] I can explain the caption → scene graph → VQA lineage
- [ ] I can sketch the 2015 CNN+RNN+2000-class VQA model
- [ ] I can explain majority voting with `do_sample=True`
- [ ] I can describe both "are VLMs looking?" experiments and what they showed
- [ ] I can define presupposition and link it to jailbreaks
- [ ] I can match VQAv2 / GQA / OK-VQA / TextVQA / VizWiz to the aspect each tests
- [ ] I can build (x, y, y′) preference triples three ways
- [ ] I can compute VQA accuracy and show why a verbose correct answer scores 0
- [ ] I can score yes/no from logits without generating text
- [ ] I have run `s3_blur_token_probs.py` and can say which tokens came from the image
- [ ] I can explain why 7/7 agreement doesn't mean the answer is right
