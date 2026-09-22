# Runnable code for the Module 4 · Lecture 2 notes

Every output pasted into the notes came from running these files on a MacBook CPU (Sept 2026).
Models used: CLIP ViT-B/32, SAM ViT-B, OWLv2, SmolVLM-256M-Instruct, all-MiniLM-L6-v2. No GPU needed.
The Qwen2.5-VL-3B examples live in the course notebook (`../../VLM_Applications_UpGrad_Tutorial_Week2_Qwen_Student_ver.ipynb`). Run those on a Colab T4.

```bash
cd learning-notes/code
export UV_CACHE_DIR=/tmp/uvcache HF_HOME=/tmp/hf        # any writable cache dir
# one helper that works in zsh AND bash (a $VAR of flags does not word-split in zsh)
vrun() { uv run --with torch --with torchvision --with transformers --with pillow --with numpy \
                --with scipy --with num2words --with pydantic --with peft --with datasets \
                --with sacrebleu --with sentence-transformers python "$@"; }
```

| Section | Script | What it shows | Model download |
|---|---|---|---|
| 0 SAM | `s0_sam_math.py` | patchify, MAE masking, attention cost, Fourier encoding, dot-product mask, conv / transposed conv, focal & Dice numbers | none |
| 0 | `s0_focal_dice.py` | Focal + Dice on a tiny object: lazy vs soft vs confident predictions | none |
| 0 | `s0_two_way_block.py` | a minimal two-way attention block with SAM's shapes | none |
| 0 | `s0_sam_point_prompt.py` | encode once, click many; 3 masks + IoU; saves `assets/sam-demo-dog-3-masks.png` | SAM ViT-B ~375 MB |
| 0 | `s0_auto_mask_grid.py` | automatic mask generation (data-engine Stage 3), about 1 min on CPU | (same) |
| 0 | `s0_grounded_sam.py` | text → OWLv2 boxes → SAM masks | OWLv2 ~600 MB |
| 1 | `s1_vision_backbones.py` | every model's ViT, read from its config | config JSON only |
| 1 | `s1_one_image_four_paradigms.py` | CLIP on the dog/bird photo | CLIP ~600 MB |
| 1 | `s1_metric_flips_ranking.py` | exact-match vs lenient metric flips the ranking | none |
| 2 | `s2_show_and_tell_toy.py` | CNN → LSTM h0 captioner, teacher-forced loss | none |
| 2 | `s2_clip_zero_shot.py` | CLIP tagging + raw cosines (the satellite-dish lesson) | CLIP |
| 2 | `s2_prompt_styles.py` | 4 caption prompt styles on one image | SmolVLM ~500 MB |
| 2 | `s2_bleu_paraphrase.py` | sacreBLEU on a correct paraphrase | none |
| 2 | `s2_clipscore.py` | CLIPScore vs hallucinated captions | CLIP |
| 2 | `s2_lora_setup.py` | LoRA wraps a VLM: 0.29% trainable | SmolVLM |
| 2/3 | `s2_s3_vlm_probes.py` | caption, presupposition (cat), blur caption | SmolVLM |
| 3 | `s3_classic_vqa_toy.py` | the 2015 CNN+LSTM+2000-class VQA model | none |
| 3 | `s3_yes_no_scoring.py` | P(yes) straight from logits | SmolVLM |
| 3 | `s3_blur_token_probs.py` | slide 25's blurred-image token-probability test | SmolVLM |
| 3 | `s3_majority_vote.py` | self-consistency voting on counting | SmolVLM |
| 3 | `s3_vqa_metric_dpo.py` | VQA accuracy metric, DPO loss, preference triples | none |
| 3 | `s3_peek_vqav2.py` | real rows of the notebook's mini-VQAv2 | dataset ~few MB |
| 4 | `s4_ocr_layout_reconstruct.py` | flat OCR text vs rebuilding the table from boxes | none |
| 4 | `s4_resolution_strip.py` | saves `assets/doc-resolution-legibility-strip.png` | none |
| 4 | `s4_anls_resolution_tokens.py` | ANLS vs exact match, DPI table, Qwen token calculator | none |
| 4 | `s4_make_invoice.py` → `s4_invoice_schema.py` | render the slide invoice; VLM → JSON → pydantic validation | SmolVLM |
| 4 | `s4_preprocess_scan.py` | deskew / contrast / crop / upscale | none (run `s4_make_invoice.py` first) |
| 4 | `s4_dave_mechanisms.py` | raw vs normalised MAE target; learned weight merging | none |
| 4 | `s4_docvlm_queries.py` | 1,200 OCR tokens → 64 learnable queries | none |
| 5 | `s5_chains_and_charts.py` | p^n chain accuracy; extract-then-reason + robustness check | none |
| 5 | `s5_make_chart.py` → `s5_vlm_chart_and_reasoning.py` | chart QA direct vs extract; helmet rule direct vs CoT | SmolVLM |
| 7 | `s7_coverage_matcher.py` | design-requirement ↔ verification matching + labelled eval | MiniLM ~90 MB |

Shared helper: `vlm_small.py` (SmolVLM-256M wrapped in the notebook's chat-template → processor → generate → trim pattern).
