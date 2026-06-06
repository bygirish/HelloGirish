# Lecture 3: The Transformer Architecture — Study Index
**Prof. Niloy Ganguly, IIT Kharagpur | upGrad IITK**

---

## Learning Path (Follow in Order)

| # | File | Topics Covered | Key Formulas |
|---|------|----------------|--------------|
| 1 | [S1_Motivation_Why_Attention.md](S1_Motivation_Why_Attention.md) | RNN limitations, Seq2Seq, Bahdanau Attention, Why attention alone wasn't enough | c_t = Σ αᵢ·hᵢ |
| 2 | [S2_Self_Attention_Mechanism.md](S2_Self_Attention_Mechanism.md) | Q/K/V framework, Self-attention recipe, Vectorized form, Numerical walkthrough | Output = softmax(QKᵀ)V |
| 3 | [S3_Encoder_Building_Blocks.md](S3_Encoder_Building_Blocks.md) | FFN, Residual Connections, Layer Norm, Scaled Attention, Positional Encoding | softmax(QKᵀ/√d_k)V |
| 4 | [S4_Multi_Head_Attention.md](S4_Multi_Head_Attention.md) | Why multiple heads, Head specialization, 2-head concrete example, PyTorch | MultiHead = Concat(heads)·W^O |
| 5 | [S5_Transformer_Decoder.md](S5_Transformer_Decoder.md) | Masked self-attention, Cross-attention, Full decoder block, Enc-dec comparison | e_ij = -∞ for j≥i |
| 6 | [S6_Autoregressive_Decoding.md](S6_Autoregressive_Decoding.md) | Decoder loop, Greedy/Beam/Sampling, KV Cache, Stop conditions | Top-p, Temperature |
| 7 | [S7_Modern_Impact_Applications.md](S7_Modern_Impact_Applications.md) | BERT vs GPT, LoRA, ViT/Whisper/CLIP, Pretraining paradigm, Emergent capabilities | LoRA: W + AB |

---

## Quick Reference — All Core Formulas

```
Self-Attention:      Output = softmax(QKᵀ/√d_k) · V
Multi-Head:          MultiHead(X) = Concat(head₁...headₕ) · W^O
FFN:                 FFN(x) = ReLU(xW₁+b₁)W₂+b₂
Residual:            x_l = F(x_{l-1}) + x_{l-1}
LayerNorm:           x' = (x-μ)/(σ+ε)
Positional Enc:      PE(pos,2i) = sin(pos/10000^(2i/d))
Causal Mask:         e_ij = -∞ for j ≥ i
Cross-Attention:     Output = softmax(Q_dec · K_enc^T / √d_k) · V_enc
Temperature:         probs = softmax(logits / T)
LoRA:                W_fine = W_pretrained + A·B (r << d)
```

---

## Architecture at a Glance

```
ENCODER (×6)                    DECODER (×6)
─────────────                   ─────────────────────────────
Input Embedding                 Output Embedding
+ Pos. Encoding                 + Pos. Encoding
     ↓                               ↓
Multi-Head                      Masked Multi-Head
Self-Attention                  Self-Attention (causal)
     ↓                               ↓
Add & Norm                      Add & Norm
     ↓                               ↓
Feed-Forward                    Multi-Head
     ↓                          Cross-Attention ← enc_out (K,V)
Add & Norm                           ↓
     ↓                          Add & Norm
  enc_out                            ↓
(K, V for                       Feed-Forward
cross-attn)                          ↓
                                Add & Norm
                                     ↓
                               Linear + Softmax
                                     ↓
                              Output Probabilities
```

---

## Interview Readiness Checklist

- [ ] Explain 5 limitations of RNNs and how Transformer addresses each
- [ ] Walk through Q, K, V with the "it/animal" example from scratch
- [ ] Derive the 4-step self-attention recipe
- [ ] Explain why we divide by √d_k
- [ ] Explain what LayerNorm does and why not BatchNorm
- [ ] Explain residual connections and their gradient effect
- [ ] Walk through 2-head attention computation numerically
- [ ] Explain why masking is needed in the decoder
- [ ] Explain cross-attention: Q from where? K, V from where?
- [ ] Describe greedy vs beam search vs temperature sampling
- [ ] Explain what KV cache is and why it matters
- [ ] Explain BERT vs GPT architectural difference and when to use each
- [ ] Explain LoRA in 60 seconds

---

## Supporting Resources in This Folder

| File | What It Contains |
|------|-----------------|
| `Lecture 3 PPT (1).pdf` | Original slide deck (71 slides) |
| `Q&A on Transformer Architecture...pdf` | Q&A session transcript |
| `lecture_3_pre_reads_NG (1).docx` | Pre-read materials |
| `Lecture Post-read materials 18_04_2026.docx` | Post-read materials |
| `session-3_transcript.docx` | Full lecture transcript |
