"""DocVLM's compression idea: 64 learnable queries cross-attend to hundreds of OCR(+layout) tokens."""
import torch, torch.nn as nn
torch.manual_seed(0)
d_ocr, d_llm, n_ocr, n_q = 768, 2048, 1200, 64

ocr_text = torch.randn(1, n_ocr, d_ocr)                        # OCR word embeddings
boxes = torch.rand(1, n_ocr, 4)                                 # (x1, y1, x2, y2) per word, normalised
layout = nn.Linear(4, d_ocr)(boxes)                             # 2-D position embedding
ocr_tokens = ocr_text + layout                                  # text + WHERE it is on the page

queries = nn.Parameter(torch.randn(1, n_q, d_ocr))              # the 64 learnable queries
xattn = nn.MultiheadAttention(d_ocr, 8, batch_first=True)
compressed, _ = xattn(queries, ocr_tokens, ocr_tokens)          # Q = queries, K/V = OCR tokens
to_llm = nn.Linear(d_ocr, d_llm)(compressed)                    # project to the (frozen) LLM's width

visual_tokens = torch.randn(1, 512, d_llm)                      # the VLM's normal image tokens
llm_input = torch.cat([visual_tokens, to_llm], dim=1)
print(f"OCR tokens {n_ocr} -> compressed {tuple(to_llm.shape)}; LLM input {tuple(llm_input.shape)}")
print(f"extra tokens added to the LLM: {n_q} instead of {n_ocr} ({n_ocr/n_q:.0f}x fewer)")
