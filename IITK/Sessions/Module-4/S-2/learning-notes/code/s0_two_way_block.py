"""A minimal SAM-style two-way attention block (shapes match SAM ViT-B/H: 64x64 image tokens, dim 256)."""
import torch, torch.nn as nn

class TwoWayBlock(nn.Module):
    def __init__(self, d=256, heads=8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.tok2img   = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp       = nn.Sequential(nn.Linear(d, 2048), nn.ReLU(), nn.Linear(2048, d))
        self.img2tok   = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(4))

    def forward(self, tokens, image):
        # 1) tokens talk to each other
        tokens = self.norms[0](tokens + self.self_attn(tokens, tokens, tokens)[0])
        # 2) token -> image: Q = tokens, K/V = image  ("what evidence do I need?")
        attn, w = self.tok2img(tokens, image, image)
        tokens = self.norms[1](tokens + attn)
        # 3) per-token MLP
        tokens = self.norms[2](tokens + self.mlp(tokens))
        # 4) image -> token: Q = image, K/V = tokens  ("re-encode the image given the prompt")
        image = self.norms[3](image + self.img2tok(image, tokens, tokens)[0])
        return tokens, image, w

torch.manual_seed(0)
image  = torch.randn(1, 64 * 64, 256)          # cached image embedding, flattened
output_tokens = torch.randn(1, 5, 256)          # 4 mask tokens + 1 IoU token (learned in real SAM)
prompt_tokens = torch.randn(1, 2, 256)          # e.g. one click + one padding point
tokens = torch.cat([output_tokens, prompt_tokens], dim=1)

blocks = [TwoWayBlock(), TwoWayBlock()]         # SAM stacks 2
for i, blk in enumerate(blocks, 1):
    tokens, image, w = blk(tokens, image)
    print(f"block {i}: tokens {tuple(tokens.shape)}  image {tuple(image.shape)}  token->image attn {tuple(w.shape)}")

# final step: mask = dot product of a mask token with every (upsampled) pixel feature
mask_token = tokens[:, 1]                                   # (1, 256)
logits = (image @ mask_token.unsqueeze(-1)).view(1, 64, 64) # SAM upsamples to 256x256 first
print("mask logits", tuple(logits.shape), "-> probabilities in [0,1] via sigmoid")
print("params per block:", sum(p.numel() for p in blocks[0].parameters()) / 1e6, "M")
