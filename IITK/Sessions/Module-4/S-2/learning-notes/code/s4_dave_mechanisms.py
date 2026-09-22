"""DAVE's two key tricks in miniature: (1) raw vs normalised MAE targets on blank patches, (2) learned weight merging."""
import torch, torch.nn as nn
torch.manual_seed(0)

print("--- 1. MAE target on a near-blank document patch ---")
patch = torch.tensor([0.98, 0.99, 0.98, 0.99])                  # white paper + scanner noise
norm_target = (patch - patch.mean()) / (patch.std(unbiased=False) + 1e-6)
print("raw target       :", patch.tolist())
print("normalised target:", [round(v, 2) for v in norm_target.tolist()], " <- noise blown up to +/-1")

print("\n--- 2. Weight-space merging of encoder copies ---")
def make_encoder(): return nn.Linear(8, 8)
copies = [make_encoder() for _ in range(3)]                     # e.g. trained next to Qwen / Llama / Phi decoders
alpha_logits = nn.Parameter(torch.zeros(3))                     # learned, not fixed at 1/3

def merged_weight():
    a = torch.softmax(alpha_logits, 0)                          # keep coefficients positive, summing to 1
    return sum(ai * c.weight for ai, c in zip(a, copies)), a

W, a = merged_weight()
print("initial alphas (= plain averaging):", [round(v, 3) for v in a.tolist()])
# toy objective: pretend copy #0 transfers best, so fit alphas to reproduce its features on some data
x = torch.randn(64, 8)
target = (x @ copies[0].weight.T).detach()
opt = torch.optim.SGD([alpha_logits], lr=5.0)
for _ in range(50):
    W, a = merged_weight()
    loss = ((x @ W.T) - target).pow(2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
print("learned alphas                   :", [round(v, 3) for v in torch.softmax(alpha_logits, 0).tolist()])
print("merged weight shape:", tuple(merged_weight()[0].shape), "-> ONE encoder, usable with any decoder")
