import math, torch, torch.nn as nn, torch.nn.functional as F

print("--- 0. Patchify: an image becomes a sequence of tokens ---")
img = torch.rand(1, 3, 1024, 1024)                         # SAM's input size
patches = img.unfold(2, 16, 16).unfold(3, 16, 16)          # (1,3,64,64,16,16)
tokens = patches.permute(0, 2, 3, 1, 4, 5).reshape(1, 64 * 64, 3 * 16 * 16)
print("image", tuple(img.shape), "-> tokens", tuple(tokens.shape), "(4096 tokens x 768 raw values)")

print("\n--- 0b. MAE masking: hide 75% of the tokens ---")
torch.manual_seed(0)
n = tokens.shape[1]
keep = torch.randperm(n)[: n // 4]                          # encoder sees only 25%
print(f"visible tokens: {len(keep)} of {n}  -> encoder cost ~{(len(keep)/n)**2:.3f}x of full attention")

print("\n--- 1. Why the ViT-H image encoder is heavy ---")
for side in (224, 1024):
    n = (side // 16) ** 2
    print(f"{side}px / 16px patches -> {n:5d} tokens -> {n*n:>11,d} attention pairs per head per layer")

print("\n--- 2. MAE target: pixels, not a vocabulary ---")
patch = torch.rand(16, 16, 3)                   # one 16x16 RGB patch
pred = patch + 0.1 * torch.randn_like(patch)    # imperfect reconstruction
print("values to predict per patch:", patch.numel(), "| MSE:", round(F.mse_loss(pred, patch).item(), 4))

print("\n--- 3. Fourier positional encoding of a click (why not raw coords) ---")
torch.manual_seed(0)
B = torch.randn(2, 8) * 10                      # random frequencies (SAM uses 128 -> 256-d)
def fourier(xy):                                 # xy normalised to [0,1]
    proj = 2 * math.pi * xy @ B
    return torch.cat([proj.sin(), proj.cos()], -1)
a, b = torch.tensor([[0.500, 0.30]]), torch.tensor([[0.501, 0.30]])
far = torch.tensor([[0.9, 0.8]])
print("raw coords distance a-b:", round((a-b).norm().item(), 4))
print("fourier cos-sim a-b    :", round(F.cosine_similarity(fourier(a), fourier(b)).item(), 4))
print("fourier cos-sim a-far  :", round(F.cosine_similarity(fourier(a), fourier(far)).item(), 4))

print("\n--- 4. Dot product of output token with pixel features -> mask ---")
w = torch.tensor([1., -1.])
feats = {"pixel A": torch.tensor([2., 0.]), "pixel B": torch.tensor([0., 2.])}
for k, f in feats.items():
    print(f"{k}: w.f = {w@f:+.0f} -> sigmoid = {torch.sigmoid(w@f):.2f}")

print("\n--- 5. Convolution shrinks, transposed convolution grows ---")
img = torch.arange(1., 10.).reshape(1, 1, 3, 3)
kern = torch.tensor([[[[0., 1.], [1., 0.]]]])
down = F.conv2d(img, kern)                       # 3x3 -> 2x2
print("conv2d 3x3 -> 2x2:\n", down[0, 0])
up = F.conv_transpose2d(down, kern)              # 2x2 -> 3x3
print("conv_transpose2d 2x2 -> 3x3:\n", up[0, 0])
up4 = nn.ConvTranspose2d(256, 32, kernel_size=2, stride=2)(torch.randn(1, 256, 64, 64))
print("SAM-style stride-2 upsample: (1,256,64,64) ->", tuple(up4.shape))

print("\n--- 6. Focal loss per pixel ---")
for pt in (0.95, 0.30):
    ce = -math.log(pt); fl = (1 - pt) ** 2 * ce
    print(f"p_t={pt:.2f}: CE={ce:.4f}  focal={fl:.6f}  shrink={ce/fl:.0f}x")

print("\n--- 7. Dice: same fraction found -> same loss, any size ---")
def dice_loss_sets(y, x, inter): return 1 - 2 * inter / (x + y)
for name, y, x, i in [("large", 10, 5, 5), ("small", 4, 2, 2), ("tiny missed", 1, 0, 0)]:
    print(f"{name:12s} |Y|={y:2d} |X|={x:2d} |X∩Y|={i:2d} -> Dice loss {dice_loss_sets(y, x, i):.3f}")

print("\n--- 8. Amortisation: 10 clicks ---")
enc, dec = 0.15, 0.05
print(f"naive: {10*(enc+dec):.2f}s   SAM: {enc + 10*dec:.2f}s")
