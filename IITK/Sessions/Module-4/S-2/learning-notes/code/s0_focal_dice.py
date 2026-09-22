import torch, torch.nn.functional as F

def focal_loss(logits, target, gamma=2.0, alpha=0.25):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = p * target + (1 - p) * (1 - target)          # prob of the CORRECT label
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    return (alpha_t * (1 - p_t) ** gamma * ce).mean()

def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits).flatten(1)
    t = target.flatten(1)
    inter = (p * t).sum(-1)
    return (1 - (2 * inter + eps) / (p.sum(-1) + t.sum(-1) + eps)).mean()

H = W = 256
target = torch.zeros(1, 1, H, W); target[..., 100:110, 100:110] = 1   # tiny 10x10 object (0.15% of pixels)

lazy = torch.full((1, 1, H, W), -4.0)          # "predict background everywhere"
good = lazy.clone(); good[..., 100:110, 100:110] = 4.0   # finds the object

sharp = torch.full((1, 1, H, W), -10.0); sharp[..., 100:110, 100:110] = 10.0   # confident + correct

for name, pred in [("all-background", lazy), ("finds, but soft", good), ("finds, confident", sharp)]:
    bce = F.binary_cross_entropy_with_logits(pred, target).item()
    print(f"{name:17s} BCE={bce:.4f}  Focal={focal_loss(pred, target).item():.6f}  "
          f"Dice={dice_loss(pred, target).item():.4f}  SAM(20F+D)={20*focal_loss(pred,target).item()+dice_loss(pred,target).item():.4f}")
