"""Mini Grounded-SAM: a text-prompted detector finds WHAT/WHERE, SAM draws the exact pixels."""
from pathlib import Path
import torch
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection, SamModel, SamProcessor

ASSETS = Path(__file__).resolve().parent.parent / "assets"
image = Image.open(ASSETS / "halluc-dog-beach-image.png").convert("RGB")

# 1) Open-vocabulary detector: text -> boxes
det_proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
det = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").eval()
queries = ["a photo of a dog", "a photo of a bird"]
inp = det_proc(text=[queries], images=image, return_tensors="pt")
with torch.no_grad():
    out = det(**inp)
W, H = image.size
side = max(W, H)                                   # OWLv2 pads to a square
res = det_proc.image_processor.post_process_object_detection(out, threshold=0.2, target_sizes=torch.tensor([[side, side]]))[0]
best = {}
for score, label, box in zip(res["scores"], res["labels"], res["boxes"]):
    q = queries[label]
    if q not in best or score > best[q][0]:
        best[q] = (score.item(), [round(v, 1) for v in box.tolist()])
for q, (s, b) in best.items():
    print(f"detector: {q!r:22s} score={s:.2f} box={b}")

# 2) SAM: box prompt -> pixel-precise mask
sam_proc = SamProcessor.from_pretrained("facebook/sam-vit-base")
sam = SamModel.from_pretrained("facebook/sam-vit-base").eval()
boxes = [[b for (_, b) in best.values()]]
p = sam_proc(image, input_boxes=boxes, return_tensors="pt")
with torch.no_grad():
    o = sam(**p, multimask_output=False)                # one mask per box: no ambiguity
masks = sam_proc.image_processor.post_process_masks(o.pred_masks, p["original_sizes"], p["reshaped_input_sizes"])[0]
for (q, (_, b)), m, iou in zip(best.items(), masks, o.iou_scores[0]):
    x1, y1, x2, y2 = b
    box_area = (x2 - x1) * (y2 - y1)
    print(f"SAM mask for {q!r:22s} pixels={int(m.sum()):5d}  (box area {box_area:6.0f})  predicted IoU={iou.item():.2f}")
