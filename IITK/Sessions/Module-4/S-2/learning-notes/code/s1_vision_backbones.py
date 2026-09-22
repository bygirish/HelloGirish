"""Every model in this lecture sees through a ViT. Read it straight from the configs (downloads JSON only)."""
from transformers import AutoConfig

def show(name, vc, size_key="image_size"):
    size = getattr(vc, size_key, None)
    patch = vc.patch_size
    extra = f"-> {(size // patch) ** 2} patch tokens" if isinstance(size, int) else ""
    print(f"{name:32s} patch={patch:>2}  depth={getattr(vc, 'num_hidden_layers', getattr(vc, 'depth', '?')):>2}  "
          f"width={getattr(vc, 'hidden_size', getattr(vc, 'embed_dim', '?'))}  image={size} {extra}")

show("openai/clip-vit-base-patch32", AutoConfig.from_pretrained("openai/clip-vit-base-patch32").vision_config)
show("llava-hf/llava-1.5-7b-hf (CLIP-L)", AutoConfig.from_pretrained("llava-hf/llava-1.5-7b-hf").vision_config)
show("facebook/sam-vit-huge", AutoConfig.from_pretrained("facebook/sam-vit-huge").vision_config)
q = AutoConfig.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct").vision_config
print(f"{'Qwen/Qwen2.5-VL-3B-Instruct':32s} patch={q.patch_size:>2}  depth={q.depth}  width={q.hidden_size}  "
      f"image=dynamic (spatial_merge={q.spatial_merge_size} -> 1 token per {q.patch_size*q.spatial_merge_size}px square)")
