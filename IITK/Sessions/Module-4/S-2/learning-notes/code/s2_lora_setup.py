"""What 'fine-tune a captioner with LoRA' means in code: freeze the VLM, train tiny adapters."""
import torch
from transformers import AutoModelForImageTextToText
from peft import LoraConfig, get_peft_model

model = AutoModelForImageTextToText.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct", dtype=torch.float32)
cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
                 target_modules=["q_proj", "v_proj"],        # the classic choice (no duplicates!)
                 task_type="CAUSAL_LM")
model = get_peft_model(model, cfg)
model.print_trainable_parameters()
