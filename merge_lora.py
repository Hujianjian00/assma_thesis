from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Base model path (downloaded model)
base_model_path = "C:/Users/assma/.cache/huggingface/hub/models--deepseek-ai--deepseek-llm-7b-chat/snapshots/afbda8b347ec881666061fa67447046fc5164ec8"

# LoRA adapter path (your fine-tuned adapter)
lora_adapter_path = "D:/documents/python codes/Thesis/github-code/deepseek-enron-lora-10k/checkpoint-3750"


# Output merged model path
merged_output_path = "D:/documents/python codes/Thesis/github-code/deepseek-merged"

# Load and merge
print("🔄 Loading base model...")
model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, lora_adapter_path)

print("🧠 Merging LoRA weights...")
merged_model = model.merge_and_unload()

print("💾 Saving merged model to:", merged_output_path)
merged_model.save_pretrained(merged_output_path)
AutoTokenizer.from_pretrained(base_model_path).save_pretrained(merged_output_path)

print("✅ Merge complete.")
