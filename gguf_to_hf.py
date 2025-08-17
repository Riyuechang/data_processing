import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/media/ifw/GameFile/linux_cache/LLMModel/Sakura-1.5B-Qwen2.5-v1.0-GGUF"
FILENAME = "sakura-1.5b-qwen2.5-v1.0-fp16.gguf"

OUTPUT_PATH = "/media/ifw/GameFile/linux_cache/LLMModel/Sakura-1.5B-Qwen2.5-v1.0-HF"


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, gguf_file=FILENAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, gguf_file=FILENAME, torch_dtype=torch.float16)

tokenizer.save_pretrained(OUTPUT_PATH)
model.save_pretrained(OUTPUT_PATH)