from glob import glob
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

files = glob("./checkpoints/*/*")
for f in files:
	lora_path = f'./out/huggingface/lora-{os.path.basename(f).replace("-Instruct", "").replace("-it", "").lower()}'
	lora_path = lora_path.replace("-3.2", "3.2")
	lora_path = lora_path.replace("-2-", "2-")
	lora_path2 = f'./out/huggingface/lora-{os.path.basename(f).replace("-Instruct", "").replace("-it", "").lower()}-reason-first'
	lora_path2 = lora_path2.replace("-3.2", "3.2")
	lora_path2 = lora_path2.replace("-2-", "2-")
	full_path = f'./out/huggingface/full-{os.path.basename(f).replace("-Instruct", "").replace("-it", "").lower()}'
	full_path = full_path.replace("-3.2", "3.2")
	full_path = full_path.replace("-2-", "2-")
	for path in [lora_path, lora_path2, full_path]:
		if not os.path.exists(path):
			continue
		state_dict = torch.load(path+"/model.pth", weights_only=True)
		model = AutoModelForCausalLM.from_pretrained(
				f,
				torch_dtype="auto",
				device_map="cuda:0",
				local_files_only=True
			)
		model.load_state_dict(state_dict)
		tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
		model.save_pretrained(path.replace("huggingface", "safetensors"))
		tokenizer.save_pretrained(path.replace("huggingface", "safetensors"))
		del model, tokenizer, state_dict
