import os
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from litgpt.prompts import Qwen2_5_DP


def load_model(checkpoint_dir, fp: str) -> tuple:
    """
    Load a model from a file path.
    Args:
        checkpoint_dir: checkpoint directory
        fp: fine-tuned model path

    Returns:
        model: the model
        tokenizer: the tokenizer
    """
    state_dict = torch.load(fp, weights_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        torch_dtype="auto",
        device_map="cuda:0",
        local_files_only=True
    )
    model.load_state_dict(state_dict)
    tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(fp), local_files_only=True)

    return model, tokenizer


def dataloader(dataset) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        generator=torch.Generator().manual_seed(1337),
        num_workers=4,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--is-reason-first", action="store_true")
    parser.add_argument("-c", "--checkpoint_dir", type=str)
    args = parser.parse_args()

    is_reason_first = args.is_reason_first
    f = args.checkpoint_dir
    finetune_path = os.path.join(
        "out/huggingface",
        f"lora-{f.split('/')[-1].replace('-Instruct', '').replace('-it', '').lower()}{"" if not is_reason_first else '-reason-first'}",
        "model.pth"
    )

    model, tokenizer = load_model(f, finetune_path)

    data = load_dataset("WIPI/dp_finetuning", token=os.getenv('HF_TOKEN'))
    test_data = []
    for entry in data['test']:
        test_data.append({"instruction": entry['input'], "output": entry['output']})



