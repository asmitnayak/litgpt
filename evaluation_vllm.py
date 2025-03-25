import json
import os
from argparse import ArgumentParser

from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from litgpt.prompts import Qwen2_5_DP, Qwen2_5, Qwen2_5_DP_FS


def load_test_dataset(batch_size=80, access_token=None):
    data = load_dataset("WIPI/dp-annotated2", token=access_token)
    #data = load_dataset("WIPI/dp_finetuning-balanced2", token=access_token)
    dl_data = []
    test_data = []
    for entry in tqdm(data['data']):
        dl_data.append({
            "instruction": Qwen2_5_DP_FS().apply(entry['input']),
            # too much RAM wastage
            # "output_df": pd.read_csv(StringIO(entry['output']), sep='|', header=None,names=['Deceptive Patterns Category', 'Deceptive Patterns Subtype', 'Reasoning'])
        })
        test_data.append({
            "input": entry['input'],
            "output": entry['output'],
        })
    dl = DataLoader(
        dl_data,
        batch_size=batch_size,
        shuffle=False,  # no shuffle might harm generate_responses_vllm_batch method
    )

    return test_data, dl

def load_model_tokenizer_sampling_params(checkpoint_dir, temperature=0.4, top_p=0.8, min_p=0.1, seed=1234, max_tokens=6200):
    model = LLM(model=checkpoint_dir, tensor_parallel_size=4, max_model_len=max_tokens)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, local_files_only=True)
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        min_p=min_p,
        top_k=15,
        max_tokens=max_tokens,
        repetition_penalty=1.05,
        seed=seed
    )
    return model, tokenizer, sampling_params

def generate_responses_vllm_batch(_model, _sampling_params, is_reason_first=False, access_token=None):
    test_data, data = load_test_dataset(access_token=os.getenv('HF_TOKEN') if not access_token else access_token)

    # colnames = ['Deceptive Patterns Category', 'Deceptive Patterns Subtype', 'Reasoning'] if is_reason_first else ['Reasoning', 'Deceptive Patterns Category', 'Deceptive Patterns Subtype']
    i = 0       # this is why shuffle=False in DataLoader

    for batch in tqdm(data):
        outputs = _model.generate(batch['instruction'], _sampling_params, use_tqdm=True)
        for _i, _outputs in enumerate(outputs):
            test_data[i]['response'] = _outputs.outputs[0].text # this is why shuffle=False in DataLoader
            i += 1

            # too much RAM wastage
            # test_data[i]["response_df"] = pd.read_csv(StringIO(_outputs.outputs[0].text), sep='|', header=None, names=colnames)
    return test_data

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--checkpoint_dir", type=str)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--min_p", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max_tokens", type=int, default=6200)
    parser.add_argument("--access_token", type=str, default=None)
    args = parser.parse_args()

    model, tokenizer, sampling_params = load_model_tokenizer_sampling_params(
        args.checkpoint_dir,
        args.temperature,
        args.top_p,
        args.min_p,
        args.seed,
        args.max_tokens
    )
    response_json = generate_responses_vllm_batch(model, sampling_params, access_token=args.access_token)
    os.makedirs('response_json2', exist_ok=True)
    filename = os.path.join('response_json2', f"{args.checkpoint_dir.rstrip('/').split('/')[-1]}.json")
    json.dump(response_json, open(filename, 'w'))


