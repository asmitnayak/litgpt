import os
import sys
import time
from pathlib import Path
from pprint import pprint
from typing import Iterator, List, Literal, Optional, Tuple, Callable, Any

import lightning as L
import torch
from lightning.fabric.plugins import BitsandbytesPrecision

from litgpt.model import GPT
from litgpt.config import Config
from litgpt.prompts import PromptStyle
from litgpt.tokenizer import Tokenizer
from litgpt.prompts import has_prompt_style, load_prompt_style
from litgpt.scripts.merge_lora import merge_lora
from litgpt.utils import (
    auto_download_checkpoint,
    check_file_size_on_cpu_and_warn,
    extend_checkpoint_dir,
    get_default_supported_precision,
    load_checkpoint,
)

from tqdm.auto import tqdm

class DPChat:
    def __init__(self,
        checkpoint_dir: Path,
        *,
        precision: Optional[str] = None,
        compile: bool = False,
        access_token: Optional[str] = None,
    ):
        """
        Chat with a model.

        Args:
            checkpoint_dir: A local path to a directory containing the model weights or a valid model name.
                You can get a list of valid model names via the `litgpt download list` command line argument.
            quantize: Whether to quantize the model and using which method:
                - bnb.nf4, bnb.nf4-dq, bnb.fp4, bnb.fp4-dq: 4-bit quantization from bitsandbytes
                - bnb.int8: 8-bit quantization from bitsandbytes
                for more details, see https://github.com/Lightning-AI/litgpt/blob/main/tutorials/quantize.md
            precision: Indicates the Fabric precision setting to use.
            compile: Whether to use compilation to speed up token generation. Will increase startup time.
            access_token: Optional API token to access models with restrictions.
        """
        self.checkpoint_dir = extend_checkpoint_dir(checkpoint_dir)
        pprint(locals())

        self.precision = precision or get_default_supported_precision(training=False)

        self.fabric = L.Fabric(devices=1, precision=precision, plugins=None)

        # Merge if this is a raw LoRA checkpoint
        self.checkpoint_path = self.checkpoint_dir / "lit_model.pth"
        if (self.checkpoint_dir / "lit_model.pth.lora").is_file() and not self.checkpoint_path.is_file():
            print("Merging LoRA weights with the base model. This won't take long and is a one-time-only thing.")
            merge_lora(self.checkpoint_dir)

        if not self.checkpoint_path.is_file():
            self.checkpoint_dir = auto_download_checkpoint(model_name=self.checkpoint_dir, access_token=self.access_token)
            self.checkpoint_path = self.checkpoint_dir / "lit_model.pth"

        check_file_size_on_cpu_and_warn(self.checkpoint_path, self.fabric.device)
        self.config = Config.from_file(checkpoint_dir / "model_config.yaml")

        with self.fabric.init_module(empty_init=True):
            self.model = GPT(self.config)
            if compile:
                print(
                    "IMPORTANT: with enabled compilation the KV-cache size is determined by model's maximum context size, which leads to "
                    "a higher memory consumption. In case of an OOM error, try to set `--compile=False`."
                )
                self.model.set_kv_cache(batch_size=1)
        load_checkpoint(self.fabric, self.model, self.checkpoint_path)
        self.model.eval()

        if compile:
            torch._dynamo.config.automatic_dynamic_shapes = True
            torch._inductor.config.triton.unique_kernel_names = True
            torch._inductor.config.coordinate_descent_tuning = True
            global next_token
            next_token = torch.compile(next_token, mode="reduce-overhead", dynamic=True)

        self.model = self.fabric.setup_module(self.model)

        self.tokenizer = Tokenizer(self.checkpoint_dir)
        self.prompt_style = (
            load_prompt_style(self.checkpoint_dir) if has_prompt_style(self.checkpoint_dir) else PromptStyle.from_config(self.config)
        )
        self.stop_tokens = self.prompt_style.stop_tokens(self.tokenizer)

        print(f"Now chatting with {self.config.name}.\n")
        L.seed_everything(1234)

    def process_prompt(self, prompt, temperature, max_new_tokens, top_k, top_p, stream=False):
        prompt = self.prompt_style.apply(prompt=prompt)
        encoded_prompt = self.tokenizer.encode(prompt, device=self.fabric.device)

        if max_new_tokens is None:
            max_returned_tokens = self.model.max_seq_length
        else:
            first_turn = self.model.mask_cache is None
            max_returned_tokens = encoded_prompt.size(0) + max_new_tokens
            if first_turn or max_returned_tokens > self.model.max_seq_length:
                self.model.max_seq_length = max_returned_tokens
                self.model.set_kv_cache(batch_size=1, device=self.fabric.device)

        y: Iterator[torch.Tensor] = self.generate(
            self.model, encoded_prompt, max_returned_tokens,
            temperature=temperature, top_k=top_k, top_p=top_p, stop_tokens=self.stop_tokens
        )
        token_generator: Iterator[str] = self.tokenizer.decode_stream(y, device=self.fabric.device)
        t0 = time.perf_counter()

        tokens_generated = 0
        generated_content = ""
        for tok in token_generator:
            tokens_generated += 1
            if stream:
                self.fabric.print(tok, end="", flush=True)
            generated_content += tok

        t = time.perf_counter() - t0

        for block in self.model.transformer.h:
            block.attn.kv_cache.reset_parameters()

        return generated_content

    def interact(self,
                 prompt: str, temperature: float, max_new_tokens: int,
                 top_k: int = 50, top_p: float = 1.0,
                 stream=False):
        """
        Args:
            prompt: The prompt to start the conversation with.
            max_new_tokens: The number of generation steps to take.
            top_k: The number of top most probable tokens to consider in the sampling process.
            top_p: If specified, it represents the cumulative probability threshold to consider in the sampling process.
                In top-p sampling, the next token is sampled from the highest probability tokens
                whose cumulative probability exceeds the threshold `top_p`. When specified,
                it must be `0 <= top_p <= 1`. Here, `top_p=0` is equivalent
                to sampling the most probable token, while `top_p=1` samples from the whole distribution.
                It can be used in conjunction with `top_k` and `temperature` with the following order
                of application:

                1. `top_k` sampling
                2. `temperature` scaling
                3. `top_p` sampling

                For more details, see https://arxiv.org/abs/1904.09751
                or https://huyenchip.com/2024/01/16/sampling.html#top_p
            temperature: A value controlling the randomness of the sampling process. Higher values result in more random
                samples.
        """
        return self.process_prompt(
                prompt,
                temperature,
                max_new_tokens,
                top_k,
                top_p,
                stream
            )

    @torch.inference_mode()
    def generate( self,
            model: GPT,
            prompt: torch.Tensor,
            max_returned_tokens: int,
            *,
            temperature: float = 1.0,
            top_k: Optional[int] = None,
            top_p: float = 1.0,
            stop_tokens: Tuple[List[int], ...] = (),
    ) -> Iterator[torch.Tensor]:
        """Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as possible.

        Arguments:
            model: The model to use.
            prompt: Tensor of shape (T) with indices of the prompt sequence.
            max_returned_tokens: The maximum number of tokens to return (given plus generated).
            temperature: Scales the predicted logits by 1 / temperature
            top_k: If specified, only sample among the tokens with the k highest probabilities.
            top_p: If specified, it represents the cumulative probability threshold to consider in the sampling process.
                In top-p sampling, the next token is sampled from the highest probability tokens
                whose cumulative probability exceeds the threshold `top_p`. When specified,
                it must be `0 <= top_p <= 1`. Here, `top_p=0` is equivalent
                to sampling the most probable token, while `top_p=1` samples from the whole distribution.
                It can be used in conjunction with `top_k` and `temperature` with the following order
                of application:

                1. `top_k` sampling
                2. `temperature` scaling
                3. `top_p` sampling

                For more details, see https://arxiv.org/abs/1904.09751
                or https://huyenchip.com/2024/01/16/sampling.html#top_p
            stop_tokens: If specified, stop generating any more token once one of this list is generated.
        """
        from litgpt.generate.base import generate_fn
        return generate_fn(
            include_prompt=False,
            include_eos=False,
            model=model,
            prompt=prompt,
            max_returned_tokens=max_returned_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_tokens=stop_tokens
        )