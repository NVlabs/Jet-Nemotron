# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from jetai.modeling import HFModel
from jetai.utils import set_benchmark_mode


def build_data(batch_size: int, prompt_len: int, max_iter: int, 
               vocab_size: int, pad_token_id: int = None, eos_token_id: int = None):
    # build a dummy dataset for evaluation
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, prompt_len, max_iter):
            self.prompt_len = prompt_len
            self.max_iter = max_iter
            self.data = [torch.randint(0, vocab_size, (prompt_len,)) for _ in range(max_iter * batch_size)]
            
            for i in range(vocab_size):
                if i != pad_token_id and i != eos_token_id:
                    replace_token = i
                    break

            # remove pad_token_id and eos_token_id in self.data
            if pad_token_id is not None:
                self.data = [torch.where(d == pad_token_id, replace_token, d) for d in self.data]
            if eos_token_id is not None:
                self.data = [torch.where(d == eos_token_id, replace_token, d) for d in self.data]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return {"input_ids": self.data[idx]}

    dataset = DummyDataset(prompt_len, max_iter)
    eval_data_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=True, 
        drop_last=False
    )

    return eval_data_loader


def build_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                trust_remote_code=True, 
                                                attn_implementation="flash_attention_2",
                                                torch_dtype=torch.bfloat16,
                                                device_map="cuda")
    model = model.eval()
    model = HFModel(model)
    return model


def print_and_save(log: str, output_dir: str):
    print(log)
    with open(os.path.join(output_dir, "log.txt"), "a") as f:
        f.write(log + "\n")


def run_generate(
    eval_data_loader: torch.utils.data.DataLoader,
    model: nn.Module, 
    tokenizer: PreTrainedTokenizer, 
    max_new_tokens: int, 
    prompt_len: int, 
    output_dir: str, 
    max_iter: int, 
    exclude_iters: int,
    prefill_chunk_size: int = None
):    
    num_seq = 0
    s = 0
    all_prefill_throughputs = []
    all_decode_throughputs = []
    
    pbar = tqdm(total=max_iter, desc="Evaluating")
    while True:
        for batch in eval_data_loader:
            input_ids = batch["input_ids"].cuda()
            attention_mask = None

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True), torch.no_grad():
                out, prefill_time, decode_time = model.generate(
                    input_ids, 
                    attention_mask, 
                    max_new_tokens=max_new_tokens,
                    stop_token_list=[],
                    pad_token_id=tokenizer.pad_token_id,
                    return_dict=True,
                    prefill_chunk_size=prefill_chunk_size,)
            
            sequences = out["sequences"]
            
            assert sequences[0].size(-1) == (max_new_tokens + prompt_len), f"Generated sequence length {sequences[0].size(-1)} does not match expected length {max_new_tokens + prompt_len}"
            
            num_seq += input_ids.size(0)
            
            prefill_tokens = input_ids.size(0) * prompt_len    
            decode_tokens = input_ids.size(0) * max_new_tokens
            
            prefill_throughput = prefill_tokens / (prefill_time / 1000)
            decode_throughput = decode_tokens / (decode_time / 1000)
            
            all_prefill_throughputs.append(prefill_throughput)
            all_decode_throughputs.append(decode_throughput)
            
            print_and_save(f"Prefill Throughput: {prefill_throughput:.2f} tokens/s", output_dir)
            print_and_save(f"Decode Throughput: {decode_throughput:.2f} tokens/s", output_dir)
            torch.cuda.empty_cache()

            pbar.update(1)
            s += 1
            if s >= max_iter:
                break
        if s >= max_iter:
            break
    
    avg_prefill_throughput = np.mean(all_prefill_throughputs[exclude_iters:])
    avg_decode_throughput = np.mean(all_decode_throughputs[exclude_iters:])
    print_and_save(f"Prefill Throughputs : {[round(x, 2) for x in all_prefill_throughputs]}", output_dir)
    print_and_save(f"Decode Throughputs : {[round(x, 2) for x in all_decode_throughputs]}", output_dir)
    print_and_save(f"Average prefill throughput (exluding first {exclude_iters} iters): {avg_prefill_throughput:.2f} tokens/s", output_dir)
    print_and_save(f"Average decode throughput (exluding first {exclude_iters} iters): {avg_decode_throughput:.2f} tokens/s", output_dir)

 
def main():
    parser = argparse.ArgumentParser(description="Measure throughput of a language model.")
    parser.add_argument("--model_name_or_path", type=str, default="jet-ai/Jet-Nemotron-2B", help="Path to the model configuration or checkpoint.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for evaluation.")
    parser.add_argument("--prompt_len", type=int, default=65536, help="Length of the prompt for evaluation.")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum number of new tokens to generate during evaluation.")
    parser.add_argument("--exclude_iters", type=int, default=3, help="Number of initial iterations to exclude from throughput calculation.")
    parser.add_argument("--max_iter", type=int, default=10, help="Maximum number of iterations to run for evaluation.")
    parser.add_argument("--prefill_chunk_size", type=int, default=2048, help="Chunk size for prefill during evaluation.")
    parser.add_argument("--output_dir", type=str, default="results/throughput", help="Directory to save evaluation results.")
    
    args = parser.parse_args()
    
    set_benchmark_mode()
        
    os.makedirs(args.output_dir, exist_ok=True)
    print_and_save("\n\n" + "=" * 20, args.output_dir)

    model = build_model(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    num_params = sum(p.numel() for p in model.parameters())
    print_and_save(f"Number of parameters: {num_params}", args.output_dir)
    
    print_and_save(f"Model: {args.model_name_or_path} | batch size: {args.batch_size} | prompt length: {args.prompt_len} | max new tokens: {args.max_new_tokens} | prefill chunk: {args.prefill_chunk_size}", args.output_dir)
    eval_data_loader = build_data(
        args.batch_size, 
        args.prompt_len, 
        args.max_iter, 
        model.vocab_size, 
        pad_token_id=tokenizer.pad_token_id, 
        eos_token_id=tokenizer.eos_token_id
    )

    run_generate(
        eval_data_loader, 
        model, 
        tokenizer, 
        args.max_new_tokens, 
        args.prompt_len, 
        args.output_dir, 
        args.max_iter, 
        args.exclude_iters,
        args.prefill_chunk_size
    )
        

if __name__ == "__main__":
    main()