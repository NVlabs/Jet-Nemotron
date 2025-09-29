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
import json
import torch
import argparse
import warnings
warnings.simplefilter('once', UserWarning)

from transformers import AutoTokenizer, AutoModelForCausalLM

from jetai.utils import (
    is_master,
    dist_init,
    dist_print,
    dist_close,
    build_config,
    get_device,
)
from jetai.evaluation.lm_eval_harness import LMEvalWrapper, LMHarnessEvaluator
from jetai.modeling import HFModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="jet-ai/Jet-Nemotron-2B", help="Path to the model configuration or checkpoint.")
    parser.add_argument("--eval_config", type=str, default="evaluation/configs/mmlu.yaml", help="Path to the evaluation configuration file.")
    parser.add_argument("--output_dir", type=str, default="results/eval/", help="Directory to save evaluation results.")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="Batch size for evaluation.")
    parser.add_argument("--max_length", type=int, default=None, help="Maximum sequence length for evaluation.")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="Maximum number of new tokens to generate during evaluation.")
    parser.add_argument("--prefill_chunk_size", type=int, default=None, help="Chunk size for prefill during evaluation.")
    parser.add_argument("--generation_num_chunks", type=int, default=1, help="Number of chunks for generation during evaluation.")
    parser.add_argument("--no_runtime_cache", action="store_true", help="Whether to disable runtime cache during evaluation.")
    parser.add_argument("--no_cache_requests", action="store_true", help="Whether to cache requests during evaluation.")
    parser.add_argument("--max_test_num", type=int, default=None, help="Maximum number of tests to run during evaluation.")
    
    # enable code eval
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    
    # parse args
    args, opt = parser.parse_known_args()

    # setup dist env
    dist_init(gpu=None, cudnn_benchmark=False)

    dist_print("Additional Arguments: ", json.dumps(opt, indent=2))

    eval_config = build_config(args.eval_config)
        
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, 
                                                 trust_remote_code=True,
                                                 attn_implementation="flash_attention_2",
                                                 torch_dtype=torch.bfloat16,
                                                 device_map="cuda")
    model = model.eval().cuda()
    model = HFModel(model)
    
    max_seq_len = args.max_length if args.max_length is not None else eval_config.get("max_length", None)
    assert max_seq_len is not None, "max_length must be specified either in the config or as an argument."
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else eval_config.get("max_new_tokens", 256)
    
    eval_wrapper = LMEvalWrapper(model, tokenizer, max_seq_len=max_seq_len,
                                    max_new_tokens=max_new_tokens,
                                    device=get_device(model),
                                    prefill_chunk_size=args.prefill_chunk_size,
                                    generation_num_chunks=args.generation_num_chunks,
                                    batch_size=args.eval_batch_size, 
                                    use_runtime_cache=not args.no_runtime_cache)
    
    evaluator = LMHarnessEvaluator(eval_wrapper, 
                                   eval_config["tasks"],
                                   subtasks=eval_config.get("subtasks", None),
                                   cache_requests=not args.no_cache_requests,
                                   max_test_num=args.max_test_num,
                                   apply_chat_template=eval_config.get("apply_chat_template", False),
                                   system_instruction=eval_config.get("system_instruction", None),
                                   keys_to_remove_in_save=eval_config.get("keys_to_remove_in_save", None))
    
    output_dir = os.path.join(args.output_dir, eval_config["name"])
    results = evaluator.evaluate(full_save_dir=os.path.join(output_dir, "full_results"))
    if is_master():
        print("Evaluation results:", json.dumps(results, indent=4))
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=4)
    
    dist_close()


if __name__ == "__main__":
    main()
