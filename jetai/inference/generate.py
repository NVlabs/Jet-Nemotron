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

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser(description="Generate text using Jet-Nemotron")
parser.add_argument("--model_name_or_path", type=str, default="jet-ai/Jet-Nemotron-2B", help="Path to the model directory")
args = parser.parse_args()

model_name_or_path = args.model_name_or_path

# For local testing, you can use the following path.
# NOTE: Be sure to download or soft link the model weights to `jetai/modeling/hf`
# model_name_or_path = "jetai/modeling/hf/"

model = AutoModelForCausalLM.from_pretrained(model_name_or_path, 
                                             trust_remote_code=True, 
                                             attn_implementation="flash_attention_2",
                                             torch_dtype=torch.bfloat16,
                                             device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = model.eval().cuda()

input_str = "Hello, I'm Jet-Nemotron from NVIDIA."

input_ids = tokenizer(input_str, return_tensors="pt").input_ids.cuda()
output = model.generate(input_ids, max_new_tokens=50, do_sample=False)
output_str = tokenizer.decode(output[0], skip_special_tokens=True)
print(output_str)