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

from typing import Optional

import torch
import torch.nn as nn
from transformers import (
    PreTrainedModel,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    PreTrainedTokenizerBase,
)

from jetai.utils import is_benchmark_mode, dist_print


class TokensStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_token_list: list[int]):
        super().__init__()
        self.stop_token_list = stop_token_list

    def __call__(self, input_ids: torch.Tensor, scores: Optional[torch.Tensor] = None) -> bool:
        finished_sequences = torch.zeros(input_ids.shape[0], dtype=torch.int32, device=input_ids.device)
        for stop_tokens in self.stop_token_list:
            for b in range(input_ids.shape[0]):
                if input_ids[b, -len(stop_tokens) :].tolist() == stop_tokens:
                    finished_sequences[b] = 1
        return finished_sequences


class HFModel(nn.Module):
    '''
    A wrapper for a Hugging Face model to provide a consistent interface for generation.
    Enable chunk-prefilling, chunk generation (contiuous batching), stopping criteria, and benchmarking.
    '''
    
    def __init__(self, model: PreTrainedModel):
        super().__init__()
        self.model = model
        self.vocab_size = getattr(self.model.config, "vocab_size", None)
        dist_print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = 1.0,
        do_sample: bool = False,
        stop_token_list: Optional[list[int]] = None,
        pad_token_id: int = None,
        return_dict: bool = False,
        return_unfinished: bool = False,
        return_kv_cache: bool = False,
        prefill_chunk_size: Optional[int] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        
        stopping_criteria_list = StoppingCriteriaList()
        if stop_token_list is not None:
            stopping_criteria = TokensStoppingCriteria(stop_token_list)
            stopping_criteria_list.append(stopping_criteria)
        
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            pad_token_id=pad_token_id,
            prefill_chunk_size=prefill_chunk_size,
            disable_compile=True
        )
        if do_sample:
            generation_config.temperature = temperature
        
        hf_out = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            use_model_defaults=False,
            return_dict_in_generate=True,
            stopping_criteria=stopping_criteria_list,
            tokenizer=tokenizer,
        )
        if is_benchmark_mode():
            hf_out, prefill_time, decode_time = hf_out

        unfinished = (hf_out.sequences[:, -1] != pad_token_id) & ~stopping_criteria_list(hf_out.sequences, None)

        if return_dict:
            out = {"sequences": hf_out.sequences}
            if return_unfinished:
                out["unfinished_sequences"] = unfinished
            if return_kv_cache:
                out["kv_cache"] = hf_out.past_key_values
            if is_benchmark_mode():
                out = (out, prefill_time, decode_time)
            return out

        if is_benchmark_mode():
            return hf_out.sequences, prefill_time, decode_time
        else:
            return hf_out.sequences
