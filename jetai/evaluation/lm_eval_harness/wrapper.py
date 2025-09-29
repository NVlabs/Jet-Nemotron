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

import json
import pandas as pd

from tqdm import tqdm
from typing import Any, Optional
from copy import deepcopy
from collections import defaultdict

import torch
import torch.nn.functional as F

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.models.utils import handle_stop_sequences
from lm_eval.utils import get_rolling_token_windows, make_disjoint_window

from transformers import PreTrainedTokenizer

from jetai.modeling import HFModel
from jetai.utils import (
    chunk_list_by_size, 
    get_device, 
    is_master,
    get_dist_size,
    get_dist_rank,
)


pd.set_option("future.no_silent_downcasting", True)


class LMEvalWrapper(LM):
    def __init__(
        self,
        model: HFModel,
        tokenizer: PreTrainedTokenizer,
        max_seq_len: int,
        batch_size: int = 1,
        max_new_tokens: Optional[int] = None,
        device: torch.device = "cuda",
        use_runtime_cache: bool = False,
        prefill_chunk_size: Optional[int] = None, # chunk_prefill
        generation_num_chunks: int = 1, # chunk_generation (like continuous batching)
        add_prefix: bool = False,
        disable_tqdm: bool = False,
    ) -> None:
        super().__init__()

        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_new_tokens = max_new_tokens
        self.generation_num_chunks = generation_num_chunks
        self.batch_size = batch_size
        self.prefill_chunk_size = prefill_chunk_size

        self._world_size = get_dist_size()
        self._rank = get_dist_rank()
        self.device = device

        self.add_prefix = add_prefix

        self.disable_tqdm = disable_tqdm

        self.batch_size = batch_size
        self.use_runtime_cache = use_runtime_cache
        self.loglikelihood_cache = None
        self.generation_cache = None

        if use_runtime_cache and is_master():
            print("Using runtime cache for LM evaluation")

    def tok_encode(self, string: str, left_truncate_len: Optional[int] = None, 
                   add_special_tokens: Optional[bool] = None, add_prefix: Optional[bool] = None) -> list[int]:
        add_prefix = add_prefix if add_prefix is not None else self.add_prefix
        # default for None - empty dict, use predefined tokenizer param
        # used for all models except for CausalLM or predefined value
        special_tokens_kwargs = {}

        # by default for CausalLM - false or self.add_prefix is set
        if add_special_tokens is None:
            special_tokens_kwargs = {"add_special_tokens": False}
        else:
            special_tokens_kwargs = {"add_special_tokens": add_special_tokens}

        encoding = self.tokenizer.encode(string, **special_tokens_kwargs)

        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            if add_prefix:
                encoding = encoding[-left_truncate_len+1:]
            else:
                encoding = encoding[-left_truncate_len:]
        
        if add_prefix:
            encoding = [self.tokenizer.bos_token_id] + encoding

        if left_truncate_len:
            assert len(encoding) <= left_truncate_len, (len(encoding), left_truncate_len)
        return encoding

    def tok_batch_encode(
        self,
        strings: list[str],
        padding_side: str = "left",
        left_truncate_len: Optional[int] = None,
        truncation: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # encode a batch of strings. converts to tensors and pads automatically, unlike tok_encode.

        encoding = self.tokenizer(
            strings,
            truncation=truncation,
            padding="longest",
            return_tensors="pt",
            padding_side=padding_side,
            add_special_tokens=False
        )
        if left_truncate_len:
            if self.add_prefix:
                left_truncate_len += 1
            encoding["input_ids"] = encoding["input_ids"][:, -left_truncate_len:]
            encoding["attention_mask"] = encoding["attention_mask"][:, -left_truncate_len:]

        if self.add_prefix:
            encoding["input_ids"] = torch.cat(
                [torch.full((encoding["input_ids"].shape[0], 1), self.tokenizer.bos_token_id, dtype=torch.long), encoding["input_ids"]],
                dim=1,
            )
            encoding["attention_mask"] = torch.cat(
                [torch.ones((encoding["attention_mask"].shape[0], 1), dtype=torch.long), encoding["attention_mask"]],
                dim=1,
            )
        
        return encoding["input_ids"], encoding["attention_mask"]

    def tok_decode(self, tokens: list[int], skip_special_tokens: bool = True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def _loglikelihood_tokens(
        self, requests: list[tuple[tuple[str, str], list[int], list[int]]]
    ) -> list[tuple[float, bool]]:
        total_requests_num = len(requests)
        sorted_indices, requests = zip(*sorted(enumerate(requests), key=lambda x: len(x[1][1] + x[1][2]), reverse=True))

        batched_requests = chunk_list_by_size(requests, self.batch_size)

        device = get_device(self.model)
        response1 = []
        response2 = []
        for batch in tqdm(
            batched_requests,
            total=len(batched_requests),
            disable=self.disable_tqdm or (self.rank != 0),
            desc=f"Running requests",
        ):
            _, batched_context_enc, batched_continuation_enc = zip(*batch)
            batched_all_enc = [
                context_enc + continuation_enc[:-1]
                for context_enc, continuation_enc in zip(batched_context_enc, batched_continuation_enc)
            ]
            assert all(len(enc) <= self.max_seq_len for enc in batched_all_enc)
            max_len = max(len(enc) for enc in batched_all_enc)
            batched_all_enc = [enc + [self.tokenizer.pad_token_id] * (max_len - len(enc)) for enc in batched_all_enc]
            batched_inp = torch.tensor(batched_all_enc, dtype=torch.long, device=device)
                        
            logits = self.model(batched_inp, attention_mask=None)["logits"]
            batched_logprob = F.log_softmax(logits, dim=-1)          
            for logprob, context_enc, continuation_enc in zip(
                batched_logprob, batched_context_enc, batched_continuation_enc
            ):
                logprob = logprob[len(context_enc) - 1 : len(context_enc) + len(continuation_enc) - 1]
                greedy_tokens = logprob.argmax(dim=-1)
                max_equal = greedy_tokens.tolist() == continuation_enc
                logprob = [logprob[i][continuation_enc[i]] for i in range(len(continuation_enc))]
                                
                response1.append(float(sum(logprob)))
                response2.append(bool(max_equal))

        # pack
        response = list(zip(response1, response2))[:total_requests_num]
        _response = sorted(zip(sorted_indices, response), key=lambda x: x[0])
        response = [x[1] for x in _response]
        return response

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        """Compute log-likelihood of generating a continuation from a context.
        Downstream tasks should attempt to use loglikelihood instead of other
        LM calls whenever possible.

        :param requests: list[Instance]
            A list of Instance objects, with property `args` which returns a tuple (context, continuation).
            `context: str`
                Context string. Implementations of LM must be able to handle an
                empty context string.
            `continuation: str`
                The continuation over which log likelihood will be calculated. If
                there is a word boundary, the space should be in the continuation.
                For example, context="hello" continuation=" world" is correct.

        :return: list[tuple[float, bool]]
            A list of pairs (logprob, isgreedy)
            `logprob: float`
                The log probability of `continuation`.
            `isgreedy`:
                Whether `continuation` would be generated by greedy sampling from `context`.
        """
        new_reqs = []
        if self.use_runtime_cache and self.loglikelihood_cache is not None:
            new_reqs = self.loglikelihood_cache
        else:
            for context, continuation in [req.args for req in requests]:
                continuation = str(continuation)
                                
                assert len(context) > 0
                rstrip_context = context.rstrip()
                n_spaces = len(context) - len(rstrip_context)
                if n_spaces > 0:
                    continuation = context[-n_spaces:] + continuation
                    context = context[:-n_spaces]
                    assert context == rstrip_context

                context_enc = [self.tokenizer.bos_token_id] if self.add_prefix else []
                context_enc += self.tokenizer.encode(context, add_special_tokens=False)
                full_enc = [self.tokenizer.bos_token_id] if self.add_prefix else []
                full_enc += self.tokenizer.encode(context + continuation, add_special_tokens=False)
                continuation_enc = full_enc[len(context_enc) :]

                new_reqs.append(((context, continuation), context_enc, continuation_enc))

        if self.use_runtime_cache and self.loglikelihood_cache is None:
            self.loglikelihood_cache = new_reqs

        return self._loglikelihood_tokens(new_reqs)

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        """Compute full log-likelihood of a string, with no truncation, for perplexity computation
        - We will use the full max context length of the model.
        - For inputs that exceed the max context length, we divide the tokenized string into chunks of up to
        the max context length.
        - IMPORTANT: Each document's loglikelihood/perplexity is computed *separately*, unlike other implementations
          which may simply concatenate multiple documents together.
        - IMPORTANT: We maximize the amount of context for each prediction. Specifically, for inputs that we break into
          multiple chunks, the last input will still a full-sized context.
          Example:
            Input tokens: [ 0 1 2 3 4 5 6 7 8 9 ]
            Prefix: BOS/EOS
            Max context length: 4
            Resulting input/prediction pairs:

                INPUT:  BOS   0   1   2
                PRED:     0   1   2   3

                INPUT:    3   4   5   6
                PRED:     4   5   6   7

                INPUT:    5   6   7   8
                PRED:             8   9

          Observe that:
            1. Each token is predicted exactly once
            2. For the last pair, we provide the full context, but only score the last two tokens

        :param requests: list[Instance]
            A list of Instance objects with property `args` which returns a tuple (context,).
            string: str
                String for which we are computing overall loglikelihood
        :return: list[tuple[float]]
            A list of tuples (logprob,)
            logprob: float
                The log probability of `context` conditioned on the BOS/EOS token.
                Can also be overridden for custom cases by `prefix_token_id`.
        """
        loglikelihoods = []

        for (string,) in [req.args for req in requests]:
            rolling_token_windows = list(
                map(
                    make_disjoint_window,
                    get_rolling_token_windows(
                        token_list=self.tokenizer.encode(string, add_special_tokens=False),
                        prefix_token=self.tokenizer.prefix_token_id,
                        max_seq_len=self.max_seq_len,
                        context_len=1,
                    ),
                )
            )
            rolling_token_windows = [(None,) + x for x in rolling_token_windows]

            string_nll = self._loglikelihood_tokens(rolling_token_windows)
            string_nll = [x[0] for x in string_nll]
            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)

        return loglikelihoods

    def _get_encoded_requests_for_generation(self, requests: list[Instance]) -> list[int, tuple[list[int], dict]]:
        context_encs = [self.tok_encode(req.args[0], left_truncate_len=self.max_seq_len-self.max_new_tokens) for req in requests]
        prompt_lens = [len(enc) for enc in context_encs]
        kwargs = [req.args[1] for req in requests]
        for kwarg in kwargs:
            if "max_gen_toks" in kwarg:
                kwarg["max_new_tokens"] = kwarg.pop("max_gen_toks")
                
        ids = list(range(len(requests)))
        return list(zip(ids, prompt_lens, context_encs, kwargs))

    def _prepare_batch_for_generation(self, batch: list[tuple[int, int, list[int], dict]]) -> dict[str, Any]:
        cids, prompt_lens, context_encs, kwargs_origin = zip(*batch)

        pad_to_len = max(map(len, context_encs))
        attn_masks = [[0] * (pad_to_len - len(enc)) +  [1] * len(enc) for enc in context_encs]
        attn_masks = torch.tensor(attn_masks, dtype=torch.long)
        context_encs = [[self.tokenizer.pad_token_id] * (pad_to_len - len(enc)) + enc for enc in context_encs]
        context_encs = torch.tensor(context_encs, dtype=torch.long)

        # we assume all gen kwargs in the batch are the same
        # this is safe to assume because the `grouper` object ensures it.
        kwargs_origin = kwargs_origin[0]
        # unpack our keyword arguments.
        kwargs = deepcopy(kwargs_origin)  # edge case for repeats > 1
        # add EOS token to stop sequences
        if "max_new_tokens" not in kwargs.keys():
            assert self.max_new_tokens is not None, "max_new_tokens must be specified either in the config or as an argument."
            kwargs["max_new_tokens"] = self.max_new_tokens
        else:
            if self.max_new_tokens is not None:
                assert kwargs["max_new_tokens"] <= self.max_new_tokens, (f"max_new_tokens in kwargs ({kwargs['max_new_tokens']}) is greater than max_new_tokens in config ({self.max_new_tokens})")
            assert kwargs["max_new_tokens"] <= self.max_seq_len, (f"max_new_tokens in kwargs ({kwargs['max_new_tokens']}) is greater than max_seq_len in config ({self.max_seq_len})")

        max_ctx_len = self.max_seq_len - kwargs["max_new_tokens"]
        assert context_encs.shape[1] <= max_ctx_len, (context_encs.shape[1], max_ctx_len)
        assert all(l <= max_ctx_len for l in prompt_lens), (max_ctx_len, prompt_lens)
        
        eos = self.tok_decode(self.tokenizer.eos_token_id, skip_special_tokens=False)
        until = handle_stop_sequences(kwargs.pop("until", None), eos=eos)
        until_tokens = [self.tok_encode(x, add_prefix=False) for x in until]
        
        device = get_device(self.model)
        context_encs = context_encs.to(device)
        attn_masks = attn_masks.to(device)

        return {
            "cids": cids,
            "prompt_lens": prompt_lens,
            "context_encs": context_encs,
            "attn_masks": attn_masks,
            "until": until,
            "until_tokens": until_tokens,
            "kwargs": kwargs,
        }

    def _make_batched_requests(self, requests: tuple[list[int], dict], batch_size: int) -> list[dict[str, Any]]:
        groups = defaultdict(list)
        
        for cid, prompt_len, context_enc, kwargs in requests:
            key = json.dumps(kwargs, sort_keys=True)
            groups[key].append((cid, prompt_len, context_enc, kwargs))
        
        batched_requests = []
        for group in groups.values():
            batched_requests.extend(chunk_list_by_size(group, batch_size))
        
        batched_requests = [self._prepare_batch_for_generation(batch) for batch in batched_requests]

        return batched_requests

    def generate_until(self, requests: list[Instance], disable_tqdm: bool = False) -> list[str]:
        original_len = len(requests)
        requests = self._get_encoded_requests_for_generation(requests)
        requests = sorted(requests, key=lambda x: x[1], reverse=True)
        
        if self.use_runtime_cache and self.generation_cache is not None:
            batched_requests = self.generation_cache
        else:
            batched_requests = self._make_batched_requests(requests, self.batch_size)
            if self.use_runtime_cache and self.generation_cache is None:
                self.generation_cache = batched_requests
        
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests",
        )

        res = []
        
        assert self.max_new_tokens % self.generation_num_chunks == 0
        max_new_tokens_chunk = self.max_new_tokens // self.generation_num_chunks
        for chunk_id in range(self.generation_num_chunks):
            remained_batch = []
            if is_master():
                print(f"generate for chunk {chunk_id}/{self.generation_num_chunks} num batched_requests: {len(batched_requests)}")
            for batch in batched_requests:
                kwargs_chunk = deepcopy(batch["kwargs"])
                kwargs_chunk["max_new_tokens"] = min(max_new_tokens_chunk, batch["kwargs"]["max_new_tokens"])
                out = self.model.generate(
                    input_ids=batch["context_encs"],
                    attention_mask=batch["attn_masks"],
                    stop_token_list=batch["until_tokens"],
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_dict=True,
                    return_unfinished=True,
                    prefill_chunk_size=self.prefill_chunk_size,
                    **kwargs_chunk,
                )
                batch_full_tokens = out["sequences"]
                unfinished = out["unfinished_sequences"]
                
                starts = torch.argmax((batch_full_tokens != self.tokenizer.pad_token_id).long(), dim=-1)
                full_tokens_list = batch_full_tokens.tolist()
                num_finished = 0
                                
                for cid, prompt_len, full_tokens, start, ufi in zip(
                    batch["cids"], batch["prompt_lens"], full_tokens_list, starts, unfinished):
                    # discard context + left-padding toks if using causal decoder-only LM
                    num_new_tokens = len(full_tokens)-batch["context_encs"].shape[1]
                    if (ufi == 0) or (num_new_tokens >= batch["kwargs"]["max_new_tokens"]):
                        cont_toks = full_tokens[start+prompt_len:]
                        s = self.tok_decode(cont_toks)

                        # use secondary stop seqs to cut off should-have-been-stopped content post-hoc
                        for term in batch["until"]:
                            if len(term) > 0:
                                s = s.split(term)[0]

                        res.append((cid, s))
                        num_finished += 1
                    else:
                        kwargs_next_chunk = deepcopy(batch["kwargs"])
                        kwargs_next_chunk["max_new_tokens"] = batch["kwargs"]["max_new_tokens"] - max_new_tokens_chunk
                        kwargs_next_chunk["until"] = batch["until"]
                        remained_batch.append((cid, prompt_len, full_tokens[start:], kwargs_next_chunk))
                
                pbar.set_postfix({"size": batch["context_encs"].size()})
                pbar.update(num_finished)
                            
            batched_requests = self._make_batched_requests(remained_batch, self.batch_size)
        
        # reorder this group of results back to original unsorted form
        res = sorted(res, key=lambda x: x[0])
        res = [x[1] for x in res]
        
        assert len(res) == len(requests), (len(res), len(requests))

        original_res = res        
        assert len(original_res) == original_len        
        
        pbar.close()

        return original_res
    
    def apply_chat_template(
        self, chat_history: list[dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        """
        Method to apply a chat template to a list of chat history between user and model.
        """
        chat_templated = self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )

        return chat_templated
