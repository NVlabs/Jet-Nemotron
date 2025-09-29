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
import lm_eval
import pandas as pd

from time import time
from typing import Optional, Any

import torch
import torch.nn as nn
import torch.distributed as dist

from jetai.evaluation.lm_eval_harness.wrapper import LMEvalWrapper
from jetai.utils import (
    is_master,
    val2tuple,
    dist_print
)


pd.set_option("future.no_silent_downcasting", True)


def prepare_for_json(obj):
    if isinstance(obj, dict):
        if any(not isinstance(key, str) for key in obj.keys()):
            return {str(k): prepare_for_json(v) for k, v in obj.items()}
        return {k: prepare_for_json(v) for k, v in obj.items()}
    
    if isinstance(obj, (list, tuple, set)):
        return [prepare_for_json(item) for item in obj]
    
    return obj


def save_to_local(results: dict, save_dir: str, keys_to_remove: list[str] = None) -> None:
    dist_print(f"Saving full results to {save_dir}")
    if is_master():
        os.makedirs(save_dir, exist_ok=True)
    
    keys_to_remove = keys_to_remove or []
    
    if "samples" in results:
        samples = results.pop("samples")
        with open(os.path.join(save_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=4, default=str)
        sample_save_dir = os.path.join(save_dir, "samples")
        os.makedirs(sample_save_dir, exist_ok=True)
        for key, value in samples.items():
            with open(os.path.join(sample_save_dir, f"{key}.json"), "w") as f:
                value = prepare_for_json(value)
                for inst in value:
                    for k in keys_to_remove:
                        if k in inst:
                            del inst[k]
                json.dump(value, f, indent=4, default=str)


class LMHarnessEvaluator:
    def __init__(self, 
        wrapper: LMEvalWrapper, 
        tasks: list[tuple[str, str]], 
        subtasks: list[tuple[str, str]] = None,
        cache_requests: bool = True,
        max_test_num: Optional[int] = None,
        apply_chat_template: bool = False,
        system_instruction: Optional[str] = None,
        keys_to_remove_in_save: Optional[str] = None
    ) -> None:

        self.wrapper = wrapper
        self.tasks = tasks
        self.subtasks = subtasks
        self.runtime_cache = {}
        self.cache_requests = cache_requests
        self.max_test_num = max_test_num
        self.apply_chat_template = apply_chat_template
        self.system_instruction = system_instruction
        self.keys_to_remove_in_save = keys_to_remove_in_save
        dist_print(f"Tasks:\n{json.dumps(self.tasks, indent=2)}")
        if self.subtasks is not None:
            dist_print(f"Sub tasks:\n{json.dumps(self.subtasks, indent=2)}")

    def _get_task_metric_score(self, task_name: str, metric_name: str, results: dict) -> float:
        try:
            eval_dict = results[task_name]
        except KeyError:
            raise ValueError(f"Task {task_name} not found in evaluation results, eval_dict keys: {results.keys()}")
        if metric_name not in eval_dict:
            raise ValueError(f"Metric {metric_name} not found in evaluation results for task {task_name}, eval_dict: {eval_dict}")
        return eval_dict[metric_name]

    def evaluate(self, 
                 model: Optional[nn.Module] = None, 
                 full_save_dir: Optional[str] = None, 
                 verbose: bool = True
        ) -> dict[str, Any]:
        
        if model is not None:
            self.wrapper.model = model
        
        is_training = self.wrapper.model.training
        self.wrapper.model.eval()

        dist_print("Tasks:", self.tasks)
        dist_print("Starting evaluation with lm-eval...")

        verbose_dict = {}
        subtask_dict = {}

        with torch.no_grad():
            st = time()
            torch.cuda.synchronize()
            out = lm_eval.simple_evaluate(
                model=self.wrapper,
                tasks=[t["name"] for t in self.tasks],
                verbosity="ERROR",
                bootstrap_iters=0,
                cache_requests=self.cache_requests,
                return_runtime_cache=self.wrapper.use_runtime_cache,
                limit=self.max_test_num,
                log_samples=verbose,
                confirm_run_unsafe_code=True,
                apply_chat_template=self.apply_chat_template,
                system_instruction=self.system_instruction,
                **self.runtime_cache
            )
            torch.cuda.synchronize()
            et = time()
            dist_print(f"lm-eval took {et - st:.2f}s")
            if self.wrapper.use_runtime_cache:
                results, self.runtime_cache = out
            else:
                results = out

        if is_master():
            full_results = results
            results = results["results"]
            for task in self.tasks:
                task_name = task["name"]
                for metric_name in val2tuple(task["metric"]):
                    verbose_dict[f"{task_name}@{metric_name}"] = self._get_task_metric_score(task_name, metric_name, results)

            if self.subtasks is not None:
                for task in self.subtasks:
                    task_name = task["name"]
                    for metric_name in val2tuple(task["metric"]):
                        subtask_dict[f"{task_name}@{metric_name}"] = self._get_task_metric_score(task_name, metric_name, results)

            if verbose and full_save_dir is not None:
                save_to_local(full_results, full_save_dir, keys_to_remove=self.keys_to_remove_in_save)

            out = {
                "avg": sum(verbose_dict.values()) / len(verbose_dict),
                "verbose": verbose_dict,
                "subtasks": subtask_dict
            }
        else:
            out = None

        out = [out]
        dist.broadcast_object_list(out, src=0)
        out = out[0]

        self.wrapper.model.train(is_training)

        return out
