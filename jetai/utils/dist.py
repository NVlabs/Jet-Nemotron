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
from datetime import timedelta
from typing import Optional

import torch
import torch.distributed


def _get_dist_local_rank() -> int:
    return int(os.environ["LOCAL_RANK"])


def dist_close() -> None:
    dist_barrier()
    torch.distributed.destroy_process_group()


def dist_init(gpu: Optional[str] = None, cudnn_benchmark: bool = False, timeout: int = 3600) -> None:
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.enabled = cudnn_benchmark
    torch.backends.cudnn.deterministic = not cudnn_benchmark

    torch.distributed.init_process_group(backend="nccl", timeout=timedelta(seconds=timeout))
    assert torch.distributed.is_initialized()

    torch.cuda.set_device(_get_dist_local_rank())


def get_dist_rank() -> int:
    return int(os.environ.get("RANK", 0))


def get_dist_size() -> int:
    return int(os.environ["WORLD_SIZE"])


def is_master() -> bool:
    return get_dist_rank() == 0


def dist_barrier() -> None:
    torch.distributed.barrier()


def dist_print(x: str, rank=0) -> None:
    if get_dist_rank() == rank:
        print(x)


def sync_tensor(tensor: torch.Tensor | float, reduce="root", dim=0) -> torch.Tensor | list[torch.Tensor]:
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.Tensor(1).fill_(tensor).cuda()
    tensor_list = [torch.empty_like(tensor) for _ in range(get_dist_size())]
    torch.distributed.all_gather(tensor_list, tensor.contiguous(), async_op=False)
    if reduce == "mean":
        return torch.mean(torch.stack(tensor_list, dim=dim), dim=dim)
    elif reduce == "sum":
        return torch.sum(torch.stack(tensor_list, dim=dim), dim=dim)
    elif reduce == "cat":
        return torch.cat(tensor_list, dim=dim)
    elif reduce == "stack":
        return torch.stack(tensor_list, dim=dim)
    elif reduce == "root":
        return tensor_list[0]
    else:
        return tensor_list