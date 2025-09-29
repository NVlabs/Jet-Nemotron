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

from typing import Any

import torch
import torch.nn as nn


def val2tuple(x: tuple | list | Any, min_len: int = 1, idx_repeat: int = -1) -> tuple:
    if isinstance(x, (list, tuple)):
        x = list(x)
    else:
        x = [x]

    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)


def chunk_list_by_size(x: list, chunk_size: int) -> list[list]:
    x_chunks = []
    for i in range(0, len(x), chunk_size):
        x_chunks.append(x[i : i + chunk_size])
    return x_chunks


def chunk_list_interleaved(x: list, chunks: int) -> list[list]:
    x_chunks = []
    for i in range(chunks):
        x_chunks.append(x[i::chunks])
    return x_chunks


def get_device(model: nn.Module) -> torch.device:
    return model.parameters().__next__().device