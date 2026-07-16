# =========================================================================
# Copyright (C) 2026. UniRank Authors. All rights reserved.
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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
# =========================================================================


import os
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
import random
import inspect


def disable_torch_compile(fn):
    compiler = getattr(torch, "compiler", None)
    if compiler is not None and hasattr(compiler, "disable"):
        return compiler.disable(fn)
    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None and hasattr(dynamo, "disable"):
        return dynamo.disable(fn)
    return fn


def seed_everything(seed=1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def build_batch_dict_from_tensor(batch_tensor, batch_columns):
    """Slice a batch tensor into a feature dictionary."""
    return {column: batch_tensor[:, index] for column, index in batch_columns}


def parse_gpu_ids(gpu_arg):
    """Parse a comma-separated GPU list; -1 selects CPU."""
    value = str(gpu_arg).strip()
    if value == "-1":
        return []
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise ValueError("--gpu cannot be empty; use --gpu -1 for CPU")
    if any(not part.isdigit() for part in parts):
        raise ValueError(f"Invalid --gpu value: {gpu_arg}; expected values such as 0,1,2,3")
    gpu_ids = [int(part) for part in parts]
    if len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError(f"Duplicate GPU IDs in --gpu: {gpu_arg}")
    return gpu_ids


def setup_visible_devices(gpu_ids):
    """Set CUDA_VISIBLE_DEVICES from physical GPU IDs."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))


def init_distributed_env():
    """Read the distributed process coordinates injected by torchrun."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    return world_size > 1, rank, local_rank, world_size, local_world_size


def is_main_process(rank):
    return rank == 0


def distributed_barrier(local_rank=None):
    """Synchronize the active process group, using the NCCL device when required."""
    if not (dist.is_available() and dist.is_initialized()):
        return
    if dist.get_backend() == "nccl" and local_rank is not None:
        dist.barrier(device_ids=[local_rank])
    else:
        dist.barrier()

def get_device(gpu=-1):
    if gpu >= 0 and torch.cuda.is_available():
        device = torch.device("cuda:" + str(gpu))
    else:
        device = torch.device("cpu")   
    return device

def get_optimizer(optimizer, params, lr, weight_decay=0):
    params = list(params)
    if len(params) == 0:
        return None
    if isinstance(optimizer, str):
        optimizer_alias = {
            "adam": "Adam",
            "adamw": "AdamW",
            "adagrad": "Adagrad",
            "sgd": "SGD",
            "sparseadam": "SparseAdam",
        }
        optimizer = optimizer_alias.get(optimizer.lower(), optimizer)
    try:
        optimizer_cls = getattr(torch.optim, optimizer)
        kwargs = {"lr": lr}
        if "weight_decay" in inspect.signature(optimizer_cls).parameters:
            kwargs["weight_decay"] = weight_decay
        optimizer = optimizer_cls(params, **kwargs)
    except Exception:
        raise NotImplementedError("optimizer={} is not supported.".format(optimizer))
    return optimizer

def get_loss(loss):
    if isinstance(loss, str):
        if loss in ["bce", "binary_crossentropy", "binary_cross_entropy"]:
            loss = "binary_cross_entropy"
    try:
        loss_fn = getattr(torch.functional.F, loss)
    except:
        try: 
            loss_fn = eval("losses." + loss)
        except:
            raise NotImplementedError("loss={} is not supported.".format(loss))       
    return loss_fn

def get_activation(activation, hidden_units=None):
    if isinstance(activation, str):
        activation_name = activation.strip().lower()
        if activation_name in ["prelu", "dice"]:
            assert type(hidden_units) == int
        if activation_name == "none":
            return None
        if activation_name == "relu":
            return nn.ReLU()
        elif activation_name == "softplus":
            return nn.Softplus()
        elif activation_name == "silu":
            return nn.SiLU()
        elif activation_name == "gelu":
            return nn.GELU()
        elif activation_name == "sigmoid":
            return nn.Sigmoid()
        elif activation_name == "mish":
            return nn.Mish()
        elif activation_name == "tanh":
            return nn.Tanh()
        elif activation_name == "softmax":
            return nn.Softmax(dim=-1)
        elif activation_name == "prelu":
            return nn.PReLU(hidden_units, init=0.1)
        elif activation_name == "dice":
            from unirank.pytorch.layers.activations import Dice
            return Dice(hidden_units)
        else:
            return getattr(nn, activation)()
    elif isinstance(activation, list):
        if hidden_units is not None:
            assert len(activation) == len(hidden_units)
            return [get_activation(act, units) for act, units in zip(activation, hidden_units)]
        else:
            return [get_activation(act) for act in activation]
    return activation

def get_initializer(initializer):
    if isinstance(initializer, str):
        try:
            initializer = eval(initializer)
        except:
            raise ValueError("initializer={} is not supported."\
                             .format(initializer))
    return initializer
