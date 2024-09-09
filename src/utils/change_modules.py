#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
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

import copy
import logging
from typing import List
from typing import Iterable, Tuple
import torch.nn as nn
from opacus.utils.module_utils import clone_module, get_submodule, parametrized_modules #, trainable_modules original trainable_modules

from utils.partially_trainable_modules import (
    Conv2d_partially_trainable,
    Linear_partially_trainable,
)


def fully_trainable_modules(module: nn.Module) -> Iterable[Tuple[str, nn.Module]]:
    """
    Recursively iterates over all submodules, returning those that
    have parameters and are trainable (ie they want a grad).
    """
    yield from (
        (m_name, m)
        for (m_name, m) in parametrized_modules(module)
        if all(p.requires_grad for p in m.parameters(recurse=False)) #original is any
    )

def fix(module: nn.Module, gamma=1.0, partially_trainable_bias=False) -> nn.Module:
    """
    Make the module and sub_modules DP compatible by running registered custom fixers.

    Args:
        module: The root module to be made compatible.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Fixed module.
    """
    module = clone_module(module)
    # iterate over all sub_modules
    # We have to get sub_module names in a list first as we will be
    # changing the modules inside the loop.
    sub_module_names = [name for name, _ in fully_trainable_modules(module)]
    for sub_module_name in sub_module_names:
        # get sub_module
        sub_module = get_submodule(module, sub_module_name)
        # if sub_module has a registered fixer
        if "Conv2d" in str(type(sub_module)) or "Linear" in str(type(sub_module)):
            # get a replacement for sub_module
            if "Linear" in str(type(sub_module)):
                new_sub_module = Linear_partially_trainable(
                    sub_module.in_features,
                    sub_module.out_features,
                    sub_module.bias is not None,
                    gamma,
                    next(sub_module.parameters()).device,
                    next(sub_module.parameters()).dtype,
                    partially_trainable_bias,
                )
            else:
                new_sub_module = Conv2d_partially_trainable(
                    sub_module.in_channels,
                    sub_module.out_channels,
                    sub_module.kernel_size,
                    sub_module.stride,
                    sub_module.padding,
                    sub_module.dilation,
                    sub_module.groups,
                    sub_module.bias is not None,
                    sub_module.padding_mode,  # TODO: refine this type
                    next(sub_module.parameters()).device,
                    next(sub_module.parameters()).dtype,
                    gamma,
                    partially_trainable_bias,
                )
            if new_sub_module.use_mask:
                new_sub_module.init_weight.data = copy.deepcopy(sub_module.weight.data)
            else:
                new_sub_module.weight_trainable.data = copy.deepcopy(sub_module.weight.data)
            if new_sub_module.test_bias:
                if partially_trainable_bias and new_sub_module.use_mask:
                    new_sub_module.init_bias.data = copy.deepcopy(sub_module.bias)
                else:
                    new_sub_module.bias_trainable.data = copy.deepcopy(sub_module.bias)

            # move new_sub_module to the same device as that of sub_module
            new_sub_module.to(next(sub_module.parameters()).device)
            # get module after replacement.
            module = replace_sub_module(
                root=module,
                sub_module_name=sub_module_name,
                new_sub_module=new_sub_module,
            )
            # log it
            print(
                f"Replaced sub_module {sub_module_name} : {sub_module}" f" with {new_sub_module}",
                flush=True,
            )
    # return fixed module
    return module


def replace_sub_module(
    root: nn.Module,
    sub_module_name: str,
    new_sub_module: nn.Module,
):
    sub_module_path = sub_module_name.split(".")
    if len(sub_module_path) == 1 and sub_module_path[0] == "":  # root is the only sub_module of root
        return new_sub_module
    else:  # replace root's descendant
        sub_module_parent = root
        for name in sub_module_path[:-1]:  # descend down to sub_module
            sub_module_parent = sub_module_parent._modules[name]
        sub_module_parent._modules[sub_module_path[-1]] = new_sub_module
    return root
