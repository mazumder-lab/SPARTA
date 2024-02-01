import time
from collections import defaultdict

import torch
import torch.nn as nn
from tqdm import tqdm

from OBC.trueobs import TrueOBS


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=""):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + "." + name1 if name != "" else name1))
    return res


def prune_block(
    block,
    loader_train,
    dev,
    argssparsity,
    algo_pruning,
    rel_damp,
):
    print("Starting ...")
    print("Ready.")

    update_layer_wise = True

    old_subset = find_layers(block, layers=[nn.Conv2d, nn.Linear])
    subset = {}
    for key_layer in old_subset:
        if old_subset[key_layer].weight.requires_grad:
            subset[key_layer] = old_subset[key_layer]

    gpts = {}
    if algo_pruning == "obc":
        for name in subset:
            gpts[name] = TrueOBS(subset[name], rel_damp=rel_damp)

    def add_batch(name):
        def tmp(_, inp, out):
            gpts[name].add_batch(inp[0].data, out.data)

        return tmp

    handles = []
    loader_tmp = loader_train
    if update_layer_wise:
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
    for batch_sgd in tqdm(loader_tmp):
        if update_layer_wise:
            input_batch_sgd, target_batch_sgd = batch_sgd
            block(input_batch_sgd.to(dev))

    if update_layer_wise:
        for h in handles:
            h.remove()

    for name in gpts:
        print(name)
        print("Pruning ...")
        sparsity = argssparsity
        if algo_pruning == "obc":
            gpts[name].prepare_unstr()
            with torch.no_grad():
                gpts[name].layer.weight.data = gpts[name].prune_unstr([sparsity])[0]

        gpts[name].free()
    if dev not in ("cpu", torch.device("cpu")):
        torch.cuda.empty_cache()
