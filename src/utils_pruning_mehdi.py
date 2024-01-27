import time
from collections import defaultdict

import torch
import torch.nn as nn
from tqdm import tqdm

from OBC.trueobs import TrueOBS

# from utils_dataset import *
# from utils_model import Conv2d_with_z, Linear_with_z, use_mask_rec
# from utils_pruning_xiang import *


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
    argsprunen,
    argsprunem,
    argsblocksize,
    # with_z,
    # gamma,
    algo_pruning,
    rel_damp,
):
    print("Starting ...")
    print("Ready.")

    update_layer_wise = True

    subset = find_layers(block, layers=[nn.Conv2d, nn.Linear])

    gpts = {}
    # if algo_pruning == "sparse_gpt":
    #     for name in subset:
    #         gpts[name] = SparseGPT(subset[name])
    if algo_pruning == "obc":
        for name in subset:
            gpts[name] = TrueOBS(subset[name], rel_damp=rel_damp)

    def add_batch(name):
        def tmp(_, inp, out):
            gpts[name].add_batch(inp[0].data, out.data)

        return tmp

    handles = []
    temp_optimizer = torch.optim.SGD(block.parameters(), 0.0, 1.0)
    criterion = torch.nn.functional.cross_entropy
    loader_tmp = loader_train
    if update_layer_wise:
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
    acc_1 = 0
    acc_2 = 0
    for batch_sgd in tqdm(loader_tmp):
        if update_layer_wise:
            (
                input_batch_sgd,
                target_batch_sgd,
            ) = batch_sgd
            acc_1 = acc_2
            acc_2 += input_batch_sgd.shape[0]
            out = block(input_batch_sgd.to(dev))

    if update_layer_wise:
        for h in handles:
            h.remove()
    # if test_update_dataset:
    #     update_dataset(*args_update_dataset)

    for name in gpts:
        print(name)
        print("Pruning ...")
        sparsity = argssparsity
        if algo_pruning == "sparse_gpt":
            gpts[name].fasterprune(
                sparsity,
                prunen=argsprunen,
                prunem=argsprunem,
                percdamp=rel_damp,
                blocksize=argsblocksize,
            )
        elif algo_pruning == "obc":
            gpts[name].prepare_unstr()
            with torch.no_grad():
                gpts[name].layer.weight.data = gpts[name].prune_unstr([sparsity])[0]
        # else:
        #     # import ipdb;ipdb.set_trace()
        #     gpts[name].pruning(
        #         sparsity,
        #         prunen=argsprunen,
        #         prunem=argsprunem,
        #         lambda_stability=rel_damp,
        #         parallel=argsblocksize,
        #         device=dev,
        #         sparsities=sparsities,
        #         sds=sds,
        #         loader_tmp=loader_tmp,
        #         arch=arch,
        #         criterion=criterion,
        #         d_losses=d_losses,
        #     )
        gpts[name].free()
        # if with_z:
        #     try:
        #         gpts[name].layer.weight_z.data[gpts[name].layer.weight == 0] = -gamma
        #         gpts[name].layer.weight_z.data[gpts[name].layer.weight != 0] = gamma
        #     except:
        #         pass

    if dev not in ("cpu", torch.device("cpu")):
        torch.cuda.empty_cache()
