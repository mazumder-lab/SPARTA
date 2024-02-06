import argparse
import copy
import os
import time

import torch
import torch.nn as nn
from datautils import *
from datautils2 import *
from modelutils import *
from quant import *
from torch.utils.data import DataLoader, Dataset, Subset
from trueobs import *

parser = argparse.ArgumentParser()

parser.add_argument("model", type=str)
parser.add_argument("dataset", type=str)
parser.add_argument("compress", type=str, choices=["quant", "nmprune", "unstr", "struct", "blocked"])
parser.add_argument("--load", type=str, default="")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--save", type=str, default="")

parser.add_argument("--nsamples", type=int, default=-1)
parser.add_argument("--batchsize", type=int, default=128)
parser.add_argument("--n_datasets", type=int, default=1)
parser.add_argument("--idx_dataset", type=int, default=0)
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--nrounds", type=int, default=1)
parser.add_argument("--noaug", action="store_true")

parser.add_argument("--wbits", type=int, default=32)
parser.add_argument("--abits", type=int, default=32)
parser.add_argument("--wperweight", action="store_true")
parser.add_argument("--wasym", action="store_true")
parser.add_argument("--wminmax", action="store_true")
parser.add_argument("--asym", action="store_true")
parser.add_argument("--aminmax", action="store_true")
parser.add_argument("--rel-damp", type=float, default=0)

parser.add_argument("--prunen", type=int, default=2)
parser.add_argument("--prunem", type=int, default=4)
parser.add_argument("--blocked_size", type=int, default=4)
parser.add_argument("--min-sparsity", type=float, default=0)
parser.add_argument("--max-sparsity", type=float, default=0)
parser.add_argument("--delta-sparse", type=float, default=0)
parser.add_argument("--sparse-dir", type=str, default="")

parser.add_argument("--l_sparsities", type=str, default="")

args = parser.parse_args()

# NEW CODE
l_sparsities = args.l_sparsities
if l_sparsities != "":
    l_sparsities = np.array(l_sparsities.split(", "), dtype=float)
else:
    l_sparsities = []
# END NEW CODE

##Change this to path of imagenet name_dataset
if "IMAGENET_PATH" in os.environ:
    IMAGENET_PATH = os.environ["IMAGENET_PATH"] + "/raw"
else:
    print("****Warning**** No IMAGENET_PATH variable", flush=True)
    # IMAGENET_PATH = ''
    IMAGENET_PATH = "/run/user/62607/loopmnt4/raw"
CIFAR10_PATH = "../../datasets"
CIFAR100_PATH = "../../datasets"
MNIST_PATH = "../../datasets"

dset_paths = {"imagenet": IMAGENET_PATH, "cifar10": CIFAR10_PATH, "cifar100": CIFAR100_PATH, "mnist": MNIST_PATH}

dset_path = dset_paths[args.dataset]


model, data_loader, test_loader = model_factory(
    args.model, dset_path, True, args.seed, args.nsamples, batch_size=args.batchsize, name_dataset=args.dataset
)

l_datasets = list_random_subsets(data_loader.dataset, args.n_datasets, seed=0)
l_dataloaders = [
    DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=8, pin_memory=True)
    for train_dataset in l_datasets
]
data_loader = l_dataloaders[args.idx_dataloader]
# TEMP - CHECK PRUNING
# old_weights = copy.deepcopy(model.state_dict())
# weights = torch.load("models_unstr/resnet18_5000.pth")
# weights['linear.weight'] = old_weights['linear.weight']
# weights['linear.bias'] = old_weights['linear.bias']
# model.load_state_dict(weights)

# d_params = dict(model.named_parameters())
# for name_param in d_params:
#     if (weights[name_param]==0).float().mean()==0.5:
#         print(name_param)
#         l_weights = torch.abs(d_params[name_param].data.flatten())
#         idx_weights = torch.argsort(l_weights)
#         n_to_prune = int(len(idx_weights)*0.7)
#         d_params[name_param].data.flatten()[idx_weights[:n_to_prune]] = 0

# test(model, test_loader)
# END CHECKING PRUNING

# data_loader, test_loader = get_loaders(
#    args.dataset, path=IMAGENET_PATH,
#    batchsize=args.batchsize, workers=args.workers,
#    nsamples=args.nsamples, seed=args.seed,
#    noaug=args.noaug
# )

if args.nrounds == -1:
    args.nrounds = 1 if "yolo" in args.model or "bert" in args.model else 10
    if args.noaug:
        args.nrounds = 1

# get_model, test, run = get_functions(args.model)

aquant = args.compress == "quant" and args.abits < 32
wquant = args.compress == "quant" and args.wbits < 32

modelp = copy.deepcopy(model)
modeld = copy.deepcopy(model)
DEV = torch.device("cuda:0")
modelp.to(DEV)
modeld.to(DEV)
modelp.eval()
modeld.eval()

if args.compress == "quant" and args.load:
    modelp.load_state_dict(torch.load(args.load))
if aquant:
    add_actquant(modelp)

layersp = find_layers(modelp)
layersd = find_layers(modeld)

SPARSE_DEFAULTS = {
    "unstr": (0, 0.99, 0.1),  # mean sparisty, maximum sparsity, change of sparsity
    "struct": (0, 0.9, 0.05),
    "blocked": (0, 0.95, 0.1),
}
sparse = args.compress in SPARSE_DEFAULTS
if sparse:
    if len(l_sparsities) == 0:
        if args.min_sparsity == 0 and args.max_sparsity == 0:
            defaults = SPARSE_DEFAULTS[args.compress]
            args.min_sparsity, args.max_sparsity, args.delta_sparse = defaults
        sparsities = []
        density = 1 - args.min_sparsity
        while density > 1 - args.max_sparsity:
            sparsities.append(1 - density)
            density *= 1 - args.delta_sparse
        sparsities.append(args.max_sparsity)  # Change this to the sparsity levels I want!
    else:
        sparsities = list(l_sparsities)
    sds = {s: copy.deepcopy(modelp).cpu().state_dict() for s in sparsities}


trueobs = {}
for name in layersp:
    layer = layersp[name]
    if isinstance(layer, ActQuantWrapper):
        layer = layer.module
    trueobs[name] = TrueOBS(layer, rel_damp=args.rel_damp)
    if aquant:
        layersp[name].quantizer.configure(args.abits, sym=args.asym, mse=not args.aminmax)
    if wquant:
        trueobs[name].quantizer = Quantizer()
        trueobs[name].quantizer.configure(
            args.wbits, perchannel=not args.wperweight, sym=not args.wasym, mse=not args.wminmax
        )

if not (args.compress == "quant" and not wquant):
    cache = {}

    def add_batch(name):
        def tmp(layer, inp, out):
            # try:
            trueobs[name].add_batch(inp[0].data, out.data)
            # except:
            #     import ipdb;ipdb.set_trace()

        return tmp

    handles = []
    for name in trueobs:
        handles.append(layersd[name].register_forward_hook(add_batch(name)))

    print("Start gradient computation")
    st_grad = time.time()
    for i in range(args.nrounds):
        print("round", i)
        for j, batch in enumerate(data_loader):
            print(i, j)
            with torch.no_grad():
                run(modeld, batch)
    print("time for grad copmutation is", time.time() - st_grad)

    for h in handles:
        h.remove()
    print("start pruning")
    st_prune = time.time()
    for name in trueobs:
        print(name)
        if args.compress == "quant":
            print("Quantizing ...")
            trueobs[name].quantize()
        if args.compress == "nmprune":
            if trueobs[name].columns % args.prunem == 0:
                print("N:M pruning ...")
                trueobs[name].nmprune(args.prunen, args.prunem)
        if sparse:
            Ws = None
            if args.compress == "unstr":
                print("Unstructured pruning ...")
                trueobs[name].prepare_unstr()
                Ws = trueobs[name].prune_unstr(sparsities)
            if args.compress == "struct":
                if not isinstance(trueobs[name].layer, nn.Conv2d):
                    size = 1
                else:
                    tmp = trueobs[name].layer.kernel_size
                    size = tmp[0] * tmp[1]
                if trueobs[name].columns / size > 3:
                    print("Structured pruning ...")
                    Ws = trueobs[name].prune_struct(sparsities, size=size)
            if args.compress == "blocked":
                if trueobs[name].columns % args.blocked_size == 0:
                    print("Blocked pruning ...")
                    trueobs[name].prepare_blocked(args.blocked_size)
                    Ws = trueobs[name].prune_blocked(sparsities)
            if Ws:
                for sparsity, W in zip(sparsities, Ws):
                    sds[sparsity][name + ".weight"] = W.reshape(sds[sparsity][name + ".weight"].shape).cpu()
        trueobs[name].free()
    print("time for pruning", time.time() - st_prune)

if sparse:
    if args.sparse_dir:
        for sparsity in sparsities:
            name = "%s_%04d.pth" % (args.model, int(sparsity * 10000))
            torch.save(sds[sparsity], os.path.join(args.sparse_dir, name))
    exit()

if aquant:
    print("Quantizing activations ...")

    def init_actquant(name):
        def tmp(layer, inp, out):
            layersp[name].quantizer.find_params(inp[0].data)

        return tmp

    handles = []
    for name in layersd:
        handles.append(layersd[name].register_forward_hook(init_actquant(name)))
    with torch.no_grad():
        run(modeld, next(iter(data_loader)))
    for h in handles:
        h.remove()

if args.save:
    torch.save(modelp.state_dict(), args.save)

test(modelp, test_loader)
