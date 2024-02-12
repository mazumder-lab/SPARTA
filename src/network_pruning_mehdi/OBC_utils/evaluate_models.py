#%%
import argparse
import copy
import os

import torch
import torch.nn as nn

from OBC_utils.datautils import *
from OBC_utils.datautils2 import *
from OBC_utils.modelutils import *
from OBC_utils.quant import *
from OBC_utils.trueobs import *
from torch.utils.data import Dataset, DataLoader, Subset
import time

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='resnet20')
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument(
    '--compress', type=str, choices=['quant', 'nmprune', 'unstr', 'struct', 'blocked'],
    default='unstr')

parser.add_argument('--load', type=str, default='')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--save', type=str, default='')

parser.add_argument('--nsamples', type=int, default=-1)
parser.add_argument('--batchsize', type=int, default=-1)
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--nrounds', type=int, default=1)
parser.add_argument('--noaug', action='store_true')

parser.add_argument('--wbits', type=int, default=32)
parser.add_argument('--abits', type=int, default=32)
parser.add_argument('--wperweight', action='store_true')
parser.add_argument('--wasym', action='store_true')
parser.add_argument('--wminmax', action='store_true')
parser.add_argument('--asym', action='store_true')
parser.add_argument('--aminmax', action='store_true')
parser.add_argument('--rel-damp', type=float, default=0)

parser.add_argument('--prunen', type=int, default=2)
parser.add_argument('--prunem', type=int, default=4)
parser.add_argument('--blocked_size', type=int, default=4)
parser.add_argument('--min-sparsity', type=float, default=0)
parser.add_argument('--max-sparsity', type=float, default=0)
parser.add_argument('--delta-sparse', type=float, default=0)
parser.add_argument('--sparse-dir', type=str, default='')

parser.add_argument('--l_sparsities', type=str, default='0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9')

args, _ = parser.parse_known_args()


# NEW CODE
l_sparsities = args.l_sparsities
if l_sparsities != "":
    l_sparsities = np.array(l_sparsities.split(", "), dtype=float)
else:
    l_sparsities = []
# END NEW CODE

##Change this to path of imagenet name_dataset
if 'IMAGENET_PATH' in os.environ:  
    IMAGENET_PATH = os.environ['IMAGENET_PATH']+"/raw"
else:
    print('****Warning**** No IMAGENET_PATH variable', flush=True)
    #IMAGENET_PATH = ''
    IMAGENET_PATH = "/run/user/62607/loopmnt4/raw"
CIFAR10_PATH = '../datasets'
MNIST_PATH = '../datasets'

dset_paths = {'imagenet':IMAGENET_PATH,'cifar10':CIFAR10_PATH,
                'mnist':MNIST_PATH}

dset_path = dset_paths[args.dataset]


for sparsity_level in l_sparsities:
    model,data_loader,test_loader = model_factory(args.model, dset_path, True, args.seed, args.nsamples, initialize_bn=False, batch_size=args.batchsize)
    state_trained = torch.load(f"models_uniform/{args.model}_{int(sparsity_level*10000)}.pth")
    model.load_state_dict(state_trained)
    model.to("cuda")
    model.train()
    compute_acc(model, data_loader, "cuda")
    model.eval()
    test_acc = compute_acc(model, test_loader, "cuda")
    print(f"Test acc for {args.model} with sparsity level of {sparsity_level}: {test_acc}%")
    number_of_non_zeros_params = np.sum([torch.sum(x.detach()!=0).item() for x in list(model.parameters())])
    number_of_params = np.sum([np.prod(x.shape) for x in list(model.parameters())])
    sparsity = 1-number_of_non_zeros_params/number_of_params
    print(f"Actual sparsity: {sparsity}")


# %%
