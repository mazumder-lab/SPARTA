import argparse

import torch
import torch.nn as nn

from database_obc import *
from torch.utils.data import DataLoader

from previous_utils.main_utils import get_model, get_dataset
from utils_training import get_item_mnist, get_item_imagenet, get_item_cifar10, initialize_dataset, load_dataset_in_memory
from pytorch_dataset_2_0 import random_split


parser = argparse.ArgumentParser()

parser.add_argument('arch', type=str)
parser.add_argument('name_dataset', type=str)
#parser.add_argument('load', type=str)
parser.add_argument('target', type=float)
parser.add_argument('--database', choices=['', 'mixed', '4block', 'unstr', '4block_8w8a'], default='')
parser.add_argument('--prefix', type=str, default='Saves_OBC')
parser.add_argument('--num_workers', type=int, default=0)

parser.add_argument('--datapath', type=str, default='')
parser.add_argument('--seed', type=int, default=0)

parser.add_argument('--batch_size_dataset', type=int, default=-1)
parser.add_argument('--nrounds', type=int, default=-1)
parser.add_argument('--noaug', action='store_true')

parser.add_argument('--skip-firstlast', action='store_true')

parser.add_argument('--bnt', action='store_true')
parser.add_argument('--bnt-batches', type=int, default=100)
parser.add_argument('--lintune', action='store_true')
parser.add_argument('--lintune-loss', action='store_true')
parser.add_argument('--lintune-epochs', type=int, default=100)
parser.add_argument('--lintune-lr', type=float, default=1e-4)
parser.add_argument('--gap', action='store_true')
parser.add_argument('--gap-epochs', type=int, default=100)
parser.add_argument('--gap-lr', type=float, default=1e-5)
parser.add_argument('--finetune', action='store_true')
parser.add_argument('--finetune-mse', action='store_true')
parser.add_argument('--finetune-epochs', type=int, default=2)
parser.add_argument('--finetune-lr', type=float, default=1e-5)
parser.add_argument('--statcorr', action='store_true')
parser.add_argument('--statcorr-samples', type=int, default=-1)
parser.add_argument('--save', type=str)

parser.add_argument('--prune_bias', type=int, default = 0,
                    help='wether to prune the bias or not')
parser.add_argument('--pretrained', type=lambda x: (str(x).lower() == 'true'), default=True)
parser.add_argument('--gamma', type=float, default = 1.0,
                help='gamma for SmoothStep')
parser.add_argument('--activation_fn', type=str, default = "relu",
                help='activation function for llm')
parser.add_argument('--n_train_kept', type=int, default = -1,
                help='number of training samples kept')
parser.add_argument('--test_almost_sequential', type=int, default = 3,
                help='If test_almost_sequential==1, we do not save the original dataset and only use the dataset created from the new model. If test_almost_sequential==2, we use this strategy only for pruning and use the original dataset for the retraining phases. If test_almost_sequential==3, we use fully parallel pruning and retraining.')

parser.add_argument('--rel_damp', type=float, default = 1e-2,
                help='rel_damp*torch.diag(H).mean() is added to the hessian for more stability.')
parser.add_argument('--lambda_fisher', type=float, default = 1e4,
                    help='lambda_fisher*WF is used in the objective function (with WF the wood-fisher approximation of the Hessian)')
parser.add_argument('--lambda_reconst', type=float, default = 1.0,
                    help='lambda_reconst*H is used in the objective function (with H the Hessian of the layer-wise reconstruction loss)')
parser.add_argument('--n_convex', type=int, default = -1,
                    help='number of convex combinations to try per layer (if set to -1, only the combination (lambda_reconst, lambda_fisher) is tried. If n_convex!=-1, then (lambda_reconst, lambda_fisher) is ignored and a list of n_convex pairs is created.')
parser.add_argument('--pruning_level', type=str, default = "layer",
                    help='Either layer or block: the saliency scores are used to prune the layers either at the layer level or block level')
parser.add_argument('--n_layers', type=int, default = -1,
                    help='number of layers to combine to form a block when pruning_level == "block"')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = parser.parse_args()

name_dataset = args.name_dataset
seed = args.seed
n_train_kept = args.n_train_kept
n_convex = args.n_convex
pruning_level = args.pruning_level
n_layers = args.n_layers

if pruning_level == "block":
    to_add = "_"+pruning_level
    if n_layers != -1:
        to_add = "_"+str(n_layers)
else:
    to_add = ""
    
if n_convex==-1:
    args.prefix += f"_{args.lambda_reconst}_{args.lambda_fisher}_{args.rel_damp}{to_add}"
else:
    args.prefix += f"_{n_convex}_{args.rel_damp}{to_add}"

path_file = '%s_%s_%dx_spdy' % (args.arch, args.database, int(args.target * 100))
path_file = os.path.join(args.prefix, "sparsity_levels", path_file)
path_file = path_file.replace('spdy', 'uniform')

name = (path_file+ '.txt').replace("sparsity_levels", "results")

name_path_weights = '%s_%04d.pth' % (args.arch, int(args.target * 10000))
if n_convex==-1:
    path_weights =  os.path.join(f"Saves_OBC_{args.lambda_reconst}_{args.lambda_fisher}_{args.rel_damp}{to_add}/models_unstr", name_path_weights)
else:
    path_weights =  os.path.join(f"Saves_OBC_{args.n_convex}_{args.rel_damp}{to_add}/models_unstr", name_path_weights)

print("Folder saves:", args.prefix)


#dataloader, testloader = get_loaders(
#    args.name_dataset, path=args.datapath,
#    batchsize=args.batch_size_dataset, workers=args.workers,
#    nsamples=args.n_train_kept, seed=args.seed,
#    noaug=args.noaug
#)
#get_model, test, run = get_functions(args.arch)

##Change this to path of imagenet name_dataset
if 'IMAGENET_PATH' in os.environ:  
    IMAGENET_PATH = os.environ['IMAGENET_PATH']+"/raw"
else:
    print('****Warning**** No IMAGENET_PATH variable', flush=True)
    #IMAGENET_PATH = ''
    IMAGENET_PATH = "/run/user/62607/loopmnt4/raw"
CIFAR10_PATH = '../datasets'
MNIST_PATH = '../datasets'
C4_PATH = "/home/gridsan/gafriat/Sparse_NN_shared/LLM/data/"
WIKITEXT_PATH = "/home/gridsan/gafriat/Sparse_NN_shared/LLM/data/"
PTB_PATH = "/home/gridsan/gafriat/Sparse_NN_shared/LLM/data/"

name_dataset_paths = {'imagenet':IMAGENET_PATH,'cifar10':CIFAR10_PATH,
                'mnist':MNIST_PATH, 'c4':C4_PATH, 'wikitext2':C4_PATH, 'ptb':C4_PATH}

name_dataset_path = name_dataset_paths[name_dataset]

print("Name dataset:", name_dataset, flush=True)
print("Path dataset:", name_dataset_path, flush=True)

model, criterion, modules_to_prune = get_model(args.arch, seed, pretrained=args.pretrained, with_z=False, gamma=args.gamma, prune_bias=args.prune_bias, activation_fn=args.activation_fn)

test_update_test_vit = True

if name_dataset == "mnist":
    get_item_func = get_item_mnist
elif name_dataset == "cifar10":
    get_item_func = get_item_cifar10
elif name_dataset == "imagenet":
    get_item_func = get_item_imagenet
else:
    get_item_func = None

test_pass_first_modules = False
train_val_dataset, test_dataset = get_dataset(name_dataset, name_dataset_path, n_train_kept, get_item_func, args.arch, seed, args.activation_fn, device, args.test_almost_sequential, test_update_test_vit, test_pass_first_modules)

if name_dataset in ["c4", "wikitext2", "ptb"]:
    n_train_kept = -1
    (train_val_dataset, train_val_attention_mask), (test_dataset, test_attention_mask) = train_val_dataset, test_dataset
    if torch.sum(torch.abs(train_val_attention_mask-test_attention_mask)).item()!=0:
        print("--- DIFFERENCE IN ATTENTION MASK ---")
        import ipdb;ipdb.set_trace()
initialize_dataset(train_val_dataset, n_train_kept, name_dataset)
initialize_dataset(test_dataset, -1, name_dataset)
train_val_dataset.return_original = False
test_dataset.return_original = False

generator_split = torch.Generator().manual_seed(seed)
train_dataset, validation_dataset = random_split(train_val_dataset, [0.8, 0.2], generator=generator_split)
generator_loader = torch.Generator()

if seed != -1:
    torch.random.manual_seed(seed)
    generator_loader = generator_loader.manual_seed(seed)

loader_train = DataLoader(train_dataset, batch_size=args.batch_size_dataset, shuffle=True, generator=generator_loader, num_workers=args.num_workers, pin_memory=True)
loader_val = DataLoader(validation_dataset, batch_size=args.batch_size_dataset, num_workers=args.num_workers, pin_memory=True)
loader_test = DataLoader(test_dataset, batch_size=args.batch_size_dataset, num_workers=args.num_workers, pin_memory=True)
loader_train.dataset.indices = [i for i in range(len(loader_train.dataset.dataset))]
loader_val.dataset.indices = []

print("Load data in memory first...", flush = True)
test_update_original = False
load_dataset_in_memory(loader_train, loader_val, n_train_kept, test_update_original)
print("Done!", flush = True)

modelp = copy.deepcopy(model)
modelp.to(device)
modelp.eval()

modelp = modelp.to(device)

modelp.load_state_dict(torch.load(path_weights, map_location=device))

acc = test(modelp, loader_test)

if not(os.path.exists(name.split("/")[0]+"/results")):
    os.mkdir(name.split("/")[0]+"/results")

print("Accuracy:", acc)

with open(name, 'w') as f:
    f.write('%.2f\n' % (acc))
