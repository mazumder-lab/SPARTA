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
parser.add_argument('--output_dir', default='output/one-shot', type=str, help='dir to save results')
parser.add_argument('--num_workers', type=int, default=0)

parser.add_argument('--seed', type=int, default=0)

parser.add_argument('--batch_size_dataset', type=int, default=-1)

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = parser.parse_args()

name_dataset = args.name_dataset
seed = args.seed

n_train_kept = args.n_train_kept

path_weights = os.path.join(args.output_dir, f'{args.arch}_sparsity={args.target}.pth')
path_weights = "/home/gridsan/gafriat/projects/CAP-main/"+path_weights

print("Folder saves:", args.output_dir)

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

model.eval()

model = model.to(device)

model.load_state_dict(torch.load(path_weights, map_location=device))

acc = test(model, loader_test)

print("Accuracy:", acc)

with open(path_weights.replace("pth", "txt"), 'w') as f:
    f.write('%.2f\n' % (acc))
