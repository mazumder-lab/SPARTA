# Adapted from code provided by the authors of SPDY [9]

import argparse
import math
import os
import random

import numpy as np
import torch

from database_obc import *
#from OBC_utils.datautils import *
#from OBC_utils.datautils2 import *
#from OBC_utils.modelutils import *
#from OBC_utils.quant import *
from torch.utils.data import Dataset, DataLoader, Subset

from previous_utils.main_utils import get_model, get_dataset
from utils_training import get_item_mnist, get_item_imagenet, get_item_cifar10, initialize_dataset, load_dataset_in_memory
from pytorch_dataset_2_0 import random_split
from utils_dataset import read_batch

parser = argparse.ArgumentParser()
parser.add_argument('arch')
parser.add_argument('name_dataset')
parser.add_argument('target', type=float)
parser.add_argument('database', choices=['mixed', '4block', 'unstr', '4block_8w8a'])
parser.add_argument('--errors', choices=['', 'squared', 'loss'], default='')
parser.add_argument('--constr', choices=['', 'bits', 'bops', 'flops', 'timingsq', 'sparsity'], default='')
parser.add_argument('--nobatchnorm', action='store_true')
parser.add_argument('--statcorr', action='store_true')
parser.add_argument('--dpbuckets', type=int, default=10000)
parser.add_argument('--dp', action='store_true')
parser.add_argument('--score', type=str, default='')
parser.add_argument('--num_workers', type=int, default=0)

parser.add_argument('--prefix', type=str, default='Saves_OBC')
parser.add_argument('--datapath', type=str, default='')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch_size_dataset', type=int, default=-1)
parser.add_argument('--nrounds', type=int, default=-1)

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

args = parser.parse_args()

name_dataset = args.name_dataset
seed = args.seed
n_train_kept = args.n_train_kept
n_convex = args.n_convex

if n_convex==-1:
    args.prefix += f"_{args.lambda_reconst}_{args.lambda_fisher}_{args.rel_damp}"
else:
    args.prefix += f"_{n_convex}_{args.rel_damp}"

print("Folder saves:", args.prefix)

#get_model, test, run = get_functions(args.arch)
#dataloader, testloader = get_loaders(
#    args.dataset, path=args.datapath,
#    batch_size_dataset=args.batch_size_dataset, workers=args.workers,
#    nsamples=args.n_train_kept, seed=args.seed,
#    noaug=True
#)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

#model, loader_train, loader_test = model_factory(args.arch, dset_path, True, seed, args.n_train_kept, batch_size=args.batch_size_dataset)
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

#modelp = get_model()
modelp = copy.deepcopy(model)
modelp.to(device)
modelp.eval()

if args.database in ['mixed', '4block_8w8a']:
    add_actquant(modelp)
layersp = find_layers(modelp)

# New code:
if "deit" in args.arch:
    new_layersp = {}
    for key in layersp:
        if not("embed" in key) and not("head" in key) and ("blocks" in key):
            new_layersp[key]=layersp[key]
layersp = new_layersp
# End new code

batches = []
for batch_sgd in tqdm(loader_train):
    input_batch_sgd, input_batch_original_sgd, target_batch_sgd, index_seen_sgd, old_index_seen_sgd = read_batch(batch_sgd)
    batches.append(run(modelp, (input_batch_sgd, target_batch_sgd), retmoved=True))

if args.database == 'mixed':
    db = QuantNMDatabase(args.arch, prefix=args.prefix)
if args.database in ['4block', 'unstr', '4block_8w8a']:
    db = SparsityDatabase(args.database, args.arch, prefix=args.prefix)

DEFAULT_CONSTR = {
    'mixed': 'bops',
    '4block': 'flops',
    'unstr': 'flops',
    '4block_8w8a': 'timingsq'
}
if not args.constr:
    args.constr = DEFAULT_CONSTR[args.database]
if not args.errors:
    args.errors = 'loss' if args.dp else 'spdy'
errors = db.load_errors(args.errors)
baseline_constr = None
if args.constr == 'bits':
    constr = db.get_bits(layersp)
if args.constr == 'bops':
    constr = db.get_bops(layersp, modelp, batches[0], run)
if args.constr == 'flops':
    constr = db.get_flops(layersp, modelp, batches[0], run)
if args.constr == 'sparsity':
    constr = db.get_sparsity(layersp, modelp, batches[0], run)
if args.constr == 'timingsq':
    baseline_constr, constr = db.get_timingsq()

print("constr is",constr)
modelp.train()
if args.nobatchnorm or args.statcorr:
    batchnorms = find_layers(modelp, [nn.BatchNorm2d])
    for bn in batchnorms.values():
        bn.eval()
if args.statcorr:
    batch = batches[0] 
    batches = [batch] 
    args.n_train_kept = args.batch_size_dataset

    #modeld = get_model()
    modeld = copy.deepcopy(model)
    modeld.to(device)
    modeld.eval()

    layersd = find_layers(modeld, layers=[nn.BatchNorm2d, nn.LayerNorm])
    layersp1 = find_layers(modelp, layers=[nn.BatchNorm2d, nn.LayerNorm])

    REDUCE = {
        2: [0],
        3: [0, 1],
        4: [0, 2, 3]
    }

    meansd = {}
    stdsd = {}
    def hookd(name):
        def tmp(layer, inp, out):
            red = REDUCE[len(out.shape)]
            meansd[name] = torch.mean(out.data, red, keepdim=True)
            stdsd[name] = torch.std(out.data, red, keepdim=True)
        return tmp
    def hookp(name):
        def tmp(layer, inp, out):
            red = REDUCE[len(out.shape)]
            meanp = torch.mean(out.data, red, keepdim=True)
            stdp = torch.std(out.data, red, keepdim=True)
            out.data -= meanp
            out.data *= stdsd[name] / (stdp + 1e-9)
            out.data += meansd[name]
        return tmp
    for name in layersd:
        layersd[name].register_forward_hook(hookd(name))
    with torch.no_grad():
        run(modeld, batch)
    for name in layersp1:
        layersp1[name].register_forward_hook(hookp(name))


layers = list(layersp.keys())
sparsities = list(errors[layers[0]].keys())
costs = [[errors[l][s] for s in sparsities] for l in layers] 
timings = [[constr[l][s] for s in sparsities] for l in layers]
costs = np.array(costs)

prunabletime = sum(max(c) for c in timings)

# # New code (not pruning everything):
# timings_prunable = np.array(timings)
# timings_prunable = timings_prunable[timings_prunable[:,0]-timings_prunable[:,-1]!=0]
# prunabletime = np.sum(np.max(timings_prunable, 1))
# # End new code

if baseline_constr is None:
    baseline_constr = prunabletime
target_constr = baseline_constr * (1-args.target) - (baseline_constr - prunabletime)

best = sum(min(c) for c in timings)
print('Max target:', baseline_constr / (best + baseline_constr - prunabletime))
bucketsize = target_constr / args.dpbuckets

for row in timings:
    for i in range(len(row)):
        row[i] = int(round(row[i] / bucketsize))

def dp(costs):
    DP = np.full((len(layers), args.dpbuckets + 1), float('inf'))
    PD = np.full((len(layers), args.dpbuckets + 1), -1)
    # DP = np.full((len(layers), max(args.dpbuckets, np.max(timings)) + 1), float('inf'))
    # PD = np.full((len(layers), max(args.dpbuckets, np.max(timings)) + 1), -1)

    for sparsity in range(len(sparsities)):
        try:
            if costs[0][sparsity] < DP[0][timings[0][sparsity]]:
                DP[0][timings[0][sparsity]] = costs[0][sparsity]
                PD[0][timings[0][sparsity]] = sparsity
        except:
            pass
    
    for layer in range(1, len(DP)):
        for sparsity in range(len(sparsities)):
            timing = timings[layer][sparsity]
            if timing == 0 and layer == len(DP) - 1:
                DP[layer] = DP[layer - 1]
                PD[layer] = 0
                continue
            if timing == 0 and layer == len(DP) - 1:
                DP[layer] = DP[layer - 1]
                PD[layer] = 0
                continue
            if timing < 1 or timing > args.dpbuckets:
                continue
            score = costs[layer][sparsity]
            tmp = DP[layer - 1][:-timing] + score
            better = tmp < DP[layer][timing:]
            if np.sum(better):
                DP[layer][timing:][better] = tmp[better]
                PD[layer][timing:][better] = sparsity

    score = np.min(DP[-1, :])
    timing = np.argmin(DP[-1, :])
    
    solution = []
    for layer in range(len(DP) - 1, -1, -1):
        solution.append(PD[layer][timing])
        timing -= timings[layer][solution[-1]]
    solution.reverse()
    return solution

def gen_costs(coefs):
    return costs * coefs.reshape((-1, 1))

def stitch_model(solution):
    config = {n: sparsities[s] for n, s in zip(layers, solution)}
    db.stitch(layersp, config)
    return modelp

@torch.no_grad()
def get_loss(model):
    loss = 0
    for batch in batches:
        loss += run(modelp, batch, loss=True)
    return loss / args.n_train_kept 

def get_score(coefs):
    costs = gen_costs(coefs)
    solution = dp(costs)
    model = stitch_model(solution)
    return get_loss(model)

if args.score:
    with open(args.score, 'r') as f:
        solution = []
        for l in f.readlines():
            splits = l.split(' ')
            sparsity = splits[0]
            name = splits[1][:-1]
            i = sparsities.index(sparsity) 
            solution.append(i)
    print(baseline_constr / (baseline_constr - prunabletime + sum(t[s] for s, t in zip(solution, timings)) * bucketsize))
    print(get_loss(stitch_model(solution)))
    exit()

def save_profile(coefs, name=''):
    solution = dp(gen_costs(coefs))
    print("solution is",solution)
    if name:
        with open(name, 'w') as f:
            for s, n in zip(solution, layers):
                f.write('%s %s\n' % (sparsities[s], n))
    else:
        for s, n in zip(solution, layers):
            print('%s %s' % (sparsities[s], n))

print('Base:', get_loss(modelp))

name = '%s_%s_%dx_spdy' % (args.arch, args.database, int(args.target * 100))

if not(os.path.exists(os.path.join(args.prefix, "sparsity_levels"))):
    os.mkdir(os.path.join(args.prefix, "sparsity_levels"))

name = os.path.join(args.prefix, "sparsity_levels", name)

if args.dp:
    name = name.replace('spdy', 'dp')
    coefs = np.ones(len(layers))
    print(get_score(np.ones(len(layers))))
    save_profile(coefs)
    save_profile(coefs, name + '.txt')
    exit()

evals = 0
print('Finding init ...')
coefs = None
score = float('inf')
for _ in range(100):
    coefs1 = np.random.uniform(0, 1, size=len(layers))
    score1 = get_score(coefs1)
    evals += 1
    print(evals)
    if score1 < score:
        print(score1)
        score = score1
        coefs = coefs1
print('Running local search ...')
for resamplings in range(int(.1 * len(layers)), 0, -1):
    print('Trying %d resamplings ...' % resamplings)
    improved = True
    while improved: 
        improved = False
        for _ in range(100):
            coefs1 = coefs.copy()
            for _ in range(resamplings):
                coefs1[random.randint(0, len(layers) - 1)] = np.random.uniform(0, 1)
            score1 = get_score(coefs1)
            evals += 1
            print(evals)
            if score1 < score:
                print(score1)
                score = score1
                coefs = coefs1
                improved = True
                break
            
print(coefs)
save_profile(coefs)
save_profile(coefs, name + '.txt')
