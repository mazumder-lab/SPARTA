#
import argparse
import torch
import plotly.graph_objects as go
import numpy as np
import os

from torch.utils.data import DataLoader
from previous_utils.main_utils import get_model, get_dataset
from utils_training import get_item_mnist, get_item_imagenet, get_item_cifar10, initialize_dataset, load_dataset_in_memory
from pytorch_dataset_2_0 import random_split
from utils_dataset import read_batch
import copy
from tqdm import tqdm
import time
#from utils_training import train_sub_modules
from utils_model import model_wrapper
import pickle
from utils_pruning import prune_blocked
from sklearn.model_selection import train_test_split

def get_loss_acc(model, loader, criterion, arch, device):
    loss = 0
    acc = 0
    n_seen_loader = 0
    with torch.no_grad():
        for batch_sgd in tqdm(loader):
            input_batch_sgd, input_batch_original_sgd, target_batch_sgd, index_seen_sgd, old_index_seen_sgd = read_batch(batch_sgd)
            pred = model(input_batch_sgd.to(device))
            if "opt" in arch:
                # with torch.autocast(device_type=model_wrapper.device, dtype=torch.float16):
                shift_logits = pred[:, :-1, :].contiguous()
                shift_labels = target_batch_sgd[:, 1:].contiguous()
                shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                loss = criterion(shift_logits, shift_labels.view(-1).to(device))
            else:
                loss = criterion(pred, target_batch_sgd.to(device))
            loss += loss.item()*target_batch_sgd.shape[0]
            acc += (torch.argmax(pred, 1)==target_batch_sgd.to(device)).float().sum().item()
            n_seen_loader += target_batch_sgd.shape[0]
    loss /= n_seen_loader
    acc /= n_seen_loader
    return loss, acc

def evaluate_sparsity(ind_new_step, ind_layer, mat_losses, mat_acc, d_params, model, mat_is_exact, folder_saves_OBC, l_path_weights, l_layers, device, loader_train, criterion, arch, test_sparsities, gpts, l_levels_pruned_weights):
    name_layer = l_layers[ind_layer]
    old_weights = copy.deepcopy(d_params[name_layer].data)
    if test_sparsities:
        weights_new_sparsity = torch.load(folder_saves_OBC+"/models_unstr/"+l_path_weights[ind_new_step], map_location=device)[name_layer]
    else:
        new_sparsity = l_levels_pruned_weights[ind_new_step]/l_n_params[ind_layer]
        weights_new_sparsity = prune_blocked(gpts[name_layer.replace(".weight", "")], 0, [new_sparsity])[0]
    # Try the pruned weights
    d_params[name_layer].data = weights_new_sparsity
    # Compute loss for pruned weights
    mat_losses[ind_new_step][ind_layer], mat_acc[ind_new_step][ind_layer] = get_loss_acc(model, loader_train, criterion, arch, device)
    mat_is_exact[ind_new_step][ind_layer] = True
    # Reset the weights to their original values
    d_params[name_layer].data = old_weights

@torch.no_grad()
def test(model, dataloader):
    train = model.training
    model.eval()
    print('Evaluating ...')
    dev = next(iter(model.parameters())).device
    preds = []
    ys = []
    for batch_sgd in tqdm(dataloader):
        input_batch_sgd, input_batch_original_sgd, target_batch_sgd, index_seen_sgd, old_index_seen_sgd = read_batch(batch_sgd)
        preds.append(torch.argmax(model(input_batch_sgd.to(dev)), 1))
        ys.append(target_batch_sgd.to(dev))
    acc = torch.mean((torch.cat(preds) == torch.cat(ys)).float()).item()
    acc *= 100
    print('%.2f' % acc)
    if model.training:
        model.train()
    return acc

def get_min_sparsity(l_n_params, l_current_step, goal_sparsity, l_sparsities, max_sparsity_start):
    n_params = np.sum(l_n_params)
    min_sparsity_unseen_layers = np.nanmax([(1/np.sum(l_n_params[l_current_step==0]))*(n_params*goal_sparsity-np.sum((l_n_params*l_sparsities[l_current_step])[l_current_step!=0])), 0.0])
    ind_min_sparsity = np.where(l_sparsities<=min_sparsity_unseen_layers)[0][-1]

    if ind_min_sparsity==0:
        ind_min_sparsity = 1
        min_sparsity_unseen_layers = l_sparsities[ind_min_sparsity]

    if max_sparsity_start!=-1:
        ind_min_sparsity = min(ind_min_sparsity, max_sparsity_start)
        min_sparsity_unseen_layers = l_sparsities[ind_min_sparsity]

    return min_sparsity_unseen_layers, ind_min_sparsity

def get_min_step(l_n_params, l_current_step, goal_sparsity, l_levels_pruned_weights, max_sparsity_start):
    n_params = np.sum(l_n_params)
    min_sparsity_unseen_layers = np.nanmax([(1/np.sum(l_n_params[l_current_step==0]))*(n_params*goal_sparsity-np.sum((l_levels_pruned_weights[l_current_step])[l_current_step!=0])), 0.0])
    min_n_params = np.min(min_sparsity_unseen_layers*l_n_params)
    ind_min_n_weights = np.where(l_levels_pruned_weights<=min_n_params)[0][-1]

    if ind_min_n_weights==0:
        ind_min_n_weights = 1
        min_n_params = l_levels_pruned_weights[ind_min_n_weights]

    if max_sparsity_start!=-1:
        ind_min_n_weights = min(ind_min_n_weights, max_sparsity_start)
        min_n_params = l_levels_pruned_weights[ind_min_n_weights]

    return min_n_params, ind_min_n_weights

def get_l_max_steps(n_layers, l_sparsities, test_sparsities, l_n_params, n_weights_step, max_sparsity):
    l_maximum_new_step = np.ones(n_layers, dtype=int)
    if test_sparsities:
        l_maximum_new_step *= (len(l_sparsities)-1)
    else:
        l_maximum_new_step = np.array(np.floor(l_n_params*max(min(max_sparsity, 1),0)/n_weights_step), dtype = int)
    return l_maximum_new_step

parser = argparse.ArgumentParser()

parser.add_argument('arch')
parser.add_argument('name_dataset')
parser.add_argument('--l_goal_sparsities', type=str, default='0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9')
parser.add_argument('--rel_damp', type=float, default = 1e-2,
                help='rel_damp*torch.diag(H).mean() is added to the hessian for more stability.')
parser.add_argument('--lambda_fisher', type=float, default = 1.0,
                    help='lambda_fisher*WF is used in the objective function (with WF the wood-fisher approximation of the Hessian)')
parser.add_argument('--lambda_reconst', type=float, default = 1.0,
                    help='lambda_reconst*H is used in the objective function (with H the Hessian of the layer-wise reconstruction loss)')
parser.add_argument('--n_convex', type=int, default = -1,
                    help='number of convex combinations to try per layer (if set to -1, only the combination (lambda_reconst, lambda_fisher) is tried. If n_convex!=-1, then (lambda_reconst, lambda_fisher) is ignored and a list of n_convex pairs is created.')
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--batch_size_dataset', type=int, default=-1)
parser.add_argument('--n_train_kept', type=int, default = -1,
                help='number of training samples kept')
parser.add_argument('--further_subsampling', type=int, default = 500,
                help='keep a subsample out of n_train_kept samples (if n_train_kept!=-1)')
parser.add_argument('--max_sparsity_start', type=int, default = -1,
                help='Maximum increase in sparsity for the fully dense layers at the start of the greedy algorithm. If set to -1, then the maximum possible sparsity is used.')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--no_approximation', type=int, default=0)
parser.add_argument('--test_sparsities', type=int, default=0, 
                    help = "Whether to increase the layers using sparisty or number of parameters")
parser.add_argument('--initial_sparsity', type=float, default=0.0,
                    help = "Whether to increase the layers using sparisty or number of parameters")
parser.add_argument('--test_recompute', type=int, default=0,
                    help = "Whether to recompute all the training losses at each sparsity goal")
parser.add_argument('--stepsize', type=int, default=1,
                    help = "We try to prune stepsize * n_params for all the layers with n_params corresponding to 1 percent of the weights of the biggest layer")
parser.add_argument('--cost_metric', type=str, default="acc",
                    help = "Either 'acc' or 'loss' for training accuracy or training loss")

args = parser.parse_args()

print(args)
# Hyperparameters
arch = args.arch
rel_damp = args.rel_damp
name_dataset = args.name_dataset
seed = args.seed
n_train_kept = args.n_train_kept
n_convex = args.n_convex
lambda_fisher = args.lambda_fisher
lambda_reconst = args.lambda_reconst
#goal_sparsity = args.goal_sparsity
l_goal_sparsities = np.array(args.l_goal_sparsities.split(", "), dtype=float)
num_workers = args.num_workers
batch_size_dataset = args.batch_size_dataset
max_sparsity_start = args.max_sparsity_start
no_approximation = args.no_approximation
test_sparsities = args.test_sparsities
initial_sparsity = args.initial_sparsity
further_subsampling = args.further_subsampling
test_recompute = args.test_recompute
stepsize = args.stepsize
cost_metric = args.cost_metric

#rel_damp = 1e-2
#lambda_fisher = 1.0
#lambda_reconst = 1.0
#n_convex = -1
#arch = "deit_tiny_patch16_224"
#arch = "mlpnet"
#name_dataset = "imagenet"
#n_train_kept = 5000
#seed = 0
#num_workers = 0
#batch_size_dataset = 128
pretrained = True 
with_z = False
gamma = 1.0
prune_bias = False
activation_fn = "relu"
test_almost_sequential = 0
pruning_level = "layer"
n_layers = -1

# Path to pruned weights
if n_convex==-1:
    folder_saves_OBC = f"Saves_OBC_{lambda_reconst}_{lambda_fisher}_{rel_damp}"
else:
    folder_saves_OBC = f"Saves_OBC_{n_convex}_{rel_damp}"

print("Folder saves:", folder_saves_OBC)

if test_sparsities:
    # l_path_weights = os.listdir(f"{folder_saves_OBC}/models_unstr")
    # l_path_weights = [x for x in l_path_weights if "pth" in x and arch in x]
    # l_sparsities = [float(x.split(arch)[1].replace(".pth", "").replace("_", ""))/10000 for x in l_path_weights]
    # idx_sort_path = np.argsort(l_sparsities)
    # l_sparsities = np.array(l_sparsities)[idx_sort_path]
    # l_path_weights = np.array(l_path_weights)[idx_sort_path]
    print(1)

# Get dataset
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
train_val_dataset, test_dataset = get_dataset(name_dataset, name_dataset_path, n_train_kept, get_item_func, arch, seed, activation_fn, device, test_almost_sequential, test_update_test_vit, test_pass_first_modules, further_subsampling)
if n_train_kept!=-1 and further_subsampling!=-1:
    n_train_kept = further_subsampling

# import ipdb; ipdb.set_trace()
# if n_train_kept!=-1 and further_subsampling!=-1:
#     current_indices = train_val_dataset.indices
#     train_val_dataset_ori = train_val_dataset.dataset
#     current_targets = train_val_dataset_ori.targets[current_indices]
#     if name_dataset == "imagenet" and further_subsampling<=1000:
#         stratify_on = current_targets%further_subsampling
#     else:
#         stratify_on = current_targets
#     indices_to_remove, indices_train_subsample, _, targets_train_subsample = train_test_split(current_indices, current_targets, test_size = further_subsampling, stratify = stratify_on, random_state=seed)
#     new_order = np.concatenate([indices_train_subsample, indices_to_remove, np.arange(len(train_val_dataset_ori.targets))[n_train_kept:]])
#     train_val_dataset.data = train_val_dataset.data[new_order]
#     train_val_dataset.targets = train_val_dataset.targets[new_order]
#     train_val_dataset = torch.utils.data.Subset(train_val_dataset, np.arange(further_subsampling))

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

loader_train = DataLoader(train_dataset, batch_size=batch_size_dataset, shuffle=False, generator=generator_loader, num_workers=num_workers, pin_memory=True)
loader_val = DataLoader(validation_dataset, batch_size=batch_size_dataset, num_workers=num_workers, pin_memory=True)
loader_test = DataLoader(test_dataset, batch_size=batch_size_dataset, num_workers=num_workers, pin_memory=True)
loader_train.dataset.indices = [i for i in range(len(loader_train.dataset.dataset))]
loader_val.dataset.indices = []

print("Load data in memory first...", flush = True)
test_update_original = False
load_dataset_in_memory(loader_train, loader_val, n_train_kept, test_update_original)
print("Done!", flush = True)

# Get model
model, criterion, modules_to_prune = get_model(arch, seed, pretrained=pretrained, with_z=False, gamma=gamma, prune_bias=prune_bias, activation_fn=activation_fn)
model.eval()
model.to(device)

if not(test_sparsities):
    l_weights_layers = np.array([np.prod(x[1].shape) for x in model.blocks.named_parameters() if "bias" not in x[0] and "norm" not in x[0]])
    n_weights_step = stepsize*int(np.ceil(np.max(l_weights_layers)/100))
    max_number_iter = int(np.ceil(np.max(l_weights_layers)/n_weights_step))
    l_levels_pruned_weights = np.arange(max_number_iter)*n_weights_step

l_new_weights = []

if pruning_level == "block":
    to_add = "_"+pruning_level
    if n_layers != -1:
        to_add = +"_"+str(n_layers)
else:
    to_add = ""
        
if n_convex==-1:
    end_name = f"{lambda_reconst}_{lambda_fisher}_{rel_damp}{to_add}"
else:
    end_name = f"{n_convex}_{rel_damp}{to_add}"

if test_sparsities:
    gpts = None
else:
    l_path_weights = None
    with open(f'gpts_{arch}_{end_name}.pickle', 'rb') as handle:
        gpts = pickle.load(handle)

# Find layers
d_params = dict(model.named_parameters())
l_layers = list(d_params.keys())
if "deit" in arch:
    l_layers = [x for x in l_layers if "blocks" in x and "norm" not in x and "bias" not in x]
l_layers = np.array(l_layers)
n_layers = len(l_layers)

# Initialize sparsities and collect number of parameters
l_current_step = np.zeros(n_layers, dtype=int)
l_n_params = np.array([np.prod(d_params[x].shape) for x in l_layers])
n_params = np.sum(l_n_params)
#l_seen = np.array([False for _ in range(n_layers)])
if test_sparsities:
    mat_losses = np.ones((len(l_sparsities), n_layers))*np.inf
    mat_acc = np.ones((len(l_sparsities), n_layers))*np.inf
    mat_is_exact = np.array([[False for _ in range(n_layers)] for _ in range(len(l_sparsities))]) # True if the cost has just been recomputed (and is exact). If false, then the corresponding cost is either +inf or old (and is only a lower bound for the orinal cost)
else:
    mat_losses = np.ones((len(l_levels_pruned_weights), n_layers))*np.inf
    mat_acc = np.ones((len(l_levels_pruned_weights), n_layers))*np.inf
    mat_is_exact = np.array([[False for _ in range(n_layers)] for _ in range(len(l_levels_pruned_weights))]) # True if the cost has just been recomputed (and is exact). If false, then the corresponding cost is either +inf or old (and is only a lower bound for the orinal cost)

if cost_metric == "loss":
    mat_metric = mat_losses
elif cost_metric == "acc":
    mat_metric = -mat_acc

l_possible_new_step = np.zeros(n_layers, dtype=int)

current_sparsity = 0.0

ind_step = 0
n_evaluations = 0
start_time = time.time()

l_initial_n_weights_pruned = l_levels_pruned_weights[None]<=l_n_params[:, None]*0.2
if initial_sparsity!=0:
    for ind_layer in range(n_layers):
        name_layer = l_layers[ind_layer]
        n_weights_to_prune = initial_sparsity*l_n_params[ind_layer]
        n_weights_to_prune = np.min(l_levels_pruned_weights[l_levels_pruned_weights>=n_weights_to_prune])
        ind_initial_step = np.where(l_levels_pruned_weights==n_weights_to_prune)[0].item()
        l_current_step[ind_layer] = ind_initial_step
        new_sparsity = n_weights_to_prune/l_n_params[ind_layer]
        weights_new_sparsity = prune_blocked(gpts[name_layer.replace(".weight", "")], 0, [new_sparsity])[0]
        d_params[name_layer].data = weights_new_sparsity
        l_possible_new_step[ind_layer] = l_current_step[ind_layer]+1

l_saved_sparsities = []
l_saved_train_losses = []
l_saved_train_acc = []

print(f"--------- List of goal sparsities: {l_goal_sparsities}")

# TEMP
# l_new_sparsities = np.array([0.6935,0.5202,0.3001,0.4001,0.6135,0.6802,0.3001,0.5502,0.3334,0.6802,0.3001,0.5902,0.6935,0.4401,0.5902,0.5702,0.3068,0.3201,0.4501,0.6902,0.3068,0.6002,0.3001,0.4801,0.6402,0.4801,0.4801,0.5702,0.5735,0.3601,0.3601,0.3301,0.6935,0.3201,0.6902,0.6902,0.3334,0.3201,0.4001,0.4601,0.3068,0.6802,0.6602,0.4601,0.6935,0.5202,0.6902,0.5602])

# for ind_layer in range(len(l_layers)):
#     new_sparsity = l_new_sparsities[ind_layer]
#     name_layer = l_layers[ind_layer]
#     weights_new_sparsity = prune_blocked(gpts[name_layer.replace(".weight", "")], 0, [new_sparsity])[0]
#     d_params[name_layer].data = weights_new_sparsity

# test_acc = test(model, loader_test)
# import ipdb;ipdb.set_trace()
#END TEMP

for goal_sparsity in l_goal_sparsities:
    l_maximum_new_step = get_l_max_steps(n_layers, None, test_sparsities, l_n_params, n_weights_step, 1.4*goal_sparsity)
    print(f"--------- Goal sparsity: {goal_sparsity}")

    if test_recompute:
        if test_sparsities:
            mat_losses = np.ones((len(l_sparsities), n_layers))*np.inf
            mat_acc = np.ones((len(l_sparsities), n_layers))*np.inf
            mat_is_exact = np.array([[False for _ in range(n_layers)] for _ in range(len(l_sparsities))]) # True if the cost has just been recomputed (and is exact). If false, then the corresponding cost is either +inf or old (and is only a lower bound for the orinal cost)
        else:
            mat_losses = np.ones((len(l_levels_pruned_weights), n_layers))*np.inf
            mat_acc = np.ones((len(l_levels_pruned_weights), n_layers))*np.inf
            mat_is_exact = np.array([[False for _ in range(n_layers)] for _ in range(len(l_levels_pruned_weights))]) # True if the cost has just been recomputed (and is exact). If false, then the corresponding cost is either +inf or old (and is only a lower bound for the orinal cost)

        if cost_metric == "loss":
            mat_metric = mat_losses
        elif cost_metric == "acc":
            mat_metric = -mat_acc

    while current_sparsity < goal_sparsity:
        ind_step += 1
        # Initialize min sparisty for unpruned layers and costs
        # if test_sparsities:
        #     min_sparsity_unseen_layers, ind_min_sparsity = get_min_sparsity(l_n_params, l_current_step, goal_sparsity, l_sparsities, max_sparsity_start)
        # else:
        min_n_params, ind_min_n_weights = get_min_step(l_n_params, l_current_step, goal_sparsity, l_levels_pruned_weights, max_sparsity_start)

        if not(test_sparsities):
            l_possible_new_step[l_current_step==0] = ind_min_n_weights
            to_try_sparsities = l_levels_pruned_weights[l_possible_new_step]/l_n_params
            print(f"Step {ind_step} - Trying sparsities {np.round(to_try_sparsities, 4)}", flush = True)
        else:
            l_possible_new_step[l_current_step==0] = ind_min_sparsity
            to_try_sparsities = l_sparsities[l_possible_new_step]
            print(f"Step {ind_step} - Trying sparsities {to_try_sparsities}", flush = True)


        # Update infinite costs
        for ind_layer in range(n_layers):
            # name_layer = l_layers[ind_layer]
            # old_weights = copy.deepcopy(d_params[name_layer].data)
            ind_new_step = l_possible_new_step[ind_layer]
            if mat_losses[ind_new_step][ind_layer] == np.inf or (no_approximation and not(mat_is_exact[ind_new_step][ind_layer])):
                # weights_new_sparsity = torch.load(folder_saves_OBC+"/models_unstr/"+l_path_weights[ind_new_step], map_location=device)[name_layer]
                # # Try the pruned weights
                # d_params[name_layer].data = weights_new_sparsity
                # # Compute loss for pruned weights
                # mat_losses[ind_new_step][ind_layer] = get_loss_acc(model, loader_train, criterion, arch, device)
                # mat_is_exact[ind_new_step][ind_layer] = True
                # # Reset the weights to their original values
                # d_params[name_layer].data = old_weights
                if to_try_sparsities[ind_layer]>=1.0:
                    mat_is_exact[ind_new_step][ind_layer] = True
                else:
                    evaluate_sparsity(ind_new_step, ind_layer, mat_losses, mat_acc, d_params, model, mat_is_exact, folder_saves_OBC, l_path_weights, l_layers, device, loader_train, criterion, arch, test_sparsities, gpts, l_levels_pruned_weights)
                    n_evaluations += 1

        # Find next layer to update
        test_next_update_found = False
        idx_updated_layer = None
        while not(test_next_update_found):
            cost_to_pick_from = copy.deepcopy(np.diag(mat_metric[l_possible_new_step]))
            cost_to_pick_from[l_possible_new_step==l_maximum_new_step]=np.inf
            next_idx_layer = np.argsort(cost_to_pick_from)[0]
            name_layer = l_layers[next_idx_layer]
            ind_new_step = l_possible_new_step[next_idx_layer]
            if mat_is_exact[ind_new_step][next_idx_layer]:
                test_next_update_found = True
                mat_is_exact[:, np.arange(n_layers)!=next_idx_layer] = False
                if test_sparsities:
                    weights_new_sparsity = torch.load(folder_saves_OBC+"/models_unstr/"+l_path_weights[ind_new_step], map_location=device)[name_layer]
                else:
                    old_sparsity = l_levels_pruned_weights[l_current_step[next_idx_layer]]/l_n_params[next_idx_layer]
                    new_sparsity = l_levels_pruned_weights[ind_new_step]/l_n_params[next_idx_layer]
                    weights_new_sparsity = prune_blocked(gpts[name_layer.replace(".weight", "")], 0, [new_sparsity])[0]
                d_params[name_layer].data = weights_new_sparsity
                idx_updated_layer = next_idx_layer
                if test_sparsities:
                    print(f"Updated layer {idx_updated_layer} from {l_sparsities[l_current_step[idx_updated_layer]]} to {l_sparsities[l_possible_new_step[idx_updated_layer]]}", flush = True)
                else:
                    print(f"Updated layer {idx_updated_layer} from {old_sparsity} to {new_sparsity}", flush = True)
                l_current_step[idx_updated_layer] = l_possible_new_step[idx_updated_layer]
            else:
                evaluate_sparsity(ind_new_step, next_idx_layer, mat_losses, mat_acc, d_params, model, mat_is_exact, folder_saves_OBC, l_path_weights, l_layers, device, loader_train, criterion, arch, test_sparsities, gpts, l_levels_pruned_weights)
                n_evaluations += 1
                #mat_losses[ind_new_step][next_idx_layer] = get_loss_acc(model, loader_train, criterion, arch, device)
                #mat_is_exact[ind_new_step][next_idx_layer] = True

        if not(no_approximation):
            # Line search to increase sparsity of found layer
                    
            # Find minimum possible sparsity
            name_layer = l_layers[idx_updated_layer]
            min_possible_step = l_current_step[idx_updated_layer]
            # Find maximum possible sparsity to not exceed the goal sparsity
            if test_sparsities:
                max_sparsity_value = (n_params*goal_sparsity - np.sum((l_n_params*l_sparsities[l_current_step])[np.arange(n_layers)!=idx_updated_layer]))/l_n_params[idx_updated_layer]
                max_idx_goal_sparsity = np.where(l_sparsities<=max_sparsity_value)[0][-1]
                max_possible_step = max(min_possible_step, max_idx_goal_sparsity)
            else:
                max_n_params = (n_params*goal_sparsity - np.sum((l_levels_pruned_weights[l_current_step])[np.arange(n_layers)!=idx_updated_layer]))
                max_idx_levels_pruning = np.where(l_levels_pruned_weights<=max_n_params)[0][-1]
                max_possible_step = max(min_possible_step, max_idx_levels_pruning)


            # Find maximum possible sparsity such that the losses for the other layer are still lower bounds
            if initial_sparsity == 0:
                l_current_step_copy = copy.deepcopy(l_current_step)
                max_step_to_keep_lower_bounds = min_possible_step
                for possible_step in range(max_possible_step+1):
                    l_current_step_copy[idx_updated_layer] = possible_step
                    if test_sparsities:
                        _, ind_min_sparsity_copy = get_min_sparsity(l_n_params, l_current_step_copy, goal_sparsity, l_sparsities, max_sparsity_start)
                        if ind_min_sparsity_copy == ind_min_sparsity:
                            max_step_to_keep_lower_bounds = possible_step
                        else:
                            break
                    else:
                        _, ind_min_n_weights_copy = get_min_step(l_n_params, l_current_step, goal_sparsity, l_levels_pruned_weights, max_sparsity_start)
                        if ind_min_n_weights_copy == ind_min_n_weights:
                            max_step_to_keep_lower_bounds = possible_step
                        else:
                            break

                max_step_to_keep_lower_bounds+=1
                max_possible_step = min(max_possible_step, max_step_to_keep_lower_bounds)

            max_possible_step = min(max_possible_step, l_maximum_new_step[idx_updated_layer])
        
            if test_sparsities:
                print(f"Line search for layer {idx_updated_layer} between {l_sparsities[min_possible_step]} and {l_sparsities[max_possible_step]}", flush = True)
            else:
                print(f"Line search for layer {idx_updated_layer} between {np.round(l_levels_pruned_weights[min_possible_step]/l_n_params[idx_updated_layer],4)} and {np.round(l_levels_pruned_weights[max_possible_step]/l_n_params[idx_updated_layer],4)}", flush = True)
           
            l_losses = copy.deepcopy(np.diag(mat_metric[l_possible_new_step]))
            l_losses[l_possible_new_step==l_maximum_new_step]=np.inf
            # l_acc = copy.deepcopy(np.diag(mat_acc[l_possible_new_step]))
            # l_acc[l_possible_new_step==l_maximum_new_step]=np.inf
            ind_cost_to_match = np.argmin(l_losses[np.arange(n_layers)!=idx_updated_layer])
            cost_to_match = l_losses[np.arange(n_layers)!=idx_updated_layer][ind_cost_to_match]
            # corres_acc = l_acc[np.arange(n_layers)!=idx_updated_layer][ind_cost_to_match]
            if cost_to_match<mat_metric[l_current_step[idx_updated_layer]][idx_updated_layer]:
                import ipdb; ipdb.set_trace()
            # acc_to_match = np.max(np.diag(mat_acc[l_possible_new_step])[np.arange(n_layers)!=idx_updated_layer])
            middle_sparsity = min_possible_step
            print(f"Minimum cost of the other layers: {cost_to_match}")
            # print(f"Corresponding accuracy: {acc_to_match}")
            while min_possible_step!=max_possible_step:
                print("min, middle, max:", min_possible_step, middle_sparsity, max_possible_step)
                middle_sparsity = int(min_possible_step/2+max_possible_step/2)
                print(f"Cost of the middle: {mat_losses[middle_sparsity][idx_updated_layer]}")
                print(f"Acc of the middle: {mat_acc[middle_sparsity][idx_updated_layer]}")
                if mat_losses[middle_sparsity][idx_updated_layer]==np.inf:
                    evaluate_sparsity(middle_sparsity, idx_updated_layer, mat_losses, mat_acc, d_params, model, mat_is_exact, folder_saves_OBC, l_path_weights, l_layers, device, loader_train, criterion, arch, test_sparsities, gpts, l_levels_pruned_weights)
                    n_evaluations += 1
                    print(f"New cost of the middle: {mat_losses[middle_sparsity][idx_updated_layer]}")
                    print(f"New acc of the middle: {mat_acc[middle_sparsity][idx_updated_layer]}")
                if mat_losses[middle_sparsity][idx_updated_layer] < cost_to_match and not(mat_is_exact[middle_sparsity][next_idx_layer]):
                    evaluate_sparsity(middle_sparsity, idx_updated_layer, mat_losses, mat_acc, d_params, model, mat_is_exact, folder_saves_OBC, l_path_weights, l_layers, device, loader_train, criterion, arch, test_sparsities, gpts, l_levels_pruned_weights)
                    n_evaluations += 1
                    # mat_losses[middle_sparsity][next_idx_layer] = get_loss_acc(model, loader_train, criterion, arch, device)
                    # mat_is_exact[ind_new_step][next_idx_layer] = True
                    print(f"New cost of the middle: {mat_losses[middle_sparsity][idx_updated_layer]}")
                    print(f"New acc of the middle: {mat_acc[middle_sparsity][idx_updated_layer]}")
                if mat_losses[middle_sparsity][idx_updated_layer] > cost_to_match:
                    max_possible_step = max(middle_sparsity-1, min_possible_step)
                    middle_sparsity = int(min_possible_step/2+max_possible_step/2)
                else:
                    min_possible_step = min(middle_sparsity+1, max_possible_step)
                    middle_sparsity = int(min_possible_step/2+max_possible_step/2)

            new_step_line_search = middle_sparsity
            if test_sparsities:
                print(f"Updated layer {idx_updated_layer} from {l_sparsities[l_current_step[idx_updated_layer]]} to {l_sparsities[new_step_line_search]}", flush = True)
            else:
                print(f"Updated layer {idx_updated_layer} from {np.round(l_levels_pruned_weights[l_current_step[idx_updated_layer]]/l_n_params[idx_updated_layer],4)} to {np.round(l_levels_pruned_weights[new_step_line_search]/l_n_params[idx_updated_layer],4)}", flush = True)
            l_current_step[idx_updated_layer] = new_step_line_search
            if test_sparsities:
                weights_new_sparsity = torch.load(folder_saves_OBC+"/models_unstr/"+l_path_weights[new_step_line_search], map_location=device)[name_layer]
            else:
                old_sparsity = l_levels_pruned_weights[l_current_step[next_idx_layer]]/l_n_params[next_idx_layer]
                new_sparsity = l_levels_pruned_weights[new_step_line_search]/l_n_params[next_idx_layer]
                weights_new_sparsity = prune_blocked(gpts[name_layer.replace(".weight", "")], 0, [new_sparsity])[0]

            d_params[name_layer].data = weights_new_sparsity

        l_possible_new_step[idx_updated_layer] = min(l_current_step[idx_updated_layer]+1, l_maximum_new_step[idx_updated_layer])

        print(" -------------------------------------------------- ", flush = True)
        mat_losses[new_step_line_search][idx_updated_layer], mat_acc[new_step_line_search][idx_updated_layer] = get_loss_acc(model, loader_train, criterion, arch, device)
        # We don't count this evaluation (we don't increase n_evaluations) as it is completely optional and only to track the performance over time. 
        current_acc = mat_acc[new_step_line_search][idx_updated_layer]
        current_loss = mat_losses[new_step_line_search][idx_updated_layer]
        if test_sparsities:
            current_sparsity = np.sum((l_sparsities[l_current_step]*l_n_params))/n_params
            print(f"Current sparisty pattern: {l_sparsities[l_current_step]}", flush = True)
        else:
            current_sparsity = np.sum(l_levels_pruned_weights[l_current_step])/n_params
            # TEMP CHECK
            l_weights_values_2 = [x[1] for x in model.blocks.named_parameters() if "norm" not in x[0] and "bias" not in x[0]]
            n_params_2 = np.sum([np.prod(x.shape) for x in l_weights_values_2])
            l_pruned_2 = np.array([(x==0).float().sum().item() for x in l_weights_values_2])
            actual_sparsity = np.sum(l_pruned_2)/n_params_2
            print(f"Check sparisty: {current_sparsity-actual_sparsity}", flush = True)
            # END TEMP CHECK
            print(f"Current sparisty pattern: {np.round(l_levels_pruned_weights[l_current_step]/l_n_params, 4)}", flush = True)
        print(f"Current sparisty: {np.round(current_sparsity, 4)}", flush = True)
        print(f"Current train accuracy: {np.round(100*current_acc, 2)}", flush = True)
        print(f"Current train loss: {np.round(current_loss, 4)}", flush = True)
        l_saved_sparsities.append(current_sparsity)
        l_saved_train_losses.append(current_loss)
        l_saved_train_acc.append(current_acc)
        print(" -------------------------------------------------- ", flush = True)

    print(f"Solution found in {n_evaluations} evaluations", flush = True)
    if test_sparsities:
        print(f"DP algorithm would have used {n_layers*len(l_sparsities)} evaluations", flush = True)
    print(f"Total time: {np.round(time.time()-start_time, 2)}s", flush = True)
    print(" -------------------------------------------------- ", flush = True)

    if goal_sparsity in [0.5, 0.6, 0.7, 0.8, 0.9]:
        # Evaluate model on test data
        test_acc = test(model, loader_test)
        print("Accuracy:", test_acc, flush = True)

        # Save sparsity distribution
        path_save_txt = f"{folder_saves_OBC}/sparsity_levels_greedy_{max_sparsity_start}_{no_approximation}_{test_sparsities}"
        if initial_sparsity!=0:
            path_save_txt += f"_{initial_sparsity}"
        if further_subsampling!=-1:
            path_save_txt += f"_fs_{further_subsampling}"
        if test_recompute==1:
            path_save_txt += "_r"
        if stepsize!=1:
            path_save_txt += f"_s_{stepsize}"

        if not(os.path.exists(path_save_txt)):
            os.mkdir(path_save_txt)

        with open(f"{path_save_txt}/{arch}_unstr_{int(goal_sparsity*100)}x_greedy.txt", 'w') as f:
            for ind_layer in range(len(l_layers)):
                current_step, name_layer = l_current_step[ind_layer], l_layers[ind_layer]
                if test_sparsities:
                    f.write('%.4f %s\n' % (l_sparsities[current_step], name_layer))
                else:
                    f.write('%.4f %s\n' % (np.round(l_levels_pruned_weights[current_step]/l_n_params[ind_layer], 4), name_layer))

        # Save results on test data
        path_save_results = f"{folder_saves_OBC}/new_results_greedy_{max_sparsity_start}_{no_approximation}_{test_sparsities}"
        if initial_sparsity!=0:
            path_save_results += f"_{initial_sparsity}"
        if further_subsampling!=-1:
            path_save_results += f"_fs_{further_subsampling}"
        if test_recompute==1:
            path_save_results += "_r"
        if stepsize!=1:
            path_save_results += f"_s_{stepsize}"

        if not(os.path.exists(path_save_results)):
            os.mkdir(path_save_results)

        path_file = '%s_%s_%dx_greedy' % (arch, "unstr", int(goal_sparsity * 100))
        path_file = os.path.join(path_save_results, path_file)
        path_model = path_file + ".pth"
        path_file = path_file + '.txt'

        with open(path_file, 'w') as f:
            f.write('%.2f\n' % (test_acc))

        torch.save(model.state_dict(), path_model)

        with open(f'{path_save_results}/l_sparsities_greedy_{arch}_{int(goal_sparsity * 100)}.npy', 'wb') as f:
            np.save(f, l_saved_sparsities)
        with open(f'{path_save_results}/l_train_acc_greedy_{arch}_{int(goal_sparsity * 100)}.npy', 'wb') as f:
            np.save(f, l_saved_train_acc)
        with open(f'{path_save_results}/l_train_losses_greedy_{arch}_{int(goal_sparsity * 100)}.npy', 'wb') as f:
            np.save(f, l_saved_train_losses)
