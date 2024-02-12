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

@torch.no_grad()
def get_loss(model, loader, criterion, arch, device):
    loss = 0
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
            loss+=loss.item()*target_batch_sgd.shape[0]
            n_seen_loader+=target_batch_sgd.shape[0]
    loss /= n_seen_loader
    return loss

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

parser = argparse.ArgumentParser()

parser.add_argument('arch')
parser.add_argument('name_dataset')
parser.add_argument('number_of_runs', type=int)
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
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--initial_sparsity', type=float, default=0.0,
                    help = "Whether to increase the layers using sparisty or number of parameters")
parser.add_argument('--lambda_sparsity', type=float, default=1e-2,
                    help = "The penalty to the number of parameters")
parser.add_argument('--test_penalty_per_layer', type=int, default=1,
                    help = "Whether to penalize the number of parameters per layer or the total number of parameters")

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
number_of_runs = args.number_of_runs
num_workers = args.num_workers
batch_size_dataset = args.batch_size_dataset
initial_sparsity = args.initial_sparsity
lambda_sparsity = args.lambda_sparsity
test_penalty_per_layer = args.test_penalty_per_layer

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
train_val_dataset, test_dataset = get_dataset(name_dataset, name_dataset_path, n_train_kept, get_item_func, arch, seed, activation_fn, device, test_almost_sequential, test_update_test_vit, test_pass_first_modules)

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

print(f"--------- Number of runs: {number_of_runs}")
# Get model
model, criterion, modules_to_prune = get_model(arch, seed, pretrained=pretrained, with_z=False, gamma=gamma, prune_bias=prune_bias, activation_fn=activation_fn)
model.eval()
model.to(device)

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

# Initialize number of parameters
l_losses = np.zeros(n_layers, dtype=int)
l_n_params = np.array([np.prod(d_params[x].shape) for x in l_layers])
n_params = np.sum(l_n_params)

current_sparsity = 0.0

# Example step
ind_step = 0
n_evaluations = 0
start_time = time.time()

# Initial sparsity
if initial_sparsity!=0:
    for ind_layer in range(n_layers):
        name_layer = l_layers[ind_layer]
        weights_new_sparsity = prune_blocked(gpts[name_layer.replace(".weight", "")], 0, [initial_sparsity])[0]
        d_params[name_layer].data = weights_new_sparsity

l_path_weights = os.listdir(f"{folder_saves_OBC}/models_unstr")
l_path_weights = [x for x in l_path_weights if "pth" in x and arch in x]
l_sparsities = [float(x.split(arch)[1].replace(".pth", "").replace("_", ""))/10000 for x in l_path_weights]
idx_sort_path = np.argsort(l_sparsities)
l_sparsities = np.array(l_sparsities)[idx_sort_path]
l_path_weights = np.array(l_path_weights)[idx_sort_path]

l_current_step = -np.ones(len(l_layers), dtype=int)

l_saved_losses = []
l_saved_sparsities = []

# Perform CD
for ind_run in range(number_of_runs):
    print(f"---- Run number {ind_run+1}/{number_of_runs}")
    for ind_layer in range(n_layers):
        name_layer = l_layers[ind_layer]
        old_weights = copy.deepcopy(d_params[name_layer].data)
        l_losses = np.zeros(len(l_path_weights))
        for ind_sparsity in range(len(l_path_weights)):
            weights_new_sparsity = torch.load(folder_saves_OBC+"/models_unstr/"+l_path_weights[ind_sparsity], map_location=device)[name_layer]
            d_params[name_layer].data = weights_new_sparsity
            l_losses[ind_sparsity] = get_loss(model, loader_train, criterion, arch, device)
        l_losses += lambda_sparsity*l_n_params[ind_layer]*(1-l_sparsities)/n_params
        ind_new_sparsity = np.argmin(l_losses)
        l_current_step[ind_layer] = ind_new_sparsity
        weights_chosen_sparsity = torch.load(folder_saves_OBC+"/models_unstr/"+l_path_weights[ind_new_sparsity], map_location=device)[name_layer]
        d_params[name_layer].data = weights_chosen_sparsity
        print(f"Updated layer {name_layer} from {(old_weights==0).float().sum().item()/l_n_params[ind_layer]} to {l_sparsities[ind_new_sparsity]}", flush = True)
        current_sparsity = np.sum([(d_params[x]==0).float().sum().item() for x in l_layers])/n_params
        l_saved_losses.append(l_losses[ind_new_sparsity])
        l_saved_sparsities.append(current_sparsity)
        print(f"Current sparsity: {current_sparsity}", flush = True)
print(f"Total time: {np.round(time.time()-start_time, 2)}s", flush = True)
print(" -------------------------------------------------- ", flush = True)

l_saved_losses = np.array(l_saved_losses)
l_saved_sparsities = np.array(l_saved_sparsities)

# Evaluate model on test data
test_acc = test(model, loader_test)
print("Accuracy:", test_acc, flush = True)

# Save sparsity distribution
path_save_txt = f"{folder_saves_OBC}/sparsity_levels_cd_{lambda_sparsity}_{int(100*current_sparsity)}"
if initial_sparsity!=0:
    path_save_txt += f"_{initial_sparsity}"

if not(os.path.exists(path_save_txt)):
    os.mkdir(path_save_txt)

with open(f"{path_save_txt}/{arch}_unstr_{int(number_of_runs)}x_cd.txt", 'w') as f:
    for ind_layer in range(len(l_layers)):
        f.write('%.4f %s\n' % (np.round((d_params[l_layers[ind_layer]].data == 0).float().sum().item()/n_params, 4), l_layers[ind_layer]))

# Save results on test data
path_save_results = f"{folder_saves_OBC}/results_cd_{lambda_sparsity}_{int(100*current_sparsity)}"
if initial_sparsity!=0:
    path_save_results += f"_{initial_sparsity}"

if not(os.path.exists(path_save_results)):
    os.mkdir(path_save_results)

path_file = '%s_%s_%dx_cd' % (arch, "unstr", int(number_of_runs))
path_file = os.path.join(path_save_results, path_file)
path_model = path_file + ".pth"
path_file = path_file + '.txt'

with open(path_file, 'w') as f:
    f.write('%.2f\n' % (test_acc))

torch.save(model.state_dict(), path_model)

print("sparsities:", l_saved_sparsities)
print("losses:", l_saved_losses)
with open(f'{path_save_results}/l_sparsities_cd_{arch}_{number_of_runs}.npy', 'wb') as f:
    np.save(f, l_saved_sparsities)
with open(f'{path_save_results}/l_losses_cd_{arch}_{number_of_runs}.npy', 'wb') as f:
    np.save(f, l_saved_losses)
