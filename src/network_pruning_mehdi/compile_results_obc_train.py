#%%
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from torch.utils.data import DataLoader

import torch
from previous_utils.main_utils import get_model, get_dataset
from utils_training import get_item_mnist, get_item_imagenet, get_item_cifar10, initialize_dataset, load_dataset_in_memory
from pytorch_dataset_2_0 import random_split
from utils_dataset import read_batch
from tqdm import tqdm

def get_acc(model, loader, device):
    acc = 0
    n_seen_loader = 0
    with torch.no_grad():
        for batch_sgd in tqdm(loader):
            input_batch_sgd, input_batch_original_sgd, target_batch_sgd, index_seen_sgd, old_index_seen_sgd = read_batch(batch_sgd)
            pred = model(input_batch_sgd.to(device))
            pred = torch.argmax(pred, 1)
            acc+=(pred==target_batch_sgd.to(device)).float().sum().item()
            n_seen_loader+=target_batch_sgd.shape[0]
    acc /= n_seen_loader
    return acc

name_dataset = "imagenet"
seed = 0
n_train_kept = 5000
num_workers = 0
batch_size_dataset = 128
rel_damp = 0.01

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

#model, loader_train, loader_test = model_factory(args.arch, dset_path, True, seed, args.n_train_kept, batch_size=batch_size_dataset)
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
train_val_dataset, test_dataset = get_dataset(name_dataset, name_dataset_path, n_train_kept, get_item_func, "deit", seed, "relu", device, 0, test_update_test_vit, test_pass_first_modules)

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

# %%
#l_arch = ["facebook/opt-125m"]#, "facebook/opt-350m"]
#l_arch = ["resnet50"]
l_arch = ["deit_tiny_patch16_224"]#, "deit_small_patch16_224", "deit_base_patch16_224"]
l_sparsity = [0.5, 0.6, 0.7, 0.8, 0.9]#, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98]
#l_sparsity = [0.5]
#l_sparsity = [0.5, 0.6, 0.7]
#metric_name = "acc"
metric_name = "ppl"
n_convex = 5

#%%
for arch in l_arch:
    model, criterion, modules_to_prune = get_model(arch, seed, pretrained=True, with_z=False, gamma=1.0, prune_bias=False, activation_fn="relu")
    model.to(device)

    l_convex_uniform = []
    l_fisher_uniform = []
    l_reconst_uniform = []

    l_convex_non_uniform = []
    l_fisher_non_uniform = []
    l_reconst_non_uniform = []

    l_CAP_non_uniform = []

    l_convex_uniform_multiple = []
    l_convex_non_uniform_multiple = []
    l_convex_non_uniform_multiple_greedy = []

    l_convex_uniform_block = []
    l_fisher_uniform_block = []
    l_reconst_uniform_block = []

    l_convex_uniform_multiple_block = []


    for goal_sparisty in l_sparsity:
        path_convex_uniform = f"Saves_OBC_1.0_1.0_0.01/models_unstr/{arch}_{int(10000*goal_sparisty)}.pth"
        path_fisher_uniform = f"Saves_OBC_0.0_1.0_0.01/models_unstr/{arch}_{int(10000*goal_sparisty)}.pth"
        path_reconst_uniform = f"Saves_OBC_1.0_0.0_0.01/models_unstr/{arch}_{int(10000*goal_sparisty)}.pth"
        path_convex_non_uniform = f"Saves_OBC_1.0_1.0_0.01/results/{arch}_unstr_{int(100*goal_sparisty)}x_dp.pth"
        path_fisher_non_uniform = f"Saves_OBC_0.0_1.0_0.01/results/{arch}_unstr_{int(100*goal_sparisty)}x_dp.pth"
        path_reconst_non_uniform = f"Saves_OBC_1.0_0.0_0.01/results/{arch}_unstr_{int(100*goal_sparisty)}x_dp.pth"
        path_CAP_non_uniform = f"/home/gridsan/gafriat/projects/CAP-main/output/one-shot/{arch}_sparsity={goal_sparisty}.pth"
        path_convex_uniform_multiple = f"Saves_OBC_{n_convex}_0.01/models_unstr/{arch}_{int(10000*goal_sparisty)}.pth"
        path_convex_non_uniform_multiple = f"Saves_OBC_{n_convex}_0.01/results/{arch}_unstr_{int(100*goal_sparisty)}x_dp.pth"
        path_convex_non_uniform_multiple_greedy = f"Saves_OBC_{n_convex}_0.01/results/{arch}_unstr_{int(100*goal_sparisty)}x_greedy.pth"
        path_convex_uniform_block = f"Saves_OBC_1.0_1.0_0.01_block/models_unstr/{arch}_{int(10000*goal_sparisty)}.pth"
        path_fisher_uniform_block = f"Saves_OBC_0.0_1.0_0.01_block/models_unstr/{arch}_{int(10000*goal_sparisty)}.pth"
        path_reconst_uniform_block = f"Saves_OBC_1.0_0.0_0.01_block/models_unstr/{arch}_{int(10000*goal_sparisty)}.pth"
        path_convex_uniform_multiple_block = f"Saves_OBC_{n_convex}_0.01_block/models_unstr/{arch}_{int(10000*goal_sparisty)}.pth"

        model.load_state_dict(torch.load(path_convex_uniform, map_location=device))
        l_convex_uniform.append(get_acc(model, loader_train, device))
        model.load_state_dict(torch.load(path_fisher_uniform, map_location=device))
        l_fisher_uniform.append(get_acc(model, loader_train, device))
        model.load_state_dict(torch.load(path_reconst_uniform, map_location=device))
        l_reconst_uniform.append(get_acc(model, loader_train, device))
        model.load_state_dict(torch.load(path_convex_non_uniform, map_location=device))
        l_convex_non_uniform.append(get_acc(model, loader_train, device))
        model.load_state_dict(torch.load(path_fisher_non_uniform, map_location=device))
        l_fisher_non_uniform.append(get_acc(model, loader_train, device))
        model.load_state_dict(torch.load(path_reconst_non_uniform, map_location=device))
        l_reconst_non_uniform.append(get_acc(model, loader_train, device))
        if os.path.exists(path_CAP_non_uniform):
            model.load_state_dict(torch.load(path_CAP_non_uniform, map_location=device))
            l_CAP_non_uniform.append(get_acc(model, loader_train, device))
        else:
            l_CAP_non_uniform.append(np.nan)
        if os.path.exists(path_convex_uniform_multiple):
            model.load_state_dict(torch.load(path_convex_uniform_multiple, map_location=device))
            l_convex_uniform_multiple.append(get_acc(model, loader_train, device))
        else:
            l_convex_uniform_multiple.append(np.nan)
        if os.path.exists(path_convex_non_uniform_multiple):
            model.load_state_dict(torch.load(path_convex_non_uniform_multiple, map_location=device))
            l_convex_non_uniform_multiple.append(get_acc(model, loader_train, device))
        else:
            l_convex_non_uniform_multiple.append(np.nan)
        if os.path.exists(path_convex_non_uniform_multiple_greedy):
            model.load_state_dict(torch.load(path_convex_non_uniform_multiple_greedy, map_location=device))
            l_convex_non_uniform_multiple_greedy.append(get_acc(model, loader_train, device))
        else:
            l_convex_non_uniform_multiple_greedy.append(np.nan)
        if os.path.exists(path_convex_uniform_block):
            model.load_state_dict(torch.load(path_convex_uniform_block, map_location=device))
            l_convex_uniform_block.append(get_acc(model, loader_train, device))
        else:
            l_convex_uniform_block.append(np.nan)
        if os.path.exists(path_fisher_uniform_block):
            model.load_state_dict(torch.load(path_fisher_uniform_block, map_location=device))
            l_fisher_uniform_block.append(get_acc(model, loader_train, device))
        else:
            l_fisher_uniform_block.append(np.nan)
        if os.path.exists(path_reconst_uniform_block):
            model.load_state_dict(torch.load(path_reconst_uniform_block, map_location=device))
            l_reconst_uniform_block.append(get_acc(model, loader_train, device))
        else:
            l_reconst_uniform_block.append(np.nan)
        if os.path.exists(path_convex_uniform_multiple_block):
            model.load_state_dict(torch.load(path_convex_uniform_multiple_block, map_location=device))
            l_convex_uniform_multiple_block.append(get_acc(model, loader_train, device))
        else:
            l_convex_uniform_multiple_block.append(np.nan)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=l_sparsity, y=l_convex_non_uniform_multiple_greedy, name=f"OBC Convex Non-Uniform - {n_convex} Greedy"))
    fig.add_trace(go.Scatter(x=l_sparsity, y=l_convex_non_uniform_multiple, name=f"OBC Convex Non-Uniform - {n_convex}"))
    fig.add_trace(go.Scatter(x=l_sparsity, y=l_convex_non_uniform, name="OBC Convex Non-Uniform"))
    fig.add_trace(go.Scatter(x=l_sparsity, y=l_fisher_non_uniform, name="OBC Fisher Non-Uniform"))
    fig.add_trace(go.Scatter(x=l_sparsity, y=l_reconst_non_uniform, name="OBC Non-Uniform"))    

    fig.add_trace(go.Scatter(x=l_sparsity, y=l_convex_uniform_multiple, name=f"OBC Convex Uniform - {n_convex}", line = dict(dash='dot')))
    fig.add_trace(go.Scatter(x=l_sparsity, y=l_convex_uniform, name="OBC Convex Uniform", line = dict(dash='dot')))
    fig.add_trace(go.Scatter(x=l_sparsity, y=l_fisher_uniform, name="OBC Fisher Uniform", line = dict(dash='dot')))
    fig.add_trace(go.Scatter(x=l_sparsity, y=l_reconst_uniform, name="OBC Uniform", line = dict(dash='dot')))
    
    fig.add_trace(go.Scatter(x=l_sparsity, y=l_convex_uniform_multiple_block, name=f"OBC Convex Uniform Block - {n_convex}", line = dict(dash='dot')))
    fig.add_trace(go.Scatter(x=l_sparsity, y=l_convex_uniform_block, name="OBC Convex Uniform Block", line = dict(dash='dot')))
    fig.add_trace(go.Scatter(x=l_sparsity, y=l_fisher_uniform_block, name="OBC Fisher Uniform Block", line = dict(dash='dot')))
    fig.add_trace(go.Scatter(x=l_sparsity, y=l_reconst_uniform_block, name="OBC Uniform Block", line = dict(dash='dot')))

    fig.add_trace(go.Scatter(x=l_sparsity, y=l_CAP_non_uniform, name="CAP Non-Uniform", line = dict(dash='dash')))
    fig.update_layout(
        title=f"Results for {arch}",
        xaxis_title="Sparsity",
        yaxis_title="Accuracy",
    )
    fig.write_html(f"comparison_{arch}_train.html")

# %%
