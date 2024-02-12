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

def evaluate_sparsity(ind_new_sparsity, ind_layer, mat_costs, d_params, model, mat_is_exact, folder_saves_OBC, l_path_weights, l_layers, device, loader_train, criterion, arch):
    name_layer = l_layers[ind_layer]
    old_weights = copy.deepcopy(d_params[name_layer].data)
    weights_new_sparsity = torch.load(folder_saves_OBC+"/models_unstr/"+l_path_weights[ind_new_sparsity], map_location=device)[name_layer]
    # Try the pruned weights
    d_params[name_layer].data = weights_new_sparsity
    # Compute loss for pruned weights
    mat_costs[ind_new_sparsity][ind_layer] = get_loss(model, loader_train, criterion, arch, device)
    mat_is_exact[ind_new_sparsity][ind_layer] = True
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

def get_min_sparsity(l_n_params, l_current_sparsities, n_params, goal_sparsity, l_sparsities):
    min_sparsity_unseen_layers = np.nanmax([(1/np.sum(l_n_params[l_current_sparsities==0]))*(n_params*goal_sparsity-np.sum((l_n_params*l_sparsities[l_current_sparsities])[l_current_sparsities!=0])), 0.0])
    ind_min_sparsity = np.where(l_sparsities<=min_sparsity_unseen_layers)[0][-1]
    if ind_min_sparsity==0:
        ind_min_sparsity = 1
        min_sparsity_unseen_layers = l_sparsities[ind_min_sparsity]
    return min_sparsity_unseen_layers, ind_min_sparsity