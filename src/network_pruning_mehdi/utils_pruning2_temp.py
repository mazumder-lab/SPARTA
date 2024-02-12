import time

import torch
import torch.nn as nn

from Sparse_GPT_utils.quant import *
from Sparse_GPT_utils.sparsegpt import *
from Sparse_GPT_utils.modelutils import *
from OBC_utils.trueobs import *

from utils_model import Linear_with_z, Conv2d_with_z, use_mask_rec
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils_pruning_xiang import *
from utils_dataset import *
#from torch.func import functional_call, vmap, grad
from torch.nn.utils._per_sample_grad import call_for_per_sample_grads
from collections import defaultdict

def assign_weights_layer(name, layer, sparsities, sparsity, l_best_weights, sds, d_losses, mat_losses, device):
    if sds!=None:
        ind_goal = np.sum(np.array(sparsities)<=sparsity)-1
        W = l_best_weights[ind_goal]
        for ind_sparsity in range(len(sparsities)):
            tried_sparsity = sparsities[ind_sparsity]
            sds[tried_sparsity][name + '.weight'] = l_best_weights[ind_sparsity].reshape(sds[tried_sparsity][name + '.weight'].shape).cpu()
            d_losses[tried_sparsity][name + '.weight'] = torch.Tensor(mat_losses[:, ind_sparsity])
    else:
        W = l_best_weights[0]

    if isinstance(layer, transformers.Conv1D):
        W = W.t()

    layer.weight.data = W.reshape(layer.weight.shape).to(device)
    

def find_best_weights_layer(pruner, ind_convex, arch, l_best_weights, W_s, l_sparsities, mat_losses, criterion, loader_tmp, device):
    W_copy = pruner.layer.weight.data.clone()
    if pruner.n_convex!=-1:
        training_mode = pruner.current_model.training
        pruner.current_model.eval()
        for ind_sparsity in range(len(l_sparsities)):
            tried_sparsity = l_sparsities[ind_sparsity]
            W = W_s[ind_sparsity]
            # if tried_sparsity==0.0:
            #     try:
            #         print(saved_weights)
            #         import ipdb;ipdb.set_trace()
            #     except:
            #         saved_weights = copy.deepcopy(W)

            pruner.layer.weight.data = W.reshape(W_copy.shape).to(device)
            # Compute loss for pruned weights
            loss_convex = 0
            n_seen_loader = 0
            with torch.no_grad():
                for batch_sgd in tqdm(loader_tmp):
                    input_batch_sgd, input_batch_original_sgd, target_batch_sgd, index_seen_sgd, old_index_seen_sgd = read_batch(batch_sgd)
                    final_out = pruner.current_model(input_batch_sgd.to(device))
                    if "opt" in arch:
                        # with torch.autocast(device_type=model_wrapper.device, dtype=torch.float16):
                        shift_logits = final_out[:, :-1, :].contiguous()
                        shift_labels = target_batch_sgd[:, 1:].contiguous()
                        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                        loss = criterion(shift_logits, shift_labels.view(-1).to(device))
                    else:
                        loss = criterion(final_out, target_batch_sgd.to(device))
                    loss_convex+=loss.item()*target_batch_sgd.shape[0]
                    n_seen_loader+=target_batch_sgd.shape[0]
            loss_convex /= n_seen_loader
            print(f"Loss for sparsity = {tried_sparsity}, (lambda_reconst, lambda_fisher)={(lambda_reconst, lambda_fisher)}:", loss_convex)
            old_best_loss = np.min(mat_losses[:, ind_sparsity])
            if loss_convex < old_best_loss:
                l_best_weights[ind_sparsity] = W.cpu()
            mat_losses[ind_convex, ind_sparsity] = loss_convex
        if training_mode:
            pruner.current_model.train()
        # Reset original weights
        pruner.layer.weight.data = W_copy.reshape(pruner.layer.weight.shape).to(device)
        # print(f"--- Weights for sparisty = {sparsities[0]}:", W_s[0])
    else:
        l_best_weights = W_s
