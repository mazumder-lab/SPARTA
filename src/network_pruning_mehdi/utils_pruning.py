import time

import torch
import torch.nn as nn

from OBC_utils.trueobs import *

from utils_model import Linear_with_z, Conv2d_with_z, use_mask_rec
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils_pruning_xiang import *
from utils_dataset import *
#from torch.func import functional_call, vmap, grad
from torch.nn.utils._per_sample_grad import call_for_per_sample_grads
from collections import defaultdict
import transformers

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=""):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + "." + name1 if name != "" else name1))
    return res

def assign_weights_layer(name, layer, sparsities, sparsity, l_best_weights, sds, d_losses, mat_losses, device):
    if sds!=None:
        ind_goal = np.sum(np.array(sparsities)<=sparsity)-1
        W = l_best_weights[ind_goal]
        for ind_sparsity in range(len(sparsities)):
            tried_sparsity = sparsities[ind_sparsity]
            sds[tried_sparsity][name + '.weight'] = l_best_weights[ind_sparsity].reshape(sds[tried_sparsity][name + '.weight'].shape).cpu()
            if d_losses!=None:
                d_losses[tried_sparsity][name + '.weight'] = torch.Tensor(mat_losses[:, ind_sparsity])
    else:
        W = l_best_weights[0]

    if isinstance(layer, transformers.Conv1D):
        W = W.t()

    layer.weight.data = W.reshape(layer.weight.shape).to(device)

def find_best_weights_layer(l_pruners, ind_convex, arch, l_best_weights, l_W_s, l_sparsities, mat_losses, criterion, loader_tmp, lambda_reconst, lambda_fisher, device):
    l_W_copy = [pruner.layer.weight.data.clone() for pruner in l_pruners]
    if l_pruners[0].n_convex!=-1:
        l_training_modes = [pruner.current_model.training for pruner in l_pruners]
        for pruner in l_pruners:
            pruner.current_model.eval()
        for ind_sparsity in range(len(l_sparsities)):
            tried_sparsity = l_sparsities[ind_sparsity]
            for ind_pruner in range(len(l_pruners)):
                W = l_W_s[ind_pruner][ind_sparsity]
                # if tried_sparsity==0.0:
                #     try:
                #         print(saved_weights)
                #         import ipdb;ipdb.set_trace()
                #     except:
                #         saved_weights = copy.deepcopy(W)

                l_pruners[ind_pruner].layer.weight.data = W.reshape(l_W_copy[ind_pruner].shape).to(device)
            # Compute loss for pruned weights
            loss_convex = 0
            n_seen_loader = 0
            with torch.no_grad():
                for batch_sgd in tqdm(loader_tmp):
                    input_batch_sgd, input_batch_original_sgd, target_batch_sgd, index_seen_sgd, old_index_seen_sgd = read_batch(batch_sgd)
                    final_out = l_pruners[0].current_model(input_batch_sgd.to(device))
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
                for ind_pruner in range(len(l_pruners)):
                    l_best_weights[ind_pruner][ind_sparsity] = l_W_s[ind_pruner][ind_sparsity].cpu()
            mat_losses[ind_convex, ind_sparsity] = loss_convex
        for ind_pruner in range(len(l_pruners)):
            if l_training_modes[ind_pruner]:
                l_pruners[ind_pruner].current_model.train()
            # Reset original weights
            l_pruners[ind_pruner].layer.weight.data = l_W_copy[ind_pruner].reshape(l_pruners[ind_pruner].layer.weight.shape).to(device)
            # print(f"--- Weights for sparisty = {sparsities[0]}:", W_s[0])
    else:
        l_best_weights = l_W_s
    return l_best_weights

def prune_blocked(pruner, ind_convex, sparsities):
    # Code from the OBC paper
    parallel = pruner.Traces[ind_convex][0].shape[1]
    blockcount = pruner.Traces[ind_convex][0].shape[0] - 1
    losses = pruner.Losses[ind_convex, :, 1:].reshape(-1)
    order = torch.argsort(losses)
    Ws = [torch.zeros((pruner.rows, pruner.columns), device=pruner.dev) for _ in sparsities]
    losses = [0] * len(sparsities) 
    for i in range(pruner.rows):
        if i % parallel == 0:
            Trace = pruner.Traces[ind_convex][i // parallel].to(pruner.dev)
        for j, sparsity in enumerate(sparsities):
            count = int(math.ceil(pruner.rows * blockcount * sparsity))
            perrow = torch.sum(
                torch.div(order[:count], blockcount, rounding_mode='trunc') == i
            ).item()
            losses[j] += torch.sum(pruner.Losses[ind_convex, i, :(perrow + 1)]).item()
            Ws[j][i, :] = Trace[perrow, i % parallel, :]
    for sparsity, loss in zip(sparsities, losses):
        print('%.4f error' % sparsity, loss)
        if DEBUG:
            tmp = pruner.layer.weight.data.clone()
            pruner.layer.weight.data = Ws[sparsities.index(sparsity)].reshape(pruner.layer.weight.shape) 
            print(torch.sum((pruner.layer(pruner.inp1) - pruner.out1) ** 2) / 128)
            pruner.layer.weight.data = tmp
    return Ws

class Pruner:

    def __init__(self, layer, algo_pruning, lambda_fisher, lambda_reconst, name, n_convex, current_model, pruning_level):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d) or isinstance(self.layer, Conv2d_with_z):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        #self.per_sample_grad = torch.zeros([0]+list(W.shape))
        self.fisher_hessian = torch.zeros(list(W.shape)+[self.columns]).double()
        self.algo_pruning = algo_pruning
        self.n_convex = n_convex
        if self.n_convex == -1:
            self.l_lambda_fisher = [lambda_fisher]
            self.l_lambda_reconst = [lambda_reconst]
        else:
            self.l_lambda_fisher = np.linspace(0, 1, n_convex)
            self.l_lambda_reconst = 1 - np.linspace(0, 1, n_convex)
            
        self.name = name
        self.current_model = current_model
        self.pruning_level = pruning_level

    def add_batch(self, inp, out, blocksize=1024):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        # if len(inp.shape) == 2:
        #     inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, Linear_with_z) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d) or isinstance(self.layer, Conv2d_with_z):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        #inp = math.sqrt(2 / self.nsamples) * inp.float()
        #self.H += inp.matmul(inp.t())
        self.H += 2 / self.nsamples * (inp.matmul(inp.t())).double()

    def prepare_pruning(self, i1, parallel, W_original, W, device, reconst_0_per_c, fisher_0_per_c, lambda_reconst, lambda_fisher):
        i2 = min(i1 + parallel, self.rows)
        count = i2 - i1
        w_old = W_original[i1:i2, :].double()
        mat_hessian = (lambda_reconst/torch.sum(reconst_0_per_c))*self.H.unsqueeze(0).repeat((count, 1, 1)).double()
        # mat_hessian = (lambda_reconst)*self.H.unsqueeze(0).repeat((count, 1, 1)).double()
        mask = torch.zeros_like(W[i1:i2, :]).bool()
        # mask[W[i1:i2, :]==0] = True
        if lambda_fisher!=0:
            #gradients_block = self.per_sample_grad[:,i1:i2,:].double().to(device)
            # try:
            #     # mat_hessian += (lambda_fisher/torch.sum(fisher_0_per_c))*torch.einsum("klm,klp->lmp", gradients_block, gradients_block)/gradients_block.shape[0]
            #     mat_hessian += torch.einsum("klm,klp->lmp", gradients_block, (lambda_fisher/(torch.sum(fisher_0_per_c).item()*gradients_block.shape[0]))*gradients_block)
            # except:
            #     import ipdb;ipdb.set_trace()
            mat_hessian +=  (lambda_fisher/torch.sum(fisher_0_per_c))*self.fisher_hessian[i1:i2,:].to(device)
        # Change dead ones to 0 (no impact on hessian)
        # deads_H = (torch.eye(self.columns, device=device)[None]*(mat_hessian==0)).bool()
        # if torch.sum(deads_H).item()>0:
        #     import ipdb;ipdb.set_trace()
        # mat_hessian[deads_H] = 1
        deads_W = mat_hessian[:,torch.eye(self.columns, device=device).bool()]==0
        w_old[deads_W] = 0
        # if torch.sum(deads_W).item()>0:
        #     import ipdb;ipdb.set_trace()
        mask[w_old==0] = True
        return i2, count, w_old, mat_hessian, mask

    def prepare_pruning_sparse_gpt(self, i1, parallel, W_original, W, device, reconst_0_per_c, fisher_0_per_c, lambda_reconst, lambda_fisher):
        i2 = min(i1 + parallel, self.rows)
        count = i2 - i1
        w_old = W_original[:, i1:i2].double()
        mat_hessian = (lambda_reconst/torch.sum(reconst_0_per_c))*self.H[i1:i2,i1:i2].unsqueeze(0).repeat((w_old.shape[0], 1, 1)).double()
        mask = torch.zeros_like(W[:, i1:i2]).bool()
        mask[W[:, i1:i2]==0] = True
        import ipdb;ipdb.set_trace()
        gradients_block = self.per_sample_grad[:,:,i1:i2].double().to(device)
        mat_hessian += (lambda_fisher/torch.sum(fisher_0_per_c))*torch.einsum("klm,klp->lmp", gradients_block, gradients_block)/gradients_block.shape[0]
        # Change dead ones to 0 (no impact on hessian)
        deads_H = (torch.eye(mat_hessian.shape[1], device=device)[None]*(mat_hessian==0)).bool()
        mat_hessian[deads_H] = 1
        deads_W = mat_hessian[:,torch.eye(mat_hessian.shape[1], device=device).bool()]==0
        w_old[deads_W] = 0
        return i2, count, w_old, mat_hessian, mask

    def pruning(
        self, sparsity, prunen=0, prunem=0, parallel=128, lambda_stability=.01, device="cpu", sparsities=None, sds=None, loader_tmp=None, arch=None, criterion=None, d_losses=None, test_prune=True
    ):
        if self.n_convex == -1:
            n_convex_iter = 1
        else:
            n_convex_iter = self.n_convex

        if sds==None:
            l_sparsities = [sparsity]
            n_sparsities = 1
        else:
            l_sparsities = sparsities
            n_sparsities = len(sparsities)

        # d_best_weights = {}
        # d_best_losses = {}
        mat_losses = np.ones((n_convex_iter, n_sparsities))*np.inf
        l_best_weights = [None]*n_sparsities

        # if sds!=None:
        #     for goal_sparsity in sparsities:
        #         d_best_weights[goal_sparsity] = None
        #         d_best_losses[goal_sparsity] = np.inf
        # else:
        #     d_best_weights[sparsity] = None
        #     d_best_losses[sparsity] = np.inf
        if self.algo_pruning=="obc_convex":
            self.Losses = torch.zeros([n_convex_iter, self.rows, self.columns + 1], device=self.dev)
            self.Traces = {}
            for ind_convex in range(n_convex_iter):
                self.Traces[ind_convex] = []

        for ind_convex in range(n_convex_iter):
            lambda_reconst = self.l_lambda_reconst[ind_convex]
            lambda_fisher = self.l_lambda_fisher[ind_convex]
            W = self.layer.weight.data.clone()
            W_original = self.layer.weight.data.clone()
            if isinstance(self.layer, nn.Conv2d) or isinstance(self.layer, Conv2d_with_z) :
                W = W.flatten(1)
                W_original = W_original.flatten(1)
            if isinstance(self.layer, transformers.Conv1D):
                W = W.t()
                W_original = W_original.t()
            W = W.float()
            W_original = W_original.float()

            if self.algo_pruning=="mp_convex":
                tick = time.time()
                indices_to_prune = torch.argsort(torch.abs(W.data.flatten()))
                W.data.flatten()[indices_to_prune[:int(np.ceil(len(indices_to_prune)*sparsity))]] = 0
                reconst_0_per_c = torch.einsum("km,mp,kp->k", W_original, self.H, W_original)
                fisher_0_per_c = torch.einsum("lm,blm,blp,lp->l", W_original, self.per_sample_grad.to(device), self.per_sample_grad.to(device), W_original)/self.per_sample_grad.shape[0]
                for i1 in range(0, self.rows, parallel):
                    i2, count, w_old, mat_hessian, mask = self.prepare_pruning(i1, parallel, W_original, W, device, reconst_0_per_c, fisher_0_per_c, lambda_reconst, lambda_fisher)
                    # End
                    mat_hessian_left = torch.einsum("lmp,lm->lmp", mat_hessian, mask.logical_not())
                    mat_hessian_left_right = torch.einsum("lmp,lp->lmp", mat_hessian_left, mask.logical_not())
                    right_side = torch.bmm(mat_hessian_left, w_old.unsqueeze(-1)).squeeze(-1)
                    #W[i1:i2, :] = torch.linalg.solve(mat_hessian_left_right + lambda_stability*torch.eye(self.columns, device=device)[None], right_side)
                    to_add = torch.mean(lambda_stability*(mat_hessian_left_right*(torch.eye(self.columns, device=device)[None])), (1,2))
                    to_add = torch.eye(self.columns, device=device)[None]*to_add[:, None, None]
                    W[i1:i2, :] = torch.linalg.solve(mat_hessian_left_right + to_add, right_side)
                # End
                # new_W = back_solve_regu(self.layer.weight.data.clone().to("cpu"), W.to("cpu"), self.per_sample_grad, lambda2, self.H.to("cpu"), linear_term.to("cpu")/2, lambda3 = lambda3)
                # new_W = torch.Tensor(new_W)
                if self.dev not in ("cpu", torch.device("cpu")):
                    torch.cuda.synchronize()
                print('time %.2f' % (time.time() - tick))
                W_s = [W]

            if self.algo_pruning=="obc_convex":
                tick = time.time()
                # self.Losses = torch.zeros([self.rows, self.columns + 1], device=self.dev)
                # self.Traces = []
                reconst_0_per_c = torch.einsum("km,mp,kp->k", W_original, self.H, W_original)
                try:
                    fisher_0_per_c = torch.einsum("km,kmp,kp->k", W_original, self.fisher_hessian.float().to(device), W_original)
                except:
                    fisher_0_per_c = torch.einsum("km,kmp,kp->k", W_original.to("cpu"), self.fisher_hessian.float(), W_original.to("cpu")).to(device)
                # import ipdb;ipdb.set_trace()
                # try:
                #     fisher_0_per_c = torch.einsum("lm,blm,blp,lp->l", W_original, self.per_sample_grad.to(device), self.per_sample_grad.to(device), W_original)/self.per_sample_grad.shape[0]
                # except:
                #     import ipdb;ipdb.set_trace()
                for i1 in range(0, self.rows, parallel):
                    i2, count, w_old, mat_hessian, mask = self.prepare_pruning(i1, parallel, W_original, W, device, reconst_0_per_c, fisher_0_per_c, lambda_reconst, lambda_fisher)
                    rangecount = torch.arange(count, device=self.dev)
                    # Add for stability
                    to_add = lambda_stability*torch.mean(torch.diagonal(mat_hessian, dim1=1, dim2=2), 1)
                    to_add = torch.eye(self.columns, device=device)[None]*to_add[:, None, None]
                    mat_hessian += to_add
                    # Check for prunable rows in w_old -> setting corresponding hessians to I:
                    idx_0_rows = torch.where(torch.max(torch.abs(w_old), 1).values==0)[0]
                    mat_hessian[idx_0_rows]+=torch.eye(self.columns, device=device)[None]
                    # Invert hessian
                    mat_hessian = torch.cholesky_inverse(torch.linalg.cholesky(mat_hessian))
                    # Code from OBC
                    start = int(torch.min(torch.sum((w_old == 0).float(), 1)).item()) + 1
                    Trace = torch.zeros((self.columns + 1, count, self.columns), device=self.dev)
                    Trace[0, :, :] = w_old
                    Trace[:start, :, :] = w_old
                    for zeros in range(start, self.columns + 1):
                        diag = torch.diagonal(mat_hessian, dim1=1, dim2=2)
                        scores = (w_old ** 2) / diag
                        scores[mask] = float('inf')
                        j = torch.argmin(scores, 1)
                        self.Losses[ind_convex, i1:i2, zeros] = scores[rangecount, j]
                        row = mat_hessian[rangecount, j, :]
                        d = diag[rangecount, j]
                        w_old -= row * (w_old[rangecount, j] / d).unsqueeze(1)
                        mask[rangecount, j] = True
                        w_old[mask] = 0
                        Trace[zeros, :, :] = w_old
                        if zeros == self.columns:
                            break
                        row /= torch.sqrt(d).unsqueeze(1)
                        mat_hessian -= torch.bmm(row.unsqueeze(2), row.unsqueeze(1))
                    self.Losses[ind_convex, i1:i2, :] /= 2
                    self.Traces[ind_convex].append(Trace.cpu())
                if self.dev not in ("cpu", torch.device("cpu")):
                    torch.cuda.synchronize()
                print('time %.2f' % (time.time() - tick))

                if self.pruning_level=="layer" and test_prune:
                    W_s = prune_blocked(self, ind_convex, l_sparsities)
                    # The following function updates l_best_weights and mat_losses:
                    l_best_weights = find_best_weights_layer([self], ind_convex, arch, [l_best_weights], [W_s], l_sparsities, mat_losses, criterion, loader_tmp, lambda_reconst, lambda_fisher, device)[0]

                # W_copy = self.layer.weight.data.clone()
                # if self.n_convex!=-1:
                #     training_mode = self.current_model.training
                #     self.current_model.eval()
                #     for ind_sparsity in range(len(l_sparsities)):
                #         tried_sparsity = l_sparsities[ind_sparsity]
                #         W = W_s[ind_sparsity]
                #         # if tried_sparsity==0.0:
                #         #     try:
                #         #         print(saved_weights)
                #         #         import ipdb;ipdb.set_trace()
                #         #     except:
                #         #         saved_weights = copy.deepcopy(W)

                #         self.layer.weight.data = W.reshape(W_copy.shape).to(device)
                #         # Compute loss for pruned weights
                #         loss_convex = 0
                #         n_seen_loader = 0
                #         with torch.no_grad():
                #             for batch_sgd in tqdm(loader_tmp):
                #                 input_batch_sgd, input_batch_original_sgd, target_batch_sgd, index_seen_sgd, old_index_seen_sgd = read_batch(batch_sgd)
                #                 final_out = self.current_model(input_batch_sgd.to(device))
                #                 if "opt" in arch:
                #                     # with torch.autocast(device_type=model_wrapper.device, dtype=torch.float16):
                #                     shift_logits = final_out[:, :-1, :].contiguous()
                #                     shift_labels = target_batch_sgd[:, 1:].contiguous()
                #                     shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                #                     loss = criterion(shift_logits, shift_labels.view(-1).to(device))
                #                 else:
                #                     loss = criterion(final_out, target_batch_sgd.to(device))
                #                 loss_convex+=loss.item()*target_batch_sgd.shape[0]
                #                 n_seen_loader+=target_batch_sgd.shape[0]
                #         loss_convex /= n_seen_loader
                #         print(f"Loss for sparsity = {tried_sparsity}, (lambda_reconst, lambda_fisher)={(lambda_reconst, lambda_fisher)}:", loss_convex)
                #         old_best_loss = np.min(mat_losses[:, ind_sparsity])
                #         if loss_convex < old_best_loss:
                #             l_best_weights[ind_sparsity] = W.cpu()
                #         mat_losses[ind_convex, ind_sparsity] = loss_convex
                #     if training_mode:
                #         self.current_model.train()
                #     # Reset original weights
                #     self.layer.weight.data = W_copy.reshape(self.layer.weight.shape).to(device)
                #     # print(f"--- Weights for sparisty = {sparsities[0]}:", W_s[0])
                # else:
                #     l_best_weights = W_s
                    
            if self.algo_pruning=="sparse_gpt_convex":
                tick = time.time()
                reconst_0_per_c = torch.einsum("km,mp,kp->k", W_original, self.H, W_original)
                fisher_0_per_c = torch.einsum("lm,blm,blp,lp->l", W_original, self.per_sample_grad.to(device), self.per_sample_grad.to(device), W_original)/self.per_sample_grad.shape[0]
                for i1 in range(0, self.rows, parallel):
                    i2, count, w_old, mat_hessian, mask = self.prepare_pruning_sparse_gpt(i1, parallel, W_original, W, device, reconst_0_per_c, fisher_0_per_c, lambda_reconst, lambda_fisher)
                    rangecount = torch.arange(count, device=self.dev)
                    # Add for stability
                    to_add = torch.mean(lambda_stability*(mat_hessian*(torch.eye(mat_hessian.shape[1], device=device)[None])), (1,2))
                    to_add = torch.eye(mat_hessian.shape[1], device=device)[None]*to_add[:, None, None]
                    mat_hessian += to_add
                    # Invert hessian
                    mat_hessian = torch.cholesky_inverse(torch.linalg.cholesky(mat_hessian))

                    import ipdb;ipdb.set_trace()
                    Q1 = torch.zeros_like(w_old)
                    Err1 = torch.zeros_like(w_old)
                    Losses1 = torch.zeros_like(w_old)

                    diag = torch.diagonal(mat_hessian, dim1=1, dim2=2)
                    tmp = (w_old ** 2) / diag
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh

                    for i in range(w_old.shape[1]):
                        w = w_old[:, i]
                        d = Hinv1[i, i]

                        if prunen != 0 and i % prunem == 0:
                            tmp = w_old[:, i:(i + prunem)] ** 2 / (torch.diag(Hinv1)[i:(i + prunem)].reshape((1, -1))) ** 2
                            mask1.scatter_(1, i + torch.topk(tmp, prunen, dim=1, largest=False)[1], True)

                        q = w.clone()
                        q[mask1[:, i]] = 0

                        if hasattr(self, 'quantizer'):
                            q = quantize(
                                q.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                            ).flatten()

                        Q1[:, i] = q
                        Losses1[:, i] = (w - q) ** 2 / d ** 2

                        err1 = (w - q) / d
                        w_old[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                        Err1[:, i] = err1

                    W[:, i1:i2] = Q1
                    Losses += torch.sum(Losses1, 1) / 2

                    W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])


                if self.dev not in ("cpu", torch.device("cpu")):
                    torch.cuda.synchronize()
                print('time %.2f' % (time.time() - tick))
                W = prune_blocked(self, ind_convex, [sparsity])[0]
        
        if self.pruning_level == "layer" and test_prune:
            assign_weights_layer(self.name, self.layer, sparsities, sparsity, l_best_weights, sds, d_losses, mat_losses, device)
        # if sds!=None:
        #     ind_goal = np.sum(np.array(sparsities)<=sparsity)-1
        #     W = l_best_weights[ind_goal]
        #     for ind_sparsity in range(len(sparsities)):
        #         tried_sparsity = sparsities[ind_sparsity]
        #         sds[tried_sparsity][self.name + '.weight'] = l_best_weights[ind_sparsity].reshape(sds[tried_sparsity][self.name + '.weight'].shape).cpu()
        #         d_losses[tried_sparsity][self.name + '.weight'] = torch.Tensor(mat_losses[:, ind_sparsity])
        # else:
        #     W = l_best_weights[0]

        # if isinstance(self.layer, transformers.Conv1D):
        #     W = W.t()

        # self.layer.weight.data = W.reshape(self.layer.weight.shape).to(device)
        
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.loader_tmp = None
        self.fisher_hessian = None
        if self.dev not in ("cpu", torch.device("cpu")):
            torch.cuda.empty_cache()

#
def zero_grad_sample(self, set_to_none: bool = True):
    # From pytorch
    r"""Sets the gradients of all optimized :class:`torch.Tensor` s to zero.

    Args:
        set_to_none (bool): instead of setting to zero, set the grads to None.
            This will in general have lower memory footprint, and can modestly improve performance.
            However, it changes certain behaviors. For example:
            1. When the user tries to access a gradient and perform manual ops on it,
            a None attribute or a Tensor full of 0s will behave differently.
            2. If the user requests ``zero_grad(set_to_none=True)`` followed by a backward pass, ``.grad``\ s
            are guaranteed to be None for params that did not receive a gradient.
            3. ``torch.optim`` optimizers have a different behavior if the gradient is 0 or None
            (in one case it does the step with a gradient of 0 and in the other it skips
            the step altogether).
    """
    foreach = self.defaults.get('foreach', False)

    if not hasattr(self, "_zero_grad_profile_name"):
        self._patch_step_function()
    if foreach:
        per_device_and_dtype_grads = defaultdict(lambda: defaultdict(list))
    with torch.autograd.profiler.record_function(self._zero_grad_profile_name):
        for group in self.param_groups:
            for p in group['params']:
                try:
                    test_grad_sample = p.grad_sample is not None
                except:
                    test_grad_sample = False
                if test_grad_sample:
                    if set_to_none:
                        p.grad_sample = None
                    else:
                        if p.grad_sample.grad_fn is not None:
                            p.grad_sample.detach_()
                        else:
                            p.grad_sample.requires_grad_(False)
                        if (not foreach or p.grad_sample.is_sparse):
                            p.grad_sample.zero_()
                        else:
                            per_device_and_dtype_grads[p.grad_sample.device][p.grad_sample.dtype].append(p.grad_sample)
        if foreach:
            for _, per_dtype_grads in per_device_and_dtype_grads.items():
                for grads in per_dtype_grads.values():
                    torch._foreach_zero_(grads)


def prune_block(block, loader_train, dev, argssparsity, argsprunen, argsprunem, argsblocksize, with_z, gamma, end_model, arch, algo_pruning, rel_damp, lambda_fisher, lambda_reconst, sparsities, sds, ind_block, n_convex, d_losses, pruning_level, n_layers, gpts_total, test_prune = True):
    print('Starting ...')
    print('Ready.')

    update_layer_wise = True
    update_gradients = algo_pruning not in ["sparse_gpt", "obc"] and not(lambda_fisher==0 and n_convex==-1)

    old_subset = find_layers(block, layers = [nn.Conv2d, nn.Linear, Linear_with_z, Conv2d_with_z])
    subset = {}
    for key_layer in old_subset:
        if old_subset[key_layer].weight.requires_grad:
            subset[key_layer] = old_subset[key_layer]

    gpts = {}
    if algo_pruning=="sparse_gpt":
        for name in subset:
            gpts[name] = SparseGPT(subset[name])
    elif algo_pruning=="obc":
        for name in subset:
            gpts[name] = TrueOBS(subset[name], rel_damp=rel_damp)
    else:
        for name in subset:
            if "deit" in arch:
                name_block = f"blocks.{ind_block}.{name}"
            elif "mlpnet" in arch:
                name_block = f"fc{ind_block+1}"
            elif "resnet" in arch:
                name_block = subset[name].weight.name_in_model
            current_model = None
            gpts[name] = Pruner(subset[name], algo_pruning, lambda_fisher, lambda_reconst, name_block, n_convex, current_model, pruning_level)

    def add_batch(name):
        def tmp(_, inp, out):
            gpts[name].add_batch(inp[0].data, out.data)
        return tmp
    handles = []
    temp_optimizer = torch.optim.SGD(block.parameters(), 0.0, 1.0)
    criterion = torch.nn.functional.cross_entropy

    loader_tmp = DataLoader(loader_train.dataset, batch_size=loader_train.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    if update_gradients:
        use_mask_rec(block, False)
        for param_block in block.named_parameters():
            if "z" in param_block[0]:
                param_block[1].requires_grad = False
        current_model = nn.Sequential(*[block, end_model])
        for name in gpts:
            gpts[name].current_model = current_model
            gpts[name].loader_tmp = loader_tmp
        # Attempt to make code faster:
        # for name in gpts:
            #gpts[name].per_sample_grad = torch.zeros([len(loader_tmp.dataset)]+list(gpts[name].per_sample_grad.shape[1:]))
        # End attempt
        # # Old 
        # loader_tmp = DataLoader(loader_train.dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        # # End Old
        
    if update_layer_wise:
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
    acc_1 = 0
    acc_2 = 0
    for batch_sgd in tqdm(loader_tmp):
        if update_layer_wise or update_gradients:
            input_batch_sgd, input_batch_original_sgd, target_batch_sgd, index_seen_sgd, old_index_seen_sgd = read_batch(batch_sgd)
            acc_1 = acc_2
            acc_2 += input_batch_sgd.shape[0]
            if not (update_gradients):
                out = block(input_batch_sgd.to(dev))
        if update_gradients:
            final_out = call_for_per_sample_grads(current_model)(input_batch_sgd.to(dev))
            if "opt" in arch:
                # with torch.autocast(device_type=model_wrapper.device, dtype=torch.float16):
                shift_logits = final_out[:, :-1, :].contiguous()
                shift_labels = target_batch_sgd[:, 1:].contiguous()
                shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                loss = criterion(shift_logits, shift_labels.view(-1).to(dev))
            else:
                loss = criterion(final_out, target_batch_sgd.to(dev))
            loss.backward()
            for name in gpts:
                grad_W = input_batch_sgd.shape[0]*gpts[name].layer.weight.grad_sample.cpu()
                if isinstance(gpts[name].layer, nn.Conv2d) or isinstance(gpts[name].layer, Conv2d_with_z):
                    grad_W = grad_W.flatten(2)
                if isinstance(gpts[name].layer, transformers.Conv1D):
                    grad_W = grad_W.t()
                #gpts[name].per_sample_grad[acc_1:acc_2] = grad_W
                grad_W = grad_W.double()
                # grad_W = grad_W.reshape(-1, gpts[name].fisher_hessian.shape[0], gpts[name].fisher_hessian.shape[1])
                # import ipdb; ipdb.set_trace()
                gpts[name].fisher_hessian += torch.einsum("klm,klp->lmp", grad_W, grad_W)

            zero_grad_sample(temp_optimizer)

    if update_gradients:
        use_mask_rec(block, True)  
        for param_block in block.named_parameters():
            if "z" in param_block[0]:
                param_block[1].requires_grad = True
                
        for name in gpts:
            gpts[name].fisher_hessian /= len(loader_tmp.dataset)


    if update_layer_wise:
        for h in handles:
            h.remove()
    # if test_update_dataset:
    #     update_dataset(*args_update_dataset)

    for name in gpts:
        print(name)
        print('Pruning ...')
        sparsity = argssparsity
        if algo_pruning == "sparse_gpt":
            gpts[name].fasterprune(
                sparsity, prunen=argsprunen, prunem=argsprunem, percdamp=rel_damp, blocksize=argsblocksize
            )
        elif algo_pruning == "obc":
            #import ipdb;ipdb.set_trace()
            gpts[name].prepare_unstr()
            with torch.no_grad():
                gpts[name].layer.weight.data = gpts[name].prune_unstr([sparsity])[0]
        else:
            #import ipdb;ipdb.set_trace()
            gpts[name].pruning(
                sparsity, prunen=argsprunen, prunem=argsprunem, lambda_stability=rel_damp, parallel=argsblocksize, device=dev, sparsities=sparsities, sds=sds, loader_tmp = loader_tmp, arch=arch, criterion=criterion, d_losses=d_losses, test_prune=test_prune
            )
        gpts[name].free()

    if test_prune:
        if pruning_level == "block":
            if n_convex == -1:
                n_convex_iter = 1
            else:
                n_convex_iter = n_convex

            l_layers = list(gpts.keys())
            new_sparsities = np.zeros((n_convex_iter, len(l_layers), len(sparsities)))
            new_l_layers = []
            for ind_layer in range(len(l_layers)):
                if ind_layer % n_layers ==0:
                    if ind_layer != 0:
                        new_l_layers.append(temp_l_layers)
                    temp_l_layers = []
                temp_l_layers.append(l_layers[ind_layer])
            new_l_layers.append(temp_l_layers)
            print("Lists of layers used for new sparsities:", new_l_layers)

            for ind_convex in range(n_convex_iter):
                acc_layer = 0
                for l_layers_temp in new_l_layers:
                    mat_losses_global = torch.zeros((2,0), device = dev)
                    for ind_layer in range(len(l_layers_temp)):
                        name_layer = l_layers_temp[ind_layer]
                        l_losses = gpts[name_layer].Losses[ind_convex,:,1:].flatten()
                        l_losses = torch.stack([torch.ones_like(l_losses)*(acc_layer+ind_layer), l_losses])
                        mat_losses_global = torch.hstack([mat_losses_global, l_losses])

                    l_idx = torch.argsort(mat_losses_global[1])
                    
                    for ind_sparsity in range(len(sparsities)):
                        target_sparsity = sparsities[ind_sparsity]
                        n_params_to_prune = int(np.ceil(len(l_idx)*target_sparsity))
                        idx_to_prune = mat_losses_global[0, l_idx[:n_params_to_prune]]
                        for ind_layer in range(len(l_layers_temp)):
                            n_layer = torch.sum((mat_losses_global[0]==(acc_layer+ind_layer)).float()).item()
                            new_sparsities[ind_convex, (acc_layer+ind_layer), ind_sparsity] = torch.sum((idx_to_prune==(acc_layer+ind_layer)).float()).item()/n_layer
                    acc_layer += len(l_layers_temp)
            mat_losses = np.ones((n_convex_iter, len(sparsities)))*np.inf
            l_best_weights = [[None]*len(sparsities) for _ in range(len(l_layers))]

            l_pruners = [gpts[l_layers[ind_layer]] for ind_layer in range(len(l_layers))]
            for ind_convex in range(n_convex_iter):
                l_W_s = []
                for ind_layer in range(len(l_layers)):
                    name_layer = l_layers[ind_layer]
                    W_s = prune_blocked(gpts[name_layer], ind_convex, new_sparsities[ind_convex, ind_layer])
                    l_W_s.append(W_s)
                # The following function updates l_best_weights and mat_losses:
                l_best_weights = find_best_weights_layer(l_pruners, ind_convex, arch, l_best_weights, l_W_s, sparsities, mat_losses, criterion, loader_tmp, l_pruners[0].l_lambda_reconst[ind_convex], l_pruners[0].l_lambda_fisher[ind_convex], dev)
            
            for ind_layer in range(len(l_layers)):
                name_layer = gpts[l_layers[ind_layer]].name
                layer = gpts[l_layers[ind_layer]].layer
                assign_weights_layer(name_layer, layer, sparsities, sparsity, l_best_weights[ind_layer], sds, d_losses, mat_losses, dev)
                
        for name in gpts:
            if with_z:
                try:
                    gpts[name].layer.weight_z.data[gpts[name].layer.weight==0] = -gamma
                    gpts[name].layer.weight_z.data[gpts[name].layer.weight!=0] = gamma
                except:
                    pass

    for name in gpts:
        if "deit" in arch:
            name_block = f"blocks.{ind_block}.{name}"
        elif "mlpnet" in arch:
            name_block = f"fc{ind_block+1}"
        elif "resnet" in arch:
            name_block = gpts[name].layer.weight.name_in_model
        gpts_total[name_block] = gpts[name]
    if dev not in ("cpu", torch.device("cpu")):
        torch.cuda.empty_cache()
