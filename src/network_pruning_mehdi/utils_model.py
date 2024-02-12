#%%
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
#%%
import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init
import math
import copy
from utils_optimizer import *

# Imports for conv2d
from typing import TypeVar, Union, Tuple, Optional, List
from itertools import repeat
import collections

import time


def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse

T = TypeVar('T')
_scalar_or_tuple_2_t = Union[T, Tuple[T, T]]
_size_2_t = _scalar_or_tuple_2_t[int]
_pair = _ntuple(2, "_pair")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def _reverse_repeat_tuple(t, n):
    r"""Reverse the order of `t` and repeat each element for `n` times.

    This can be used to translate padding arg used by Conv and Pooling modules
    to the ones used by `F.pad`.
    """
    return tuple(x for x in reversed(t) for _ in range(n))

# End of the imports for conv2d
def compute_z_from_tensor(input_tensor, gamma, type_function):
    if type_function == "smoothstep":
        condition_1 = input_tensor <= -gamma/2
        condition_2 = input_tensor >= gamma/2
        smooth_zs = (-2 /(gamma**3)) * (input_tensor**3) + (3/(2 * gamma)) * input_tensor + 0.5

        return torch.where(condition_1, torch.zeros_like(input_tensor), 
                            torch.where(condition_2, torch.ones_like(input_tensor), smooth_zs))
    elif type_function == "sigmoid":
        return torch.sigmoid(input_tensor/gamma)

def compute_z_rec(module):
    str_module = module.__str__().lower()
    d_children = dict(module.named_children())
    test_children = len(d_children)>0
    if test_children:
        for name_child in d_children:
            child = d_children[name_child]
            compute_z_rec(child)
    # elif not("relu" in str_module) and not("layernorm" in str_module) and not("embedding" in str_module) and not("maxpool" in str_module) and not("avgpool" in str_module) and not("norm" in str_module) and not("linear(" in str_module) and not("view" in str_module) and not("log_softmax_mlpnet" in str_module):
    elif "with_z" in str_module:
        try:
            module.compute_z()
        except:
            import ipdb;ipdb.set_trace()

def set_require_grad_rec(module, is_grad_required):
    str_module = module.__str__().lower()
    d_children = dict(module.named_children())
    test_children = len(d_children)>0
    if test_children:
        for name_child in d_children:
            child = d_children[name_child]
            set_require_grad_rec(child, is_grad_required)
    elif not("relu" in str_module) and not("maxpool" in str_module) and not("avgpool" in str_module) and not("view" in str_module) and not("log_softmax_mlpnet" in str_module):
        #and not("layernorm" in str_module) and not("embedding" in str_module)
        try:
            module.requires_grad_(is_grad_required)
        except:
            pass
            #import ipdb;ipdb.set_trace()

def set_require_grad_with_name_rec(module, is_grad_required, name):
    d_children = dict(module.named_children())
    test_children = len(d_children)>0
    if test_children:
        for name_child in d_children:
            child = d_children[name_child]
            if name_child == name:
                child.requires_grad_(is_grad_required)
            else:
                set_require_grad_with_name_rec(child, is_grad_required, name)

def set_phase_decoder_rec(module, phase):
    str_module = module.__str__().lower()
    d_children = dict(module.named_children())
    test_children = len(d_children)>0
    if str_module[:15]=="optdecoderlayer":
        module.phase = phase
    elif test_children:
        for name_child in d_children:
            child = d_children[name_child]
            set_phase_decoder_rec(child, phase)

def get_phase_decoder_rec(module):
    str_module = module.__str__().lower()
    d_children = dict(module.named_children())
    test_children = len(d_children)>0
    tot_phase = 0
    if str_module[:15]=="optdecoderlayer":
        tot_phase = max(tot_phase, module.phase)
    elif test_children:
        for name_child in d_children:
            child = d_children[name_child]
            tot_phase = max(tot_phase, get_phase_decoder_rec(child))
    return tot_phase

def save_grad_layer_wise_rec(module):
    str_module = module.__str__().lower()
    d_children = dict(module.named_children())
    test_children = len(d_children)>0
    if test_children:
        for name_child in d_children:
            child = d_children[name_child]
            save_grad_layer_wise_rec(child)
    elif not("relu" in str_module) and not("layernorm" in str_module) and not("embedding" in str_module) and not("maxpool" in str_module) and not("avgpool" in str_module) and not("norm" in str_module) and not("linear(" in str_module) and not("view" in str_module) and not("log_softmax_mlpnet" in str_module):
        current_impact = 0
        if module.type_pruning=="layer_wise" or module.type_pruning=="smalles_grad_with_z":
            current_impact = copy.deepcopy(module.weight.grad*(-module.weight.data)+module.weight_z.grad*(-module.gamma/2-module.weight_z.data))
        elif module.type_pruning=="smallest_grad":
            current_impact = copy.deepcopy(module.weight.grad*(-module.weight.data))
        current_impact = torch.abs(current_impact)
        module.impact_of_pruning += current_impact

def initialize_pruning_rec(module, type_pruning, type_function):
    str_module = module.__str__().lower()
    d_children = dict(module.named_children())
    test_children = len(d_children)>0
    if test_children:
        for name_child in d_children:
            child = d_children[name_child]
            initialize_pruning_rec(child, type_pruning, type_function)
    #elif not("relu" in str_module) and not("layernorm" in str_module) and not("embedding" in str_module) and not("maxpool" in str_module) and not("avgpool" in str_module) and not("norm" in str_module) and not("linear(" in str_module) and not("view" in str_module) and not("log_softmax_mlpnet" in str_module):
    elif "with_z" in str_module:
        try:
            module.type_pruning = type_pruning
            module.type_function = type_function
            module.n_weights = np.prod(module.weight.shape)
            module.step_pruning = 0
            if type_pruning=="layer_wise":
                module.layer_wise_loss = 0
                module.impact_of_pruning = 0
            if "smallest_grad" in type_pruning:
                module.impact_of_pruning = 0
        except:
            print(str_module)
            import ipdb;ipdb.set_trace()

def reinitialize_pruning_rec(module, type_pruning):
    str_module = module.__str__().lower()
    d_children = dict(module.named_children())
    test_children = len(d_children)>0
    if test_children:
        for name_child in d_children:
            child = d_children[name_child]
            reinitialize_pruning_rec(child, type_pruning)
    #elif not("relu" in str_module) and not("layernorm" in str_module) and not("embedding" in str_module) and not("maxpool" in str_module) and not("avgpool" in str_module) and not("norm" in str_module) and not("linear(" in str_module) and not("view" in str_module) and not("log_softmax_mlpnet" in str_module):
    elif "with_z" in str_module:
        try:
            if "layer_wise" == type_pruning:
                module.layer_wise_loss = 0
                module.impact_of_pruning = 0
            if "smallest_grad" in type_pruning:
                module.impact_of_pruning = 0
        except:
            import ipdb;ipdb.set_trace()

def use_mask_rec(module, use_mask):
    str_module = module.__str__().lower()
    d_children = dict(module.named_children())
    test_children = len(d_children)>0
    if test_children:
        for name_child in d_children:
            child = d_children[name_child]
            use_mask_rec(child, use_mask)
    #elif not("relu" in str_module) and not("layernorm" in str_module) and not("embedding" in str_module) and not("maxpool" in str_module) and not("avgpool" in str_module) and not("norm" in str_module) and not("linear(" in str_module) and not("view" in str_module) and not("log_softmax_mlpnet" in str_module):
    elif "with_z" in str_module:
        module.use_mask = use_mask

def compute_n_z_rec(module, test_grad, include_batchnorm, type_function):
    if type_function == "smoothstep":
        tol_z_zero = 1e-3
    else:
        tol_z_zero = 0.0
    n_z = 0
    str_module = module.__str__().lower()
    d_children = dict(module.named_children())
    test_children = len(d_children)>0
    if test_children:
        for name_child in d_children:
            child = d_children[name_child]
            n_z += compute_n_z_rec(child, test_grad, include_batchnorm, type_function)
    #elif not("relu" in str_module) and not("layernorm" in str_module) and not("embedding" in str_module) and not("maxpool" in str_module) and not("avgpool" in str_module) and not("norm" in str_module) and not("linear(" in str_module) and not("view" in str_module) and not("log_softmax_mlpnet" in str_module):
    elif "with_z" in str_module:
        try:
            if not(test_grad) or module.weight.requires_grad:
                n_z += torch.sum(module.z>tol_z_zero, dtype=float).detach().item()
                if module.test_bias and module.prune_bias:
                    n_z += torch.sum(module.z_2>tol_z_zero, dtype=float).detach().item()
        except:
            import ipdb;ipdb.set_trace()
    elif "norm" in str_module and include_batchnorm:
        if not(test_grad) or module.weight.requires_grad:
            n_z += np.sum([np.prod(x.shape) for x in module.parameters()])
    return n_z

def compute_n_z_close_to_1_rec(module, tol_z_1, test_grad, include_batchnorm):
    n_z = 0
    str_module = module.__str__().lower()
    d_children = dict(module.named_children())
    test_children = len(d_children)>0
    if test_children:
        for name_child in d_children:
            child = d_children[name_child]
            n_z += compute_n_z_close_to_1_rec(child, tol_z_1, test_grad, include_batchnorm)
    #elif not("relu" in str_module) and not("layernorm" in str_module) and not("embedding" in str_module) and not("maxpool" in str_module) and not("avgpool" in str_module) and not("norm" in str_module) and not("linear(" in str_module) and not("view" in str_module) and not("log_softmax_mlpnet" in str_module):
    elif "with_z" in str_module:
        try:
            if not(test_grad) or module.weight.requires_grad:
                n_z+=torch.sum(module.z>=tol_z_1, dtype=float).detach().item()
                if module.test_bias and module.prune_bias:
                    n_z+=torch.sum(module.z_2>=tol_z_1, dtype=float).detach().item()
        except:
            import ipdb;ipdb.set_trace()
    elif "norm" in str_module and include_batchnorm:
        if not(test_grad) or module.weight.requires_grad:
            n_z += np.sum([np.prod(x.shape) for x in module.parameters()])
    return n_z 

def compute_n_weight_z_close_to_1_rec(module, tol_z_1, test_grad, include_batchnorm):
    n_z = 0
    str_module = module.__str__().lower()
    d_children = dict(module.named_children())
    test_children = len(d_children)>0
    if test_children:
        for name_child in d_children:
            child = d_children[name_child]
            n_z += compute_n_z_close_to_1_rec(child, tol_z_1, test_grad, include_batchnorm)
    #elif not("relu" in str_module) and not("layernorm" in str_module) and not("embedding" in str_module) and not("maxpool" in str_module) and not("avgpool" in str_module) and not("norm" in str_module) and not("linear(" in str_module) and not("view" in str_module) and not("log_softmax_mlpnet" in str_module):
    elif "with_z" in str_module:
        try:
            if not(test_grad) or module.weight.requires_grad:
                n_z+=torch.sum(module.weight_z>=tol_z_1, dtype=float).detach().item()
                if module.test_bias and module.prune_bias:
                    n_z+=torch.sum(module.bias_z>=tol_z_1, dtype=float).detach().item()
        except:
            import ipdb;ipdb.set_trace()
    elif "norm" in str_module and include_batchnorm:
        if not(test_grad) or module.weight.requires_grad:
            n_z += np.sum([np.prod(x.shape) for x in module.parameters()])
    return n_z 

def reset_z_rec(module, tol_z_1, prop_reset, type_pruning, generator, test_grad, type_reset, method_pruning, threshold_restart, test_mult_reset):
    n_reset = 0
    input_tol = tol_z_1
    str_module = module.__str__().lower()
    d_children = dict(module.named_children())
    test_children = len(d_children)>0
    if test_children:
        for name_child in d_children:
            child = d_children[name_child]
            n_reset += reset_z_rec(child, tol_z_1, prop_reset, type_pruning, generator, test_grad, type_reset, method_pruning, threshold_restart, test_mult_reset)
    #elif not("relu" in str_module) and not("layernorm" in str_module) and not("embedding" in str_module) and not("maxpool" in str_module) and not("avgpool" in str_module) and not("norm" in str_module) and not("linear(" in str_module) and not("view" in str_module) and not("log_softmax_mlpnet" in str_module):
    elif "with_z" in str_module:   
        if not(test_grad) or module.weight.requires_grad:
            #module.compute_z()
            old_z = module.z
            if module.test_bias and module.prune_bias:
                old_bias_z = module.z_2
            n_reset += module.reset_weight_z(input_tol, prop_reset, type_pruning, generator, type_reset, method_pruning, threshold_restart)
            module.compute_z()
            new_z = module.z
            module.weight.data[new_z==0] = 0
            if test_mult_reset:
                module.weight.data[new_z>0] *= (old_z/new_z)[new_z>0]
            if module.test_bias and module.prune_bias:
                new_bias_z = module.z_2
                module.bias.data[new_bias_z==0] = 0
                if test_mult_reset:
                    module.bias.data[new_bias_z>0] *= (old_bias_z/new_bias_z)[new_bias_z>0]
    return n_reset

def compute_losses_rec(module, device, entropy_reg, selection_reg, l2_reg, l2_original_reg, original_module):
    tol_instability_ent = 1e-6
    tol_instability_l2 = 1e-3
    entropy_loss = torch.tensor(0.0).to(device)
    selection_loss = torch.tensor(0.0).to(device)
    l2_loss = torch.tensor(0.0).to(device)
    l2_original_loss = torch.tensor(0.0).to(device)
    str_module = module.__str__().lower()
    d_children = dict(module.named_children())
    if original_module!=None:
        d_children_original = dict(original_module.named_children())
    test_children = len(d_children)>0
    if test_children:
        for name_child in d_children:
            child = d_children[name_child]
            if original_module!=None:
                child_original = d_children_original[name_child]
            else:
                child_original = None
            entropy_loss_child, selection_loss_child, l2_loss_child, l2_original_loss_child = compute_losses_rec(child, device, entropy_reg, selection_reg, l2_reg, l2_original_reg, child_original)
            entropy_loss += entropy_loss_child
            selection_loss += selection_loss_child
            l2_loss += l2_loss_child
            l2_original_loss += l2_original_loss_child
    #elif not("relu" in str_module) and not("layernorm" in str_module) and not("embedding" in str_module) and not("maxpool" in str_module) and not("avgpool" in str_module) and not("norm" in str_module) and not("linear(" in str_module) and not("view" in str_module) and not("log_softmax_mlpnet" in str_module):
    elif "with_z" in str_module:
        try:
            if entropy_reg!=0:
                first_step_ent = module.z*torch.log(module.z+tol_instability_ent) + (1-module.z)*torch.log(1-module.z+tol_instability_ent)
                first_step_ent[first_step_ent>0] = 0
                entropy_loss += -(torch.sum(first_step_ent))
                if module.test_bias and module.prune_bias:
                    first_step_ent = module.z_2*torch.log(module.z_2+tol_instability_ent) + (1-module.z_2)*torch.log(1-module.z_2+tol_instability_ent)
                    first_step_ent[first_step_ent>0] = 0
                    entropy_loss += -(torch.sum(first_step_ent))
            if selection_reg !=0:
                selection_loss += torch.sum(module.z)
                if module.test_bias and module.prune_bias:
                    selection_loss += torch.sum(module.z_2)
            if l2_reg !=0:
                #l2_loss += torch.sum(module.weight**2)
                l2_loss += torch.sum(((module.weight*(((1-module.z)/(tol_instability_l2+module.z)).detach()))**2))
                if module.test_bias and module.prune_bias:
                    # l2_loss += torch.sum(module.bias**2)
                    l2_loss += torch.sum(((module.bias*(((1-module.z_2)/(tol_instability_l2+module.z_2)).detach()))**2))
            if l2_original_reg !=0:
                l2_original_loss += (torch.sum(((module.weight.view(-1)*module.z.view(-1)-original_module.weight.view(-1))**2)))
                if module.test_bias and module.prune_bias:
                    l2_original_loss += (torch.sum(((module.bias.view(-1)*module.z_2.view(-1)-original_module.bias.view(-1))**2)))
            # return entropy_loss, selection_loss, l2_loss, selection_lagrangian_loss, l2_original_loss
            if entropy_loss<0:
                import ipdb;ipdb.set_trace()
        except:
            import ipdb;ipdb.set_trace()
    return entropy_loss, selection_loss, l2_loss, l2_original_loss

def compute_layer_wise_loss_rec(module, device):
    layer_wise_loss = torch.tensor(0.0).to(device)
    str_module = module.__str__().lower()
    d_children = dict(module.named_children())
    test_children = len(d_children)>0
    if test_children:
        for name_child in d_children:
            child = d_children[name_child]
            layer_wise_loss_child = compute_layer_wise_loss_rec(child, device)
            layer_wise_loss += layer_wise_loss_child
    #elif not("relu" in str_module) and not("layernorm" in str_module) and not("embedding" in str_module) and not("maxpool" in str_module) and not("avgpool" in str_module) and not("norm" in str_module) and not("linear(" in str_module) and not("view" in str_module) and not("log_softmax_mlpnet" in str_module):
    elif "with_z" in str_module:
        layer_wise_loss += module.layer_wise_loss
    return layer_wise_loss

def get_idx_keep(module, name_module, d_list):
    d_list[name_module+"_row"] = False
    d_list[name_module+"_column"] = False
    if module.bias!=None:
        d_list[name_module+"_row_bias"] = False
    if not("batch" in module.__str__().lower()):
        if np.prod(module.z.shape)>0:
            len_shape = len(module.z.shape)
            sum_rows_z = torch.sum(module.z, tuple([1]+[2+i for i in range(len_shape-2)]))
            if module.test_bias and module.prune_bias:
                sum_rows_z += module.z_2
            sum_columns_z = torch.sum(module.z, tuple([0]+[2+i for i in range(len_shape-2)]))
            if torch.min(sum_rows_z) == 0:
                d_list[name_module+"_row"] = True
                d_list[name_module+"_idx_keep_row"] = torch.where(sum_rows_z!=0)[0]
                if module.test_bias and module.prune_bias:
                    d_list[name_module+"_idx_keep_row_bias"] = torch.where(sum_rows_z!=0)[0]
                    d_list[name_module+"_row_bias"] = True
            if torch.min(sum_columns_z) == 0:
                d_list[name_module+"_column"] = True
                d_list[name_module+"_idx_keep_column"] = torch.where(sum_columns_z!=0)[0]
        # if module.test_bias and module.prune_bias:
        #     d_list[name_module+"_row_bias"] = False
        #     if np.prod(module.z_2.shape)>0:
        #         sum_rows_z = module.z_2
        #         if torch.min(sum_rows_z) == 0:
        #             d_list[name_module+"_idx_keep_row_bias"] = torch.where(sum_rows_z!=0)[0]
        #             d_list[name_module+"_row_bias"] = True

def classical_pruning(module, tol_reset = 1e-2):
    module.compute_z()

    # Prune the z close to 0
    condition_weight = torch.abs(module.z) <= tol_reset
    module.weight_z.data[condition_weight] = -module.gamma
    module.weight.data[condition_weight] = 0

    # Set the z close to 1
    condition_weight = torch.abs(module.z) >= 1-tol_reset
    module.weight_z.data[condition_weight] = module.gamma

    if module.test_bias and module.prune_bias:
        # Prune the z close to 0
        condition_bias = torch.abs(module.z_2) <= tol_reset
        module.bias_z.data[condition_bias] = -module.gamma
        module.bias.data[condition_bias] = 0

        # Set the z close to 1
        condition_weight = torch.abs(module.z_2) >= 1-tol_reset
        module.bias_z.data[condition_weight] = module.gamma

def remove_weights(self, d_modules, optimizer, d_list, l_name_modules):
    optimizer_name = optimizer.__class__.__name__
    
    # TO DELETE LATE
    # list_params_old_test_matching = list(optimizer.param_groups[0]["params"])
    # END

    if "momentum" in optimizer.defaults:
        momentum = optimizer.defaults["momentum"]
    else:
        momentum = -1

    copy_optimizer = initialize_optimizer(test_different_lr = self.test_different_lr, model=self.model, optimizer_name=optimizer_name, steps_per_epoch=self.steps_per_epoch, lr = optimizer.defaults["lr"], val_second_lr=self.val_second_lr, momentum=momentum, weight_decay=optimizer.defaults["weight_decay"])
    
    try:
        copy_optimizer._step_count = optimizer._step_count
    except:
        pass

    # for idx_param in range(len(optimizer.param_groups)):
    #     list_params = list(optimizer.param_groups[idx_param]["params"])
    #     for i in range(len(list_params)):
    #         group_param = list_params[i]
    #         copy_optimizer.state[group_param] = copy.deepcopy(optimizer.state[group_param])

    for name_module in l_name_modules:
        module = d_modules[name_module]
        test_exp_avg = False
        test_exp_avg_sq = False
        test_momentum_buffer = False
        test_step = False
        if ("batch" in module.__str__().lower() and d_list[name_module+"_row"]) or (not("batch" in module.__str__().lower()) and (d_list[name_module+"_row"] or d_list[name_module+"_column"])):
            if "exp_avg" in optimizer.state[module.weight]:
                test_exp_avg = True
                old_state_exp_avg = copy.deepcopy(optimizer.state[module.weight]["exp_avg"])
                if not("batch" in module.__str__().lower()):
                    old_state_exp_avg_z = copy.deepcopy(optimizer.state[module.weight_z]["exp_avg"])
            if "exp_avg_sq" in optimizer.state[module.weight]:
                test_exp_avg_sq = True
                old_state_exp_avg_sq = copy.deepcopy(optimizer.state[module.weight]["exp_avg_sq"])
                if not("batch" in module.__str__().lower()):
                    old_state_exp_avg_sq_z = copy.deepcopy(optimizer.state[module.weight_z]["exp_avg_sq"])
            if "momentum_buffer" in optimizer.state[module.weight] and optimizer.state[module.weight]["momentum_buffer"]!=None:
                test_momentum_buffer = True
                old_state_mom_buff = copy.deepcopy(optimizer.state[module.weight]["momentum_buffer"])
                if not("batch" in module.__str__().lower()):
                    old_state_mom_buff_z = copy.deepcopy(optimizer.state[module.weight_z]["momentum_buffer"])
            if "step" in optimizer.state[module.weight]:
                test_step = True
                old_step = copy.deepcopy(optimizer.state[module.weight]["step"])
                if not("batch" in module.__str__().lower()):
                    old_step_z = copy.deepcopy(optimizer.state[module.weight_z]["step"])
        if d_list[name_module+"_row"]:
            idx_keep_row = d_list[name_module+"_idx_keep_row"]
            copy_grad = copy.deepcopy(module.weight.grad[idx_keep_row])
            module.weight = Parameter(module.weight[idx_keep_row])
            module.weight.grad = copy_grad
            
            if not("batch" in module.__str__().lower()):
                copy_grad = copy.deepcopy(module.weight_z.grad[idx_keep_row])
                module.weight_z = Parameter(module.weight_z[idx_keep_row])
                module.weight_z.grad = copy_grad
                if "linear_with" in module.__str__().lower():
                    module.out_features = len(idx_keep_row)
                elif "conv" in module.__str__().lower():
                    module.out_channels = len(idx_keep_row)
            else:
                module.num_features = len(idx_keep_row)
                module.running_mean = module.running_mean[idx_keep_row]
                module.running_var = module.running_var[idx_keep_row]

            if test_exp_avg:
                old_state_exp_avg = old_state_exp_avg[idx_keep_row]
                if not("batch" in module.__str__().lower()):
                    old_state_exp_avg_z = old_state_exp_avg_z[idx_keep_row]
            if test_exp_avg_sq:
                old_state_exp_avg_sq = old_state_exp_avg_sq[idx_keep_row]
                if not("batch" in module.__str__().lower()):
                    old_state_exp_avg_sq_z = old_state_exp_avg_sq_z[idx_keep_row]
            if test_momentum_buffer:
                old_state_mom_buff = old_state_mom_buff[idx_keep_row]
                if not("batch" in module.__str__().lower()):
                    old_state_mom_buff_z = old_state_mom_buff_z[idx_keep_row]
        if not("batch" in module.__str__().lower()) and d_list[name_module+"_column"]:
            idx_keep_column = d_list[name_module+"_idx_keep_column"]
            copy_grad = copy.deepcopy(module.weight.grad[:,idx_keep_column])
            module.weight = Parameter(module.weight[:,idx_keep_column])
            module.weight.grad = copy_grad
            if "linear_with" in module.__str__().lower():
                module.in_features = len(idx_keep_column)
            elif "conv" in module.__str__().lower():
                module.in_channels = len(idx_keep_column)
            copy_grad = copy.deepcopy(module.weight_z.grad[:,idx_keep_column])
            module.weight_z = Parameter(module.weight_z[:,idx_keep_column])
            module.weight_z.grad = copy_grad
            if test_exp_avg:
                old_state_exp_avg = old_state_exp_avg[:,idx_keep_column]
                old_state_exp_avg_z = old_state_exp_avg_z[:,idx_keep_column]
            if test_exp_avg_sq:
                old_state_exp_avg_sq = old_state_exp_avg_sq[:,idx_keep_column]
                old_state_exp_avg_sq_z = old_state_exp_avg_sq_z[:,idx_keep_column]
            if test_momentum_buffer:
                old_state_mom_buff = old_state_mom_buff[:,idx_keep_column]                      
                old_state_mom_buff_z = old_state_mom_buff_z[:,idx_keep_column]
            if name_module == "fc1":
                self.model.idx_keep_input = self.model.idx_keep_input[idx_keep_column.cpu()]

        if test_exp_avg:
            copy_optimizer.state[module.weight]["exp_avg"] = old_state_exp_avg
            if not("batch" in module.__str__().lower()):
                copy_optimizer.state[module.weight_z]["exp_avg"] = old_state_exp_avg_z
        elif module.weight in optimizer.state and "exp_avg" in optimizer.state[module.weight]:
            copy_optimizer.state[module.weight]["exp_avg"] = copy.deepcopy(optimizer.state[module.weight]["exp_avg"])
            if not("batch" in module.__str__().lower()):
                copy_optimizer.state[module.weight_z]["exp_avg"] = copy.deepcopy(optimizer.state[module.weight_z]["exp_avg"])
        if test_exp_avg_sq:
            copy_optimizer.state[module.weight]["exp_avg_sq"] = old_state_exp_avg_sq
            if not("batch" in module.__str__().lower()):
                copy_optimizer.state[module.weight_z]["exp_avg_sq"] = old_state_exp_avg_sq_z
        elif module.weight in optimizer.state and "exp_avg_sq" in optimizer.state[module.weight]:
            copy_optimizer.state[module.weight]["exp_avg_sq"] = copy.deepcopy(optimizer.state[module.weight]["exp_avg_sq"])
            if not("batch" in module.__str__().lower()):
                copy_optimizer.state[module.weight_z]["exp_avg_sq"] = copy.deepcopy(optimizer.state[module.weight_z]["exp_avg_sq"])
        if test_momentum_buffer:
            copy_optimizer.state[module.weight]["momentum_buffer"] = old_state_mom_buff
            if not("batch" in module.__str__().lower()):
                copy_optimizer.state[module.weight_z]["momentum_buffer"] = old_state_mom_buff_z
        elif module.weight in optimizer.state and "momentum_buffer" in optimizer.state[module.weight] and optimizer.state[module.weight]["momentum_buffer"]!=None:
            copy_optimizer.state[module.weight]["momentum_buffer"] = copy.deepcopy(optimizer.state[module.weight]["momentum_buffer"])
            if not("batch" in module.__str__().lower()):
                copy_optimizer.state[module.weight_z]["momentum_buffer"] = copy.deepcopy(optimizer.state[module.weight_z]["momentum_buffer"])
        if test_step:
            copy_optimizer.state[module.weight]["step"] = old_step
            if not("batch" in module.__str__().lower()):
                copy_optimizer.state[module.weight_z]["step"] = old_step_z
        elif module.weight in optimizer.state and "step" in optimizer.state[module.weight]:
            copy_optimizer.state[module.weight]["step"] = copy.deepcopy(optimizer.state[module.weight]["step"])
            if not("batch" in module.__str__().lower()):
                copy_optimizer.state[module.weight_z]["step"] = copy.deepcopy(optimizer.state[module.weight_z]["step"])

        test_exp_avg_bias = False
        test_exp_avg_sq_bias = False
        test_momentum_buffer_bias = False
        test_step_bias = False
        #if not("batch" in module.__str__().lower()) and module.bias!=None:
        if module.bias!=None:
            if d_list[name_module+"_row_bias"]:
                if "exp_avg" in optimizer.state[module.bias]:
                    test_exp_avg_bias = True
                    old_state_exp_avg_bias = copy.deepcopy(optimizer.state[module.bias]["exp_avg"])
                    if not("batch" in module.__str__().lower()):
                        old_state_exp_avg_z_bias = copy.deepcopy(optimizer.state[module.bias_z]["exp_avg"])
                if "exp_avg_sq" in optimizer.state[module.bias]:
                    test_exp_avg_sq_bias = True
                    old_state_exp_avg_sq_bias = copy.deepcopy(optimizer.state[module.bias]["exp_avg_sq"])
                    if not("batch" in module.__str__().lower()):
                        old_state_exp_avg_sq_z_bias = copy.deepcopy(optimizer.state[module.bias_z]["exp_avg_sq"])
                if "momentum_buffer" in optimizer.state[module.bias]:
                    test_momentum_buffer_bias = True
                    old_state_mom_buff_bias = copy.deepcopy(optimizer.state[module.bias]["momentum_buffer"])
                    if not("batch" in module.__str__().lower()):
                        old_state_mom_buff_z_bias = copy.deepcopy(optimizer.state[module.bias_z]["momentum_buffer"])
                if "step" in optimizer.state[module.bias]:
                    test_step_bias = True
                    old_step_bias = copy.deepcopy(optimizer.state[module.bias]["step"])
                    if not("batch" in module.__str__().lower()):
                        old_step_z_bias = copy.deepcopy(optimizer.state[module.bias_z]["step"])

                idx_keep_row_bias = d_list[name_module+"_idx_keep_row_bias"]
                copy_grad = copy.deepcopy(module.bias.grad[idx_keep_row_bias])
                module.bias = Parameter(module.bias[idx_keep_row_bias])
                module.bias.grad = copy_grad
                if not("batch" in module.__str__().lower()):
                    copy_grad = copy.deepcopy(module.bias_z.grad[idx_keep_row_bias])
                    module.bias_z = Parameter(module.bias_z[idx_keep_row_bias])
                    module.bias_z.grad = copy_grad

                if test_exp_avg_bias:
                    old_state_exp_avg_bias = old_state_exp_avg_bias[idx_keep_row_bias]
                    if not("batch" in module.__str__().lower()):
                        old_state_exp_avg_z_bias = old_state_exp_avg_z_bias[idx_keep_row_bias]
                if test_exp_avg_sq_bias:
                    old_state_exp_avg_sq_bias = old_state_exp_avg_sq_bias[idx_keep_row_bias]
                    if not("batch" in module.__str__().lower()):
                        old_state_exp_avg_sq_z_bias = old_state_exp_avg_sq_z_bias[idx_keep_row_bias]
                if test_momentum_buffer_bias:
                    old_state_mom_buff_bias = old_state_mom_buff_bias[idx_keep_row_bias]
                    if not("batch" in module.__str__().lower()):
                        old_state_mom_buff_z_bias = old_state_mom_buff_z_bias[idx_keep_row_bias]

            if test_exp_avg_bias:
                copy_optimizer.state[module.bias]["exp_avg"] = old_state_exp_avg_bias
                if not("batch" in module.__str__().lower()):
                    copy_optimizer.state[module.bias_z]["exp_avg"] = old_state_exp_avg_z_bias
            elif module.bias in optimizer.state and "exp_avg" in optimizer.state[module.bias]:
                copy_optimizer.state[module.bias]["exp_avg"] = copy.deepcopy(optimizer.state[module.bias]["exp_avg"])
                if not("batch" in module.__str__().lower()):
                    copy_optimizer.state[module.bias_z]["exp_avg"] = copy.deepcopy(optimizer.state[module.bias_z]["exp_avg"])
            if test_exp_avg_sq_bias:
                copy_optimizer.state[module.bias]["exp_avg_sq"] = old_state_exp_avg_sq_bias
                if not("batch" in module.__str__().lower()):
                    copy_optimizer.state[module.bias_z]["exp_avg_sq"] = old_state_exp_avg_sq_z_bias
            elif module.bias in optimizer.state and "exp_avg_sq" in optimizer.state[module.bias]:
                copy_optimizer.state[module.bias]["exp_avg_sq"] = copy.deepcopy(optimizer.state[module.bias]["exp_avg_sq"])
                if not("batch" in module.__str__().lower()):
                    copy_optimizer.state[module.bias_z]["exp_avg_sq"] = copy.deepcopy(optimizer.state[module.bias_z]["exp_avg_sq"])
            if test_momentum_buffer_bias:
                copy_optimizer.state[module.bias]["momentum_buffer"] = old_state_mom_buff_bias
                if not("batch" in module.__str__().lower()):
                    copy_optimizer.state[module.bias_z]["momentum_buffer"] = old_state_mom_buff_z_bias
            elif module.bias in optimizer.state and "momentum_buffer" in optimizer.state[module.bias]:
                copy_optimizer.state[module.bias]["momentum_buffer"] = copy.deepcopy(optimizer.state[module.bias]["momentum_buffer"])
                if not("batch" in module.__str__().lower()):
                    copy_optimizer.state[module.bias_z]["momentum_buffer"] = copy.deepcopy(optimizer.state[module.bias_z]["momentum_buffer"])
            if test_step_bias:
                copy_optimizer.state[module.bias]["step"] = old_step_bias
                if not("batch" in module.__str__().lower()):
                    copy_optimizer.state[module.bias_z]["step"] = old_step_z_bias
            elif module.bias in optimizer.state and "step" in optimizer.state[module.bias]:
                copy_optimizer.state[module.bias]["step"] = copy.deepcopy(optimizer.state[module.bias]["step"])
                if not("batch" in module.__str__().lower()):
                    copy_optimizer.state[module.bias_z]["step"] = copy.deepcopy(optimizer.state[module.bias_z]["step"])
    
    #del optimizer
    # optimizer = copy_optimizer
    optimizer_name = copy_optimizer.__class__.__name__
    if "momentum" in copy_optimizer.defaults:
        momentum = copy_optimizer.defaults["momentum"]
    else:
        momentum = -1
    final_optimizer = initialize_optimizer(test_different_lr = self.test_different_lr, model=self.model, optimizer_name=optimizer_name, steps_per_epoch=self.steps_per_epoch, lr = copy_optimizer.defaults["lr"], val_second_lr=self.val_second_lr, momentum=momentum, weight_decay=copy_optimizer.defaults["weight_decay"])
    try:
        final_optimizer._step_count = copy_optimizer._step_count
    except:
        pass

    for idx_param in range(len(copy_optimizer.param_groups)):
        list_params_new = list(final_optimizer.param_groups[idx_param]["params"])
        for i in range(len(list_params_new)):
            group_param_new = list_params_new[i]
            final_optimizer.state[group_param_new] = copy.deepcopy(copy_optimizer.state[group_param_new])
    
    del copy_optimizer

    # TO DELETE LATER
    # list_params_new_test_matching = list(final_optimizer.param_groups[0]["params"])
    
    # if len(list_params_new_test_matching)!=len(list_params_old_test_matching):
    #     print("ERROR OF LENGTH")
    #     import ipdb;ipdb.set_trace()

    # for i in range(len(list_params_old_test_matching)):
    #     param_old = list_params_old_test_matching[i]
    #     param_new = list_params_new_test_matching[i]
    #     if i in [0,1]:
    #         name_module = "fc1"
    #     if i in [2,3]:
    #         name_module = "fc2"
    #     if i in [4,5]:
    #         name_module = "fc3"
        
    #     for key in ["exp_avg", "exp_avg_sq"]:
    #         old_mat = optimizer.state[param_old][key]
    #         new_mat = final_optimizer.state[param_new][key]
    #         if d_list[name_module+"_row"]:
    #             old_mat = old_mat[d_list[name_module+"_idx_keep_row"]]
    #         if d_list[name_module+"_column"]:
    #             old_mat = old_mat[:,d_list[name_module+"_idx_keep_column"]]
    #         if torch.sum(new_mat != old_mat)>0:
    #             print("STATE NOT MATCHING")
    #             import ipdb;ipdb.set_trace()
    #     if d_list[name_module+"_row"]:
    #         param_old = param_old[d_list[name_module+"_idx_keep_row"]]
    #     if d_list[name_module+"_column"]:
    #         param_old = param_old[:,d_list[name_module+"_idx_keep_column"]]
    #     if torch.sum(param_new != param_old)>0:
    #         print("Param NOT MATCHING")
    #         import ipdb;ipdb.set_trace()
    # END
    return final_optimizer

def advanced_dense_pruning_weights(d_modules, d_list, l_name_modules):
    for name_module in l_name_modules:
        module = d_modules[name_module]
        if d_list[name_module+"_row"]:
            idx_keep_row = d_list[name_module+"_idx_keep_row"]
            idx_remove_row = np.array([ind for ind in range(module.weight.shape[0]) if not(ind in idx_keep_row)])
            module.weight.data[idx_remove_row] = 0
            if not("batch" in module.__str__().lower()):
                module.weight_z.data[idx_remove_row] = -module.gamma
        if not("batch" in module.__str__().lower()) and d_list[name_module+"_column"]:
            idx_keep_column = d_list[name_module+"_idx_keep_column"]
            idx_remove_column = np.array([ind for ind in range(module.weight.shape[1]) if not(ind in idx_keep_column)])
            module.weight.data[:,idx_remove_column] = 0
            module.weight_z.data[:,idx_remove_column] = -module.gamma
        if not("batch" in module.__str__().lower()) and module.test_bias:
            if d_list[name_module+"_row_bias"]:
                idx_keep_row_bias = d_list[name_module+"_idx_keep_row_bias"]
                idx_remove_row_bias = np.array([ind for ind in range(module.bias.shape[0]) if not(ind in idx_keep_row_bias)])
                module.bias.data[idx_remove_row_bias] = 0
                module.bias_z.data[idx_remove_row_bias] = -module.gamma
    return None

def prune_models_external(self, d_modules, optimizer, dense_to_sparse=False):
    # if self.step_temp == 201:
    #     import ipdb;ipdb.set_trace()
    test_pruned = False
    l_name_modules = list(d_modules.keys())
    #l_name_modules = [x for x in l_name_modules if len(d_modules[x].__str__().lower().split(":"))==1 and (("conv" in d_modules[x].__str__().lower()) or ("batch" in d_modules[x].__str__().lower()) or ("linear_with" in d_modules[x].__str__().lower()))]
    l_name_modules = [x for x in l_name_modules if len(d_modules[x].__str__().lower().split(":"))==1 and (("with_z" in d_modules[x].__str__().lower()) or ("batch" in d_modules[x].__str__().lower()))]
    d_list = {}
    # Perform classical pruning (setting manually the weights to -1) and collect the potential rows/columns to prune
    # The row of the previous module has the same shape and the column of the next module (except if this is a batch norm layer)
    for ind_module in range(len(l_name_modules)):
        x = l_name_modules[ind_module]
        get_idx_keep(d_modules[x], x, d_list)
        if not("batch" in d_modules[x].__str__().lower()):
            # Case where x is not a bn
            classical_pruning(d_modules[x])
            # if dense_to_sparse:
            if ind_module>0:
                x_previous = l_name_modules[ind_module-1]
                if d_list[x_previous+"_row"]:
                    if d_list[x+"_column"]:
                        list1 = d_list[x_previous+"_idx_keep_row"]
                        list2 = d_list[x+"_idx_keep_column"]
                        intersection = torch.Tensor(np.intersect1d(list1.cpu(), list2.cpu())).long()
                        d_list[x_previous+"_idx_keep_row"] = intersection
                        d_list[x+"_idx_keep_column"] = intersection
                        if d_modules[x_previous].bias!=None:
                            d_list[x_previous+"_idx_keep_row_bias"] = intersection
                            d_list[x_previous+"_row_bias"] = True
                    else:
                        list1 = d_list[x_previous+"_idx_keep_row"]
                        d_list[x+"_idx_keep_column"] = list1
                        d_list[x+"_column"] = True
                else:
                    if d_list[x+"_column"]:
                        list2 = d_list[x+"_idx_keep_column"]
                        d_list[x_previous+"_idx_keep_row"] = list2
                        d_list[x_previous+"_row"] = True
                        if d_modules[x_previous].bias!=None:
                            d_list[x_previous+"_idx_keep_row_bias"] = list2
                            d_list[x_previous+"_row_bias"] = True
        else:
            # Case where x is a bn
            if ind_module>0:# and dense_to_sparse:
                x_previous = l_name_modules[ind_module-1]
                if d_list[x_previous+"_row"]:
                    if d_list[x+"_row"]:
                        list1 = d_list[x_previous+"_idx_keep_row"]
                        list2 = d_list[x+"_idx_keep_row"]
                        intersection = torch.Tensor(np.intersect1d(list1.cpu(), list2.cpu())).long()
                        if d_modules[x_previous].bias!=None:
                            d_list[x_previous+"_idx_keep_row_bias"] = intersection
                            d_list[x_previous+"_row_bias"] = True
                        d_list[x_previous+"_idx_keep_row"] = intersection
                        d_list[x+"_idx_keep_row"] = intersection
                    else:
                        list1 = d_list[x_previous+"_idx_keep_row"]
                        d_list[x+"_idx_keep_row"] = list1
                        d_list[x+"_row"] = True
                else:
                    if d_list[x+"_row"]:
                        list2 = d_list[x+"_idx_keep_row"]
                        d_list[x_previous+"_idx_keep_row"] = list2
                        d_list[x_previous+"_row"] = True
                        if d_modules[x_previous].bias!=None:
                            d_list[x_previous+"_idx_keep_row_bias"] = list2
                            d_list[x_previous+"_row_bias"] = True

    # Update the rows/column to prune backwards (based on the list of modules)
    # if dense_to_sparse:
    for ind_module in range(len(l_name_modules)-1,0,-1):
        x = l_name_modules[ind_module]
        x_previous = l_name_modules[ind_module-1]
        if not("batch" in d_modules[x].__str__().lower()):
            if d_list[x_previous+"_row"]:
                if d_list[x+"_column"]:
                    list1 = d_list[x_previous+"_idx_keep_row"]
                    list2 = d_list[x+"_idx_keep_column"]
                    intersection = torch.Tensor(np.intersect1d(list1.cpu(), list2.cpu())).long()
                    d_list[x_previous+"_idx_keep_row"] = intersection
                    d_list[x+"_idx_keep_column"] = intersection
                    if d_modules[x_previous].bias!=None:
                        d_list[x_previous+"_idx_keep_row_bias"] = intersection
                        d_list[x_previous+"_row_bias"] = True
                else:
                    list1 = d_list[x_previous+"_idx_keep_row"]
                    d_list[x+"_idx_keep_column"] = list1
                    d_list[x+"_column"] = True
            else:
                if d_list[x+"_column"]:
                    list2 = d_list[x+"_idx_keep_column"]
                    d_list[x_previous+"_idx_keep_row"] = list2
                    d_list[x_previous+"_row"] = True
                    if d_modules[x_previous].bias!=None:
                        d_list[x_previous+"_idx_keep_row_bias"] = list2
                        d_list[x_previous+"_row_bias"] = True
        else:
            if d_list[x_previous+"_row"]:
                if d_list[x+"_row"]:
                    list1 = d_list[x_previous+"_idx_keep_row"]
                    list2 = d_list[x+"_idx_keep_row"]
                    intersection = torch.Tensor(np.intersect1d(list1.cpu(), list2.cpu())).long()
                    if d_modules[x_previous].bias!=None:
                        d_list[x_previous+"_idx_keep_row_bias"] = intersection
                        d_list[x_previous+"_row_bias"] = True
                    d_list[x_previous+"_idx_keep_row"] = intersection
                    d_list[x+"_idx_keep_row"] = intersection
                else:
                    list1 = d_list[x_previous+"_idx_keep_row"]
                    d_list[x+"_idx_keep_row"] = list1
                    d_list[x+"_row"] = True
            else:
                if d_list[x+"_row"]:
                    list2 = d_list[x+"_idx_keep_row"]
                    d_list[x_previous+"_idx_keep_row"] = list2
                    d_list[x_previous+"_row"] = True
                    if d_modules[x_previous].bias!=None:
                        d_list[x_previous+"_idx_keep_row_bias"] = list2
                        d_list[x_previous+"_row_bias"] = True

    test_pruned = any([not("batch" in d_modules[x].__str__().lower()) and ((d_list[x+"_row"] or d_list[x+"_column"])) for x in l_name_modules])
    self.d_list = d_list
        # if test_pruned:
        # if self.step_temp == 200:
        #     copy_exp_avg_1 = copy.deepcopy(optimizer.state[self.model.fc1.weight]["exp_avg"].numpy())
        #     copy_exp_avg_sq_1 = copy.deepcopy(optimizer.state[self.model.fc1.weight]["exp_avg_sq"].numpy())
        #     copy_grad_1 = copy.deepcopy(self.model.fc1.weight.grad.numpy())
        #     copy_exp_avg_2 = copy.deepcopy(optimizer.state[self.model.fc2.weight]["exp_avg"].numpy())
        #     copy_exp_avg_sq_2 = copy.deepcopy(optimizer.state[self.model.fc2.weight]["exp_avg_sq"].numpy())
        #     copy_grad_2 = copy.deepcopy(self.model.fc2.weight.grad.numpy())
        #     copy_exp_avg_3 = copy.deepcopy(optimizer.state[self.model.fc3.weight]["exp_avg"].numpy())
        #     copy_exp_avg_sq_3 = copy.deepcopy(optimizer.state[self.model.fc3.weight]["exp_avg_sq"].numpy())
        #     copy_grad_3 = copy.deepcopy(self.model.fc3.weight.grad.numpy())
        #     import ipdb;ipdb.set_trace()
    if dense_to_sparse:
        if test_pruned:
            final_optimizer = remove_weights(self, d_modules, optimizer, d_list, l_name_modules)
        else:
            final_optimizer = optimizer
    else:
        # if test_pruned:
        #     advanced_dense_pruning_weights(d_modules, d_list, l_name_modules)
        final_optimizer = optimizer
        
    return final_optimizer, test_pruned

def prune_models_external_sigmoid(model, tol_reset):
    d_modules = dict(model.named_modules())
    l_name_modules = list(d_modules.keys())
    l_name_modules = [x for x in l_name_modules if len(d_modules[x].__str__().lower().split(":"))==1 and (("conv" in d_modules[x].__str__().lower()) or ("batch" in d_modules[x].__str__().lower()) or ("linear_with" in d_modules[x].__str__().lower()))]
    for ind_module in range(len(l_name_modules)):
        x = l_name_modules[ind_module]
        if not("batch" in d_modules[x].__str__().lower()):
            classical_pruning(d_modules[x], tol_reset)

class View_mlpnet(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)

class Log_softmax_mlpnet(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.log_softmax(x, dim=1)
    
class Avgpool_2d_resnet18(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return F.avg_pool2d(x, 4)
    
class View_resnet(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)
    
class View_mobilenet(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(-1, x.size(1))


class Linear_with_z(torch.nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, gamma: float = 1.0,
                 device=None, dtype=None, prune_bias=True) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gamma=gamma
        self.is_sparse = False
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.weight_z = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.test_bias = bias
        self.prune_bias=prune_bias
        self.use_mask=True

        if self.test_bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
            if self.prune_bias:
                self.bias_z = Parameter(torch.empty(out_features, **factory_kwargs))
            else:
                self.register_parameter('bias_z', None)
        else:
            self.register_parameter('bias', None)
            self.register_parameter('bias_z', None)
        self.reset_parameters()

    def compute_z(self) -> None:
        self.z = compute_z_from_tensor(self.weight_z, self.gamma, self.type_function)

        if self.test_bias and self.prune_bias:
            self.z_2 = compute_z_from_tensor(self.bias_z, self.gamma, self.type_function)
            if self.use_mask:
                self.bias_final = self.bias*self.z_2
            else:
                self.bias_final = self.bias#*1.0
        elif self.test_bias:
            self.bias_final = self.bias#*1.0
        else:
            self.bias_final = self.bias

    def forward(self, input: Tensor) -> Tensor:
        self.compute_z()
        #output_linear = (self.weight*self.z).matmul(input.T).T
        if self.use_mask:
            output_linear = F.linear(input, self.weight*self.z, self.bias_final)
        else:
            output_linear = F.linear(input, self.weight, self.bias_final)
        if self.type_pruning == "layer_wise":
            self.layer_wise_loss = torch.sum(output_linear**2)
            # self.layer_wise_loss.backward(retain_graph=True)
            # with torch.no_grad():
            #     self.impact_of_pruning += copy.deepcopy(self.weight.grad*(-self.weight.data)+self.weight_z.grad*(-self.gamma/2-self.weight_z.data))
        return output_linear

    @torch.no_grad()
    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        #self.weight_z.data.uniform_(-self.gamma/100, self.gamma/100)
        self.weight_z.data.uniform_(self.gamma, self.gamma)

        if self.test_bias:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
            #self.bias_z.data.uniform_(-self.gamma/100, self.gamma/100)
            if self.prune_bias:
                self.bias_z.data.uniform_(self.gamma, self.gamma)

    @torch.no_grad()
    def reset_weight_z(self, tol_z_1, prop_reset=1.0, type_pruning="magnitude", generator=None, type_reset="layer_wise", method_pruning="schedule", threshold_restart=1e-4) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        n_to_reset = int(prop_reset*self.n_weights)
        n_reset = 0
        if "magnitude_with_z" in type_pruning:
            idx_weight = torch.argsort(torch.abs(self.weight*self.z).view(-1))
        elif "magnitude" in type_pruning:
            idx_weight = torch.argsort(torch.abs(self.weight).view(-1))
        elif type_pruning=="layer_wise" or "smallest_grad" in type_pruning:
            if self.step_pruning == 0:
                print("-- First step --")
                idx_weight = torch.argsort(torch.abs(self.weight).view(-1))
            else:
                idx_weight = torch.argsort(torch.abs(self.impact_of_pruning).view(-1))
            self.step_pruning += 1
        
        # TO DELETE LATER
        # self.weight_before = copy.deepcopy(self.weight)
        # self.idx_weight_before = copy.deepcopy(idx_weight)
        # END

        idx_weight = idx_weight[self.z.view(-1)[idx_weight]>=tol_z_1]
        if self.test_bias and self.prune_bias:
            idx_bias = torch.argsort(torch.abs(self.bias).view(-1))
            idx_bias = idx_bias[self.z_2.view(-1)[idx_bias]>=tol_z_1]
            n_bias_left = len(idx_bias)
            n_weights_left = len(idx_weight)
            n_to_reset_weight = int(n_to_reset*n_weights_left/(n_bias_left+n_weights_left))
            n_to_reset_bias = int(n_to_reset*n_bias_left/(n_bias_left+n_weights_left))
        else:
            n_to_reset_weight = n_to_reset

        if method_pruning=="schedule":
            idx_weight = idx_weight[:n_to_reset_weight]
        elif method_pruning in ["threshold", "both"]:
            n_to_reset_weight = min(n_to_reset_weight, torch.sum(torch.abs(self.weight).view(-1)[self.z.view(-1)>=tol_z_1] <= threshold_restart).item())
            idx_weight = idx_weight[:n_to_reset_weight]
        else:
            print("ERROR: method_pruning should be either shedule, threshold or both")
        idx_weight = torch.sort(idx_weight).values

        # TO DELETE LATER
        # self.idx_weight_after = copy.deepcopy(idx_weight)
        # END

        n_reset+=len(idx_weight)
        self.weight_z.view(-1).data[idx_weight] = self.weight_z.view(-1).data[idx_weight].uniform_(-self.gamma/100, self.gamma/100, generator=generator).to(self.weight)
        if self.test_bias and self.prune_bias:
            if method_pruning=="schedule":
                idx_bias = idx_bias[:n_to_reset_bias]
            elif method_pruning in ["threshold", "both"]:
                n_to_reset_bias = min(n_to_reset_bias, torch.sum(torch.abs(self.bias).view(-1)[self.z_2.view(-1)>=tol_z_1] <= threshold_restart).item())
                idx_bias = idx_bias[:n_to_reset_bias]
            else:
                print("ERROR: method_pruning should be either shedule, threshold or both")
            n_reset+=len(idx_bias)
            self.bias_z.view(-1).data[idx_bias] = self.bias_z.view(-1).data[idx_bias].uniform_(-self.gamma/100, self.gamma/100, generator=generator).to(self.weight)
            
        return n_reset

    @torch.no_grad()
    def sparsify(self) -> None:
        sparse_weight = self.weight.data.to_sparse()
        self.sparse_weight_values = Parameter(sparse_weight.values())
        self.sparse_weight_indices = Parameter(sparse_weight.indices(), requires_grad=False)
        self.sparse_weight_size = sparse_weight.size()
        del self.weight
        sparse_weight_z = self.weight_z.data.to_sparse()
        self.sparse_weight_z_values = Parameter(sparse_weight_z.values())
        del self.weight_z
        if self.test_bias and self.prune_bias:
            sparse_bias = self.bias.data.to_sparse()
            self.sparse_bias_values = Parameter(sparse_bias.values())
            self.sparse_bias_indices = Parameter(sparse_bias.indices(), requires_grad=False)
            self.sparse_bias_size = sparse_bias.size()
            del self.bias
            sparse_bias_z = self.bias_z.data.to_sparse()
            self.sparse_bias_z_values = Parameter(sparse_bias_z.values())
            del self.bias_z
        self.is_sparse = True

    @torch.no_grad()
    def freeze_z(self) -> None:
        self.weight_z.requires_grad = False
        if self.test_bias and self.prune_bias:
            self.bias_z.requires_grad = False

    @torch.no_grad()
    def unfreeze_z(self) -> None:
        self.weight_z.requires_grad = True
        if self.test_bias and self.prune_bias:
            self.bias_z.requires_grad = True

    @torch.no_grad()
    def freeze_weight(self) -> None:
        self.weight.requires_grad = False
        if self.test_bias:
            self.bias.requires_grad = False

    @torch.no_grad()
    def unfreeze_weight(self) -> None:
        self.weight.requires_grad = True
        if self.test_bias:
            self.bias.requires_grad = True

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    def __deepcopy__(self, memo):
        out = Linear_with_z(
                in_features = self.in_features,
                out_features = self.out_features,
                bias = self.test_bias,
                gamma = self.gamma,
                prune_bias= self.prune_bias)
    
        out.in_features = self.in_features
        out.out_features = self.out_features
        out.gamma = self.gamma
        out.is_sparse = self.is_sparse
        out.weight = copy.deepcopy(self.weight, memo)
        out.weight_z = copy.deepcopy(self.weight_z, memo)
        out.test_bias = self.test_bias
        out.prune_bias = self.prune_bias
        out.use_mask = self.use_mask

        if self.test_bias:
            out.bias = copy.deepcopy(self.bias, memo)
            if self.prune_bias:
                out.bias_z = copy.deepcopy(self.bias_z, memo)
        return out

def conv3x3(in_planes, out_planes, stride=1, gamma=1.0, with_z=True, prune_bias=True):
    """3x3 convolution with padding"""
    if with_z:
        return Conv2d_with_z(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, gamma=gamma, prune_bias=prune_bias)
    else:
        return torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class _ConvNd_with_z(torch.nn.Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']
    __annotations__ = {'bias': Optional[torch.Tensor]}

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        ...

    in_channels: int
    _reversed_padding_repeated_twice: List[int]
    out_channels: int
    kernel_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Union[str, Tuple[int, ...]]
    dilation: Tuple[int, ...]
    transposed: bool
    output_padding: Tuple[int, ...]
    groups: int
    padding_mode: str
    weight: Tensor
    bias: Optional[Tensor]

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, ...],
                 stride: Tuple[int, ...],
                 padding: Tuple[int, ...],
                 dilation: Tuple[int, ...],
                 transposed: bool,
                 output_padding: Tuple[int, ...],
                 groups: int,
                 bias: bool,
                 padding_mode: str,
                 device=None,
                 dtype=None,
                 gamma:float = 1.0,
                 prune_bias=True) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_strings = {'same', 'valid'}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(
                    "Invalid padding string {!r}, should be one of {}".format(
                        padding, valid_padding_strings))
            if padding == 'same' and any(s != 1 for s in stride):
                raise ValueError("padding='same' is not supported for strided convolutions")

        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        self.gamma=gamma
        self.prune_bias=prune_bias
        self.use_mask=True
        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
            if padding == 'same':
                for d, k, i in zip(dilation, kernel_size,
                                   range(len(kernel_size) - 1, -1, -1)):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                        total_padding - left_pad)
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)

        if transposed:
            self.weight = Parameter(torch.empty(
                (in_channels, out_channels // groups, *kernel_size), **factory_kwargs))
            self.weight_z = Parameter(torch.empty(
                (in_channels, out_channels // groups, *kernel_size), **factory_kwargs))
        else:
            self.weight = Parameter(torch.empty(
                (out_channels, in_channels // groups, *kernel_size), **factory_kwargs))
            self.weight_z = Parameter(torch.empty(
                (out_channels, in_channels // groups, *kernel_size), **factory_kwargs))
        
        if bias:
            self.bias = Parameter(torch.empty(out_channels, **factory_kwargs))
            if self.prune_bias:
                self.bias_z = Parameter(torch.empty(out_channels, **factory_kwargs))
            else:
                self.register_parameter('bias_z', None)
            self.test_bias = True
        else:
            self.register_parameter('bias', None)
            self.register_parameter('bias_z', None)
            self.test_bias = False

        self.reset_parameters()

    def compute_z(self) -> None:
        self.z = compute_z_from_tensor(self.weight_z, self.gamma, self.type_function)

        if self.test_bias and self.prune_bias:
            self.z_2 = compute_z_from_tensor(self.bias_z, self.gamma, self.type_function)
            if self.use_mask:
                self.bias_final = self.bias*self.z_2
            else:
                self.bias_final = self.bias#*1.0
        elif self.test_bias:
            self.bias_final = self.bias#*1.0
        else:
            self.bias_final = self.bias

    @torch.no_grad()
    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight_z.data.uniform_(self.gamma, self.gamma)

        if self.test_bias:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)
                if self.prune_bias:
                    self.bias_z.data.uniform_(self.gamma, self.gamma)

    @torch.no_grad()
    def reset_weight_z(self, tol_z_1, prop_reset=1.0, type_pruning="magnitude", generator=None, type_reset="layer_wise", method_pruning="schedule", threshold_restart=1e-4) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        n_to_reset = int(prop_reset*self.n_weights)
        n_reset = 0
        if "magnitude_with_z" in type_pruning:
            idx_weight = torch.argsort(torch.abs(self.weight*self.z).view(-1))
        elif "magnitude" in type_pruning:
            idx_weight = torch.argsort(torch.abs(self.weight).view(-1))
        elif type_pruning=="layer_wise" or "smallest_grad" in type_pruning:
            if self.step_pruning == 0:
                print("-- First step --")
                idx_weight = torch.argsort(torch.abs(self.weight).view(-1))
            else:
                idx_weight = torch.argsort(torch.abs(self.impact_of_pruning).view(-1))
            self.step_pruning += 1
        
        # TO DELETE LATER
        # self.weight_before = copy.deepcopy(self.weight)
        # self.idx_weight_before = copy.deepcopy(idx_weight)
        # END

        idx_weight = idx_weight[self.z.view(-1)[idx_weight]>=tol_z_1]
        if self.test_bias and self.prune_bias:
            idx_bias = torch.argsort(torch.abs(self.bias).view(-1))
            idx_bias = idx_bias[self.z_2.view(-1)[idx_bias]>=tol_z_1]
            n_bias_left = len(idx_bias)
            n_weights_left = len(idx_weight)
            n_to_reset_weight = int(n_to_reset*n_weights_left/(n_bias_left+n_weights_left))
            n_to_reset_bias = int(n_to_reset*n_bias_left/(n_bias_left+n_weights_left))
        else:
            n_to_reset_weight = n_to_reset

        if method_pruning=="schedule":
            idx_weight = idx_weight[:n_to_reset_weight]
        elif method_pruning in ["threshold", "both"]:
            n_to_reset_weight = min(n_to_reset_weight, torch.sum(torch.abs(self.weight).view(-1) <= threshold_restart).item())
            idx_weight = idx_weight[:n_to_reset_weight]
        else:
            print("ERROR: method_pruning should be either shedule, threshold or both")
        idx_weight = torch.sort(idx_weight).values

        # TO DELETE LATER
        # self.idx_weight_after = copy.deepcopy(idx_weight)
        # END

        n_reset+=len(idx_weight)
        self.weight_z.view(-1).data[idx_weight] = self.weight_z.view(-1).data[idx_weight].uniform_(-self.gamma/100, self.gamma/100, generator=generator).to(self.weight)
        if self.test_bias and self.prune_bias:
            if method_pruning=="schedule":
                idx_bias = idx_bias[:n_to_reset_bias]
            elif method_pruning in ["threshold", "both"]:
                n_to_reset_bias = min(n_to_reset_bias, torch.sum(torch.abs(self.bias).view(-1) <= threshold_restart).item())
                idx_bias = idx_bias[:n_to_reset_bias]
            else:
                print("ERROR: method_pruning should be either shedule, threshold or both")
            n_reset+=len(idx_bias)
            self.bias_z.view(-1).data[idx_bias] = self.bias_z.view(-1).data[idx_bias].uniform_(-self.gamma/100, self.gamma/100, generator=generator).to(self.weight)
            
        return n_reset

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'

class Conv2d_with_z(_ConvNd_with_z):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None,
        gamma:float=1.0,
        prune_bias=True
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, gamma=gamma, prune_bias=prune_bias, **factory_kwargs)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        self.compute_z()
        if self.use_mask:
            return self._conv_forward(input, self.weight*self.z, self.bias_final)
        else:
            return self._conv_forward(input, self.weight, self.bias_final)

def compute_sparsity(model, n_params_original, type_compute_sparsity, gamma, to_prune_within_block):
    if type_compute_sparsity == "total":
        number_of_non_zero_params = np.sum([torch.sum(x[1]!=0).item() if ("z" not in x[0] and x[0] in to_prune_within_block) else 0 for x in model.named_parameters()])
    else:
        number_of_non_zero_params = np.sum([torch.sum(x[1]>-gamma/2).item() if ("z" in x[0] and x[0] in to_prune_within_block) else 0 for x in model.named_parameters()])
    return 1-number_of_non_zero_params/n_params_original

def compute_sparsity_storage(model, n_params_original):
    number_params = np.sum([np.prod(x[1].shape) if "z" not in x[0] else 0 for x in model.named_parameters()])
    return 1-number_params/n_params_original

class model_wrapper():
    def __init__(self, model, optimizer, seed, entropy_reg, selection_reg, l2_reg, device, dense_to_sparse, test_different_lr, steps_per_epoch, val_second_lr, momentum, weight_decay, tol_z_1, input_channel, type_pruning, generator, type_reset, method_pruning, threshold_restart, selection_lagrangian_reg, entropy_lagrangian_reg, l2_original_reg, original_model, type_function, test_mult_reset, test_reset_to_orignal, prune_bias, type_compute_sparsity, gamma) -> None:
        self.model = model
        self.optimizer = optimizer
        self.prune_bias = prune_bias
        self.type_compute_sparsity = type_compute_sparsity
        self.seed = seed
        self.gamma = gamma
        self.entropy_reg = entropy_reg
        self.selection_reg = selection_reg
        self.l2_reg = l2_reg
        self.device = device
        self.dense_to_sparse = dense_to_sparse
        self.test_different_lr = test_different_lr
        self.steps_per_epoch = steps_per_epoch
        self.val_second_lr = val_second_lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.n_params_original = np.sum([np.prod(x[1].shape) if ("z" not in x[0]) else 0 for x in model.named_parameters()])
        self.n_params_original_z = np.sum([np.prod(x[1].shape) if ("z" in x[0]) else 0 for x in model.named_parameters()])
        if "VisionTransformer" in self.model.__str__():
            try:
                self.n_params_original_layers = np.sum([np.prod(x[1].shape) if ("z" not in x[0]) else 0 for x in model.blocks.named_parameters()])
                self.n_params_original_layers_z = np.sum([np.prod(x[1].shape) if ("z" in x[0]) else 0 for x in model.blocks.named_parameters()])
            except:
                self.n_params_original_layers = -1
            self.to_prune_within_block = [x[0] for x in self.model.blocks.named_parameters() if x[1].requires_grad]
        elif "OPT" in self.model.__str__():
            try:
                self.n_params_original_layers = np.sum([np.prod(x[1].shape) if ("z" not in x[0]) else 0 for x in model.model.decoder.layers.named_parameters()])
                self.n_params_original_layers_z = np.sum([np.prod(x[1].shape) if ("z" in x[0]) else 0 for x in model.model.decoder.layers.named_parameters()])
            except:
                self.n_params_original_layers = -1
            self.to_prune_within_block = [x[0] for x in self.model.named_parameters() if x[1].requires_grad]
        else:
            self.n_params_original_layers = -1
            self.to_prune_within_block = [x[0] for x in self.model.named_parameters() if x[1].requires_grad]
        self.tol_z_1 = tol_z_1
        self.type_pruning = type_pruning
        self.generator = generator
        self.type_reset = type_reset
        self.method_pruning = method_pruning
        self.threshold_restart = threshold_restart
        self.selection_lagrangian_reg = selection_lagrangian_reg
        self.entropy_lagrangian_reg = entropy_lagrangian_reg
        self.l2_original_reg = l2_original_reg
        self.original_model = original_model
        if self.original_model!=None:
            self.original_model.to(device)
        self.sparsity_level_selection = 0.0
        self.sparsity_level_entropy = 0.0
        self.test_sparsity_reached = False
        self.type_function = type_function
        self.test_mult_reset = test_mult_reset
        self.test_reset_to_orignal = test_reset_to_orignal
        if dense_to_sparse:
            self.model.idx_keep_input = np.arange(input_channel)

    def compute_layer_wise_loss(self):
        return compute_layer_wise_loss_rec(self.model, self.device)
    
    def save_grad_layer_wise(self):
        return save_grad_layer_wise_rec(self.model)

    def initialize_pruning(self):
        return initialize_pruning_rec(self.model, self.type_pruning, self.type_function)

    def reinitialize_pruning(self):
        return reinitialize_pruning_rec(self.model, self.type_pruning)

    def get_sparsity(self):
        if self.type_compute_sparsity == "total":
            n_total_params_to_use = self.n_params_original
        else:
            n_total_params_to_use = self.n_params_original_z
        return compute_sparsity(self.model, n_total_params_to_use, self.type_compute_sparsity, self.gamma, self.to_prune_within_block)

    def get_final_sparsity(self):
        if self.type_compute_sparsity == "total":
            if self.n_params_original_layers==-1:
                n_total_params_to_use = self.n_params_original
            else:
                n_total_params_to_use = self.n_params_original_layers
        else:
            if self.n_params_original_layers==-1:
                n_total_params_to_use = self.n_params_original_z
            else:
                n_total_params_to_use = self.n_params_original_layers_z

        if self.n_params_original_layers==-1:
            return compute_sparsity(self.model, n_total_params_to_use, self.type_compute_sparsity, self.gamma, self.to_prune_within_block)
        try:
            return compute_sparsity(self.model.model.decoder.layers, n_total_params_to_use, self.type_compute_sparsity, self.gamma, self.to_prune_within_block)
        except:
            return compute_sparsity(self.model.blocks, n_total_params_to_use, self.type_compute_sparsity, self.gamma, self.to_prune_within_block)

    def get_sparsity_storage(self):
        return compute_sparsity_storage(self.model, self.n_params_original)

    def get_losses(self) -> float:

        if self.n_params_original_z == 0:
            return torch.tensor(0), torch.tensor(0), torch.tensor(0)
        else:
            max_entropy_reg = -torch.log(torch.tensor(0.5+1e-6))

            if self.selection_lagrangian_reg!=None:
                selection_reg = self.selection_lagrangian_reg
            else:
                selection_reg = self.selection_reg

            if self.entropy_lagrangian_reg != None:
                entropy_reg = self.entropy_lagrangian_reg
            else:
                entropy_reg = self.entropy_reg

            entropy_loss, selection_loss, l2_loss, l2_original_loss = compute_losses_rec(self.model, self.device, entropy_reg, selection_reg, self.l2_reg, self.l2_original_reg, self.original_model)
            # print(l2_original_loss)
            # print("1.",selection_lagrangian_loss.item(), (self.threshold_restart*self.n_params_original_z*0.1).item())

            selection_loss /= self.n_params_original_z

            if self.selection_lagrangian_reg!=None:
                selection_loss -= (1-self.sparsity_level_selection)

            entropy_loss /= self.n_params_original_z
            if self.entropy_lagrangian_reg!= None:
                entropy_loss_diff = entropy_loss - (1-self.sparsity_level_entropy)*max_entropy_reg

            l2_loss /= self.n_params_original_z
            if self.l2_original_reg!=0 and l2_original_loss.item()>=1:
                l2_original_loss /= l2_original_loss.item()
            l2_original_loss *= self.l2_original_reg

            if self.selection_lagrangian_reg!=None and (selection_loss<0 or self.selection_lagrangian_reg<0):
                self.selection_lagrangian_reg.data = torch.tensor(self.selection_reg)
                selection_loss = 0.0
                self.sparsity_level_selection+=min(0.1, self.goal_sparsity-self.sparsity_level_selection)
                # print(self.sparsity_level_selection)
                # print("2.",self.selection_lagrangian_reg.item(), selection_lagrangian_loss.item())
                # print("2.",100*self.l2_reg, selection_lagrangian_loss.item())

            if self.entropy_lagrangian_reg!=None and (entropy_loss_diff<0 or self.entropy_lagrangian_reg<0):
                self.entropy_lagrangian_reg.data = torch.tensor(self.entropy_reg)
                entropy_loss_diff = 0.0
                self.sparsity_level_entropy+=min(0.1, self.goal_sparsity-self.sparsity_level_entropy)
            
            if self.entropy_lagrangian_reg!= None:
                entropy_loss += entropy_loss_diff

            entropy_loss *= entropy_reg
            selection_loss *= selection_reg
            l2_loss *= self.l2_reg

            if l2_loss<0:
                import ipdb;ipdb.set_trace()
            if selection_loss<0:
                import ipdb;ipdb.set_trace()
            if entropy_loss<0:
                import ipdb;ipdb.set_trace()
            if l2_original_loss<0:
                import ipdb;ipdb.set_trace()
            #print(entropy_loss.item(), selection_loss.item(), l2_loss.item(), selection_lagrangian_loss.item(), l2_original_loss.item())
            return entropy_loss, selection_loss, l2_loss+l2_original_loss
            # entropy_loss = torch.tensor(0.0).to(self.device)
            # selection_loss = torch.tensor(0.0).to(self.device)
            # l2_loss = torch.tensor(0.0).to(self.device)
            # d_children = dict(self.model.named_children())
            # for name_child in d_children:
            #     if not("relu" in name_child) and not("avgpool" in name_child) and not("bn" in name_child):
            #         child = d_children[name_child]
            #         if self.entropy_reg!=0:
            #             entropy_loss += -self.entropy_reg*(torch.sum(child.z*torch.log(child.z+tol_instability) + (1-child.z)*torch.log(1-child.z+tol_instability)))
            #             if child.bias != None:
            #                 entropy_loss += -self.entropy_reg*(torch.sum(child.z_2*torch.log(child.z_2+tol_instability) + (1-child.z_2)*torch.log(1-child.z_2+tol_instability)))
            #         if self.selection_reg !=0:
            #             selection_loss += self.selection_reg*torch.sum(child.z)
            #             if child.bias != None:
            #                 selection_loss += self.selection_reg*torch.sum(child.z_2)
            #         if self.l2_reg !=0:
            #             l2_loss += self.l2_reg*torch.sum(child.weight**2)
            #             if child.bias != None:
            #                 l2_loss += self.l2_reg*torch.sum(child.bias**2)
            # return entropy_loss, selection_loss, l2_loss

    def maximum_value(self):
        l_params = list(self.model.parameters())
        max_value = -np.inf
        for param in l_params:
            max_value = np.max([torch.max(torch.abs(param)).item(), max_value])
        return max_value

    def maximum_gradient(self):
        l_params = list(self.model.parameters())
        max_value = -np.inf
        for param in l_params:
            try:
                max_value = np.max([torch.max(torch.abs(param.grad)).item(), max_value])
            except:
                max_value = np.max([-np.inf, max_value])
        return max_value

    def compute_z(self) -> float:
        compute_z_rec(self.model)
        # d_children = dict(self.model.named_children())
        # for name_child in d_children:
        #     child = d_children[name_child]
        #     test_children = False
        #     try:
        #         child.named_children()
        #         test_children = True
        #     except:
        #         pass
        #     if test_children:
        #         child.compute_z()
        #     # if "layer" in name_child:
        #     #     d_children_layer = dict(child.named_children())
        #     #     for name_child_layer in d_children_layer:
        #     #         child_layer = d_children_layer[name_child_layer]
        #     #         if not("relu" in name_child_layer) and not("avgpool" in name_child_layer) and not("bn" in name_child_layer):
        #     #             child_layer.compute_z()
        #     elif not("relu" in name_child) and not("avgpool" in name_child) and not("bn" in name_child):
        #         try:
        #             child.compute_z()
        #         except:
        #             import ipdb;ipdb.set_trace()

    @torch.no_grad()
    def get_n_z(self, test_grad=True, include_batchnorm=True) -> float:
        return compute_n_z_rec(self.model, test_grad, include_batchnorm, self.type_function)
        # n_z = 0
        # d_children = dict(self.model.named_children())
        # for name_child in d_children:
        #     if not("relu" in name_child) and not("avgpool" in name_child) and not("bn" in name_child):
        #         child = d_children[name_child]
        #         n_z += torch.sum(child.z>0, dtype=float)
        # return n_z.detach().item()

    @torch.no_grad()
    def get_n_z_close_to_1(self, test_grad=True, include_batchnorm=True) -> float:
        return compute_n_z_close_to_1_rec(self.model, self.tol_z_1, test_grad, include_batchnorm)
        # n_z = 0
        # d_children = dict(self.model.named_children())
        # for name_child in d_children:
        #     if not("relu" in name_child) and not("avgpool" in name_child) and not("bn" in name_child):
        #         child = d_children[name_child]
        #         n_z += torch.sum(child.z>=self.tol_z_1, dtype=float)
        # return n_z.detach().item()
    
    @torch.no_grad()
    def get_n_weigth_z_close_to_1(self, test_grad=True, include_batchnorm=True) -> float:
        return compute_n_weight_z_close_to_1_rec(self.model, self.tol_z_1, test_grad, include_batchnorm)
        # n_z = 0
        # d_children = dict(self.model.named_children())
        # for name_child in d_children:
        #     if not("relu" in name_child) and not("avgpool" in name_child) and not("bn" in name_child):
        #         child = d_children[name_child]
        #         n_z += torch.sum(child.z>=self.tol_z_1, dtype=float)
        # return n_z.detach().item()

    @torch.no_grad()
    def reset_z(self, prop_reset = 0.5, test_grad=True) -> float:
        #if self.type_pruning == "layer_wise":
        if self.type_reset=="layer_wise":
            n_reset = reset_z_rec(self.model, self.tol_z_1, prop_reset=prop_reset, type_pruning=self.type_pruning, generator=self.generator, test_grad=test_grad, type_reset=self.type_reset, method_pruning=self.method_pruning, threshold_restart=self.threshold_restart, test_mult_reset= self.test_mult_reset)
        else:
            # # if initialization:
            # #     input_tol = 0.0
            # # else:
            # input_tol = self.tol_z_1
            # n_reset = 0
            # with torch.no_grad():
            #     d_children = dict(self.model.named_children())
            #     for name_child in d_children:
            #         if not("relu" in name_child) and not("avgpool" in name_child) and not("bn" in name_child):
            #             child = d_children[name_child]
            #             # if initialization:
            #             #     old_z = torch.ones_like(child.weight_z)
            #             #     if child.test_bias:
            #             #         old_bias_z = torch.ones_like(child.bias_z)
            #             # else:
            #             child.compute_z()
            #             old_z = child.z
            #             if child.test_bias:
            #                 old_bias_z = child.z_2

            #             n_reset += child.reset_weight_z(input_tol, prop_reset)
            #             child.compute_z()
            #             new_z = child.z
            #             child.weight.data[new_z>0] *= (old_z/new_z)[new_z>0]
            #             if child.test_bias:
            #                 new_bias_z = child.z_2
            #                 child.bias.data[new_bias_z>0] *= (old_bias_z/new_bias_z)[new_bias_z>0]
            # self.add_loss = 0
            n_reset = 0
            n_to_reset = int(prop_reset*self.n_params_original_z)
            n_to_reset = max(n_to_reset,1)
            d_modules = dict(self.model.named_modules())
            l_name_modules = list(d_modules.keys())
            # l_name_modules = [x for x in l_name_modules if len(d_modules[x].__str__().lower().split(":"))==1 and (("conv" in d_modules[x].__str__().lower()) or ("linear_with" in d_modules[x].__str__().lower()))]
            l_name_modules = [x for x in l_name_modules if len(d_modules[x].__str__().lower().split(":"))==1 and ("with_z" in d_modules[x].__str__().lower())]

            start_time = time.time()
            torch_weights = torch.zeros((0, 3))
            torch_z = torch.zeros((0))
            n_modules = len(l_name_modules)
            for ind_module in range(n_modules):
                name_module = l_name_modules[ind_module]
                module = d_modules[name_module]
                torch_z = torch.cat([torch_z, module.z.view(-1).detach().cpu()])
                if "magnitude_with_z" in self.type_pruning:
                    weights_metric = torch.abs(module.weight*module.z)
                elif "magnitude" in self.type_pruning:
                    weights_metric = torch.abs(module.weight)
                    #weights_metric = weights_metric/torch.norm(weights_metric, 2)
                elif self.type_pruning=="layer_wise" or "smallest_grad" in self.type_pruning:
                    if module.step_pruning == 0:
                        print("-- First step --")
                        weights_metric = torch.abs(module.weight)
                    else:
                        weights_metric = torch.abs(module.impact_of_pruning)
                    module.step_pruning += 1

                if "_H" in self.type_pruning:
                    percdamp=.01
                    current_H = copy.deepcopy(self.gpts[name_module].H)
                    dead = torch.diag(current_H) == 0
                    current_H[dead, dead] = 1
                    damp = percdamp * torch.mean(torch.diag(current_H))
                    diag = torch.arange(current_H.shape[0], device=self.device)
                    current_H[diag, diag] += damp
                    H_inv = torch.linalg.cholesky(current_H)
                    H_inv = torch.cholesky_inverse(H_inv)
                    H_inv = torch.linalg.cholesky(H_inv, upper=True)
                    weights_metric = torch.abs(weights_metric / (torch.diag(H_inv).reshape((1, -1))))

                weights_metric = weights_metric.view(-1)
                n_weights = len(weights_metric)
                torch_weights = torch.cat([torch_weights, torch.stack([torch.arange(n_weights), ind_module*torch.ones(n_weights), weights_metric.detach().cpu()]).T])
                if module.test_bias and module.prune_bias:
                    torch_z = torch.cat([torch_z, module.z_2.view(-1).detach().cpu()])
                    if "magnitude_with_z" in self.type_pruning:
                        bias_metric = torch.abs(module.bias*module.z_2).view(-1)
                    elif "magnitude" in self.type_pruning:
                        bias_metric = torch.abs(module.bias).view(-1)
                    elif self.type_pruning=="layer_wise" or "smallest_grad" in self.type_pruning:
                        if module.step_pruning == 0:
                            print("-- First step --")
                            bias_metric = torch.abs(module.bias).view(-1)
                        else:
                            bias_metric = torch.abs(module.impact_of_pruning_2).view(-1)
                    n_bias = len(bias_metric)
                    torch_weights = torch.cat([torch_weights, torch.stack([torch.arange(n_bias), (ind_module+n_modules)*torch.ones(n_bias), bias_metric.detach().cpu()]).T])

            print("Time to concat: ", time.time()-start_time)
            start_time = time.time()
            selection_z = torch_z>=self.tol_z_1
            #torch_weights = torch_weights[selection_z][idx_smallest_n_to_reset]

            # torch_z = torch_z[torch.argsort(torch_weights[:,-1])]
            # torch_weights = torch_weights[torch.argsort(torch_weights[:,-1])]
            # print(f"Number of weights smaller than 1e-5: {torch.sum(torch_weights[selection_z][:,-1]<=1e-5)}")
            # print(f"Number of weights smaller than 1e-4: {torch.sum(torch_weights[selection_z][:,-1]<=1e-4)}")
            # print(f"Number of weights smaller than 1e-3: {torch.sum(torch_weights[selection_z][:,-1]<=1e-3)}")
            # TO CHANGE LATER:
            # inds_for_loss = torch_weights[selection_z][:n_to_reset, :2]
            # if self.selection_lagrangian_reg!=None:
            #     try:
            #         self.selection_lagrangian_reg.data = torch.tensor(self.l2_reg)
            #     except:
            #         import ipdb;ipdb.set_trace()
            # print(inds_for_loss.shape)
            # End

            if self.method_pruning == "schedule":
                idx_smallest_n_to_reset = torch.topk(torch_weights[selection_z][:,-1], n_to_reset, largest=False, sorted=False).indices
                #torch_weights = torch_weights[selection_z][:n_to_reset]
                torch_weights = torch_weights[selection_z][idx_smallest_n_to_reset]
            elif self.method_pruning in ["threshold", "both"]:
                n_to_reset_threshold = min(n_to_reset, torch.sum(torch_weights[selection_z][:,-1]<=self.threshold_restart).item())
                idx_smallest_n_to_reset = torch.topk(torch_weights[selection_z][:,-1], n_to_reset_threshold, largest=False, sorted=False).indices
                #torch_weights = torch_weights[selection_z][:n_to_reset_threshold]
                torch_weights = torch_weights[selection_z][idx_smallest_n_to_reset]
            else:
                print("ERROR: method_pruning should be either shedule, threshold or both")

            print("Time to sort them: ", time.time()-start_time)
            start_time = time.time()

            for ind_module in range(len(l_name_modules)):
                idx_to_reset = torch_weights[torch_weights[:,1]==ind_module][:,0].long()
                # idx_to_reset = torch.sort(idx_to_reset).values
                name_module = l_name_modules[ind_module]
                module = d_modules[name_module]

                #idx_weight_to_add_loss = inds_for_loss[inds_for_loss[:,1]==ind_module][:,0].long()
                #idx_weight_to_add_loss = torch_weights[torch_weights[:,1]==ind_module][:,0].long()

                # idx_weight_to_add_loss = torch.sort(idx_weight_to_add_loss).values
                module.idx_weight_to_add_loss = idx_to_reset
                # self.add_loss += torch.sum(torch.abs(module.weight.view(-1)[idx_to_add_loss]))

                old_z = module.z
                if module.test_bias and module.prune_bias:
                    old_bias_z = module.z_2
                n_reset+=len(idx_to_reset)
                module.weight_z.view(-1).data[idx_to_reset] = module.weight_z.view(-1).data[idx_to_reset].uniform_(-module.gamma/100, module.gamma/100, generator=self.generator).to(module.weight)
                
                if module.test_bias and module.prune_bias:
                    idx_bias = torch_weights[torch_weights[:,1]==(ind_module+n_modules)][:,0].long()

                    #idx_bias_to_add_loss = inds_for_loss[inds_for_loss[:,1]==(ind_module+n_modules)][:,0].long()
                    #idx_bias_to_add_loss = torch_weights[torch_weights[:,1]==(ind_module+n_modules)][:,0].long()

                    # idx_bias_to_add_loss = torch.sort(idx_bias_to_add_loss).values
                    module.idx_bias_to_add_loss = idx_bias
                    # self.add_loss += torch.sum(torch.abs(module.bias.view(-1)[idx_to_add_loss]))

                    # idx_bias = torch.sort(idx_bias).values
                    n_reset+=len(idx_bias)
                    module.bias_z.view(-1).data[idx_bias] = module.bias_z.view(-1).data[idx_bias].uniform_(-module.gamma/100, module.gamma/100, generator=self.generator).to(module.weight)

                module.compute_z()
                new_z = module.z
                module.weight.data[new_z==0] = 0
                if self.test_mult_reset:
                    module.weight.data[new_z>0] *= (old_z/new_z)[new_z>0]
                if module.test_bias and module.prune_bias:
                    new_bias_z = module.z_2
                    module.bias.data[new_bias_z==0] = 0
                    if self.test_mult_reset:
                        module.bias.data[new_bias_z>0] *= (old_bias_z/new_bias_z)[new_bias_z>0]
        print("Time to reset everything: ", time.time()-start_time)
        start_time = time.time()

        return n_reset

    @torch.no_grad()
    def set_require_grad(self, is_grad_required) -> float:
        set_require_grad_rec(self.model, is_grad_required)
        return None
    
    @torch.no_grad()
    def multiply_weight(self) -> None:
        parameters = self.model.named_parameters()
        for parameter in parameters:
            if not("_z" in parameter[0]) and not("indices" in parameter[0]):
                parameter[1].data = 2*parameter[1].data

    @torch.no_grad()
    def multiply_weight_copy(self) -> None:
        parameters = self.model.named_parameters()
        for parameter in parameters:
            if not("_z" in parameter[0]) and not("indices" in parameter[0]):
                parameter[1].data = 2*parameter[1].data

    @torch.no_grad()
    def sparsify(self) -> None:
        d_children = dict(self.model.named_children())
        for name_child in d_children:
            if not("relu" in name_child) and not("avgpool" in name_child) and not("bn" in name_child):
                child = d_children[name_child]
                child.sparsify()

    @torch.no_grad()
    def phase_training_z(self) -> None:
        d_children = dict(self.model.named_children())
        for name_child in d_children:
            if not("relu" in name_child) and not("avgpool" in name_child) and not("bn" in name_child):
                child = d_children[name_child]
                #child.unfreeze_z()
                child.freeze_weight()

    @torch.no_grad()
    def phase_training_weight(self) -> None:
        d_children = dict(self.model.named_children())
        for name_child in d_children:
            if not("relu" in name_child) and not("avgpool" in name_child) and not("bn" in name_child):
                child = d_children[name_child]
                #child.freeze_z()
                child.unfreeze_weight()

    @torch.no_grad()
    def freeze_all_z(self, d_named_parameters) -> None:
        for x in d_named_parameters:
                if "_z" in x:
                    d_named_parameters[x].requires_grad = False

    def unfreeze_all_z(self, d_named_parameters) -> None:
        for x in d_named_parameters:
                if "_z" in x:
                    d_named_parameters[x].requires_grad = True

    # def unfreeze_all_z(self) -> None:
    #     d_children = dict(self.model.named_children())
    #     for name_child in d_children:
    #         if not("relu" in name_child) and not("maxpool" in name_child) and not("avgpool" in name_child) and not("norm" in name_child) and not("view" in name_child) and not("log_softmax_mlpnet" in name_child):
    #             child = d_children[name_child]
    #             child.freeze_z()
    #             # child.unfreeze_weight()

    @torch.no_grad()
    def prune_models(self):
        if self.type_function == "smoothstep":
            d_modules = dict(self.model.named_modules())
            optimizer, test_pruned = prune_models_external(self, d_modules, self.optimizer, self.dense_to_sparse)
            self.optimizer = optimizer
        else:
            optimizer = self.optimizer
            test_pruned = False
        # if test_pruned:
        #     print("Weights were pruned, dense to sparse", self.dense_to_sparse)
        return optimizer, test_pruned
        # test_pruned = False
        # if self.dense_to_sparse:
        #     with torch.no_grad():
        #         d_children = dict(self.model.named_children())
        #         for name_child in d_children:
        #             if not("relu" in name_child) and not("avgpool" in name_child) and not("bn" in name_child):
        #                 child = d_children[name_child]
        #                 if len(child.z)>0 and torch.min(child.z) == 0:
        #                     optimizer_name = optimizer.__class__.__name__
        #                     idx_keep = torch.where(child.z!=0)[0]
        #                     copy_grad_weight = copy.deepcopy(child.sparse_weight_values.grad[idx_keep])
        #                     copy_grad_weight_z = copy.deepcopy(child.sparse_weight_z_values.grad[idx_keep])
        #                     sparse_weight_values_temp = Parameter(child.sparse_weight_values[idx_keep])
        #                     sparse_weight_z_values_temp = Parameter(child.sparse_weight_z_values[idx_keep])
        #                     if optimizer_name == "SGD":
        #                         optimizer.state[sparse_weight_values_temp]["momentum_buffer"] = copy.deepcopy(optimizer.state[child.sparse_weight_values].pop("momentum_buffer")[idx_keep])
        #                         optimizer.state[sparse_weight_z_values_temp]["momentum_buffer"] = copy.deepcopy(optimizer.state[child.sparse_weight_z_values].pop("momentum_buffer")[idx_keep])
        #                     elif optimizer_name == "Adam":
        #                         optimizer.state[sparse_weight_values_temp]["exp_avg"] = copy.deepcopy(optimizer.state[child.sparse_weight_values].pop("exp_avg")[idx_keep])
        #                         optimizer.state[sparse_weight_z_values_temp]["exp_avg"] = copy.deepcopy(optimizer.state[child.sparse_weight_z_values].pop("exp_avg")[idx_keep])
        #                         optimizer.state[sparse_weight_values_temp]["exp_avg_sq"] = copy.deepcopy(optimizer.state[child.sparse_weight_values].pop("exp_avg_sq")[idx_keep])
        #                         optimizer.state[sparse_weight_z_values_temp]["exp_avg_sq"] = copy.deepcopy(optimizer.state[child.sparse_weight_z_values].pop("exp_avg_sq")[idx_keep])
        #                     child.sparse_weight_values = sparse_weight_values_temp
        #                     child.sparse_weight_z_values = sparse_weight_z_values_temp
        #                     child.sparse_weight_values.grad = copy_grad_weight
        #                     child.sparse_weight_z_values.grad = copy_grad_weight_z
        #                     child.sparse_weight_indices = Parameter(child.sparse_weight_indices[:,idx_keep], requires_grad=False)
                            
        #                     if child.bias:
        #                         # NEED TO IMPLEMENT PRUNING FOR BIAS
        #                         import ipdb;ipdb.set_trace()
        #                     test_pruned = True

        #     if test_pruned:
        #         # self.update_indice_param()
        #         optimizer_name = optimizer.__class__.__name__
        #         copy_optimizer = initialize_optimizer(self.test_different_lr, self.model, optimizer_name, self.steps_per_epoch, optimizer.defaults["lr"], self.val_second_lr, self.momentum, self.weight_decay)
        #         try:
        #             copy_optimizer._step_count = optimizer._step_count
        #         except:
        #             pass
        #         for idx_param in range(len(optimizer.param_groups)):
        #             for key in optimizer.param_groups[idx_param]:
        #                 if key!="params":
        #                     copy_optimizer.param_groups[idx_param][key] = optimizer.param_groups[idx_param][key]
        #             list_params_old = list(optimizer.param_groups[idx_param]["params"])
        #             list_params_new = list(copy_optimizer.param_groups[idx_param]["params"])
        #             for i in range(len(list_params_new)):
        #                 group_param_old = list_params_old[i]
        #                 group_param_new = list_params_new[i]
        #                 copy_optimizer.state[group_param_new] = copy.deepcopy(optimizer.state[group_param_old])
        #     else:
        #         copy_optimizer = optimizer
        # else:
        #     with torch.no_grad():
        #         d_children = dict(self.model.named_children())
        #         for name_child in d_children:
        #             if not("relu" in name_child) and not("avgpool" in name_child) and not("bn" in name_child):
        #                 child = d_children[name_child]
        #                 if torch.min(child.z) == 0:
        #                     child.weight_z.data[child.z==0] = -child.gamma
        #                     test_pruned = True
        #     copy_optimizer = optimizer
        # return copy_optimizer, test_pruned

# %%
