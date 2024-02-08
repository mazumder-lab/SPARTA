import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed):
    """
    Set the seed for reproducibility in Python's random, NumPy, and PyTorch.

    Args:
    seed (int): The seed value to set.
    """
    random.seed(seed)  # Set seed for Python's standard random library
    np.random.seed(seed)  # Set seed for NumPy
    torch.manual_seed(seed)  # Set seed for PyTorch CPU operations

    if torch.cuda.is_available():
        # Set seed for PyTorch CUDA operations
        torch.cuda.manual_seed(seed)
        # Set seed for all GPUs (if using more than one)
        torch.cuda.manual_seed_all(seed)
        # Ensure CUDA operations are deterministic
        torch.backends.cudnn.deterministic = True
        # Disable dynamic algorithm selection for convolution
        torch.backends.cudnn.benchmark = False


def count_parameters(model, all_param_flag=False):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or all_param_flag)


@torch.no_grad()
def layerwise_magnitude_pruning(net_state_dict, new_net_state_dict, sparsity):
    for name in new_net_state_dict:
        if "mask" in name:
            original_name = name.replace("mask_", "").replace("_trainable", "")
            param = net_state_dict[original_name]
            threshold_value = torch.quantile(param.abs().reshape(-1), 1 - sparsity)
            layerwise_mask = (param > threshold_value).float()
            new_net_state_dict[name] = layerwise_mask.view_as(new_net_state_dict[name])
    return new_net_state_dict


@torch.no_grad()
def global_magnitude_pruning(net_state_dict, new_net_state_dict, sparsity):
    pvec = []
    for name in new_net_state_dict:
        if "mask" in name:
            original_name = name.replace("mask_", "").replace("_trainable", "")
            pvec.append(net_state_dict[original_name].reshape(-1))
    pvec = torch.cat(pvec).abs()
    threshold_value = torch.quantile(pvec, 1 - sparsity)
    global_mask = (pvec > threshold_value).float()
    pointer = 0
    for name in new_net_state_dict:
        if "mask" in name:
            param = new_net_state_dict[name]
            num_params = param.numel()
            new_net_state_dict[name] = global_mask[pointer : pointer + num_params].view_as(param)
            pointer += num_params
    return new_net_state_dict


@torch.no_grad()
def update_magnitude_mask(net: nn.Module, args):
    net_state_dict = net.state_dict()
    for original_name in net_state_dict:
        if "init" in original_name:
            name_mask = original_name.replace("init_", "mask_") + "_trainable"
            name_weight = original_name.replace("init_", "") + "_trainable"
            real_weight = net_state_dict[original_name] + net_state_dict[name_mask] * net_state_dict[name_weight]
            idx_weights = torch.argsort(
                real_weight.abs().flatten(),
                descending=args.magnitude_descending if args.magnitude_descending is not None else False,
            )
            idx_weights = idx_weights[: int(len(idx_weights) * (1 - args.sparsity))]
            new_tensor = torch.ones_like(real_weight).flatten()
            new_tensor[idx_weights] = 0
            net_state_dict[original_name] = real_weight
            net_state_dict[name_mask] = new_tensor.view_as(real_weight)
            net_state_dict[name_weight] = torch.zeros_like(real_weight)
    net.load_state_dict(net_state_dict)
    return net


@torch.no_grad()
def update_noisy_grad_mask(net: nn.Module, args):
    net_state_dict = net.state_dict()
    named_parameters = dict(net.named_parameters())
    for original_name in net_state_dict:
        if "init" in original_name:
            name_mask = original_name.replace("init_", "mask_") + "_trainable"
            name_weight = original_name.replace("init_", "") + "_trainable"
            real_weight = net_state_dict[original_name] + net_state_dict[name_mask] * net_state_dict[name_weight]
            noisy_grad = named_parameters[name_weight].grad
            # NOTE we just changed descending to True to keep smallest gradients in norm
            idx_weights = torch.argsort(noisy_grad.abs().flatten(), descending=args.magnitude_descending)
            idx_weights = idx_weights[: int(len(idx_weights) * (1 - args.sparsity))]
            new_tensor = torch.ones_like(noisy_grad).flatten()
            new_tensor[idx_weights] = 0
            net_state_dict[original_name] = real_weight
            net_state_dict[name_mask] = new_tensor.view_as(real_weight)
            net_state_dict[name_weight] = torch.zeros_like(real_weight)
    net.load_state_dict(net_state_dict)
    return net


# @torch.no_grad()
# def update_global_magnitude_mask(net: nn.Module, args):
#     net_state_dict = net.state_dict()
#     pvec = []
#     for original_name in net_state_dict:
#         if "init" in original_name:
#             name_mask = original_name.replace("init_", "mask_") + "_trainable"
#             name_weight = original_name.replace("init_", "") + "_trainable"
#             real_weight = net_state_dict[original_name] + net_state_dict[name_mask] * net_state_dict[name_weight]
#             net_state_dict[original_name] = real_weight
#             pvec.append(real_weight.reshape(-1))
#     pvec = torch.cat(pvec).abs()
#     threshold_value = torch.quantile(pvec, args.sparsity)
#     global_mask = (pvec > threshold_value).float()
#     pointer = 0
#     for original_name in net_state_dict:
#         if "init" in original_name:
#             param = net_state_dict[original_name]
#             name_mask = original_name.replace("init_", "mask_") + "_trainable"
#             name_weight = original_name.replace("init_", "") + "_trainable"
#             num_params = param.numel()
#             net_state_dict[name_mask] = global_mask[pointer : pointer + num_params].view_as(param)
#             net_state_dict[name_weight] = torch.zeros_like(param)
#             pointer += num_params
#     net.load_state_dict(net_state_dict)
#     return net


# @torch.no_grad()
# def update_global_noisy_grad_mask(net: nn.Module, args):
#     net_state_dict = net.state_dict()
#     named_parameters = dict(net.named_parameters())
#     for original_name in net_state_dict:
#         if "init" in original_name:
#             name_mask = original_name.replace("init_", "mask_") + "_trainable"
#             name_weight = original_name.replace("init_", "") + "_trainable"
#             real_weight = net_state_dict[original_name] + net_state_dict[name_mask] * net_state_dict[name_weight]
#             noisy_grad = named_parameters[name_weight].grad
#             # NOTE we just changed descending to True to keep smallest gradients in norm
#             idx_weights = torch.argsort(noisy_grad.abs().flatten(), descending=args.magnitude_descending)
#             idx_weights = idx_weights[: int(len(idx_weights) * (1 - args.sparsity))]
#             new_tensor = torch.ones_like(noisy_grad).flatten()
#             new_tensor[idx_weights] = 0
#             net_state_dict[original_name] = real_weight
#             net_state_dict[name_mask] = new_tensor.view_as(real_weight)
#             net_state_dict[name_weight] = torch.zeros_like(real_weight)
#     net.load_state_dict(net_state_dict)
#     return net


def smooth_crossentropy(pred, gold, smoothing=0.0):
    n_class = pred.size(1)

    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
    log_prob = F.log_softmax(pred, dim=1)

    return F.kl_div(input=log_prob, target=one_hot, reduction="none").sum(-1)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def compute_test_stats(net, testloader, epoch_number, device, criterion, outF):
    print("Computing test stats")

    # [T.1] Switch the net to eval mode
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    # [T.2] Cycle through all test batches
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets).mean()

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if batch_idx % 10 == 0:  # TODO fix
                print(
                    "Epoch: %d, Loss: %.3f | Acc: %.3f%% (%d/%d)"
                    % (
                        epoch_number,
                        test_loss / (batch_idx + 1),
                        100.0 * correct / total,
                        correct,
                        total,
                    )
                )
    acc = 100.0 * correct / total
    print("For epoch: {}, test loss: {} and accuracy: {}".format(epoch_number, test_loss / (batch_idx + 1), acc))
    outF.write("For epoch: {}, test loss: {} and accuracy: {}".format(epoch_number, test_loss / (batch_idx + 1), acc))
    outF.write("\n")
    outF.flush()

    return acc, test_loss / (batch_idx + 1)
