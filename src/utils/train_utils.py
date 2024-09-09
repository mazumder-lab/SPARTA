import argparse
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torchvision.transforms import Resize


def set_seed(seed):
    """
    Set the seed for reproducibility in Python's random, NumPy, and PyTorch.

    Args:
    seed (int): The seed value to set.
    """
    random.seed(seed)  
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def count_parameters(model, all_param_flag=False):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or all_param_flag)


def use_finetune_optimizer(parameter_ls, momentum, wd):
    print("Using SGD.")
    base_optimizer = torch.optim.SGD(parameter_ls, momentum=momentum, weight_decay=wd)
    return base_optimizer


def use_lr_scheduler(optimizer, len_dataset, batch_size, classifier_lr, lr, num_epochs, warm_up=0.2, use_cosine_more_epochs=True):
    steps_per_epoch = int(math.ceil(len_dataset / batch_size))
    # TODO improve this
    print("steps_per_epoch: {}".format(steps_per_epoch))
    epochs = num_epochs if use_cosine_more_epochs is False else int(num_epochs * 1.2)
    lr_schedule = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[classifier_lr, lr, lr],
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=warm_up,
    )
    return lr_schedule


@torch.no_grad()
def layerwise_magnitude_pruning(net_state_dict, new_net_state_dict, sparsity, descending=False):
    for name in new_net_state_dict:
        if "mask" in name and "bias" not in name:
            original_name = name.replace("mask_", "").replace("_trainable", "")
            idx_weights = torch.argsort(net_state_dict[original_name].abs().flatten(), descending=descending)
            idx_weights = idx_weights[: int(len(idx_weights) * (1 - sparsity))]
            param = new_net_state_dict[name]
            layerwise_mask = param.flatten()
            layerwise_mask[idx_weights] = 0
            new_net_state_dict[name] = layerwise_mask.view_as(param)
    return new_net_state_dict


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


def compute_test_stats(net, testloader, epoch_number, device, criterion, outF=None, to_resize=False):
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
            if to_resize:
                outputs = net(Resize(224)(inputs))
            else:
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
    if outF is not None:
        outF.write(
            "For epoch: {}, test loss: {} and accuracy: {}".format(epoch_number, test_loss / (batch_idx + 1), acc)
        )
        outF.write("\n")
        outF.flush()
    return acc, test_loss / (batch_idx + 1)
