import argparse
import gc
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from opacus.validators import ModuleValidator
from torchvision.transforms import Resize

from models.deit import (
    deit_base_patch16_224,
    deit_small_patch16_224,
    deit_tiny_patch16_224,
)
from models.resnet import ResNet18
from models.wide_resnet import Wide_ResNet
from opacus_per_sample.optimizer_per_sample import DPOptimizerPerSample


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


def train_vanilla_single_step(
    net,
    inputs,
    targets,
    criterion,
    trainloader,
    batch_idx,
    optimizer,
    lr_scheduler,
    lsr,
    batch_size,
    epoch,
    to_resize=False,
):
    if to_resize:
        inputs = Resize(224)(inputs)

    # Forward pass through network
    outputs = net(inputs)
    loss = criterion(outputs, targets, smoothing=lsr).mean()

    # Backward pass
    loss.mean().backward()
    # check if we are at the end of a true batch
    is_updated_logical_batch = not optimizer._check_skip_next_step(pop_next=False)

    # optimizer won't actually make a step unless logical batch is over
    optimizer.step()
    # optimizer won't actually clear gradients unless logical batch is over
    optimizer.zero_grad()
    # If all learning rates are set to 0.0, don't update the weights, it will be fixed elsewhere
    all_lrs_zeros = all(group["lr"] == 0.0 for group in optimizer.param_groups)
    # Step when there is a logical step
    if is_updated_logical_batch and not all_lrs_zeros:
        lr_scheduler.step()
    return outputs, loss


def train_single_epoch(
    net,
    trainloader,
    epoch_number,
    device,
    criterion,
    optimizer: DPOptimizerPerSample,
    lr_scheduler,
    lsr,
    print_batch_stat_freq,
    outF,
    epoch,
    batch_size,
    to_resize=False,
):
    print("Commencing training for epoch number: {}".format(epoch_number), flush=True)
    # [T.1] Convert model to training mode
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    # [T.2] Zero out gradient before commencing training for a full epoch
    optimizer.zero_grad()
    gc.collect()
    torch.cuda.empty_cache()

    # [T.3] Cycle through all batches for 1 epoch
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Run single step of training
        outputs, loss = train_vanilla_single_step(
            net=net,
            inputs=inputs,
            targets=targets,
            criterion=criterion,
            trainloader=trainloader,
            batch_idx=batch_idx,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            lsr=lsr,
            batch_size=batch_size,
            epoch=epoch,
            to_resize=to_resize,
        )

        # Collect stats
        train_loss += loss.item()
        total += targets.size(0)

        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        if batch_idx % print_batch_stat_freq == 0:
            for param_group in optimizer.param_groups:
                print("Current lr: {}".format(param_group["lr"]))
            print(
                "Epoch: %d, Batch: %d, Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (
                    epoch_number,
                    batch_idx,
                    train_loss / (batch_idx + 1),
                    100.0 * correct / total,
                    correct,
                    total,
                ),
                flush=True,
            )

    # Print epoch-end stats
    acc = 100.0 * correct / total
    print(
        "For epoch number: {}, train loss: {} and accuracy: {}".format(epoch_number, train_loss / (batch_idx + 1), acc)
    )


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


def compute_masked_net_stats(masked_net: nn.Module, trainloader, epoch, device, criterion, model_name, num_classes):
    if model_name == "resnet18":
        test_net = ResNet18(num_classes=num_classes)
    elif model_name == "wrn2810":
        test_net = Wide_ResNet(
            depth=28,
            widen_factor=10,
            dropout_rate=0.0,
            num_classes=num_classes,
        )
    elif model_name == "deit_tiny_patch16_224":
        test_net = deit_tiny_patch16_224(pretrained=False, num_classes=num_classes).to("cpu")
    elif model_name == "deit_small_patch16_224":
        test_net = deit_small_patch16_224(pretrained=False, num_classes=num_classes).to("cpu")
    elif model_name == "deit_base_patch16_224":
        test_net = deit_base_patch16_224(pretrained=False, num_classes=num_classes).to("cpu")
    test_net.train()
    test_net = ModuleValidator.fix(test_net.to("cpu"))

    masked_net_state_dict = masked_net.state_dict()
    test_net_state_dict = test_net.state_dict()
    for original_name in masked_net_state_dict:
        if "init" in original_name:
            name_mask = original_name.replace("init_", "mask_") + "_trainable"
            name = original_name.replace("_module.", "").replace("init_", "")
            param = masked_net_state_dict[original_name] * masked_net_state_dict[name_mask]
            test_net_state_dict[name] = param
        elif "_trainable" not in original_name:
            test_net_state_dict[original_name.replace("_module.", "")] = masked_net_state_dict[original_name]
    test_net.load_state_dict(test_net_state_dict)
    test_net.to(device)

    compute_test_stats(
        net=test_net,
        testloader=trainloader,
        epoch_number=epoch,
        device=device,
        criterion=criterion,
        to_resize="deit" in model_name,
    )

    if model_name != "wrn2810":
        for original_name in masked_net_state_dict:
            if "init" in original_name:
                name_mask = original_name.replace("init_", "mask_") + "_trainable"
                name = original_name.replace("_module.", "").replace("init_", "")
                param = masked_net_state_dict[original_name]
                test_net_state_dict[name] = param
            elif "_trainable" not in original_name:
                test_net_state_dict[original_name.replace("_module.", "")] = masked_net_state_dict[original_name]
        test_net.load_state_dict(test_net_state_dict)
    else:
        test_net = None
    return test_net


def count_parameters(model, all_param_flag=False):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or all_param_flag)


def use_finetune_optimizer(parameter_ls, momentum, wd):
    print("Using SGD.")
    base_optimizer = torch.optim.SGD(parameter_ls, momentum=momentum, weight_decay=wd)
    return base_optimizer


def use_lr_scheduler(
    optimizer, len_dataset, batch_size, classifier_lr, lr, num_epochs, warm_up=0.2, use_cosine_more_epochs=True
):
    steps_per_epoch = int(math.ceil(len_dataset / batch_size))
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
