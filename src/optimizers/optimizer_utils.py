import math

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler


def update_magnitude_mask(net: nn.Module, args):
    net_state_dict = net.state_dict()
    for original_name in net_state_dict:
        if "init" in original_name:
            name_mask = original_name.replace("init_", "mask_") + "_trainable"
            name_weight = original_name.replace("init_", "") + "_trainable"
            real_weight = net_state_dict[original_name] + net_state_dict[name_mask] * net_state_dict[name_weight]
            idx_weights = torch.argsort(real_weight.flatten(), descending=False)
            idx_weights = idx_weights[: int(len(idx_weights) * (1 - args.sparsity))]
            new_tensor = torch.ones_like(real_weight).flatten()
            new_tensor[idx_weights] = 0
            net_state_dict[original_name] = real_weight
            net_state_dict[name_mask] = new_tensor.view_as(real_weight)
            net_state_dict[name_weight] = torch.zeros_like(real_weight)
    net.load_state_dict(net_state_dict)
    return net


def update_noisy_grad_mask(net: nn.Module, args):
    net_state_dict = net.state_dict()
    for original_name in net_state_dict:
        if "init" in original_name:
            name_mask = original_name.replace("init_", "mask_") + "_trainable"
            name_weight = original_name.replace("init_", "") + "_trainable"
            real_weight = net_state_dict[original_name] + net_state_dict[name_mask] * net_state_dict[name_weight]
            idx_weights = torch.argsort(real_weight.flatten(), descending=False)
            idx_weights = idx_weights[: int(len(idx_weights) * (1 - args.sparsity))]
            new_tensor = torch.ones_like(real_weight).flatten()
            new_tensor[idx_weights] = 0
            net_state_dict[original_name] = real_weight
            net_state_dict[name_mask] = new_tensor.view_as(real_weight)
            net_state_dict[name_weight] = torch.zeros_like(real_weight)
    net.load_state_dict(net_state_dict)
    return net

    return net


def use_finetune_optimizer(parameter_ls, momentum, wd):
    print("Using SGD.")
    base_optimizer = torch.optim.SGD(parameter_ls, momentum=momentum, weight_decay=wd)
    return base_optimizer


class LinearWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, total_steps, last_epoch=-1):
        self.total_steps = total_steps
        super(LinearWarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            base_lr["initial_lr"] * min(1, (self.last_epoch + 1) / (self.total_steps + 1))
            for base_lr in self.optimizer.param_groups
        ]


def use_warmup_cosine_scheduler(optimizer, num_epochs, total_steps):
    warmup_scheduler = LinearWarmupScheduler(optimizer, total_steps=total_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    return warmup_scheduler, cosine_scheduler


def use_lr_scheduler(optimizer, args, world_size, warm_up=0.2):
    steps_per_epoch = int(math.ceil(50000 / (args.batch_size * args.accum_steps * world_size)))
    # TODO improve this
    print("steps_per_epoch: {}".format(steps_per_epoch))
    lr_schedule = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[args.classifier_lr, args.lr],
        epochs=int(args.num_epochs * 1.6),  # TODO fix this
        steps_per_epoch=steps_per_epoch,
        pct_start=warm_up,
    )
    return lr_schedule
