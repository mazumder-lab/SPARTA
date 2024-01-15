import math

import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler


def use_finetune_optimizer(parameter_ls, momentum, wd):
    print("Using SGD.")
    base_optimizer = torch.optim.SGD(parameter_ls, momentum=momentum, weight_decay=wd)
    return base_optimizer


class LinearWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, total_steps, last_epoch=-1):
        self.total_steps = total_steps
        super(LinearWarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * min(1, self.last_epoch / self.total_steps) for base_lr in self.optimizer.param_groups]


def use_warmup_cosine_scheduler(optimizer, args, total_steps):
    warmup_scheduler = LinearWarmupScheduler(optimizer, total_steps=total_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)
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
