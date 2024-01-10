import math

import torch
import torch.optim.lr_scheduler as lr_scheduler


def use_finetune_optimizer(parameter_ls, momentum, wd):
    print("Using SGD.")
    base_optimizer = torch.optim.SGD(parameter_ls, momentum=momentum, weight_decay=wd)
    return base_optimizer


def use_lr_scheduler(optimizer, args, world_size, warm_up=0.2):
    optimizer_for_lr_scheduling = optimizer

    if args.lr_schedule_type == "onecycle":
        steps_per_epoch = int(
            math.ceil(50000 / (args.batch_size * args.accum_steps * world_size))
        )  # TODO improve this
        print("steps_per_epoch: {}".format(steps_per_epoch))
        lr_schedule = lr_scheduler.OneCycleLR(
            optimizer_for_lr_scheduling,
            max_lr=[args.classifier_lr, args.lr],
            epochs=args.num_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=warm_up,
        )
    return lr_schedule


def use_optimizer(network, args):
    if args.optimizer == "sgd":
        print("Using SGD")
        base_optimizer = torch.optim.SGD(
            network.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.wd,
        )
    elif args.optimizer == "adamw":
        print("Using Adam with weight decay")
        base_optimizer = torch.optim.AdamW(
            network.parameters(), lr=args.lr, weight_decay=args.wd
        )
    return base_optimizer
