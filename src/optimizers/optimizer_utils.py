import math

import torch
import torch.optim.lr_scheduler as lr_scheduler


def use_finetune_optimizer(parameter_ls, momentum, wd):
    print("Using SGD.")
    base_optimizer = torch.optim.SGD(parameter_ls, momentum=momentum, weight_decay=wd)
    return base_optimizer


def use_lr_scheduler(optimizer, args, world_size, warm_up=0.2):
    if args.lr_schedule_type == "onecycle":
        steps_per_epoch = int(math.ceil(50000 / (args.batch_size * args.accum_steps * world_size)))
        # TODO improve this
        print("steps_per_epoch: {}".format(steps_per_epoch))
        lr_schedule = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[args.classifier_lr, args.lr],
            epochs=args.num_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=warm_up,
        )
    return lr_schedule
