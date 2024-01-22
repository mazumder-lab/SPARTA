# %%
import argparse
import math
import os

import torch
import torch.cuda
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator

from conf.global_settings import CHECKPOINT_PATH, MAX_PHYSICAL_BATCH_SIZE
from dataset_utils import get_train_and_test_dataloader
from finegrain_utils.resnet_mehdi import ResNet18_partially_trainable
from models.resnet import ResNet18
from optimizers.optimizer_utils import use_finetune_optimizer
from train_cifar import train_single_epoch
from utils.train_utils import (
    compute_test_stats,
    count_parameters,
    set_seed,
    smooth_crossentropy,
)


def use_lr_scheduler(optimizer, batch_size, classifier_lr, lr, num_epochs, warm_up=0.2):
    steps_per_epoch = int(math.ceil(50000 / batch_size))
    # TODO improve this
    print("steps_per_epoch: {}".format(steps_per_epoch))
    lr_schedule = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[classifier_lr, lr],
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=warm_up,
    )
    return lr_schedule


def test_adaptive_mask():
    """The goal of this test is to verify that using the mask as entirely
    zeros or ones is entirely equivalent to finetuning using
    requires_grad=False/True for each layer."""
    dataset = "cifar10"
    classifier_lr = 0.4
    lr = 0.05
    momentum = 0.9
    wd = 0.0
    batch_size = 500
    clipping = 0.8
    epsilon = 1.0
    delta = 1e-5
    warm_up = 0.01
    num_epochs = 50
    out_file = "nb2_outfile_test_adaptive_mask_seed0.txt"
    INDICES_LIST = [1, 14, 17, 20, 32, 35, 37, 40, 43, 46, 54, 55, 59, 60, 61]

    train_loader, test_loader = get_train_and_test_dataloader(
        dataset=dataset,
        batch_size=batch_size,
    )
    print("train and test data loaders are ready")

    net = ResNet18(num_classes=100)
    net.train()
    net = ModuleValidator.fix(net.to("cpu"))
    device = torch.device(f"cuda:{0}") if torch.cuda.is_available() else "cpu"
    net.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=torch.device("cpu")))
    net.linear = nn.Linear(
        in_features=net.linear.in_features,
        out_features=10,
        bias=net.linear.bias is not None,
    )
    # Indices Mask
    new_net = ResNet18_partially_trainable(num_classes=10, with_mask=True)
    new_net.train()
    new_net = ModuleValidator.fix(new_net.to("cpu"))

    net_state_dict = net.state_dict()
    net_state_dict_id = {name: idx for idx, name in enumerate(net_state_dict)}
    new_net_state_dict = new_net.state_dict()

    for name in new_net_state_dict:
        if "mask" in name:
            original_name = name.replace("mask_", "").replace("_trainable", "")
            idx = net_state_dict_id[original_name]
            if idx not in INDICES_LIST:
                new_net_state_dict[name] = torch.zeros_like(new_net_state_dict[name])
        elif "init" in name:
            original_name = name.replace("init_", "")
            new_net_state_dict[name] = net_state_dict[original_name]
        elif "_trainable" not in name:
            # TODO fix this
            new_net_state_dict[name] = net_state_dict[name]

    new_net.load_state_dict(new_net_state_dict)
    net = new_net

    for name, param in net.named_parameters():
        if ("_trainable" not in name) and ("init" not in name):
            idx = net_state_dict_id[name]
            if idx not in INDICES_LIST:
                param.requires_grad = False

    # usual definitions
    net = net.to(device)

    trainable_indices = []
    trainable_names = []
    classifier_params = []
    other_params = []
    for idx, (name, param) in enumerate(net.named_parameters()):
        # the classifier layer is always trainable. it will have its own learning rate classifier_lr
        if "linear" in name:
            trainable_indices.append(idx)
            trainable_names.append(name)
            classifier_params.append(param)
        # every other parameter which is trainable is added to other_parameters. learning rate is lr
        elif param.requires_grad:
            trainable_indices.append(idx)
            trainable_names.append(name)
            other_params.append(param)
    nb_trainable_params = count_parameters(net)

    # STEP [4] - Create loss function and optimizer
    criterion = smooth_crossentropy  # torch.nn.CrossEntropyLoss()
    parameter_ls = [
        {"params": classifier_params, "lr": classifier_lr},
        {"params": other_params, "lr": lr},
    ]
    # The optimizer is always sgd for now
    optimizer = use_finetune_optimizer(parameter_ls=parameter_ls, momentum=momentum, wd=wd)

    privacy_engine = PrivacyEngine()
    (
        net,
        optimizer,
        train_loader,
    ) = privacy_engine.make_private_with_epsilon(
        module=net,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=num_epochs,
        target_epsilon=epsilon,
        target_delta=delta,
        max_grad_norm=clipping,
    )

    print(f"Using sigma={optimizer.noise_multiplier} and C={clipping}")
    print("loss function and optimizer created")

    # TODO incorporate world size
    lr_scheduler = use_lr_scheduler(optimizer, batch_size, classifier_lr, lr, num_epochs, warm_up)

    # STEP [5] - Run epoch-wise training and validation
    print("training for {} epochs".format(num_epochs))
    addr = out_file
    outF = open(addr, "w")
    outF.write(f"The indices of trainable parameters are: {trainable_indices}.")
    outF.write("\n")
    outF.write(f"The names of trainable parameters are: {trainable_names}.")
    outF.write("\n")
    outF.write(f"The number of trainable parameters is: {nb_trainable_params}.")
    outF.write("\n")
    outF.write(f"Using sigma={optimizer.noise_multiplier} and C={clipping}")
    outF.write("\n")
    outF.flush()

    test_acc_epochs = []
    with BatchMemoryManager(
        data_loader=train_loader,
        max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
        optimizer=optimizer,
    ) as memory_safe_data_loader:
        for epoch in range(num_epochs):
            # Run training for single epoch
            train_single_epoch(
                net=net,
                trainloader=memory_safe_data_loader,
                epoch_number=epoch,
                device=device,
                criterion=criterion,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                clip_gradient=0.0,
                grad_clip_cst=False,
                lsr=0.0,
                accum_steps=1,
                print_batch_stat_freq=1,
                outF=outF,
                batch_size=batch_size,
                epoch=epoch,
                lr_schedule_type="onecycle",
                use_dp=True,
                world_size=1,
            )
            # Compute test accuracy
            test_acc, test_loss = compute_test_stats(
                net=net,
                testloader=test_loader,
                epoch_number=epoch,
                device=device,
                criterion=criterion,
                outF=outF,
            )
            test_acc_epochs.append(test_acc)


set_seed(0)
test_adaptive_mask()

# %%

# %%

# %%

# %%
