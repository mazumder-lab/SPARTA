# %%
import argparse
import math
import os
import pickle

import torch
import torch.cuda
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator

from conf.global_settings import CHECKPOINT_PATH, MAX_PHYSICAL_BATCH_SIZE
from dataset_utils import get_train_and_test_dataloader
from finegrain_utils.resnet_mehdi import ResNet18_partially_trainable
from models.resnet import ResNet18
from opacus_per_sample.privacy_engine_per_sample import PrivacyEnginePerSample
from optimizers.optimizer_utils import use_finetune_optimizer
from utils.train_utils import (
    compute_test_stats,
    count_parameters,
    set_seed,
    smooth_crossentropy,
    str2bool,
)

FINAL_EPOCH = 10


def train_single_epoch(
    net,
    trainloader,
    epoch_number,
    device,
    criterion,
    optimizer,
    lr_scheduler,
    clip_gradient,
    grad_clip_cst,
    lsr,
    accum_steps,
    print_batch_stat_freq,
    outF,
    epoch,
    batch_size,
    lr_schedule_type="warmup_cosine",
    use_dp=False,
    use_w_tilde=False,
    use_fisher_mask_with_true_grads=False,
    sparsity=1.0,
    correction_coefficient=0.1,
    world_size=1,
):
    print("Commencing training for epoch number: {}".format(epoch_number))

    # [T.0] decompose lr_schedulers if we are using two
    if lr_schedule_type == "warmup_cosine":
        # lr_scheduler is warmup here and it is used during 0th epoch only
        lr_scheduler, cosine_scheduler = lr_scheduler

    # [T.1] Convert model to training mode
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    # [T.2] Zero out gradient before commencing training for a full epoch
    if epoch == FINAL_EPOCH:
        optimizer.compute_fisher_mask = True
        optimizer.use_w_tilde = use_w_tilde
        optimizer.use_fisher_mask_with_true_grads = use_fisher_mask_with_true_grads
    optimizer.zero_grad()

    # [T.3] Cycle through all batches for 1 epoch
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Run single step of training
        outputs, loss, old_net = train_vanilla_single_step(
            net=net,
            inputs=inputs,
            targets=targets,
            criterion=criterion,
            accum_steps=accum_steps,
            trainloader=trainloader,
            batch_idx=batch_idx,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            clip_gradient=clip_gradient,
            grad_clip_cst=grad_clip_cst,
            lsr=lsr,
            batch_size=batch_size,
            epoch=epoch,
            lr_schedule_type=lr_schedule_type,
            use_dp=use_dp,
            sparsity=sparsity,
            correction_coefficient=correction_coefficient,
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
                    train_loss * accum_steps / (batch_idx + 1),
                    100.0 * correct / total,
                    correct,
                    total,
                )
            )
    if lr_schedule_type == "warmup_cosine":
        cosine_scheduler.step()
    # Print epoch-end stats
    acc = 100.0 * correct / total
    print(
        "For epoch number: {}, train loss: {} and accuracy: {}".format(
            epoch_number, train_loss * accum_steps / (batch_idx + 1), acc
        )
    )
    return old_net


def train_vanilla_single_step(
    net,
    inputs,
    targets,
    criterion,
    accum_steps,
    trainloader,
    batch_idx,
    optimizer,
    lr_scheduler,
    clip_gradient,
    grad_clip_cst,
    lsr,
    batch_size,
    epoch,
    lr_schedule_type="warmup_cosine",
    use_dp=False,
    sparsity=1.0,
    correction_coefficient=0.1,
):
    # Forward pass through network
    outputs = net(inputs)
    loss = criterion(outputs, targets, smoothing=lsr).mean()

    # Normalize loss to account for gradient accumulation
    loss = loss / accum_steps

    # Backward pass
    loss.mean().backward()
    if use_dp:
        # check if we are at the end of a true batch
        is_updated_logical_batch = not (optimizer._check_skip_next_step(pop_next=False))
    nodp_or_logical_batch = (not use_dp) or is_updated_logical_batch

    # Weights update (TODO include all)
    if (batch_idx + 1) % accum_steps == 0 or batch_idx == len(trainloader) - 1:
        if clip_gradient and not use_dp:
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip_cst)
        # optimizer won't actually make a step unless logical batch is over
        optimizer.step()
        # optimizer won't actually clear gradients unless logical batch is over
        old_net = None
        if is_updated_logical_batch and epoch == FINAL_EPOCH and optimizer.compute_fisher_mask:
            optimizer.get_fisher_mask(sparsity, correction_coefficient)
            print("Starting to print")
            net_state_dict = net.state_dict()
            mask_names = [name for name in net_state_dict if "mask" in name]
            for p, mask_name in zip(optimizer.param_groups[1]["params"], mask_names):
                net_state_dict[mask_name] = p.mask.view_as(net_state_dict[mask_name])
            net.load_state_dict(net_state_dict)
            optimizer.clear_momentum_buffer()
            optimizer.clear_hessian()
            old_net = compute_masked_net_stats(net, trainloader, epoch, device, criterion)

        optimizer.zero_grad()
        # Step when there is a logical step
        if (lr_schedule_type != "warmup_cosine") and nodp_or_logical_batch:
            lr_scheduler.step()
        elif (epoch == 0) and (lr_schedule_type == "warmup_cosine") and nodp_or_logical_batch:
            lr_scheduler.step()

    # Return stuff
    return outputs, loss, old_net


def compute_masked_net_stats(masked_net, trainloader, epoch, device, criterion):
    test_net = ResNet18(num_classes=10)
    test_net.train()
    test_net = ModuleValidator.fix(test_net.to("cpu"))

    masked_net_state_dict = masked_net.state_dict()
    test_net_state_dict = test_net.state_dict()
    for original_name in masked_net_state_dict:
        if "init" in original_name:
            name_mask = original_name.replace("init_", "mask_") + "_trainable"
            name_weight = original_name.replace("init_", "") + "_trainable"
            name = original_name.replace("_module.", "").replace("init_", "")
            param = (
                masked_net_state_dict[original_name]
                + masked_net_state_dict[name_mask] * masked_net_state_dict[name_weight]
            )
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
    )

    for original_name in masked_net_state_dict:
        if "init" in original_name:
            name_mask = original_name.replace("init_", "mask_") + "_trainable"
            name_weight = original_name.replace("init_", "") + "_trainable"
            name = original_name.replace("_module.", "").replace("init_", "")
            param = masked_net_state_dict[original_name] + masked_net_state_dict[name_weight]
            test_net_state_dict[name] = param
    test_net.load_state_dict(test_net_state_dict)
    return test_net


def use_lr_scheduler(optimizer, batch_size, classifier_lr, lr, num_epochs, warm_up=0.2):
    steps_per_epoch = int(math.ceil(50000 / batch_size))
    # TODO improve this
    print("steps_per_epoch: {}".format(steps_per_epoch))
    lr_schedule = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[classifier_lr, lr, lr],
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=warm_up,
    )
    return lr_schedule


# Initialize the parser
parser = argparse.ArgumentParser(description="Experiment Configuration")

# Adding arguments
parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset name")
parser.add_argument("--classifier_lr", type=float, default=0.2, help="Classifier learning rate")
parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
parser.add_argument("--wd", type=float, default=0.0, help="Weight decay")
parser.add_argument("--batch_size", type=int, default=500, help="Batch size")
parser.add_argument("--clipping", type=float, default=1.0, help="Gradient clipping")
parser.add_argument("--epsilon", type=float, default=1.0, help="Epsilon")
parser.add_argument("--delta", type=float, default=1e-5, help="Delta")
parser.add_argument("--warm_up", type=float, default=0.01, help="Warm-up")
parser.add_argument("--correction_coefficient", type=float, default=0.1, help="correction_coefficient")
parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
parser.add_argument("--sparsity", type=float, default=0.2, help="Sparsity")
parser.add_argument("--out_file", type=str, default="outfile_per_sample_test.txt", help="Output file name")
parser.add_argument(
    "--use_w_tilde",
    type=str2bool,
    nargs="?",
    default=True,
    help="Use W Tilde.",
)
parser.add_argument(
    "--use_fisher_mask_with_true_grads",
    type=str2bool,
    nargs="?",
    default=False,
    help="Use Fisher Mask with True Grads.",
)

# Parse the arguments
args = parser.parse_args()


set_seed(0)
"""The goal of this test is to verify that using the mask as entirely
zeros or ones is entirely equivalent to finetuning using
requires_grad=False/True for each layer."""
# Unpacking the configurations
dataset = args.dataset
classifier_lr = args.classifier_lr
lr = args.lr
momentum = args.momentum
wd = args.wd
batch_size = args.batch_size
clipping = args.clipping
epsilon = args.epsilon
delta = args.delta
warm_up = args.warm_up
num_epochs = args.num_epochs
sparsity = args.sparsity
out_file = args.out_file
use_w_tilde = args.use_w_tilde
use_fisher_mask_with_true_grads = args.use_fisher_mask_with_true_grads
correction_coefficient = args.correction_coefficient


print(f"Dataset: {dataset}")
print(f"Classifier Learning Rate: {classifier_lr}")
print(f"Learning Rate: {lr}")
print(f"Momentum: {momentum}")
print(f"Weight Decay: {wd}")
print(f"Batch Size: {batch_size}")
print(f"Gradient Clipping: {clipping}")
print(f"Epsilon: {epsilon}")
print(f"Delta: {delta}")
print(f"Warm-Up: {warm_up}")
print(f"Number of Epochs: {num_epochs}")
print(f"Sparsity: {sparsity}")
print(f"Output File: {out_file}")
print(f"Use W Tilde: {use_w_tilde}")
print(f"correction_coefficient: {correction_coefficient}")
print(f"Use Fisher Mask with True Grads: {use_fisher_mask_with_true_grads}")


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
new_net_state_dict = new_net.state_dict()
# %%

for name in new_net_state_dict:
    if "mask" in name:
        assert torch.all(new_net_state_dict[name] == 1)
    elif "init" in name:
        original_name = name.replace("init_", "")
        new_net_state_dict[name] = net_state_dict[original_name]
    elif "_trainable" not in name:
        new_net_state_dict[name] = net_state_dict[name]

new_net.load_state_dict(new_net_state_dict)
net = new_net
del new_net
# %%
net = net.to(device)

trainable_indices = []
trainable_names = []
classifier_params = []
conv_params = []
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
        if "conv" in name or "shortcut.0" in name:
            conv_params.append(param)
        else:
            other_params.append(param)
nb_trainable_params = count_parameters(net)

# # STEP [4] - Create loss function and optimizer
criterion = smooth_crossentropy  # torch.nn.CrossEntropyLoss()
parameter_ls = [
    {"params": classifier_params, "lr": classifier_lr},
    {"params": conv_params, "lr": lr},
    {"params": other_params, "lr": lr},
]
# The optimizer is always sgd for now
optimizer = use_finetune_optimizer(parameter_ls=parameter_ls, momentum=momentum, wd=wd)

privacy_engine = PrivacyEnginePerSample()
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

# # TODO incorporate world size
lr_sched = use_lr_scheduler(optimizer, batch_size, classifier_lr, lr, num_epochs, warm_up)

# # STEP [5] - Run epoch-wise training and validation
print("training for {} epochs".format(num_epochs))
addr = out_file
outF = open(addr, "w")
print(f"The indices of trainable parameters are: {trainable_indices}.")
print("\n")
print(f"The names of trainable parameters are: {trainable_names}.")
print("\n")
print(f"The number of trainable parameters is: {nb_trainable_params}.")
print("\n")
print(f"Using sigma={optimizer.noise_multiplier} and C={clipping}")
print("\n")

test_acc_epochs = []
with BatchMemoryManager(
    data_loader=train_loader,
    max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
    optimizer=optimizer,
) as memory_safe_data_loader:
    for epoch in range(num_epochs):
        # Run training for single epoch
        ret = train_single_epoch(
            net=net,
            trainloader=memory_safe_data_loader,
            epoch_number=epoch,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            lr_scheduler=lr_sched,
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
            use_w_tilde=use_w_tilde,
            use_fisher_mask_with_true_grads=use_fisher_mask_with_true_grads,
            world_size=1,
            sparsity=sparsity,
            correction_coefficient=correction_coefficient,
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
        if ret is not None:
            old_net = ret

print("Starting Sparsity Analysis.\n")
old_net.to(device)
net_state_dict = net.state_dict()
old_net_state_dict = old_net.state_dict()
overall_frozen = []
for original_name in net_state_dict:
    if "init" in original_name:
        name_mask = original_name.replace("init_", "mask_") + "_trainable"
        name_weight = original_name.replace("init_", "") + "_trainable"
        name = original_name.replace("_module.", "").replace("init_", "")
        param = net_state_dict[original_name] + net_state_dict[name_mask] * net_state_dict[name_weight]
        if name in old_net_state_dict:
            diff_param = (param - old_net_state_dict[name]) if not args.use_zero_pruning else param
            ones_frozen = (diff_param == 0).float().reshape(-1)
            overall_frozen.append(ones_frozen)
            print(f"Percentage of frozen in {name}: {torch.mean(ones_frozen)}.\n")
overall_frozen = torch.cat(overall_frozen)
print(f"Overall percentage of frozen parameters: {torch.mean(overall_frozen)}.\n")

# set_seed(0)
# # test_adaptive_mask()

# %%
