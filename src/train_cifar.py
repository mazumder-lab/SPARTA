import argparse
import math
import os
import pickle

import torch
import torch.cuda
import torch.multiprocessing as mp
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator
from sklearn.metrics import d2_pinball_score
from torch.nn.parallel import DistributedDataParallel as DDP

from conf.global_settings import (
    CHECKPOINT_PATH,
    INDICES_LIST,
    MASK_1_PATH,
    MASK_10_PATH,
    MASK_20_PATH,
    MASK_30_PATH,
    MASK_50_PATH,
    MASK_60_PATH,
    MASK_70_PATH,
    MASK_80_PATH,
    MASK_90_PATH,
    MAX_PHYSICAL_BATCH_SIZE,
)
from dataset_utils import get_train_and_test_dataloader

# from models.resnet import ResNet18, ResNet50
from finegrain_utils.resnet_mehdi import ResNet18_partially_trainable
from loralib import apply_lora, mark_only_lora_as_trainable
from models.resnet import ResNet18, ResNet50
from models.wide_resnet import Wide_ResNet
from optimizers.optimizer_utils import (
    update_magnitude_mask,
    update_noisy_grad_mask,
    use_finetune_optimizer,
    use_lr_scheduler,
    use_warmup_cosine_scheduler,
)
from utils.train_utils import (
    compute_test_stats,
    count_parameters,
    set_seed,
    smooth_crossentropy,
    str2bool,
)


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
    optimizer.zero_grad()

    # [T.3] Cycle through all batches for 1 epoch
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Run single step of training
        outputs, loss = train_vanilla_single_step(
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
    outF.write(
        "For epoch number: {}, train loss: {} and accuracy: {}".format(
            epoch_number, train_loss * accum_steps / (batch_idx + 1), acc
        )
    )
    outF.write("\n")
    outF.flush()


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
):
    # Forward pass through network
    outputs = net(inputs)
    loss = criterion(outputs, targets, smoothing=lsr).mean()

    # Normalize loss to account for gradient accumulation
    loss = loss / accum_steps

    # TODO create branch with l2_loss and mask set to ones.
    # d_params = dict(net.named_parameters())
    # l_trainable = [x for x in d_params.keys() if "_trainable" in x and "mask" not in x]
    # l2_loss = sum([(d_params[x] ** 2).sum() for x in l_trainable])
    # print(l2_loss)

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
        optimizer.zero_grad()
        # Step when there is a logical step
        if (lr_schedule_type != "warmup_cosine") and nodp_or_logical_batch:
            lr_scheduler.step()
        elif (epoch == 0) and (lr_schedule_type == "warmup_cosine") and nodp_or_logical_batch:
            lr_scheduler.step()

    # Return stuff
    return outputs, loss


#########################################################
def main_trainer(rank, world_size, args, use_cuda):
    if world_size > 1:  # initialize multi process procedure
        torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    # TODO log this using MLFlow
    print("Parsed args: {}".format(args))

    # STEP [2] - Create train and test dataloaders
    train_loader, test_loader = get_train_and_test_dataloader(
        dataset=args.dataset,
        batch_size=args.batch_size,
        world_size=world_size,
        rank=rank,
    )
    print("train and test data loaders are ready")

    args.pretrained = True
    # STEP [3] - Create model. If the model is pretrained, it is assumed that it is pretrained on CIFAR100 that's why else 100 in the code.
    if args.model == "resnet18":
        net = ResNet18(num_classes=args.num_classes if not args.pretrained else 100)
        mask_net = ResNet18(num_classes=args.num_classes if not args.pretrained else 100)
    elif args.model == "resnet50":
        net = ResNet50(num_classes=args.num_classes if not args.pretrained else 100)
    elif args.model == "WRN-28-10":
        net = Wide_ResNet(
            depth=28,
            widen_factor=10,
            dropout_rate=0.0,
            num_classes=args.num_classes if not args.pretrained else 100,
        )  # TODO introduce a parameter for dropout
    else:
        raise Exception("unsupported model type provided")

    # All models that would be potentially used for dp should use gn: group normalization instead of bn: batch normalization.
    # Note: the name of the module after Module Validator would still be bn but the module module itself is a gn layer.
    if args.use_gn:
        # This part uses opacus modulevalidator fix to modify the architecture and change all BN with GN.
        # Down the line, we can change the architecture directly in models with GN.
        net.train()
        net = ModuleValidator.fix(net.to("cpu"))
        print(net)

    if use_cuda:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = "cpu"

    if args.pretrained:
        net.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=torch.device("cpu")))
        net.linear = nn.Linear(
            in_features=net.linear.in_features,
            out_features=args.num_classes,
            bias=net.linear.bias is not None,
        )

    if args.mask_available:
        # same mask happens to have two different names on different instances. #TODO fix it.
        sparsity_value = args.sparsity if not args.mask_reversed else 1 - args.sparsity
        if math.isclose(sparsity_value, 0.2, abs_tol=1e-9):
            MASK_PATH = MASK_20_PATH
        elif math.isclose(sparsity_value, 0.5, abs_tol=1e-9):
            MASK_PATH = MASK_50_PATH
        elif math.isclose(sparsity_value, 0.7, abs_tol=1e-9):
            MASK_PATH = MASK_70_PATH
        elif math.isclose(sparsity_value, 0.6, abs_tol=1e-9):
            MASK_PATH = MASK_60_PATH
        elif math.isclose(sparsity_value, 0.8, abs_tol=1e-9):
            MASK_PATH = MASK_80_PATH
        elif math.isclose(sparsity_value, 0.9, abs_tol=1e-9):
            MASK_PATH = MASK_90_PATH
        elif math.isclose(sparsity_value, 0.1, abs_tol=1e-9):
            MASK_PATH = MASK_10_PATH
        elif math.isclose(sparsity_value, 0.01, abs_tol=1e-9):
            MASK_PATH = MASK_1_PATH
        elif math.isclose(sparsity_value, 0.3, abs_tol=1e-9):
            MASK_PATH = MASK_30_PATH
        with open(MASK_PATH, "rb") as file:
            # data = pickle.load(file)
            # mask = data["mask"]
            mask_net.load_state_dict(torch.load(MASK_PATH, map_location=torch.device("cpu")))
            mask = {}
            for name, param in mask_net.named_parameters():
                mask[name] = (param.data == 0.0).float()

    if args.mask_available and args.mask_reversed:
        for name in mask:
            # flips 0 and 1 values
            mask[name] = 1 - mask[name]

    if args.use_magnitude_mask:
        sparsity = args.sparsity
        new_net = ResNet18_partially_trainable(num_classes=args.num_classes, with_mask=True)
        if args.use_gn:
            new_net.train()
            new_net = ModuleValidator.fix(new_net.to("cpu"))
        # Get the state dictionaries of both networks
        net_state_dict = net.state_dict()
        new_net_state_dict = new_net.state_dict()
        for name in new_net_state_dict:
            if "mask" in name:
                use_convexity = 0 <= args.cvx_reversed_obc <= 1
                original_name = name.replace("mask_", "").replace("_trainable", "")
                if args.mask_available and not use_convexity:
                    new_net_state_dict[name] = mask[original_name].view_as(new_net_state_dict[name])
                else:
                    idx_weights = torch.argsort(
                        net_state_dict[original_name].abs().flatten(), descending=args.magnitude_descending
                    )
                    idx_weights = idx_weights[: int(len(idx_weights) * (1 - sparsity))]
                    param = new_net_state_dict[name]
                    new_tensor = param.flatten()
                    new_tensor[idx_weights] = 0
                    new_net_state_dict[name] = new_tensor.view_as(param)
                    if use_convexity:
                        obc_mask = mask[original_name].view_as(new_net_state_dict[name])
                        magnitude_mask = new_tensor.view_as(param)
                        convexity_mask = (
                            args.cvx_reversed_obc * obc_mask + (1 - args.cvx_reversed_obc) * magnitude_mask
                        )
                        new_net_state_dict[name] = convexity_mask

            elif "init" in name:
                original_name = name.replace("init_", "")
                param = net_state_dict[original_name]
                mask_name = name.replace("init_", "mask_") + "_trainable"
                if args.use_zero_pruning:
                    # elementwise multiplication
                    param = param * new_net_state_dict[mask_name].view_as(param)
                new_net_state_dict[name] = param
            elif "_trainable" not in name:
                # TODO fix this
                new_net_state_dict[name] = net_state_dict[name]

        new_net.load_state_dict(new_net_state_dict)
        net, old_net = new_net, net
        del new_net

    if args.finetune_strategy == "lora":
        apply_lora(net, r=args.lora_rank, use_lora_linear=False)
        mark_only_lora_as_trainable(net, bias="lora_only")
    elif args.finetune_strategy == "linear_probing":
        for name, param in net.named_parameters():
            if "linear" not in name:
                param.requires_grad = False
    elif args.finetune_strategy == "lp_gn":
        for name, param in net.named_parameters():
            if ("linear" not in name) and ("bn" not in name):
                param.requires_grad = False
    elif args.finetune_strategy == "conf_indices":
        for idx, (_, param) in enumerate(net.named_parameters()):
            if idx not in INDICES_LIST:
                param.requires_grad = False
    elif args.finetune_strategy == "all_layers":
        # keep all parameters trainable
        pass

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

    if world_size > 1:  # call DDP to parallelize the data
        net = DDP(net, device_ids=[rank])
    print("model created")

    # STEP [4] - Create loss function and optimizer
    criterion = smooth_crossentropy  # torch.nn.CrossEntropyLoss()
    parameter_ls = [
        {"params": classifier_params, "lr": args.classifier_lr},
        {"params": other_params, "lr": args.lr},
    ]
    # The optimizer is always sgd for now
    if args.optimizer == "sgd":
        optimizer = use_finetune_optimizer(parameter_ls=parameter_ls, momentum=args.momentum, wd=args.wd)

    if args.use_dp:
        privacy_engine = PrivacyEngine()
        (
            net,
            optimizer,
            train_loader,
        ) = privacy_engine.make_private_with_epsilon(
            module=net,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=args.num_epochs,
            target_epsilon=args.epsilon,
            target_delta=args.delta,
            max_grad_norm=args.clipping,
        )

    if args.use_dp:
        print(f"Using sigma={optimizer.noise_multiplier} and C={args.clipping}")
    print("loss function and optimizer created")

    if args.lr_schedule_type == "onecycle":
        lr_scheduler = use_lr_scheduler(optimizer=optimizer, args=args, world_size=world_size, warm_up=args.warm_up)
    elif args.lr_schedule_type == "warmup_cosine":
        # TODO incorporate world size
        lr_scheduler = use_warmup_cosine_scheduler(
            optimizer=optimizer, num_epochs=args.num_epochs, total_steps=len(train_loader)
        )

    # STEP [5] - Run epoch-wise training and validation
    print("training for {} epochs".format(args.num_epochs))
    if rank > 0:  # Write on the save file only on the first gpu.
        addr = args.out_file + "1"
    else:
        addr = args.out_file
    outF = open(addr, "w")
    print(args)
    outF.write(str(args))
    outF.write("\n")
    outF.write(f"The indices of trainable parameters are: {trainable_indices}.")
    outF.write("\n")
    outF.write(f"The names of trainable parameters are: {trainable_names}.")
    outF.write("\n")
    outF.write(f"The number of trainable parameters is: {nb_trainable_params}.")
    outF.write("\n")
    if args.use_dp:
        outF.write(f"Using sigma={optimizer.noise_multiplier} and C={args.clipping}")
        outF.write("\n")
    outF.flush()

    test_acc_epochs = []
    if args.use_dp:
        with BatchMemoryManager(
            data_loader=train_loader,
            max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
            optimizer=optimizer,
        ) as memory_safe_data_loader:
            for epoch in range(args.num_epochs):
                # Run training for single epoch
                train_single_epoch(
                    net=net,
                    trainloader=memory_safe_data_loader,
                    epoch_number=epoch,
                    device=device,
                    criterion=criterion,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    clip_gradient=args.clip_gradient,
                    grad_clip_cst=args.grad_clip_cst,
                    lsr=args.lsr,
                    accum_steps=args.accum_steps,
                    print_batch_stat_freq=args.print_batch_stat_freq,
                    outF=outF,
                    batch_size=args.batch_size,
                    epoch=epoch,
                    lr_schedule_type=args.lr_schedule_type,
                    world_size=world_size,
                    use_dp=True,
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
                if args.use_adaptive_magnitude_mask and ((epoch + 1) % 10 == 0):
                    if args.type_mask == "magnitude":
                        net = update_magnitude_mask(net, args)
                    elif args.type_mask == "noisy_grad_magnitude":
                        net = update_noisy_grad_mask(net, args)

    else:
        for epoch in range(args.num_epochs):
            if world_size > 1:  # sync dataloader in case of multiple gpus
                train_loader.sampler.set_epoch(epoch)
            # Run training for single epoch
            train_single_epoch(
                net=net,
                trainloader=train_loader,
                epoch_number=epoch,
                device=device,
                criterion=criterion,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                clip_gradient=args.clip_gradient,
                grad_clip_cst=args.grad_clip_cst,
                lsr=args.lsr,
                accum_steps=args.accum_steps,
                print_batch_stat_freq=args.print_batch_stat_freq,
                outF=outF,
                batch_size=args.batch_size,
                epoch=epoch,
                lr_schedule_type=args.lr_schedule_type,
                world_size=world_size,
                use_dp=False,
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
    print("training complete")

    if args.use_magnitude_mask:
        outF.write("Starting Sparsity Analysis.")
        old_net.to(device)
        net_state_dict = net.state_dict()
        old_net_state_dict = old_net.state_dict()
        for original_name in net_state_dict:
            if "init" in original_name:
                name_mask = original_name.replace("init_", "mask_") + "_trainable"
                name_weight = original_name.replace("init_", "") + "_trainable"
                name = original_name.replace("_module.", "").replace("init_", "")
                param = net_state_dict[original_name] + net_state_dict[name_mask] * net_state_dict[name_weight]
                if name in old_net_state_dict:
                    diff_param = (param - old_net_state_dict[name]) if not args.use_zero_pruning else param
                    outF.write(f"Sparsity in {name}: {torch.mean((diff_param == 0).float())}")

    if world_size == 1:  # save the model
        torch.save(net.state_dict(), args.save_file)
    elif rank == 0:
        torch.save(net.module.state_dict(), args.save_file)
    else:
        print("world_size is not 1 and rank is not 0")
        torch.save(net.module.state_dict(), args.save_file)

    # Print average of top 5 test accuracies
    # avg_best_test_accuracy = sum(sorted(test_acc_epochs)[-5:]) / 5
    # print("Overall test accuracy: {}".format(avg_best_test_accuracy))
    # outF.write("Overall test accuracy: {}".format(avg_best_test_accuracy))

    # Print last test accuracy obtained
    last_test_accuracy = test_acc_epochs[-1]
    print("Test accuracy: {}".format(last_test_accuracy))
    outF.write("Test accuracy: {}".format(last_test_accuracy))
    outF.write("\n")
    outF.flush()


# TODO use MLFlow to dump results in a user-friendly format

if __name__ == "__main__":
    # STEP [1] - Parse command line arguments
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10/100 training on CNNs/ViTs/MLP-Mixers")

    # Data loader arguments
    parser.add_argument(
        "--dataset",
        default="cifar10",
        type=str,
        help="type of dataset (cifar10 or cifar100)",
    )
    parser.add_argument(
        "--batch_size",
        default=1000,
        type=int,
        help="batch size (per gpu) for training sets. The effective batch size is num_gpu*batch_size",
    )

    # Model arguments
    parser.add_argument(
        "--model",
        default="resnet18",
        type=str,
        choices=["resnet18", "resnet50", "WRN-28-10"],
        help="type of model for image classification on CIFAR datasets",
    )
    parser.add_argument(
        "--num_classes",
        default=10,
        type=int,
        choices=[10, 100],
        help="number of classes. 10 for CIFAR10, 100 for CIFAR100",
    )

    # Learning rate arguments
    parser.add_argument(
        "--lr_schedule_type",
        default="onecycle",
        type=str,
        choices=["warmup_cosine", "onecycle"],
        help="type of learning rate scheduler",
    )
    parser.add_argument(
        "--classifier_lr",
        default=0.8,
        type=float,
        help="learning rate for classification layer",
    )
    parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
    # Loss function
    parser.add_argument("--lsr", default=0.0, type=float, help="label smoothing")
    parser.add_argument("--warm_up", default=0.2, type=float, help="warm up for one cycle")
    parser.add_argument("--num_epochs", default=200, type=int, help="number of epochs")
    # Optimizer arguments
    parser.add_argument(
        "--optimizer",
        default="sgd",
        type=str,
        choices=["sgd"],
        help="type of optimizer used",
    )
    parser.add_argument("--momentum", default=0.0, type=float, help="momentum")
    parser.add_argument("--wd", default=0.0, type=float, help="weight decay")
    parser.add_argument(
        "--clip_gradient",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="clips the gradient.",
    )
    parser.add_argument("--grad_clip_cst", default=0.0, type=float, help="constant of gradient clipping")
    # Training arguments
    parser.add_argument(
        "--use_adaptive_lr",
        type=str2bool,
        nargs="?",
        default=False,
        help="whether to divide lr by clipping constant.",
    )
    parser.add_argument(
        "--finetune_strategy",
        type=str,
        choices=["linear_probing", "lp_gn", "conf_indices", "lora", "all_layers"],
        default="all_layers",
        help="how to finetune the model.",
    )
    parser.add_argument("--lora_rank", default=0, type=int, help="lora rank value")
    parser.add_argument(
        "--accum_steps",
        default=1,
        type=int,
        help="number of gradient accumulation steps",
    )
    parser.add_argument(
        "--print_batch_stat_freq",
        default=1,
        type=int,
        help="print batch statistics after every few eopochs specified by this argument",
    )
    # Changing batch normalization by group normalization
    parser.add_argument(
        "--use_gn",
        type=str2bool,
        nargs="?",
        default=True,
        help="uses opacus validator to change the batch norms by group normalization layers.",
    )
    parser.add_argument(
        "--use_magnitude_mask",
        type=str2bool,
        nargs="?",
        default=False,
        help="uses magnitude mask before training the network.",
    )
    parser.add_argument(
        "--mask_available",
        type=str2bool,
        nargs="?",
        default=False,
        help="We have access to obc mask (fixed).",
    )
    parser.add_argument(
        "--mask_reversed",
        type=str2bool,
        nargs="?",
        default=False,
        help="flips training and fixed parameters.",
    )
    parser.add_argument(
        "--cvx_reversed_obc",
        type=float,
        default=-1,
        help="apply mask = alpha m_reversed_obc + (1 - alpha) m_magnitude.",
    )
    parser.add_argument(
        "--use_zero_pruning",
        type=str2bool,
        nargs="?",
        default=False,
        help="zeroes out non-trainable weights as opposed to simply freezing them.",
    )
    parser.add_argument(
        "--use_adaptive_magnitude_mask",
        type=str2bool,
        nargs="?",
        default=False,
        help="uses magnitude mask before training the network.",
    )
    parser.add_argument(
        "--magnitude_descending",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="magnitude from smaller to bigger.",
    )
    parser.add_argument(
        "--type_mask",
        type=str,
        choices=["magnitude", "noisy_grad_magnitude", ""],
        default="",
        help="chooses type of mask to be applied if adaptive magnitude mask is true.",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.0,
        help="percentage of weights that are non-trainable.",
    )

    parser.add_argument(
        "--use_dp",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="uses opacus validator to change the batch norms by group normalization layers.",
    )

    # Add DP parameters
    parser.add_argument(
        "--epsilon",
        type=float,
        default=-1,
        help="privacy parameter, if -1 non-private training.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=-1,
        help="privacy parameter, if -1 non-private training",
    )
    # For now this is different from grad_clipping constant because the other one will not be used.
    parser.add_argument(
        "--clipping",
        type=float,
        default=0.0,
        help="gradient clipping constant C: max_grad_norm in opacus for DP finetuning",
    )

    # Logging arguments
    parser.add_argument(
        "--experiment_dir",
        default=".",
        type=str,
        help="name of directory where we put the experiments results",
    )
    # Logging arguments
    parser.add_argument(
        "--out_file",
        default="output_file.txt",
        type=str,
        help="output file for logging",
    )
    parser.add_argument(
        "--save_file",
        default="output_net.pt",
        type=str,
        help="output file for saving the network",
    )
    # Random seed
    parser.add_argument("--seed", default=0, type=int, help="RNG seed")
    parser.add_argument("--SLURM_JOB_ID", type=int, default=-1)
    parser.add_argument("--TASK_ID", type=int, default=-1)
    # For DDP, not set by user
    parser.add_argument("--local_rank", type=int)

    args = parser.parse_args()

    set_seed(args.seed)

    args.out_file = os.path.join(
        args.experiment_dir, str(args.SLURM_JOB_ID) + "_" + str(args.TASK_ID) + "_" + args.out_file
    )
    args.save_file = os.path.join(
        args.experiment_dir, str(args.SLURM_JOB_ID) + "_" + str(args.TASK_ID) + "_" + args.save_file
    )
    torch.backends.cudnn.benchmark = True

    use_cuda = torch.cuda.is_available()
    print("use_cuda={use_cuda}.")
    world_size = torch.cuda.device_count()

    # set to False by default. only used in constants search experiments.
    if args.use_adaptive_lr:
        print("We are using the learning_rate / clipping constant.")
        args.lr = args.lr / (args.grad_clip_cst if not args.use_dp else args.clipping)
        args.classifier_lr = args.classifier_lr / (args.grad_clip_cst if not args.use_dp else args.clipping)

    # if you want to use a predefined mask, don't update it after k epochs.
    if args.mask_available:
        args.use_adaptive_magnitude_mask = False

    # These are not used in dp. Other parameters are going to substitute them
    if args.use_dp:
        # Opacus is going to handle gradient clipping on its own. These parameters are for non-dp training.
        args.clip_gradient = False
        args.grad_clip_cst = 0.0
        # Poisson sampling in DP does not allow accumulation of steps. If a large batch size is needed, and there are memory constraints, use BatchMemoryManager instead (which is used)
        args.accum_steps = 1

    if world_size > 1:  # multiple gpus available
        mp.spawn(
            main_trainer,
            args=(world_size, args, use_cuda),
            nprocs=world_size,
            join=True,
        )
    else:  # single gpu
        main_trainer(0, 1, args, use_cuda)
