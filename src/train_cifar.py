import argparse
import copy
import gc
import math
import os
import pickle
import time
from tabnanny import check

import torch
import torch.cuda
import torch.nn as nn
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator
from torchvision.transforms import Resize

from finegrain_utils.resnet_mehdi import ResNet18_partially_trainable
from finegrain_utils.wide_resnet_mehdi import WRN2810_partially_trainable
from models.deit import (
    deit_base_patch16_224,
    deit_small_patch16_224,
    deit_tiny_patch16_224,
)
from models.resnet import ResNet18, ResNet50
from models.wide_resnet import Wide_ResNet
from opacus_per_sample.optimizer_per_sample import DPOptimizerPerSample
from opacus_per_sample.privacy_engine_per_sample import PrivacyEnginePerSample
from utils.change_modules import fix
from utils.dataset_utils import get_train_and_test_dataloader
from utils.train_utils import (
    compute_test_stats,
    count_parameters,
    layerwise_magnitude_pruning,
    set_seed,
    smooth_crossentropy,
    str2bool,
    use_finetune_optimizer,
    use_lr_scheduler,
)


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
):
    if "deit" in args.model:
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


#########################################################
def main_trainer(args, use_cuda):
    # TODO log this using MLFlow
    print("Parsed args: {}".format(args))

    if use_cuda:
        torch.cuda.set_device(0)
        device = torch.device(f"cuda:{0}")
    else:
        device = "cpu"
    # STEP [2] - Create train and test dataloaders
    train_loader, test_loader = get_train_and_test_dataloader(
        dataset=args.dataset,
        batch_size=args.batch_size,
    )
    print("train and test data loaders are ready")

    # STEP [3] - Create model and load pretrained weights (with Group Normalization).
    if args.model == "resnet18":
        net = ResNet18(num_classes=args.num_classes)
        checkpoint_path = "../checkpoints/lsr=01train_resnet_gn.pt"
    elif args.model == "wrn2810":
        net = Wide_ResNet(
            depth=28,
            widen_factor=10,
            dropout_rate=0.0,
            num_classes=args.num_classes,
        )
        checkpoint_path = "../checkpoints/wrn_2810_imagenet32_gn.pt"
    elif args.model == "deit_tiny_patch16_224":
        net = deit_tiny_patch16_224(pretrained=False, num_classes=args.num_classes).to("cpu")
        checkpoint_path = "../checkpoints/deit_tiny_patch16_224-a1311bcf.pth"
    elif args.model == "deit_small_patch16_224":
        net = deit_small_patch16_224(pretrained=False, num_classes=args.num_classes).to("cpu")
        checkpoint_path = "../checkpoints/deit_small_patch16_224-cd65a155.pth"
    elif args.model == "deit_base_patch16_224":
        net = deit_base_patch16_224(pretrained=False, num_classes=args.num_classes).to("cpu")
        checkpoint_path = "../checkpoints/deit_base_patch16_224-b5f2ef4d.pth"
    else:
        raise Exception("unsupported model type provided.")

    if args.use_gn:
        net.train()
        net = ModuleValidator.fix(net.to("cpu"))
        print(net)

    if args.model == "resnet18":
        pretrained_weights = torch.load(checkpoint_path, map_location="cpu")
        if args.num_classes != 100:  # pretrained on cifar100
            del pretrained_weights["linear.weight"]
            del pretrained_weights["linear.bias"]
    elif args.model == "wrn2810":
        pretrained_weights = torch.load(checkpoint_path, map_location="cpu")
        if args.num_classes != 1000:  # pretrained on ImageNet32
            del pretrained_weights["linear.weight"]
            del pretrained_weights["linear.bias"]
    else:
        pretrained_weights = torch.load(checkpoint_path, map_location="cpu")["model"]
        if args.num_classes != 1000:  # All pretrained on ImageNet
            del pretrained_weights["head.weight"]
            del pretrained_weights["head.bias"]
    net.load_state_dict(pretrained_weights, strict=False)

    # STEP [4] - Decide on which masking procedure to follow. Introduce the masking formulation W_old + m . W if the method is finegrained.
    # For DeiTs the embedding layer is frozen no matter the masking type.
    if "deit" in args.model:
        net.cls_token.requires_grad = False
        net.pos_embed.requires_grad = False

    if args.method_name == "linear_probing":
        for name, param in net.named_parameters():
            if "linear" not in name:
                param.requires_grad = False
    elif args.method_name == "lp_gn":
        for name, param in net.named_parameters():
            if "deit" in args.model:
                if "head" not in name:
                    param.requires_grad = False
            else:
                if ("linear" not in name) and ("bn" not in name):
                    param.requires_grad = False
    elif args.method_name == "first_last":
        for idx, (name, param) in enumerate(net.named_parameters()):
            if ("linear" not in name) and ("bn" not in name) and (idx >= 15):
                param.requires_grad = False
    elif args.method_name == "all_layers":
        # keep all parameters trainable
        pass

    # Print the name of Trainable parameters
    print("--- List of trainable params ---", flush=True)
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name, flush=True)
    print("--- Done ---", flush=True)

    if args.mask_type:  # != ""
        # If we are using any type if masking, then introduce the partially trainable $W_{\text{old} + m \odot W$ formulation
        if args.model == "resnet18":
            new_net = ResNet18_partially_trainable(num_classes=args.num_classes, with_mask=True)
        elif args.model == "wrn2810":
            new_net = WRN2810_partially_trainable(num_classes=args.num_classes, partially_trainable_bias=False)
        elif "deit" in args.model:
            new_net = copy.deepcopy(net)
            new_net = fix(new_net, partially_trainable_bias=True)

        if args.use_gn:
            new_net.train()
            new_net = ModuleValidator.fix(new_net.to("cpu"))
        # Get the state dictionaries of both networks
        net_state_dict = net.state_dict()
        new_net_state_dict = new_net.state_dict()

        # Create an initial mask with magnitude pruning if it is being used solely or as a convex combination
        if args.method_name == "mp_weights":
            new_net_state_dict = layerwise_magnitude_pruning(
                net_state_dict,
                new_net_state_dict,
                args.sparsity,
                descending=False,
            )

        # Now copy the initial weights in the right place in the new formulation and delete the previous architecture if it is not used.
        for name in new_net_state_dict:
            if "init" in name:
                original_name = name.replace("init_", "")
                new_net_state_dict[name] = net_state_dict[original_name]
            elif "_trainable" not in name:
                # This is for parameters that remain unchanged in the new formulation
                new_net_state_dict[name] = net_state_dict[name]

        new_net.load_state_dict(new_net_state_dict)
        net, old_net = new_net, net
        if args.model == "wrn2810":
            del old_net
        del new_net, net_state_dict, new_net_state_dict

    # STEP [5] - Seperate trainable parameters from linear (those are randomly initialized so lr is very high)
    net = net.to(device)

    trainable_indices = []
    trainable_names = []
    classifier_params = []
    conv_params = []
    other_params = []
    for idx, (name, param) in enumerate(net.named_parameters()):
        # the classifier layer is always trainable. it will have its own learning rate classifier_lr
        if "linear" in name or ("head" in name and "deit" in args.model):
            trainable_indices.append(idx)
            trainable_names.append(name)
            classifier_params.append(param)
            print("Classifier layer:", name)
        # every other parameter which is trainable is added to other_parameters. learning rate is lr
        elif param.requires_grad:
            trainable_indices.append(idx)
            trainable_names.append(name)
            if (("conv" in name or "shortcut.0" in name) and "deit" not in args.model) or (
                "blocks" in name and "norm" not in name and "deit" in args.model
            ):
                conv_params.append(param)
                print("Conv type layer:", name)
            else:
                other_params.append(param)
                print("Other layer:", name)
    nb_trainable_params = count_parameters(net)
    print("Model created.")

    # STEP [6] - Create loss function and optimizer
    criterion = smooth_crossentropy  # torch.nn.CrossEntropyLoss()
    parameter_ls = [
        {"params": classifier_params, "lr": args.classifier_lr},
        {"params": conv_params, "lr": args.lr},
        {"params": other_params, "lr": args.lr},
    ]
    # The optimizer is always sgd for now
    if args.optimizer == "sgd":
        optimizer = use_finetune_optimizer(parameter_ls=parameter_ls, momentum=args.momentum, wd=args.wd)

    privacy_engine = PrivacyEnginePerSample()
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
    print(f"Using sigma={optimizer.noise_multiplier} and C={args.clipping}")
    print("loss function and optimizer created")

    if args.lr_schedule_type == "onecycle":
        lr_scheduler = use_lr_scheduler(
            optimizer,
            args.batch_size,
            args.classifier_lr,
            args.lr,
            args.num_epochs,
            args.warm_up,
        )

    # STEP [7] - Run epoch-wise training and validation
    print("training for {} epochs".format(args.num_epochs))
    addr = args.out_file
    outF = open(addr, "w")
    print(args)
    outF.write(str(args))
    outF.write("\n")
    outF.write(f"The indices of trainable parameters are: {trainable_indices}.")
    print(f"The indices of trainable parameters are: {trainable_indices}.", flush=True)
    outF.write("\n")
    outF.write(f"The names of trainable parameters are: {trainable_names}.")
    print(f"The names of trainable parameters are: {trainable_names}.", flush=True)
    outF.write("\n")
    outF.write(f"The number of trainable parameters is: {nb_trainable_params}.")
    print(f"The number of trainable parameters is: {nb_trainable_params}.", flush=True)
    outF.write("\n")
    outF.write(f"Using sigma={optimizer.noise_multiplier} and C={args.clipping}")
    print(f"Using sigma={optimizer.noise_multiplier} and C={args.clipping}", flush=True)
    outF.write("\n")
    outF.flush()

    start_time = time.time()
    test_acc_epochs = []
    with BatchMemoryManager(
        data_loader=train_loader,
        max_physical_batch_size=args.max_physical_batch_size,
        optimizer=optimizer,
    ) as memory_safe_data_loader:
        old_net = net
        for epoch in range(args.num_epochs):
            # Run training for single epoch
            if args.mask_type == "optimization" and epoch == args.epoch_mask_finding:
                optimizer.compute_mask = True
                optimizer.method_name = args.method_name
                if args.use_fixed_w_mask_finding:
                    original_lrs = [group["lr"] for group in optimizer.param_groups]
                    for group in optimizer.param_groups:
                        group["lr"] = 0.0

            train_single_epoch(
                net=net,
                trainloader=memory_safe_data_loader,
                epoch_number=epoch,
                device=device,
                criterion=criterion,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                lsr=args.lsr,
                print_batch_stat_freq=args.print_batch_stat_freq,
                outF=outF,
                batch_size=args.batch_size,
                epoch=epoch,
            )

            gc.collect()
            torch.cuda.empty_cache()
            ret = None
            if args.mask_type == "optimization" and epoch == args.epoch_mask_finding and args.use_fixed_w_mask_finding:
                for group, original_lr in zip(optimizer.param_groups, original_lrs):
                    group["lr"] = original_lr
            if args.mask_type == "optimization" and epoch == args.epoch_mask_finding and optimizer.compute_mask:
                net_state_dict = net.state_dict()
                if "deit" in args.model:
                    init_weights = [
                        net_state_dict[name] for name in net.state_dict() if "init" in name and "blocks" in name
                    ]
                else:
                    init_weights = [net_state_dict[name] for name in net.state_dict() if "init" in name]
                del net_state_dict
                print(
                    f"Start the mask finding procedure with the method_name={args.method_name}",
                    flush=True,
                )
                optimizer.get_optimization_method_mask(init_weights, args.sparsity)

                print("Starting to print")
                net_state_dict = net.state_dict()
                if "deit" in args.model:
                    init_names = [name for name in net.state_dict() if "init" in name and "blocks" in name]
                else:
                    init_names = [name for name in net_state_dict if "init" in name]
                for p, init_name in zip(optimizer.param_groups[1]["params"], init_names):
                    name_mask = init_name.replace("init_", "mask_") + "_trainable"
                    name_weight = init_name.replace("init_", "") + "_trainable"
                    net_state_dict[name_mask] = p.mask.view_as(net_state_dict[name_mask])
                    if args.use_delta_weight_optim:
                        real_weight = (
                            net_state_dict[init_name] + net_state_dict[name_weight]
                        )  # * net_state_dict[name_mask] // The assumption is that the mask is initially all ones for the optimization methods
                    else:
                        real_weight = (
                            net_state_dict[init_name] + net_state_dict[name_weight] * net_state_dict[name_mask]
                        )
                    net_state_dict[init_name] = real_weight
                    net_state_dict[name_weight] = torch.zeros_like(real_weight)
                net.load_state_dict(net_state_dict)
                optimizer.clear_momentum_grad()
                ret = compute_masked_net_stats(
                    net,
                    memory_safe_data_loader,
                    epoch,
                    device,
                    criterion,
                    model_name=args.model,
                    num_classes=args.num_classes,
                )

            # Compute test accuracy
            test_acc, test_loss = compute_test_stats(
                net=net,
                testloader=test_loader,
                epoch_number=epoch,
                device=device,
                criterion=criterion,
                outF=outF,
                to_resize="deit" in args.model,
            )
            test_acc_epochs.append(test_acc)
            if ret is not None and (
                args.model != "wrn2810"
            ):  # The 2nd condition is not needed here as it is already captured in train_single_epoch but I leave it for comprehension
                old_net = ret

            epsilon = privacy_engine.get_epsilon(args.delta)
            print(f"Current privacy budget spent: {epsilon}.")
            if epsilon > args.epsilon:
                print(f"Stopping training at epoch={epoch} with epsilon={epsilon}.")
                break
        del ret

    # STEP [8] - Run sparsity checks on all parameters
    if args.mask_type and args.model != "wrn2810":
        outF.write("Starting Sparsity Analysis.\n")
        print("Starting Sparsity Analysis.\n", flush=True)
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
                    diff_param = param - old_net_state_dict[name]
                    ones_frozen = (diff_param == 0).float().reshape(-1)
                    overall_frozen.append(ones_frozen)
                    outF.write(f"Percentage of frozen in {name}: {torch.mean(ones_frozen)}.\n")
                    print(
                        f"Percentage of frozen in {name}: {torch.mean(ones_frozen)}",
                        flush=True,
                    )
        overall_frozen = torch.cat(overall_frozen)
        outF.write(f"Overall percentage of frozen parameters: {torch.mean(overall_frozen)}.\n")
        print(
            f"Overall percentage of frozen parameters: {torch.mean(overall_frozen)}",
            flush=True,
        )

    total_time = time.time() - start_time
    outF.write(f"Time spent: {total_time}.")
    print(f"Time spent: {total_time}.", flush=True)
    # Print last test accuracy obtained
    last_test_accuracy = test_acc_epochs[-1]
    outF.write("Test accuracy: {}".format(last_test_accuracy))
    print("Test accuracy: {}".format(last_test_accuracy))
    outF.write("\n")
    outF.flush()


def compute_masked_net_stats(masked_net, trainloader, epoch, device, criterion, model_name, num_classes):
    if model_name == "resnet18":
        test_net = ResNet18(num_classes=num_classes)
    elif model_name == "resnet50":
        test_net = ResNet50(num_classes=num_classes)
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
        choices=[
            "resnet18",
            "resnet50",
            "wrn2810",
            "deit_tiny_patch16_224",
            "deit_small_patch16_224",
            "deit_base_patch16_224",
        ],
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
        choices=["constant", "onecycle"],
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
    # Training arguments
    parser.add_argument(
        "--print_batch_stat_freq",
        default=1,
        type=int,
        help="print batch statistics after every few eopochs specified by this argument",
    )
    parser.add_argument(
        "--max_physical_batch_size",
        default=100,
        type=int,
        help="DP training max physical batch size",
    )
    parser.add_argument(
        "--epoch_mask_finding",
        default=10,
        type=int,
        help="epoch after which we switch from all-layer finetuning to sparse finetuning.",
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
        "--method_name",
        type=str,
        choices=[
            "linear_probing",
            "lp_gn",
            "conf_indices",
            "lora",
            "first_last",
            "all_layers",
            "mp_weights",
            "optim_weights_noisy_grads",
            "optim_averaged_noisy_grads",
            "optim_averaged_clipped_grads",
            "row_pruning_noisy_grads",
            "block_pruning_noisy_grads",
            "",
        ],
        default="",
        help="chooses type of mask to be applied. The default '' is equivalent to 'all_layers'.",
    )
    parser.add_argument(
        "--use_delta_weight_optim",
        type=str2bool,
        nargs="?",
        default=True,
        help="uses the delta_weight after optimization or not.",
    )
    parser.add_argument(
        "--use_fixed_w_mask_finding",
        type=str2bool,
        nargs="?",
        default=True,
        help="update_w or not during dp_sgd.",
    )
    parser.add_argument(
        "--use_cosine_more_epochs",
        type=str2bool,
        nargs="?",
        default=True,
        help="lr doesn't collapse near end of training.",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.0,
        help="percentage of weights that are non-trainable.",
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
    # Random seed
    parser.add_argument("--seed", default=0, type=int, help="RNG seed")
    parser.add_argument("--SLURM_JOB_ID", type=int, default=-1)
    parser.add_argument("--TASK_ID", type=int, default=-1)
    # For DDP, not set by user

    args = parser.parse_args()

    set_seed(args.seed)

    args.out_file = os.path.join(
        "results_folder",
        args.experiment_dir,
        args.model + "_" + args.dataset + "_" + args.method_name + str(args.seed) + "_" + str(args.SLURM_JOB_ID) + "_" + str(args.TASK_ID) + ".txt",
    )

    use_cuda = torch.cuda.is_available()
    print(f"use_cuda={use_cuda}.")

    # These are constraints to run on V100 GPUs with 32GB of RAM
    if "deit" in args.model and "base" in args.model:
        args.max_physical_batch_size = 10
    elif "deit" in args.model:
        args.max_physical_batch_size = 100

    if args.dataset == "cifar100":
        args.epoch_mask_finding = 10

    args.mask_type = ""
    if args.method_name in [
        "mp_weights",
        "mp_adaptive_weights",
        "mp_adaptive_noisy_grads",
    ]:
        args.mask_type = "magnitude_pruning"
    elif args.method_name in [
        "optim_weights_noisy_grads",
        "optim_averaged_noisy_grads",
        "optim_averaged_clipped_grads",
        "row_pruning_noisy_grads",
        "block_pruning_noisy_grads",
    ]:
        args.mask_type = "optimization"

    # These are not used in dp. Other parameters are going to substitute them
    main_trainer(args, use_cuda)
