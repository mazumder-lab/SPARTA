import argparse
import copy
import gc
import os
import time

import torch
import torch.cuda
import torch.nn as nn
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator

from models.deit import (
    deit_base_patch16_224,
    deit_small_patch16_224,
    deit_tiny_patch16_224,
)
from models.resnet import ResNet18
from models.wide_resnet import Wide_ResNet
from opacus_per_sample.privacy_engine_per_sample import PrivacyEnginePerSample
from utils.change_modules import fix, fully_trainable_modules
from utils.dataset_utils import get_train_and_test_dataloader
from utils.train_utils import (
    compute_test_stats,
    count_parameters,
    layerwise_magnitude_pruning,
    set_seed,
    smooth_crossentropy,
    str2bool,
    train_single_epoch,
    use_finetune_optimizer,
    use_lr_scheduler,
)
from peft import (
    LoraConfig,
    get_peft_model
)

def masking_cond(name):
    return (resnet_masking_cond(name) and "resnet" in args.model) or (deit_masking_cond(name) and "deit" in args.model)

def deit_masking_cond(name):
    return ("blocks" in name) or ("norm" in name) or ("patch_embed" in name)

def resnet_masking_cond(name):
    return ("conv" in name) or ("shortcut" in name)

def last_layer_cond(name):
    return ("linear" in name and "resnet" in args.model) or ("head" in name and "deit" in args.model)


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
        checkpoint_path = "../checkpoints/resnet18_cifar100_gn.pt"
    elif args.model == "wideresnet2810":
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

    net.train()
    net = ModuleValidator.fix(net.to("cpu"))
    print(net)

    if args.model == "resnet18":
        pretrained_weights = torch.load(checkpoint_path, map_location="cpu")
        if args.num_classes != 100:  # pretrained on cifar100
            del pretrained_weights["linear.weight"]
            del pretrained_weights["linear.bias"]
    elif args.model == "wideresnet2810":
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
            if "deit" in args.model:
                if "head" not in name:
                    param.requires_grad = False
            else:
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
    elif args.method_name == "dp_bitfit":
        for name, param in net.named_parameters():
            if "deit" in args.model:
                if ("head" not in name) and ("bias" not in name):
                    param.requires_grad = False
            else:
                if ("linear" not in name) and ("bn" not in name) and ("bias" not in name):
                    param.requires_grad = False
    elif args.method_name == "lora":
        target_modules = [name for name, module in fully_trainable_modules(net) if type(module) in [nn.Linear, nn.Conv2d] and not last_layer_cond(name)]
        peft_config = LoraConfig(
            r=args.num_ranks,
            target_modules=target_modules,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias)
        net = get_peft_model(net, peft_config)
        for name, param in net.named_parameters():
            if last_layer_cond(name):
                param.requires_grad = True
    elif args.method_name == "all_layers":
        # keep all parameters trainable
        pass

    # Print the name of Trainable parameters
    print("--- List of trainable params ---", flush=True)
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name, flush=True)
    print("--- Done ---", flush=True)

    if args.mask_type:
        # If we are using any type if masking, then introduce the partially trainable $W_{\text{old} + m \odot W$ formulation
        new_net = copy.deepcopy(net)
        new_net = fix(new_net, partially_trainable_bias=False)
        # Get the state dictionaries of both networks
        net_state_dict = net.state_dict()
        new_net_state_dict = new_net.state_dict()
        # Create an initial mask with magnitude pruning if it is being used solely or as a convex combination
        if args.mask_type == "magnitude_pruning":
            new_net_state_dict = layerwise_magnitude_pruning(
                net_state_dict,
                new_net_state_dict,
                args.sparsity,
                descending=False,
            )
            for name in new_net_state_dict:
                if "mask" in name and last_layer_cond(name):
                    new_net_state_dict[name] = torch.ones_like(new_net_state_dict[name])

        elif args.mask_type == "random":
            for name in new_net_state_dict:
                if "mask" in name and last_layer_cond(name):
                    new_net_state_dict[name] = torch.ones_like(new_net_state_dict[name])
                elif "mask" in name:
                    num_elements = new_net_state_dict[name].numel()
                    random_mask = torch.ones(num_elements)
                    random_mask[: int(num_elements * (1 - args.sparsity))] = 0
                    random_mask = random_mask[torch.randperm(num_elements)]
                    new_net_state_dict[name] = random_mask.view_as(new_net_state_dict[name])

        elif args.mask_type == "optimization" and args.use_last_layer_only_init:
            for name in new_net_state_dict:
                if "mask" in name and masking_cond(name):
                    new_net_state_dict[name] = torch.zeros_like(new_net_state_dict[name])

        # Now copy the initial weights in the right place in the new formulation and delete the previous architecture if it is not used.
        for name in new_net_state_dict:
            if "init" in name:
                original_name = name.replace("init_", "")
                new_net_state_dict[name] = net_state_dict[original_name]
            elif "_trainable" not in name:
                # This is for parameters that remain unchanged in the new formulation
                new_net_state_dict[name] = net_state_dict[name]
            
        new_net.load_state_dict(new_net_state_dict)
        net, old_net = new_net, net.to("cpu")
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
        if param.requires_grad:
            trainable_indices.append(idx)
            trainable_names.append(name)
            if last_layer_cond(name):
                classifier_params.append(param)
                print("Classifier layer:", name)
            elif masking_cond(name):
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
    else:
        raise Exception("SGD is the only optimizer implemented for now.")

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
            len(train_loader.dataset),
            args.batch_size,
            args.classifier_lr,
            args.lr,
            args.num_epochs,
            args.warm_up,
        )
    else:
        raise Exception("Onecycle is the only scheduler implemented for now.")

    # STEP [7] - Run epoch-wise training and validation
    print("training for {} epochs".format(args.num_epochs))
    addr = args.out_file
    outF = open(addr, "w")
    outF.write(str(args))
    outF.write("\n")
    outF.write(f"The indices of trainable parameters are: {trainable_indices}.")
    outF.write("\n")
    outF.write(f"The names of trainable parameters are: {trainable_names}.")
    outF.write("\n")
    outF.write(f"The number of trainable parameters is: {nb_trainable_params}.")
    outF.write("\n")
    outF.write(f"Using sigma={optimizer.noise_multiplier} and C={args.clipping}")
    outF.write("\n")
    outF.flush()

    start_time = time.time()
    with BatchMemoryManager(
        data_loader=train_loader,
        max_physical_batch_size=args.max_physical_batch_size,
        optimizer=optimizer,
    ) as memory_safe_data_loader:
        for epoch in range(args.num_epochs):
            # Run training for single epoch
            if args.mask_type == "optimization" and epoch == args.epoch_mask_finding:
                optimizer.method_name = args.method_name
                if args.use_fixed_w_mask_finding:
                    original_lrs = [group["lr"] for group in optimizer.param_groups]
                    for group in optimizer.param_groups:
                        group["lr"] = 0.0
                if args.use_last_layer_only_init:
                    # We relax the mask when lr==0.0 so that row_groups receive private absolute gradients and we can rank them by importance
                    net_state_dict = net.state_dict()
                    for name_mask in net_state_dict:
                        if "mask" in name_mask and masking_cond(name_mask):
                            name_weight = name_mask.replace("mask_", "")
                            net_state_dict[name_mask] = torch.ones_like(net_state_dict[name_mask])
                            # Very important to set previously screened variables to 0 once they can become trainable so as not to change output.
                            net_state_dict[name_weight] = torch.zeros_like(net_state_dict[name_weight])
                    net.load_state_dict(net_state_dict)

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
                to_resize="deit" in args.model,
            )
            gc.collect()
            torch.cuda.empty_cache()
            if args.mask_type == "optimization" and epoch == args.epoch_mask_finding and args.use_fixed_w_mask_finding:
                for group, original_lr in zip(optimizer.param_groups, original_lrs):
                    group["lr"] = original_lr
            if args.mask_type == "optimization" and epoch == args.epoch_mask_finding:
                print(f"Start the mask finding procedure with the method_name={args.method_name}", flush=True)
                init_weights = None
                if args.method_name in [
                    "row_pruning_noisy_grads",
                    "row_pruning_weighted_noisy_grads",
                    "optim_averaged_noisy_grads",
                    "optim_averaged_clipped_grads",
                    "optim_weights_noisy_grads",
                    "optim_weights_clipped_grads",
                ]:
                    net_state_dict = net.state_dict()
                    init_weights = []
                    for init_name in net.state_dict():
                        if "init" in init_name and not last_layer_cond(init_name):
                            name_mask = init_name.replace("init_", "mask_") + "_trainable"
                            name_weight = init_name.replace("init_", "") + "_trainable"
                            real_weight = (
                                net_state_dict[init_name] + net_state_dict[name_weight] * net_state_dict[name_mask]
                            )
                            init_weights.append(real_weight)
                    del net_state_dict

                optimizer.get_optimization_method_mask(args.sparsity, init_weights)

                # Update the masks
                net_state_dict = net.state_dict()
                init_names = [name for name in net_state_dict if "init" in name and masking_cond(name)]
                masked_params = [p for p in optimizer.param_groups[1]["params"] if p.mask is not None]
                for p, init_name in zip(masked_params, init_names):
                    name_mask = init_name.replace("init_", "mask_") + "_trainable"
                    name_weight = init_name.replace("init_", "") + "_trainable"
                    real_weight = net_state_dict[init_name] + net_state_dict[name_weight] * net_state_dict[name_mask]
                    net_state_dict[init_name] = real_weight
                    net_state_dict[name_weight] = torch.zeros_like(real_weight)
                    net_state_dict[name_mask] = p.mask.view_as(net_state_dict[name_mask])
                net.load_state_dict(net_state_dict)
                old_net = copy.deepcopy(net).to("cpu")
                optimizer.clear_momentum_grad()

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

            epsilon = privacy_engine.get_epsilon(args.delta)
            print(f"Current privacy budget spent: {epsilon}.")
            if epsilon > args.epsilon:
                print(f"Stopping training at epoch={epoch} with epsilon={epsilon}.")
                break

    # STEP [8] - Run sparsity checks on all parameters
    gc.collect()
    torch.cuda.empty_cache()
    if args.mask_type:
        outF.write("Starting Sparsity Analysis.\n")
        print("Starting Sparsity Analysis.\n", flush=True)
        old_net.to(device)
        net_state_dict = net.state_dict()
        old_net_state_dict = old_net.state_dict()
        overall_frozen = []
        for init_name in net_state_dict:
            if "init" in init_name:
                original_name = init_name.replace("init_", "").replace("_module.", "")
                name_mask = init_name.replace("init_", "mask_") + "_trainable"
                name_weight = init_name.replace("init_", "") + "_trainable"
                param = net_state_dict[init_name] + net_state_dict[name_mask] * net_state_dict[name_weight]
                old_name = original_name if original_name in old_net_state_dict else init_name
                diff_param = param - old_net_state_dict[old_name]
                ones_frozen = (diff_param == 0).float().reshape(-1)
                overall_frozen.append(ones_frozen)
                outF.write(f"Percentage of frozen in {original_name}: {torch.mean(ones_frozen)}.\n")
                print(
                    f"Percentage of frozen in {original_name}: {torch.mean(ones_frozen)}",
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
    outF.write("Test accuracy: {}".format(test_acc))
    print("Test accuracy: {}".format(test_acc))
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
        choices=[
            "resnet18",
            "resnet50",
            "wideresnet2810",
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
        choices=[10, 11, 100],
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
    parser.add_argument(
        "--method_name",
        type=str,
        choices=[
            "linear_probing",
            "lp_gn",
            "lora",
            "all_layers",
            "mp_weights",
            "dp_bitfit",
            "optim_weights_noisy_grads",
            "optim_weights_clipped_grads",
            "optim_averaged_noisy_grads",
            "optim_averaged_clipped_grads",
            "row_pruning_noisy_grads",
            "row_pruning_weighted_noisy_grads",
            "random_masking",
            "",
        ],
        default="",
        help="chooses type of mask to be applied. The default '' is equivalent to 'all_layers'.",
    )
    parser.add_argument(
        "--use_fixed_w_mask_finding",
        type=str2bool,
        nargs="?",
        default=True,
        help="update_w or not during dp_sgd.",
    )
    parser.add_argument(
        "--use_last_layer_only_init",
        type=str2bool,
        nargs="?",
        default=False,
        help="start training with the last layer only, instead of all_layers.",
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
    parser.add_argument(
        "--num_ranks",
        type=int,
        default=16,
        help="number of LoRA ranks",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=0.5,
        help="lora alpha",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.0,
        help="lora dropout",
    )
    parser.add_argument(
        "--lora_bias",
        default="all",
        type=str,
        choices=[
            "all",
            "none",
        ],
        help="train all biases or none of them",
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
    
    os.makedirs(os.path.join("results_folder", args.experiment_dir), exist_ok=True)
    args.out_file = os.path.join(
        "results_folder",
        args.experiment_dir,
        args.model
        + "_"
        + args.dataset
        + "_"
        + args.method_name
        + "_"
        + str(args.epsilon)
        + "_"
        + str(args.sparsity)
        + "_"
        + str(args.epoch_mask_finding)
        + "_"
        + str(args.use_last_layer_only_init)
        + "_"
        + str(args.seed)
        + "_"
        + str(args.TASK_ID)
        + ".txt",
    )

    use_cuda = torch.cuda.is_available()
    print(f"use_cuda={use_cuda}.")

    args.mask_type = ""
    if args.method_name in [
        "mp_weights",
    ]:
        args.mask_type = "magnitude_pruning"
    elif args.method_name in [
        "random_masking",
    ]:
        args.mask_type = "random"
    elif args.method_name in [
        "optim_weights_noisy_grads",
        "optim_weights_clipped_grads",
        "optim_averaged_noisy_grads",
        "optim_averaged_clipped_grads",
        "row_pruning_noisy_grads",
        "row_pruning_weighted_noisy_grads",
    ]:
        args.mask_type = "optimization"

    main_trainer(args, use_cuda)
