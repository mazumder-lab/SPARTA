import argparse
import math
import os
import pickle
import time

import torch
import torch.cuda
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator
from torch.nn.parallel import DistributedDataParallel as DDP

from conf.global_settings import (
    BATCH_FINAL,
    CHECKPOINT_PATH,
    CHECKPOINT_WRN_PATH,
    EPOCH_MASK_FINDING,
    EXPERIMENTAL_DIVISION_COEFF,
    INDICES_LIST,
    MAX_PHYSICAL_BATCH_SIZE,
)
from dataset_utils import get_train_and_test_dataloader

# from models.resnet import ResNet18, ResNet50
from finegrain_utils.resnet_mehdi import ResNet18_partially_trainable
from finegrain_utils.wide_resnet_mehdi import WRN2810_partially_trainable
from models.resnet import ResNet18, ResNet50
from models.wide_resnet import Wide_ResNet
from opacus_per_sample.optimizer_per_sample import DPOptimizerPerSample
from opacus_per_sample.privacy_engine_per_sample import PrivacyEnginePerSample
from optimizers.optimizer_utils import (
    use_finetune_optimizer,
    use_warmup_cosine_scheduler,
)
from utils.train_utils import (
    compute_test_stats,
    count_parameters,
    global_magnitude_pruning,
    layerwise_magnitude_pruning,
    set_seed,
    smooth_crossentropy,
    str2bool,
    update_global_magnitude_mask,
    update_global_noisy_grad_mask,
    update_magnitude_mask,
    update_noisy_grad_mask,
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
    lr_schedule_type="warmup_cosine",
    sparsity=1.0,
    mask_type="",
    method_name="",
    use_w_tilde=False,
    correction_coefficient=0.1,
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
    if mask_type == "optimization" and epoch == EPOCH_MASK_FINDING:
        optimizer.compute_fisher_mask = True
        optimizer.use_w_tilde = use_w_tilde
        optimizer.method_name = method_name
    optimizer.zero_grad()

    old_net = None
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
            lr_schedule_type=lr_schedule_type,
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
                )
            )
        if mask_type == "optimization" and epoch == EPOCH_MASK_FINDING and batch_idx == BATCH_FINAL:
            break

    if mask_type == "optimization" and epoch == EPOCH_MASK_FINDING and optimizer.compute_fisher_mask:
        net_state_dict = net.state_dict()
        init_weights = [net_state_dict[name] for name in net_state_dict if "init" in name]
        # if add_precision_clipping_and_noise:
        #     optimizer.get_H_inv_fisher_mask(init_weights, sparsity, correction_coefficient)
        # else:
        #     optimizer.get_fisher_mask(init_weights, sparsity, correction_coefficient)
        optimizer.get_optimization_method_mask(init_weights, sparsity, correction_coefficient)

        print("Starting to print")
        init_names = [name for name in net_state_dict if "init" in name]
        for p, init_name in zip(optimizer.param_groups[1]["params"], init_names):
            name_mask = init_name.replace("init_", "mask_") + "_trainable"
            name_weight = init_name.replace("init_", "") + "_trainable"
            real_weight = net_state_dict[init_name] + net_state_dict[name_weight] # * net_state_dict[name_mask] // The assumption is that the mask is initially all ones for the optimization methods
            net_state_dict[init_name] = real_weight
            net_state_dict[name_mask] = p.mask.view_as(net_state_dict[name_mask])
            net_state_dict[name_weight] = torch.zeros_like(real_weight)
        net.load_state_dict(net_state_dict)
        optimizer.clear_momentum_grad()
        old_net = compute_masked_net_stats(net, trainloader, epoch, device, criterion)

    if lr_schedule_type == "warmup_cosine":
        cosine_scheduler.step()
    # Print epoch-end stats
    acc = 100.0 * correct / total
    print(
        "For epoch number: {}, train loss: {} and accuracy: {}".format(epoch_number, train_loss / (batch_idx + 1), acc)
    )
    return old_net


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
    lr_schedule_type="warmup_cosine",
):
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
    # Step when there is a logical step
    if (lr_schedule_type != "warmup_cosine") and is_updated_logical_batch:
        lr_scheduler.step()
    elif (epoch == 0) and (lr_schedule_type == "warmup_cosine") and is_updated_logical_batch:
        lr_scheduler.step()
    # Return stuff
    return outputs, loss


#########################################################
def main_trainer(args, use_cuda):
    # TODO log this using MLFlow
    print("Parsed args: {}".format(args))

    # STEP [2] - Create train and test dataloaders
    train_loader, test_loader = get_train_and_test_dataloader(
        dataset=args.dataset,
        batch_size=args.batch_size,
    )
    print("train and test data loaders are ready")

    # STEP [3] - Create model and load pretrained weights (with Group Normalization).
    if args.model == "resnet18":
        net = ResNet18(num_classes=100)
    elif args.model == "resnet50":
        net = ResNet50(num_classes=100)
    elif args.model == "wrn2810":
        net = Wide_ResNet(
            depth=28,
            widen_factor=10,
            dropout_rate=0.3,
            num_classes=1000,
        )
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
        torch.cuda.set_device(0)
        device = torch.device(f"cuda:{0}")
    else:
        device = "cpu"

    if args.model == "resnet18":
        net.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=torch.device("cpu")))
    elif args.model == "wrn2810":
        net.load_state_dict(torch.load(CHECKPOINT_WRN_PATH, map_location=torch.device("cpu")))
    net.linear = nn.Linear(
        in_features=net.linear.in_features,
        out_features=args.num_classes,
        bias=net.linear.bias is not None,
    )

    # STEP [4] - Decide on which masking procedure to follow. Introduce the masking formulation W_old + m . W if the method is finegrained.
    if args.finetune_strategy == "linear_probing":
        for name, param in net.named_parameters():
            if "linear" not in name:
                param.requires_grad = False
    elif args.finetune_strategy == "lp_gn":
        for name, param in net.named_parameters():
            if ("linear" not in name) and ("bn" not in name):
                param.requires_grad = False
    elif args.finetune_strategy == "first_last":
        for idx, (name, param) in enumerate(net.named_parameters()):
            if ("linear" not in name) and ("bn" not in name) and (idx >= 15):
                param.requires_grad = False
    elif args.finetune_strategy == "conf_indices":
        for idx, (_, param) in enumerate(net.named_parameters()):
            if idx not in INDICES_LIST:
                param.requires_grad = False
    elif args.finetune_strategy == "all_layers":
        # keep all parameters trainable
        pass

    if args.mask_type:
        # If we are using any type if masking, then introduce the partially trainable $W_{\text{old} + m \odot W$ formulation
        if args.model == "resnet18":
            new_net = ResNet18_partially_trainable(num_classes=args.num_classes, with_mask=True)
        elif args.model == "wrn2810":
            new_net = WRN2810_partially_trainable(num_classes=args.num_classes, partially_trainable_bias=False)
        if args.use_gn:
            new_net.train()
            new_net = ModuleValidator.fix(new_net.to("cpu"))
        # Get the state dictionaries of both networks
        net_state_dict = net.state_dict()
        new_net_state_dict = new_net.state_dict()

        # Create an initial mask with magnitude pruning if it is being used solely or as a convex combination
        if args.method_name == "mp_weights":
            new_net_state_dict = layerwise_magnitude_pruning(
                net_state_dict, new_net_state_dict, args.sparsity, args.magnitude_descending
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
            optimizer, args.batch_size, args.classifier_lr, args.lr, args.num_epochs, args.warm_up
        )
    elif args.lr_schedule_type == "warmup_cosine":
        # TODO incorporate world size
        lr_scheduler = use_warmup_cosine_scheduler(
            optimizer=optimizer, num_epochs=args.num_epochs, total_steps=len(train_loader)
        )

    # STEP [7] - Run epoch-wise training and validation
    print("training for {} epochs".format(args.num_epochs))
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
    outF.write(f"Using sigma={optimizer.noise_multiplier} and C={args.clipping}")
    outF.write("\n")
    outF.flush()

    start_time = time.time()
    test_acc_epochs = []
    with BatchMemoryManager(
        data_loader=train_loader,
        max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
        optimizer=optimizer,
    ) as memory_safe_data_loader:
        old_net = net
        for epoch in range(args.num_epochs):
            # Run training for single epoch
            ret = train_single_epoch(
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
                lr_schedule_type=args.lr_schedule_type,
                sparsity=args.sparsity,
                mask_type=args.mask_type,
                method_name=args.method_name,
                use_w_tilde=args.use_w_tilde,
                correction_coefficient=args.correction_coefficient,
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

            epsilon = privacy_engine.get_epsilon(args.delta)
            print(f"Current privacy budget spent: {epsilon}.")
            if epsilon > args.epsilon:
                print(f"Stopping training at epoch={epoch} with epsilon={epsilon}.")
                break

            if ((args.mask_type == "magnitude") and ("adaptive" in args.method_name)) and ((epoch + 1) % 10 == 0):
                # reset momentum buffer in optimizer
                optimizer.clear_momentum_grad()
                if args.method_name == "mp_adaptive_weights":
                    net = update_magnitude_mask(net, args)
                elif args.method_name == "mp_adaptive_noisy_grads":
                    net = update_noisy_grad_mask(net, args)

    # STEP [8] - Run sparsity checks on all parameters
    if args.mask_type:
        outF.write("Starting Sparsity Analysis.\n")
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
                    diff_param = (param - old_net_state_dict[name])
                    ones_frozen = (diff_param == 0).float().reshape(-1)
                    overall_frozen.append(ones_frozen)
                    outF.write(f"Percentage of frozen in {name}: {torch.mean(ones_frozen)}.\n")
        overall_frozen = torch.cat(overall_frozen)
        outF.write(f"Overall percentage of frozen parameters: {torch.mean(overall_frozen)}.\n")

    total_time = time.time() - start_time
    outF.write(f"Time spent: {total_time}.")
    # Print last test accuracy obtained
    last_test_accuracy = test_acc_epochs[-1]
    print("Test accuracy: {}".format(last_test_accuracy))
    outF.write("Test accuracy: {}".format(last_test_accuracy))
    outF.write("\n")
    outF.flush()


def compute_masked_net_stats(masked_net, trainloader, epoch, device, criterion):
    test_net = ResNet18(num_classes=10)
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
    )

    for original_name in masked_net_state_dict:
        if "init" in original_name:
            name_mask = original_name.replace("init_", "mask_") + "_trainable"
            name = original_name.replace("_module.", "").replace("init_", "")
            param = masked_net_state_dict[original_name]
            test_net_state_dict[name] = param
        elif "_trainable" not in original_name:
            test_net_state_dict[original_name.replace("_module.", "")] = masked_net_state_dict[original_name]
    test_net.load_state_dict(test_net_state_dict)
    return test_net


def use_lr_scheduler(optimizer, batch_size, classifier_lr, lr, num_epochs, warm_up=0.2):
    steps_per_epoch = int(math.ceil(50000 / batch_size))
    # TODO improve this
    print("steps_per_epoch: {}".format(steps_per_epoch))
    lr_schedule = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[classifier_lr, lr, lr],
        epochs=int(num_epochs * 1.2),
        steps_per_epoch=steps_per_epoch,
        pct_start=warm_up,
    )
    return lr_schedule


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
        choices=["resnet18", "resnet50", "wrn2810"],
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
    # Training arguments
    parser.add_argument(
        "--finetune_strategy",
        type=str,
        choices=["linear_probing", "lp_gn", "conf_indices", "lora", "first_last", "all_layers"],
        default="all_layers",
        help="how to finetune the model.",
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
        "--mask_type",
        choices=["magnitude", "optimization", ""],
        default="",
        help="chooses type of mask to be applied if adaptive magnitude mask is true.",
    )
    parser.add_argument(
        "--method_name",
        type=str,
        choices=[
            "mp_weights",
            "mp_adaptive_weights",
            "mp_adaptive_noisy_grads",
            "mp_weights_grads",
            "optim_weights_noisy_grads",
            "optim_averaged_noisy_grads",
            "optim_fisher_with_true_grads",
            "optim_fisher_with_clipped_true_grads",
            "optim_fisher_with_noisy_grads",
            "optim_fisher_noisy_hessian",
            "optim_noisy_precision",
            "",
        ],
        default="",
        help="chooses type of mask to be applied if adaptive magnitude mask is true.",
    )
    parser.add_argument(
        "--magnitude_descending",
        type=str2bool,
        nargs="?",
        default=False,
        help="magnitude from smaller to bigger.",
    )
    parser.add_argument(
        "--use_w_tilde",
        type=str2bool,
        nargs="?",
        default=False,
        help="Use W Tilde - only for maks of type optimization.",
    )
    parser.add_argument(
        "--correction_coefficient",
        type=float,
        default=0.1,
        help="correction_coefficient - only for maks of type optimization if use_w_tilde is True.",
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

    args = parser.parse_args()

    set_seed(args.seed)

    args.out_file = os.path.join(
        "results_folder", args.experiment_dir, str(args.SLURM_JOB_ID) + "_" + str(args.TASK_ID) + "_" + args.out_file
    )
    args.save_file = os.path.join(
        "results_folder", args.experiment_dir, str(args.SLURM_JOB_ID) + "_" + str(args.TASK_ID) + "_" + args.save_file
    )
    torch.backends.cudnn.benchmark = True

    use_cuda = torch.cuda.is_available()
    print("use_cuda={use_cuda}.")

    # These are not used in dp. Other parameters are going to substitute them
    main_trainer(args, use_cuda)
