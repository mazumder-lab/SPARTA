# %%
import argparse
import copy
import math
import os
import sys

import torch
import torch.cuda
import torch.multiprocessing as mp
import torch.nn as nn
import torch_pruning as tp
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator
from torch.nn.parallel import DistributedDataParallel as DDP

from conf.global_settings import CHECKPOINT_PATH, INDICES_LIST, MAX_PHYSICAL_BATCH_SIZE
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
from train_cifar import train_single_epoch, train_vanilla_single_step
from utils.train_utils import (
    compute_test_stats,
    count_parameters,
    set_seed,
    smooth_crossentropy,
    str2bool,
)

sys.path.append("/Users/mmakni/Desktop/NetworkPruning")
sys.path.append("/Users/mmakni/Desktop/NetworkPruning/Lagrangian-Heuristic")
from pruners.Layer_pruner import LayerPruner

# %%
train_loader, test_loader = get_train_and_test_dataloader(
    dataset="cifar10",
    batch_size=128,
    world_size=1,
    rank=0,
)

net = ResNet18(num_classes=100)
net.train()
net = ModuleValidator.fix(net.to("cpu"))
net.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=torch.device("cpu")))
net.linear = nn.Linear(
    in_features=net.linear.in_features,
    out_features=10,
    bias=net.linear.bias is not None,
)
net.eval()
criterion = smooth_crossentropy

# %%
modules_to_prune = []
for name, param in net.named_parameters():
    layer_name, param_name = ".".join(name.split(".")[:-1]), name.split(".")[-1]
    if "conv" in layer_name:
        modules_to_prune.append(name)
# %%
model_pruned = copy.deepcopy(net)
# %%
pruner = LayerPruner(
    model_pruned,
    modules_to_prune,
    train_loader.dataset,
    train_loader,
    test_loader,
    200,
    criterion,
    0.0001,
    0,
    "cuda",
    "Back",
    0,
    1,
    0,
    0,
    False,
    False,
)

# %%
w_pruned = pruner.prune_unstr(0.5)
# %%

# %%

# %%

# # %%
# imp = tp.importance.GroupTaylorImportance()
# example_inputs = torch.randn(1, 3, 32, 32)
# ignored_layers = []
# for m in net.modules():
#     if not isinstance(m, nn.Conv2d):
#         ignored_layers.append(m)

# pruner = tp.pruner.MetaPruner(  # We can always choose MetaPruner if sparse training is not required.
#     net,
#     example_inputs,
#     importance=imp,
#     pruning_ratio=0.6,  # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
#     # pruning_ratio_dict = {model.conv1: 0.2, model.layer2: 0.8}, # customized pruning ratios for layers or blocks
#     ignored_layers=ignored_layers,
# )
# # %%
# base_macs, base_nparams = tp.utils.count_ops_and_params(net, example_inputs)
# if isinstance(imp, tp.importance.GroupTaylorImportance):
#     # Taylor expansion requires gradients for importance estimation
#     inputs, targets = next(iter(train_loader))
#     outputs = net(inputs)
#     loss = criterion(outputs, targets, smoothing=0.0).mean()
#     # Backward pass
#     loss.mean().backward()

# # %%
# pruner.step()
# macs, nparams = tp.utils.count_ops_and_params(net, example_inputs)
# # %%

# # %%

# # %%

# # %%
