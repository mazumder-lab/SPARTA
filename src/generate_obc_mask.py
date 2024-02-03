# %%
import argparse
import math
import os
import pickle

import torch
import torch.cuda
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from opacus.validators import ModuleValidator

from conf.global_settings import CHECKPOINT_PATH
from dataset_utils import get_train_and_test_dataloader
from finegrain_utils.resnet_mehdi import ResNet18_partially_trainable
from models.resnet import ResNet18
from utils.train_utils import set_seed
from utils_pruning_mehdi import prune_block

parser = argparse.ArgumentParser(description="Generate obc masks.")
parser.add_argument(
    "--sparsity",
    default=0.5,
    type=float,
    help="mask sparsity",
)
args = parser.parse_args()


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


set_seed(0)
"""The goal of this test is to verify that using the mask as entirely
zeros or ones is entirely equivalent to finetuning using
requires_grad=False/True for each layer."""
dataset = "cifar10"
batch_size = 128
sparsity = args.sparsity
checkpoint_path = "../checkpoints/new_obc_eval"
os.makedirs(checkpoint_path, exist_ok=True)
out_pickle = f"{checkpoint_path}/resnet18_mask{int(sparsity*100)}.pkl"

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

net = net.to(device)
net.train()
with torch.no_grad():
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
# new eval line added here.
net.eval()
prune_block(net, train_loader, device, sparsity, "obc", 1e-2)


net = net.to("cpu")
mask = {}

for name, param in net.named_parameters():
    mask[name] = (param.data == 0.0).float()

with open(out_pickle, "wb") as f:
    pickle.dump({"net": net, "mask": mask}, f)
