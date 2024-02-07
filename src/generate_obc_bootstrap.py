import argparse
import os
import pickle

import torch
import torch.cuda
from opacus.validators import ModuleValidator

from models.resnet import ResNet18

parser = argparse.ArgumentParser(description="Generate obc masks.")
parser.add_argument(
    "--sparsity",
    default=0.5,
    type=float,
    help="mask sparsity",
)
args = parser.parse_args()


sparsity = args.sparsity
n_datasets = 20
checkpoint_path = "../checkpoints/models_unstr_bootstrap"
os.makedirs(checkpoint_path, exist_ok=True)
out_pickle = f"{checkpoint_path}/resnet18_obc_bootstrap_{int(sparsity*100)}.pkl"

net = ResNet18(num_classes=10)
net.train()
net = ModuleValidator.fix(net.to("cpu"))
device = torch.device(f"cuda:{0}") if torch.cuda.is_available() else "cpu"

mask = {name: torch.zero_like(param) for name, param in net.named_parameters()}


for idx in range(n_datasets):
    print(idx)
    net.load_state_dict(
        torch.load(
            checkpoint_path + f"/{idx}_{n_datasets}_resnet18_{args.sparsity * 10000}", map_location=torch.device("cpu")
        )
    )
    for name, param in net.named_parameters():
        mask[name] += (param != 0.0).float()

for name, _ in net.named_parameters():
    mask[name] /= n_datasets
    print(torch.mean(mask[name]))

with open(out_pickle, "wb") as file:
    pickle.dump(mask, file)
