""" configurations for this project

author baiyu
"""

import os
from datetime import datetime

# mean and std of cifar100 dataset
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

# CIFAR100_TEST_MEAN = (0.5088964127604166, 0.48739301317401956, 0.44194221124387256)
# CIFAR100_TEST_STD = (0.2682515741720801, 0.2573637364478126, 0.2770957707973042)
EPOCH_MASK_FINDING = 2
BATCH_FINAL = 20
MAX_PHYSICAL_BATCH_SIZE = 100
# This value could be changed for experimental checks when we reduce the noise variance. Should be 1.0 for dp guarantees.
EXPERIMENTAL_DIVISION_COEFF = 1.0

# directory to save weights file
OBC_PATH = "../checkpoints/"
BOOTSTRAP_PATH = "../checkpoints/models_unstr_bootstrap/"
CHITA_PATH = "../checkpoints/chita_masks/"
CVX_CHITA_PATH = "../checkpoints/cvx_chita/"
CHECKPOINT_PATH = "../checkpoints/lsr=01train_resnet_gn.pt"
CHECKPOINT_WRN_PATH = "../checkpoints/wrn_2810_imagenet32_gn.pt"

IMAGENET32_PATH = "/home/gridsan/mmakni/BoxImageNet32/raw"

MASK_1_PATH = "resnet18_100.pth"
MASK_10_PATH = "resnet18_1000.pth"
MASK_20_PATH = "resnet18_2000.pth"
MASK_30_PATH = "resnet18_3000.pth"
MASK_40_PATH = "resnet18_4000.pth"
MASK_50_PATH = "resnet18_5000.pth"
MASK_60_PATH = "resnet18_6000.pth"
MASK_70_PATH = "resnet18_7000.pth"
MASK_80_PATH = "resnet18_8000.pth"
MASK_90_PATH = "resnet18_9000.pth"
BOOTSTRAP_MASK_10_PATH = "resnet18_obc_bootstrap_10.pkl"
BOOTSTRAP_MASK_20_PATH = "resnet18_obc_bootstrap_20.pkl"
BOOTSTRAP_MASK_30_PATH = "resnet18_obc_bootstrap_30.pkl"
BOOTSTRAP_MASK_40_PATH = "resnet18_obc_bootstrap_40.pkl"
BOOTSTRAP_MASK_50_PATH = "resnet18_obc_bootstrap_50.pkl"
BOOTSTRAP_MASK_60_PATH = "resnet18_obc_bootstrap_60.pkl"
BOOTSTRAP_MASK_70_PATH = "resnet18_obc_bootstrap_70.pkl"
BOOTSTRAP_MASK_80_PATH = "resnet18_obc_bootstrap_80.pkl"
BOOTSTRAP_MASK_90_PATH = "resnet18_obc_bootstrap_90.pkl"
CHITA_MASK_80_PATH = "chita_model_80.pth"
CHITA_MASK_50_PATH = "chita_model_50.pth"
CHITA_MASK_20_PATH = "chita_model_20.pth"

CVX_CHITA_MASK_10_PATH = "resnet18_1000.pth"
CVX_CHITA_MASK_20_PATH = "resnet18_2000.pth"
CVX_CHITA_MASK_30_PATH = "resnet18_3000.pth"
CVX_CHITA_MASK_40_PATH = "resnet18_4000.pth"
CVX_CHITA_MASK_50_PATH = "resnet18_5000.pth"
CVX_CHITA_MASK_60_PATH = "resnet18_6000.pth"
CVX_CHITA_MASK_70_PATH = "resnet18_7000.pth"
CVX_CHITA_MASK_80_PATH = "resnet18_8000.pth"
CVX_CHITA_MASK_90_PATH = "resnet18_9000.pth"


INDICES_LIST = [1, 14, 17, 20, 32, 35, 37, 40, 43, 46, 54, 55, 59, 60, 61]
# INDICES_LIST = [
#     0,
#     6,
#     7,
#     9,
#     10,
#     11,
#     14,
#     15,
#     21,
#     29,
#     30,
#     32,
#     33,
#     34,
#     36,
#     38,
#     40,
#     41,
#     42,
#     45,
#     49,
#     50,
#     51,
#     53,
#     54,
#     56,
#     57,
#     60,
#     61,
# ]
# total training epoches
EPOCH = 200
MILESTONES = [60, 120, 160]

# initial learning rate
# INIT_LR = 0.1

# time of we run the script
TIME_NOW = datetime.now().strftime("%A_%d_%B_%Y_%Hh_%Mm_%Ss")

# tensorboard log dir
LOG_DIR = "runs"

# save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10
