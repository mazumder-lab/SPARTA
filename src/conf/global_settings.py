""" configurations for this project

author baiyu
"""
import os
from datetime import datetime

# CIFAR100 dataset path (python version)
# CIFAR100_PATH = '/nfs/private/cifar100/cifar-100-python'

# mean and std of cifar100 dataset
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

# CIFAR100_TEST_MEAN = (0.5088964127604166, 0.48739301317401956, 0.44194221124387256)
# CIFAR100_TEST_STD = (0.2682515741720801, 0.2573637364478126, 0.2770957707973042)

MAX_PHYSICAL_BATCH_SIZE = 100
# directory to save weights file
CHECKPOINT_PATH = "../checkpoints/lsr=01train_resnet_gn.pt"
MASK_20_PATH = "../checkpoints/mask20_resnet18.pkl"
MASK_10_PATH = "../checkpoints/resnet18_mask10.pkl"
MASK_1_PATH = "../checkpoints/resnet18_mask1.pkl"
MASK_50_PATH = "../checkpoints/resnet18_mask50.pkl"
MASK_80_PATH = "../checkpoints/resnet18_mask80.pkl"
INDICES_LIST = [1, 14, 17, 20, 32, 35, 37, 40, 43, 46, 54, 55, 59, 60, 61]
# INDICES_LIST = [1, 14, 17, 20, 32, 35, 37, 40, 43, 46, 54, 55, 59, 60, 61]
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
