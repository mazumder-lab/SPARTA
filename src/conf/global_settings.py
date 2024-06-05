import os
from datetime import datetime

EPOCH_MASK_FINDING = 5
BATCH_FINAL = 50000
MAX_PHYSICAL_BATCH_SIZE = 50


# directory to save weights file
CHECKPOINT_PATH = "../checkpoints/lsr=01train_resnet_gn.pt"
CHECKPOINT_WRN_PATH = "../checkpoints/wrn_2810_imagenet32_gn.pt"

IMAGENET32_PATH = "/home/gridsan/mmakni/BoxImageNet32/raw"
