#!/bin/bash

# Loading the required module
source /etc/profile

# Load modules
module load anaconda/2022a 
module load /home/gridsan/groups/datasets/ImageNet/modulefile 

echo $LLSUB_RANK
echo $LLSUB_SIZE

LD_LIBRARY_PATH=/home/gridsan/gafriat/.conda/envs/pruning/lib/python3.10/site-packages/nvidia/cudnn/lib

# Run the script
python /home/gridsan/gafriat/projects/network_pruning/to_run_experiments.py
python /home/gridsan/gafriat/projects/network_pruning/to_run_supercloud_gpu.py $LLSUB_RANK $LLSUB_SIZE