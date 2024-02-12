#!/bin/bash

# Loading the required module
source /etc/profile

# Load modules
module load anaconda/2022a
module load /home/gridsan/groups/datasets/ImageNet/modulefile 

echo $LLSUB_RANK
echo $LLSUB_SIZE

# Run the script
python /home/gridsan/gafriat/projects/network_pruning/to_run_experiments.py
python /home/gridsan/gafriat/projects/network_pruning/to_run_supercloud_cpu.py $LLSUB_RANK $LLSUB_SIZE