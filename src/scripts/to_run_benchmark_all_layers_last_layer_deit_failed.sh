#!/bin/bash

# Loading the required module
source /etc/profile

# module load anaconda/2023a 
module load /home/gridsan/groups/datasets/ImageNet/modulefile 

echo $LLSUB_RANK
echo $LLSUB_SIZE

LD_LIBRARY_PATH=/home/gridsan/gafriat/.conda/envs/pruning/lib/python3.10/site-packages/nvidia/cudnn/lib

# Run the script
python to_run_benchmark_all_layers_last_layer_deit_failed.py $LLSUB_RANK $LLSUB_SIZE