#!/bin/bash

# Loading the required module
source /etc/profile

# Load modules
module load anaconda/2022a
module load /home/gridsan/groups/datasets/ImageNet/modulefile 

echo $LLSUB_RANK
echo $LLSUB_SIZE

echo "Path:"
echo $IMAGENET_PATH

# Run the script
#python /home/gridsan/gafriat/projects/network_pruning/to_run_experiments.py
#python /home/gridsan/gafriat/projects/network_pruning/to_run_supercloud_cpu.py $LLSUB_RANK $LLSUB_SIZE

#python main.py --lr 0.005 --dense_to_sparse 0 --folder_saves Saves_test_TEMP --goal_sparsity 0.5 --optimizer_name SGD --type_reset ensemble --type_pruning magnitude --arc mobilenetv1 --name_dataset imagenet --n_epochs 10 --n_train_kept 10000 --num_workers 16 --mode layer_wise --loss_func layer_wise