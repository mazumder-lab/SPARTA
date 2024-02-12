#!/bin/bash

# Loading the required module
source /etc/profile

# Load modules
module load anaconda/2022a 
module load /home/gridsan/groups/datasets/ImageNet/modulefile 

echo $LLSUB_RANK
echo $LLSUB_SIZE

# Run the script
python main.py --lr 0.005 --dense_to_sparse 0 --folder_saves Saves_layer_wise_test --goal_sparsity 0.9 --optimizer_name SGD --type_reset ensemble --type_pruning magnitude --arc mobilenetv1 --name_dataset imagenet --mode layer_wise --loss_func layer_wise --n_epochs 100 --T_max_cos 100 --l2_reg 1e-4