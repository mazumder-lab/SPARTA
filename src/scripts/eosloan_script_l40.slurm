#!/bin/bash 
#SBATCH --job-name=dp_resnet
#SBATCH --time=1-00:00
#SBATCH --partition=ou_sloan_teaching
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --output=test_job_%j.out
#SBATCH --error=test_job_%j.err
#SBATCH --array=0


echo "1. GPU Information from nvidia-smi:"
nvidia-smi

EXPERIMENT_DIR="L40_test_results_resnet18_deit"

TASK_ID=$SLURM_ARRAY_TASK_ID
echo $TASK_ID

epsilons=(1.0)
epsilon=${epsilons[$(($TASK_ID % 1))]}
TASK_ID=$((TASK_ID/1))

clippings=(1.0)
clipping=${clippings[$(($TASK_ID % 1))]}
TASK_ID=$((TASK_ID/1))

classifier_lrs=(0.1) 
lrs=(0.005)
classifier_lr=${classifier_lrs[$(($TASK_ID % 1))]}
lr=${lrs[$(($TASK_ID % 1))]}
TASK_ID=$((TASK_ID/1))

sparisities=(0.2)
sparsity=${sparisities[$(($TASK_ID % 1))]}
TASK_ID=$((TASK_ID/1))

seeds=(0)
seed=${seeds[$(($TASK_ID % 1))]}
TASK_ID=$((TASK_ID/1))

epochs=50
batch_size=500
use_fixed_w_mask_finding=1
use_last_layer_only_init=1
model="resnet18"
name_dataset="cifar100"
epoch_mask_finding=10
num_classes=100

# Check and create directory if it doesn't exist
cd ..
if [ ! -d "results_folder" ]; then
    mkdir -p "results_folder"
fi
cd results_folder
if [ ! -d "$EXPERIMENT_DIR" ]; then
    mkdir -p "$EXPERIMENT_DIR"
fi
cd ..

python3 train_cifar.py --use_last_layer_only_init ${use_last_layer_only_init} --method_name "row_pruning_noisy_grads" --max_physical_batch_size 10 --epoch_mask_finding ${epoch_mask_finding} --dataset ${name_dataset} --batch_size ${batch_size} --model ${model} --num_classes ${num_classes} --classifier_lr ${classifier_lr} --lr ${lr} --lsr 0.0 --wd 0.0 --momentum 0.9 --lr_schedule_type "onecycle" --num_epochs ${epochs} --warm_up 0.02 --use_gn True --sparsity ${sparsity} --epsilon ${epsilon} --delta 1e-5 --clipping ${clipping} --experiment_dir ${EXPERIMENT_DIR} --seed ${seed} --TASK_ID ${TASK_ID}
