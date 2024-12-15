#!/bin/bash 
#SBATCH -t 1-00:00 #Request runtime of 4 days
#SBATCH --partition sched_mit_sloan_gpu_r8
#SBATCH --gres=gpu:volta:1
#SBATCH -o ../results_folder/revised_final_results_tiny_deit/output_logs/output_run_%A_%a.txt
#SBATCH -e ../results_folder/revised_final_results_tiny_deit/error_logs/error_run_%A_%a.txt
#SBATCH --array=0-39

EXPERIMENT_DIR="revised_final_results_base_deit"

TASK_ID=$SLURM_ARRAY_TASK_ID
echo $TASK_ID
# Loading the required module
source activate pruning


epsilons=(1.0)
epsilon=${epsilons[$(($TASK_ID % 1))]}
TASK_ID=$((TASK_ID/1))

clippings=(0.75 0.5)
clipping=${clippings[$(($TASK_ID % 2))]}
TASK_ID=$((TASK_ID/2))

classifier_lrs=(0.1 0.1 0.01 0.005) 
lrs=(0.005 0.001 0.01 0.05)
classifier_lr=${classifier_lrs[$(($TASK_ID % 4))]}
lr=${lrs[$(($TASK_ID % 4))]}
TASK_ID=$((TASK_ID/4))

sparisities=(0.2)
sparsity=${sparisities[$(($TASK_ID % 1))]}
TASK_ID=$((TASK_ID/1))

seeds=(0 42 2024 256 16)
seed=${seeds[$(($TASK_ID % 5))]}
TASK_ID=$((TASK_ID/5))

epochs=50
batch_size=500
use_fixed_w_mask_finding=True
use_last_layer_only_init=True
model="deit_base_patch16_224"
name_dataset="cifar100"
epoch_mask_finding=10
num_classes=10

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

python train_cifar.py --use_last_layer_only_init ${use_last_layer_only_init} --method_name "row_pruning_noisy_grads" --max_physical_batch_size 100 --epoch_mask_finding ${epoch_mask_finding} --dataset ${name_dataset} --batch_size ${batch_size} --model ${model} --num_classes ${num_classes} --classifier_lr ${classifier_lr} --lr ${lr} --lsr 0.0 --wd 0.0 --momentum 0.9 --lr_schedule_type "onecycle" --num_epochs ${epochs} --warm_up 0.02 --use_gn True --sparsity ${sparsity} --epsilon ${epsilon} --delta 1e-5 --clipping ${clipping} --experiment_dir ${EXPERIMENT_DIR} --seed ${seed} --TASK_ID ${TASK_ID}
