#!/bin/bash 
#SBATCH -c 20
#SBATCH -t 2-00:0 #Request runtime of 2 days
#SBATCH --gres=gpu:volta:2
#SBATCH --mem=340G
#SBATCH --exclusive
#SBATCH -o ../results_folder/wrn2810/experiments_wrn2810_cif100_row_pruning_noisy_grads_update/output_logs/output_run_%A_%a.txt
#SBATCH -e ../results_folder/wrn2810/experiments_wrn2810_cif100_row_pruning_noisy_grads_update/error_logs/error_run_%A_%a.txt
#SBATCH --array=0-39

EXPERIMENT_DIR="wrn2810/experiments_wrn2810_cif100_row_pruning_noisy_grads_update"

TASK_ID=$SLURM_ARRAY_TASK_ID
echo $TASK_ID
# Loading the required module
module load anaconda/2023a
source activate pruning


epsilons=(1.0)
epsilon=${epsilons[$(($TASK_ID % 1))]}
TASK_ID=$((TASK_ID/1))

sparisities=(0.1 0.2)
sparsity=${sparisities[$(($TASK_ID % 2))]}
TASK_ID=$((TASK_ID/2))

clippings=(1.0 0.75 0.5 0.25 0.1)
clipping=${clippings[$(($TASK_ID % 5))]}
TASK_ID=$((TASK_ID/5))

classifier_lrs=(0.05 0.05 0.025 0.025) 
lrs=(0.005 0.002 0.0025 0.001)
classifier_lr=${classifier_lrs[$(($TASK_ID % 4))]}
lr=${lrs[$(($TASK_ID % 4))]}
TASK_ID=$((TASK_ID/4))

batch_sizes=(500) 
batch_size=${batch_sizes[$(($TASK_ID % 1))]}
TASK_ID=$((TASK_ID/1))

epochss=(50) 
epochs=${epochss[$(($TASK_ID % 1))]}
TASK_ID=$((TASK_ID/1))

use_delta_weight_optims=(True)
use_delta_weight_optim=${use_delta_weight_optims[$(($TASK_ID % 1))]}
TASK_ID=$((TASK_ID/1))

use_fixed_w_mask_findings=(True)
use_fixed_w_mask_finding=${use_fixed_w_mask_findings[$(($TASK_ID % 1))]}
TASK_ID=$((TASK_ID/1))

# Check and create directory if it doesn't exist
cd ..
cd results_folder
if [ ! -d "$EXPERIMENT_DIR" ]; then
    mkdir -p "$EXPERIMENT_DIR"
fi
cd ..

python3 -m train_cifar --use_delta_weight_optim ${use_delta_weight_optim} --dataset "cifar100" --batch_size ${batch_size} --model "wrn2810" --num_classes 100 --classifier_lr ${classifier_lr} --lr ${lr} --lsr 0.0 --wd 0.0 --momentum 0.9 --lr_schedule_type "onecycle" --num_epochs ${epochs} --magnitude_descending False --warm_up 0.01 --finetune_strategy "all_layers"   --use_gn True --method_name "row_pruning_noisy_grads"   --mask_type "optimization" --sparsity ${sparsity} --epsilon ${epsilon} --delta 1e-5 --clipping ${clipping} --experiment_dir ${EXPERIMENT_DIR} --out_file "row_pruning_noisy_grads.txt"  --seed 0  --SLURM_JOB_ID $SLURM_JOB_ID --TASK_ID $SLURM_ARRAY_TASK_ID