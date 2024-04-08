#!/bin/bash 
#SBATCH -c 20
#SBATCH -t 2-00:0 #Request runtime of 2 days
#SBATCH --gres=gpu:volta:1
#SBATCH --mem=170G
#SBATCH -o ../results_folder/wrn2810/experiment_optim_mp_w_noisy_grads_extra_noise/output_logs/output_run_%A_%a.txt
#SBATCH -e ../results_folder/wrn2810/experiment_optim_mp_w_noisy_grads_extra_noise/error_logs/error_run_%A_%a.txt
#SBATCH --array=0-3

EXPERIMENT_DIR="wrn2810/experiment_optim_mp_w_noisy_grads_extra_noise"

TASK_ID=$SLURM_ARRAY_TASK_ID
echo $TASK_ID
# Loading the required module
module load anaconda/2023a
source activate pruning


epsilons=(1.0)
epsilon=${epsilons[$(($TASK_ID % 1))]}
TASK_ID=$((TASK_ID/1))

clippings=(1.0)
clipping=${clippings[$(($TASK_ID % 1))]}
TASK_ID=$((TASK_ID/1))

batch_sizes=(500) 
batch_size=${batch_sizes[$(($TASK_ID % 1))]}
TASK_ID=$((TASK_ID/1))

epochss=(50) 
epochs=${epochss[$(($TASK_ID % 1))]}
TASK_ID=$((TASK_ID/1))

classifier_lrs=(0.2) 
lrs=(0.01)
classifier_lr=${classifier_lrs[$(($TASK_ID % 1))]}
lr=${lrs[$(($TASK_ID % 1))]}
TASK_ID=$((TASK_ID/1))

# sparisities=(0.2 0.8 0.1 0.5 0.9 0.3 0.6 0.4)
# sparsity=${sparisities[$(($TASK_ID % 8))]}
# TASK_ID=$((TASK_ID/8))

sparisities=(0.2 0.8)
sparsity=${sparisities[$(($TASK_ID % 2))]}
TASK_ID=$((TASK_ID/2))

use_delta_weight_optims=(False True)
use_delta_weight_optim=${use_delta_weight_optims[$(($TASK_ID % 2))]}
TASK_ID=$((TASK_ID/2))

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

python3 -m train_cifar --use_delta_weight_optim ${use_delta_weight_optim} --use_w_tilde True --correction_coefficient 0.01 --dataset "cifar10" --batch_size ${batch_size} --model "wrn2810" --num_classes 10 --classifier_lr ${classifier_lr} --lr ${lr} --lsr 0.0 --wd 0.0 --momentum 0.9 --lr_schedule_type "onecycle" --num_epochs ${epochs} --magnitude_descending False --warm_up 0.01 --finetune_strategy "all_layers"   --use_gn True --method_name "optim_mp_w_noisy_grads_extra_noise"   --mask_type "optimization" --sparsity ${sparsity} --epsilon ${epsilon} --delta 1e-5 --clipping ${clipping} --experiment_dir ${EXPERIMENT_DIR} --out_file "optim_mp_w_noisy_grads_extra_noise.txt"  --seed 0  --SLURM_JOB_ID $SLURM_JOB_ID --TASK_ID $SLURM_ARRAY_TASK_ID
