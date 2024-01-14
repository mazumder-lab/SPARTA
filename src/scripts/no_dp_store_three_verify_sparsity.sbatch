#!/bin/bash 
#SBATCH -c 20
#SBATCH -t 2-00:0 #Request runtime of 2 days
#SBATCH --gres=gpu:volta:1
#SBATCH -o ../no_dp_store_three_sparsity/output_logs/output_run_%A_%a.txt
#SBATCH -e ../no_dp_store_three_sparsity/error_logs/error_run_%A_%a.txt
#SBATCH --array=0-1

EXPERIMENT_DIR="no_dp_store_three_sparsity"

TASK_ID=$SLURM_ARRAY_TASK_ID
echo $TASK_ID
# Loading the required module
module load anaconda/2023a
source activate pruning

epsilons=(1.0 8.0)
epsilon=${epsilons[$(($TASK_ID % 2))]}
TASK_ID=$((TASK_ID/2))

clippings=(1.0)
clipping=${clippings[$(($TASK_ID % 1))]}
TASK_ID=$((TASK_ID/1))

batch_sizes=(250) 
batch_size=${batch_sizes[$(($TASK_ID % 1))]}
TASK_ID=$((TASK_ID/1))

classifier_lrs=(0.2) 
classifier_lr=${classifier_lrs[$(($TASK_ID % 1))]}
TASK_ID=$((TASK_ID/1))

lrs=(0.01) 
lr=${lrs[$(($TASK_ID % 1))]}
TASK_ID=$((TASK_ID/1))

adaptive_bools=(False) 
adaptive_bool=${adaptive_bools[$(($TASK_ID % 1))]}
TASK_ID=$((TASK_ID/1))

# Check and create directory if it doesn't exist
cd ..
if [ ! -d "$EXPERIMENT_DIR" ]; then
    mkdir -p "$EXPERIMENT_DIR"
fi
python3 -m train_cifar --dataset "cifar10" --batch_size ${batch_size} --model "resnet18" --num_classes 10 --classifier_lr ${classifier_lr} --lr ${lr} --lsr 0.0 --use_adaptive_lr ${adaptive_bool} --wd 0.0 --momentum 0.9 --clip_gradient False --num_epochs 1 --accum_steps 1 --warm_up 0.01 --finetune_strategy "all_layers" --use_gn True --use_magnitude_mask True --sparsity 0.0 --use_dp False --experiment_dir ${EXPERIMENT_DIR} --out_file "out_file.txt" --seed 0  --SLURM_JOB_ID $SLURM_JOB_ID --TASK_ID $SLURM_ARRAY_TASK_ID
