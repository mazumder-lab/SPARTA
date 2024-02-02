#!/bin/bash 
#SBATCH -c 20
#SBATCH -t 2-00:0 #Request runtime of 2 days
#SBATCH --gres=gpu:volta:1
#SBATCH -o ../test_obc_magnitude_masks_nodp/output_logs/output_run_%A_%a.txt
#SBATCH -e ../test_obc_magnitude_masks_nodp/error_logs/error_run_%A_%a.txt
#SBATCH --array=0-15

EXPERIMENT_DIR="benchmarking_new_obc_mask"

TASK_ID=$SLURM_ARRAY_TASK_ID
echo $TASK_ID
# Loading the required module
module load anaconda/2023a
source activate pruning


mask_availables=(False True)
mask_available=${mask_availables[$(($TASK_ID % 2))]}
TASK_ID=$((TASK_ID/2))

batch_sizes=(128) 
batch_size=${batch_sizes[$(($TASK_ID % 1))]}
TASK_ID=$((TASK_ID/1))

classifier_lrs=(0.2 0.4) 
lrs=(0.01 0.01)
classifier_lr=${classifier_lrs[$(($TASK_ID % 2))]}
lr=${lrs[$(($TASK_ID % 2))]}
TASK_ID=$((TASK_ID/2))

sparsities=(0.5 0.7) 
sparsity=${sparsities[$(($TASK_ID % 2))]}
TASK_ID=$((TASK_ID/2))

epochss=(50 100) 
epochs=${epochss[$(($TASK_ID % 2))]}
TASK_ID=$((TASK_ID/2))

# Check and create directory if it doesn't exist
cd ..
if [ ! -d "$EXPERIMENT_DIR" ]; then
    mkdir -p "$EXPERIMENT_DIR"
fi

python3 -m train_cifar --dataset "cifar10" --batch_size ${batch_size} --model "resnet18" --num_classes 10 --classifier_lr ${classifier_lr} --lr ${lr} --lsr 0.1 --wd 1e-5 --momentum 0.9 --lr_schedule_type "onecycle" --num_epochs ${epochs} --magnitude_descending False --warm_up 0.01 --finetune_strategy "all_layers"   --use_gn True --use_magnitude_mask True  --mask_available ${mask_available} --use_adaptive_magnitude_mask False  --type_mask "" --sparsity ${sparsity} --use_dp False --experiment_dir ${EXPERIMENT_DIR} --out_file "test_nodp_mask.txt"        --seed 0  --SLURM_JOB_ID $SLURM_JOB_ID --TASK_ID $SLURM_ARRAY_TASK_ID
