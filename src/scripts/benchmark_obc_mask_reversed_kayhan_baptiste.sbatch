#!/bin/bash 
#SBATCH -c 20
#SBATCH -t 2-00:0 #Request runtime of 2 days
#SBATCH --gres=gpu:volta:1
#SBATCH -o ../benchmarking_obc_mask_reversed_kayhan_8_9/output_logs/output_run_%A_%a.txt
#SBATCH -e ../benchmarking_obc_mask_reversed_kayhan_8_9/error_logs/error_run_%A_%a.txt
#SBATCH --array=0-47

EXPERIMENT_DIR="benchmarking_obc_mask_reversed_kayhan_8_9"

TASK_ID=$SLURM_ARRAY_TASK_ID
echo $TASK_ID
# Loading the required module
module purge
module load anaconda/2023a-pytorch
source activate pruning


epsilons=(1.0)
epsilon=${epsilons[$(($TASK_ID % 1))]}
TASK_ID=$((TASK_ID/1))

clippings=(1.0 0.75 0.5)
clipping=${clippings[$(($TASK_ID % 3))]}
TASK_ID=$((TASK_ID/3))

batch_sizes=(500) 
batch_size=${batch_sizes[$(($TASK_ID % 1))]}
TASK_ID=$((TASK_ID/1))

sparsities=(0.8 0.9) 
sparsity=${sparsities[$(($TASK_ID % 2))]}
TASK_ID=$((TASK_ID/2))

classifier_lrs=(0.2 0.4 0.4 0.2) 
lrs=(0.01 0.01 0.005 0.005)
classifier_lr=${classifier_lrs[$(($TASK_ID % 4))]}
lr=${lrs[$(($TASK_ID % 4))]}
TASK_ID=$((TASK_ID/4))

epochss=(50 100) 
epochs=${epochss[$(($TASK_ID % 2))]}
TASK_ID=$((TASK_ID/2))

# sparsities=(0.9 0.5 0.1 0.01) 
# sparsity=${sparsities[$(($TASK_ID % 4))]}
# TASK_ID=$((TASK_ID/4))


# Check and create directory if it doesn't exist
cd ..
if [ ! -d "$EXPERIMENT_DIR" ]; then
    mkdir -p "$EXPERIMENT_DIR"
fi

python3 -m train_cifar --dataset "cifar10" --batch_size ${batch_size} --model "resnet18" --num_classes 10 --classifier_lr ${classifier_lr} --lr ${lr} --lsr 0.0 --wd 0.0 --momentum 0.9 --lr_schedule_type "onecycle" --num_epochs ${epochs} --magnitude_descending False --warm_up 0.01 --finetune_strategy "all_layers"   --use_gn True --use_magnitude_mask True  --mask_available True --mask_reversed True --use_adaptive_magnitude_mask False  --type_mask "" --sparsity ${sparsity} --use_dp True --epsilon ${epsilon} --delta 1e-5 --clipping ${clipping} --experiment_dir ${EXPERIMENT_DIR} --out_file "obc_mask_reversed_out_file.txt"        --seed 0  --SLURM_JOB_ID $SLURM_JOB_ID --TASK_ID $SLURM_ARRAY_TASK_ID
