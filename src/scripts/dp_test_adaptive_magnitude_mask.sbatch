#!/bin/bash 
#SBATCH -c 20
#SBATCH -t 2-00:0 #Request runtime of 2 days
#SBATCH --gres=gpu:volta:1
#SBATCH -o ../dp_finegrained_to_add/output_logs/output_run_%A_%a.txt
#SBATCH -e ../dp_finegrained_to_add/error_logs/error_run_%A_%a.txt
#SBATCH --array=0-23

EXPERIMENT_DIR="dp_finegrained_to_add"

TASK_ID=$SLURM_ARRAY_TASK_ID
echo $TASK_ID
# Loading the required module
module load anaconda/2023a
source activate pruning

type_masks=("noisy_grad_magnitude" "magnitude")
type_mask=${type_masks[$(($TASK_ID % 2))]}
TASK_ID=$((TASK_ID/2))

epsilons=(1.0)
epsilon=${epsilons[$(($TASK_ID % 1))]}
TASK_ID=$((TASK_ID/1))

clippings=(1.0)
clipping=${clippings[$(($TASK_ID % 1))]}
TASK_ID=$((TASK_ID/1))

# sparsities=(0.01 0.1 0.5) 
# sparsity=${sparsities[$(($TASK_ID % 3))]}
# TASK_ID=$((TASK_ID/3))

sparsities=(0.01 0.1 0.5) 
sparsity=${sparsities[$(($TASK_ID % 3))]}
TASK_ID=$((TASK_ID/3))

epochss=(50 100) 
epochs=${epochss[$(($TASK_ID % 2))]}
TASK_ID=$((TASK_ID/2))

batch_sizes=(500) 
batch_size=${batch_sizes[$(($TASK_ID % 1))]}
TASK_ID=$((TASK_ID/1))

classifier_lrs=(0.8 0.2) 
classifier_lr=${classifier_lrs[$(($TASK_ID % 2))]}
TASK_ID=$((TASK_ID/2))

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

python3 -m train_cifar --dataset "cifar10" --batch_size ${batch_size} --model "resnet18" --num_classes 10 --classifier_lr ${classifier_lr} --lr ${lr} --lsr 0.0 --use_adaptive_lr ${adaptive_bool} --wd 0.0 --momentum 0.9 --lr_schedule_type "warmup_cosine" --clip_gradient False --num_epochs ${epochs} --finetune_strategy "all_layers" --use_gn True --use_magnitude_mask True --use_adaptive_magnitude_mask True --type_mask ${type_mask} --sparsity ${sparsity} --use_dp True --epsilon ${epsilon} --delta 1e-5 --clipping ${clipping} --experiment_dir "sparsity_fine_grained" --out_file "out_file.txt" --seed 0  --SLURM_JOB_ID $SLURM_JOB_ID --TASK_ID $SLURM_ARRAY_TASK_ID
