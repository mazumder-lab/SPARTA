#!/bin/bash 
#SBATCH -c 20
#SBATCH -t 2-00:0 #Request runtime of 2 days
#SBATCH --gres=gpu:volta:1
#SBATCH -o ../testing_baselines_adaptive_new/output_logs/output_run_%A_%a.txt
#SBATCH -e ../testing_baselines_adaptive_new/error_logs/error_run_%A_%a.txt
#SBATCH --array=0-77

EXPERIMENT_DIR="testing_baselines_adaptive"

TASK_ID=$SLURM_ARRAY_TASK_ID
echo $TASK_ID
# Loading the required module
module load anaconda/2023a-pytorch
source activate pruning


epsilons=(1.0)
epsilon=${epsilons[$(($TASK_ID % 1))]}
TASK_ID=$((TASK_ID/1))

clippings=(1.0 0.75 0.5)
clipping=${clippings[$(($TASK_ID % 3))]}
TASK_ID=$((TASK_ID/3))
# clippings=(1.0)
# clipping=${clippings[$(($TASK_ID % 1))]}
# TASK_ID=$((TASK_ID/1))

batch_sizes=(500) 
batch_size=${batch_sizes[$(($TASK_ID % 1))]}
TASK_ID=$((TASK_ID/1))

classifier_lrs=(0.2) 
lrs=(0.01)
classifier_lr=${classifier_lrs[$(($TASK_ID % 1))]}
lr=${lrs[$(($TASK_ID % 1))]}
TASK_ID=$((TASK_ID/1))

sparsities=(0.01 0.1 0.2 0.3 0.5 0.7 0.8 0.9)
# sparsities=(0.5 0.6 0.7 0.8 0.9) 
sparsity=${sparsities[$(($TASK_ID % 8))]}
TASK_ID=$((TASK_ID/8))

epochss=(50 100) 
epochs=${epochss[$(($TASK_ID % 2))]}
TASK_ID=$((TASK_ID/2))

# Check and create directory if it doesn't exist
cd ..
if [ ! -d "$EXPERIMENT_DIR" ]; then
    mkdir -p "$EXPERIMENT_DIR"
fi

python3 -m train_cifar --dataset "cifar10" --batch_size ${batch_size} --model "resnet18" --num_classes 10 --classifier_lr ${classifier_lr} --lr ${lr} --lsr 0.0 --wd 0.0 --momentum 0.9 --lr_schedule_type "onecycle" --num_epochs ${epochs} --magnitude_descending False --warm_up 0.01 --finetune_strategy "all_layers"   --use_gn True --use_magnitude_mask True  --use_global_magnitude True  --use_adaptive_magnitude_mask True --type_mask ""                     --sparsity ${sparsity} --use_dp True --epsilon ${epsilon} --delta 1e-5 --clipping ${clipping} --experiment_dir ${EXPERIMENT_DIR} --out_file "largest_adaptive_global_magnitude_out_file.txt"     --seed 0  --SLURM_JOB_ID $SLURM_JOB_ID --TASK_ID $SLURM_ARRAY_TASK_ID
python3 -m train_cifar --dataset "cifar10" --batch_size ${batch_size} --model "resnet18" --num_classes 10 --classifier_lr ${classifier_lr} --lr ${lr} --lsr 0.0 --wd 0.0 --momentum 0.9 --lr_schedule_type "onecycle" --num_epochs ${epochs} --magnitude_descending False --warm_up 0.01 --finetune_strategy "all_layers"   --use_gn True --use_magnitude_mask True  --use_global_magnitude False --use_adaptive_magnitude_mask True --type_mask ""                     --sparsity ${sparsity} --use_dp True --epsilon ${epsilon} --delta 1e-5 --clipping ${clipping} --experiment_dir ${EXPERIMENT_DIR} --out_file "largest_adaptive_layerwise_magnitude_out_file.txt"  --seed 0  --SLURM_JOB_ID $SLURM_JOB_ID --TASK_ID $SLURM_ARRAY_TASK_ID
python3 -m train_cifar --dataset "cifar10" --batch_size ${batch_size} --model "resnet18" --num_classes 10 --classifier_lr ${classifier_lr} --lr ${lr} --lsr 0.0 --wd 0.0 --momentum 0.9 --lr_schedule_type "onecycle" --num_epochs ${epochs} --magnitude_descending True  --warm_up 0.01 --finetune_strategy "all_layers"   --use_gn True --use_magnitude_mask True  --use_global_magnitude True  --use_adaptive_magnitude_mask True --type_mask ""                     --sparsity ${sparsity} --use_dp True --epsilon ${epsilon} --delta 1e-5 --clipping ${clipping} --experiment_dir ${EXPERIMENT_DIR} --out_file "smallest_adaptive_global_magnitude_out_file.txt"    --seed 0  --SLURM_JOB_ID $SLURM_JOB_ID --TASK_ID $SLURM_ARRAY_TASK_ID
python3 -m train_cifar --dataset "cifar10" --batch_size ${batch_size} --model "resnet18" --num_classes 10 --classifier_lr ${classifier_lr} --lr ${lr} --lsr 0.0 --wd 0.0 --momentum 0.9 --lr_schedule_type "onecycle" --num_epochs ${epochs} --magnitude_descending True  --warm_up 0.01 --finetune_strategy "all_layers"   --use_gn True --use_magnitude_mask True  --use_global_magnitude False --use_adaptive_magnitude_mask True --type_mask ""                     --sparsity ${sparsity} --use_dp True --epsilon ${epsilon} --delta 1e-5 --clipping ${clipping} --experiment_dir ${EXPERIMENT_DIR} --out_file "smallest_adaptive_layerwise_magnitude_out_file.txt" --seed 0  --SLURM_JOB_ID $SLURM_JOB_ID --TASK_ID $SLURM_ARRAY_TASK_ID
