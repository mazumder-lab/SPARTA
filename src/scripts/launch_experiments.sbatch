#!/bin/bash
#SBATCH --job-name=sparta_experiment
#SBATCH --time=1-00:00
#SBATCH --partition=mit_preemptable
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
# SBATCH --gres=gpu:1
#SBATCH --output=sparta_experiment-row-prune-%j.out
#SBATCH --error=sparta_experiment-row-prune-%j.err
#SBATCH --array=0-5

EXPERIMENT_DIR="sparta_experiment"
RESULTS_DIR="results_folder/$EXPERIMENT_DIR"

cd ..
mkdir -p "$RESULTS_DIR"

# methods=("all_layers" "lp_gn" "dp_bitfit" "mp_weights" "row_pruning_noisy_grads" "lora")
# methods=("row_pruning_noisy_grads" "lora" "dp_bitfit" "mp_weights")
methods=("row_pruning_noisy_grads")
seeds=(0)
epsilons=(4.0)
datasets=("cifar100" "cifar10")
models=("wideresnet2810" "deit_small_patch16_224" "resnet18")

use_fixed_w_mask_finding=1
clipping=1.0
batch_size=500
lr=0.01
classifier_lr=0.01

# Compute indices
num_methods=${#methods[@]}
num_seeds=${#seeds[@]}
num_epsilons=${#epsilons[@]}
num_datasets=${#datasets[@]}
num_models=${#models[@]}

method_idx=$((SLURM_ARRAY_TASK_ID % num_methods))
seed_idx=$(((SLURM_ARRAY_TASK_ID / num_methods) % num_seeds))
epsilon_idx=$(((SLURM_ARRAY_TASK_ID / (num_methods * num_seeds)) % num_epsilons))
dataset_idx=$(((SLURM_ARRAY_TASK_ID / (num_methods * num_seeds * num_epsilons)) % num_datasets))
model_idx=$(((SLURM_ARRAY_TASK_ID / (num_methods * num_seeds * num_epsilons * num_datasets)) % num_models))

method=${methods[$method_idx]}
seed=${seeds[$seed_idx]}
epsilon=${epsilons[$epsilon_idx]}
dataset=${datasets[$dataset_idx]}
model=${models[$model_idx]}

# Set num_classes based on dataset
if [ "$dataset" == "cifar10" ]; then
  num_classes=10
elif [ "$dataset" == "cifar100" ]; then
  num_classes=100
else
  echo "Unknown dataset: $dataset"
  exit 1
fi

base_args="--max_physical_batch_size 100 --dataset $dataset --batch_size $batch_size --model $model --num_classes $num_classes --classifier_lr $classifier_lr --lr $lr --lsr 0.0 --wd 0.0 --momentum 0.9 --lr_schedule_type onecycle --num_epochs 50 --warm_up 0.02 --epsilon $epsilon --delta 1e-5 --clipping $clipping --experiment_dir $EXPERIMENT_DIR"

if [ "$method" == "all_layers" ] || [ "$method" == "lp_gn" ] || [ "$method" == "dp_bitfit" ]; then
  python train_cifar.py --method_name "$method" $base_args --epoch_mask_finding -1 --seed $seed --TASK_ID $SLURM_ARRAY_TASK_ID
elif [ "$method" == "mp_weights" ]; then
  python train_cifar.py --method_name "$method" $base_args --epoch_mask_finding -1 --sparsity 0.2 --seed $seed --TASK_ID $SLURM_ARRAY_TASK_ID
elif [ "$method" == "row_pruning_noisy_grads" ]; then
  python train_cifar.py --method_name "$method" $base_args --epoch_mask_finding 10 --sparsity 0.2 --use_last_layer_only_init true --seed $seed --TASK_ID $SLURM_ARRAY_TASK_ID
fi
