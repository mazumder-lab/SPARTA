#!/bin/bash 
#SBATCH -c 20
#SBATCH -t 2-00:0 #Request runtime of 2 days
#SBATCH --gres=gpu:volta:1
#SBATCH -o ./output_logs/output_run_%A_%a.txt
#SBATCH -e ./error_logs/error_run_%A_%a.txt
#SBATCH --array=0-7

TASK_ID=$SLURM_ARRAY_TASK_ID
echo $TASK_ID
# Loading the required module
module purge
module load anaconda/2023a
source activate pruning

python main_trueobs.py resnet18 cifar10 unstr --sparse-dir models_unstr_bootstrap --l_sparsities "0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9" --n_datasets 20 --idx_dataset ${TASK_ID}