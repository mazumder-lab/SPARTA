#!/bin/bash 
#SBATCH -c 20
#SBATCH -t 2-00:0 #Request runtime of 2 days
#SBATCH --gres=gpu:volta:1
#SBATCH -o ../test_generate_obc_mask_public/output_logs/output_run_%A_%a.txt
#SBATCH -e ../test_generate_obc_mask_public/error_logs/error_run_%A_%a.txt
#SBATCH --array=0-7

TASK_ID=$SLURM_ARRAY_TASK_ID
echo $TASK_ID
# Loading the required module
module purge
module load anaconda/2023a-pytorch
source activate pruning

sparsities=(0.01 0.1 0.2 0.3 0.5 0.7 0.8 0.9) 
sparsity=${sparsities[$(($TASK_ID % 8))]}
TASK_ID=$((TASK_ID/8))

cd ..
cd ..

python3 -m generate_obc_mask_public_data --sparsity ${sparsity}
