#!/bin/bash 
#SBATCH -c 20
#SBATCH -t 2-00:0 #Request runtime of 2 days
#SBATCH --gres=gpu:volta:1
#SBATCH --mem=170G
#SBATCH -o ../test_fisher_obc_mask_kayhan/output_logs/output_run_%A_%a.txt
#SBATCH -e ../test_fisher_obc_mask_kayhan/error_logs/error_run_%A_%a.txt
#SBATCH --array=0-15

TASK_ID=$SLURM_ARRAY_TASK_ID
echo $TASK_ID

module load anaconda/2023a-pytorch
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

classifier_lrs=(0.2) 
lrs=(0.01)
classifier_lr=${classifier_lrs[$(($TASK_ID % 1))]}
lr=${lrs[$(($TASK_ID % 1))]}
TASK_ID=$((TASK_ID/1))

sparsities=(0.2 0.8) 
sparsity=${sparsities[$(($TASK_ID % 2))]}
TASK_ID=$((TASK_ID/2))

epochss=(50) 
epochs=${epochss[$(($TASK_ID % 1))]}
TASK_ID=$((TASK_ID/1))

use_w_tildes=(True True True False)
correction_coefficients=(0.1 1.0 0.01 0.0)
use_w_tilde=${use_w_tildes[$(($TASK_ID % 4))]}
TASK_ID=$((TASK_ID/4))

use_fisher_mask_with_true_gradss=(True False)
use_fisher_mask_with_true_grads=${use_fisher_mask_with_true_gradss[$(($TASK_ID % 2))]}
TASK_ID=$((TASK_ID/2))

cd ..
cd ..

python3 -m test_per_sample_opacus.py --sparsity ${sparsity} --use_w_tilde ${use_w_tilde} --use_fisher_mask_with_true_grads ${use_fisher_mask_with_true_grads} --correction_coefficient ${correction_coefficient}
