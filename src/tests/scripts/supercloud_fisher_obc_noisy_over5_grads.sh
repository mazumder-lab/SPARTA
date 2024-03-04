#!/bin/bash 
#SBATCH -c 40
#SBATCH -t 2-00:0 #Request runtime of 2 days
#SBATCH --gres=gpu:volta:1
#SBATCH --mem=340G
#SBATCH -o ../fisher_mask_over5_grads/output_logs/output_run_%A_%a.txt
#SBATCH -e ../fisher_mask_over5_grads/error_logs/error_run_%A_%a.txt
#SBATCH --array=0-8

TASK_ID=$SLURM_ARRAY_TASK_ID
echo $TASK_ID

module purge
module load anaconda/2023a-pytorch
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pruning

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

epochss=(50) 
epochs=${epochss[$(($TASK_ID % 1))]}
TASK_ID=$((TASK_ID/1))

use_fisher_mask_with_true_gradss=(False)
use_fisher_mask_with_true_grads=${use_fisher_mask_with_true_gradss[$(($TASK_ID % 1))]}
TASK_ID=$((TASK_ID/1))

use_clipped_true_gradss=(False)
use_clipped_true_grads=${use_clipped_true_gradss[$(($TASK_ID % 1))]}
TASK_ID=$((TASK_ID/1))

sparsities=(0.2 0.5 0.8 0.9 0.1 0.3 0.4 0.7 0.6) 
sparsity=${sparsities[$(($TASK_ID % 2))]}
TASK_ID=$((TASK_ID/2))

use_w_tildes=(True)
correction_coefficients=(0.01)
use_w_tilde=${use_w_tildes[$(($TASK_ID % 1))]}
correction_coefficient=${correction_coefficients[$(($TASK_ID % 1))]}
TASK_ID=$((TASK_ID/1))


cd ..
cd ..

python3 -m test_per_sample_opacus.py --sparsity ${sparsity} --use_w_tilde ${use_w_tilde} --use_fisher_mask_with_true_grads ${use_fisher_mask_with_true_grads} --use_clipped_true_grads ${use_clipped_true_grads} --correction_coefficient ${correction_coefficient} --num_epochs ${epochs} --epsilon ${epsilon} --clipping ${clipping} --batch_size ${batch_size} 
