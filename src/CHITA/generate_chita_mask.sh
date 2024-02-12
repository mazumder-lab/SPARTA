#!/bin/bash
#SBATCH -c 20
#SBATCH -t 2-00:0 #Request runtime of 2 days
#SBATCH --gres=gpu:volta:1
#SBATCH -o ./logs/output_run_%A_%a.txt #redirect output to output_JOBID.txt
#SBATCH -e ./logs/error_run_%A_%a.txt #redirect errors to error_JOBID.txt
#SBATCH --array=0-7

module load anaconda/2023a
source activate pruning

TASK_ID=$SLURM_ARRAY_TASK_ID
echo $TASK_ID

alsgo="Heuristic_CD"
num_stages=50
sparsity_schedule="poly"
fisher_subsample_size=500

l2s=(0.01 1)
l2=${l2s[$((TASK_ID % 2))]}
TASK_ID=$((TASK_ID/2))

sparsities=(0.2 0.5 0.8 0.9)
base_levels=(0.05 0.1 0.3 0.4)
sparsity=${sparsities[$((TASK_ID % 4))]}
base_level=${base_levels[$((TASK_ID % 4))]}
TASK_ID=$((TASK_ID/4))

fisher_mini_bsz=32

python3 -u run_experiment.py --arch resnet18 --dset cifar100 --num_workers 40 --exp_name full_comp_aug_2_test --exp_id ${SLURM_JOB_ID} --compute_training_losses False --restrict_support True --shuffle_train True --use_wbar True --use_activeset True --test_batch_size 400 --fisher_subsample_size ${fisher_subsample_size} --fisher_mini_bsz ${fisher_mini_bsz} --fisher_data_bsz ${fisher_mini_bsz} --num_iterations 10 --num_stages ${num_stages} --seed 1 --first_order_term True --compute_trace_H False --recompute_X True --sparsity ${sparsity} --base_level ${base_level}  --l2 ${l2} --sparsity_schedule ${sparsity_schedule} --algo ${algo} --normalize False





