#!/bin/bash
#SBATCH -c 40
#SBATCH -t 2-00:0 #Request runtime of 2 days
#SBATCH --gres=gpu:volta:2
#SBATCH -o ../cvx_fischer_layerwise/output_logs/output_run_%A_%a.txt
#SBATCH -e ../cvx_fischer_layerwise/error_logs/error_run_%A_%a.txt
#SBATCH --array=0

EXPERIMENT_DIR="cvx_fischer_layerwise"

TASK_ID=$SLURM_ARRAY_TASK_ID
echo $TASK_ID
# Loading the required module
source activate pruning


python main.py --lr 0.01 --dense_to_sparse 0 --optimizer_name SGD --type_reset ensemble --type_pruning magnitude --arch resnet18 --name_dataset cifar100 --n_epochs 0 --T_max_cos 200 --n_train_kept -1 --patience 50 --num_workers 0 --l2_reg 0.001 --entropy_reg 0.001 --selection_reg 0.001 --l2_original_reg 0 --seed 50 --gamma 1.0 --test_constraint_weights 0 --test_different_lr 0 --type_decay None --pretrained True --goal_sparsity 0.5 --test_normalized_sgd 1 --mode layer_wise --loss_func layer_wise --pruning_rate_cte 0.1 --test_early_stopping 0 --threshold_weights 0 --lambda_loss 100.0 --test_repeat_if_sparsity_not_reached 1 --loss_last_block layer_wise --retrain_last_block 1 --test_mult_reset 1 --test_reset_to_orignal 0 --test_start_sparse_gpt 0 --test_start_obc 0 --test_start_convex 1 --algo_pruning obc_convex --rel_damp 0.01 --lambda_fisher 1.0 --lambda_reconst 1.0 --prune_bias 0 --type_compute_sparsity prunable --test_adaptive_lr 1 --patience_adaptive_lr 5 --patience_freeze 10 --test_wait_for_pruning 1 --test_almost_sequential 3 --tol_ent_reg 0.01 --tol_sel_reg 0.01 --n_incr_gradual_pruning -1 --goal_sparsity_discrete -1 --folder_saves Saves_test_opt --test_save_all_sparsities 1 --n_parallel 32 --n_convex -1 --pruning_level layer
