#!/bin/bash 
#SBATCH -c 20
#SBATCH -t 2-00:0 #Request runtime of 2 days
#SBATCH --gres=gpu:volta:1
#SBATCH -o ../nb2_test_adaptive_mask/output_logs/output_run_%A_%a.txt
#SBATCH -e ../nb2_test_adaptive_mask/error_logs/error_run_%A_%a.txt
#SBATCH --array=0
source activate pruning

cd ..
cd ..

python3 -m test_adaptive_mask
