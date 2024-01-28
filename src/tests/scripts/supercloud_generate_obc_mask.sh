#!/bin/bash 
#SBATCH -c 20
#SBATCH -t 2-00:0 #Request runtime of 2 days
#SBATCH --gres=gpu:volta:1
#SBATCH -o ../test_generate_obc_mask/output_logs/output_run_%A_%a.txt
#SBATCH -e ../test_generate_obc_mask/error_logs/error_run_%A_%a.txt
#SBATCH --array=0

module load anaconda/2023a
source activate pruning

cd ..
cd ..

python3 -m generate_ob_mask
