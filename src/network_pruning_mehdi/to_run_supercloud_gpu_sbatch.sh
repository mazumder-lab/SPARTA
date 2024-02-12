#!/bin/bash 
#SBATCH --partition=xeon-g6-volta 
#SBATCH --constraint=xeon-g6
#SBATCH -o ../logs/output_run_%A_%a.txt #redirect output to output_JOBID.txt
#SBATCH -e ../logs/error_run_%A_%a.txt #redirect errors to error_JOBID.txt
##SBATCH --exclusive
#SBATCH --gres=gpu:volta:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --array=0

source /etc/profile

module purge
module load anaconda/2022a
module load /home/gridsan/groups/datasets/ImageNet/modulefile

python /home/gridsan/gafriat/projects/network_pruning/to_run_experiments.py
python /home/gridsan/gafriat/projects/network_pruning/to_run_supercloud_gpu_sbatch.py