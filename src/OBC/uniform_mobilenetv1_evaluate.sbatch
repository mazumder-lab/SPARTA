#!/bin/bash 
#SBATCH -c 20
#SBATCH -t 0-72:0 #Request runtime of 30 minutes
#SBATCH --gres=gpu:volta:1
#SBATCH -o ./logs/output_run_%A_%a.txt #redirect output to output_JOBID.txt
#SBATCH -e ./logs/error_run_%A_%a.txt #redirect errors to error_JOBID.txt
#SBATCH --array=0-0

source /etc/profile

# Load modules
module purge
module load anaconda/2022a
module load /home/gridsan/groups/datasets/ImageNet/modulefile 

#if [ ! -d "$IMAGENET_PATH" ]; then
#    echo "Modulefile Previously loaded"
#    export IMAGENET_PATH=/run/user/$UID/imagenet
#fi

sleep 3

python evaluate_models.py --model mobilenetv1 --dataset imagenet --nsamples 5000 --l_sparsities "0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9" --batchsize 128