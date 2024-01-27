#!/bin/bash 
#SBATCH -c 20
#SBATCH -t 0-72:0 #Request runtime of 30 minutes
#SBATCH --gres=gpu:volta:1
#SBATCH -o ./logs/output_run_%A_%a.txt #redirect output to output_JOBID.txt
#SBATCH -e ./logs/error_run_%A_%a.txt #redirect errors to error_JOBID.txt
#SBATCH --array=0-8


module purge
module load anaconda/2022a
module load /home/gridsan/groups/datasets/ImageNet/modulefile

TASK_ID=$SLURM_ARRAY_TASK_ID
echo $TASK_ID

#if [ ! -d "$IMAGENET_PATH" ]; then
#    echo "Modulefile Previously loaded"
#    export IMAGENET_PATH=/run/user/$UID/imagenet
#fi

targets=(0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9)
target=${targets[$(($TASK_ID % 9))]}
TASK_ID=$((TASK_ID/9))


sleep 3
echo $target

python spdy.py resnet20 cifar10 ${target} unstr --dp --constr sparsity --nsamples -1 --batchsize 128
