#!/bin/bash 
#SBATCH -c 20
#SBATCH -t 2-00:0 #Request runtime of 2 days
#SBATCH --gres=gpu:volta:1
#SBATCH -o ../output_logs/output_run_%A_%a.txt #redirect output to output_JOBID.txt
#SBATCH -e ../error_logs/error_run_%A_%a.txt #redirect errors to error_JOBID.txt
#SBATCH --array=0-0

TASK_ID=$SLURM_ARRAY_TASK_ID
echo $TASK_ID

# Loading the required module
module load anaconda/2023a
source activate pruning

cd ..
python3 -m train_cifar --dataset "cifar10" --batch_size 128 --model "resnet18" --num_classes 10 --classifier_lr 0.8 --lr 0.01 --lsr 0.1 --wd 0.0 --momentum 0.9 --clip_gradient False --num_epochs 200 --accum_steps 1 --warm_up 0.05 --use_gn True --use_dp False --out_file "out_file.txt" --seed 0