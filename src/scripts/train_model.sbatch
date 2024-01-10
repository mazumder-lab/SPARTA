#!/bin/bash 
#SBATCH -c 20
#SBATCH -t 2-00:0 #Request runtime of 2 days
#SBATCH --gres=gpu:volta:1
#SBATCH -o ./output_logs/output_run_%A_%a.txt #redirect output to output_JOBID.txt
#SBATCH -e ./error_logs/error_run_%A_%a.txt #redirect errors to error_JOBID.txt
#SBATCH --array=0-7

TASK_ID=$SLURM_ARRAY_TASK_ID
echo $TASK_ID

# Loading the required module
module load anaconda/2023a
source activate pruning


#constants=(1.0 1.2 0.8 2.0 5.0 0.5 1.5 0.1)
constants=(0.7 0.6 0.5 0.4 0.75 0.65 0.85 0.9)
constant=${constants[$TASK_ID]}

python3 -m train_cifar --dataset "cifar100"  --batch_size 128 --model "resnet18" --num_classes 100 --lr 0.1 --lsr 0.1 --wd 5e-4 --momentum 0.9 --clip_gradient True --grad_clip_cst ${constant} --num_epochs 200 --accum_steps 1 --warm_up 0.05 --use_gn True --out_file "model18_100_clipping_gn_new.txt" --save_file "model18_100_clipping_gn_new.pt" --seed 42