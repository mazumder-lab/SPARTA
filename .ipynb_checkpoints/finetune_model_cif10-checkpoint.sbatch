#!/bin/bash 
#SBATCH -c 20
#SBATCH -t 2-00:0 #Request runtime of 2 days
#SBATCH --gres=gpu:volta:1
#SBATCH -o ./output_logs/output_run_%A_%a.txt #redirect output to output_JOBID.txt
#SBATCH -e ./error_logs/error_run_%A_%a.txt #redirect errors to error_JOBID.txt
#SBATCH --array=0-8

TASK_ID=$SLURM_ARRAY_TASK_ID
echo $TASK_ID

# Loading the required module
module load anaconda/2023a
source activate pruning

constants=(0.05 0.1 0.01 0.2 0.3 0.5 1.0 2.0 10.0)
#constants=(0.05 0.1 0.01 0.2 0.3)
#constants=(0.4 0.5 0.8 1.0 2.0)
constant=${constants[$TASK_ID]}

python3 -m train_cifar --dataset "cifar10"  --batch_size 128 --model "resnet18" --num_classes 10 --classifier_lr 0.8 --lr 0.01 --lsr 0.1 --wd 0.0 --momentum 0.9 --clip_gradient True --grad_clip_cst ${constant} --adaptive_lr False --linear_probing True --num_epochs 200 --accum_steps 1 --warm_up 0.005 --use_gn True --out_file "linear_probing_True_adaptive_lr_correct_lr_scheduler_lsr01_model18_10_clipping_gn_cif10.txt" --save_file "linear_probing_True_adaptive_lr_correct_lr_scheduler_lsr01_model18_10_clipping_gn_cif10.pt" --seed 42

#python3 -m train_cifar --dataset "cifar10"  --batch_size 128 --model "resnet18" --num_classes 10 --classifier_lr 0.8 --lr 0.01 --lsr 0.1 --wd 0.0 --momentum 0.9 --clip_gradient False --grad_clip_cst 0.0 --num_epochs 200 --accum_steps 1 --warm_up 0.05 --use_gn True --out_file "warmup05_correct_lr_scheduler_lsr01_model18_10_clipping_gn_cif10.txt" --save_file "warmup05_correct_lr_scheduler_lsr01_model18_10_clipping_gn_cif10.pt" --seed 42

#python3 -m train_cifar --dataset "cifar10"  --batch_size 128 --model "resnet18" --num_classes 10 --lr 0.1 --lsr 0.1 --wd 5e-4 --momentum 0.9 --clip_gradient True --grad_clip_cst ${constant} --num_epochs 50 --accum_steps 1 --warm_up 0.05 --use_gn True --out_file "model18_10_clipping_gn_cif10.txt" --save_file "model18_10_clipping_gn_cif10.pt" --seed 42