#!/bin/bash 
#SBATCH -c 20
#SBATCH -t 2-00:0 #Request runtime of 2 days
#SBATCH --gres=gpu:volta:1
#SBATCH -o ../output_logs/output_run_%A_%a.txt #redirect output to output_JOBID.txt
#SBATCH -e ../error_logs/error_run_%A_%a.txt #redirect errors to error_JOBID.txt
#SBATCH --array=0-3

TASK_ID=$SLURM_ARRAY_TASK_ID
echo $TASK_ID

# Loading the required module
module load anaconda/2023a
source activate pruning

TASK_ID=$SLURM_ARRAY_TASK_ID

clip_gradients=(True False)
clip_gradient=${clip_gradients[$(($TASK_ID % 2))]}
TASK_ID=$((TASK_ID/2))

accum_stepss=(1 5)
accum_steps=${accum_stepss[$(($TASK_ID % 2))]}

cd ..
#lr 0.8 for classifier params
python3 -m train_cifar --dataset "cifar10" --model "resnet18" --num_classes 10 --classifier_lr 0.8 --lr 0.1 --lsr 0.1 --batch_size 256 --accum_steps ${accum_steps} --warm_up 0.05 --num_epochs 200 --optimizer sgd --momentum 0.9 --wd 0.0 --finetune_strategy lora --lora_rank 4 --clip_gradient ${clip_gradient} --grad_clip_cst 1.0 --use_gn True --use_dp False --out_file "_outfile.txt" --save_file "_save_file.pt" --seed 42

