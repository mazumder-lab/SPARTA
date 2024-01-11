#!/bin/bash 
#SBATCH -c 20
#SBATCH -t 2-00:0 #Request runtime of 2 days
#SBATCH --gres=gpu:volta:1
#SBATCH -o ../output_logs/output_run_%A_%a.txt #redirect output to output_JOBID.txt
#SBATCH -e ../error_logs/error_run_%A_%a.txt #redirect errors to error_JOBID.txt
#SBATCH --array=0

TASK_ID=$SLURM_ARRAY_TASK_ID
echo $TASK_ID

# Loading the required module
module load anaconda/2023a
source activate pruning

cd ..
#lr 0.8 for all trainable classifier/lora params
python3 -m train_cifar --dataset "cifar10" --model "resnet18" --num_classes 10 --classifier_lr 0.8 --lr 0.8 --lsr 0.1 --batch_size 256 --accum_steps 1 --warm_up 0.05 --num_epochs 200 --optimizer sgd --momentum 0.9 --wd 0.0 --finetune_strategy lora --lora_rank 4 --clip_gradient False --use_gn True --use_dp False --out_file "outfile.txt" --save_file "save_file.pt" --seed 42  --SLURM_JOB_ID $SLURM_JOB_ID --TASK_ID $SLURM_ARRAY_TASK_ID

