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

#constants=(0.05 0.1 0.01 0.2 0.3 0.5 1.0 2.0 10.0)
# constant=${constants[$TASK_ID]}

cd ..
#lr 0.2 for newly introduced lora params
python3 -m train_cifar --dataset "cifar10"  --batch_size 128 --model "resnet18" --num_classes 10 --classifier_lr 0.8 --lr 0.2 --lsr 0.1 --warm_up 0.05 --num_epochs 200 --optimizer sgd --momentum 0.9 --wd 0.0 --clip_gradient False --grad_clip_cst 0.0 --finetune_strategy lora --accum_steps 1 --use_gn True --out_file "lora_lsr01_model18_clipping_gn_cif10.txt" --save_file "lora_lsr01_model18_clipping_gn_cif10.pt" --seed 42

