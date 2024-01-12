#!/bin/bash 
#SBATCH -c 20
#SBATCH -t 2-00:0 #Request runtime of 2 days
#SBATCH --gres=gpu:volta:1
#SBATCH -o ../output_logs/output_run_%A_%a.txt #redirect output to output_JOBID.txt
#SBATCH -e ../error_logs/error_run_%A_%a.txt #redirect errors to error_JOBID.txt
#SBATCH --array=0-29

TASK_ID=$SLURM_ARRAY_TASK_ID
echo $TASK_ID

# Loading the required module
module load anaconda/2023a
source activate pruning

TASK_ID=$SLURM_ARRAY_TASK_ID

batch_sizes=(250 100) 
batch_size=${batch_sizes[$(($TASK_ID % 2))]}
TASK_ID=$((TASK_ID/2))

epsilons=(1.0 8.0)
epsilon=${epsilons[$(($TASK_ID % 2))]}
TASK_ID=$((TASK_ID/2))

#clippings=(1.0 0.9 0.8 0.7 0.5 0.25)
clippings=(1.0 0.9 0.8 0.7)
#clippings=(0.01 0.03 0.05 0.1 0.2 0.5 1.0)
clipping=${clippings[$TASK_ID]}



cd ..
python3 -m train_cifar --dataset "cifar10" --batch_size ${batch_size} --model "resnet18" --num_classes 10 --classifier_lr 0.2 --lr 0.003 --lsr 0.0 --wd 0.0 --momentum 0.9 --clip_gradient False --num_epochs 50 --accum_steps 1 --warm_up 0.01 --finetune_strategy "conf_indices" --use_gn True --use_dp True --epsilon ${epsilon} --delta 1e-5 --clipping ${clipping} --out_file "subset_out_file.txt" --seed 0  --SLURM_JOB_ID $SLURM_JOB_ID --TASK_ID $SLURM_ARRAY_TASK_ID
python3 -m train_cifar --dataset "cifar10" --batch_size ${batch_size} --model "resnet18" --num_classes 10 --classifier_lr 0.2 --lr 0.003 --lsr 0.0 --wd 0.0 --momentum 0.9 --clip_gradient False --num_epochs 50 --accum_steps 1 --warm_up 0.01 --finetune_strategy "all_layers" --use_gn True --use_dp True --epsilon ${epsilon} --delta 1e-5 --clipping ${clipping} --out_file "all_layers_out_file.txt" --seed 0  --SLURM_JOB_ID $SLURM_JOB_ID --TASK_ID $SLURM_ARRAY_TASK_ID
