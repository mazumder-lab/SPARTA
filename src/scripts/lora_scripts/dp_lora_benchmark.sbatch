#!/bin/bash 
#SBATCH -c 20
#SBATCH -t 2-00:0 #Request runtime of 2 days
#SBATCH --gres=gpu:volta:1
#SBATCH -o ../dp_lora_benchmarking/output_logs/output_run_%A_%a.txt #redirect output to output_JOBID.txt
#SBATCH -e ../dp_lora_benchmarking/error_logs/error_run_%A_%a.txt #redirect errors to error_JOBID.txt
#SBATCH --array=0-3

EXPERIMENT_DIR="dp_lora_benchmarking"

TASK_ID=$SLURM_ARRAY_TASK_ID
echo $TASK_ID

# Loading the required module
module purge
module load anaconda/2023a-pytorch
source activate pruning

TASK_ID=$SLURM_ARRAY_TASK_ID


lora_ranks=(4 3)
lora_rank=${lora_ranks[$(($TASK_ID % 2))]}
TASK_ID=$((TASK_ID/2))


lora_lrs=(0.2 0.05)
lora_lr=${lora_lrs[$(($TASK_ID % 2))]}
TASK_ID=$((TASK_ID/2))


batch_sizes=(500)
batch_size=${batch_sizes[$(($TASK_ID % 1))]}

cd ..
if [ ! -d "$EXPERIMENT_DIR" ]; then
    mkdir -p "$EXPERIMENT_DIR"
fi


python3 -m train_cifar --dataset "cifar10" --model "resnet18" --num_classes 10 --classifier_lr 0.2 --lr ${lora_lr} --lsr 0.0 --batch_size ${batch_size} --warm_up 0.01 --num_epochs 50 --optimizer sgd --momentum 0.9 --wd 0.0 --finetune_strategy lora --lora_rank ${lora_rank} --use_gn True --use_dp True --epsilon 1 --delta 1e-5 --clipping 1.0 --out_file "outfile_dplora.txt" --seed 0 --SLURM_JOB_ID $SLURM_JOB_ID --TASK_ID $SLURM_ARRAY_TASK_ID --experiment_dir ${EXPERIMENT_DIR}

