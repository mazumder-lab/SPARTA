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

#if [ ! -d "$IMAGENET_PATH" ]; then
#    echo "Modulefile Previously loaded"
#    export IMAGENET_PATH=/run/user/$UID/imagenet
#fi

TASK_ID=$SLURM_ARRAY_TASK_ID
echo $TASK_ID

txtpaths=("mobilenetv1_unstr_50x_dp.txt" "mobilenetv1_unstr_55x_dp.txt" "mobilenetv1_unstr_60x_dp.txt" "mobilenetv1_unstr_65x_dp.txt" "mobilenetv1_unstr_70x_dp.txt" "mobilenetv1_unstr_75x_dp.txt" "mobilenetv1_unstr_80x_dp.txt" "mobilenetv1_unstr_85x_dp.txt" "mobilenetv1_unstr_90x_dp.txt")
txtpath=${txtpaths[$((TASK_ID % 9))]}
TASK_ID=$((TASK_ID/9))
sleep 3

python postproc.py mobilenetv1 imagenet ${txtpath} --database unstr --nsamples 5000 --bnt --bnt-batches 100 --batchsize 128