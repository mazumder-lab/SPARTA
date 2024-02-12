#!/bin/bash 
#SBATCH --partition=xeon-g6-volta 
#SBATCH --constraint=xeon-g6
#SBATCH -t 0-168:0 #Request runtime of 30 minutes
#SBATCH -o ../logs/output_run_%A_%a.txt #redirect output to output_JOBID.txt
#SBATCH -e ../logs/error_run_%A_%a.txt #redirect errors to error_JOBID.txt
#SBATCH --gres=gpu:volta:2
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=20
#SBATCH --array=0


TASK_ID=0
EXP_ID="0_0"

echo $TASK_ID
echo $EXP_ID

module load /home/gridsan/groups/datasets/ImageNet/modulefile 

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=$((12073 + TASK_ID))
echo $MASTER_PORT

#export MASTER_ADDR=$master_addr

export OMP_NUM_THREADS=24

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
--nproc_per_node=4 main.py --lr 0.005 --arch mobilenetv1 --name_dataset imagenet --optimizer_name SGD --selection_reg 0.1 --l2_reg 1e-05 --pretrained True --goal_sparsity 0.8 --folder_saves Saves_combined_pretrained_imagenet