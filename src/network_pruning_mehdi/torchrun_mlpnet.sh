#!/bin/bash 
#SBATCH --partition=xeon-g6-volta 
#SBATCH --constraint=xeon-g6
#SBATCH -o ../logs/output_run_%A_%a.txt #redirect output to output_JOBID.txt
#SBATCH -e ../logs/error_run_%A_%a.txt #redirect errors to error_JOBID.txt
##SBATCH --exclusive
#SBATCH --gres=gpu:volta:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --array=0

module purge
module load anaconda/2022a
module load /home/gridsan/groups/datasets/ImageNet/modulefile

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=29400
export WORLD_SIZE=4

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
#export MASTER_ADDR="d-12-3-1"
echo "MASTER_ADDR="$MASTER_ADDR

# srun python3 -u main.py --lr 0.005 --arch mlpnet --name_dataset mnist --optimizer_name Adam --selection_reg 0.1 --l2_reg 1e-03 --pretrained True --goal_sparsity 0.98 --folder_saves Saves_combined_pretrained_mlpnet_test --num_workers 40 --test_distributed 1
# srun /home/gridsan/gafriat/.conda/envs/additive/bin/python -u main.py --lr 0.005 --arch mlpnet --name_dataset mnist --optimizer_name Adam --selection_reg 0.1 --l2_reg 1e-03 --pretrained True --goal_sparsity 0.98 --folder_saves Saves_combined_pretrained_mlpnet_test --num_workers 40 --test_distributed 1
# srun /home/gridsan/gafriat/.conda/ envs/pytorch1_6/bin/python -u main.py --lr 0.005 --arch mlpnet --name_dataset mnist --optimizer_name Adam --selection_reg 0.1 --l2_reg 1e-03 --pretrained True --goal_sparsity 0.98 --folder_saves Saves_combined_pretrained_mlpnet_test --num_workers 40 --test_distributed 1
# torchrun --nnodes=4 --nproc_per_node=20 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 main.py --lr 0.005 --arch mlpnet --name_dataset mnist --optimizer_name Adam --selection_reg 0.1 --l2_reg 1e-03 --pretrained True --goal_sparsity 0.98 --folder_saves Saves_combined_pretrained_mlpnet_test --num_workers 20 --test_distributed 1
torchrun --nnodes=2 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 main.py --lr 0.005 --arch mlpnet --name_dataset mnist --optimizer_name Adam --selection_reg 0.1 --l2_reg 1e-03 --pretrained True --goal_sparsity 0.98 --folder_saves Saves_combined_pretrained_mlpnet_test --num_workers 20 --test_distributed 1