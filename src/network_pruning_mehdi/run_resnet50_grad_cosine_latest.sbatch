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

module purge
module load anaconda/2022a
#module load /home/gridsan/groups/datasets/ImageNet/modulefile

TASK_ID=$SLURM_ARRAY_TASK_ID
TASK_ID=0
EXP_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

CHECKPOINT_PATH=""
FIRST_EPOCH=0

echo $TASK_ID

echo $EXP_ID

algos=("Heuristic_LSBlock" "MP")
block_sizes=(500 -1)
split_types=(1 -1)
algo=${algos[0]}
block_size=${block_sizes[0]}
split_type=${split_types[0]}

nums_stages=(1 1 16)



sparsity_schedule="poly"

training_schedules=("cosine_fast" "cosine_slow" "cosine_one")
training_schedule=${training_schedules[TASK_ID%3]}
num_stages=${nums_stages[TASK_ID%3]}
TASK_ID=$((TASK_ID/3))


if [ $training_schedule == "cosine_fast" ] 
then 
    max_lr=0.05
    min_lr=0.000005
    prune_every=15
    nprune_epochs=6
    nepochs=100
    warm_up=0
    gamma_ft=-1
fi
if [ $training_schedule == "cosine_fast_gamma" ] 
then 
    max_lr=0.05
    min_lr=0.000005
    prune_every=15
    nprune_epochs=7
    nepochs=150
    warm_up=0
    gamma_ft=0.8
fi
if [ $training_schedule == "cosine_one" ] 
then 
    max_lr=0.256
    min_lr=0.000005
    prune_every=1
    nprune_epochs=1
    nepochs=100
    warm_up=5
    gamma_ft=-1
fi
if [ $training_schedule == "cosine_slow" ]
then
    max_lr=0.005
    min_lr=0.000005
    prune_every=4
    nprune_epochs=16
    nepochs=100
    warm_up=0
    gamma_ft=-1
fi
if [ $training_schedule == "cosine_fast_slr" ]
then
    max_lr=0.005
    min_lr=0.000005
    prune_every=10
    nprune_epochs=5
    nepochs=100
    warm_up=0
    gamma_ft=-1
fi

echo $max_lr

seed=1

fisher_subsample_sizes=(500)
fisher_subsample_size=${fisher_subsample_sizes[0]}

l2s=(0.0001 0.001)
l2=${l2s[0]}

fisher_mini_bszs=(1)
fisher_mini_bsz=16

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=12345
export WORLD_SIZE=8

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

srun python3 -u run_experiment_gradual_new.py --arch resnet50 --dset imagenet --num_workers 40 --exp_name grad_dec12 --exp_id ${EXP_ID} --compute_training_losses False --restrict_support True --shuffle_train True --use_wbar True --use_activeset True --test_batch_size 512 --train_batch_size 256  --fisher_subsample_size ${fisher_subsample_size} --fisher_mini_bsz ${fisher_mini_bsz} --fisher_data_bsz ${fisher_mini_bsz} --num_iterations 1 --num_stages ${num_stages} --seed ${seed} --first_order_term False --compute_trace_H False --recompute_X True --sparsity 0.98 --base_level 0.3 --outer_base_level 0.5  --l2 ${l2} --sparsity_schedule ${sparsity_schedule} --training_schedule ${training_schedule} --algo ${algo} --normalize False --block_size ${block_size}  --split_type ${split_type} --max_lr ${max_lr} --min_lr ${min_lr} --prune_every ${prune_every} --nprune_epochs ${nprune_epochs} --nepochs ${nepochs} --gamma_ft ${gamma_ft} --warm_up ${warm_up} 