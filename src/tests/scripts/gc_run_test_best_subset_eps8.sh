#!/bin/bash 
# Loading the required module
source activate pruning

cd ..
cd ..

CUDA_VISIBLE_DEVICES=0 python3 -m train_cifar --dataset "cifar10" --batch_size 500 --model "resnet18" --num_classes 10 --lr_schedule_type "warmup_cosine" --classifier_lr 0.4 --lr 0.05 --lsr 0.0 --use_adaptive_lr False --wd 0.0 --momentum 0.9 --clip_gradient False --num_epochs 50 --accum_steps 1 --warm_up 0.01 --finetune_strategy "conf_indices" --use_gn True --use_dp True --epsilon 8.0 --delta 1e-5 --clipping 1.1 --out_file "test_best_subset_eps8.txt" --seed 0  --SLURM_JOB_ID "slurm_id" --TASK_ID $SLURM_ARRAY_TASK_ID
