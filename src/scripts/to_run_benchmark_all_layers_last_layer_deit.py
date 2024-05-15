#%%
import os
import sys

to_run = True
supercloud_environment = "/home/gridsan/gafriat/.conda/envs/dp_pruning_2/bin/python"
EXPERIMENT_DIR = "benchmarking_all_exp_deit"

os.chdir("..")

# Check and create directory if it doesn't exist
if not(os.path.exists(EXPERIMENT_DIR)):
    try:
        os.mkdir(EXPERIMENT_DIR)
    except:
        pass

l_epsilons = [1.0]
l_clippings=[0.5, 0.75, 1.0]
l_batch_sizes=[500]
l_epochs=[50]
l_lrs=[(0.2,0.01), (0.2, 0.005), (0.1, 0.005), (0.1, 0.001)]
l_models = ["deit_tiny_patch16_224", "deit_small_patch16_224", "deit_base_patch16_224"]
# (classifier_lr, lr)
l_scripts_to_run = []
for model in l_models:
    for epsilon in l_epsilons:
        for clipping in l_clippings:
            for batch_size in l_batch_sizes:
                for epochs in l_epochs:
                    for (classifier_lr, lr) in l_lrs:
                        command = f'{supercloud_environment} train_cifar_ori.py --dataset "cifar10" --batch_size {batch_size} --model {model} --num_classes 10 --classifier_lr {classifier_lr} --lr {lr} --lsr 0.0 --wd 0.0 --momentum 0.9 --lr_schedule_type "onecycle" --num_epochs {epochs} --magnitude_descending False --warm_up 0.02 --finetune_strategy "all_layers"   --use_gn True --use_magnitude_mask False --use_adaptive_magnitude_mask False --type_mask ""  --sparsity 0.0 --use_dp True --epsilon {epsilon} --delta 1e-5 --clipping {clipping} --experiment_dir {EXPERIMENT_DIR} --out_file "all_layers_out_file.txt" --seed 0 --accum_steps 1'
                        l_scripts_to_run.append(command)
                        command = f'{supercloud_environment} train_cifar_ori.py --dataset "cifar10" --batch_size {batch_size} --model {model} --num_classes 10 --classifier_lr {classifier_lr} --lr {lr} --lsr 0.0 --wd 0.0 --momentum 0.9 --lr_schedule_type "onecycle" --num_epochs {epochs} --magnitude_descending False --warm_up 0.02 --finetune_strategy "lp_gn"        --use_gn True --use_magnitude_mask False --use_adaptive_magnitude_mask False  --type_mask "" --sparsity 0.0 --use_dp True --epsilon {epsilon} --delta 1e-5 --clipping {clipping} --experiment_dir {EXPERIMENT_DIR} --out_file "last_layer_lp_gn.txt"    --seed 0 --accum_steps 1'
                        l_scripts_to_run.append(command)
                        command = f'{supercloud_environment} train_cifar_ori.py --dataset "cifar100" --batch_size {batch_size} --model {model} --num_classes 100 --classifier_lr {classifier_lr} --lr {lr} --lsr 0.0 --wd 0.0 --momentum 0.9 --lr_schedule_type "onecycle" --num_epochs {epochs} --magnitude_descending False --warm_up 0.02 --finetune_strategy "all_layers"   --use_gn True --use_magnitude_mask False --use_adaptive_magnitude_mask False --type_mask ""  --sparsity 0.0 --use_dp True --epsilon {epsilon} --delta 1e-5 --clipping {clipping} --experiment_dir {EXPERIMENT_DIR} --out_file "all_layers_out_file.txt" --seed 0 --accum_steps 1'
                        l_scripts_to_run.append(command)
                        command = f'{supercloud_environment} train_cifar_ori.py --dataset "cifar100" --batch_size {batch_size} --model {model} --num_classes 100 --classifier_lr {classifier_lr} --lr {lr} --lsr 0.0 --wd 0.0 --momentum 0.9 --lr_schedule_type "onecycle" --num_epochs {epochs} --magnitude_descending False --warm_up 0.02 --finetune_strategy "lp_gn"        --use_gn True --use_magnitude_mask False --use_adaptive_magnitude_mask False  --type_mask "" --sparsity 0.0 --use_dp True --epsilon {epsilon} --delta 1e-5 --clipping {clipping} --experiment_dir {EXPERIMENT_DIR} --out_file "last_layer_lp_gn.txt"    --seed 0 --accum_steps 1'
                        l_scripts_to_run.append(command)

print("Number of scripts:", len(l_scripts_to_run))
#%%
if to_run:
    sys.path.append("./")
    my_task_id = int(sys.argv[1])
    num_tasks = int(sys.argv[2])

    sub_l_script = l_scripts_to_run[my_task_id:len(l_scripts_to_run):num_tasks]

    for ind_trial in range(len(sub_l_script)):
        script = sub_l_script[ind_trial]
        print(script)
        os.system(script)

    # %%
