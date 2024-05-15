#%%
import os
import sys

to_run = True
EXPERIMENT_DIR = "benchmarking_all_exp_deit"

os.chdir("..")

# Check and create directory if it doesn't exist
if not(os.path.exists(EXPERIMENT_DIR)):
    try:
        os.mkdir(EXPERIMENT_DIR)
    except:
        pass

l_epsilons = [1.0]
l_clippings=[0.75, 1.0]
#l_clippings=[1.0]
l_batch_sizes=[500]
l_epochs=[50]
l_lrs=[(0.2,0.01), (0.2, 0.005), (0.1, 0.005), (0.1, 0.001)]
l_sparisities=[0.1, 0.2, 0.5, 0.8, 0.9]
l_use_delta_weight_optims = [1]
l_use_fixed_w_mask_findings = [1]
seed = 0


l_models = ["deit_base_patch16_224"]
l_name_datasets = ["cifar10"]#, "cifar10"]

# (classifier_lr, lr)
l_scripts_to_run = []
for use_fixed_w_mask_finding in l_use_fixed_w_mask_findings:
    for use_delta_weight_optim in l_use_delta_weight_optims:
        for sparsity in l_sparisities:
            for model in l_models:
                for epsilon in l_epsilons:
                    for clipping in l_clippings:
                        for batch_size in l_batch_sizes:
                            for epochs in l_epochs:
                                for (classifier_lr, lr) in l_lrs:
                                    for name_dataset in l_name_datasets:
                                        num_classes = int(name_dataset.replace("cifar", ""))
                                        command = f'python train_cifar.py --use_delta_weight_optim {use_delta_weight_optim} --dataset {name_dataset} --batch_size {batch_size} --model {model} --num_classes {num_classes} --classifier_lr {classifier_lr} --lr {lr} --lsr 0.0 --wd 0.0 --momentum 0.9 --lr_schedule_type "onecycle" --num_epochs {epochs} --magnitude_descending False --warm_up 0.01 --finetune_strategy "all_layers"   --use_gn True --method_name "row_pruning_noisy_grads"   --mask_type "optimization" --sparsity {sparsity} --epsilon {epsilon} --delta 1e-5 --clipping {clipping} --experiment_dir {EXPERIMENT_DIR} --out_file "row_pruning_noisy_grads.txt"  --seed {seed}  --SLURM_JOB_ID 0 --TASK_ID 0'
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
