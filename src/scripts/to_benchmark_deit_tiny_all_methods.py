# %%
import os
import sys
from itertools import product

to_run = True
EXPERIMENT_DIR = "revised_init_results_tiny_cif100"

os.chdir("..")

# Check and create directory if it doesn't exist
if not (os.path.exists(os.path.join("results_folder", EXPERIMENT_DIR))):
    try:
        os.mkdir(os.path.join("results_folder", EXPERIMENT_DIR))
    except:
        pass

l_epsilons = [1.0]
l_clippings = [1.0, 0.75]
l_batch_sizes = [500]
l_epochs = [50]
l_lrs = [(0.1, 0.005), (0.1, 0.001), (0.01, 0.01), (0.005, 0.005)]
l_sparisities = [0.0, 0.2]
l_use_delta_weight_optims = [1]
l_use_fixed_w_mask_findings = [1]
l_use_cosine_more_epochs = [1]
l_models = ["deit_tiny_patch16_224"]
l_name_datasets = ["cifar100"]
l_epoch_mask_finding = [10]
use_last_layer_only_init = True
l_seeds = [0]

# %%
l_scripts_to_run = []
for cnt, (
    use_fixed_w_mask_finding,
    use_delta_weight_optim,
    use_cosine_more_epochs,
    model,
    epsilon,
    clipping,
    batch_size,
    epochs,
    (classifier_lr, lr),
    name_dataset,
) in enumerate(
    product(
        l_use_fixed_w_mask_findings,
        l_use_delta_weight_optims,
        l_use_cosine_more_epochs,
        l_models,
        l_epsilons,
        l_clippings,
        l_batch_sizes,
        l_epochs,
        l_lrs,
        l_name_datasets,
    )
):
    for seed in l_seeds:
        num_classes = int(name_dataset.replace("cifar", ""))
        for sparsity in l_sparisities:
            command = f'python train_cifar.py --method_name "mp_weights" --max_physical_batch_size 100 --epoch_mask_finding -1 --use_delta_weight_optim {use_delta_weight_optim} --use_cosine_more_epochs {use_cosine_more_epochs} --dataset {name_dataset} --batch_size {batch_size} --model {model} --num_classes {num_classes} --classifier_lr {classifier_lr} --lr {lr} --lsr 0.0 --wd 0.0 --momentum 0.9 --lr_schedule_type "onecycle" --num_epochs {epochs} --warm_up 0.02 --use_gn True --sparsity {sparsity} --epsilon {epsilon} --delta 1e-5 --clipping {clipping} --experiment_dir {EXPERIMENT_DIR} --seed {seed} --TASK_ID {cnt}'
            l_scripts_to_run.append(command)
            command = f'python train_cifar.py --method_name "dp_bitfit"  --max_physical_batch_size 100 --epoch_mask_finding -1 --use_delta_weight_optim {use_delta_weight_optim} --use_cosine_more_epochs {use_cosine_more_epochs} --dataset {name_dataset} --batch_size {batch_size} --model {model} --num_classes {num_classes} --classifier_lr {classifier_lr} --lr {lr} --lsr 0.0 --wd 0.0 --momentum 0.9 --lr_schedule_type "onecycle" --num_epochs {epochs} --warm_up 0.02 --use_gn True --sparsity {sparsity} --epsilon {epsilon} --delta 1e-5 --clipping {clipping} --experiment_dir {EXPERIMENT_DIR} --seed {seed} --TASK_ID {cnt}'
            l_scripts_to_run.append(command)
            for epoch_mask_finding in l_epoch_mask_finding:
                command = f'python train_cifar.py --use_last_layer_only_init {use_last_layer_only_init} --method_name "row_pruning_noisy_grads"      --max_physical_batch_size 100 --epoch_mask_finding {epoch_mask_finding} --use_delta_weight_optim {use_delta_weight_optim} --use_cosine_more_epochs {use_cosine_more_epochs} --dataset {name_dataset} --batch_size {batch_size} --model {model} --num_classes {num_classes} --classifier_lr {classifier_lr} --lr {lr} --lsr 0.0 --wd 0.0 --momentum 0.9 --lr_schedule_type "onecycle" --num_epochs {epochs} --warm_up 0.02 --use_gn True --sparsity {sparsity} --epsilon {epsilon} --delta 1e-5 --clipping {clipping} --experiment_dir {EXPERIMENT_DIR} --seed {seed} --TASK_ID {cnt}'
                l_scripts_to_run.append(command)
                command = f'python train_cifar.py --use_last_layer_only_init {use_last_layer_only_init} --method_name "block_pruning_noisy_grads"    --max_physical_batch_size 100 --epoch_mask_finding {epoch_mask_finding} --use_delta_weight_optim {use_delta_weight_optim} --use_cosine_more_epochs {use_cosine_more_epochs} --dataset {name_dataset} --batch_size {batch_size} --model {model} --num_classes {num_classes} --classifier_lr {classifier_lr} --lr {lr} --lsr 0.0 --wd 0.0 --momentum 0.9 --lr_schedule_type "onecycle" --num_epochs {epochs} --warm_up 0.02 --use_gn True --sparsity {sparsity} --epsilon {epsilon} --delta 1e-5 --clipping {clipping} --experiment_dir {EXPERIMENT_DIR} --seed {seed} --TASK_ID {cnt}'
                l_scripts_to_run.append(command)
                command = f'python train_cifar.py --use_last_layer_only_init {use_last_layer_only_init} --method_name "random_masking"               --max_physical_batch_size 100 --epoch_mask_finding {epoch_mask_finding} --use_delta_weight_optim {use_delta_weight_optim} --use_cosine_more_epochs {use_cosine_more_epochs} --dataset {name_dataset} --batch_size {batch_size} --model {model} --num_classes {num_classes} --classifier_lr {classifier_lr} --lr {lr} --lsr 0.0 --wd 0.0 --momentum 0.9 --lr_schedule_type "onecycle" --num_epochs {epochs} --warm_up 0.02 --use_gn True --sparsity {sparsity} --epsilon {epsilon} --delta 1e-5 --clipping {clipping} --experiment_dir {EXPERIMENT_DIR} --seed {seed} --TASK_ID {cnt}'
                l_scripts_to_run.append(command)



print("Number of scripts:", len(l_scripts_to_run))
# %%
if to_run:
    sys.path.append("./")
    my_task_id = int(sys.argv[1])
    num_tasks = int(sys.argv[2])

    sub_l_script = l_scripts_to_run[my_task_id : len(l_scripts_to_run) : num_tasks]

    for ind_trial in range(len(sub_l_script)):
        script = sub_l_script[ind_trial]
        print(script)
        os.system(script)
