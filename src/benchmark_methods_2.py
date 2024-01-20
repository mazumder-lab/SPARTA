import os
import subprocess

# Parameters
epsilons = [1.0]
clippings = [1.0, 0.75, 0.5]
batch_sizes = [250, 500]
epochss = [50, 100]
classifier_lrs = [0.2, 0.4]
lrs = [0.01, 0.05]
sparsities = [0.1, 0.01, 0.2]
experiment_dir = "benchmarking_all_exp_gc_2"
total_tasks = 18  # Total number of tasks in your SLURM array


# Function to run a command
def run_command(command):
    subprocess.run(command, shell=True)


# Check and create directory if it doesn't exist
os.makedirs(experiment_dir, exist_ok=True)

# Main loop to run experiments
for task_id in range(1, total_tasks):
    current_task_id = task_id
    epsilon = epsilons[current_task_id % 1]
    current_task_id //= 1

    clipping = clippings[current_task_id % 3]
    current_task_id //= 3

    batch_size = 5000
    epochs = 100

    classifier_lr = classifier_lrs[current_task_id % 2]
    lr = lrs[current_task_id % 2]
    current_task_id //= 2

    sparsity = sparsities[current_task_id % 3]
    current_task_id //= 3

    # Define the command to run
    base_command = f"CUDA_VISIBLE_DEVICES=2 python3 -m train_cifar --dataset cifar10 --batch_size {batch_size} --model resnet18 --num_classes 10 --classifier_lr {classifier_lr} --lr {lr} --lsr 0.0 --wd 0.0 --momentum 0.9 --lr_schedule_type onecycle --num_epochs {epochs} --warm_up 0.01 --use_gn True --use_dp True --epsilon {epsilon} --delta 1e-5 --clipping {clipping} --experiment_dir {experiment_dir} --seed 0 --SLURM_JOB_ID 0 --TASK_ID {task_id}"
    noisy_grad_command = (
        base_command
        + f" --magnitude_descending False --finetune_strategy all_layers --use_magnitude_mask True --use_adaptive_magnitude_mask True --type_mask noisy_grad_magnitude --sparsity {sparsity} --out_file noisy_grad_magnitude_out_file.txt"
    )
    largest_adaptive_magnitude_command = (
        base_command
        + f" --magnitude_descending True --finetune_strategy all_layers --use_magnitude_mask True --use_adaptive_magnitude_mask True --type_mask magnitude --sparsity {sparsity} --out_file largest_adaptive_magnitude_out_file.txt"
    )
    largest_fixed_magnitude_command = (
        base_command
        + f" --magnitude_descending True --finetune_strategy all_layers --use_magnitude_mask True --use_adaptive_magnitude_mask False --type_mask '' --sparsity {sparsity} --out_file largest_fixed_magnitude_out_file.txt"
    )
    subset_command = (
        base_command
        + " --magnitude_descending False --finetune_strategy conf_indices --use_magnitude_mask False --use_adaptive_magnitude_mask False --type_mask '' --sparsity 0.0 --out_file subset_out_file.txt"
    )
    all_layers_command = (
        base_command
        + " --magnitude_descending False --finetune_strategy all_layers --use_magnitude_mask False --use_adaptive_magnitude_mask False --type_mask '' --sparsity 0.0 --out_file all_layers_out_file.txt"
    )
    smallest_adaptive_magnitude_command = (
        base_command
        + f" --magnitude_descending False --finetune_strategy all_layers --use_magnitude_mask True --use_adaptive_magnitude_mask True --type_mask magnitude --sparsity {sparsity} --out_file smallest_adaptive_magnitude_out_file.txt"
    )
    smalles_fixed_magnitude_command = (
        base_command
        + f" --magnitude_descending False --finetune_strategy all_layers --use_magnitude_mask True --use_adaptive_magnitude_mask False --type_mask '' --sparsity {sparsity} --out_file smallest_fixed_magnitude_out_file.txt"
    )

    commands = [
        noisy_grad_command,
        largest_adaptive_magnitude_command,
        subset_command,
        all_layers_command,
        smallest_adaptive_magnitude_command,
        smalles_fixed_magnitude_command,
    ]
    # Running different configurations
    for command in commands:
        run_command(command)
