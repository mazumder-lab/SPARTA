#%%
import os
import sys

to_run = True
supercloud_env = "/home/gridsan/gafriat/.conda/envs/pruning/bin/python"

#l_goal_sparisties = [0.5, 0.6, 0.7, 0.8, 0.9]#, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98]
l_lambdas_sparsity = [1e-3, 5e-3, 1e-2, 5e-2]
initial_sparsity = 0.0
max_sparsity_start = 1
n_rounds = 3

l_arch = ["deit_tiny_patch16_224"]#, "deit_small_patch16_224"]
#arch = "deit_small_patch16_224"
#arch = "deit_base_patch16_224"
l_commands = []

for arch in l_arch:

    for n_convex in [-1]:

        for lambda_sparsity in l_lambdas_sparsity:
            if n_convex == -1:
                l_commands += [
                    f"cd_non_uniform.py {arch} imagenet {n_rounds} --n_train_kept 5000 --batch_size_dataset 128 --lambda_fisher 1.0 --lambda_reconst 1.0 --rel_damp 1e-2 --initial_sparsity {initial_sparsity} --lambda_sparsity {lambda_sparsity}",
                    #f"greedy_non_uniform.py {arch} imagenet {n_rounds} --n_train_kept 5000 --batch_size_dataset 128 --lambda_fisher 0.0 --lambda_reconst 1.0 --rel_damp 1e-2 --initial_sparsity {initial_sparsity} --lambda_sparsity {lambda_sparsity}",
                    #f"greedy_non_uniform.py {arch} imagenet {n_rounds} --n_train_kept 5000 --batch_size_dataset 128 --lambda_fisher 1.0 --lambda_reconst 0.0 --rel_damp 1e-2 --initial_sparsity {initial_sparsity} --lambda_sparsity {lambda_sparsity}"
                ]
            else:
                l_commands += [
                    f"cd_non_uniform.py {arch} imagenet {n_rounds} --n_train_kept 5000 --batch_size_dataset 128 --n_convex {n_convex} --rel_damp 1e-2 --initial_sparsity {initial_sparsity} --lambda_sparsity {lambda_sparsity}"
                ]

l_scripts_to_run = [supercloud_env+" "+command for command in l_commands]

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
