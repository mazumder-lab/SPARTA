#%%
import os
import sys

to_run = True
supercloud_env = "/home/gridsan/gafriat/.conda/envs/pruning/bin/python"

#l_goal_sparisties = [0.5, 0.6, 0.7, 0.8, 0.9]#, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98]
l_goal_sparisties = [0.9]
l_initial_sparsity = [0.2, 0.4, 0.6]
l_further_subsampling = [500, 1000, -1]
max_sparsity_start = 1
no_approximation = 1

l_arch = ["deit_tiny_patch16_224"]#, "deit_small_patch16_224"]
#arch = "deit_small_patch16_224"
#arch = "deit_base_patch16_224"
l_commands = []

for arch in l_arch:

    for n_convex in [-1]:

        for goal_sparsity in l_goal_sparisties:
            for initial_sparsity in l_initial_sparsity:
                if n_convex == -1:
                    l_commands += [
                        f"greedy_non_uniform.py {arch} imagenet {goal_sparsity} --n_train_kept 5000 --batch_size_dataset 128 --lambda_fisher 1.0 --lambda_reconst 1.0 --rel_damp 1e-2 --max_sparsity_start {max_sparsity_start} --no_approximation {no_approximation} --initial_sparsity {initial_sparsity}",
                        #f"greedy_non_uniform.py {arch} imagenet {goal_sparsity} --n_train_kept 5000 --batch_size_dataset 128 --lambda_fisher 0.0 --lambda_reconst 1.0 --rel_damp 1e-2 --max_sparsity_start {max_sparsity_start} --no_approximation {no_approximation} --initial_sparsity {initial_sparsity}",
                        #f"greedy_non_uniform.py {arch} imagenet {goal_sparsity} --n_train_kept 5000 --batch_size_dataset 128 --lambda_fisher 1.0 --lambda_reconst 0.0 --rel_damp 1e-2 --max_sparsity_start {max_sparsity_start} --no_approximation {no_approximation} --initial_sparsity {initial_sparsity}"
                    ]
                else:
                    l_commands += [
                        f"greedy_non_uniform.py {arch} imagenet {goal_sparsity} --n_train_kept 5000 --batch_size_dataset 128 --n_convex {n_convex} --rel_damp 1e-2 --max_sparsity_start {max_sparsity_start} --no_approximation {no_approximation} --initial_sparsity {initial_sparsity}"
                    ]

l_scripts_to_run = [supercloud_env+" "+command for command in l_commands]

#%%
l_scripts_to_run = []
l_further_subsampling = [-1]
l_test_recompute = [1]
max_sparsity_start = -1
no_approximation = 0

l_goal_sparisties = ["\"0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9\""]
l_initial_sparsity = [0.2]

l_arch = ["deit_tiny_patch16_224"]#, "deit_small_patch16_224"]
#arch = "deit_small_patch16_224"
#arch = "deit_base_patch16_224"
l_commands = []

for arch in l_arch:

    for n_convex in [-1]:
        for test_recompute in l_test_recompute:
            for goal_sparsities in l_goal_sparisties:
                for initial_sparsity in l_initial_sparsity:
                    for further_subsampling in l_further_subsampling:
                        if n_convex == -1:
                            l_commands += [
                                f"greedy_non_uniform.py {arch} imagenet --l_goal_sparsities {goal_sparsities} --n_train_kept 5000 --batch_size_dataset 128 --lambda_fisher 1.0 --lambda_reconst 1.0 --rel_damp 1e-2 --max_sparsity_start {max_sparsity_start} --no_approximation {no_approximation} --initial_sparsity {initial_sparsity} --further_subsampling {further_subsampling} --test_recompute {test_recompute}",
                                #f"greedy_non_uniform.py {arch} imagenet {goal_sparsity} --n_train_kept 5000 --batch_size_dataset 128 --lambda_fisher 0.0 --lambda_reconst 1.0 --rel_damp 1e-2 --max_sparsity_start {max_sparsity_start} --no_approximation {no_approximation} --initial_sparsity {initial_sparsity}",
                                #f"greedy_non_uniform.py {arch} imagenet {goal_sparsity} --n_train_kept 5000 --batch_size_dataset 128 --lambda_fisher 1.0 --lambda_reconst 0.0 --rel_damp 1e-2 --max_sparsity_start {max_sparsity_start} --no_approximation {no_approximation} --initial_sparsity {initial_sparsity}"
                            ]
                        else:
                            l_commands += [
                                f"greedy_non_uniform.py {arch} imagenet --l_goal_sparsities {goal_sparsities} --n_train_kept 5000 --batch_size_dataset 128 --n_convex {n_convex} --rel_damp 1e-2 --max_sparsity_start {max_sparsity_start} --no_approximation {no_approximation} --initial_sparsity {initial_sparsity} --further_subsampling {further_subsampling} --test_recompute {test_recompute}"
                            ]

l_scripts_to_run += [supercloud_env+" "+command for command in l_commands]

l_goal_sparisties = ["\"0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9\""]
l_initial_sparsity = [0.3]

l_arch = ["deit_tiny_patch16_224"]#, "deit_small_patch16_224"]
#arch = "deit_small_patch16_224"
#arch = "deit_base_patch16_224"
l_commands = []

for arch in l_arch:

    for n_convex in [-1]:
        for test_recompute in l_test_recompute:
            for goal_sparsities in l_goal_sparisties:
                for initial_sparsity in l_initial_sparsity:
                    for further_subsampling in l_further_subsampling:
                        if n_convex == -1:
                            l_commands += [
                                f"greedy_non_uniform.py {arch} imagenet --l_goal_sparsities {goal_sparsities} --n_train_kept 5000 --batch_size_dataset 128 --lambda_fisher 1.0 --lambda_reconst 1.0 --rel_damp 1e-2 --max_sparsity_start {max_sparsity_start} --no_approximation {no_approximation} --initial_sparsity {initial_sparsity} --further_subsampling {further_subsampling} --test_recompute {test_recompute}",
                                #f"greedy_non_uniform.py {arch} imagenet {goal_sparsity} --n_train_kept 5000 --batch_size_dataset 128 --lambda_fisher 0.0 --lambda_reconst 1.0 --rel_damp 1e-2 --max_sparsity_start {max_sparsity_start} --no_approximation {no_approximation} --initial_sparsity {initial_sparsity}",
                                #f"greedy_non_uniform.py {arch} imagenet {goal_sparsity} --n_train_kept 5000 --batch_size_dataset 128 --lambda_fisher 1.0 --lambda_reconst 0.0 --rel_damp 1e-2 --max_sparsity_start {max_sparsity_start} --no_approximation {no_approximation} --initial_sparsity {initial_sparsity}"
                            ]
                        else:
                            l_commands += [
                                f"greedy_non_uniform.py {arch} imagenet --l_goal_sparsities {goal_sparsities} --n_train_kept 5000 --batch_size_dataset 128 --n_convex {n_convex} --rel_damp 1e-2 --max_sparsity_start {max_sparsity_start} --no_approximation {no_approximation} --initial_sparsity {initial_sparsity} --further_subsampling {further_subsampling} --test_recompute {test_recompute}"
                            ]

l_scripts_to_run += [supercloud_env+" "+command for command in l_commands]

l_goal_sparisties = ["\"0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9\""]
l_initial_sparsity = [0.4]

l_arch = ["deit_tiny_patch16_224"]#, "deit_small_patch16_224"]
#arch = "deit_small_patch16_224"
#arch = "deit_base_patch16_224"
l_commands = []

for arch in l_arch:

    for n_convex in [-1]:
        for test_recompute in l_test_recompute:
            for goal_sparsities in l_goal_sparisties:
                for initial_sparsity in l_initial_sparsity:
                    for further_subsampling in l_further_subsampling:
                        if n_convex == -1:
                            l_commands += [
                                f"greedy_non_uniform.py {arch} imagenet --l_goal_sparsities {goal_sparsities} --n_train_kept 5000 --batch_size_dataset 128 --lambda_fisher 1.0 --lambda_reconst 1.0 --rel_damp 1e-2 --max_sparsity_start {max_sparsity_start} --no_approximation {no_approximation} --initial_sparsity {initial_sparsity} --further_subsampling {further_subsampling} --test_recompute {test_recompute}",
                                #f"greedy_non_uniform.py {arch} imagenet {goal_sparsity} --n_train_kept 5000 --batch_size_dataset 128 --lambda_fisher 0.0 --lambda_reconst 1.0 --rel_damp 1e-2 --max_sparsity_start {max_sparsity_start} --no_approximation {no_approximation} --initial_sparsity {initial_sparsity}",
                                #f"greedy_non_uniform.py {arch} imagenet {goal_sparsity} --n_train_kept 5000 --batch_size_dataset 128 --lambda_fisher 1.0 --lambda_reconst 0.0 --rel_damp 1e-2 --max_sparsity_start {max_sparsity_start} --no_approximation {no_approximation} --initial_sparsity {initial_sparsity}"
                            ]
                        else:
                            l_commands += [
                                f"greedy_non_uniform.py {arch} imagenet --l_goal_sparsities {goal_sparsities} --n_train_kept 5000 --batch_size_dataset 128 --n_convex {n_convex} --rel_damp 1e-2 --max_sparsity_start {max_sparsity_start} --no_approximation {no_approximation} --initial_sparsity {initial_sparsity} --further_subsampling {further_subsampling} --test_recompute {test_recompute}"
                            ]

l_scripts_to_run += [supercloud_env+" "+command for command in l_commands]

l_goal_sparisties = ["\"0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9\""]
l_initial_sparsity = [0.5]

l_arch = ["deit_tiny_patch16_224"]#, "deit_small_patch16_224"]
#arch = "deit_small_patch16_224"
#arch = "deit_base_patch16_224"
l_commands = []

for arch in l_arch:

    for n_convex in [-1]:
        for test_recompute in l_test_recompute:
            for goal_sparsities in l_goal_sparisties:
                for initial_sparsity in l_initial_sparsity:
                    for further_subsampling in l_further_subsampling:
                        if n_convex == -1:
                            l_commands += [
                                f"greedy_non_uniform.py {arch} imagenet --l_goal_sparsities {goal_sparsities} --n_train_kept 5000 --batch_size_dataset 128 --lambda_fisher 1.0 --lambda_reconst 1.0 --rel_damp 1e-2 --max_sparsity_start {max_sparsity_start} --no_approximation {no_approximation} --initial_sparsity {initial_sparsity} --further_subsampling {further_subsampling} --test_recompute {test_recompute}",
                                #f"greedy_non_uniform.py {arch} imagenet {goal_sparsity} --n_train_kept 5000 --batch_size_dataset 128 --lambda_fisher 0.0 --lambda_reconst 1.0 --rel_damp 1e-2 --max_sparsity_start {max_sparsity_start} --no_approximation {no_approximation} --initial_sparsity {initial_sparsity}",
                                #f"greedy_non_uniform.py {arch} imagenet {goal_sparsity} --n_train_kept 5000 --batch_size_dataset 128 --lambda_fisher 1.0 --lambda_reconst 0.0 --rel_damp 1e-2 --max_sparsity_start {max_sparsity_start} --no_approximation {no_approximation} --initial_sparsity {initial_sparsity}"
                            ]
                        else:
                            l_commands += [
                                f"greedy_non_uniform.py {arch} imagenet --l_goal_sparsities {goal_sparsities} --n_train_kept 5000 --batch_size_dataset 128 --n_convex {n_convex} --rel_damp 1e-2 --max_sparsity_start {max_sparsity_start} --no_approximation {no_approximation} --initial_sparsity {initial_sparsity} --further_subsampling {further_subsampling} --test_recompute {test_recompute}"
                            ]

l_scripts_to_run += [supercloud_env+" "+command for command in l_commands]

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
