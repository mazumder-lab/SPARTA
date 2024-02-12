#%%
import os
import sys

to_run = True
supercloud_env = "/home/gridsan/gafriat/.conda/envs/pruning/bin/python"

l_goal_sparisties = [0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98]

arch = "deit_tiny_patch16_224"
#arch = "deit_small_patch16_224"
#arch = "deit_base_patch16_224"
pruning_level = "block"

l_n_convex = [-1]
n_layers = 2

test_run_database = False
test_run_dp = False
test_run_postproc = False
test_run_postproc_uniform = True
test_run_postproc_CAP = False

l_commands = []

for n_convex in l_n_convex:
    if test_run_database:
        if n_convex == -1:
            l_commands += [
                f"database_obc.py {arch} imagenet unstr loss --n_train_kept 5000 --batch_size_dataset 128 --lambda_fisher 1.0 --lambda_reconst 1.0 --rel_damp 1e-2",
                f"database_obc.py {arch} imagenet unstr loss --n_train_kept 5000 --batch_size_dataset 128 --lambda_fisher 0.0 --lambda_reconst 1.0 --rel_damp 1e-2",
                f"database_obc.py {arch} imagenet unstr loss --n_train_kept 5000 --batch_size_dataset 128 --lambda_fisher 1.0 --lambda_reconst 0.0 --rel_damp 1e-2"
            ]
        else:
            l_commands += [
                f"database_obc.py {arch} imagenet unstr loss --n_train_kept 5000 --batch_size_dataset 128 --rel_damp 1e-2 --n_convex {n_convex}"
            ]

    if test_run_dp:
        if n_convex == -1:
            for goal_sparsity in l_goal_sparisties:
                l_commands += [
                    f"spdy_obc.py {arch} imagenet {goal_sparsity} unstr --dp --constr sparsity --n_train_kept 5000 --batch_size_dataset 128 --lambda_fisher 1.0 --lambda_reconst 1.0 --rel_damp 1e-2",
                    f"spdy_obc.py {arch} imagenet {goal_sparsity} unstr --dp --constr sparsity --n_train_kept 5000 --batch_size_dataset 128 --lambda_fisher 1.0 --lambda_reconst 0.0 --rel_damp 1e-2",
                    f"spdy_obc.py {arch} imagenet {goal_sparsity} unstr --dp --constr sparsity --n_train_kept 5000 --batch_size_dataset 128 --lambda_fisher 0.0 --lambda_reconst 1.0 --rel_damp 1e-2"
                ]
        else:
            for goal_sparsity in l_goal_sparisties:
                l_commands += [
                    f"spdy_obc.py {arch} imagenet {goal_sparsity} unstr --dp --constr sparsity --n_train_kept 5000 --batch_size_dataset 128 --rel_damp 1e-2 --n_convex {n_convex}"
                ]

    if test_run_postproc:
        if n_convex == -1:
            for goal_sparsity in l_goal_sparisties:
                l_commands += [
                    f"postproc_obc.py {arch} imagenet {goal_sparsity} --dp --database unstr --n_train_kept 5000 --batch_size_dataset 128 --lambda_fisher 1.0 --lambda_reconst 1.0 --rel_damp 1e-2",
                    f"postproc_obc.py {arch} imagenet {goal_sparsity} --dp --database unstr --n_train_kept 5000 --batch_size_dataset 128 --lambda_fisher 1.0 --lambda_reconst 0.0 --rel_damp 1e-2",
                    f"postproc_obc.py {arch} imagenet {goal_sparsity} --dp --database unstr --n_train_kept 5000 --batch_size_dataset 128 --lambda_fisher 0.0 --lambda_reconst 1.0 --rel_damp 1e-2",
                ]
        else:
            for goal_sparsity in l_goal_sparisties:
                l_commands += [
                    f"postproc_obc.py {arch} imagenet {goal_sparsity} --dp --database unstr --n_train_kept 5000 --batch_size_dataset 128 --rel_damp 1e-2 --n_convex {n_convex}"
                ]

    if test_run_postproc_uniform:
        if n_convex == -1:
            for goal_sparsity in l_goal_sparisties:
                l_commands += [
                    f"postproc_uniform_obc.py {arch} imagenet {goal_sparsity} --database unstr --n_train_kept 5000 --batch_size_dataset 128 --lambda_fisher 1.0 --lambda_reconst 1.0 --rel_damp 1e-2 --pruning_level {pruning_level} --n_layers {n_layers}",
                    #f"postproc_uniform_obc.py {arch} imagenet {goal_sparsity} --database unstr --n_train_kept 5000 --batch_size_dataset 128 --lambda_fisher 1.0 --lambda_reconst 0.0 --rel_damp 1e-2 --pruning_level {pruning_level} --n_layers {n_layers}",
                    #f"postproc_uniform_obc.py {arch} imagenet {goal_sparsity} --database unstr --n_train_kept 5000 --batch_size_dataset 128 --lambda_fisher 0.0 --lambda_reconst 1.0 --rel_damp 1e-2 --pruning_level {pruning_level} --n_layers {n_layers}",
                ]
        else:
            for goal_sparsity in l_goal_sparisties:
                l_commands += [
                    f"postproc_uniform_obc.py {arch} imagenet {goal_sparsity} --database unstr --n_train_kept 5000 --batch_size_dataset 128 --rel_damp 1e-2 --n_convex {n_convex} --pruning_level {pruning_level}"
                ]


if test_run_postproc_CAP:
    for goal_sparsity in l_goal_sparisties:
        l_commands += [
            f"postproc_CAP.py {arch} imagenet {goal_sparsity} --n_train_kept 5000 --batch_size_dataset 128",
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

    # %%
