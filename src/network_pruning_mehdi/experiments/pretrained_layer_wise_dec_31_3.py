#%%
import numpy as np
import json
import os

l_trials_values = []
l_names = []

# obc deit small
d_params = {}
d_params["learning_rate"] = [0.01]
d_params["dense_to_sparse"] = [0]
d_params["optimizer_name"] = ["SGD"]
d_params["type_reset"] = ["ensemble"]
d_params["type_pruning"] = ["magnitude"]
d_params["arch"] = ["deit_small_patch16_224"]
d_params["name_dataset"] = ["imagenet"]
d_params["n_epochs"] = [0]
d_params["T_max_cos"] = [200]
d_params["n_train_kept"] = [5000]
d_params["patience"] = [50]
d_params["num_workers"] = [0]
d_params["l2_reg"] = [1e-3]
d_params["entropy_reg"] = [1e-3]
d_params["selection_reg"] = [1e-3]
d_params["l2_original_reg"] = [0]
d_params["seed"] = [50]
d_params["gamma"] = [1e0]
d_params["test_constraint_weights"] = [0]
#d_params["val_second_lr"] = [0.001]
d_params["test_different_lr"] = [0]
d_params["type_decay"] = ["None"]
d_params["pretrained"] = ["True"]
d_params["goal_sparsity"] = [0.5]
d_params["test_normalized_sgd"] = [1]
d_params["mode"] = ["layer_wise"]
d_params["loss_func"] = ["layer_wise"]
d_params["pruning_rate_cte"] = [0.1]
d_params["test_early_stopping"] = [0, 1]
d_params["threshold_weights"] = [0]
d_params["lambda_loss"] = [100.0]
d_params["test_repeat_if_sparsity_not_reached"] = [1]
d_params["loss_last_block"] = ["layer_wise"]

d_params["retrain_last_block"] = [1]
d_params["test_mult_reset"] = [1]
d_params["test_reset_to_orignal"] = [0]
d_params["test_start_sparse_gpt"] = [0]
d_params["test_start_obc"] = [1]
d_params["rel_damp"] = [0]
d_params["test_start_convex"] = [0]
d_params["prune_bias"] = [0]
d_params["type_compute_sparsity"] = ["prunable"]
d_params["test_adaptive_lr"] = [1]
d_params["patience_adaptive_lr"] = [5]
d_params["patience_freeze"] = [10]
d_params["test_wait_for_pruning"] = [1]
d_params["test_almost_sequential"] = [0, 3]
d_params["tol_ent_reg"] = [1e-2]
d_params["tol_sel_reg"] = [1e-2]
d_params["n_incr_gradual_pruning"] = [-1]
d_params["goal_sparsity_discrete"] = [0.5]
#d_params["type_pruning_schedule"] = ["linear", "exponential"]
d_params["folder_saves"] = ["Saves_opt_dec_31"]

#%%
command = "python main.py"
for key in d_params:
    if key == "folder_saves":
        command += f" --{key} Saves_test_opt"
    elif key == "learning_rate":
        command += f" --lr {d_params[key][0]}"
    else:
        command += f" --{key} {d_params[key][0]}"

print(command)
#%%
with open("experiments/"+os.path.basename(__file__).split(".")[0]+".json", "w") as outfile:
    json.dump(d_params, outfile)