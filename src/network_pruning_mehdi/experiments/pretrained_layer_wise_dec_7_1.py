#%%
import numpy as np
import json
import os

l_trials_values = []
l_names = []

# Gridsearch
d_params = {}
d_params["learning_rate"] = [0.001]
d_params["weight_decay"] = [0.00001]
d_params["dense_to_sparse"] = [0]
d_params["optimizer_name"] = ["SGD"]
d_params["batch_size_dataset"] = [1]
d_params["type_reset"] = ["ensemble"]
d_params["type_pruning"] = ["magnitude"]
d_params["arch"] = ["facebook/opt-125m"]
d_params["name_dataset"] = ["c4"]
d_params["n_epochs"] = [200]
d_params["T_max_cos"] = [200]
d_params["n_train_kept"] = [128]
d_params["patience"] = [20]
d_params["num_workers"] = [0]
d_params["l2_reg"] = [1e-3]
d_params["entropy_reg"] = [1e-3]
d_params["selection_reg"] = [1e-3]
d_params["l2_original_reg"] = [0]
d_params["seed"] = [50]
d_params["gamma"] = [1e0]
d_params["test_constraint_weights"] = [0]
d_params["test_different_lr"] = [1]
d_params["val_second_lr"] = [0.001]
d_params["type_decay"] = ["None"]
d_params["pretrained"] = ["True"]
d_params["goal_sparsity"] = [0.5]
d_params["test_normalized_sgd"] = [1]
d_params["mode"] = ["layer_wise"]
d_params["loss_func"] = ["layer_wise"]
d_params["pruning_rate_cte"] = [0.1] # Add 0.05 maybe?
d_params["test_early_stopping"] = [1] # Add 0,2 maybe?
d_params["threshold_weights"] = [1]
d_params["lambda_loss"] = [100.0]
d_params["test_repeat_if_sparsity_not_reached"] = [1]
d_params["loss_last_block"] = ["layer_wise"]
d_params["retrain_last_block"] = [1]
d_params["test_mult_reset"] = [1]
d_params["test_reset_to_orignal"] = [0]
d_params["test_start_sparse_gpt"] = [0]
d_params["prune_bias"] = [0]
d_params["type_compute_sparsity"] = ["prunable"]
d_params["test_adaptive_lr"] = [1]
d_params["patience_adaptive_lr"] = [5]
d_params["patience_freeze"] = [10]
d_params["test_wait_for_pruning"] = [1]
d_params["test_almost_sequential"] = [0]
d_params["tol_ent_reg"] = [0]
d_params["tol_sel_reg"] = [1e-1, 5e-2]
d_params["folder_saves"] = ["Saves_opt_dec_7_1"]

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
name_script = ".".join(os.path.basename(__file__).split(".")[:-1])
with open("experiments/"+name_script+".json", "w") as outfile:
    json.dump(d_params, outfile)
# %%
