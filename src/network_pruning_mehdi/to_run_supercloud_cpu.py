#%%
import os
import sys
import itertools
import numpy as np
from utils_experiments import get_name_study, test_trial_done
import json

sys.path.append("./")
my_task_id = int(sys.argv[1])
num_tasks = int(sys.argv[2])

supercloud_env = "python"
supercloud_env = "/home/gridsan/gafriat/.conda/envs/network_pruning/bin/python"
supercloud_script = "/home/gridsan/gafriat/projects/soft_trees_ensemble/main.py"

l_experiments = ["SAM_vs_SGD_regression", "SAM_vs_SGD_classification"]

l_trials_values = []
l_names = []

for experiment in l_experiments:
    with open("experiments/"+experiment+".json", "r") as f:
            d_params = json.load(f)
    l_params = list(d_params.items())
    l_params_values = [l_params[i][1] for i in range(len(l_params))]
    l_params_names = [l_params[i][0] for i in range(len(l_params))]
    l_trials_values_batch = list(itertools.product(*l_params_values))
    l_names_values_batch = [l_params_names for i in range(len(l_trials_values_batch))]
    l_trials_values += l_trials_values_batch
    l_names += l_names_values_batch

print("Total number of scripts=", len(l_names))
#%%
l_trials_values_remaining = []
l_names_remaining = []
max_len = 0
for i in range(len(l_names)):
    trial_values = l_trials_values[i]
    name_args = l_names[i]
    args = {}
    for j in range(len(name_args)):
        args[name_args[j]] = trial_values[j]
    name_study = get_name_study(**args)
    max_len = max(max_len, len(name_study))
    if "type_model" not in args:
         args["type_model"] = "soft_trees"
    test_deja_train, n_trials_done, test_file_name = test_trial_done(name_study, args["n_trials"], args["n_repeat"], args["folder_saves"], args["type_model"])
    if not(test_deja_train):
        l_trials_values_remaining.append(trial_values)
        l_names_remaining.append(name_args)

print("Number of initial scripts = ", len(l_trials_values))
print("Number of remaining scripts = ", len(l_trials_values_remaining))
print("")
print("------------------")
print("Remaining scripts:")
print("------------------")
#%%
l_names = []
for i in range(len(l_names_remaining)):
    trial_values = l_trials_values_remaining[i]
    name_args = l_names_remaining[i]
    args = {}
    for j in range(len(name_args)):
        args[name_args[j]] = trial_values[j]
    name_study = get_name_study(**args)
    print(name_study, ",", 0)
    l_names.append(name_study)

l_names = np.array(l_names)
l_names_unique = np.unique(l_names)
l_names_count = np.zeros(len(l_names_unique))
# %%
for i in range(len(l_names_unique)):
    l_names_count[i] += len(np.where(l_names_unique[i] == l_names)[0])
# %%
l_names_dupplicates = l_names_unique[l_names_count>1]
# %%
print("")
print("Number of dupplicate scripts = ", len(l_names_dupplicates))
print("")
print("-------------------")
print("Dupplicate scripts:")
print("-------------------")

for name in l_names_dupplicates:
    print(name)

l_trials_values = l_trials_values_remaining
l_names = l_names_remaining
#%%
sub_l_trials_values = l_trials_values[my_task_id:len(l_trials_values):num_tasks]
sub_l_names = l_names[my_task_id:len(l_trials_values):num_tasks]

for ind_trial in range(len(sub_l_trials_values)):
    trial_values = sub_l_trials_values[ind_trial]
    l_params_names = sub_l_names[ind_trial]
    hyper_param = ""
    for i in range(len(l_params_names)):
        name = l_params_names[i]
        hyper_param+=" --"+name+" "+str(trial_values[i])
    print(supercloud_env+" "+supercloud_script+hyper_param)
    os.system(supercloud_env+" "+supercloud_script+hyper_param)

# %%
