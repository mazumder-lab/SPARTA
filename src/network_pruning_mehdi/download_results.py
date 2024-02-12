#%%
import numpy as np
import os
import sys
import itertools
from utils_experiments import get_name_study, test_trial_done
import json
from main import *
import copy

default_arguments = vars(arguments)

folder_local_saves = "Saves_test_early_stopping2"
l_experiments = ["pretrained_test_imagenet_layer_wise_oct_24"]

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

if not(os.path.exists(folder_local_saves)):
    os.mkdir(folder_local_saves)
# else:
#     shutil.rmtree(folder_local_saves)
#     os.mkdir(folder_local_saves)
#%%
l_trials_values_remaining = []
l_names_remaining = []
l_names_study = []

for i in range(len(l_names)):
    trial_values = l_trials_values[i]
    name_args = l_names[i]
    args = {}
    for j in range(len(name_args)):
        args[name_args[j]] = trial_values[j]
    args_final = copy.deepcopy(default_arguments)
    args_final["learning_rate"] = args_final.pop("lr")
    for key in args:
        args_final[key] = args[key]
    
    name_study = get_name_study(**args_final)
    test_deja_train, n_trials_done, test_file_name = test_trial_done(name_study, args_final["n_trials"], args_final["n_repeat"], args_final["folder_saves"])
    if not(test_deja_train):
        l_trials_values_remaining.append(trial_values)
        l_names_remaining.append(name_args)
        #print(name_study, ",", n_trials_done)
        l_names_study.append(name_study)
    path_study = f"{args_final['folder_saves']}/study_{name_study}"
    command = f"scp -r gafriat@txe1-login.mit.edu:/home/gridsan/gafriat/projects/network_pruning/{path_study} /Users/afriatg/Desktop/Research/network_pruning/{folder_local_saves}"
    os.system(command)

print("Number of initial scripts = ", len(l_trials_values))
print("Number of remaining scripts = ", len(l_trials_values_remaining))
print("")
print("------------------")
print("Remaining scripts:")
print("------------------")

#%%
l_names_study = np.array(l_names_study)
l_names_unique = np.unique(l_names_study)
l_names_count = np.zeros(len(l_names_unique))
# %%
for i in range(len(l_names_unique)):
    l_names_count[i] += len(np.where(l_names_unique[i] == l_names_study)[0])
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

# %%
