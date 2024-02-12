#%%
import numpy as np
import os
import sys
import itertools
from utils_experiments import get_name_study, test_trial_done
import json
from main import *
import copy

to_run = True
default_arguments = vars(arguments)

supercloud_script = "/home/gridsan/gafriat/projects/network_pruning/main.py"

#l_experiments = ["pretrained_resnet20_layer_wise_block_vs_one_2"]
#l_experiments = ["pretrained_test_imagenet", "pretrained_test_imagenet_2"]
#l_experiments = ["pretrained_test_imagenet_layer_wise"]
#l_experiments = ["pretrained_test_imagenet_layer_wise_oct_3"]
#l_experiments = ["pretrained_test_imagenet_layer_wise_oct_6"]
#l_experiments = ["pretrained_test_imagenet_layer_wise_oct_7"]
#l_experiments = ["pretrained_test_imagenet_layer_wise_oct_17_part2", "pretrained_test_imagenet_layer_wise_oct_19_part2"]
#l_experiments = ["pretrained_test_imagenet_layer_wise_oct_20"]
#l_experiments = ["pretrained_test_imagenet_layer_wise_oct_23"]
#l_experiments = ["pretrained_test_imagenet_layer_wise_oct_24"]
#l_experiments = ["pretrained_test_imagenet_layer_wise_oct_30"]
#l_experiments = ["pretrained_test_imagenet_layer_wise_oct_32"]
#l_experiments = ["pretrained_test_imagenet_layer_wise_nov_6_1_bis"]
#l_experiments = ["pretrained_test_imagenet_layer_wise_nov_7"]
#l_experiments = ["pretrained_test_imagenet_layer_wise_nov_8"]
#l_experiments = ["pretrained_test_imagenet_layer_wise_nov_10"]
#l_experiments = ["pretrained_test_imagenet_layer_wise_nov_13_1", "pretrained_test_imagenet_layer_wise_nov_13_2"]
#l_experiments = ["pretrained_test_imagenet_layer_wise_nov_13_3"]
#l_experiments = ["pretrained_test_imagenet_layer_wise_nov_13_4"]
#l_experiments = ["pretrained_test_imagenet_layer_wise_nov_14"]
#l_experiments = ["pretrained_test_imagenet_layer_wise_nov_14_1"]
#l_experiments = ["pretrained_test_imagenet_layer_wise_nov_16_1", "pretrained_test_imagenet_layer_wise_nov_16_2"]
#l_experiments = ["pretrained_test_imagenet_layer_wise_nov_17_1", "pretrained_test_imagenet_layer_wise_nov_17_2"]
#l_experiments = ["pretrained_test_imagenet_layer_wise_nov_18"]
#l_experiments = ["pretrained_test_imagenet_layer_wise_nov_19_2"]
#l_experiments = ["pretrained_test_imagenet_layer_wise_nov_19_2"]
#l_experiments = ["pretrained_test_imagenet_layer_wise_nov_21_version_2_bis"]
#l_experiments = ["pretrained_test_imagenet_layer_wise_nov_23_3"]
#l_experiments = ["pretrained_test_imagenet_layer_wise_nov_23_4"]
# l_experiments = ["pretrained_test_imagenet_layer_wise_nov_27"]
#l_experiments = ["pretrained_layer_wise_nov_30", "pretrained_layer_wise_nov_30_2"]
#l_experiments = ["pretrained_layer_wise_dec_3"]
#l_experiments = ["pretrained_layer_wise_dec_3_2", "pretrained_layer_wise_dec_3_3"]
#l_experiments = ["pretrained_layer_wise_dec_5"]
#l_experiments = ["pretrained_layer_wise_dec_6"]
#l_experiments = ["pretrained_layer_wise_dec_6_1"]
#l_experiments = ["pretrained_layer_wise_dec_7_1", "pretrained_layer_wise_dec_7_2"]
l_experiments = ["pretrained_layer_wise_dec_7_2"]

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
        print(name_study, ",", n_trials_done)
        l_names_study.append(name_study)

#%%
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

l_trials_values = l_trials_values_remaining
l_names = l_names_remaining
#%%
if to_run:
    sys.path.append("./")
    my_task_id = int(sys.argv[1])
    num_tasks = int(sys.argv[2])

    sub_l_trials_values = l_trials_values[my_task_id:len(l_trials_values):num_tasks]
    sub_l_names = l_names[my_task_id:len(l_trials_values):num_tasks]

    for ind_trial in range(len(sub_l_trials_values)):
        trial_values = sub_l_trials_values[ind_trial]
        l_params_names = sub_l_names[ind_trial]
        hyper_param = ""
        for i in range(len(l_params_names)):
            name = l_params_names[i]
            hyper_param+=" --"+name+" "+str(trial_values[i])
            if name == "arch":
                if "opt" in str(trial_values[i]):
                    supercloud_env = "/home/gridsan/gafriat/.conda/envs/network_pruning/bin/python"
                else:
                    supercloud_env = "python"
        hyper_param = hyper_param.replace("learning_rate", "lr")
        print(supercloud_script+hyper_param)
        os.system(supercloud_env+" "+supercloud_script+hyper_param)

# %%
