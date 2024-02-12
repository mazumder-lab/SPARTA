#%%
import numpy as np
import os
import sys
import itertools
from utils_experiments import get_name_study, test_trial_done
import json
from main import *
import copy
import shutil

def create_script(path_file, command_to_run):
    with open(path_file, "w") as f:
        f.write("#!/bin/bash")
        f.write("\n")
        f.write("\n")
        f.write("#SBATCH --partition=xeon-g6-volta")
        f.write("\n")
        f.write("#SBATCH --constraint=xeon-g6")
        f.write("\n")
        f.write("#SBATCH -o ../logs/output_run_%A_%a.txt #redirect output to output_JOBID.txt")
        f.write("\n")
        f.write("#SBATCH -e ../logs/error_run_%A_%a.txt #redirect errors to error_JOBID.txt")
        f.write("\n")
        f.write("#SBATCH --gres=gpu:volta:1")
        f.write("\n")
        f.write("#SBATCH --nodes=1")
        f.write("\n")
        f.write("#SBATCH --ntasks-per-node=1")
        f.write("\n")
        f.write("#SBATCH --cpus-per-task=20")
        f.write("\n")
        f.write("#SBATCH --array=0")
        f.write("\n")
        f.write("#SBATCH --mail-type=FAIL")
        f.write("\n")
        f.write("#SBATCH --mail-user=afriatg@mit.edu")
        f.write("\n")
        f.write("\n")
        f.write("source /etc/profile")
        f.write("\n")
        f.write("module purge")
        f.write("\n")
        f.write("module load anaconda/2022a")
        f.write("\n")
        f.write("module load /home/gridsan/groups/datasets/ImageNet/modulefile")
        f.write("\n")
        f.write("\n")
        f.write(command_to_run)
        f.write("\n")    

name_folder = "to_run_sbatch"
if not(os.path.exists(name_folder)):
        os.mkdir(name_folder)
else:
    shutil.rmtree(name_folder)
    os.mkdir(name_folder)

default_arguments = vars(arguments)

supercloud_env = "python"
#supercloud_env = "/home/gridsan/gafriat/.conda/envs/additive/bin/python"
supercloud_script = "/home/gridsan/gafriat/projects/network_pruning/main.py"

l_experiments = ["pretrained_resnet20_layer_wise_block_vs_one"]

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
for ind_trial in range(len(l_trials_values)):
    trial_values = l_trials_values[ind_trial]
    l_params_names = l_names[ind_trial]
    hyper_param = ""
    for i in range(len(l_params_names)):
        name = l_params_names[i]
        hyper_param+=" --"+name+" "+str(trial_values[i])
    hyper_param = hyper_param.replace("learning_rate", "lr")
    print(supercloud_script+hyper_param)
    path_file = f"{name_folder}/script_{ind_trial}.sh"
    create_script(path_file, supercloud_env+" "+supercloud_script+hyper_param)
    os.system("chmod u+x "+path_file)
    os.system("sbatch "+path_file)

# %%
