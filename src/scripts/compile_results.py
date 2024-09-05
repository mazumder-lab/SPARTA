import copy
import os

import numpy as np
import pandas as pd
# import plotly.graph_objects as go


# EXPERIMENT_DIR = "final_results_resnet18"
# EXPERIMENT_DIR = "mm_final_results_tiny_cif10"
EXPERIMENT_DIR = "final_results_tiny_organ"
# EXPERIMENT_DIR = "final_results_tiny_organ_eps05"
# EXPERIMENT_DIR = "final_results_tiny_organ_eps75"


PATH_TO_RESULTS = "../results_folder/"

path_files = PATH_TO_RESULTS + EXPERIMENT_DIR + "/"
print(path_files)
l_columns = ["dataset", "model", "batch_size", "epsilon", "delta", "clipping", "warm_up", "classifier_lr", "lr", "sparsity", "epoch_mask_finding", "method_name", "use_last_layer_only_init", "seed", "acc_t0", "Test acc", "Time"]

l_results_test = []
l_time = []
l_hyperparameters = []

for file in os.listdir(path_files):
    with open(path_files + file, "rb") as f:
        lines = f.readlines()
    if "Time spent" not in str(lines[-1]):
        continue
    for ind_line in range(len(lines)):
        line = lines[ind_line]
        if "Namespace" in str(line):
            hyperparameters = {}
            for key_val in str(line).split("(")[1].split(")")[0].split(", "):
                key,val = key_val.split("=")
                hyperparameters[key]=val
            l_results_test.append([])
            l_hyperparameters.append(hyperparameters)
        if "test loss" in str(line):
            test_loss = float(str(line).split("test loss: ")[1].split(" and")[0])
            test_acc = float(str(line).split("accuracy: ")[1].split("\\")[0])
            l_results_test[-1].append((copy.deepcopy(test_loss), copy.deepcopy(test_acc)))
        if "Time spent" in str(line):
            time_spent = float(str(line).split("Time spent: ")[1].split(".Test accuracy:")[0])
            l_time.append(time_spent)
    hyperparameters["acc_t0"] = l_results_test[-1][int(hyperparameters["epoch_mask_finding"])][1] if hyperparameters["epoch_mask_finding"] != "-1" else 0


l_finished_epoch = np.array([len(results_test) for results_test in l_results_test])
l_final_results_test_acc = np.array([results_test[-1][1] for results_test in l_results_test])
l_hyperparameters_name = list(l_hyperparameters[0].keys())
l_time = np.array(l_time)


dict_results = {}
for hyperparameters_name in l_hyperparameters_name:
    dict_results[hyperparameters_name] = []

for hyperparameters in l_hyperparameters:
    for key in hyperparameters:
        if key in dict_results:
            dict_results[key].append(hyperparameters[key])

dict_results["Test acc"] = l_final_results_test_acc
dict_results["Finished epochs"] = l_finished_epoch
dict_results["Time"] = l_time

df_results = pd.DataFrame.from_dict(dict_results)
df_results = df_results[l_columns]

df_results = df_results.groupby(l_columns[:-4]).agg(
    {'Test acc': ['mean', 'std'], 'seed': 'count', 'Time': 'mean', 'acc_t0': 'mean'}).sort_values(
        by=('Test acc', 'mean'), ascending=False).reset_index()

df_results.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df_results.columns.values]
df_results.rename(columns={'Test acc_mean': 'Test acc', 'Test acc_std': 'acc_std'}, inplace=True)
df_results["std_err_mean"] = df_results["acc_std"] / np.sqrt(df_results["seed_count"])
df_results["Time_mean"] = df_results["Time_mean"] / 3600
grouped_df = df_results.loc[df_results.groupby(['method_name', 'sparsity', 'use_last_layer_only_init', 'epoch_mask_finding'])['Test acc'].idxmax()]


# df_results = df_results.groupby(l_columns[:-2]).agg(
#     {'Test acc': ['mean', 'std'], 'seed': 'count'}).sort_values(
#         by=('Test acc', 'mean'), ascending=False).reset_index()
    
# grouped_df = df_results.loc[df_results.groupby(['method_name', 'sparsity', 'epoch_mask_finding'])['Test acc mean'].idxmax()]

# TO plot best learning curves
# df_best = pd.read_csv('csv_res/res_global.csv')
# df_best = df_best[df_best['finetune_strategy']=="'all_layers'"]

# fig = go.Figure()


# for ind_exp in range(len(df_best)):
#     num_exp = df_best["Unnamed: 0"].iloc[ind_exp]
#     name_dataset = df_best['dataset'].iloc[ind_exp]
#     name_model = df_best['model'].iloc[ind_exp]
#     l_train_acc = np.array([x[1] for x in l_results_train[num_exp]])
#     fig.add_trace(go.Scatter(
#             #name='Measurement',
#             x=np.arange(len(l_train_acc)),
#             y=l_train_acc,
#             mode='lines',
#             # line=dict(color=color[0], width=4),
#             showlegend=True,
#             textfont_size=20,
#             name=f"{name_model}_{name_dataset}"
#         ))

# fig.write_html(f"plots.html")



# df_results = df_results[l_columns].groupby(l_columns[:-3]).agg({'Train acc': ['mean', 'mad'], 'Test acc': ['mean', 'mad']})
df_results.to_csv(f"{path_files}{EXPERIMENT_DIR}_csv_results.csv")
grouped_df.to_csv(f"{path_files}{EXPERIMENT_DIR}_csv_grouped_results.csv")
