import copy
import os

import numpy as np
import pandas as pd
# import plotly.graph_objects as go


# EXPERIMENT_DIR = "to_benchmark_resnet18_all_methods_exp2/"
# EXPERIMENT_DIR = "to_benchmark_deit_tiny_all_methods_cif100_exp3"
EXPERIMENT_DIR = "to_benchmark_deit_tiny_all_methods_cif10_exp3"


PATH_TO_RESULTS = "../results_folder/"

path_files = PATH_TO_RESULTS + EXPERIMENT_DIR + "/"
print(path_files)
l_columns = ["dataset", "model", "batch_size", "epsilon", "delta", "clipping", "warm_up", "classifier_lr", "lr", "epoch_mask_finding", "method_name", "seed", "Test acc"]

l_results_test = []
l_hyperparameters = []
l_results_test_current = []

for file in os.listdir(path_files):
    with open(path_files + file, "rb") as f:
        lines = f.readlines()
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

l_finished_epoch = np.array([len(results_test) for results_test in l_results_test])
l_final_results_test_acc = np.array([results_test[-1][1] for results_test in l_results_test])
l_hyperparameters_name = list(l_hyperparameters[0].keys())

dict_results = {}
for hyperparameters_name in l_hyperparameters_name:
    dict_results[hyperparameters_name] = []

for hyperparameters in l_hyperparameters:
    for key in hyperparameters:
        dict_results[key].append(hyperparameters[key])

dict_results["Test acc"] = l_final_results_test_acc
dict_results["Finished epochs"] = l_finished_epoch

df_results = pd.DataFrame.from_dict(dict_results).sort_values(by="Test acc", ascending=False)
grouped_df = df_results.loc[df_results.groupby(['method_name', 'sparsity', 'epoch_mask_finding'])['Test acc'].idxmax()].sort_values(by='Test acc', ascending=False)

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
