import copy
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go

#l_logs = ["LLSUB.3521752"]#, "LLSUB.3521839"]
#l_logs = ["LLSUB.3521839"]
#l_logs = ["LLSUB.2593868"]
#l_logs = ["LLSUB.2086846"]
#l_logs = ["LLSUB.1101612"]
l_logs = ["LLSUB.1138433", "LLSUB.3480263"]

# With "use_new_version"
l_columns = ["dataset", "model", "batch_size", "classifier_lr", "lr", "finetune_strategy", "accum_steps", "epsilon", "delta", "clipping", "use_new_version", "Train acc", "Test acc", "seed"]

# Without "use_new_version"
# l_columns = ["dataset", "model", "batch_size", "classifier_lr", "lr", "finetune_strategy", "accum_steps", "epsilon", "delta", "clipping", "Train acc", "Test acc", "seed"]

l_results_train = []
l_results_test = []
l_hyperparameters = []
l_results_train_current = []
l_results_test_current = []

for log_name in l_logs:
    l_path_nodes = os.listdir(log_name)
    l_path_nodes = [x for x in l_path_nodes if x[0]=="p"]
    for path_node in l_path_nodes:
        current_path_to_node = "/".join([log_name, path_node])
        l_path_tasks = os.listdir(current_path_to_node)
        for path_tasks in l_path_tasks:
            current_path_to_task = "/".join([log_name, path_node, path_tasks])
            print(f"Reading {current_path_to_task}")
            with open(current_path_to_task, "rb") as f:
                lines = f.readlines()
            for ind_line in range(len(lines)):
                line = lines[ind_line]
                if "Namespace" in str(line) and "Commencing" in str(lines[ind_line+1]):
                    hyperparameters = {}
                    for key_val in str(line).split("(")[1].split(")")[0].split(", "):
                        key,val = key_val.split("=")
                        hyperparameters[key]=val
                    l_results_train.append([])
                    l_results_test.append([])
                    l_hyperparameters.append(hyperparameters)
                if "train loss" in str(line):
                    train_loss = float(str(line).split("train loss: ")[1].split(" and")[0])
                    train_acc = float(str(line).split("accuracy: ")[1].split("\\")[0])
                    # l_results_train_current.append((copy.deepcopy(train_loss), copy.deepcopy(train_acc)))
                    l_results_train[-1].append((copy.deepcopy(train_loss), copy.deepcopy(train_acc)))
                if "test loss" in str(line):
                    test_loss = float(str(line).split("test loss: ")[1].split(" and")[0])
                    test_acc = float(str(line).split("accuracy: ")[1].split("\\")[0])
                    # l_results_test_current.append((copy.deepcopy(test_loss), copy.deepcopy(test_acc)))
                    l_results_test[-1].append((copy.deepcopy(test_loss), copy.deepcopy(test_acc)))

l_finished_epoch = np.array([len(results_test) for results_test in l_results_test])
l_final_results_test_acc = np.array([results_test[-1][1] for results_test in l_results_test])
l_final_results_train_acc = np.array([results_train[-1][1] for results_train in l_results_train])
l_hyperparameters_name = list(l_hyperparameters[0].keys())

dict_results = {}
for hyperparameters_name in l_hyperparameters_name:
    dict_results[hyperparameters_name] = []

for hyperparameters in l_hyperparameters:
    for key in hyperparameters:
        dict_results[key].append(hyperparameters[key])

dict_results["Test acc"] = l_final_results_test_acc
dict_results["Train acc"] = l_final_results_train_acc
dict_results["Finished epochs"] = l_finished_epoch

df_results = pd.DataFrame.from_dict(dict_results).sort_values(by="Test acc", ascending=False)

def mean_plus_minus_mad(x):
    mean = x.mean()
    mad = x.mad()
    return mean, mad

import ipdb;ipdb.set_trace()

df_best = pd.read_csv('csv_res/res_global.csv')
df_best = df_best[df_best['finetune_strategy']=="'all_layers'"]

fig = go.Figure()

for ind_exp in range(len(df_best)):
    num_exp = df_best["Unnamed: 0"].iloc[ind_exp]
    name_dataset = df_best['dataset'].iloc[ind_exp]
    name_model = df_best['model'].iloc[ind_exp]
    l_train_acc = np.array([x[1] for x in l_results_train[num_exp]])
    fig.add_trace(go.Scatter(
            #name='Measurement',
            x=np.arange(len(l_train_acc)),
            y=l_train_acc,
            mode='lines',
            # line=dict(color=color[0], width=4),
            showlegend=True,
            textfont_size=20,
            name=f"{name_model}_{name_dataset}"
        ))

fig.write_html(f"plots.html")

#df_results.to_csv("res_tot.csv")

if True:
    df_results = df_results[l_columns].groupby(l_columns[:-3]).agg({'Train acc': ['mean', 'mad'], 'Test acc': ['mean', 'mad']})
    df_results.to_csv("results_comparison_2.csv")
else:
    #df_results.to_csv("results_comparison_private_vision_opacus.csv")
    df_results[l_columns[:-1]].to_csv("results_comparison_2.csv")
