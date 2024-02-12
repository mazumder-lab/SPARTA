#%%
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go

name_results = "results_layer_wise_jan_10_2024"

# %%
#l_arch = ["facebook/opt-125m"]#, "facebook/opt-350m"]
#l_arch = ["resnet50"]
l_arch = ["deit_small_patch16_224"]
l_sparsity = [0.5, 0.6, 0.7, 0.8, 0.9]
#l_sparsity = [0.5]
#l_sparsity = [0.5, 0.6, 0.7]
#metric_name = "acc"
metric_name = "ppl"

# def get_n_params_dense(arch):
#     if arch == "mobilenetv1":
#         return 4210088
#     elif arch == "resnet50":
#         return 25503912

for arch in l_arch:
    arch = arch.replace("facebook/", "")
    for goal_sparisty in l_sparsity:
        path = f"{name_results}/study_{arch}_{goal_sparisty}_{metric_name}/data_total.csv"
        df = pd.read_csv(path, index_col=0)
        # df["Actual sparsity"] = 1-df["Number of z left"]/get_n_params_dense(arch)
        #l_keys  = list(df.keys())
        #df[["test_early_stopping", "patience", "retrain_last_block", "type_decay", "Train acc", "Validation acc", "Test acc", "Sparsity", "Time training", "Best epoch"]].to_csv(name_results+f"/table_{arch}_{goal_sparisty}.csv")
        #df[["test_early_stopping", "Learning rate", "pruning_rate_cte", f"Train {metric_name}", f"Validation {metric_name}", f"Test {metric_name}", "Sparsity", "Time training", "Best epoch"]].to_csv(name_results+f"/table_{arch}_{goal_sparisty}.csv")
        #df[["test_normalized_sgd", "pruning_rate_cte", "lambda_loss", f"Train {metric_name}", f"Validation {metric_name}", f"Test {metric_name}", "Sparsity", "Time training", "Best epoch"]].to_csv(name_results+f"/table_{arch}_{goal_sparisty}.csv")
        #df[["threshold_weights", "Learning rate", f"Train {metric_name}", f"Validation {metric_name}", f"Test {metric_name}", "Sparsity", "Time training", "Best epoch"]].to_csv(name_results+f"/table_{arch}_{goal_sparisty}.csv")
        #df[["Learning rate", "test_different_lr", "val_second_lr", "test_normalized_sgd", f"Train {metric_name}", f"Validation {metric_name}", f"Test {metric_name}", "Sparsity", "Time training", "Best epoch"]].to_csv(name_results+f"/table_{arch}_{goal_sparisty}.csv")
        #df[["test_start_sparse_gpt", f"Train {metric_name}", f"Validation {metric_name}", f"Test {metric_name}", "Sparsity", "Time training", "Best epoch"]].to_csv(name_results+f"/table_{arch}_{goal_sparisty}.csv")
        #df[["Learning rate", "patience_adaptive_lr", "patience_freeze", f"Train {metric_name}", f"Validation {metric_name}", f"Test {metric_name}", "Sparsity", "Time training", "Best epoch"]].to_csv(name_results+f"/table_{arch}_{goal_sparisty}.csv")
        #df[["patience_freeze", "test_start_sparse_gpt", "n_epochs", f"Train {metric_name}", f"Validation {metric_name}", f"Test {metric_name}", "Sparsity", "Time training", "Best epoch"]].to_csv(name_results+f"/table_{arch}_{goal_sparisty}.csv")
        #df[["learning_rate", "type_decay", "test_adaptive_lr", "n_incr_gradual_pruning", "type_pruning_schedule", f"Train {metric_name}", f"Validation {metric_name}", f"Test {metric_name}", "Sparsity", "Time training", "Best epoch"]].sort_values(by=f"Test {metric_name}").to_csv(name_results+f"/table_{arch}_{goal_sparisty}.csv")
        #df[["learning_rate", f"Train {metric_name}", f"Validation {metric_name}", f"Test {metric_name}", "Sparsity", "Time training", "Best epoch"]].sort_values(by=f"Test {metric_name}").to_csv(name_results+f"/table_{arch}_{goal_sparisty}.csv")
        #df[["goal_sparsity", "test_start_sparse_gpt", "test_start_obc", "test_almost_sequential", "test_early_stopping", "rel_damp", f"Train {metric_name}", f"Validation {metric_name}", f"Test {metric_name}", "Sparsity", "Time training", "Best epoch"]].sort_values(by=f"Test {metric_name}", ascending=False).to_csv(name_results+f"/table_{arch}_{goal_sparisty}.csv")
        #df[["goal_sparsity", "test_almost_sequential", "rel_damp", f"Train {metric_name}", f"Validation {metric_name}", f"Test {metric_name}", "Sparsity", "Time training", "Best epoch"]].sort_values(by=f"Test {metric_name}", ascending=False).to_csv(name_results+f"/table_{arch}_{goal_sparisty}.csv")
        #df[["goal_sparsity", "test_almost_sequential", "rel_damp", "lambda_fisher", f"Train {metric_name}", f"Validation {metric_name}", f"Test {metric_name}", "Sparsity", "Time training", "Best epoch"]].sort_values(by=f"Test {metric_name}", ascending=False).to_csv(name_results+f"/table_{arch}_{goal_sparisty}.csv")
        df[["goal_sparsity", "rel_damp", "lambda_reconst", "lambda_fisher", f"Train {metric_name}", f"Validation {metric_name}", f"Test {metric_name}", "Sparsity", "Time training", "Best epoch"]].sort_values(by=f"Test {metric_name}", ascending=False).to_csv(name_results+f"/table_{arch}_{goal_sparisty}.csv")

#%%
for arch in l_arch:
    l_convex = []
    l_fisher = []
    l_reconst = []

    for goal_sparisty in l_sparsity:
        df = pd.read_csv(name_results+f"/table_{arch}_{goal_sparisty}.csv", index_col=0)
        #best_convex = df[(df["lambda_reconst"]!=0)*(df["lambda_fisher"]!=0)].iloc[0]["Test ppl"]
        best_convex = df[(df["lambda_reconst"]==1.0)*(df["lambda_fisher"]==1.0)].iloc[0]["Test ppl"]
        best_fisher = df[(df["lambda_reconst"]==0)*(df["lambda_fisher"]!=0)].iloc[0]["Test ppl"]
        best_reconst = df[(df["lambda_reconst"]!=0)*(df["lambda_fisher"]==0)].iloc[0]["Test ppl"]
        l_convex.append(best_convex)
        l_fisher.append(best_fisher)
        l_reconst.append(best_reconst)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=l_sparsity, y=l_convex, name="OBC Convex"))
    fig.add_trace(go.Scatter(x=l_sparsity, y=l_fisher, name="OBC Fisher"))
    fig.add_trace(go.Scatter(x=l_sparsity, y=l_reconst, name="OBC"))
    fig.update_layout(
        title=f"Results for {arch}",
        xaxis_title="Sparsity",
        yaxis_title="Accuracy",
    )
    fig.write_html(f"comparison_{arch}.html")

# %%
# folder_saves = "Saves_test_resnet50_mobilenetv1_oct_6"
# metric_name = "accuracy"
# l_dir = [x for x in os.listdir(folder_saves) if "."!=x[0]]

# for path_study in l_dir:
#     path_l_sparsity = folder_saves+"/"+path_study+"/best_trial/repeat_0/l_sparsity.npy"
#     path_l_sparsity_storage = folder_saves+"/"+path_study+"/best_trial/repeat_0/l_sparsity_storage.npy"
#     path_l_in_sample_loss = folder_saves+"/"+path_study+"/best_trial/repeat_0/l_in_sample_loss.npy"
#     path_l_in_sample_metric = folder_saves+"/"+path_study+"/best_trial/repeat_0/l_in_sample_metric.npy"
#     path_l_lr = folder_saves+"/"+path_study+"/best_trial/repeat_0/l_lr.npy"
#     path_l_n_z = folder_saves+"/"+path_study+"/best_trial/repeat_0/l_n_z.npy"
#     path_l_times_epochs = folder_saves+"/"+path_study+"/best_trial/repeat_0/l_times_epochs.npy"
#     path_l_validation_loss = folder_saves+"/"+path_study+"/best_trial/repeat_0/l_validation_loss.npy"
#     path_l_validation_metric = folder_saves+"/"+path_study+"/best_trial/repeat_0/l_validation_metric.npy"

#     l_n_z = np.load(path_l_n_z)
#     l_lr = np.load(path_l_lr)
#     start_module = np.where(l_lr==np.max(l_lr))[0]
#     start_module = np.hstack([start_module, np.array([len(l_lr)])])
#     l_n_params_dense = np.ones(len(l_lr))
#     for i in range(len(start_module)-1):
#         l_n_params_dense[start_module[i]:start_module[i+1]] *= l_n_z[start_module[i]]
#     new_l_sparsity = 1-l_n_z/l_n_params_dense
#     dict_list = {}
#     dict_list["l_in_sample_loss/In-sample loss"] = np.load(path_l_in_sample_loss)
#     dict_list["l_in_sample_metric/In-sample "+metric_name] = np.load(path_l_in_sample_metric)
#     dict_list["l_validation_loss/Validation loss"] = np.load(path_l_validation_loss)
#     dict_list["l_validation_metric/Validation "+metric_name] = np.load(path_l_validation_metric)
#     dict_list["l_times_epochs/Time per epoch"] = np.load(path_l_times_epochs)
#     dict_list["l_lr/Learning rate"] = np.load(path_l_lr)
#     dict_list["l_n_z/Number of z"] = np.load(path_l_n_z)
#     dict_list["l_sparsity/Sparsity"] = new_l_sparsity

#     fig = go.Figure()
#     for key_list in dict_list:
#             l_names = key_list.split("/")
#             name_save = l_names[0]
#             name_plot = l_names[1]
#             fig.add_trace(go.Scatter(x=np.arange(len(dict_list[key_list])), y=dict_list[key_list], name=name_plot))
#             if name_plot=="Learning rate":
#                 fig.add_trace(go.Scatter(x=np.arange(len(dict_list[key_list])), y=2/dict_list[key_list], name="2/lr"))

#     fig.update_layout(title="Summary for "+path_study[6:])
#     fig.write_html(folder_saves+"/"+path_study+"/best_trial/repeat_0/summary.html")

# # %%

# %%
