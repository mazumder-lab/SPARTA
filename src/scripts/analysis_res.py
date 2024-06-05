# %%
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go

l_columns = [
    "dataset",
    "model",
    "batch_size",
    "sparsity",
    "finetune_strategy",
    "epsilon",
    "delta",
    "use_fixed_w_mask_finding",
    "use_delta_weight_optim",
    "use_w_tilde",
    "classifier_lr",
    "lr",
    "clipping",
    "Train acc",
    "Test acc",
    "Finished epochs",
]

# date = "_april_31"
date = "_may_13"
# type_method = "_clipped_grads"
type_method = "_row_pruning_noisy_grads"

# Read all the exp
res_tot = pd.read_csv(f"res_tot{date}.csv", index_col=0)

# Remove unfinsihed ones
res_tot = res_tot[res_tot["Finished epochs"] >= 0]

# Create a table per dataset, network and type of training
l_models = np.unique(res_tot["model"])
l_dataset = np.unique(res_tot["dataset"])
# l_finetune_strategy = ["'all_layers'", "'lp_gn'"]
l_finetune_strategy = np.unique(res_tot["finetune_strategy"])
try:
    l_sparsity = np.unique(res_tot["sparsity"])
    test_sparsity = True
except:
    test_sparsity = False

df_best_res = pd.DataFrame([])
path_save = f"csv_res{date}"

if not (os.path.exists(path_save)):
    os.mkdir(path_save)

l_finished_exp = []
for model in l_models:
    for dataset in l_dataset:
        for finetune_strategy in l_finetune_strategy:
            if test_sparsity:
                l_sparsity_actual = l_sparsity
            else:
                l_sparsity_actual = [None]
            for sparsity in l_sparsity_actual:
                condition = (
                    (res_tot["model"] == model)
                    * (res_tot["dataset"] == dataset)
                    * (res_tot["finetune_strategy"] == finetune_strategy)
                )
                to_add = ""
                if test_sparsity:
                    condition = condition * (res_tot["sparsity"] == sparsity)
                    to_add += f"_{int(100*sparsity)}"
                res_model = res_tot[condition]
                res_model[l_columns].sort_values(by="Test acc", ascending=False).to_csv(
                    f"""{path_save}/res_{model.replace("'","")}_{dataset.replace("'","")}_{finetune_strategy.replace("'","")}{to_add}.csv"""
                )
                try:
                    df_best_res = pd.concat([df_best_res, res_model.iloc[[0]]])
                    l_finished_exp.append(len(res_model))
                except:
                    print(f"No result for {model}, {dataset}, sparsity={sparsity}")
# %%
if not (os.path.exists("res_global")):
    os.mkdir("res_global")

df_best_res["Finished exp"] = l_finished_exp
df_best_res[l_columns + ["Finished exp"]].sort_values(
    by="Test acc", ascending=False
).to_csv(f"""res_global/res_global{type_method}.csv""")

# %%
df_clipped_grad = pd.read_csv("""res_global/res_global_clipped_grads.csv""")
df_row_pruning = pd.read_csv("""res_global/res_global_row_pruning_noisy_grads.csv""")

# %%
l_models = np.unique(df_clipped_grad["model"])
l_dataset = np.unique(df_clipped_grad["dataset"])
for model in l_models:
    for dataset in l_dataset:
        fig = go.Figure()
        if "deit_tiny" in model:
            if dataset == "'cifar10'":
                acc_last_layer = 85.39
                acc_all_layers = 89.06
            elif dataset == "'cifar100'":
                acc_last_layer = 57.13
                acc_all_layers = 50.55
        elif "deit_small" in model:
            if dataset == "'cifar10'":
                acc_last_layer = 89.92
                acc_all_layers = 93.99
            elif dataset == "'cifar100'":
                acc_last_layer = 63.06
                acc_all_layers = 62.25

        fig.add_trace(
            go.Scatter(
                # name='Measurement',
                x=[1.0],
                y=[acc_all_layers],
                mode="markers",
                marker=dict(size=10, symbol="cross"),
                # line=dict(color=color[0], width=4),
                showlegend=True,
                textfont_size=20,
                name="all layers",
            )
        )

        fig.add_trace(
            go.Scatter(
                # name='Measurement',
                x=[0.0],
                y=[acc_last_layer],
                mode="markers",
                marker=dict(size=10, symbol="x"),
                # line=dict(color=color[0], width=4),
                showlegend=True,
                textfont_size=20,
                name="last layer",
            )
        )

        for type_method in ["_clipped_grads", "_row_pruning_noisy_grads"]:
            df_best_res = pd.read_csv(f"""res_global/res_global{type_method}.csv""")
            l_sparsity = np.array(
                df_best_res[
                    (df_best_res["dataset"] == dataset)
                    * (df_best_res["model"] == model)
                ]["sparsity"]
            )
            l_test_acc = np.array(
                df_best_res[
                    (df_best_res["dataset"] == dataset)
                    * (df_best_res["model"] == model)
                ]["Test acc"]
            )

            l_test_acc = l_test_acc[np.argsort(l_sparsity)]
            l_sparsity = l_sparsity[np.argsort(l_sparsity)]

            fig.add_trace(
                go.Scatter(
                    # name='Measurement',
                    x=l_sparsity,
                    y=l_test_acc,
                    mode="lines",
                    line=dict(width=4),
                    showlegend=True,
                    textfont_size=20,
                    name=type_method[1:].replace("_", " "),
                )
            )

        fig.write_html(f"res_global/plots_{model}_{dataset}{date}.html")

    # %%
