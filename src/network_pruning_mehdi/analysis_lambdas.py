#%%
import torch
import plotly.graph_objects as go
import numpy as np
import os

n_convex = 5
rel_damp = 0.01
folder_saves_plots = f"Plots_OBC_{n_convex}_{rel_damp}"
if not(os.path.exists(folder_saves_plots)):
    os.mkdir(folder_saves_plots)

l_archs = ["deit_tiny_patch16_224", "deit_small_patch16_224"]
l_sparsity = [0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98]

for arch in l_archs:
    folder_arch = folder_saves_plots+"/"+arch
    if not(os.path.exists(folder_arch)):
        os.mkdir(folder_arch)
    for sparsity in l_sparsity:
        print(f"Plots for ({arch}, {sparsity}) ...", flush=True)
        name = '%s_%04d.pth' % (arch, int(sparsity * 10000))
        l_lambda_fisher = np.linspace(0, 1, n_convex)
        l_lambda_reconst = 1 - l_lambda_fisher
        l_convex_comb = [str((l_lambda_reconst[i], l_lambda_fisher[i])) for i in range(len(l_lambda_fisher))]

        #%%
        fig = go.Figure()
        weight_losses = torch.load(f"Saves_OBC_{n_convex}_{rel_damp}/losses/{name}")
        mat_losses = np.zeros((0, n_convex))
        l_keys = []
        for key in weight_losses:
            if ("norm" not in key) and ("blocks" in key) and ("embed" not in key) and ("bias" not in key):
                fig.add_trace(go.Scatter(x=l_convex_comb, y=weight_losses[key], name=key))
                mat_losses = np.vstack((mat_losses, weight_losses[key].numpy()))
                l_keys.append(key)
        fig.update_layout(
            title=f"Results for {arch}",
            xaxis_title="(lambda_reconst, lambda_fisher)",
            yaxis_title="Loss",
        )
        fig.write_html(f"{folder_arch}/lambdas_{arch}_{n_convex}_{sparsity}.html")

        # %%
        fig = go.Figure()
        for ind_convex_comb in range(len(l_convex_comb)):
            convex_comb = l_convex_comb[ind_convex_comb]
            l_losses = mat_losses[:,ind_convex_comb]/np.min(mat_losses, 1)
            fig.add_trace(go.Scatter(x=l_keys, y=l_losses, name=convex_comb, mode='markers'))

        fig.update_layout(
            title=f"Results for {arch} with sparisty = {sparsity}",
            xaxis_title="layer",
            yaxis_title="Normalized loss",
        )
        fig.write_html(f"{folder_arch}/lambdas_{arch}_{n_convex}_{sparsity}_2.html")
        print(f"Done", flush=True)

        # %%
