#%%
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from previous_utils.main_utils import get_model
import torch
from tqdm import tqdm

# %%
#l_arch = ["facebook/opt-125m"]#, "facebook/opt-350m"]
#l_arch = ["resnet50"]
l_arch = ["deit_tiny_patch16_224", "deit_small_patch16_224"]#, "deit_base_patch16_224"]
l_sparsity = [0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98]
#l_sparsity = [0.5]
#l_sparsity = [0.5, 0.6, 0.7]
#metric_name = "acc"
metric_name = "ppl"
n_convex = 5

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_sparsities(model):
    l_results = [(x[0], (x[1]==0).float().mean().item()) for x in model.named_parameters() if "norm" not in x[0] and "bias" not in x[0] and "blocks" in x[0]]
    l_layers = [x[0] for x in l_results]
    l_sparsities = [x[1] for x in l_results]
    return l_layers, l_sparsities

if not(os.path.exists("sparsity_levels")):
    os.mkdir("sparsity_levels")

#%%
for arch in l_arch:

    l_convex_non_uniform = []
    l_fisher_non_uniform = []
    l_reconst_non_uniform = []

    l_CAP_non_uniform = []
    l_convex_uniform_multiple = []
    l_convex_non_uniform_multiple = []

    l_convex_uniform_block = []
    l_fisher_uniform_block = []
    l_reconst_uniform_block = []
    l_convex_uniform_multiple_block = []


    for goal_sparisty in tqdm(l_sparsity):
        model, criterion, modules_to_prune = get_model(arch, 0, pretrained=True, with_z=False, gamma=1.0, prune_bias=False, activation_fn="relu")
        model.to(device)
        l_layers, _ = get_sparsities(model)

        path_convex_non_uniform = f"Saves_OBC_1.0_1.0_0.01/results/{arch}_unstr_{int(100*goal_sparisty)}x_dp.pth"
        path_fisher_non_uniform = f"Saves_OBC_0.0_1.0_0.01/results/{arch}_unstr_{int(100*goal_sparisty)}x_dp.pth"
        path_reconst_non_uniform = f"Saves_OBC_1.0_0.0_0.01/results/{arch}_unstr_{int(100*goal_sparisty)}x_dp.pth"

        path_CAP_non_uniform = f"/home/gridsan/gafriat/projects/CAP-main/output/one-shot/{arch}_sparsity={goal_sparisty}.pth"

        path_convex_non_uniform_multiple = f"Saves_OBC_{n_convex}_0.01/results/{arch}_unstr_{int(100*goal_sparisty)}x_dp.pth"

        path_convex_uniform_block = f"Saves_OBC_1.0_1.0_0.01_block/models_unstr/{arch}_{int(10000*goal_sparisty)}.pth"
        path_fisher_uniform_block = f"Saves_OBC_0.0_1.0_0.01_block/models_unstr/{arch}_{int(10000*goal_sparisty)}.pth"
        path_reconst_uniform_block = f"Saves_OBC_1.0_0.0_0.01_block/models_unstr/{arch}_{int(10000*goal_sparisty)}.pth"
        path_convex_uniform_multiple_block = f"Saves_OBC_{n_convex}_0.01_block/models_unstr/{arch}_{int(10000*goal_sparisty)}.pth"

        if os.path.exists(path_convex_non_uniform):
            model.load_state_dict(torch.load(path_convex_non_uniform, map_location=device))
            l_convex_non_uniform = get_sparsities(model)[1]
        else:
            l_convex_non_uniform = [np.nan]*len(l_layers)
        if os.path.exists(path_fisher_non_uniform):
            model.load_state_dict(torch.load(path_fisher_non_uniform, map_location=device))
            l_fisher_non_uniform = get_sparsities(model)[1]
        else:
            l_fisher_non_uniform = [np.nan]*len(l_layers)
        if os.path.exists(path_reconst_non_uniform):
            model.load_state_dict(torch.load(path_reconst_non_uniform, map_location=device))
            l_reconst_non_uniform = get_sparsities(model)[1]
        else:
            l_reconst_non_uniform = [np.nan]*len(l_layers)
        if os.path.exists(path_CAP_non_uniform):
            model.load_state_dict(torch.load(path_CAP_non_uniform, map_location=device))
            l_CAP_non_uniform = get_sparsities(model)[1]
        else:
            l_CAP_non_uniform = [np.nan]*len(l_layers)
        if os.path.exists(path_convex_non_uniform_multiple):
            model.load_state_dict(torch.load(path_convex_non_uniform_multiple, map_location=device))
            l_convex_non_uniform_multiple = get_sparsities(model)[1]
        else:
            l_convex_non_uniform_multiple = [np.nan]*len(l_layers)
        if os.path.exists(path_convex_uniform_block):
            model.load_state_dict(torch.load(path_convex_uniform_block, map_location=device))
            l_convex_uniform_block = get_sparsities(model)[1]
        else:
            l_convex_uniform_block = [np.nan]*len(l_layers)
        if os.path.exists(path_fisher_uniform_block):
            model.load_state_dict(torch.load(path_fisher_uniform_block, map_location=device))
            l_fisher_uniform_block = get_sparsities(model)[1]
        else:
            l_fisher_uniform_block = [np.nan]*len(l_layers)
        if os.path.exists(path_reconst_uniform_block):
            model.load_state_dict(torch.load(path_reconst_uniform_block, map_location=device))
            l_reconst_uniform_block = get_sparsities(model)[1]
        else:
            l_reconst_uniform_block = [np.nan]*len(l_layers)
        if os.path.exists(path_convex_uniform_multiple_block):
            model.load_state_dict(torch.load(path_convex_uniform_multiple_block, map_location=device))
            l_convex_uniform_multiple_block = get_sparsities(model)[1]
        else:
            l_convex_uniform_multiple_block = [np.nan]*len(l_layers)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=l_layers, y=l_convex_non_uniform_multiple, name=f"OBC Convex Non-Uniform - {n_convex}"))
        fig.add_trace(go.Bar(x=l_layers, y=l_convex_non_uniform, name="OBC Convex Non-Uniform"))
        fig.add_trace(go.Bar(x=l_layers, y=l_fisher_non_uniform, name="OBC Fisher Non-Uniform"))
        fig.add_trace(go.Bar(x=l_layers, y=l_reconst_non_uniform, name="OBC Non-Uniform"))

        fig.add_trace(go.Bar(x=l_layers, y=l_convex_uniform_multiple, name=f"OBC Convex Uniform - {n_convex}"))
        fig.add_trace(go.Bar(x=l_layers, y=l_convex_uniform_multiple_block, name=f"OBC Convex Uniform Block - {n_convex}"))

        fig.add_trace(go.Bar(x=l_layers, y=l_convex_uniform_block, name="OBC Convex Uniform Block"))
        fig.add_trace(go.Bar(x=l_layers, y=l_fisher_uniform_block, name="OBC Fisher Uniform Block"))
        fig.add_trace(go.Bar(x=l_layers, y=l_reconst_uniform_block, name="OBC Uniform Block"))

        fig.add_trace(go.Bar(x=l_layers, y=l_CAP_non_uniform, name="CAP Non-Uniform"))

        fig.update_layout(
            title=f"Results for {arch} and sparsity = {goal_sparisty}",
            xaxis_title="Layer",
            yaxis_title="Sparsity",
        )
        fig.write_html(f"sparsity_levels/{arch}_{goal_sparisty}.html")

# %%
