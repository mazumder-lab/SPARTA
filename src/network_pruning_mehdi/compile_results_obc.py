#%%
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# %%
#l_arch = ["facebook/opt-125m"]#, "facebook/opt-350m"]
#l_arch = ["resnet50"]
l_arch = ["deit_tiny_patch16_224"]#, "deit_small_patch16_224", "deit_base_patch16_224"]
l_sparsity = [0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98]
#l_sparsity = [0.5]
#l_sparsity = [0.5, 0.6, 0.7]
#metric_name = "acc"
metric_name = "ppl"
n_convex = 5

#%%
for arch in l_arch:
    l_convex_uniform = []
    l_fisher_uniform = []
    l_reconst_uniform = []

    l_convex_non_uniform = []
    l_fisher_non_uniform = []
    l_reconst_non_uniform = []

    l_CAP_non_uniform = []
    l_convex_uniform_multiple = []
    l_convex_non_uniform_multiple = []
    l_convex_non_uniform_multiple_greedy = []

    l_convex_uniform_block_2 = []
    l_convex_uniform_block = []
    l_fisher_uniform_block = []
    l_reconst_uniform_block = []
    l_convex_uniform_multiple_block = []

    d_convex_uniform_greedy = {}
    d_convex_uniform_cd = {}
    for (number_of_runs, initial_sparsity) in [(3, 0.0)]:
        for lambda_sparsity in [0.001, 0.005, 0.01, 0.05]:
            file = [x for x in os.listdir("Saves_OBC_1.0_1.0_0.01") if f"results_cd_{lambda_sparsity}_" in x]
            if (number_of_runs, initial_sparsity) not in d_convex_uniform_cd:
                d_convex_uniform_cd[(number_of_runs, initial_sparsity)] = ([],[])
            if len(file)>0:
                file = file[0]
                sparisty = int(file.split(f"results_cd_{lambda_sparsity}_")[1])
                path_convex_uniform_cd = f"Saves_OBC_1.0_1.0_0.01/results_cd_{lambda_sparsity}_{sparisty}"
                if initial_sparsity != 0:
                    path_convex_uniform_cd+= f"_{initial_sparsity}"
                path_convex_uniform_cd+=f"/{arch}_unstr_{number_of_runs}x_cd.txt"
                if os.path.exists(path_convex_uniform_cd):
                    d_convex_uniform_cd[(number_of_runs, initial_sparsity)][0].append(sparisty/100)
                    d_convex_uniform_cd[(number_of_runs, initial_sparsity)][1].append(float(open(path_convex_uniform_cd, "r").read().rstrip()))

    for goal_sparisty in l_sparsity:
        for (max_sparsity_start, no_approximation, test_sparsities, initial_sparsity) in [(-1, 0, 0, 0.2), (-1, 0, 0, 0.4), (-1, 0, 0, 0.5)]:
            path_convex_uniform_convex = f"Saves_OBC_1.0_1.0_0.01/results_greedy_{max_sparsity_start}_{no_approximation}_{test_sparsities}_{initial_sparsity}/{arch}_unstr_{int(100*goal_sparisty)}x_greedy.txt"
            if (max_sparsity_start, no_approximation, test_sparsities, initial_sparsity) not in d_convex_uniform_greedy:
                d_convex_uniform_greedy[(max_sparsity_start, no_approximation, test_sparsities, initial_sparsity)] = []
            if os.path.exists(path_convex_uniform_convex):
                d_convex_uniform_greedy[(max_sparsity_start, no_approximation, test_sparsities, initial_sparsity)].append(float(open(path_convex_uniform_convex, "r").read().rstrip()))
            else:
                d_convex_uniform_greedy[(max_sparsity_start, no_approximation, test_sparsities, initial_sparsity)].append(np.nan)

        path_convex_uniform = f"Saves_OBC_1.0_1.0_0.01/results/{arch}_unstr_{int(100*goal_sparisty)}x_uniform.txt"
        path_fisher_uniform = f"Saves_OBC_0.0_1.0_0.01/results/{arch}_unstr_{int(100*goal_sparisty)}x_uniform.txt"
        path_reconst_uniform = f"Saves_OBC_1.0_0.0_0.01/results/{arch}_unstr_{int(100*goal_sparisty)}x_uniform.txt"

        path_convex_non_uniform = f"Saves_OBC_1.0_1.0_0.01/results/{arch}_unstr_{int(100*goal_sparisty)}x_dp.txt"
        path_fisher_non_uniform = f"Saves_OBC_0.0_1.0_0.01/results/{arch}_unstr_{int(100*goal_sparisty)}x_dp.txt"
        path_reconst_non_uniform = f"Saves_OBC_1.0_0.0_0.01/results/{arch}_unstr_{int(100*goal_sparisty)}x_dp.txt"

        path_CAP_non_uniform = f"/home/gridsan/gafriat/projects/CAP-main/output/one-shot/{arch}_sparsity={goal_sparisty}.txt"

        path_convex_uniform_multiple = f"Saves_OBC_{n_convex}_0.01/results/{arch}_unstr_{int(100*goal_sparisty)}x_uniform.txt"
        path_convex_non_uniform_multiple = f"Saves_OBC_{n_convex}_0.01/results/{arch}_unstr_{int(100*goal_sparisty)}x_dp.txt"
        path_convex_non_uniform_multiple_greedy = f"Saves_OBC_{n_convex}_0.01/results/{arch}_unstr_{int(100*goal_sparisty)}x_greedy.txt"

        path_convex_uniform_block_2 = f"Saves_OBC_1.0_1.0_0.01_block_2/results/{arch}_unstr_{int(100*goal_sparisty)}x_uniform.txt"
        path_convex_uniform_block = f"Saves_OBC_1.0_1.0_0.01_block/results/{arch}_unstr_{int(100*goal_sparisty)}x_uniform.txt"
        path_fisher_uniform_block = f"Saves_OBC_0.0_1.0_0.01_block/results/{arch}_unstr_{int(100*goal_sparisty)}x_uniform.txt"
        path_reconst_uniform_block = f"Saves_OBC_1.0_0.0_0.01_block/results/{arch}_unstr_{int(100*goal_sparisty)}x_uniform.txt"
        path_convex_uniform_multiple_block = f"Saves_OBC_{n_convex}_0.01_block/results/{arch}_unstr_{int(100*goal_sparisty)}x_uniform.txt"


        l_convex_uniform.append(float(open(path_convex_uniform, "r").read().rstrip()))
        l_fisher_uniform.append(float(open(path_fisher_uniform, "r").read().rstrip()))
        l_reconst_uniform.append(float(open(path_reconst_uniform, "r").read().rstrip()))
        l_convex_non_uniform.append(float(open(path_convex_non_uniform, "r").read().rstrip()))
        l_fisher_non_uniform.append(float(open(path_fisher_non_uniform, "r").read().rstrip()))
        l_reconst_non_uniform.append(float(open(path_reconst_non_uniform, "r").read().rstrip()))

        if os.path.exists(path_CAP_non_uniform):
            l_CAP_non_uniform.append(float(open(path_CAP_non_uniform, "r").read().rstrip()))
        else:
            l_CAP_non_uniform.append(np.nan)
        if os.path.exists(path_convex_uniform_multiple):
            l_convex_uniform_multiple.append(float(open(path_convex_uniform_multiple, "r").read().rstrip()))
        else:
            l_convex_uniform_multiple.append(np.nan)
        if os.path.exists(path_convex_non_uniform_multiple):
            l_convex_non_uniform_multiple.append(float(open(path_convex_non_uniform_multiple, "r").read().rstrip()))
        else:
            l_convex_non_uniform_multiple.append(np.nan)
        if os.path.exists(path_convex_non_uniform_multiple_greedy):
            l_convex_non_uniform_multiple_greedy.append(float(open(path_convex_non_uniform_multiple_greedy, "r").read().rstrip()))
        else:
            l_convex_non_uniform_multiple_greedy.append(np.nan)


        if os.path.exists(path_convex_uniform_block_2):
            l_convex_uniform_block_2.append(float(open(path_convex_uniform_block_2, "r").read().rstrip()))
        else:
            l_convex_uniform_block_2.append(np.nan)
        if os.path.exists(path_convex_uniform_block):
            l_convex_uniform_block.append(float(open(path_convex_uniform_block, "r").read().rstrip()))
        else:
            l_convex_uniform_block.append(np.nan)
        if os.path.exists(path_fisher_uniform_block):
            l_fisher_uniform_block.append(float(open(path_fisher_uniform_block, "r").read().rstrip()))
        else:
            l_fisher_uniform_block.append(np.nan)
        if os.path.exists(path_reconst_uniform_block):
            l_reconst_uniform_block.append(float(open(path_reconst_uniform_block, "r").read().rstrip()))
        else:
            l_reconst_uniform_block.append(np.nan)
        if os.path.exists(path_convex_uniform_multiple_block):
            l_convex_uniform_multiple_block.append(float(open(path_convex_uniform_multiple_block, "r").read().rstrip()))
        else:
            l_convex_uniform_multiple_block.append(np.nan)

    fig = go.Figure()

    l_best_greedy = np.array([np.nan]*len(l_sparsity))
    for key in d_convex_uniform_greedy:
        fig.add_trace(go.Scatter(x=l_sparsity, y=d_convex_uniform_greedy[key], name=f"OBC Convex Non-Uniform Fast Greedy - {key}"))
        l_best_greedy = np.nanmax(np.stack([l_best_greedy, d_convex_uniform_greedy[key]]), 0)
    for key in d_convex_uniform_cd:
        fig.add_trace(go.Scatter(x=d_convex_uniform_cd[key][0], y=d_convex_uniform_cd[key][1], name=f"OBC Convex Non-Uniform CD - {key}"))

    fig.add_trace(go.Scatter(x=l_sparsity, y=l_best_greedy, name=f"OBC Convex Non-Uniform best Greedy"))
    fig.add_trace(go.Scatter(x=l_sparsity, y=l_convex_non_uniform_multiple_greedy, name=f"OBC Convex Non-Uniform - {n_convex} Greedy - "))
    fig.add_trace(go.Scatter(x=l_sparsity, y=l_convex_non_uniform_multiple, name=f"OBC Convex Non-Uniform - {n_convex}"))
    fig.add_trace(go.Scatter(x=l_sparsity, y=l_convex_non_uniform, name="OBC Convex Non-Uniform"))
    fig.add_trace(go.Scatter(x=l_sparsity, y=l_fisher_non_uniform, name="OBC Fisher Non-Uniform"))
    fig.add_trace(go.Scatter(x=l_sparsity, y=l_reconst_non_uniform, name="OBC Non-Uniform"))

    fig.add_trace(go.Scatter(x=l_sparsity, y=l_convex_uniform_multiple, name=f"OBC Convex Uniform - {n_convex}", line = dict(dash='dot')))
    fig.add_trace(go.Scatter(x=l_sparsity, y=l_convex_uniform, name="OBC Convex Uniform", line = dict(dash='dot')))
    fig.add_trace(go.Scatter(x=l_sparsity, y=l_fisher_uniform, name="OBC Fisher Uniform", line = dict(dash='dot')))
    fig.add_trace(go.Scatter(x=l_sparsity, y=l_reconst_uniform, name="OBC Uniform", line = dict(dash='dot')))
    
    fig.add_trace(go.Scatter(x=l_sparsity, y=l_convex_uniform_multiple_block, name=f"OBC Convex Uniform Block - {n_convex}", line = dict(dash='dot')))
    fig.add_trace(go.Scatter(x=l_sparsity, y=l_convex_uniform_block_2, name="OBC Convex Uniform Block (2)", line = dict(dash='dot')))
    fig.add_trace(go.Scatter(x=l_sparsity, y=l_convex_uniform_block, name="OBC Convex Uniform Block", line = dict(dash='dot')))
    fig.add_trace(go.Scatter(x=l_sparsity, y=l_fisher_uniform_block, name="OBC Fisher Uniform Block", line = dict(dash='dot')))
    fig.add_trace(go.Scatter(x=l_sparsity, y=l_reconst_uniform_block, name="OBC Uniform Block", line = dict(dash='dot')))

    fig.add_trace(go.Scatter(x=l_sparsity, y=l_CAP_non_uniform, name="CAP Non-Uniform", line = dict(dash='dash')))

    fig.update_layout(
        title=f"Results for {arch}",
        xaxis_title="Sparsity",
        yaxis_title="Accuracy",
    )
    fig.write_html(f"comparison_{arch}.html")

# %%
