#%%
import plotly.graph_objects as go
import numpy as np

#%%
l_sparsities_5000 = np.load("Saves_OBC_1.0_1.0_0.01/results_greedy_-1_0_0_0.2/l_sparsities_greedy_deit_tiny_patch16_224_90.npy")
l_sparsities_4000 = np.load("Saves_OBC_1.0_1.0_0.01/results_greedy_-1_0_0_0.2_4000/l_sparsities_greedy_deit_tiny_patch16_224_90.npy")
l_sparsities_3000 = np.load("Saves_OBC_1.0_1.0_0.01/results_greedy_-1_0_0_0.2_3000/l_sparsities_greedy_deit_tiny_patch16_224_90.npy")
l_sparsities_2000 = np.load("Saves_OBC_1.0_1.0_0.01/results_greedy_-1_0_0_0.2_2000/l_sparsities_greedy_deit_tiny_patch16_224_90.npy")
l_sparsities_1000 = np.load("Saves_OBC_1.0_1.0_0.01/results_greedy_-1_0_0_0.2_1000/l_sparsities_greedy_deit_tiny_patch16_224_90.npy")
l_sparsities_500 = np.load("Saves_OBC_1.0_1.0_0.01/results_greedy_-1_0_0_0.2_500/l_sparsities_greedy_deit_tiny_patch16_224_90.npy")
#%%
l_train_acc_5000 = np.load("Saves_OBC_1.0_1.0_0.01/results_greedy_-1_0_0_0.2/l_train_acc_greedy_deit_tiny_patch16_224_90.npy")
l_train_acc_4000 = np.load("Saves_OBC_1.0_1.0_0.01/results_greedy_-1_0_0_0.2_2000/l_train_acc_greedy_deit_tiny_patch16_224_90.npy")
l_train_acc_3000 = np.load("Saves_OBC_1.0_1.0_0.01/results_greedy_-1_0_0_0.2_2000/l_train_acc_greedy_deit_tiny_patch16_224_90.npy")
l_train_acc_2000 = np.load("Saves_OBC_1.0_1.0_0.01/results_greedy_-1_0_0_0.2_2000/l_train_acc_greedy_deit_tiny_patch16_224_90.npy")
l_train_acc_1000 = np.load("Saves_OBC_1.0_1.0_0.01/results_greedy_-1_0_0_0.2_1000/l_train_acc_greedy_deit_tiny_patch16_224_90.npy")
l_train_acc_500 = np.load("Saves_OBC_1.0_1.0_0.01/results_greedy_-1_0_0_0.2_500/l_train_acc_greedy_deit_tiny_patch16_224_90.npy")
#%%
l_train_losses_5000 = np.load("Saves_OBC_1.0_1.0_0.01/results_greedy_-1_0_0_0.2/l_train_losses_greedy_deit_tiny_patch16_224_90.npy")
l_train_losses_4000 = np.load("Saves_OBC_1.0_1.0_0.01/results_greedy_-1_0_0_0.2/l_train_losses_greedy_deit_tiny_patch16_224_90.npy")
l_train_losses_3000 = np.load("Saves_OBC_1.0_1.0_0.01/results_greedy_-1_0_0_0.2/l_train_losses_greedy_deit_tiny_patch16_224_90.npy")
l_train_losses_2000 = np.load("Saves_OBC_1.0_1.0_0.01/results_greedy_-1_0_0_0.2_2000/l_train_losses_greedy_deit_tiny_patch16_224_90.npy")
l_train_losses_1000 = np.load("Saves_OBC_1.0_1.0_0.01/results_greedy_-1_0_0_0.2_1000/l_train_losses_greedy_deit_tiny_patch16_224_90.npy")
l_train_losses_500 = np.load("Saves_OBC_1.0_1.0_0.01/results_greedy_-1_0_0_0.2_500/l_train_losses_greedy_deit_tiny_patch16_224_90.npy")
# %%

arch = "deit_tiny_patch16_224"

fig = go.Figure()

fig.add_trace(go.Scatter(x=l_sparsities_5000, y=l_train_acc_5000, name=f"OBC Convex Non-Uniform Fast Greedy 5000 - Accuracy"))
fig.add_trace(go.Scatter(x=l_sparsities_5000, y=l_train_losses_5000, name=f"OBC Convex Non-Uniform Fast Greedy 5000 - Losses"))

fig.add_trace(go.Scatter(x=l_sparsities_4000, y=l_train_acc_4000, name=f"OBC Convex Non-Uniform Fast Greedy 4000 - Accuracy"))
fig.add_trace(go.Scatter(x=l_sparsities_4000, y=l_train_losses_4000, name=f"OBC Convex Non-Uniform Fast Greedy 4000 - Losses"))

fig.add_trace(go.Scatter(x=l_sparsities_3000, y=l_train_acc_3000, name=f"OBC Convex Non-Uniform Fast Greedy 3000 - Accuracy"))
fig.add_trace(go.Scatter(x=l_sparsities_3000, y=l_train_losses_3000, name=f"OBC Convex Non-Uniform Fast Greedy 3000 - Losses"))

fig.add_trace(go.Scatter(x=l_sparsities_2000, y=l_train_acc_2000, name=f"OBC Convex Non-Uniform Fast Greedy 2000 - Accuracy"))
fig.add_trace(go.Scatter(x=l_sparsities_2000, y=l_train_losses_2000, name=f"OBC Convex Non-Uniform Fast Greedy 2000 - Losses"))

fig.add_trace(go.Scatter(x=l_sparsities_1000, y=l_train_acc_1000, name=f"OBC Convex Non-Uniform Fast Greedy 1000 - Accuracy"))
fig.add_trace(go.Scatter(x=l_sparsities_1000, y=l_train_losses_1000, name=f"OBC Convex Non-Uniform Fast Greedy 1000 - Losses"))

fig.add_trace(go.Scatter(x=l_sparsities_500, y=l_train_acc_500, name=f"OBC Convex Non-Uniform Fast Greedy 500 - Accuracy"))
fig.add_trace(go.Scatter(x=l_sparsities_500, y=l_train_losses_500, name=f"OBC Convex Non-Uniform Fast Greedy 500 - Losses"))

fig.update_layout(
    title=f"Results for {arch}",
    xaxis_title="Sparsity",
    yaxis_title="Accuracy",
)
fig.write_html(f"comparison_n_samples_{arch}_train.html")

# %%