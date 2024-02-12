#%%
import torch
import plotly.graph_objects as go
import numpy as np
import os

n_convex = 5
rel_damp = 1e-2
arch = "deit_tiny_patch16_224"
sparsity = 0.5

name = '%s_%04d.pth' % (arch, int(sparsity * 10000))

weight_losses = torch.load(f"Saves_OBC_{n_convex}_{rel_damp}/losses/{name}")
l_keys = []
for key in weight_losses:
    if ("norm" not in key) and ("blocks" in key) and ("embed" not in key) and ("bias" not in key):
        l_keys.append(key)
l_keys

# %%
from previous_utils.main_utils import get_model
model, _, _ = get_model(arch, 0, True, False, 0.0, False, "relu")
# %%
d_params = dict(model.named_parameters())
l_n_params = []
for key in l_keys:
    l_n_params.append(np.prod(d_params[key].shape))
# %%
l_n_params = np.array(l_n_params)
l_ind_seen = np.zeros(len(l_n_params))
l_s = np.zeros(len(l_n_params))
N = np.sum(l_n_params)
goal_sparsity = 0.5
# %%
for k in range(len(l_n_params)):
    s_k = (1/np.sum(l_n_params[l_ind_seen==0]))*(N*goal_sparsity-np.sum(l_n_params[l_ind_seen==1]*l_s[l_ind_seen==1]))
    ind_seen = np.argmax(l_n_params[l_ind_seen==0])
    l_ind_seen[ind_seen] = 1
    l_s[ind_seen] = s_k
    print(f"Minimum sparisty at step {k} is: {s_k}")
# %%
