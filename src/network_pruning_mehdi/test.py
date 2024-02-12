#%%
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
#%%
import torch
from typing import Any, Optional
from torch import Tensor, nn
import copy
import math
import optuna
import itertools
from sklearn.metrics import accuracy_score, roc_auc_score
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import roc_auc_score
# from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, LinearLR, CosineAnnealingLR, LambdaLR
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, LinearLR, CosineAnnealingLR, LambdaLR
import time

# https://www.kaggle.com/competitions/cs419m/data

# ------------------------
# --- Models per block ---
# ------------------------

import torch
from torch.utils.data import Dataset, DataLoader

#weight_proba_main = Parameter(torch.Tensor(self.n_main, self.n_trees, 2**self.depth-1, 1))
#%%
import torch
import torch_sparse
from torch_sparse import coalesce
from torch.nn.parameter import Parameter
import copy

index = torch.tensor([[1, 0, 1, 0, 2, 1],
                      [0, 1, 1, 1, 0, 0],
                      [0, 1, 1, 1, 0, 0]])
value = torch.Tensor([[1], [3], [4], [5], [6], [7]])

index, value = coalesce(index, value, m=3, n=2)

# %%
torch.manual_seed(1)
weights = Parameter(torch.rand(5,3))
weights.data[2,2] = 0
weights.data[2,1] = 0
weights.data[4,2] = 0
sparse_weight = copy.deepcopy(weights.data).to_sparse()
#%%
sparse_weight_values = Parameter(sparse_weight.values())
sparse_weight_indices = Parameter(sparse_weight.indices(), requires_grad=False)
sparse_weight_size = sparse_weight.size()
#%%
sparse_weight = torch.sparse_coo_tensor(sparse_weight_indices, sparse_weight_values, sparse_weight_size)

#%%
optimizer = torch.optim.Adam([sparse_weight_values], 0.1)

index = torch.stack(torch.where(weights!=0))
value = weights[tuple(index)]
sample_vec = torch.rand(64,3)
# %%
pred = torch.einsum("jk,bk -> jb", weights,sample_vec)
# %%
pred2 = torch.sparse.mm(sparse_weight, sample_vec.T)
# %%
torch.sum(pred != pred2)
# %%
loss = torch.mean((torch.ones_like(pred2)-pred2))**2
loss.backward()
# %%
optimizer.step()
# %%
from utils_model import *
# %%
linear_layer = Linear_with_z(10, 10, False)
# %%
linear_layer(torch.ones(10))
# %%
linear_layer.sparsify()
# %%
linear_layer.sparse_weight_z_indices

# %%
