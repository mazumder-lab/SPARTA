#%%
from utils_model import *
# %%
linear_layer = Linear_with_z(10, 5, False)
# %%
linear_layer(torch.ones(100, 10))
# %%
linear_layer.sparsify()
#%%
linear_layer(torch.ones(100,10))

# %%
