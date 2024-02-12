#%%
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, LinearLR, CosineAnnealingLR, LambdaLR
import torch
from previous_utils.main_utils import get_model
import numpy as np
import matplotlib.pyplot as plt

model, criterion, modules_to_prune = get_model("mlpnet", 0, pretrained=False, with_z=True)
T_max = 12
n_steps = 12

optimizer = torch.optim.SGD(model.parameters(), lr = 0.1, momentum=0.9, weight_decay=1e-3)
scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-5)

l_lr = []
l_lr.append(optimizer.param_groups[0]['lr'])
for i in range(n_steps):
    scheduler.step()
    l_lr.append(optimizer.param_groups[0]['lr'])

l_lr = np.array(l_lr)
plt.plot(l_lr)
# %%

e1 = 0
e2 = 12
l_lr_riade = np.arange(n_steps+1)
l_lr_riade = 0.00001+0.5*(0.1-0.00001)*(1+np.cos(np.pi*(l_lr_riade-e1)/(e2-e1)))
plt.plot(l_lr_riade)

# %%
l_lr-l_lr_riade
# %%
