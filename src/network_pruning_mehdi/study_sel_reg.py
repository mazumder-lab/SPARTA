#%%
import numpy as np
from utils_experiments import convert_to_small_str
begin_name = "study_mlpnet_lr_0.01_500_mnist_SGD_es_patience_50_val_loss_gamma_1_sel_reg_"
end_name = "_ent_reg_0.1_l2_reg_0_wd_3.752e-5_mom_0.9_auc"
l_sel_reg = [1e-05, 5e-05, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
#l_sel_reg = list(np.linspace(0.005, 0.05, 21))
l_sel_reg_str = [convert_to_small_str(x) for x in l_sel_reg]
#%%

goal_sparsity = np.linspace(0,1,100)
corres_sel_reg = np.zeros(100)
corres_val_metric = np.zeros(100)

for i in range(100):
    best_val_metric = -np.inf
    for sel_reg_str in l_sel_reg_str:
        sel_reg = float(sel_reg_str)
        name_study = begin_name+sel_reg_str+end_name
        path_l_z = "Saves_sparsity_vs_sel_reg/"+name_study+"/best_trial/repeat_0/l_n_z.npy"
        path_l_val_metric = "Saves_sparsity_vs_sel_reg/"+name_study+"/best_trial/repeat_0/l_validation_metric.npy"
        l_z = np.load(path_l_z)
        l_val_metric = np.load(path_l_val_metric)
        l_sparsity = 1-l_z/32360
        if len(l_sparsity[l_sparsity>=goal_sparsity[i]])>=1:
            ind_epoch = np.argmax(l_val_metric[l_sparsity>=goal_sparsity[i]])
            if best_val_metric<l_val_metric[ind_epoch]:
                best_val_metric = l_val_metric[ind_epoch]
                corres_sel_reg[i] = sel_reg
                corres_val_metric[i] = l_val_metric[ind_epoch]

# %%
import matplotlib.pyplot as plt
plt.figure()
plt.plot(goal_sparsity, corres_sel_reg)
plt.savefig("sel_reg_vs_sparsity_1.png")
# %%
plt.figure()
plt.plot(goal_sparsity, corres_val_metric)
plt.savefig("val_metric_vs_sparsity_1.png")

# %%
wrong_sparsity = 0.7935105067985166
n_params = (2*32360)*(1-wrong_sparsity)
sparsity = 1-n_params/32360
# %%
n_params = (2*32360)*(0.21083127317676142)
sparsity = 1-n_params/32360

# %%
n_params = (2*32360)*(1-0.7934796044499381)
sparsity = 1-n_params/32360

# %%
n_params = (2*32360)*(1-0.7892305315203956)
sparsity = 1-n_params/32360

# %%
n_params = (2*32360)*(1-0.7934796044499381)
sparsity = 1-n_params/32360
sparsity
# %%
n_params = (2*32360)*(1-0.7930006180469715)
sparsity = 1-n_params/32360
sparsity

# %%
