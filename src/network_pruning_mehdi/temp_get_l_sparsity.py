#%%
import matplotlib.pyplot as plt
import numpy as np
import os
#%%
d_saved_sparsities = {}
l_lambdas_sparisty = [0.001, 0.005, 0.01, 0.05]
for lambda_sparsity in l_lambdas_sparisty:
    l_saved_sparsities = []
    with open(f"cd_{lambda_sparsity}.txt", "rb") as f:
        lines = list(f.readlines())
        for line in lines:
            line = str(line)
            test_sparisty = line[:9]=="b'Current"
            if test_sparisty:
                current_sparsity = float(line.split("b'Current sparsity: ")[1].rstrip().replace("\\n'", ""))
                l_saved_sparsities.append(current_sparsity)
    d_saved_sparsities[lambda_sparsity] = l_saved_sparsities
# %%
for lambda_sparsity in l_lambdas_sparisty:
    final_sparsity = int(100*d_saved_sparsities[lambda_sparsity][-1])
    path_save_sparsity = f"/home/gridsan/gafriat/projects/network_pruning/Saves_OBC_1.0_1.0_0.01/results_cd_{lambda_sparsity}_{final_sparsity}/l_sparsities_cd_deit_tiny_patch16_224_3.npy"
    old_name = f"/home/gridsan/gafriat/projects/network_pruning/Saves_OBC_1.0_1.0_0.01/results_cd_{lambda_sparsity}_{final_sparsity}/old_l_sparsities_cd_deit_tiny_patch16_224_3.npy"
    try:
        os.remove(f"{path_save_sparsity}.npy")
    except:
        pass
    os.rename(path_save_sparsity, old_name)
    with open(f'{path_save_sparsity}', 'wb') as f:
        np.save(f, d_saved_sparsities[lambda_sparsity])

# %%
#plt.plot(d_saved_sparsities[0.005])
# %%
