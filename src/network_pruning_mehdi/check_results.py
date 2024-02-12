from previous_utils.main_utils import get_model, get_dataset
import torch
# %%
model, _, _ = get_model(arch, 0, True, False, 1.0, False, "relu")
# %%
path_1 = "TEMP/deit_tiny_patch16_224_unstr_70x_greedy.pth"
path_2 = "TEMP/deit_tiny_patch16_224_unstr_50x_greedy.pth"
model.load_state_dict(torch.load(path_2, map_location="cpu"))
# %%
#train_val_dataset, test_dataset = get_dataset("imagnet", , n_train_kept, get_item_func, arch, seed, activation_fn, device, test_almost_sequential, test_update_test_vit, test_pass_first_modules, further_subsampling)
#%%
l_layers = [x[1] for x in model.blocks.named_parameters() if "norm" not in x[0] and "bias" not in x[0]]
sum([(x==0).float().sum().item() for x in l_layers])/np.sum([np.prod(x.shape) for x in l_layers])
# %%
