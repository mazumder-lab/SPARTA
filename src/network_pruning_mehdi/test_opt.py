#%%
from Sparse_GPT_utils.opt import *
from Sparse_GPT_utils.datautils import get_loaders

print("Imports done!", flush = True)

model = get_opt("facebook/opt-125m", "/home/gridsan/gafriat/Sparse_NN_shared/LLM/model/", cached=True, with_z = False, gamma=1.0)
import ipdb;ipdb.set_trace()

# %%
params_z = np.sum([np.prod(x[1].shape) for x in list(model.named_parameters()) if "_z" in x[0]])
print(params_z)
# %%
params_non_z = np.sum([np.prod(x[1].shape) for x in list(model.named_parameters()) if "_z" not in x[0]])
print(params_non_z)
#%%
print(params_z/params_non_z)
# %%
dataloader, testloader = get_loaders(
        "c4", "/home/gridsan/gafriat/Sparse_NN_shared/LLM/data/", nsamples=128, seed=0, model="facebook/opt-125m", seqlen=model.seqlen
    )
#%%
model.to("cuda")
acc = 0
model.eval()
for batch in dataloader:
    model(batch[0].to("cuda"))
    acc+=1
    print(f"Acc: {acc}")
# %%
import ipdb;ipdb.set_trace()
