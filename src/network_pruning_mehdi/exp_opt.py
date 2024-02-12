#%%
import torch
import torch.nn as nn

def get_opt(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM
    model = OPTForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = model.config.max_position_embeddings
    return model

model = get_opt("facebook/opt-125m")
model.eval()

# %%
from utils_model import Linear_with_z
list(list(list(list(list(list(model.children())[0].children())[0].children())[3].children())[0].children())[0].children())[0] = Linear_with_z(768, 768, True)

# %%
