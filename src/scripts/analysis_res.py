#%%
import pandas as pd

l_columns = ["dataset", "model", "batch_size", "accum_steps", "finetune_strategy", "epsilon", "delta", "classifier_lr", "lr", "clipping", "Train acc", "Test acc"]

# Read all the exp
res_tot = pd.read_csv("res_tot.csv", index_col=0)

# Remove unfinsihed ones
res_tot = res_tot[res_tot["Finished epochs"]>=50]

# Create a table per dataset, network and type of training
l_models = ["'deit_tiny_patch16_224'", "'deit_small_patch16_224'", "'deit_base_patch16_224'"]
l_dataset = ["'cifar10'", "'cifar100'"]
l_finetune_strategy = ["'all_layers'", "'lp_gn'"]

df_best_res = pd.DataFrame([])

l_finished_exp = []
for model in l_models:
    for dataset in l_dataset:
        for finetune_strategy in l_finetune_strategy:
            res_model = res_tot[(res_tot["model"]==model)*(res_tot["dataset"]==dataset)*(res_tot["finetune_strategy"]==finetune_strategy)]
            res_model[l_columns].sort_values(by="Test acc", ascending=False).to_csv(f"""csv_res/res_{model.replace("'","")}_{dataset.replace("'","")}_{finetune_strategy.replace("'","")}.csv""")
            df_best_res = pd.concat([df_best_res, res_model.iloc[[0]]])
            l_finished_exp.append(len(res_model))
# %%
df_best_res["Finished exp"] = l_finished_exp
df_best_res[l_columns+["Finished exp"]].sort_values(by="Test acc", ascending=False).to_csv(f"""csv_res/res_global.csv""")
# %%
