#%%
from utils_experiments import *
import copy
from main import *

def loss_function_without_penalization(x,y):
    return torch.nn.MSELoss()(torch.squeeze(x),torch.squeeze(y))

metric_best_model = "ppl" #acc or ppl or loss
evaluation_metric = "ppl" #acc or ppl
default_arguments = vars(arguments)

#%%
if __name__ == '__main__':
    # ---
    name_folder_saves = "results_layer_wise_jan_10_2024"
    #l_experiments = ["pretrained_test_imagenet", "pretrained_test_imagenet_2", "pretrained_test_cifar10", "pretrained_test_mnist"]
    #l_experiments = ["pretrained_test_imagenet_layer_wise_oct_17_part1", "pretrained_test_imagenet_layer_wise_oct_17_part2", "pretrained_test_imagenet_layer_wise_oct_19_part1", "pretrained_test_imagenet_layer_wise_oct_19_part2"]
    #l_experiments = ["pretrained_test_imagenet_layer_wise_oct_24"]
    #l_experiments = ["pretrained_test_imagenet_layer_wise_oct_30"]
    #l_experiments = ["pretrained_test_imagenet_layer_wise_oct_31"]
    #l_experiments = ["pretrained_test_imagenet_layer_wise_oct_32"]
    #l_experiments = ["pretrained_test_imagenet_layer_wise_nov_6_1_bis"]
    #l_experiments = ["pretrained_test_imagenet_layer_wise_nov_6_2", "pretrained_test_imagenet_layer_wise_nov_10"]
    #l_experiments = ["pretrained_test_imagenet_layer_wise_nov_13_1", "pretrained_test_imagenet_layer_wise_nov_13_2", "pretrained_test_imagenet_layer_wise_nov_13_3"]
    #l_experiments = ["pretrained_test_imagenet_layer_wise_nov_13_4", "pretrained_test_imagenet_layer_wise_nov_14"]
    #l_experiments = ["pretrained_test_imagenet_layer_wise_nov_16_1", "pretrained_test_imagenet_layer_wise_nov_17_1", "pretrained_test_imagenet_layer_wise_nov_17_2"]
    #l_experiments = ["pretrained_test_imagenet_layer_wise_nov_14_1"]
    #l_experiments = ["pretrained_test_imagenet_layer_wise_nov_19"]
    #l_experiments = ["pretrained_test_imagenet_layer_wise_nov_20"]
    #l_experiments = ["pretrained_test_imagenet_layer_wise_nov_21_version_2", "pretrained_test_imagenet_layer_wise_nov_21_version_2_bis"]
    #l_experiments = ["pretrained_test_imagenet_layer_wise_nov_24"]
    #l_experiments = ["pretrained_layer_wise_nov_30", "pretrained_layer_wise_nov_30_2"]
    #l_experiments = ["pretrained_layer_wise_dec_2"]
    #l_experiments = ["pretrained_layer_wise_dec_3"]
    #l_experiments = ["pretrained_layer_wise_dec_3_2", "pretrained_layer_wise_dec_3_3"]
    #l_experiments = ["pretrained_layer_wise_dec_18_1", "pretrained_layer_wise_dec_18_2", "pretrained_layer_wise_dec_18_3"]
    #l_experiments = ["pretrained_layer_wise_dec_19"]
    #l_experiments = ["pretrained_layer_wise_dec_31_1", "pretrained_layer_wise_dec_31_2", "pretrained_layer_wise_dec_31_3"]
    #l_experiments = ["pretrained_layer_wise_jan_3_2024_2"]
    #l_experiments = ["pretrained_layer_wise_jan_7_2024_1", "pretrained_layer_wise_jan_7_2024_2"]
    l_experiments = ["pretrained_layer_wise_jan_10_2024_1", "pretrained_layer_wise_jan_10_2024_2"]
    l_arch = ["deit_small_patch16_224"]
    #l_arch = ["resnet50"]
    #l_arch = ["facebook/opt-125m"]#, "facebook/opt-350m"]

    l_sparsity = [0.5, 0.6, 0.7, 0.8, 0.9]
    #l_sparsity = [0.5]
    #l_sparsity = [0.9]

    type_of_task = "classification"

    # Run python files
    for experiment in l_experiments:
        os.system("python experiments/"+experiment+".py")

    if not(os.path.exists(name_folder_saves)):
        os.mkdir(name_folder_saves)
    else:
        shutil.rmtree(name_folder_saves)
        os.mkdir(name_folder_saves)

    # ["spambase", "churn", "mice_protein", "madelon", "gisette", "dorothea", "spambase", "churn", "mice_protein"]
    for goal_sparsity in l_sparsity:
        for arch in l_arch:

            print("---- Gathering results for "+arch+" ----")
            n_scripts_study = 0
            l_tables = []

            l_tables_our_method = []
            l_tables_other = []

            for experiment in l_experiments:
                with open("experiments/"+experiment+".json", "r") as f:
                    d_params = json.load(f)

                if not("type_model" in d_params):
                    d_params["type_model"]=["our_method"]

                for key in default_arguments:
                    if not(key in d_params) and not(key == "lr"):
                        d_params[key] = [default_arguments[key]]
                if "our_method" in d_params["type_model"]:
                    d_params_copy = copy.deepcopy(d_params)
                    d_params_copy["type_model"] = ["our_method"]
                    if arch in d_params["arch"]:
                        d_params_copy["arch"] = [arch]
                        if goal_sparsity in d_params["goal_sparsity"]:
                            d_params_copy["goal_sparsity"] = [goal_sparsity]
                            l_tables_our_method.append(d_params_copy)
                            n_script_study_added = 1
                            for param in d_params_copy:
                                n_script_study_added *= len(d_params_copy[param])
                            n_scripts_study += n_script_study_added
                if "other" in d_params["type_model"]:
                    d_params_copy = copy.deepcopy(d_params)
                    d_params_copy["type_model"] = ["other"]
                    if arch in d_params["arch"]:
                        d_params_copy["arch"] = [arch]
                        if goal_sparsity in d_params["goal_sparsity"]:
                            d_params_copy["goal_sparsity"] = [goal_sparsity]
                            l_tables_other.append(d_params_copy)
                            n_script_study_added = 1
                            for param in d_params_copy:
                                n_script_study_added *= len(d_params_copy[param])
                            n_scripts_study += n_script_study_added

            l_tables.append(l_tables_our_method)
            l_tables.append(l_tables_other)
            
            arch = arch.replace("facebook/", "")
            
            path_save = name_folder_saves+"/study_"+arch+"_"+str(goal_sparsity)+"_"+metric_best_model
            if not(os.path.exists(path_save)):
                os.mkdir(path_save)

            if n_scripts_study>0:
                l_data, data_final = summarize_results(l_tables=l_tables, path_save=path_save, type_of_task=type_of_task, metric_best_model=metric_best_model, n_scripts=n_scripts_study, evaluation_metric=evaluation_metric)
            else:
                print("No study for", arch)

# %%
