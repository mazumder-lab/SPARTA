#%%
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pandas as pd
import numpy as np
#%%
import torch
import optuna
from optuna.trial import TrialState
import logging
import sys
# import matplotlib
# matplotlib.use("Agg")
import shutil
import json
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
import itertools
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import scipy.io
from sklearn.impute import SimpleImputer
import pickle
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
import random

# -------------------------------
# --- Fine tuning with Optuna ---
# -------------------------------

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return None

class SaveStudyOptuna:
      def __init__(self, name_study, folder_saves):
            self.name_study = name_study
            self.folder_saves = folder_saves
            try:
                  if not(os.path.exists(self.folder_saves)):
                        os.mkdir(self.folder_saves)
            except:
                  pass
            if not(os.path.exists(self.folder_saves+"/study_"+self.name_study)):
                  os.mkdir(self.folder_saves+"/study_"+self.name_study)
      def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
            with open(self.folder_saves+"/study_"+self.name_study+"/save_study.pkl", 'wb') as handle:
                  pickle.dump(study, handle, protocol=pickle.HIGHEST_PROTOCOL)
            if len(study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])) > 1:
                  param_imp_plot = optuna.visualization.plot_param_importances(study)
                  param_imp_plot.write_html(self.folder_saves+"/study_"+self.name_study+"/param_importance.html")
                  progress_over_time = optuna.visualization.plot_optimization_history(study)
                  progress_over_time.write_html(self.folder_saves+"/study_"+self.name_study+"/progress_over_time.html")
                  param_influence = optuna.visualization.plot_slice(study)
                  param_influence.write_html(self.folder_saves+"/study_"+self.name_study+"/param_influence.html")
                  for key, value in study.best_trial.params.items():
                        optuna.visualization.plot_slice(study, [key]).write_html(self.folder_saves+"/study_"+self.name_study+"/"+key+"_influence.html")

def conduct_fine_tuning(objective, name_study, timeout, n_trials, save_study=None, n_jobs=2, folder_saves="TSML_saves", type_of_task="regression"):
      optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
      callback_item = SaveStudyOptuna(name_study, folder_saves)
      if save_study==None:
            if type_of_task == "regression":
                  study = optuna.create_study(direction="minimize")
            else:
                  study = optuna.create_study(direction="maximize")
            sampler = optuna.samplers.TPESampler()
            study.sampler = sampler
      else:
            study = save_study
            sampler = optuna.samplers.TPESampler()
            study.sampler = sampler
      study.optimize(objective, timeout=timeout, n_trials=n_trials, n_jobs=n_jobs, callbacks=[callback_item])
      pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
      complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
      print("Study statistics: ")
      print("  Number of finished trials: ", len(study.trials))
      print("  Number of pruned trials: ", len(pruned_trials))
      print("  Number of complete trials: ", len(complete_trials))

      print("Best trial for study_"+name_study+" :")
      trial = study.best_trial
      print("  Value: ", trial.value)

      best_params = trial.params
      print("  Params: ")
      for key, value in best_params.items():
            print("    {}: {}".format(key, value))

      with open(folder_saves+"/study_"+name_study+"/best_params.pkl", 'wb') as handle:
            pickle.dump(best_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

def save_results(dict_list, best_model, d_results, name_study, dict_params, trial, folder_saves, ind_repeat, sds):
        try:
              number_trial = trial.number
        except:
              number_trial = 0
        if not(os.path.exists(folder_saves)):
                os.mkdir(folder_saves)
        if not(os.path.exists(folder_saves+"/study_"+name_study)):
                os.mkdir(folder_saves+"/study_"+name_study)
        if not(os.path.exists(folder_saves+"/study_"+name_study+"/trial_"+str(number_trial))):
                os.mkdir(folder_saves+"/study_"+name_study+"/trial_"+str(number_trial))
        if not(os.path.exists(folder_saves+"/study_"+name_study+"/trial_"+str(number_trial)+"/repeat_"+str(ind_repeat))):
                os.mkdir(folder_saves+"/study_"+name_study+"/trial_"+str(number_trial)+"/repeat_"+str(ind_repeat))
        if not(os.path.exists(folder_saves+"/study_"+name_study+"/trial_"+str(number_trial)+"/repeat_"+str(ind_repeat)+"/history")):
                os.mkdir(folder_saves+"/study_"+name_study+"/trial_"+str(number_trial)+"/repeat_"+str(ind_repeat)+"/history")
        if not(os.path.exists(folder_saves+"/study_"+name_study+"/trial_"+str(number_trial)+"/params.json")):
                with open(folder_saves+"/study_"+name_study+"/trial_"+str(number_trial)+"/params.json", "w") as outfile:
                    json.dump(dict_params, outfile)

        l_params_names = np.sort(list(dict_params.keys()))
        l_params_values = [dict_params[l_params_names[i]] for i in range(len(l_params_names))]

        fig = go.Figure()
        for key_list in dict_list:
                l_names = key_list.split("/")
                name_save = l_names[0]
                name_plot = l_names[1]
                fig.add_trace(go.Scatter(x=np.arange(len(dict_list[key_list])), y=dict_list[key_list], name=name_plot))
                with open(folder_saves+"/study_"+name_study+"/trial_"+str(number_trial)+"/repeat_"+str(ind_repeat)+"/"+name_save+".npy", 'wb') as f:
                    np.save(f, dict_list[key_list])
                if name_plot=="Learning rate":
                    fig.add_trace(go.Scatter(x=np.arange(len(dict_list[key_list])), y=2/dict_list[key_list], name="2/lr"))

        fig.update_layout(title="Summary for "+name_study)
        fig.write_html(folder_saves+"/study_"+name_study+"/trial_"+str(number_trial)+"/repeat_"+str(ind_repeat)+"/summary.html")

        with open(folder_saves+"/study_"+name_study+"/trial_"+str(number_trial)+"/repeat_"+str(ind_repeat)+"/results.json", "w") as outfile:
                json.dump(d_results, outfile)

        path_model = folder_saves+"/study_"+name_study+"/trial_"+str(number_trial)+"/repeat_"+str(ind_repeat)+"/model"
        torch.save(best_model.state_dict(), path_model)
        with open(folder_saves+"/study_"+name_study+"/trial_"+str(number_trial)+"/repeat_"+str(ind_repeat)+"/test_save_done.npy", 'wb') as f:
                np.save(f, True)

def delete_models(name_study, folder_saves, n_repeat, type_of_task, delete_pass=False):
      d_repeats = {}
      list_dir = os.listdir(folder_saves+"/study_"+name_study)
      trial_folders = np.array([x for x in list_dir if "trial" in x])
      print(trial_folders)
      l_metric = []
      l_to_keep = []
      for i in range(len(trial_folders)):
            model_folder = trial_folders[i]
            try:
                  validation_metric = 0
                  for ind_repeat in range(n_repeat):
                        with open(folder_saves+"/study_"+name_study+"/"+model_folder+"/repeat_"+str(ind_repeat)+"/test_save_done.npy", 'rb') as f:
                              test_save_done = np.load(f)
                        with open(folder_saves+"/study_"+name_study+"/"+model_folder+"/repeat_"+str(ind_repeat)+"/results.json", "r") as f:
                              dict_results = json.load(f)
                        if type_of_task=="classification":
                              try:
                                    validation_metric += dict_results["val_acc"]/n_repeat
                              except:
                                    validation_metric += dict_results["val_ppl"]/n_repeat
                        elif type_of_task == "regression":
                              validation_metric += dict_results["val_mse"]/n_repeat
                  l_metric.append(validation_metric)
                  l_to_keep.append(i)
            except:
                  if delete_pass:
                        shutil.rmtree(folder_saves+"/study_"+name_study+"/"+model_folder)
                  else:
                        print("pass")
                        pass
      trial_folders = trial_folders[l_to_keep]
      if len(l_metric)>0:
            if type_of_task == "regression":
                  best_metric_ind = np.argmin(l_metric)
            elif type_of_task == "classification":
                  best_metric_ind = np.argmax(l_metric)
            for i in range(len(trial_folders)):
                  if i!=best_metric_ind:
                        try:
                              shutil.rmtree(folder_saves+"/study_"+name_study+"/"+trial_folders[i])
                              print(trial_folders[i]+" deleted")
                        except:
                              pass

def gather_list_models(name_study, ind_repeat, folder_saves, device):
      path_models = folder_saves+"/study_"+name_study+"/best_trial/repeat_"+str(ind_repeat)+"/history/all_models"
      path_save = folder_saves+"/study_"+name_study+"/best_trial/repeat_"+str(ind_repeat)+"/history/all_models.pth"
      if os.path.exists(path_models):
            list_models_path = os.listdir(path_models)
            list_models_path = [x for x in list_models_path if "model" in x]
            list_models_state_dict = [torch.load(path_models+"/model_"+str(i), map_location=device) for i in range(len(list_models_path))]
            torch.save(list_models_state_dict, path_save)
            shutil.rmtree(path_models)
            print("The models have successfully been saved as a list")
      elif not(os.path.exists(path_save)):
            print("The models cannot be found")
      else:
            print("The models have already been saved as a list")            

def read_results(name_study, ind_repeat, type_of_task = "classification", folder_saves = ""):
      try:
            name_trial = os.listdir(folder_saves+"/study_"+name_study)
      except:
            print("The study doesn't exist,", folder_saves+"/study_"+name_study)

      name_trial = [x for x in name_trial if "trial" in x]
      if len(name_trial)>1:
            print("More than one model found")
      if len(name_trial)==0:
            print("No model found")

      if "best_trial" in name_trial:
            name_trial = "best_trial"
      else:
            name_trial = name_trial[0]
      try:
            with open(folder_saves+"/study_"+name_study+"/"+name_trial+"/params.json", "r") as f:
                  dict_params = json.load(f)
      except:
            print("couldn't read dict_params")

      with open(folder_saves+"/study_"+name_study+"/"+name_trial+"/repeat_"+str(ind_repeat)+"/results.json", "r") as f:
            dict_results = json.load(f)

      if type_of_task == "classification":
            try:
                  in_sample_metric = np.array([dict_results["train_acc"], dict_results["train_auc"]])
            except:
                  in_sample_metric = np.array([dict_results["train_ppl"], dict_results["train_auc"]])
            try:
                  validation_metric = np.array([dict_results["val_acc"], dict_results["val_auc"]])
            except:
                  validation_metric = np.array([dict_results["val_ppl"], dict_results["val_auc"]])
            try:
                  test_metric = np.array([dict_results["test_acc"], dict_results["test_auc"]])
            except:
                  test_metric = np.array([dict_results["test_ppl"], dict_results["test_auc"]])
      elif type_of_task == "regression":
            in_sample_metric = dict_results["train_mse"]
            validation_metric = dict_results["val_mse"]
            test_metric = dict_results["test_mse"]

      best_ep = dict_results["best_ep"]
      time_training = dict_results["time_training"]
      n_z = dict_results["n_z"]
      if "n_params" in dict_results:
            n_params = dict_results["n_params"]
      else:
            n_params = None
      if "sparsity" in dict_results:
            sparsity = dict_results["sparsity"]
      else:
            sparsity = None
      return dict_params, in_sample_metric, validation_metric, test_metric, best_ep, time_training, n_z, n_params, sparsity

# -------------------
# --- Experiments ---
# -------------------

def convert_to_small_str(number):
      if int(number)==number:
            return str(int(number))
      if len(str(number))>5:
            power_of_ten = -int(np.floor(np.log(abs(number))/np.log(10)))
            number = number*10**(power_of_ten)
            number = np.round(number, 3)
            return str(number)+"e-"+str(power_of_ten)
      return str(number)

def get_name_study(arch,
                   weight_decay,
                   momentum,
                   pretrained,
                   n_epochs,
                   timeout,
                   n_trials,
                   name_dataset,
                   learning_rate,
                   batch_size_dataset,
                   test_early_stopping,
                   test_save_all_models,
                   optimizer_name,
                   min_lr,
                   max_lr,
                   type_decay,
                   gamma_lr_decay,
                   T_max_cos,
                   eta_min_cos,
                   start_lr_decay,
                   end_lr_decay,
                   path_load_weights,
                   type_of_task,
                   test_compute_accurate_in_sample_loss,
                   n_repeat,
                   folder_saves,
                   warmup_steps,
                   patience,
                   gamma,
                   entropy_reg,
                   selection_reg,
                   l2_reg,
                   metric_early_stopping,
                   device,
                   period_milestones,
                   metric_best_model,
                   test_different_lr,
                   dense_to_sparse,
                   seed,
                   val_second_lr,
                   goal_sparsity,
                   n_restart,
                   tol_z_1,
                   type_training,
                   num_workers,
                   type_pruning,
                   local_rank,
                   test_distributed,
                   mode,
                   loss_func,
                   type_reset,
                   n_train_kept,
                   threshold_weights,
                   method_pruning,
                   threshold_restart, 
                   test_constraint_weights,
                   test_one_layer_pruning,
                   test_prop_goal_sparsity,
                   l2_original_reg,
                   test_normalized_sgd,
                   type_function,
                   pruning_rate_cte,
                   lambda_loss,
                   test_load_data_first,
                   test_repeat_if_sparsity_not_reached,
                   loss_last_block,
                   retrain_last_block,
                   test_mult_reset,
                   test_reset_to_orignal,
                   test_start_sparse_gpt,
                   prune_bias,
                   type_compute_sparsity,
                   test_adaptive_lr,
                   patience_adaptive_lr,
                   patience_freeze,
                   test_wait_for_pruning,
                   test_almost_sequential,
                   tol_ent_reg,
                   tol_sel_reg,
                   goal_sparsity_discrete,
                   activation_fn,
                   n_incr_gradual_pruning,
                   test_start_convex,
                   type_pruning_schedule,
                   test_start_obc,
                   rel_damp,
                   lambda_fisher,
                   lambda_reconst,
                   algo_pruning,
                   test_save_all_sparsities,
                   n_parallel,
                   n_convex,
                   pruning_level,
                   n_layers,
                   type_model = "our_model"):
      
      if "facebook/" in arch:
            name_study = arch.replace("facebook/","")
            if activation_fn=="leaky_relu":
                  name_study+="_lr"
            elif activation_fn!="relu":
                  name_study+="_"+activation_fn
      elif "deit_tiny" in arch:
            name_study = "d_t"
      elif "deit_small" in arch:
            name_study = "d_s"
      elif "deit_base" in arch:
            name_study = "d_b"
      else:
            name_study = arch
      if test_start_sparse_gpt:
            name_study+="_sgpt"
            if rel_damp!=1e-2:
                  name_study+="_"+convert_to_small_str(rel_damp)
      elif test_start_convex:
            name_study+="_cvx"
            if algo_pruning=="obc_convex":
                  name_study+="_obc"
            elif algo_pruning=="sparse_gpt_convex":
                  name_study+="_sgpt"
            elif algo_pruning!="mp_convex":
                  name_study+="_"+algo_pruning
            if n_convex!=-1:
                  name_study+="_"+convert_to_small_str(n_convex)
            else:
                  name_study+="_"+convert_to_small_str(lambda_reconst)
                  name_study+="_"+convert_to_small_str(lambda_fisher)
            if rel_damp!=1e-2:
                  name_study+="_"+convert_to_small_str(rel_damp)
      elif test_start_obc:
            name_study+="_obc"
            if rel_damp!=0:
                  name_study+="_"+convert_to_small_str(rel_damp)
      if test_start_convex or test_start_sparse_gpt or test_start_obc:
            if pruning_level!="layer":
                  if pruning_level=="block":
                        name_study+="_bl"
                        if n_layers!=-1:
                              name_study+="_"+convert_to_small_str(n_layers)
                  else:
                        name_study+="_"+pruning_level
      if test_almost_sequential==1:
            name_study+="_as"
      if test_almost_sequential==2:
            name_study+="_as2"
      if test_almost_sequential==3:
            name_study+="_par"
      if learning_rate != -1:
            name_study += "_lr_"+convert_to_small_str(learning_rate)
      if test_adaptive_lr:
            name_study += "_adpt_"+convert_to_small_str(patience_adaptive_lr)
      if val_second_lr!=-1:
            name_study+="_"+convert_to_small_str(val_second_lr)
      name_study+="_"+convert_to_small_str(batch_size_dataset)
      name_study+="_"+name_dataset
      if n_train_kept!=-1:
            name_study+="_"+convert_to_small_str(n_train_kept)
      if test_normalized_sgd:
            name_study+="_norm"
      name_study+="_"+optimizer_name
      if type_function == "sigmoid":
            name_study+="_sig"
      elif type_function == "smoothstep":
            name_study+="_smst"
      if type_training != "combined":
            name_study+="_"+type_training
      if n_epochs!=1000:
            name_study+="_n_ep_"+convert_to_small_str(n_epochs)
      if test_wait_for_pruning:
            name_study+="_af_pru"
      if patience_freeze!=1:
            name_study+="_frz_"+convert_to_small_str(patience_freeze)
      if not(prune_bias):
            name_study+="_npb"
      if (type_decay!="None"):
            name_study+="_"+type_decay
            if type_decay=="linear":
                  name_study+="_linear_start_"+convert_to_small_str(start_lr_decay)+"_end_"+convert_to_small_str(end_lr_decay)
            elif type_decay=="exponential":
                  name_study+="_"+convert_to_small_str(gamma_lr_decay)
            elif type_decay=="cosine":
                  if T_max_cos!=-1:
                        name_study+="_T_max_"+convert_to_small_str(T_max_cos)
                  if eta_min_cos!=1e-5:
                        name_study += "_min_lr_"+convert_to_small_str(eta_min_cos)
            elif type_decay=="multi_lr":
                  name_study+="_p_"+convert_to_small_str(period_milestones)+"_g_"+convert_to_small_str(gamma_lr_decay)
            elif type_decay=="ramp":
                  name_study+="_"+convert_to_small_str(warmup_steps)
      if test_early_stopping==0 and retrain_last_block:
            name_study+="_es0"
            if patience!=-1:
                  name_study+="_p_"+convert_to_small_str(patience)
            name_study += "_"+metric_early_stopping
            if test_compute_accurate_in_sample_loss:
                  name_study+="_acc_loss"
      elif test_early_stopping==1:
            name_study+="_es"
            if patience!=-1:
                  name_study+="_p_"+convert_to_small_str(patience)
            name_study += "_"+metric_early_stopping
            if test_compute_accurate_in_sample_loss:
                  name_study+="_acc_loss"
      elif test_early_stopping not in [0,1]:
            name_study+="_es_"+convert_to_small_str(test_early_stopping)
            if patience!=-1:
                  name_study+="_p_"+convert_to_small_str(patience)
            name_study += "_"+metric_early_stopping
            if test_compute_accurate_in_sample_loss:
                  name_study+="_acc_loss"
            
      if mode=="ensemble":
            name_study += "_ens"
      elif mode=="layer_wise":
            name_study += "_lw"
      else:
            print("MODE NOT RECOGNIZED")

      if loss_func=="layer_wise":
            name_study += "_loss_lw"
      elif loss_func!="classic":
            print("LOSS FUNCTION NOT RECOGNIZED")

      if mode=="layer_wise" and loss_func=="layer_wise":
            if test_one_layer_pruning:
                  name_study += "_one_l"
            elif threshold_weights!=1:
                  name_study += "_"+convert_to_small_str(threshold_weights)

      if method_pruning!="schedule":
            name_study += "_"+method_pruning+"_"+convert_to_small_str(threshold_restart)

      if num_workers!=0:
            name_study+="_n_w_"+convert_to_small_str(num_workers)
      if "magnitude_with_z" in type_pruning:
            name_study+="_m_w_z"
      elif "_H" in type_pruning:
            name_study+="_H"
      if type_reset=="layer_wise":
            name_study+="_res_lw"
      elif type_reset=="ensemble":
            name_study+="_res_ens"
      else:
            print("TYPE RESET NOT RECOGNIZED")

      if tol_ent_reg!=1e-2:
            name_study+="_te_"+convert_to_small_str(tol_ent_reg)
      if tol_sel_reg!=1e-2:
            name_study+="_ts_"+convert_to_small_str(tol_sel_reg)
      if not(test_mult_reset):
            name_study+="_no_m"
      if test_reset_to_orignal:
            name_study+="_to_ori"

      name_study+="_gam_"+convert_to_small_str(gamma)
      name_study+="_sel_"+convert_to_small_str(selection_reg)
      name_study+="_ent_"+convert_to_small_str(entropy_reg)
      name_study+="_l2_"+convert_to_small_str(l2_reg)
      if l2_original_reg!=0:
            name_study+="_ori_"+convert_to_small_str(l2_original_reg)
      name_study+="_wd_"+convert_to_small_str(weight_decay)
      name_study+="_mom_"+convert_to_small_str(momentum)

      if pruning_rate_cte!=-1:
            name_study+="_cte_sch_"+convert_to_small_str(pruning_rate_cte)

      if test_constraint_weights:
            name_study+="_lag"

      if str(pretrained).lower()!="true":
            name_study+="_scratch"
      name_study+="_"+metric_best_model
      if goal_sparsity!=0.0:
            name_study+="_"+convert_to_small_str(goal_sparsity)
            if type_compute_sparsity == "prunable":
                  name_study += "_prun"
            if test_prop_goal_sparsity==1:
                  name_study+="_prop"
            if goal_sparsity_discrete!=goal_sparsity and test_start_sparse_gpt:
                  name_study+="_"+convert_to_small_str(goal_sparsity_discrete)
            if n_incr_gradual_pruning!=-1:
                  if type_pruning_schedule=="exponential":
                        letter_to_add = "e"
                  elif type_pruning_schedule=="linear":
                        letter_to_add = "l"
                  name_study+=f"_gp{letter_to_add}_"+convert_to_small_str(n_incr_gradual_pruning)
      if n_restart!=0:
            name_study+="_n_res_"+convert_to_small_str(n_restart)
      if tol_z_1!=1.0:
            name_study+="_tol_z_"+convert_to_small_str(tol_z_1)
      if seed!=0:
            name_study+="_seed_"+convert_to_small_str(seed)
      if test_different_lr!=0:
            name_study+="_dlr"
      if lambda_loss!=1:
            name_study+="_ll_"+convert_to_small_str(lambda_loss)
      if mode=="layer_wise" and loss_func=="layer_wise":
            if test_repeat_if_sparsity_not_reached:
                  name_study+="_tr"
            if loss_last_block=="layer_wise":
                  name_study+="_llb_lw"
                  if retrain_last_block:
                        name_study+="_rt"
      if n_repeat!=1:
            name_study+="_"+convert_to_small_str(n_repeat)
      return name_study


def test_trial_done(name_study, n_trials, n_repeats, folder_saves = "TSML_saves"):
      test_deja_train = False
      n_trials_done = 0
      save_study_done = None
      test_file_name = False
      try:
            if ("study_"+name_study) in os.listdir(folder_saves):
                  test_file_name = True
                  if len(os.listdir(folder_saves+"/study_"+name_study))>=1:
                        #     with open(folder_saves+"/study_"+name_study+"/save_study.pkl", "rb") as f:
                        #         save_study_done = pickle.load(f)
                        for ind_repeat in range(n_repeats):
                              with open(folder_saves+"/study_"+name_study+"/best_trial/repeat_"+str(ind_repeat)+"/results.json", "r") as f:
                                    dict_results = json.load(f)
                        #     n_trials_done = len(save_study_done.trials)
                        device = 'cuda' if torch.cuda.is_available() else 'cpu'
                        params = torch.load(folder_saves+"/study_"+name_study+"/best_trial/repeat_"+str(ind_repeat)+"/model", map_location=device).keys()
                        test_deja_train = True
      except:
            pass
      #     if n_trials_done>=n_trials:
      #         test_deja_train = True
      return test_deja_train, n_trials_done, test_file_name

def summarize_results(l_tables, path_save = None, type_of_task = "regression", metric_best_model="auc", n_scripts=10, evaluation_metric="acc"):
      l_data = []
      data_final = pd.DataFrame()
      data_total = pd.DataFrame()
      n_scripts_done = 0
      for l_d_table in l_tables:
            if len(l_d_table)==0:
                  continue
            print("Loading",len(l_d_table), "tables")
            data = pd.DataFrame()
            for d_table in l_d_table:
                  common_params = {}
                  diff_params = {}
                  for key in d_table:
                        if len(d_table[key])>1:
                              diff_params[key] = d_table[key]
                        else:
                              common_params[key] = d_table[key][0]
                  data_temp = pd.DataFrame()
                  l_params = list(diff_params.items())
                  l_params_values = [l_params[i][1] for i in range(len(l_params))]
                  l_params_names = [l_params[i][0] for i in range(len(l_params))]
                  l_trials_values_temp = list(itertools.product(*l_params_values))
                  l_names_values_temp = [l_params_names for i in range(len(l_trials_values_temp))]
                  for key in common_params:
                        data_temp[key]=[common_params[key] for _ in range(len(l_names_values_temp))]
                  for i in range(len(diff_params)):
                        data_temp[l_names_values_temp[0][i]]=[l_trials_values_temp[k][i] for k in range(len(l_trials_values_temp))]
                  l_trials_done = []
                  if type_of_task=="regression":
                        l_train_metric = []
                        l_val_metric = []
                        l_test_metric = []
                        l_train_metric_MAD = []
                        l_val_metric_MAD = []
                        l_test_metric_MAD = []
                  else:
                        l_train_metric = []
                        l_val_metric = []
                        l_test_metric = []
                        l_train_auc = []
                        l_val_auc = []
                        l_test_auc = []
                        l_train_metric_MAD = []
                        l_val_metric_MAD = []
                        l_test_metric_MAD = []
                        l_train_auc_MAD = []
                        l_val_auc_MAD = []
                        l_test_auc_MAD = []
                  l_lr = []
                  l_best_ep = []
                  l_time_training = []
                  l_n_z = []
                  l_n_z_MAD = []
                  l_sparsity = []
                  l_sparsity_MAD = []

                  #{"best_ep", "train_acc", "val_acc", "test_acc", "train_auc", "val_auc", "test_auc", "n_z", "n_params", "goal_sparsity_reached", "time_training"}
                  for i in range(len(data_temp)):
                        args_name = dict(data_temp.iloc[i])
                        list_keys = list(args_name.keys())
                        for key in list_keys:
                              if type(key)==tuple:
                                    values_key = args_name.pop(key)
                                    for k in range(len(key)):
                                          key_element = key[k]
                                          value_element = values_key[k]
                                          args_name[key_element] = value_element
                        
                        name_study = get_name_study(**args_name)

                        # ---- Get data
                        n_params_study = 0
                        if type_of_task=="classification":
                              train_metric_study = np.zeros((0,2))
                              val_metric_study =  np.zeros((0,2))
                              test_metric_study =  np.zeros((0,2))
                        elif type_of_task=="regression":
                              train_metric_study =  np.zeros((0,))
                              val_metric_study = np.zeros((0,))
                              test_metric_study = np.zeros((0,))
                        best_ep_study = 0
                        time_training_study = np.zeros((0,))
                        n_z = np.zeros((0,))
                        sparsity = np.zeros((0,))
                        n_repeat = args_name["n_repeat"]
                        n_scripts_done += 1
                        print("Progress:", str(np.round(100*n_scripts_done/n_scripts, 3))+"%", flush=True)
                        # try:
                        if not("n_trials" in args_name):
                              args_name["n_trials"] = 1
                        test_deja_train, n_trials_done, test_file_name = test_trial_done(name_study, args_name["n_trials"], n_repeats=args_name["n_repeat"], folder_saves=args_name["folder_saves"])
                        l_trials_done.append(n_trials_done)
                        # except:
                        # l_trials_done.append(np.nan)
                        for ind_repeat in range(n_repeat):
                              try:
                                    dict_params, in_sample_metric, validation_metric, test_metric, best_ep, time_training, n_z_current, n_params, sparsity_current  = read_results(name_study, ind_repeat=ind_repeat, type_of_task=type_of_task, folder_saves=args_name["folder_saves"])
                                    if sparsity_current == None:
                                          path_model = args_name["folder_saves"]+"/study_"+name_study+f"/best_trial/repeat_{ind_repeat}/model"
                                          saved_weights = torch.load(path_model)
                                          n_params_original = 0
                                          n_params_non_zero = 0
                                          for key in list(saved_weights.keys()):
                                                if "_z" not in key:
                                                      n_params_original += np.prod(saved_weights[key].shape)
                                                      n_params_non_zero += torch.sum(saved_weights[key]!=0).item()
                                          sparsity_current = 1-n_params_non_zero/n_params_original
                                    try:
                                          time_training_study = np.hstack([time_training_study, time_training])
                                    except:
                                          time_training_study = np.hstack([time_training_study, np.array([np.nan])])
                                          print("No time_training")
                                    try:
                                          n_z = np.hstack([n_z, n_z_current])
                                    except:
                                          n_z = np.hstack([n_z, np.nan])
                                          print("No n_z")
                                    try:
                                          sparsity = np.hstack([sparsity, sparsity_current])
                                    except:
                                          sparsity = np.hstack([sparsity, np.nan])
                                          print("No sparsity")
                                    try:
                                          if type_of_task == "regression":
                                                train_metric_study = np.hstack([train_metric_study, in_sample_metric])
                                                val_metric_study = np.hstack([val_metric_study, validation_metric])
                                                test_metric_study = np.hstack([test_metric_study, test_metric])
                                          else:
                                                if not(args_name["name_dataset"] in ["c4", "wikitext2", "ptb"]):
                                                      in_sample_metric[0] *= 100
                                                      validation_metric[0] *= 100
                                                      test_metric[0] *= 100
                                                train_metric_study = np.vstack([train_metric_study, in_sample_metric])
                                                val_metric_study = np.vstack([val_metric_study, validation_metric])
                                                test_metric_study = np.vstack([test_metric_study, test_metric])
                                    except:
                                          try:
                                                if not(args_name["name_dataset"] in ["c4", "wikitext2", "ptb"]):
                                                      in_sample_metric*=100
                                                      validation_metric*=100
                                                      test_metric*=100
                                                train_metric_study = np.hstack([train_metric_study, in_sample_metric])
                                                val_metric_study += np.hstack([val_metric_study, validation_metric])
                                                test_metric_study += np.hstack([test_metric_study, test_metric])
                                          except:
                                                print("No metric")
                                                if type_of_task=="regression":
                                                      train_metric_study = np.hstack([train_metric_study, np.array([np.nan])])
                                                      val_metric_study = np.hstack([val_metric_study, np.array([np.nan])])
                                                      test_metric_study = np.hstack([test_metric_study, np.array([np.nan])])
                                                elif type_of_task=="classification":
                                                      train_metric_study = np.vstack([train_metric_study, np.array([np.nan, np.nan])])
                                                      val_metric_study = np.vstack([val_metric_study, np.array([np.nan, np.nan])])
                                                      test_metric_study = np.vstack([test_metric_study, np.array([np.nan, np.nan])])
                                    try:
                                          lr_study = dict_params["lr"]
                                          best_ep_study += best_ep/n_repeat
                                    except:
                                          lr_study = np.nan
                                          best_ep_study = np.nan
                                          print("No lr/best_ep")
                                    try:
                                          n_params_study += n_params/n_repeat
                                    except:
                                          n_params_study = np.nan
                                          print("No n_params")
                              except:
                                    # try:
                                    #       dict_params, in_sample_metric, validation_metric, test_metric, best_ep, time_training, n_z_i, n_z_ij, n_features_used, n_params = read_results_keras(name_study, folder_saves=args_name["folder_saves"])
                                    # except:
                                    print("Couldn't read results", flush=True)
                                    time_training_study = np.hstack([time_training_study, np.array([np.nan])])
                                    if type_of_task=="regression":
                                          train_metric_study = np.hstack([train_metric_study, np.array([np.nan])])
                                          val_metric_study = np.hstack([val_metric_study, np.array([np.nan])])
                                          test_metric_study = np.hstack([test_metric_study, np.array([np.nan])])
                                    elif type_of_task=="classification":
                                          train_metric_study = np.vstack([train_metric_study, np.array([np.nan, np.nan])])
                                          val_metric_study = np.vstack([val_metric_study, np.array([np.nan, np.nan])])
                                          test_metric_study = np.vstack([test_metric_study, np.array([np.nan, np.nan])])
                                    lr_study = np.nan
                                    best_ep_study = np.nan
                                    n_params_study = np.nan
                                    n_z = np.nan
                                    sparsity = np.nan
                        # ----
                        numbers_decimal = 2
                        l_time_training.append(np.median(time_training_study))
                        l_n_z.append(np.median(n_z))
                        l_n_z_MAD.append(np.round(np.mean(np.abs(train_metric_study[:,0]-np.median(train_metric_study[:,0]))),numbers_decimal))
                        l_sparsity.append(np.median(sparsity))
                        l_sparsity_MAD.append(np.round(np.mean(np.abs(sparsity-np.median(sparsity))),numbers_decimal))
                        if type_of_task=="regression":
                              l_train_metric.append(np.round(np.median(train_metric_study),2))
                              l_val_metric.append(np.round(np.median(val_metric_study),2))
                              l_test_metric.append(np.round(np.median(test_metric_study),2))
                              l_train_metric_MAD.append(np.round(np.std(train_metric_study),numbers_decimal))
                              l_val_metric_MAD.append(np.round(np.std(val_metric_study),numbers_decimal))
                              l_test_metric_MAD.append(np.round(np.std(test_metric_study),numbers_decimal))
                        else:
                              l_train_metric.append(np.round(np.median(train_metric_study[:,0]),2))
                              l_train_auc.append(np.round(100*np.median(train_metric_study[:,1]),2))
                              l_val_metric.append(np.round(np.median(val_metric_study[:,0]),2))
                              l_val_auc.append(np.round(100*np.median(val_metric_study[:,1]),2))
                              l_test_metric.append(np.round(np.median(test_metric_study[:,0]),2))
                              l_test_auc.append(np.round(100*np.median(test_metric_study[:,1]),2))
                              l_train_metric_MAD.append(np.round(np.sum(np.abs(train_metric_study[:,0]-np.median(train_metric_study[:,0])))/n_repeat,numbers_decimal))
                              l_train_auc_MAD.append(np.round(100*np.sum(np.abs(train_metric_study[:,1]-np.median(train_metric_study[:,1])))/n_repeat,numbers_decimal))
                              l_val_metric_MAD.append(np.round(np.sum(np.abs(val_metric_study[:,0]-np.median(val_metric_study[:,0])))/n_repeat,numbers_decimal))
                              l_val_auc_MAD.append(np.round(100*np.sum(np.abs(val_metric_study[:,1]-np.median(val_metric_study[:,1])))/n_repeat,numbers_decimal))
                              l_test_metric_MAD.append(np.round(np.sum(np.abs(test_metric_study[:,0]-np.median(test_metric_study[:,0])))/n_repeat,numbers_decimal))
                              l_test_auc_MAD.append(np.round(100*np.sum(np.abs(test_metric_study[:,1]-np.median(test_metric_study[:,1])))/n_repeat,numbers_decimal))
                        l_lr.append(lr_study)
                        l_best_ep.append(best_ep_study)
                  data_temp["Number of completed trials"] = l_trials_done
                  if type_of_task=="regression":
                        data_temp["Train mse"] = l_train_metric
                        data_temp["Validation mse"] = l_val_metric
                        data_temp["Test mse"] = l_test_metric
                        data_temp["Train mse MAD"] = l_train_metric_MAD
                        data_temp["Validation mse MAD"] = l_val_metric_MAD
                        data_temp["Test mse MAD"] = l_test_metric_MAD
                  elif type_of_task=="classification":
                        data_temp[f"Train {evaluation_metric}"] = l_train_metric
                        data_temp[f"Train {evaluation_metric} MAD"] = l_train_metric_MAD
                        data_temp[f"Validation {evaluation_metric}"] = l_val_metric
                        data_temp[f"Validation {evaluation_metric} MAD"] = l_val_metric_MAD
                        data_temp[f"Test {evaluation_metric}"] = l_test_metric
                        data_temp[f"Test {evaluation_metric} MAD"] = l_test_metric_MAD
                        data_temp["Train auc"] = l_train_auc
                        data_temp["Train auc MAD"] = l_train_auc_MAD
                        data_temp["Validation auc"] = l_val_auc
                        data_temp["Validation auc MAD"] = l_val_auc_MAD
                        data_temp["Test auc"] = l_test_auc
                        data_temp["Test auc MAD"] = l_test_auc_MAD
                  data_temp["Sparsity"] = l_sparsity
                  data_temp["Sparsity MAD"] = l_sparsity_MAD
                  data_temp["Time training"] = l_time_training
                  data_temp["Number of z left"] = l_n_z
                  data_temp["Number of z left MAD"] = l_n_z_MAD
                  data_temp["Best epoch"] = l_best_ep
                  data_temp["Learning rate"] = l_lr
                  data = pd.concat([data, data_temp], ignore_index=True)
            if type_of_task=="regression":
                  l_data.append(data.sort_values(by="Validation mse"))
            elif type_of_task=="classification":
                  if metric_best_model=="acc":
                        l_data.append(data.sort_values(by="Validation acc", ascending=False))
                  elif metric_best_model=="auc":
                        l_data.append(data.sort_values(by="Validation auc", ascending=False))
                  elif metric_best_model=="ppl":
                        l_data.append(data.sort_values(by="Validation ppl", ascending=True))
      data_total = l_data[0]
      if type_of_task=="regression":
            data_final = pd.DataFrame(l_data[0].iloc[np.argmin(l_data[0]["Validation mse"])]).transpose()
      elif type_of_task=="classification":
            if metric_best_model=="acc":
                  data_final = pd.DataFrame(l_data[0].iloc[np.argmax(l_data[0]["Validation acc"])]).transpose()
            elif metric_best_model=="auc":
                  data_final = pd.DataFrame(l_data[0].iloc[np.argmax(l_data[0]["Validation auc"])]).transpose()
            elif metric_best_model=="ppl":
                  data_final = pd.DataFrame(l_data[0].iloc[np.argmin(l_data[0]["Validation ppl"])]).transpose()
      for data in l_data[1:]:
            data_total = pd.concat([data_total, data], ignore_index=True)
            if type_of_task=="regression":
                  data_final = pd.concat([data_final, pd.DataFrame(data.iloc[np.argmin(data["Validation mse"])]).transpose()], ignore_index=True)
            elif type_of_task=="classification":
                  if metric_best_model=="acc":
                        data_final = pd.concat([data_final, pd.DataFrame(data.iloc[np.argmax(data["Validation acc"])]).transpose()], ignore_index=True)
                  elif metric_best_model=="auc":
                        data_final = pd.concat([data_final, pd.DataFrame(data.iloc[np.argmax(data["Validation auc"])]).transpose()], ignore_index=True)
                  elif metric_best_model=="auc":
                        data_final = pd.concat([data_final, pd.DataFrame(data.iloc[np.argmin(data["Validation auc"])]).transpose()], ignore_index=True)
      if type_of_task=="regression":
            data_final = data_final.sort_values(by="Validation mse")
      elif type_of_task=="classification":
            if metric_best_model=="acc":
                  data_final = data_final.sort_values(by="Test acc", ascending=False)
            elif metric_best_model=="auc":
                  data_final = data_final.sort_values(by="Test auc", ascending=False)
            elif metric_best_model=="ppl":
                  data_final = data_final.sort_values(by="Test ppl", ascending=True)
      if type_of_task=="regression":
            data_total = data_total.sort_values(by="Validation mse")
      elif type_of_task=="classification":
            if metric_best_model=="acc":
                  data_total = data_total.sort_values(by="Validation acc", ascending=False)
            elif metric_best_model=="auc":
                  data_total = data_total.sort_values(by="Validation auc", ascending=False)
            elif metric_best_model=="ppl":
                  data_total = data_total.sort_values(by="Validation ppl", ascending=True)

      if path_save!=None:
            if os.path.exists(path_save):
                  print("Deleting old data ...")
                  shutil.rmtree(path_save)
            print("Saving the data...")
            os.mkdir(path_save)
            data_final.to_csv(path_save+"/data_final.csv")
            for i in range(len(l_data)):
                  l_data[i].to_csv(path_save+"/data_"+str(i)+".csv")
            data_total.to_csv(path_save+"/data_total.csv")
      return l_data, data_final

def get_grid_search(d_params):
    l_params = list(d_params.items())
    l_params_values = [l_params[i][1] for i in range(len(l_params))]
    l_params_names = [l_params[i][0] for i in range(len(l_params))]            
    l_trials_values = list(itertools.product(*l_params_values))
    for ind_name in range(len(l_params_names)):
        name = l_params_names[ind_name]
        if type(name)==tuple:
            for ind_trial_value in range(len(l_trials_values)):
                trial_value = l_trials_values[ind_trial_value]
                new_trial_value = trial_value[:ind_name]+trial_value[ind_name+1:]+trial_value[ind_name]
                l_trials_values[ind_trial_value] = new_trial_value
            l_params_names.pop(ind_name)
            l_params_names += list(name)
    l_names_values = [l_params_names for i in range(len(l_trials_values))]
    return l_trials_values, l_names_values

print("Import utils done")
# %%
