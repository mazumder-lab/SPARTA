#%%
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
#%%
import torch
from typing import Any, Optional
from torch import nn
from torch.nn.parameter import Parameter
import copy
import optuna
import itertools
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, LinearLR, CosineAnnealingLR, LambdaLR
import time
from tqdm import tqdm
from PIL import Image
from typing import Tuple
from utils_pruning import *

# from enum import Enum
from typing import cast
import inspect
import warnings
from torch.cuda.amp.grad_scaler import OptState

# https://www.kaggle.com/competitions/cs419m/data

# ------------------------
# --- Models per block ---
# ------------------------

import torch
#from new_pytorch_dataloader_2_0 import DataLoader
from torch.utils.data import DataLoader
# from multiprocessing import Manager
from utils_model import set_require_grad_rec, prune_models_external_sigmoid, set_phase_decoder_rec, get_phase_decoder_rec
from utils_dataset import *

import psutil

# ---------------------
# --- Class Dataset ---
# ---------------------

class Dataset(torch.utils.data.Dataset):
      'Characterizes a dataset for PyTorch'
      def __init__(self, x, y):
            'Initialization'
            self.x = x
            self.y = y

      def __len__(self):
            'Denotes the total number of samples'
            return self.x.shape[0]

      def __getitem__(self, index):
            'Generates one sample of data'
            return self.x[index], self.y[index]

def is_uninitialized_parameter(x: Any) -> bool:
    if not hasattr(nn.parameter, 'UninitializedParameter'):
        return False
    return isinstance(x, nn.parameter.UninitializedParameter)

def get_auc_combinations(data_numpy_train, y_numpy_train, data_numpy_val, y_numpy_val, l_combinations, idx_combination, seed):
    if seed==-1:
          model = DecisionTreeClassifier(max_depth=3)
    else:
          model = DecisionTreeClassifier(max_depth=3, random_state=seed)
          
    model.fit(data_numpy_train[:,l_combinations[idx_combination]], y_numpy_train)
    pred = model.predict_proba(data_numpy_val[:,l_combinations[idx_combination]])
    auc = compute_auc(y_numpy_val, pred)
    return auc

def get_mse_combinations(data_numpy_train, y_numpy_train, data_numpy_val, y_numpy_val, l_combinations, idx_combination, seed):
    if seed==-1:
          model = DecisionTreeRegressor(max_depth=3)
    else:
          model = DecisionTreeRegressor(max_depth=3, random_state=seed)
    model.fit(data_numpy_train[:,l_combinations[idx_combination]], y_numpy_train)
    pred = model.predict(data_numpy_val[:,l_combinations[idx_combination]])
    mse = np.mean((y_numpy_val-pred)**2)
    return mse


# --------------------------
# --- Training per block ---
# --------------------------

def loss_function_regression_without_penalization(x,y):
      return torch.nn.MSELoss()(x,y)

def loss_function_classification_without_penalization(x,y):
      if len(x.shape)>=2:
            return torch.nn.NLLLoss()(torch.log(x+1e-6),y)
      else:
            return torch.nn.BCELoss()(x,y)
      
def initialize_scheduler(type_decay, optimizer, gamma_lr_decay, n_epochs, period_milestones, start_lr_decay, end_lr_decay, T_max_cos, eta_min_cos, warmup_steps):
      if type_decay!="None":
            print("type_decay diff from None", type_decay)
            if type_decay=="divergence":
                  scheduler = ExponentialLR(optimizer, gamma=gamma_lr_decay)
                  current_min_loss = np.inf
                  acc_divergence = 0
            if type_decay=="exponential":
                  scheduler = ExponentialLR(optimizer, gamma=gamma_lr_decay)
            if type_decay=="multi_lr":
                  n_milesones = n_epochs//period_milestones
                  milestones = period_milestones*np.arange(n_milesones)[1:]
                  scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma_lr_decay)
            if type_decay=="linear":
                  scheduler = LinearLR(optimizer, start_factor=start_lr_decay, end_factor=end_lr_decay, total_iters=n_epochs)
            if type_decay=="cosine":
                  scheduler = CosineAnnealingLR(optimizer, T_max=T_max_cos, eta_min=eta_min_cos)
            if type_decay=="ramp":
                  def warmup(current_step: int):
                        if current_step < warmup_steps:
                              # current_step / warmup_steps * base_lr
                              return float(current_step / warmup_steps)
                        else:
                              # (num_training_steps - current_step) / (num_training_steps - warmup_steps) * base_lr
                              return max(0.0, float(n_epochs - current_step) / float(max(1, n_epochs - warmup_steps)))
                  scheduler = LambdaLR(optimizer, lr_lambda=warmup)
      else:
            scheduler = None
      return scheduler

def get_prop_reset_mnist(sparsity, first_term):
      if sparsity >= 0.975:
            prop_reset=min(first_term, 0.0001)
      elif 0.975 > sparsity >= 0.95:
            prop_reset=min(first_term, 0.0005)
      elif 0.95 > sparsity >= 0.9:
            prop_reset=min(first_term, 0.001)
      elif 0.9 > sparsity >= 0.85:
            prop_reset=min(first_term, 0.005)
      elif 0.85 > sparsity >= 0.80:
            prop_reset=min(first_term, 0.01)
      elif 0.80 > sparsity >= 0.75:
            prop_reset=min(first_term, 0.05)
      else:
            prop_reset=min(first_term, 0.1)
      return prop_reset

def get_prop_reset_resnet20(sparsity, first_term):
      # return get_prop_reset_mnist(sparsity, first_term)
      if sparsity >= 0.9:
            prop_reset=min(first_term, 0.0001)
      # elif 0.9 > sparsity >= 0.85:
      #       prop_reset=min(first_term, 0.005)
      # elif 0.85 > sparsity >= 0.80:
      #       prop_reset=min(first_term, 0.01)
      # elif 0.80 > sparsity >= 0.75:
      #       prop_reset=min(first_term, 0.05)

      # elif 0.9 > sparsity >= 0.6:
      #       prop_reset=min(first_term, 0.005)
      # elif 0.6 > sparsity >= 0.5:
      #       prop_reset=min(first_term, 0.01)
      # elif 0.5 > sparsity >= 0.35:
      #       prop_reset=min(first_term, 0.05)
      # else:
      #       prop_reset=min(first_term, 0.1)
      elif 0.9 > sparsity >= 0.2:
            prop_reset=min(first_term, 0.01)
      elif 0.2 > sparsity >= 0.15:
            prop_reset=min(first_term, 0.05)
      else:
            prop_reset=min(first_term, 0.1)
      return prop_reset

def get_layer_outputs(name, d_layer_output, test_detach):
    def hook(model, input, output):
            if test_detach:
                  d_layer_output[name] = output.detach()
            else:
                  d_layer_output[name] = output
    return hook

def compute_acc(model,dataloader,device='cpu',verbose=False):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    i = 0
    with torch.no_grad():
        for batch_sgd in tqdm(dataloader):
            input_batch_sgd, input_batch_original_sgd, target_batch_sgd, index_seen_sgd, old_index_seen_sgd = read_batch(batch_sgd)
            i+=1
            images, labels = input_batch_sgd.to(device), target_batch_sgd.to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            if len(labels.shape)>=2:
                labels = labels[:,0]
            correct += (predicted == labels).sum().item()
            if verbose and i%10 == 0:
                print(total, correct)

            del images,labels,outputs

    return 100 * correct / total

def compute_accuracy_two_models(model, original_model, dataloader, device='cpu', verbose=False):
      correct_original = 0
      correct = 0
      total = 0
      # since we're not training, we don't need to calculate the gradients for our outputs
      i = 0
      with torch.no_grad():
            for batch_sgd in tqdm(dataloader):
                  i+=1
                  input_batch_sgd, input_batch_original_sgd, target_batch_sgd, index_seen_sgd, old_index_seen_sgd = read_batch(batch_sgd)
                  input_batch_sgd, input_batch_original_sgd, target_batch_sgd = input_batch_sgd.to(device), input_batch_original_sgd.to(device), target_batch_sgd.to(device)
                  # calculate outputs by running images through the network
                  outputs = model(input_batch_sgd)
                  outputs_orignal = original_model(input_batch_original_sgd)
                  # the class with the highest energy is what we choose as prediction
                  _, predicted = torch.max(outputs.data, 1)
                  _, predicted_original = torch.max(outputs_orignal.data, 1)
                  total += target_batch_sgd.size(0)
                  correct += (predicted == target_batch_sgd).sum().item()
                  correct_original += (predicted_original == target_batch_sgd).sum().item()
                  if verbose and i%10 == 0:
                        print(total, correct)
      print("Model acc:", 100 * correct / total)
      print("Original model acc:", 100 * correct_original / total)
      return 100 * correct / total, 100 * correct_original / total

def get_item_mnist(self, index: int) -> Tuple[Any, Any]:
      """
      Args:
      index (int): Index

      Returns:
      tuple: (image, target) where target is index of the target class.
      """
      new_index = index-self.required_increment[index]
      img, target = self.data[new_index], int(self.targets[new_index])

      if self.is_original:
            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.fromarray(img.numpy(), mode="L")

            if self.transform is not None:
                  img = self.transform(img)

            if self.target_transform is not None:
                  target = self.target_transform(target)

      target = torch.Tensor([target, new_index, index]).long()

      if not(self.is_original):
            try:
                  sample_original = self.data_output_original[new_index]
            except:
                  sample_original = img
      
      if not(self.is_original):
            return img, sample_original, target
      else:
            return img, target

def get_item_cifar10(self, index: int) -> Tuple[Any, Any]:
      """
      Args:
      index (int): Index

      Returns:
      tuple: (image, target) where target is index of the target class.
      """
      new_index = index-self.required_increment[index]
      img, target = self.data[new_index], int(self.targets[new_index])

      if self.is_original:
            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.fromarray(img)

            if self.transform is not None:
                  img = self.transform(img)

            if self.target_transform is not None:
                  target = self.target_transform(target)

      target = torch.Tensor([target, new_index, index]).long()

      if not(self.is_original):
            try:
                  sample_original = self.data_output_original[new_index]
            except:
                  sample_original = img
      
      if not(self.is_original):
            return img, sample_original, target
      else:
            return img, target

def get_item_imagenet(self, index: int) -> Tuple[Any, Any]:
      """
      Args:
      index (int): Index

      Returns:
      tuple: (sample, target) where target is class_index of the target class.
      """
      if self.is_original:
            new_index = -1
            path, target = self.samples[index]
            target = int(target)
            sample = self.loader(path)
            if self.transform is not None:
                  sample = self.transform(sample)

            if self.target_transform is not None:
                  target = self.target_transform(target)
      else:
            new_index = (index - self.required_increment[index]).item()
            sample, target = self.data[new_index], self.targets[new_index]
            try:
                  sample_original = self.data_output_original[new_index]
            except:
                  sample_original = sample

      target = torch.Tensor([target, new_index, index]).long()
      
      if not(self.is_original):
            return sample, sample_original, target
      else:
            return sample, target

def initialize_dataset(dataset, n_train_kept, name_dataset):
      if n_train_kept==-1:
            sub_dataset = dataset
            if name_dataset == "imagenet":
                  sub_dataset.required_increment = np.zeros(len(sub_dataset.samples), dtype=int) #Manager().Array('i', np.zeros(len(self.samples), dtype=int))
            else:
                  sub_dataset.required_increment = np.zeros(dataset.data.shape[0], dtype=int) #Manager().Array('i', np.zeros(len(self.samples), dtype=int))
      else:
            sub_dataset = dataset.dataset
            sub_dataset.required_increment = np.zeros(n_train_kept, dtype=int) #Manager().Array('i', np.zeros(n_train_kept, dtype=int))
      sub_dataset.new_data = []
      sub_dataset.new_original_data = []
      sub_dataset.old_indices = []
      sub_dataset.is_original = True

def compute_metrics(model_wrapper, original_model, loader_train, loader_val, name_dataset, n_train_kept, type_of_task, criterion, test_early_stopping_used, module_training, is_last_module, loss_last_block, lambda_loss, scaler_y, test_update_loader, compute_train_loss):
      train_loss_with_pen = np.nan
      train_loss_with_no_pen = np.nan
      metric_train = np.nan
      model_wrapper.model.eval()
      if test_early_stopping_used==0:
            val_loss, metric_val = np.nan, np.nan
      else:
            if module_training:
                  if is_last_module and loss_last_block=="mce":
                        if name_dataset in ["c4", "wikitext2", "ptb"]:
                              print("Computing val ppl...")
                              val_loss, metric_val = get_loss_perplexity(model_wrapper, loader_val, criterion)
                        else:
                              print("Computing val acc and val loss...")
                              metric_val, val_loss, _ = get_loss_metric(model_wrapper, loader_val, type_of_task, criterion, scaler_y)
                  elif is_last_module and loss_last_block=="layer_wise":
                        # TO DO: TWO OPERATIONS HERE COULD BE REDUCED TO ONE
                        _, val_loss = get_loss_metric_module(model_wrapper, original_model, loader_val, type_of_task, criterion, scaler_y, test_update_loader=test_update_loader, name_dataset=name_dataset, n_train_kept=n_train_kept)
                        val_loss *= lambda_loss
                        if name_dataset in ["c4", "wikitext2", "ptb"]:
                              _, metric_val = get_loss_perplexity(model_wrapper, loader_val, criterion)
                        else:
                              metric_val = compute_acc(model_wrapper.model, loader_val, model_wrapper.device, verbose=False)
                  else:
                        print("Computing val loss...")
                        metric_val, val_loss = get_loss_metric_module(model_wrapper, original_model, loader_val, type_of_task, criterion, scaler_y, test_update_loader=test_update_loader, name_dataset=name_dataset, n_train_kept=n_train_kept)
                        metric_val = np.nan
                        val_loss *= lambda_loss
            else:
                  if type_of_task=="regression":
                        metric_val, val_loss, _ = get_loss_metric(model_wrapper, loader_val, type_of_task, criterion, scaler_y)
                  else:
                        if name_dataset in ["c4", "wikitext2", "ptb"]:
                              val_loss, metric_val = get_loss_perplexity(model_wrapper, loader_val, criterion)
                        else:
                              metric_val, val_loss, _ = get_loss_metric(model_wrapper, loader_val, type_of_task, criterion, scaler_y)

      if compute_train_loss:
            if module_training and (loss_last_block == "layer_wise" or not(is_last_module)):
                  _, train_loss_with_no_pen = get_loss_metric_module(model_wrapper, original_model, loader_train, type_of_task, criterion, scaler_y, test_update_loader=test_update_loader, name_dataset=name_dataset, n_train_kept=n_train_kept)
                  entropy_loss, selection_loss, l2_loss = model_wrapper.get_losses()
                  train_loss_with_pen = train_loss_with_no_pen + entropy_loss.item() + selection_loss.item() + l2_loss.item()
            else:
                  if name_dataset in ["c4", "wikitext2", "ptb"]:
                        train_loss_with_no_pen, metric_train = get_loss_perplexity(model_wrapper, loader_train, criterion)
                        entropy_loss, selection_loss, l2_loss = model_wrapper.get_losses()
                        train_loss_with_pen = train_loss_with_no_pen + entropy_loss.item() + selection_loss.item() + l2_loss.item()
                  else:
                        metric_train, train_loss_with_no_pen, train_loss_with_pen = get_loss_metric(model_wrapper, loader_train, type_of_task, criterion, scaler_y)
      model_wrapper.model.train()
      return train_loss_with_no_pen, train_loss_with_pen, metric_train, val_loss, metric_val

def actions_when_sparsity_becomes_reached(goal_sparsity, epoch, model_wrapper, d_named_parameters, test_early_stopping, loader_train, loader_val):  
      print("----", flush = True)
      print("Goal sparsity ("+str(goal_sparsity)+") reached at epoch", epoch, flush=True)
      print("----", flush = True)
      model_wrapper.freeze_all_z(d_named_parameters)
      if model_wrapper.type_function == "sigmoid":
            prune_models_external_sigmoid(model_wrapper.model, 1e-3)

      if test_early_stopping == 2:
            loader_train.dataset.indices = [i for i in range(len(loader_train.dataset.dataset))]
            loader_val.dataset.indices = []

# class OptState(Enum):
#     READY = 0
#     UNSCALED = 1
#     STEPPED = 2

# From Pytorch
def new_gradient_scaler_step(gradient_scaler, optimizer, test_normalized_sgd, d_named_parameters, model_wrapper, *args, **kwargs):
      """
      :meth:`step` carries out the following two operations:

      1.  Internally invokes ``unscale_(optimizer)`` (unless :meth:`unscale_` was explicitly called for ``optimizer``
      earlier in the iteration).  As part of the :meth:`unscale_`, gradients are checked for infs/NaNs.
      2.  If no inf/NaN gradients are found, invokes ``optimizer.step()`` using the unscaled
      gradients.  Otherwise, ``optimizer.step()`` is skipped to avoid corrupting the params.

      ``*args`` and ``**kwargs`` are forwarded to ``optimizer.step()``.

      Returns the return value of ``optimizer.step(*args, **kwargs)``.

      Args:
      optimizer (torch.optim.Optimizer):  Optimizer that applies the gradients.
      args:  Any arguments.
      kwargs:  Any keyword arguments.

      .. warning::
      Closure use is not currently supported.
      """
      if (not gradient_scaler._enabled):
            #torch.nn.utils.clip_grad_norm_(model_wrapper.model.parameters(), 1.0, "inf")
            torch.nn.utils.clip_grad_value_(model_wrapper.model.parameters(), 1.0)
            return optimizer.step(*args, **kwargs)

      if "closure" in kwargs:
            raise RuntimeError("Closure use is not currently supported if GradScaler is enabled.")

      gradient_scaler._check_scale_growth_tracker("step")

      optimizer_state = gradient_scaler._per_optimizer_states[id(optimizer)]

      if optimizer_state["stage"] is OptState.STEPPED:
            raise RuntimeError("step() has already been called since the last update().")

      retval = None

      if (hasattr(optimizer, "_step_supports_amp_scaling") and optimizer._step_supports_amp_scaling):
            kwargs_ = kwargs
            has_grad_scaler_kwarg = "grad_scaler" in inspect.signature(optimizer.step).parameters
            if has_grad_scaler_kwarg:
                  warnings.warn(
                        "GradScaler is going to stop passing itself as a keyword argument to the passed "
                        "optimizer. In the near future GradScaler registers `grad_scale: Tensor` and "
                        "`found_inf: Tensor` to the passed optimizer and let the optimizer use them directly.",
                        FutureWarning)
                  kwargs_.update({"grad_scaler": gradient_scaler})
            else:
                  if optimizer_state["stage"] is OptState.READY:
                        gradient_scaler._check_inf_per_device(optimizer)
                  scaler = gradient_scaler._get_scale_async()
                  found_inf = cast(
                        torch.Tensor,
                        sum([
                        t.to(scaler.device, non_blocking=True) for t in optimizer_state["found_inf_per_device"].values()
                        ])
                  )
                  optimizer.grad_scale = None if optimizer_state["stage"] == OptState.UNSCALED else scaler
                  optimizer.found_inf = found_inf
            retval = optimizer.step(*args, **kwargs_)
            optimizer_state["stage"] = OptState.STEPPED
            if not has_grad_scaler_kwarg:
                  del optimizer.grad_scale
                  del optimizer.found_inf
            return retval

      if optimizer_state["stage"] is OptState.READY:
            gradient_scaler.unscale_(optimizer)
            if test_normalized_sgd:
                  for x in d_named_parameters:
                        if "_z" in x:
                              if d_named_parameters[x].grad != None:
                                    d_named_parameters[x].grad[d_named_parameters[x].grad>0] /= d_named_parameters[x].grad[d_named_parameters[x].grad>0]
                                    d_named_parameters[x].grad[d_named_parameters[x].grad<0] /= -d_named_parameters[x].grad[d_named_parameters[x].grad<0]


      assert len(optimizer_state["found_inf_per_device"]) > 0, "No inf checks were recorded for this optimizer."

      retval = gradient_scaler._maybe_opt_step(optimizer, optimizer_state, *args, **kwargs)

      optimizer_state["stage"] = OptState.STEPPED

      return retval


def train_neural_network(name_study, name_model, model_wrapper, dataset, optimizer, criterion, n_epochs, batch_size_dataset, path_save, test_early_stopping, trial, test_save_all_models=False, type_decay="exponential", gamma_lr_decay=np.exp(-np.log(25)/10000), T_max_cos=10, eta_min_cos=1e-5, start_lr_decay=1e-2, end_lr_decay=1e-5, warmup_steps=100, type_of_task = "regression", test_compute_accurate_in_sample_loss = 0, folder_saves = "TSML_saves", ind_repeat=0, patience=50, metric_early_stopping="val_loss", period_milestones=25, goal_sparsity=0.0, type_training="combined", n_restart=0, num_workers=4, mode="ensemble", loss_func_and_model=("classic", None), is_last_module = False, module_training=False, name_dataset="mnist", n_train_kept = -1, n_rounds = -1, current_round = -1, test_normalized_sgd=0, pruning_rate_cte=-1, lambda_loss=1.0, test_repeat_if_sparsity_not_reached=1, loss_last_block="mce", retraining_of_last_block=False, copy_indices_train=None, copy_indices_val=None, test_adaptive_lr=False, patience_adaptive_lr=10, patience_freeze=1, test_wait_for_pruning=0, test_almost_sequential=0, tol_ent_reg=1e-2, tol_sel_reg=1e-2, test_update_dataset=True, test_one_layer_pruning=False):
      test_early_stopping_used = copy.deepcopy(test_early_stopping)

      loader_train, loader_val, loader_test, scaler_y = dataset

      model_wrapper.goal_sparsity = goal_sparsity
      loss_func, original_model = loss_func_and_model
      if loss_func == "layer_wise":
            d_layer_output_original = {}
            d_layer_output = {}
            d_modules_original = dict(original_model.named_modules())
            d_modules = dict(model_wrapper.model.named_modules())
            l_name_modules = list(d_modules.keys())
            l_name_modules = [x for x in l_name_modules if len(d_modules[x].__str__().lower().split(":"))==1 and (("conv" in d_modules[x].__str__().lower()) or ("linear_with" in d_modules[x].__str__().lower()))]

            if mode!="layer_wise" and not(module_training):
                  for name_module in l_name_modules:
                        d_modules[name_module].register_forward_hook(get_layer_outputs(name_module, d_layer_output, False))
                        d_modules_original[name_module].register_forward_hook(get_layer_outputs(name_module, d_layer_output_original, True))

      d_named_parameters = dict(model_wrapper.model.named_parameters())
      model_wrapper.unfreeze_all_z(d_named_parameters)

      if test_print_ram:
            print('1. Train RAM memory % used:', psutil.virtual_memory()[2], flush = True)
            print('1. Train final RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, flush = True)
            print('1. Train final Cuda memory:', torch.cuda.memory_allocated()/10**9)

      n_batches = len(loader_train)

      l_lr = []
      l_in_sample_loss = []
      l_in_sample_loss_no_pen = []
      l_in_sample_metric = []
      l_validation_loss = []
      l_validation_metric = []
      l_n_z = []
      l_n_params = []
      l_times_epochs = []
      l_sparsity = []
      l_sparsity_storage = []

      best_ep = -1
      # best_train_loss = np.inf
      # best_val_loss = np.inf
      # best_val_metric = -np.inf
      best_model = copy.deepcopy(model_wrapper.model)

      # BEGIN computing initial metrics
      print("Evaluating initial metrics...", flush=True)
      test_update_loader = False

      compute_train_loss = (test_early_stopping_used==0)

      train_loss_with_no_pen, train_loss_with_pen, metric_train, val_loss, metric_val = compute_metrics(model_wrapper, original_model, loader_train, loader_val, name_dataset, n_train_kept, type_of_task, criterion, test_early_stopping_used, module_training, is_last_module, loss_last_block, lambda_loss, scaler_y, test_update_loader, compute_train_loss)

      best_train_loss = np.inf

      if test_early_stopping_used==0:
            best_train_loss = train_loss_with_pen
            best_val_metric = metric_val
      else:
            best_val_loss = val_loss
            best_val_metric = metric_val
            print("--- Initial Validation Loss:", best_val_loss, "---", flush=True)
            print("--- Initial Validation Metric:", best_val_metric, "---", flush=True)

      print("Done", flush=True)
      # END
      
      # generator = torch.Generator()
      # if model_wrapper.seed != -1:
      #       torch.random.manual_seed(model_wrapper.seed)
      #       generator = generator.manual_seed(model_wrapper.seed)
      
      # loader_train = DataLoader(train_dataset, batch_size=batch_size_dataset, shuffle = True, generator=generator, num_workers=num_workers, pin_memory=True)
      # loader_val = DataLoader(validation_dataset, batch_size=batch_size_dataset, num_workers=num_workers, pin_memory=True)
      # loader_test = DataLoader(test_dataset, batch_size=batch_size_dataset, num_workers=num_workers, pin_memory=True)

      # Initialize z and sparsity
      model_wrapper.compute_z()
      if type_training == "combined":
            n_restart = 0.5

      if test_print_ram:
            print('2. Train RAM memory % used:', psutil.virtual_memory()[2], flush = True)
            print('2. Train final RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, flush = True)
            print('2. Train final Cuda memory:', torch.cuda.memory_allocated()/10**9)

      l_initial_lr = []
      if optimizer!=None:
            for idx_group_param in range(len(optimizer.param_groups)):
                  l_initial_lr.append(optimizer.param_groups[idx_group_param]["lr"])
            
      epoch_counter = -1
      number_of_epochs_increase = 0
      max_number_of_epochs_increase = 2

      acc_nan_loss = 0

      if mode == "layer_wise":
            model_wrapper.set_require_grad(False)
            if loss_func!="layer_wise":
                  d_modules = dict(model_wrapper.model.named_modules())
                  l_name_modules = list(d_modules.keys())
                  l_name_modules = [x for x in l_name_modules if len(d_modules[x].__str__().lower().split(":"))==1 and (("conv" in d_modules[x].__str__().lower()) or ("linear_with" in d_modules[x].__str__().lower()))]
            l_name_modules += [-1]
      else:
            l_name_modules = [-1]
      
      acc_module = -1
      # scheduler = initialize_scheduler(type_decay, optimizer, gamma_lr_decay, n_epochs, period_milestones, start_lr_decay, end_lr_decay, T_max_cos, eta_min_cos, warmup_steps)
      activation = {}

      if test_print_ram:
            print('3. Train RAM memory % used:', psutil.virtual_memory()[2], flush = True)
            print('3. Train final RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, flush = True)
            print('3. Train final Cuda memory:', torch.cuda.memory_allocated()/10**9)

      # if name_dataset in ["c4", "wikitext2", "ptb"]:
      #       print("Initializing layer norm module...", flush=True)
      #       for batch_sgd in tqdm(loader_train):
      #             input_batch_sgd, input_batch_original_sgd, target_batch_sgd, index_seen_sgd, old_index_seen_sgd = read_batch(batch_sgd)
      #             model_wrapper.model(input_batch_sgd.to(model_wrapper.device))

      for name_module in l_name_modules:
            if mode == "layer_wise" and name_module!=-1:
                  if loss_func=="layer_wise" and not(module_training):
                        handle_original = d_modules_original[name_module].register_forward_hook(get_layer_outputs(name_module, d_layer_output_original, True))
                        handle = d_modules[name_module].register_forward_hook(get_layer_outputs(name_module, d_layer_output, False))
            elif mode == "layer_wise" and name_module==-1:
                  model_wrapper.set_require_grad(True)
            
            sparsity = model_wrapper.get_sparsity()
            if goal_sparsity==0 and model_wrapper.n_params_original_z==0:
                  sparsity=0
            test_sparsity_reached = (sparsity>=goal_sparsity)

            if test_sparsity_reached:
                  epoch = -1
                  actions_when_sparsity_becomes_reached(goal_sparsity, epoch, model_wrapper, d_named_parameters, test_early_stopping, loader_train, loader_val)
                  if test_early_stopping == 2:
                        test_early_stopping_used = 0

            # if mode=="layer_wise":
            #       try:
            #             if sparsity_total>=goal_sparsity:
            #                   test_sparsity_reached = True
            #       except:
            #             pass
            
            initial_selection_reg = copy.deepcopy(model_wrapper.selection_reg)
            initial_entropy_reg = copy.deepcopy(model_wrapper.entropy_reg)
            just_reset = False
            # prop_reset_const = 0.1

            acc_no_sparisity_change = 0
            n_epochs_no_improvement = 0
            n_epochs_small_improvements = 0
            n_mult_sel = 0
            n_mult_ent = 0
            n_epochs_no_improvement_freeze = 0
            phase_freeze = False
            acc_module+=1
            if acc_module>=1:
                  print("--- Loading best weights previous layer ---")
                  with torch.no_grad():
                        l_param_model = list(model_wrapper.model.parameters())
                        l_param_best_model = list(best_model.parameters())
                        for j in range(len(l_param_model)):
                              l_param_model[j].data = copy.deepcopy(l_param_best_model[j].data)
                  print("--- Done ---")
            if mode == "layer_wise" and name_module!=-1:
                  d_modules[name_module].requires_grad_(True)
            for i in range(int(2*n_restart)):
                  if type_training == "alternate":
                        if i%2==0:
                              # n_epochs = 3
                              # T_max_cos = 3
                              # Z training phase
                              if i>=1:
                                    # Load best weights of the weight training phase
                                    with torch.no_grad():
                                          l_param_model = list(model_wrapper.model.parameters())
                                          l_param_best_model = list(best_model.parameters())
                                          for j in range(len(l_param_model)):
                                                l_param_model[j].data = copy.deepcopy(l_param_best_model[j].data)
                              model_wrapper.phase_training_z()
                        else:
                              # n_epochs = 2
                              # T_max_cos = 2
                              # Weight training phase
                              model_wrapper.phase_training_weight()
                              if i==(2*n_restart-1):
                                    n_epochs = int(n_epochs*2.5)
                                    n_epochs = 12
                                    T_max_cos = 12

                  # if name_dataset=="imagenet" and module_training and model_wrapper.ind_model_wrap==0:
                  #       n_epochs_used = 1
                  # else:
                  #       n_epochs_used = n_epochs
                  # TO DELETE LATER
                  n_epochs_used = copy.deepcopy(n_epochs)
                  if T_max_cos==-1:
                        T_max_cos_used = n_epochs_used
                  else:
                        T_max_cos_used = T_max_cos

                  # END

                  if optimizer != None:
                        scheduler = initialize_scheduler(type_decay, optimizer, gamma_lr_decay, n_epochs, period_milestones, start_lr_decay, end_lr_decay, T_max_cos_used, eta_min_cos, warmup_steps)

                        for idx_group_param in range(len(optimizer.param_groups)):
                              optimizer.param_groups[idx_group_param]["lr"] = l_initial_lr[idx_group_param]

                  # To delete later
                  model_wrapper.step_temp = 0
                  # End
                  if "layer_wise" in model_wrapper.type_pruning or "smallest_grad" in model_wrapper.type_pruning:
                        model_wrapper.initialize_pruning()
                  
                  

                  n_z = model_wrapper.get_n_z(test_grad=True, include_batchnorm=False)
                  
                  # if n_train_kept == -1:
                  #       loader_train.dataset.dataset.targets = torch.Tensor(loader_train.dataset.dataset.targets)
                  # else:
                  #       loader_train.dataset.dataset.dataset.targets = torch.Tensor(loader_train.dataset.dataset.dataset.targets)
                  
                  # # TO DELETE, SANITY CHECK
                  # if module_training and is_last_module:
                  #       n_epochs_used = 1
                  # # END
                  
                  if module_training and (n_z == 0 or n_epochs_used == 0) and not(is_last_module):
                        # Transform dataset
                        add_text = ""
                        if test_one_layer_pruning and get_phase_decoder_rec(model_wrapper.model)!=0:
                              add_text += f" - Phase {get_phase_decoder_rec(model_wrapper.model)}"
                        if retraining_of_last_block:
                              add_text += " (Retraining)"
                        print(f"Round {current_round+1}/{n_rounds}"+add_text, flush = True)
                        if test_almost_sequential in [1, 3]:
                              test_update_original = False
                        else:
                              test_update_original = True
                        if test_update_dataset:
                              print("Updating dataset ...", flush=True)
                              update_dataset(model_wrapper.model, model_wrapper.device, original_model, loader_train, loader_val, n_train_kept, test_update_original, copy_indices_train, copy_indices_val, test_almost_sequential)
                        print("Done", flush=True)
                        # End transform dataset
                        n_epochs_used = 0

                  print("Number of epochs:", n_epochs_used, flush=True)

                  # # TO DELETE, SANITY CHECK
                  # test_update_loader = False
                  # metric_val, val_loss = get_loss_metric_module(model_wrapper, original_model, loader_val, type_of_task, criterion, scaler_y, test_update_loader=test_update_loader, name_dataset=name_dataset, n_train_kept=n_train_kept)
                  # import ipdb;ipdb.set_trace()
                  # # END
                  epoch = -1

                  # if name_dataset in ["c4", "wikitext2", "ptb"]:
                  #       use_amp = True
                  # else:
                  #       use_amp = False
                  use_amp = False

                  gradient_scaler = torch.cuda.amp.GradScaler(enabled = use_amp)

                  while epoch < n_epochs_used-1:
                        epoch += 1
                        if test_print_ram:
                              print('4. Train RAM memory % used:', psutil.virtual_memory()[2], flush = True)
                              print('4. Train final RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, flush = True)
                              print('4. Train final Cuda memory:', torch.cuda.memory_allocated()/10**9)
                        if (n_epochs_no_improvement < patience):# or not(test_early_stopping_used):
                              epoch_counter+=1
                              start_epoch = time.time()
                              if test_sparsity_reached and test_adaptive_lr:
                                    print("n_epochs_small_improvements = ", n_epochs_small_improvements)
                              if test_sparsity_reached:
                                    print("n_epochs_no_improvement =", n_epochs_no_improvement)
                              elif not(phase_freeze):
                                    n_z_close_to_1 = model_wrapper.get_n_z_close_to_1(test_grad=True, include_batchnorm=False)
                                    n_z = model_wrapper.get_n_z(test_grad=True, include_batchnorm=False)
                                    ratio_z_to_1 = n_z_close_to_1/n_z
                                    # Version 1
                                    if epoch != 0:
                                          if not(just_reset) and ratio_z_to_1-old_ratio<=tol_ent_reg and sparsity-old_sparsity<=tol_sel_reg:
                                                n_mult_ent += 1
                                                model_wrapper.entropy_reg *= 2
                                          elif sparsity-old_sparsity<=tol_sel_reg:
                                                n_mult_sel += 1
                                                model_wrapper.selection_reg *= 2
                                                acc_no_sparisity_change += 1
                                          if sparsity<old_sparsity:
                                                print("-------- Sparsity become smaller --------")
                                          # if epoch >= 2:
                                          #       print("diff entropy loss: ", (old_entropy_loss-approx_entropy_loss)/model_wrapper.entropy_reg)
                                          #       print("diff ratio: ", ratio_z_to_1-old_ratio)
                                          #       print("diff selection loss: ", (old_selection_loss-approx_selection_loss)/model_wrapper.selection_reg)
                                          #       print("diff sparstiy: ", np.abs(sparsity-old_sparsity))
                                    # End version 1

                                    # Version 2
                                    # if epoch >=2:
                                    #       diff_ent_loss = (old_entropy_loss-approx_entropy_loss)/model_wrapper.entropy_reg
                                    #       diff_sel_loss = (old_selection_loss-approx_selection_loss)/model_wrapper.selection_reg
                                    #       if diff_ent_loss<=tol_ent_reg and diff_sel_loss<=tol_sel_reg:
                                    #             n_mult_ent += 1
                                    #             model_wrapper.entropy_reg *= 2
                                    #       elif diff_sel_loss<=tol_sel_reg:
                                    #             n_mult_sel += 1
                                    #             model_wrapper.selection_reg *= 2
                                    #             acc_no_sparisity_change += 1
                                    #       if epoch >= 2:
                                    #             print("diff entropy loss: ", diff_ent_loss)
                                    #             print("diff ratio: ", ratio_z_to_1-old_ratio)
                                    #             print("diff selection loss: ", diff_sel_loss)
                                    #             print("diff sparstiy: ", np.abs(sparsity-old_sparsity))
                                    # End version 2

                                    # if acc_no_sparisity_change>=3:
                                    #       acc_no_sparisity_change = 0
                                    #       model_wrapper.selection_reg *= 2
                                    old_ratio = copy.deepcopy(ratio_z_to_1)
                                    # if epoch>=1:
                                    #       old_entropy_loss = copy.deepcopy(approx_entropy_loss)
                                    #       old_selection_loss = copy.deepcopy(approx_selection_loss)
                                    just_reset = False
                                    print("Ratio of z equal to 1:", ratio_z_to_1, flush=True)
                                    if ratio_z_to_1 >= 0.95:
                                          first_term = 1.05*(goal_sparsity-sparsity+1e-4)
                                          if model_wrapper.method_pruning=="threshold":
                                                prop_reset = first_term
                                          elif model_wrapper.method_pruning in ["schedule", "both"] and pruning_rate_cte==-1:
                                                if "mlpnet" in name_model:
                                                      prop_reset = get_prop_reset_mnist(sparsity, first_term)
                                                elif "resnet20" in name_model:
                                                      prop_reset = get_prop_reset_resnet20(sparsity, first_term)
                                                elif "resnet50" in name_model:
                                                      prop_reset = get_prop_reset_mnist(sparsity, first_term)
                                                else:
                                                      prop_reset = get_prop_reset_mnist(sparsity, first_term)
                                          elif model_wrapper.method_pruning in ["schedule", "both"] and pruning_rate_cte!=-1:
                                                prop_reset = min(first_term, pruning_rate_cte)
                                          else:
                                                print("ERROR: method_pruning should be either threshold, schedule or both")
                                          if mode=="layer_wise":
                                                prop_reset = np.nanmax([prop_reset, 1/n_z, 1/model_wrapper.n_params_original_z])
                                          
                                          # model_wrapper.sparsity_level_selection += prop_reset
                                          #model_wrapper.sparsity_level_selection = 0.9
                                          
                                          # TO DELETE
                                          # if model_wrapper.selection_lagrangian_reg!=None:
                                          #       prop_reset = 1.0
                                          #       #prop_reset=min(first_term, 0.1)
                                          # else:
                                          #       prop_reset=min(first_term, prop_reset_const)
                                          # END
                                          
                                          # TO DELETE
                                          # prop_reset = 1.0
                                          # END
                                          model_wrapper.compute_z()
                                          if model_wrapper.test_reset_to_orignal:
                                                model_wrapper.model.load_state_dict(original_model.state_dict(), strict=False)
                                          n_reset = model_wrapper.reset_z(prop_reset=prop_reset)
                                          if n_mult_sel>=2:
                                                model_wrapper.selection_reg /= 4
                                                n_mult_sel -= 2
                                          if n_mult_ent>=2:
                                                model_wrapper.entropy_reg /= 4
                                                n_mult_ent -= 2
                                          print("-----", flush=True)
                                          print(str(n_reset)+" z-weights have been reset", n_z_close_to_1/n_z, flush=True)
                                          print("-----", flush=True)
                                          
                                          just_reset = True
                                          # weight_z_list = list(model_wrapper.model.named_modules())
                                          # weight_z_concat = torch.cat([x[1].weight_z.view(-1) for x in weight_z_list if ("conv" in x[0] or "fc" in x[0])])
                                          # if len(weight_z_concat[weight_z_concat<1].view(-1))>0:
                                          #       print(weight_z_concat[weight_z_concat<1])
                                          # print((list(model_wrapper.model.modules()))[1])

                              loss_pred_in_sample = 0
                              approx_loss_in_sample_with_pen = 0
                              approx_entropy_loss = 0
                              approx_selection_loss = 0
                              approx_l2_loss = 0
                              if type_of_task=="classification":
                                    approx_metric_train = 0
                              model_wrapper.model.train()
                              current_lr = optimizer.param_groups[0]["lr"]
                              print("current lr:", current_lr)
                              n_seen = 0
                              acc_batch = 0
                              if "layer_wise" in model_wrapper.type_pruning or "smallest_grad" in model_wrapper.type_pruning:
                                    model_wrapper.reinitialize_pruning()
                              
                              for batch_sgd in tqdm(loader_train):
                                    if test_print_ram:
                                          print('1. RAM memory % used:', psutil.virtual_memory()[2], flush = True)
                                          print('1. RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, flush = True)
                                          print('1. Cuda memory:', torch.cuda.memory_allocated()/10**9)
                                    # print("HERE:", loader_train.dataset.dataset.index_seen)
                                    if test_print_time:
                                          try:
                                                print("Time 0:", time.time()-start_time_loop)
                                          except:
                                                start_time_loop = time.time()
                                    input_batch_sgd, input_batch_original_sgd, target_batch_sgd, index_seen_sgd, old_index_seen_sgd = read_batch(batch_sgd)
                                    acc_batch += 1
                                    model_wrapper.step_temp += 1
                                    n_batch = input_batch_sgd.shape[0]
                                    n_seen += n_batch
                                    optimizer.zero_grad()
                                    if test_print_time:
                                          print("Time 0 bis:",time.time()-start_time_loop)
                                          start_time_loop = time.time()
                                    if test_print_ram:
                                          print('2. RAM memory % used:', psutil.virtual_memory()[2], flush = True)
                                          print('2. RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, flush = True)
                                          print('2. Cuda memory:', torch.cuda.memory_allocated()/10**9)

                                    # with torch.autocast(device_type=model_wrapper.device, dtype=torch.float16, enabled=use_amp):
                                    output = model_wrapper.model(input_batch_sgd.to(model_wrapper.device))
                                    # else:
                                    #       output = model_wrapper.model(input_batch_sgd.to(model_wrapper.device))

                                    if test_print_time:
                                          print("Time 1:",time.time()-start_time_loop)
                                          start_time_loop = time.time()
                                    if test_print_ram:
                                          print('3. RAM memory % used:', psutil.virtual_memory()[2], flush = True)
                                          print('3. RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, flush = True)
                                          print('3. Cuda memory:', torch.cuda.memory_allocated()/10**9)
                                    if model_wrapper.type_pruning=="layer_wise":
                                          tot_layer_wise_loss = model_wrapper.compute_layer_wise_loss()
                                          tot_layer_wise_loss.backward(retain_graph = True)
                                          model_wrapper.save_grad_layer_wise()
                                          optimizer.zero_grad()
                                    if test_print_time:
                                          print("Time 1 bis:",time.time()-start_time_loop)
                                          start_time_loop = time.time()
                                    if test_print_ram:
                                          print('4. RAM memory % used:', psutil.virtual_memory()[2], flush = True)
                                          print('4. RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, flush = True)
                                          print('4. Cuda memory:', torch.cuda.memory_allocated()/10**9)

                                    y_truth = target_batch_sgd
                                    y_truth = y_truth.to(model_wrapper.device)
                                    if test_print_time:
                                          print("Time 2:",time.time()-start_time_loop)
                                          start_time_loop = time.time()
                                    if test_print_ram:
                                          print('5. RAM memory % used:', psutil.virtual_memory()[2], flush = True)
                                          print('5. RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, flush = True)
                                          print('5. Cuda memory:', torch.cuda.memory_allocated()/10**9)

                                    if test_print_time:
                                          print("Time 3:",time.time()-start_time_loop)
                                          start_time_loop = time.time()
                                    if test_print_ram:
                                          print('6. RAM memory % used:', psutil.virtual_memory()[2], flush = True)
                                          print('6. RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, flush = True)
                                          print('6. Cuda memory:', torch.cuda.memory_allocated()/10**9)
                                    if loss_func == "layer_wise":
                                          with torch.no_grad():
                                                # if name_dataset in ["c4", "wikitext2", "ptb"]:
                                                #       with torch.autocast(device_type=model_wrapper.device, dtype=torch.float16):
                                                if test_almost_sequential==2 and not(test_sparsity_reached):
                                                      output_original = original_model(input_batch_sgd.to(model_wrapper.device))
                                                else:
                                                      output_original = original_model(input_batch_original_sgd.to(model_wrapper.device))
                                                # else:
                                                #       output_original = original_model(input_batch_original_sgd.to(model_wrapper.device))
                                                if test_print_time:
                                                      print("Time 4:",time.time()-start_time_loop)
                                                      start_time_loop = time.time()
                                                if test_print_ram:
                                                      print('7. RAM memory % used:', psutil.virtual_memory()[2], flush = True)
                                                      print('7. RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, flush = True)
                                                      print('7. Cuda memory:', torch.cuda.memory_allocated()/10**9)

                                          # if mode=="layer_wise" and name_module!=-1:
                                          #       loss = torch.mean((d_layer_output[name_module] - d_layer_output_original[name_module])**2)
                                          if name_module==-1 and (module_training and is_last_module and loss_last_block=="mce"):
                                                if name_dataset in ["c4", "wikitext2", "ptb"]:
                                                #       with torch.autocast(device_type=model_wrapper.device, dtype=torch.float16):
                                                      shift_logits = output[:, :-1, :].contiguous()
                                                      shift_labels = target_batch_sgd[:, 1:].contiguous()
                                                      shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                                                      loss = criterion(shift_logits, shift_labels.view(-1).to(model_wrapper.device))
                                                else:
                                                      loss = criterion(output, y_truth)
                                          elif module_training and (loss_last_block=="layer_wise" or not(is_last_module)):
                                                # if name_dataset in ["c4", "wikitext2", "ptb"]:
                                                #       with torch.autocast(device_type=model_wrapper.device, dtype=torch.float16):
                                                loss = torch.mean((output-output_original)**2)*lambda_loss
                                                # else:
                                                #       loss = torch.mean((output-output_original)**2)*lambda_loss
                                          else:
                                                loss = 0
                                                for key_layer in d_layer_output:
                                                      # if name_dataset in ["c4", "wikitext2", "ptb"]:
                                                      #       with torch.autocast(device_type=model_wrapper.device, dtype=torch.float16):
                                                      loss += torch.mean((d_layer_output[key_layer] - d_layer_output_original[key_layer])**2)*lambda_loss
                                                      # else:
                                                      #       loss += torch.mean((d_layer_output[key_layer] - d_layer_output_original[key_layer])**2)*lambda_loss
                                          if test_print_time:
                                                print("Time 5:",time.time()-start_time_loop)
                                                start_time_loop = time.time()
                                          if test_print_ram:
                                                print('8. RAM memory % used:', psutil.virtual_memory()[2], flush = True)
                                                print('8. RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, flush = True)
                                                print('8. Cuda memory:', torch.cuda.memory_allocated()/10**9)
                                          # Updating dataset
                                          # if epoch==(n_epochs_used-1) and not(is_last_module):
                                          #       update_loader_first(loader_train, output, target_batch_sgd, index_seen_sgd, old_index_seen_sgd, n_train_kept)
                                          # else:
                                          #       loader_train.dataset.dataset.index_seen = Manager().list()
                                          #       loader_train.dataset.dataset.old_index_seen = Manager().list()
                                          if test_print_time:
                                                print("Time 6:",time.time()-start_time_loop)
                                                start_time_loop = time.time()
                                          if test_print_ram:
                                                print('9. RAM memory % used:', psutil.virtual_memory()[2], flush = True)
                                                print('9. RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, flush = True)
                                                print('9. Cuda memory:', torch.cuda.memory_allocated()/10**9)

                                    else:
                                          # if name_dataset in ["c4", "wikitext2", "ptb"]:
                                          #       with torch.autocast(device_type=model_wrapper.device, dtype=torch.float16):
                                          shift_logits = output[:, :-1, :].contiguous()
                                          shift_labels = target_batch_sgd[:, 1:].contiguous()
                                          shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                                          loss = criterion(shift_logits, shift_labels.view(-1).to(model_wrapper.device))
                                          # else:
                                          #       loss = criterion(output, y_truth.long())
                                          if test_print_time:
                                                print("Time 7:",time.time()-start_time_loop)
                                                start_time_loop = time.time()
                                          if test_print_ram:
                                                print('9. RAM memory % used:', psutil.virtual_memory()[2], flush = True)
                                                print('9. RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, flush = True)
                                                print('9. Cuda memory:', torch.cuda.memory_allocated()/10**9)

                              
                                    #model_wrapper.compute_z()
                                    #n_z_close_to_1 = model_wrapper.get_n_z_close_to_1(test_grad=True, include_batchnorm=False)
                                    #n_weight_close_to_1 = model_wrapper.get_n_weigth_z_close_to_1(test_grad=True, include_batchnorm=False)
                                    #print(f"4. Number of z equal to 1: {n_z_close_to_1, n_weight_close_to_1}")

                                    loss_pred_in_sample += n_batch*loss.item()

                                    if test_print_time:
                                          print("Time 7 bis:",time.time()-start_time_loop)
                                          start_time_loop = time.time()
                                    # if name_dataset in ["c4", "wikitext2", "ptb"]:
                                    #       with torch.autocast(device_type=model_wrapper.device, dtype=torch.float16):
                                    entropy_loss, selection_loss, l2_loss = model_wrapper.get_losses()
                                    # else:
                                    #       entropy_loss, selection_loss, l2_loss = model_wrapper.get_losses()
                                    if test_print_time:
                                          print("Time 7 bis bis:",time.time()-start_time_loop)
                                          start_time_loop = time.time()
                                    # if model_wrapper.step_temp == 446:
                                    #       # if not(model_wrapper.dense_to_sparse):
                                    #       #       np.save("output_dense_445.npy", output.data.numpy())
                                    #       import ipdb;ipdb.set_trace()
                                    approx_selection_loss += n_batch*selection_loss.item()
                                    approx_entropy_loss += n_batch*entropy_loss.item()
                                    approx_l2_loss += n_batch*l2_loss.item()
                                    
                                    if is_last_module:
                                          if name_dataset in ["c4", "wikitext2", "ptb"]:
                                                if loss_func == "layer_wise":
                                                      shift_logits = output[:, :-1, :].contiguous()
                                                      shift_labels = target_batch_sgd[:, 1:].contiguous()
                                                      shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                                                      approx_metric_train += criterion(shift_logits, shift_labels.view(-1).to(model_wrapper.device)).float().detach() * n_batch # * model_wrapper.model.seqlen
                                                else:
                                                      approx_metric_train += loss.float().detach() * n_batch #* model_wrapper.model.seqlen
                                          else:
                                                approx_metric_train += 100*torch.sum(torch.argmax(torch.softmax(output, 1), 1) == y_truth).item()
                                    elif module_training:
                                          approx_metric_train = np.nan
                                    
                                    # if name_dataset in ["c4", "wikitext2", "ptb"]:
                                    #       with torch.autocast(device_type=model_wrapper.device, dtype=torch.float16):
                                    if np.isnan(loss.item()):
                                          print("---", flush = True)
                                          print(f"{acc_nan_loss}# in-sample loss is nan", flush = True)
                                          print("", flush = True)
                                          print("---", flush = True)
                                          acc_nan_loss+=1
                                    if np.isnan(entropy_loss.item()):
                                          print("---", flush = True)
                                          print(f"{acc_nan_loss}# entropy loss is nan", flush = True)
                                          print("", flush = True)
                                          print("---", flush = True)
                                          acc_nan_loss+=1
                                    if np.isnan(selection_loss.item()):
                                          print("---", flush = True)
                                          print(f"{acc_nan_loss}# selection loss is nan", flush = True)
                                          print("", flush = True)
                                          print("---", flush = True)
                                          acc_nan_loss+=1
                                    if np.isnan(l2_loss.item()):
                                          print("---", flush = True)
                                          print(f"{acc_nan_loss}# l2 loss is nan", flush = True)
                                          print("", flush = True)
                                          print("---", flush = True)
                                          acc_nan_loss+=1
                                    # if acc_nan_loss>0:
                                    #       import ipdb;ipdb.set_trace()
                                    loss += entropy_loss + selection_loss + l2_loss

                                    # else:
                                    #       loss += entropy_loss + selection_loss + l2_loss

                                    # End autocast
                                    approx_loss_in_sample_with_pen += n_batch*loss.item()
                                    if test_print_time:
                                          print("Time 8:",time.time()-start_time_loop)
                                          start_time_loop = time.time()
                                    if test_print_ram:
                                          print('10. RAM memory % used:', psutil.virtual_memory()[2], flush = True)
                                          print('10. RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, flush = True)
                                          print('10. Cuda memory:', torch.cuda.memory_allocated()/10**9)

                                    #if name_dataset in ["c4", "wikitext2", "ptb"]:
                                    gradient_scaler.scale(loss).backward()
                                    #else:
                                    #      loss.backward()  # Derive gradients.
                                    if test_normalized_sgd and not(use_amp):
                                          for x in d_named_parameters:
                                                if "_z" in x:
                                                      if d_named_parameters[x].grad != None:
                                                            d_named_parameters[x].grad[d_named_parameters[x].grad>0] /= d_named_parameters[x].grad[d_named_parameters[x].grad>0]
                                                            d_named_parameters[x].grad[d_named_parameters[x].grad<0] /= -d_named_parameters[x].grad[d_named_parameters[x].grad<0]
                                    
                                    #model_wrapper.compute_z()
                                    #n_z_close_to_1 = model_wrapper.get_n_z_close_to_1(test_grad=True, include_batchnorm=False)
                                    #n_weight_close_to_1 = model_wrapper.get_n_weigth_z_close_to_1(test_grad=True, include_batchnorm=False)
                                    #print(f"5. Number of z equal to 1: {n_z_close_to_1, n_weight_close_to_1}")

                                    if test_print_time:
                                          print("Time 9:",time.time()-start_time_loop)
                                          start_time_loop = time.time()
                                    if test_print_ram:
                                          print('11. RAM memory % used:', psutil.virtual_memory()[2], flush = True)
                                          print('11. RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, flush = True)
                                          print('11. Cuda memory:', torch.cuda.memory_allocated()/10**9)

                                    if model_wrapper.type_pruning=="smallest_grad":
                                          model_wrapper.save_grad_layer_wise()
                                    # if model_wrapper.step_temp == 446:
                                    #       if not(model_wrapper.dense_to_sparse):
                                    #             grad_1_dense = model_wrapper.model.fc1.weight.grad.numpy()
                                    #             grad_z_1_dense = model_wrapper.model.fc1.weight_z.grad.numpy()
                                    #             grad_2_dense = model_wrapper.model.fc2.weight.grad.numpy()
                                    #             grad_z_2_dense = model_wrapper.model.fc2.weight_z.grad.numpy()
                                    #             grad_3_dense = model_wrapper.model.fc3.weight.grad.numpy()
                                    #             grad_z_3_dense = model_wrapper.model.fc3.weight_z.grad.numpy()
                                    #             np.save("grad_1_dense.npy", grad_1_dense)
                                    #             np.save("grad_z_1_dense.npy", grad_z_1_dense)
                                    #             np.save("grad_2_dense.npy", grad_2_dense)
                                    #             np.save("grad_z_2_dense.npy", grad_z_2_dense)
                                    #             np.save("grad_3_dense.npy", grad_3_dense)
                                    #             np.save("grad_z_3_dense.npy", grad_z_3_dense)
                                    #       else:
                                    #             grad_1_sparse = model_wrapper.model.fc1.weight.grad.numpy()
                                    #             grad_z_1_sparse = model_wrapper.model.fc1.weight_z.grad.numpy()
                                    #             grad_2_sparse = model_wrapper.model.fc2.weight.grad.numpy()
                                    #             grad_z_2_sparse = model_wrapper.model.fc2.weight_z.grad.numpy()
                                    #             grad_3_sparse = model_wrapper.model.fc3.weight.grad.numpy()
                                    #             grad_z_3_sparse = model_wrapper.model.fc3.weight_z.grad.numpy()
                                    #       import ipdb;ipdb.set_trace()
                                    #model_wrapper.compute_z()
                                    #n_z_close_to_1 = model_wrapper.get_n_z_close_to_1(test_grad=True, include_batchnorm=False)
                                    #n_weight_close_to_1 = model_wrapper.get_n_weigth_z_close_to_1(test_grad=True, include_batchnorm=False)
                                    #print(f"6. Number of z equal to 1: {n_z_close_to_1, n_weight_close_to_1}")
                                    # if name_dataset in ["c4", "wikitext2", "ptb"]:
                                    #gradient_scaler.step(optimizer)
                                    new_gradient_scaler_step(gradient_scaler, optimizer, test_normalized_sgd, d_named_parameters, model_wrapper)
                                    gradient_scaler.update()
                                    # max_value_model = model_wrapper.maximum_value()
                                    # max_value_model_grad = model_wrapper.maximum_gradient()
                                    # print("Maximum value:", max_value_model)
                                    # print("Maximum gradient:", max_value_model_grad)

                                    # else:
                                    #       optimizer.step()  # Update parameters based on gradients.

                                    if test_print_time:
                                          print("Time 10:",time.time()-start_time_loop)
                                          start_time_loop = time.time()
                                    if test_print_ram:
                                          print('12. RAM memory % used:', psutil.virtual_memory()[2], flush = True)
                                          print('12. RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, flush = True)
                                          print('12. Cuda memory:', torch.cuda.memory_allocated()/10**9)

                                    optimizer, test_pruned = model_wrapper.prune_models()

                                    if test_print_time:
                                          print("Time 11:",time.time()-start_time_loop)
                                          start_time_loop = time.time()
                                    if test_print_ram:
                                          print('13. RAM memory % used:', psutil.virtual_memory()[2], flush = True)
                                          print('13. RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, flush = True)
                                          print('13. Cuda memory:', torch.cuda.memory_allocated()/10**9)

                                    if model_wrapper.dense_to_sparse:
                                          if type_decay=="multi_lr":
                                                copy_scheduler = MultiStepLR(optimizer, milestones=scheduler.milestones, gamma=scheduler.gamma)
                                                copy_scheduler.load_state_dict(scheduler.state_dict())
                                                scheduler = copy_scheduler
                                          if type_decay=="cosine":
                                                copy_scheduler = CosineAnnealingLR(optimizer, T_max=T_max_cos, eta_min=eta_min_cos)
                                                copy_scheduler.load_state_dict(scheduler.state_dict())
                                                scheduler = copy_scheduler
                                    #model_wrapper.compute_z()
                                    #n_z_close_to_1 = model_wrapper.get_n_z_close_to_1(test_grad=True, include_batchnorm=False)
                                    #n_weight_close_to_1 = model_wrapper.get_n_weigth_z_close_to_1(test_grad=True, include_batchnorm=False)
                                    #print(f"10. Number of z equal to 1: {n_z_close_to_1, n_weight_close_to_1}")
                                    if test_print_ram:
                                          print('2. RAM memory % used:', psutil.virtual_memory()[2], flush = True)
                                          print('2. RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, flush = True)

                              # l_children = list(model_wrapper.model.children())
                              # print("first:", l_children[0].weight.shape)
                              # print("second:", l_children[1].weight.shape)
                              # print("third:", l_children[2].weight.shape)
                              #print("Cuda memory loop:", torch.cuda.memory_allocated("cuda"))
                              approx_loss_in_sample_no_pen = loss_pred_in_sample/n_seen
                              approx_loss_in_sample_with_pen = approx_loss_in_sample_with_pen/n_seen
                              approx_selection_loss = approx_selection_loss/n_seen
                              approx_entropy_loss = approx_entropy_loss/n_seen
                              approx_l2_loss = approx_l2_loss/n_seen
                              if type_of_task=="classification":
                                    approx_metric_train /= n_seen
                                    if name_dataset in ["c4", "wikitext2", "ptb"] and is_last_module:
                                          #approx_metric_train /= model_wrapper.model.seqlen
                                          approx_metric_train = torch.exp(approx_metric_train).item()
                              try:
                                    approx_loss_in_sample_no_pen = approx_loss_in_sample_no_pen.item()
                              except:
                                    pass
                              optimizer.zero_grad()
                              model_wrapper.model.eval()
                              if test_print_ram:
                                    print('14. RAM memory % used:', psutil.virtual_memory()[2], flush = True)
                                    print('14. RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, flush = True)
                                    print('14. Cuda memory:', torch.cuda.memory_allocated()/10**9)
                              test_repeat_for_sparisty = test_repeat_if_sparsity_not_reached and not(test_sparsity_reached) and number_of_epochs_increase<max_number_of_epochs_increase

                              compute_train_loss = (test_early_stopping_used==0) and test_compute_accurate_in_sample_loss

                              train_loss_with_no_pen_here, train_loss_with_pen_here, metric_train_here, val_loss, metric_val = compute_metrics(model_wrapper, original_model, loader_train, loader_val, name_dataset, n_train_kept, type_of_task, criterion, test_early_stopping_used, module_training, is_last_module, loss_last_block, lambda_loss, scaler_y, test_update_loader, compute_train_loss)

                              if test_compute_accurate_in_sample_loss:
                                    train_loss_with_no_pen, train_loss_with_pen, metric_train = train_loss_with_no_pen_here, train_loss_with_pen_here, metric_train_here

                              if test_repeat_for_sparisty and epoch==(n_epochs_used-1):
                                    number_of_epochs_increase += 1
                                    n_epochs_used += n_epochs
                                    print(f"---- New number of epochs {n_epochs_used} ----")
                                    scheduler = initialize_scheduler(type_decay, optimizer, gamma_lr_decay, n_epochs, period_milestones, start_lr_decay, end_lr_decay, T_max_cos_used, eta_min_cos, warmup_steps)
                                    for idx_group_param in range(len(optimizer.param_groups)):
                                          optimizer.param_groups[idx_group_param]["lr"] = l_initial_lr[idx_group_param]
                                    model_wrapper.entropy_reg /= 2**n_mult_ent
                                    n_mult_ent = 0
                                    model_wrapper.selection_reg /= 2**n_mult_sel
                                    n_mult_sel = 0

                              if test_early_stopping_used==1:
                                    l_error_array = np.array(l_validation_loss)
                                    current_metric = val_loss
                              else:
                                    l_error_array = np.array(l_in_sample_loss_no_pen)
                                    current_metric = approx_loss_in_sample_no_pen
                              l_diff = l_error_array[1:] - l_error_array[:-1]
                              l_diff = l_diff[l_diff>0]
                              if len(l_diff)>10 and not(test_sparsity_reached):
                                    new_diff = current_metric - l_error_array[-1]
                                    print("--- new_diff:", new_diff)
                                    if new_diff >= 3*np.std(l_diff[-20:]) and not(phase_freeze):
                                          print("--- Freezing all weight_z and bias_z ---")
                                          print("n_epochs_no_improvement_freeze =", n_epochs_no_improvement_freeze, flush = True)
                                          # if just_reset:
                                          #       prop_reset_const/=2
                                          model_wrapper.freeze_all_z(d_named_parameters)
                                          phase_freeze = True
                                          best_metric_freeze = copy.deepcopy(current_metric)
                                    if phase_freeze and (best_metric_freeze<current_metric or (best_metric_freeze-current_metric)/best_metric_freeze<=1e-3):
                                          print("--- imp:", (best_metric_freeze-current_metric)/best_metric_freeze)
                                          n_epochs_no_improvement_freeze += 1
                                          print("n_epochs_no_improvement_freeze =", n_epochs_no_improvement_freeze, flush = True)
                                          if n_epochs_no_improvement_freeze == patience_freeze:
                                                print("--- Unfreezing all weight_z and bias_z ---", flush = True)
                                                model_wrapper.unfreeze_all_z(d_named_parameters)
                                                phase_freeze = False
                                                n_epochs_no_improvement_freeze = 0
                                                model_wrapper.model.load_state_dict(best_model.state_dict())
                                                if n_mult_sel>=2:
                                                      model_wrapper.selection_reg /= 4
                                                      n_mult_sel -= 2
                                                if n_mult_ent>=2:
                                                      model_wrapper.entropy_reg /= 4
                                                      n_mult_ent -= 2
                                                # model_wrapper.selection_reg = initial_selection_reg
                                                # model_wrapper.entropy_reg = initial_entropy_reg
                                    elif phase_freeze and best_metric_freeze>current_metric:
                                          n_epochs_no_improvement_freeze = 0
                                          best_metric_freeze = copy.deepcopy(current_metric)
                                          print("n_epochs_no_improvement_freeze =", n_epochs_no_improvement_freeze, flush = True)

                              if test_save_all_models:
                                    path_save_all_models = folder_saves+"/study_"+name_study+"/trial_"+str(trial.number)+"/repeat_"+str(ind_repeat)+"/history/all_models"
                                    if not(os.path.exists(folder_saves+"/study_"+name_study+"/trial_"+str(trial.number))):
                                          os.mkdir(folder_saves+"/study_"+name_study+"/trial_"+str(trial.number))
                                    if not(os.path.exists(folder_saves+"/study_"+name_study+"/trial_"+str(trial.number)+"/repeat_"+str(ind_repeat))):
                                          os.mkdir(folder_saves+"/study_"+name_study+"/trial_"+str(trial.number)+"/repeat_"+str(ind_repeat))
                                    if not(os.path.exists(folder_saves+"/study_"+name_study+"/trial_"+str(trial.number)+"/repeat_"+str(ind_repeat)+"/history")):
                                          os.mkdir(folder_saves+"/study_"+name_study+"/trial_"+str(trial.number)+"/repeat_"+str(ind_repeat)+"/history")
                                    if not(os.path.exists(path_save_all_models)):
                                          os.mkdir(path_save_all_models)
                                    torch.save(model_wrapper.model.state_dict(), path_save_all_models+"/model_"+str(epoch))

                              old_sparsity = copy.deepcopy(sparsity)
                              if mode == "layer_wise":
                                    n_z = model_wrapper.get_n_z(test_grad=True)
                                    n_z_total = model_wrapper.get_n_z(test_grad=False)
                                    n_params_current = np.sum([np.prod(x[1].shape) for x in model_wrapper.model.named_parameters() if "_z" not in x[0]])
                                    n_params_original_z = np.sum([d_modules[x].n_weights for x in l_name_modules if x!=-1 and d_modules[x].weight.requires_grad])
                                    n_params_dense_total = model_wrapper.n_params_original_z
                                    #sparsity = 1 - n_z/model_wrapper.n_params_original_z
                                    # sparsity = 1 - n_z/n_params_original_z
                                    sparsity_total = 1 - n_z_total/n_params_dense_total
                                    # sparsity_storage = 1 - n_params_current/n_params_dense_total
                              elif mode=="ensemble":
                                    n_z = model_wrapper.get_n_z(test_grad=False, include_batchnorm=False)
                                    n_params_current = np.sum([np.prod(x[1].shape) for x in model_wrapper.model.named_parameters() if "_z" not in x[0]])
                                    n_params_original_z = model_wrapper.n_params_original_z
                                    # sparsity = 1 - n_z/n_params_original_z
                                    # sparsity_storage = 1 - n_params_current/n_params_original_z

                              sparsity = model_wrapper.get_sparsity()
                              if goal_sparsity==0 and model_wrapper.n_params_original_z==0:
                                    sparsity=0
                              sparsity_storage = model_wrapper.get_sparsity_storage()
                              if len(l_n_z)>0:
                                    old_sparsity = 1 - l_n_z[-1]/n_params_original_z
                                    sparsity_increases_before_goal_being_reached = not(test_sparsity_reached) and (sparsity>old_sparsity)
                              else:
                                    sparsity_increases_before_goal_being_reached = False


                              sparsity_becomes_reached = not(test_sparsity_reached) and (sparsity>=goal_sparsity)
                              
                              if mode=="layer_wise":
                                    sparsity_becomes_reached = sparsity_becomes_reached or (not(test_sparsity_reached) and (sparsity_total>=goal_sparsity))
                              condition_sparsity = sparsity_increases_before_goal_being_reached or sparsity_becomes_reached

                              if sparsity>=goal_sparsity:
                                    test_sparsity_reached = True
                              else:
                                    if test_wait_for_pruning:
                                          n_epochs_used += 1

                              if mode=="layer_wise":
                                    if sparsity_total>=goal_sparsity:
                                          test_sparsity_reached = True

                              model_wrapper.test_sparsity_reached = test_sparsity_reached

                              if sparsity_becomes_reached:
                                    actions_when_sparsity_becomes_reached(goal_sparsity, epoch, model_wrapper, d_named_parameters, test_early_stopping, loader_train, loader_val)
                                    if test_early_stopping == 2:
                                          test_early_stopping_used = 0
                              
                              tol_adaptative_lr = 0.5
                              if test_early_stopping_used == 1:
                                    if (type_of_task=="regression"):
                                          if (metric_val < best_val_metric) or condition_sparsity:
                                                if test_sparsity_reached and test_adaptive_lr and best_val_metric>0 and 100*abs(best_val_metric-metric_val)/best_val_metric<=tol_adaptative_lr:
                                                      n_epochs_small_improvements +=1
                                                elif test_sparsity_reached and test_adaptive_lr:
                                                      n_epochs_small_improvements = 0
                                                print("--- CONDITION IMPROVEMENT (regression es) ---")
                                                best_val_metric = metric_val
                                                best_ep = epoch_counter
                                                if path_save!=None:
                                                      torch.save(model_wrapper.model.state_dict(), path_save)
                                                best_model = copy.deepcopy(model_wrapper.model)
                                                n_epochs_no_improvement = 0
                                          else:
                                                if test_sparsity_reached:
                                                      n_epochs_no_improvement += 1
                                                      if test_sparsity_reached and test_adaptive_lr:
                                                            n_epochs_small_improvements = 0

                                    if (type_of_task=="classification"):
                                          if metric_early_stopping in ["val_loss", "val_perplexity"]:
                                                print("--- Current val loss:", val_loss)
                                                print("--- Current best val loss:", best_val_loss)
                                                condition_improvement = (val_loss < best_val_loss) or condition_sparsity
                                                if test_sparsity_reached and test_adaptive_lr and best_val_loss>0 and 100*abs(best_val_loss-val_loss)/best_val_loss<=tol_adaptative_lr:
                                                      n_epochs_small_improvements +=1
                                                elif test_sparsity_reached and test_adaptive_lr:
                                                      n_epochs_small_improvements = 0
                                          elif metric_early_stopping == "val_accuracy":
                                                condition_improvement = (metric_val > best_val_metric) or condition_sparsity
                                                if test_sparsity_reached and test_adaptive_lr and best_val_metric>0 and 100*abs(best_val_metric-metric_val)/best_val_metric<tol_adaptative_lr:
                                                      n_epochs_small_improvements +=1
                                                elif test_sparsity_reached and test_adaptive_lr:
                                                      n_epochs_small_improvements = 0
                                          if condition_improvement:
                                                print("--- CONDITION IMPROVEMENT (classification es) ---")
                                                best_val_loss = val_loss
                                                best_val_metric = metric_val
                                                best_ep = epoch_counter
                                                if path_save!=None:
                                                      torch.save(model_wrapper.model.state_dict(), path_save)
                                                best_model = copy.deepcopy(model_wrapper.model)
                                                n_epochs_no_improvement = 0
                                          else:
                                                if test_sparsity_reached:
                                                      n_epochs_no_improvement += 1
                                                      if test_sparsity_reached and test_adaptive_lr:
                                                            n_epochs_small_improvements = 0

                              else:
                                    if not(test_compute_accurate_in_sample_loss):
                                          train_loss_with_pen = approx_loss_in_sample_with_pen
                                          train_loss_with_no_pen = approx_loss_in_sample_no_pen
                                    # if module_training and (loss_last_block == "layer_wise" or not(is_last_module)):
                                    #       metric_in_sample = train_loss_with_no_pen
                                    # else:
                                    #       metric_in_sample = train_loss_with_pen
                                    print("--- Current train loss:", train_loss_with_pen)
                                    print("--- Current best train loss:", best_train_loss)
                                    if train_loss_with_pen < best_train_loss or condition_sparsity:
                                          if test_sparsity_reached and test_adaptive_lr and 100*abs(best_train_loss-train_loss_with_pen)/best_train_loss<=tol_adaptative_lr:
                                                n_epochs_small_improvements +=1
                                          elif test_sparsity_reached and test_adaptive_lr:
                                                n_epochs_small_improvements = 0
                                          print("--- CONDITION IMPROVEMENT (no es) ---")
                                          best_train_loss = train_loss_with_pen
                                          best_val_metric = metric_val
                                          best_ep = epoch_counter
                                          if path_save!=None:
                                                torch.save(model_wrapper.model.state_dict(), path_save)
                                          best_model = copy.deepcopy(model_wrapper.model)
                                          n_epochs_no_improvement = 0
                                    else:
                                          if test_sparsity_reached:
                                                n_epochs_no_improvement += 1

                              if test_sparsity_reached and test_adaptive_lr and n_epochs_small_improvements==patience_adaptive_lr:
                                    optimizer.param_groups[0]["lr"] *= 2
                                    n_epochs_small_improvements = 0

                              # Updating dataset    
                              if module_training and not(test_repeat_for_sparisty) and epoch==(n_epochs_used-1) and not(is_last_module):
                                    print("Updating model parameters ...", flush=True)
                                    model_wrapper.model.load_state_dict(best_model.state_dict())
                                    if test_almost_sequential in [1, 3]:
                                          test_update_original = False
                                    else:
                                          test_update_original = True

                                    # TEMP CHECK TO DELETE
                                    # test_begin = False
                                    # try:
                                    #       model_wrapper.model.eval()
                                    #       model_wrapper.model.to("cpu")
                                    #       model_wrapper.model.attention_mask = model_wrapper.model.attention_mask.to("cpu")
                                    #       train_data = loader_train.dataset.dataset.data[copy_indices_train]
                                    #       val_data = loader_train.dataset.dataset.data[copy_indices_val]
                                    #       output_train_data = model_wrapper.model(train_data)
                                    #       output_val_data = model_wrapper.model(val_data)
                                    #       model_wrapper.model.to("cuda")
                                    #       model_wrapper.model.attention_mask = model_wrapper.model.attention_mask.to("cuda")
                                    #       test_begin = True
                                    # except:
                                    #       pass
                                    # END  TEMP CHECK TO DELETE
                                    if test_update_dataset:
                                          print("Updating dataset ...", flush=True)
                                          update_dataset(model_wrapper.model, model_wrapper.device, original_model, loader_train, loader_val, n_train_kept, test_update_original, copy_indices_train, copy_indices_val, test_almost_sequential)

                                    # TEMP CHECK TO DELETE
                                    # if test_begin:
                                    #       max_diff_train = torch.max(torch.abs(loader_train.dataset.dataset.data[copy_indices_train]-output_train_data)).item()
                                    #       max_diff_val = torch.max(torch.abs(loader_train.dataset.dataset.data[copy_indices_val]-output_val_data)).item()
                                    #       print("------------------------")
                                    #       print("max diff train:", max_diff_train)
                                    #       print("max diff val:", max_diff_val)
                                    #       print("------------------------")
                                    # model_wrapper.model.train()
                                    # END  TEMP CHECK TO DELETE
                                    print("Done", flush=True)

                              test_model_stuck = False
                              tol_epsilon = 1e-6
                              if test_compute_accurate_in_sample_loss:
                                    print_loss_no_pen = "Exact in-sample loss with no pen"
                                    print_loss_with_pen = "Exact in-sample loss with pen"
                                    if type_of_task == "regression":
                                          if epoch >=1:
                                                test_model_stuck = abs(value_loss_with_pen-train_loss_with_pen)<=tol_epsilon and abs(value_loss_no_pen-train_loss_with_no_pen)<=tol_epsilon and abs(value_in_sample_metric - metric_train)<=tol_epsilon
                                          value_in_sample_metric = metric_train
                                          print_metric = "Exact in-sample MSE"
                                    elif type_of_task == "classification":
                                          if epoch >=1:
                                                test_model_stuck = abs(value_loss_with_pen-train_loss_with_pen)<=tol_epsilon and abs(value_loss_no_pen - train_loss_with_no_pen)<=tol_epsilon and abs(value_in_sample_metric - metric_train)<=tol_epsilon
                                          value_in_sample_metric = metric_train
                                          if name_dataset in ["c4", "wikitext2", "ptb"]:
                                                print_metric = "Exact in-sample perplexity"
                                          else:
                                                print_metric = "Exact in-sample accuracy"
                                    value_loss_with_pen = train_loss_with_pen
                                    value_loss_no_pen = train_loss_with_no_pen
                              else:
                                    print_loss_no_pen = "Approx in-sample loss with no pen"
                                    print_loss_with_pen = "Approx in-sample loss with pen"
                                    if type_of_task == "regression":
                                          if epoch >=1:
                                                test_model_stuck = abs(value_in_sample_metric-approx_loss_in_sample_no_pen)<=tol_epsilon and abs(value_loss_no_pen - approx_loss_in_sample_no_pen)<=tol_epsilon and abs(value_loss_with_pen - approx_loss_in_sample_with_pen)<=tol_epsilon
                                          value_in_sample_metric = approx_loss_in_sample_no_pen
                                          print_metric = "Approx in-sample MSE"
                                    elif type_of_task == "classification":
                                          if epoch >=1:
                                                test_model_stuck = abs(value_in_sample_metric-approx_metric_train)<=tol_epsilon and abs(value_loss_no_pen - approx_loss_in_sample_no_pen)<=tol_epsilon and abs(value_loss_with_pen - approx_loss_in_sample_with_pen)<=tol_epsilon
                                          value_in_sample_metric = approx_metric_train
                                          if name_dataset in ["c4", "wikitext2", "ptb"]:
                                                print_metric = "Approx in-sample perplexity"
                                          else:
                                                print_metric = "Approx in-sample accuracy"
                                    value_loss_no_pen = approx_loss_in_sample_no_pen
                                    value_loss_with_pen = approx_loss_in_sample_with_pen

                              if name_dataset in ["c4", "wikitext2", "ptb"]:
                                    to_pring_val = "Val perplexity"
                              else:
                                    to_pring_val = "Val acc"

                              to_print = ""
                              
                              add_text = ""
                              if test_one_layer_pruning and get_phase_decoder_rec(model_wrapper.model)!=0:
                                    add_text += f" - Phase {get_phase_decoder_rec(model_wrapper.model)}"
                              if retraining_of_last_block:
                                    add_text += " (Retraining)"


                              if module_training:
                                    to_print += f"Round {current_round+1}/{n_rounds}{add_text}, "
                              else:
                                    to_print += ""
                              
                              if type_of_task=="regression":
                                    if mode=="layer_wise":
                                          to_print += f'Epoch: {epoch:03d}, {print_loss_no_pen}: {value_loss_no_pen:.4f}, {print_loss_with_pen}: {value_loss_with_pen:.4f}, Validation loss: {val_loss:.4f}, {print_metric}: {value_in_sample_metric:.4f}, Validation MSE: {metric_val:.4f}, lr: {current_lr:.4f}, n_z: {n_z:.4f}, layer sparsity: {sparsity:4f}, total sparsity: {sparsity_total:4f}, sparsity storage: {sparsity_storage:4f}'
                                    elif mode=="ensemble":
                                          to_print += f'Epoch: {epoch:03d}, {print_loss_no_pen}: {value_loss_no_pen:.4f}, {print_loss_with_pen}: {value_loss_with_pen:.4f}, Validation loss: {val_loss:.4f}, {print_metric}: {value_in_sample_metric:.4f}, Validation MSE: {metric_val:.4f}, lr: {current_lr:.4f}, n_z: {n_z:.4f}, sparsity: {sparsity:4f}, sparsity storage: {sparsity_storage:4f}'
                              elif type_of_task == "classification":
                                    if mode=="layer_wise":
                                          to_print += f'Epoch: {epoch:03d}, {print_loss_no_pen}: {value_loss_no_pen:.4f}, {print_loss_with_pen}: {value_loss_with_pen:.4f}, Validation loss: {val_loss:.4f}, {print_metric}: {value_in_sample_metric:.4f}, {to_pring_val}: {metric_val:.4f}, lr: {current_lr:.4f}, n_z: {n_z:.4f}, layer sparsity: {sparsity:4f}, total sparsity: {sparsity_total:4f}, sparsity storage: {sparsity_storage:4f}'
                                    elif mode=="ensemble":
                                          to_print += f'Epoch: {epoch:03d}, {print_loss_no_pen}: {value_loss_no_pen:.4f}, {print_loss_with_pen}: {value_loss_with_pen:.4f}, Validation loss: {val_loss:.4f}, {print_metric}: {value_in_sample_metric:.4f}, {to_pring_val}: {metric_val:.4f}, lr: {current_lr:.4f}, n_z: {n_z:.4f}, sparsity: {sparsity:4f}, sparsity storage: {sparsity_storage:4f}'
                              to_print += f", RAM Used (GB): {(psutil.virtual_memory()[3]/1000000000):.2f}"
                              
                              to_print += f", selection_loss {approx_selection_loss:.4f}"
                              to_print += f", entropy_loss {approx_entropy_loss:.4f}"
                              to_print += f", l2_loss {approx_l2_loss:.4f}"

                              to_print += f", selection_reg {model_wrapper.selection_reg:.4f}"
                              to_print += f", entropy_reg {model_wrapper.entropy_reg:.4f}"
                              to_print += f", l2_reg {model_wrapper.l2_reg:.4f}"

                              if model_wrapper.selection_lagrangian_reg!=None:
                                    to_print += f", selection_reg {model_wrapper.selection_lagrangian_reg.data}"
                                    to_print += f", sparsity_level_selection {model_wrapper.sparsity_level_selection}"
                              if model_wrapper.entropy_lagrangian_reg!=None:
                                    to_print += f", entropy_reg {model_wrapper.entropy_lagrangian_reg.data}"
                                    to_print += f", sparsity_level_entropy {model_wrapper.sparsity_level_entropy}"
                              print(to_print, flush=True)
                              l_lr.append(current_lr)
                              l_in_sample_loss.append(value_loss_with_pen)
                              l_in_sample_loss_no_pen.append(value_loss_no_pen)
                              l_validation_loss.append(val_loss)
                              l_sparsity.append(sparsity)
                              l_sparsity_storage.append(sparsity_storage)

                              if mode=="layer_wise":
                                    l_n_z.append(n_z_total)
                              elif mode=="ensemble":
                                    l_n_z.append(n_z)
                              l_n_params.append(n_params_current)
                              if type_of_task == "regression":
                                    l_validation_metric.append(metric_val)
                                    l_in_sample_metric.append(value_in_sample_metric)
                              elif type_of_task == "classification":
                                    l_validation_metric.append(metric_val)
                                    l_in_sample_metric.append(value_in_sample_metric)
                              l_times_epochs.append(time.time()-start_epoch)
                              if np.isnan(value_loss_with_pen):
                                    print("---", flush = True)
                                    print("Loss became NaN: end of the training", flush = True)
                                    print("---", flush = True)

                                    if module_training and not(is_last_module): #if mode=="layer_wise" and loss_func == "layer_wise" and epoch < (n_epochs_used-1):
                                          print("Updating model parameters ...", flush=True)
                                          model_wrapper.model.load_state_dict(best_model.state_dict())
                                          if test_almost_sequential in [1, 3]:
                                                test_update_original = False
                                          else:
                                                test_update_original = True
                                          if test_update_dataset:
                                                print("Updating dataset ...", flush=True)
                                                update_dataset(model_wrapper.model, model_wrapper.device, original_model, loader_train, loader_val, n_train_kept, test_update_original, copy_indices_train, copy_indices_val, test_almost_sequential)
                                          print("Done", flush=True)
                                    break

                              if test_model_stuck:
                                    print("---", flush = True)
                                    print("Model got stuck: end of the training", flush = True)
                                    print("---", flush = True)

                                    if module_training and not(is_last_module):# and epoch < (n_epochs_used-1):
                                          print("Updating model parameters ...", flush=True)
                                          model_wrapper.model.load_state_dict(best_model.state_dict())
                                          if test_almost_sequential in [1, 3]:
                                                test_update_original = False
                                          else:
                                                test_update_original = True
                                          if test_update_dataset:
                                                print("Updating dataset ...", flush=True)
                                                update_dataset(model_wrapper.model, model_wrapper.device, original_model, loader_train, loader_val, n_train_kept, test_update_original, copy_indices_train, copy_indices_val, test_almost_sequential)
                                          print("Done", flush=True)
                                    break

                              if type_decay!="None":
                                    scheduler.step()
                              if trial!=None:
                                    if trial.should_prune():
                                          raise optuna.exceptions.TrialPruned()
                        else:
                              if mode == "layer_wise" and name_module!=-1:
                                    d_modules[name_module].requires_grad_(False)
                              print("Early stopping at epoch", epoch)

                              if module_training and not(is_last_module):#epoch < (n_epochs_used-1):
                                    print("Updating model parameters ...", flush=True)
                                    model_wrapper.model.load_state_dict(best_model.state_dict())
                                    if test_almost_sequential in [1, 3]:
                                          test_update_original = False
                                    else:
                                          test_update_original = True
                                    if test_update_dataset:
                                          print("Updating dataset ...", flush=True)
                                          update_dataset(model_wrapper.model, model_wrapper.device, original_model, loader_train, loader_val, n_train_kept, test_update_original, copy_indices_train, copy_indices_val, test_almost_sequential)
                                    print("Done", flush=True)
                              break

            if mode == "layer_wise" and name_module!=-1:
                  d_modules[name_module].requires_grad_(False)
            if mode == "layer_wise" and loss_func=="layer_wise" and name_module!=-1 and not(module_training):
                  handle.remove()
                  handle_original.remove()

      if (module_training and is_last_module) or not(module_training):
            print("Updating model parameters ...", flush=True)
            model_wrapper.model.load_state_dict(best_model.state_dict())
            print("Done", flush=True)

      if epoch_counter>=0:
            n_z_final = l_n_z[epoch_counter]
      else:
            n_z_final = model_wrapper.get_n_z(test_grad=False)
      return l_in_sample_loss, l_in_sample_loss_no_pen, l_validation_loss, l_in_sample_metric, l_validation_metric, l_times_epochs, l_lr, l_n_z, l_sparsity, l_sparsity_storage, l_n_params, best_model, best_ep, n_z_final, test_sparsity_reached

def process_results_neural_network(l_in_sample_loss, l_in_sample_loss_no_pen, l_validation_loss, l_in_sample_metric, l_validation_metric, l_times_epochs, l_lr, l_n_z, l_sparsity, l_sparsity_storage, l_n_params, best_model, loader_train, loader_val, loader_test, type_of_task, model_wrapper, scaler_y, metric_name, best_ep, n_z_final, test_sparsity_reached, name_dataset, n_train_kept, copy_indices_train, copy_indices_val, arch):
      l_in_sample_loss = np.array(l_in_sample_loss)
      l_in_sample_loss_no_pen = np.array(l_in_sample_loss_no_pen)
      l_validation_loss = np.array(l_validation_loss)
      l_in_sample_metric = np.array(l_in_sample_metric)
      l_validation_metric = np.array(l_validation_metric)
      l_times_epochs = np.array(l_times_epochs)
      l_lr = np.array(l_lr)
      l_n_z = np.array(l_n_z)
      l_sparsity = np.array(l_sparsity)
      l_sparsity_storage = np.array(l_sparsity_storage)
      l_n_params = np.array(l_n_params)
      if name_dataset in ["c4", "wikitext2", "ptb"]:
            l_model_wrap = model_wrapper
            print("Computing metrics for training, validation and test set ...", flush = True)
            in_sample_metric, validation_metric, test_metric = evaluate_llm(l_model_wrap, loader_train, loader_val, loader_test, n_train_kept, l_model_wrap[0].device, copy_indices_train, copy_indices_val, 1)
      elif "deit" in arch:
            l_model_wrap = model_wrapper
            update_datasets_till_last_block(l_model_wrap, loader_train, loader_val, loader_test, n_train_kept, l_model_wrap[0].device, copy_indices_train, copy_indices_val, 1)
            print("Computing metric training set ...", flush = True)
            in_sample_metric = evaluate_neural_network(l_model_wrap[-1].model, loader_train, type_of_task, l_model_wrap[0].device, scaler_y=scaler_y)
            print("Computing metric validation set ...", flush = True)
            validation_metric = evaluate_neural_network(l_model_wrap[-1].model, loader_val, type_of_task, l_model_wrap[0].device, scaler_y=scaler_y)
            print("Computing metric test set ...", flush = True)
            test_metric = evaluate_neural_network(l_model_wrap[-1].model, loader_test, type_of_task, l_model_wrap[0].device, scaler_y=scaler_y)
      else:
            print("Computing metric training set ...", flush = True)
            in_sample_metric = evaluate_neural_network(best_model, loader_train, type_of_task, model_wrapper.device, scaler_y=scaler_y)
            print("Computing metric validation set ...", flush = True)
            validation_metric = evaluate_neural_network(best_model, loader_val, type_of_task, model_wrapper.device, scaler_y=scaler_y)
            print("Computing metric test set ...", flush = True)
            test_metric = evaluate_neural_network(best_model, loader_test, type_of_task, model_wrapper.device, scaler_y=scaler_y)
      
      try:
            in_sample_metric = in_sample_metric.cpu()
            validation_metric = validation_metric.cpu()
            test_metric = test_metric.cpu()
      except:
            pass
      
      try:
            in_sample_metric = in_sample_metric.item()
            validation_metric = validation_metric.item()
            test_metric = test_metric.item()
      except:
            pass

      dict_list = {}
      dict_list["l_in_sample_loss/In-sample loss"] = l_in_sample_loss
      dict_list["l_in_sample_loss_no_pen/In-sample loss no pen"] = l_in_sample_loss_no_pen
      dict_list["l_in_sample_metric/In-sample "+metric_name] = l_in_sample_metric
      dict_list["l_validation_loss/Validation loss"] = l_validation_loss
      dict_list["l_validation_metric/Validation "+metric_name] = l_validation_metric
      dict_list["l_times_epochs/Time per epoch"] = l_times_epochs
      dict_list["l_lr/Learning rate"] = l_lr
      dict_list["l_n_z/Number of z"] = l_n_z
      dict_list["l_n_params/Number of parameters"] = l_n_params
      dict_list["l_sparsity/Sparsity"] = l_sparsity
      dict_list["l_sparsity_storage/Sparsity storage"] = l_sparsity_storage
      
      d_results = {}
      d_results["best_ep"] = best_ep
      if type_of_task == "regression":
            d_results["train_mse"] = in_sample_metric
            d_results["val_mse"] = validation_metric
            d_results["test_mse"] = test_metric
      elif type_of_task == "classification":
            if name_dataset in ["c4", "wikitext2", "ptb"]:
                  d_results["train_ppl"] = in_sample_metric[0]
                  d_results["val_ppl"] = validation_metric[0]
                  d_results["test_ppl"] = test_metric[0]  
                  d_results["train_auc"] = in_sample_metric[1]
                  d_results["val_auc"] = validation_metric[1]
                  d_results["test_auc"] = test_metric[1]
            else:
                  d_results["train_acc"] = in_sample_metric[0]
                  d_results["val_acc"] = validation_metric[0]
                  d_results["test_acc"] = test_metric[0]  
                  d_results["train_auc"] = in_sample_metric[1]
                  d_results["val_auc"] = validation_metric[1]
                  d_results["test_auc"] = test_metric[1]

      d_results["n_z"] = n_z_final
      d_results["goal_sparsity_reached"] = test_sparsity_reached
      return d_results, validation_metric, best_model, dict_list

def get_n_params(module):
      d_params = dict(module.named_parameters())
      return np.sum([np.prod(d_params[x].shape) for x in d_params if "z" in x])

def get_n_params_non_z(module):
      d_params = dict(module.named_parameters())
      return np.sum([np.prod(d_params[x].shape) for x in d_params if "z" not in x])


def get_modules_layer_wise_rec(module, l_modules):
      if module.__str__().lower()[:10] == "basicblock":
            l_modules.append(module)
      elif module.__str__().lower()[:10] == "bottleneck":
            l_modules.append(module)
      elif module.__str__().lower()[:4] == "conv":
            l_modules.append(module)
      elif module.__str__().lower()[:9] == "batchnorm":
            l_modules.append(module)
      elif module.__str__().lower()[:9] == "groupnorm":
            l_modules.append(module)
      elif module.__str__().lower()[:4] == "relu":
            l_modules.append(module)
      elif module.__str__().lower()[:7] == "avgpool":
            l_modules.append(module)
      else:
            d_children = dict(module.named_children())
            if len(d_children)>0:
                  for child in d_children:
                        get_modules_layer_wise_rec(d_children[child], l_modules)
            else:
                  l_modules.append(module)
      return l_modules

def get_modules_layer_wise_llm(model, attention_mask, test_one_layer_pruning):
      l_modules_before_threshold = list(model.model.decoder.layers)
      for module in l_modules_before_threshold:
            module.attention_mask = attention_mask
            module.return_hidden_only = True
      if test_one_layer_pruning:
            tmp_list = []
            if model.model.decoder.final_layer_norm is not None:
                  tmp_list+=[model.model.decoder.final_layer_norm]
            if model.model.decoder.project_out is not None:
                  tmp_list+=[model.model.decoder.project_out]
                  l_modules_before_threshold+=[nn.Sequential(*tmp_list)]
                  l_modules_before_threshold+=[model.lm_head]
            else:
                  tmp_list += [model.lm_head]
                  l_modules_before_threshold+=[nn.Sequential(*tmp_list)]
      else:
            if model.model.decoder.final_layer_norm is not None:
                  l_modules_before_threshold+=[model.model.decoder.final_layer_norm]
            if model.model.decoder.project_out is not None:
                  l_modules_before_threshold+=[model.model.decoder.project_out]
            l_modules_before_threshold+=[model.lm_head]
      for module in l_modules_before_threshold:
            module.seqlen = model.seqlen
      return l_modules_before_threshold

class Pos_embed(torch.nn.Module):
      def __init__(self, model):
            super().__init__()
            self.func = model._pos_embed

      def forward(self, x):
            return self.func(x)

class Global_pool(torch.nn.Module):
      def __init__(self, global_pool, num_prefix_tokens):
            super().__init__()
            self.global_pool = global_pool
            self.num_prefix_tokens = num_prefix_tokens

      def forward(self, x):
            if self.global_pool == 'avg':
                  return x[:, self.num_prefix_tokens:].mean(dim=1)
            elif self.global_pool:
                  return x[:, 0]  # class token

def get_modules_layer_wise_deit(model):
      #l_modules_before_threshold = [nn.Sequential(*[model.patch_embed, Pos_embed(model), model.patch_drop, model.norm_pre])]
      l_modules_before_threshold = []
      l_modules_before_threshold += list(model.blocks)
      end_list = []
      end_list += [model.norm]
      if model.attn_pool is not None:
            end_list += [model.attn_pool]
      else:
            end_list += [Global_pool(model.global_pool, model.num_prefix_tokens)]
      end_list += [model.fc_norm, model.head_drop, model.head]
      l_modules_before_threshold+=[nn.Sequential(*end_list)]
      return l_modules_before_threshold


def process_to_one_layer(list_to_change):
      l_temp = []
      l_downsamples = []
      for ind_sub_module in range(len(list_to_change)):
            current_module = list_to_change[ind_sub_module]
            if current_module.__str__().lower()[:10] == "basicblock" or current_module.__str__().lower()[:10] == "bottleneck":
                  d_modules_residual_block = dict(current_module.named_modules())
                  l_keys = list(d_modules_residual_block.keys())[1:]
                  l_keys = [x for x in l_keys if "downsample" not in x]
                  for ind_key in range(len(l_keys)):
                        if ind_key==0:
                              test_save_initial_data = 1
                              if current_module.downsample != None:
                                    test_downsample = 1
                              else:
                                    test_downsample = 0
                        else:
                              test_save_initial_data = 0
                              test_downsample = 0
                        if ind_key==(len(l_keys)-2):
                              test_add_initial_data = 1
                        else:
                              test_add_initial_data = 0
                        l_temp.append((d_modules_residual_block[l_keys[ind_key]],test_save_initial_data, test_add_initial_data, test_downsample))
                  if current_module.downsample != None:
                        l_downsamples.append(current_module.downsample)
            else:
                  l_temp.append((current_module, 0, 0, 0))
      return l_temp, l_downsamples

def process_l_modules_w_threshold(l_modules_before_threshold, threshold, test_one_layer_pruning, arch, l_modules_before_threshold_original):
      l_modules = []
      l_modules_original = []
      acc_n_params = 0
      new_element = []
      new_element_original = []
      force_stop_next = False
      while len(l_modules_before_threshold)>0:
            if test_one_layer_pruning and not(("opt" in arch or "deit" in arch)):
                  new_module, test_save_initial_data, test_add_initial_data, test_downsample  = l_modules_before_threshold.pop(0)
                  new_module_original, _, _, _  = l_modules_before_threshold_original.pop(0)

                  new_module.test_save_initial_data = test_save_initial_data
                  new_module.test_add_initial_data = test_add_initial_data
                  new_module.test_downsample = test_downsample

                  new_module_original.test_save_initial_data = test_save_initial_data
                  new_module_original.test_add_initial_data = test_add_initial_data
                  new_module_original.test_downsample = test_downsample

            else:
                  new_module = l_modules_before_threshold.pop(0)
                  new_module_original = l_modules_before_threshold_original.pop(0)

                  new_module.test_save_initial_data = False
                  new_module.test_add_initial_data = False
                  new_module.test_downsample = False

                  new_module_original.test_save_initial_data = False
                  new_module_original.test_add_initial_data = False
                  new_module_original.test_downsample = False

            n_new_params = get_n_params(new_module)
            acc_n_params += n_new_params
            if not(test_one_layer_pruning and (test_save_initial_data or force_stop_next)) and (acc_n_params<=threshold or len(new_element)==0 or acc_n_params == 0 or (not(test_one_layer_pruning) and n_new_params==0) or ((acc_n_params-n_new_params)==0)) and not(("opt" in arch or "deit" in arch) and len(l_modules_before_threshold)==0):
                  new_element.append(new_module)
                  new_element_original.append(new_module_original)
                  force_stop_next = False
            else:
                  if len(new_element) > 1:
                        test_save_initial_data_final, test_add_initial_data_final, test_downsample_final = False, False, False
                        for element in new_element:
                              test_save_initial_data_final = test_save_initial_data_final or element.test_save_initial_data
                              test_add_initial_data_final = test_add_initial_data_final or element.test_add_initial_data
                              test_downsample_final = test_downsample_final or element.test_downsample
                        new_element = nn.Sequential(*new_element)
                        new_element_original = nn.Sequential(*new_element_original)

                        new_element.test_save_initial_data = test_save_initial_data_final
                        new_element.test_add_initial_data = test_add_initial_data_final
                        new_element.test_downsample = test_downsample_final
                        
                        new_element_original.test_save_initial_data = test_save_initial_data_final
                        new_element_original.test_add_initial_data = test_add_initial_data_final
                        new_element_original.test_downsample = test_downsample_final
                  else:
                        new_element = new_element[0]
                        new_element_original = new_element_original[0]
                  l_modules.append(new_element)
                  l_modules_original.append(new_element_original)
                  new_element = [new_module]
                  new_element_original = [new_module_original]
                  acc_n_params = get_n_params(new_module)
                  force_stop_next = False
            if (test_one_layer_pruning and not("opt" in arch or "deit" in arch)) and test_add_initial_data:
                  force_stop_next = True
      if len(new_element) > 1:
            test_save_initial_data_final, test_add_initial_data_final, test_downsample_final = False, False, False
            for element in new_element:
                  test_save_initial_data_final = test_save_initial_data_final or element.test_save_initial_data
                  test_add_initial_data_final = test_add_initial_data_final or element.test_add_initial_data
                  test_downsample_final = test_downsample_final or element.test_downsample
            new_element = nn.Sequential(*new_element)
            new_element_original = nn.Sequential(*new_element_original)
            
            new_element.test_save_initial_data = test_save_initial_data_final
            new_element.test_add_initial_data = test_add_initial_data_final
            new_element.test_downsample = test_downsample_final
            
            new_element_original.test_save_initial_data = test_save_initial_data_final
            new_element_original.test_add_initial_data = test_add_initial_data_final
            new_element_original.test_downsample = test_downsample_final
      else:
            new_element = new_element[0]
            new_element_original = new_element_original[0]
      l_modules.append(new_element)
      l_modules_original.append(new_element_original)
      return l_modules, l_modules_original

def process_l_modules_w_threshold_original(l_modules_before_threshold, l_modules_with_z, test_one_layer_pruning):
      l_modules = []
      acc_ind = 0
      for ind_element in range(len(l_modules_with_z)):
            new_element = []
            n_modules = 1
            try:
                  n_modules = len(l_modules_with_z[ind_element])
                  print(n_modules)
            except:
                  pass
            for _ in range(n_modules):
                  if test_one_layer_pruning:
                        new_element.append(l_modules_before_threshold[acc_ind][0])
                  else:
                        new_element.append(l_modules_before_threshold[acc_ind])
                  acc_ind+=1
            if len(new_element) > 1:
                  new_element = nn.Sequential(*new_element)
            else:
                  new_element = new_element[0]
            new_element.test_save_initial_data = l_modules_with_z[ind_element].test_save_initial_data
            new_element.test_add_initial_data = l_modules_with_z[ind_element].test_add_initial_data
            new_element.test_downsample = l_modules_with_z[ind_element].test_downsample
            l_modules.append(new_element)
      return l_modules

# ----------------------------
# --- Evaluation per block ---
# ----------------------------

def compute_auc(y_true, y_pred):
    if len(y_pred.shape)==2:
        if y_pred.shape[1] == 2:
            y_pred = y_pred[:,1]
    auc = roc_auc_score(
        y_true,
        y_pred,
        multi_class="ovo"
    )
    return auc

def get_loss_metric_old(model_wrap, dataset, type_of_task, criterion, scaler_y=None, test_train=False):
      model_wrap.model.eval()
      data = dataset.data
      y = dataset.targets.to(model_wrap.device)
      pred = torch.zeros((0)).to(model_wrap.device)
      with torch.no_grad():
            size_batch = 300
            n_batches = int(np.ceil(data.shape[0]/size_batch))
            for ind_batch in range(n_batches):
                  idx_start = ind_batch*size_batch
                  idx_end = (ind_batch+1)*size_batch
                  pred = torch.concat([pred, model_wrap.model(data[idx_start:idx_end].to(model_wrap.device))])
      if type_of_task == "regression":
            mse = criterion(pred, y).item()
            try:
                  mse = mse.cpu()
            except:
                  pass
            loss_with_no_pen = mse
            entropy_loss, selection_loss, l2_loss = model_wrap.get_losses()
            loss_with_pen = loss_with_no_pen + entropy_loss.item() + selection_loss.item() + l2_loss.item()
            return mse, loss_with_no_pen, loss_with_pen
      elif type_of_task == "classification":
            if len(pred.shape)>=2:
                  acc = 100*torch.mean((torch.argmax(pred.detach(), dim=1)==y).float())
            else:
                  acc = 100*torch.mean((torch.round(pred.detach())==y).float())
            try:
                  acc = acc.cpu()
            except:
                  pass
            loss_with_no_pen = criterion(pred.detach(), y).item()
            entropy_loss, selection_loss, l2_loss = model_wrap.get_losses()
            loss_with_pen = loss_with_no_pen + entropy_loss.item() + selection_loss.item() + l2_loss.item()
            return acc, loss_with_no_pen, loss_with_pen

def evaluate_neural_network(model, loader, type_of_task, device, scaler_y=None):
      model.eval()
      pred = torch.zeros((0)).to(device)
      y = torch.zeros((0)).to(device)
      with torch.no_grad():
            for batch_sgd in tqdm(loader):
                  #print(f"Evaluation RAM Used (GB): {(psutil.virtual_memory()[3]/1000000000):.2f}", flush = True)
                  input_batch_sgd, input_batch_original_sgd, target_batch_sgd, index_seen_sgd, old_index_seen_sgd = read_batch(batch_sgd)
                  pred = torch.concat([pred, model(input_batch_sgd.to(device))])
                  y = torch.concat([y, target_batch_sgd.to(device)])

      if type_of_task=="regression":
            in_sample_pred = scaler_y.inverse_transform(pred.detach().numpy()[:,np.newaxis])[:,0]
            in_sample_truth = scaler_y.inverse_transform(y.numpy()[:,np.newaxis])[:,0]
            metric = np.mean((in_sample_pred - in_sample_truth)**2)
      elif type_of_task=="classification":
            y = y.long()
            #print(f"Evaluation RAM Used (GB): {(psutil.virtual_memory()[3]/1000000000):.2f}", flush = True)
            try:
                  pred_numpy = pred.cpu()
                  #print(f"Evaluation RAM Used (GB): {(psutil.virtual_memory()[3]/1000000000):.2f}", flush = True)
                  pred_numpy = pred_numpy.detach()
                  #print(f"Evaluation RAM Used (GB): {(psutil.virtual_memory()[3]/1000000000):.2f}", flush = True)
            except:
                  pass
            #print(f"Evaluation RAM Used (GB): {(psutil.virtual_memory()[3]/1000000000):.2f}", flush = True)
            pred_numpy = torch.softmax(pred_numpy, 1)
            #print(f"Evaluation RAM Used (GB): {(psutil.virtual_memory()[3]/1000000000):.2f}", flush = True)
            pred_numpy = pred_numpy.numpy()
            #print(f"Evaluation RAM Used (GB): {(psutil.virtual_memory()[3]/1000000000):.2f}", flush = True)
            try:
                  corres_y_numpy = y.cpu()
                  #print(f"Evaluation RAM Used (GB): {(psutil.virtual_memory()[3]/1000000000):.2f}", flush = True)
            except:
                  pass
            corres_y_numpy = corres_y_numpy.numpy()
            #print(f"Evaluation RAM Used (GB): {(psutil.virtual_memory()[3]/1000000000):.2f}", flush = True)
            #auc = compute_auc(corres_y_numpy, pred_numpy)
            auc = -1
            if len(pred.shape)>=2:
                  acc = torch.mean((torch.argmax(pred.detach(), dim=1)==y).float()).item()
                  #print(f"Evaluation RAM Used (GB): {(psutil.virtual_memory()[3]/1000000000):.2f}", flush = True)
            else:
                  acc = torch.mean((torch.round(pred.detach())==y).float()).item()
                  #print(f"Evaluation RAM Used (GB): {(psutil.virtual_memory()[3]/1000000000):.2f}", flush = True)
            metric = (acc, auc)
      return metric

def evaluate_llm(l_model_wrap, loader_train, loader_val, loader_test, n_train_kept, device, copy_indices_train, copy_indices_val, test_almost_sequential):
      # for ind_wrap in range(len(l_model_wrap)-1):
      #       model_wrap = l_model_wrap[ind_wrap]
      #       model_wrap.model.to(device)
      #       print(f"Updating dataset (round {ind_wrap+1}/{len(l_model_wrap)-1}) ...", flush=True)
            
      #       # # TEMP CHECK TO DELETE
      #       # try:
      #       #       model_wrap.model.to("cpu")
      #       #       model_wrap.model.attention_mask = model_wrap.model.attention_mask.to("cpu")
      #       #       train_idx = copy.deepcopy(loader_train.dataset.indices)
      #       #       val_idx = copy.deepcopy(loader_val.dataset.indices)
      #       #       train_data = loader_train.dataset.dataset.data[train_idx]
      #       #       val_data = loader_train.dataset.dataset.data[val_idx]
      #       #       output_train_data = model_wrap.model(train_data)
      #       #       output_val_data = model_wrap.model(val_data)
      #       #       model_wrap.model.to("cuda")
      #       #       model_wrap.model.attention_mask = model_wrap.model.attention_mask.to("cuda")
      #       update_dataset(model_wrap.model, model_wrap.device, None, loader_train, loader_val, n_train_kept, False, copy_indices_train, copy_indices_val, test_almost_sequential)
      #       #       max_diff_train = torch.max(torch.abs(loader_train.dataset.dataset.data[train_idx]-output_train_data)).item()
      #       #       max_diff_val = torch.max(torch.abs(loader_train.dataset.dataset.data[val_idx]-output_val_data)).item()
      #       #       print("max diff train:", max_diff_train)
      #       #       print("max diff val:", max_diff_val)
      #       # except:
      #       #       pass
      #       # # END TEMP CHECK TO DELETE

      #       # if ind_wrap==0:
      #       #       import ipdb;ipdb.set_trace()
      #       update_dataset(model_wrap.model, model_wrap.device, None, loader_test, None, n_train_kept, False, copy_indices_train, copy_indices_val, test_almost_sequential)
      #       print("Done", flush=True)
      #       model_wrap.model.to("cpu")
      #       # if ind_wrap==0:
      #       #       import ipdb;ipdb.set_trace()
      update_datasets_till_last_block(l_model_wrap, loader_train, loader_val, loader_test, n_train_kept, device, copy_indices_train, copy_indices_val, test_almost_sequential)
      last_block = l_model_wrap[-1]
      last_block.model.to(device)
      criterion = torch.nn.functional.cross_entropy
      _, train_ppl = get_loss_perplexity(last_block, loader_train, criterion)
      _, val_ppl = get_loss_perplexity(last_block, loader_val, criterion)
      _, test_ppl = get_loss_perplexity(last_block, loader_test, criterion)
      train_auc = -1
      val_auc = -1
      test_auc = -1
      train_metric = (train_ppl, train_auc)
      val_metric = (val_ppl, val_auc)
      test_metric = (test_ppl, test_auc)
      return train_metric, val_metric, test_metric

def update_datasets_till_last_block(l_model_wrap, loader_train, loader_val, loader_test, n_train_kept, device, copy_indices_train, copy_indices_val, test_almost_sequential):
      for ind_wrap in range(len(l_model_wrap)-1):
            model_wrap = l_model_wrap[ind_wrap]
            model_wrap.model.to(device)
            print(f"Updating dataset (round {ind_wrap+1}/{len(l_model_wrap)-1}) ...", flush=True)
            update_dataset(model_wrap.model, model_wrap.device, None, loader_train, loader_val, n_train_kept, False, copy_indices_train, copy_indices_val, test_almost_sequential)
            update_dataset(model_wrap.model, model_wrap.device, None, loader_test, None, -1, False, None, None, test_almost_sequential)
            print("Done", flush=True)
            model_wrap.model.to("cpu")

def get_loss_metric(model_wrap, loader, type_of_task, criterion, scaler_y=None, test_train=False):
      model_wrap.model.eval()
      pred = torch.zeros((0)).to(model_wrap.device)
      y = torch.zeros((0)).to(model_wrap.device)
      with torch.no_grad():
            for batch_sgd in tqdm(loader):
                  if test_print_ram:
                        print('13. RAM memory % used:', psutil.virtual_memory()[2], flush = True)
                        print('13. RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, flush = True)
                        print('13. Cuda memory:', torch.cuda.memory_allocated()/10**9)
                  input_batch_sgd, input_batch_original_sgd, target_batch_sgd, index_seen_sgd, old_index_seen_sgd = read_batch(batch_sgd)
                  pred = torch.concat([pred, model_wrap.model(input_batch_sgd.to(model_wrap.device))])
                  y = torch.concat([y, target_batch_sgd.to(model_wrap.device)])

      if type_of_task == "regression":
            mse = criterion(pred, y).item()
            loss_with_no_pen = mse
            entropy_loss, selection_loss, l2_loss = model_wrap.get_losses()
            loss_with_pen = loss_with_no_pen + entropy_loss.item() + selection_loss.item() + l2_loss.item()
            return mse, loss_with_no_pen, loss_with_pen
      elif type_of_task == "classification":
            y = y.long()
            if len(pred.shape)>=2:
                  acc = torch.mean((torch.argmax(pred.detach(), dim=1)==y).float()).item()
            else:
                  acc = torch.mean((torch.round(pred.detach())==y).float()).item()
            acc*=100
            loss_with_no_pen = criterion(pred.detach(), y).item()
            entropy_loss, selection_loss, l2_loss = model_wrap.get_losses()
            loss_with_pen = loss_with_no_pen + entropy_loss.item() + selection_loss.item() + l2_loss.item()
            return acc, loss_with_no_pen, loss_with_pen

def get_loss_metric_module(model_wrap, original_model, loader, type_of_task, criterion, scaler_y=None, test_train=False, is_last_module=True, test_update_loader=False, name_dataset="mnist", n_train_kept=-1):
      model_wrap.model.eval()
      current_mse = 0
      n_seen = 0
      with torch.no_grad():
            for batch_sgd in tqdm(loader):
                  if test_print_ram:
                        print('Get loss metric RAM memory % used:', psutil.virtual_memory()[2], flush = True)
                        print('Get loss metric RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, flush = True)
                        print('Get loss metric Cuda memory:', torch.cuda.memory_allocated()/10**9)
                  input_batch_sgd, input_batch_original_sgd, target_batch_sgd, index_seen_sgd, old_index_seen_sgd = read_batch(batch_sgd)
                  n_seen += input_batch_sgd.shape[0]
                  # if input_batch_sgd.dtype!= list(model_wrap.model.parameters())[0].dtype:
                  #       use_amp = True
                  # else:
                  #       use_amp = False
                  use_amp = False
                  # with torch.autocast(device_type=model_wrap.device, dtype=torch.float16, enabled=use_amp):
                  output = model_wrap.model(input_batch_sgd.to(model_wrap.device))
                  output_original = original_model(input_batch_original_sgd.to(model_wrap.device))
                  # End autocast
                  current_mse += torch.mean((output-output_original)**2).item()*output.shape[0]
      current_mse = current_mse/n_seen
      return current_mse, current_mse

def get_loss_perplexity(model_wrap, loader, criterion):
      model_wrap.model.eval()
      nlls = []
      n_seen = 0
      loss = 0
      #loader_ppl = DataLoader(loader.dataset, batch_size=1, num_workers=0, pin_memory=True)
      loader_ppl = loader
      with torch.no_grad():
            for batch_sgd in tqdm(loader_ppl):
                  if test_print_ram:
                        print('Get loss metric RAM memory % used:', psutil.virtual_memory()[2], flush = True)
                        print('Get loss metric RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, flush = True)
                        print('Get loss metric Cuda memory:', torch.cuda.memory_allocated()/10**9)
                  input_batch_sgd, input_batch_original_sgd, target_batch_sgd, index_seen_sgd, old_index_seen_sgd = read_batch(batch_sgd)
                  # if input_batch_sgd.dtype!= list(model_wrap.model.parameters())[0].dtype:
                  #       use_amp = True
                  # else:
                  #       use_amp = False
                  use_amp = False
                  # with torch.autocast(device_type=model_wrap.device, dtype=torch.float16, enabled=use_amp):
                  lm_logits = model_wrap.model(input_batch_sgd.to(model_wrap.device))
                  # End autocast
                  shift_logits = lm_logits[:, :-1, :].contiguous()
                  shift_labels = target_batch_sgd[:, 1:].contiguous()
                  shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                  n_samples = lm_logits.shape[0]
                  loss_batch = criterion(shift_logits, shift_labels.view(-1).to(model_wrap.device))
                  loss += loss_batch*n_samples
                  neg_log_likelihood = loss_batch.float() * n_samples * model_wrap.model.seqlen
                  n_seen += n_samples * model_wrap.model.seqlen
                  nlls.append(neg_log_likelihood)
      ppl = torch.exp(torch.stack(nlls).sum() / n_seen)
      loss /= n_samples
      return loss.item(), ppl.item()

# function generate_schedule from CHITA-main
def generate_schedule(num_stages, base_level,sparsity_level,schedule):
      repeat=1
      if num_stages == 1:
            return [sparsity_level]
      if schedule == 'exp':
            sparsity_multiplier = (sparsity_level - base_level)*np.power(2, num_stages-1)/(np.power(2, num_stages-1) - 1)
            l =[base_level + sparsity_multiplier*((np.power(2, stage) - 1)/np.power(2, stage)) for stage in range(num_stages)]
            return [x for x in l for _ in range(repeat)]
      elif schedule == 'poly':
            l= [sparsity_level + (base_level-sparsity_level)*np.power(1 - (stage/(num_stages-1)), 3) for stage in range(num_stages)]
            return [x for x in l for _ in range(repeat)]
      elif schedule == 'const':
            return [sparsity_level for stage in range(num_stages)]
      elif schedule == 'linear':
            return [base_level + stage*(sparsity_level - base_level)/(num_stages-1) for stage in range(num_stages)]
      elif schedule == 'MFAC':
            sparsity_multiplier = ((1. - sparsity_level) / (1. - base_level)) ** (1./num_stages)
            return [1. - ((1. - base_level) * (sparsity_multiplier**(stage+1))) for stage in range(num_stages)]

def train_sub_modules(model_wrap, test_distributed, arguments, device, optimizer, original_module, is_last_module, name_study, name_model, dataset, criterion, n_epochs, batch_size_dataset, test_early_stopping, trial, test_save_all_models, type_decay, gamma_lr_decay, T_max_cos, eta_min_cos, start_lr_decay, end_lr_decay, warmup_steps, type_of_task, test_compute_accurate_in_sample_loss, folder_saves, ind_repeat, patience, metric_early_stopping, period_milestones, goal_sparsity, type_training, n_restart, num_workers, loss_func, name_dataset, n_train_kept, l_model_wrap, ind_model_wrap, l_in_sample_loss, l_in_sample_loss_no_pen, l_validation_loss, l_in_sample_metric, l_validation_metric, l_times_epochs, l_lr, l_n_z, l_sparsity, l_sparsity_storage, l_n_params, best_ep, n_z_final, test_normalized_sgd, pruning_rate_cte, lambda_loss, test_repeat_if_sparsity_not_reached, loss_last_block, retraining_of_last_block, copy_indices_train, copy_indices_val, test_adaptive_lr, patience_adaptive_lr, patience_freeze, test_wait_for_pruning, test_almost_sequential, tol_ent_reg, tol_sel_reg, n_incr_gradual_pruning, goal_sparsity_discrete, test_start_sparse_gpt, test_start_convex, type_pruning_schedule, end_model, test_one_layer_pruning, arch, test_start_obc, rel_damp, lambda_fisher, lambda_reconst, algo_pruning, sparsities, sds, n_parallel, n_convex, d_losses, pruning_level, n_layers, gpts, test_prune=True):
      if test_print_ram:
            print('1. Train sub RAM memory % used:', psutil.virtual_memory()[2], flush = True)
            print('1. Train sub final RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, flush = True)
            print('1. Train sub final Cuda memory:', torch.cuda.memory_allocated()/10**9)
      set_require_grad_rec(model_wrap.model, True)
      if test_print_ram:
            print('2. Train sub RAM memory % used:', psutil.virtual_memory()[2], flush = True)
            print('2. Train sub final RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, flush = True)
            print('2. Train sub final Cuda memory:', torch.cuda.memory_allocated()/10**9)
      if test_distributed:
            model_wrap.model.cuda(arguments.gpu)
            model_wrap.model = torch.nn.parallel.DistributedDataParallel(model_wrap.model, device_ids=[arguments.gpu])
      else:
            model_wrap.model.to(device)
      if test_print_ram:
            print('3. Train sub RAM memory % used:', psutil.virtual_memory()[2], flush = True)
            print('3. Train sub final RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, flush = True)
            print('3. Train sub final Cuda memory:', torch.cuda.memory_allocated()/10**9)
      mode_model_wrap = "ensemble"
      original_module.to(device)
      if test_print_ram:
            print('4. Train sub RAM memory % used:', psutil.virtual_memory()[2], flush = True)
            print('4. Train sub final RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, flush = True)
            print('4. Train sub final Cuda memory:', torch.cuda.memory_allocated()/10**9)
      module_training = True
      loader_train, loader_val, loader_test, scaler_y = dataset
      if n_incr_gradual_pruning!=-1:
            if type_pruning_schedule=="linear":
                  l_goal_sparisties = np.linspace(0, goal_sparsity_discrete, n_incr_gradual_pruning+1)[1:]
            elif type_pruning_schedule=="exponential":
                  l_goal_sparisties = generate_schedule(n_incr_gradual_pruning+1, 0, goal_sparsity_discrete, "exp")[1:]
            if len(l_goal_sparisties)>=1:
                  l_goal_sparisties[-1] = goal_sparsity
            else:
                  l_goal_sparisties = [goal_sparsity]
            n_epochs_used = int(np.ceil(n_epochs/n_incr_gradual_pruning))
      else:
           l_goal_sparisties = [goal_sparsity]
           n_epochs_used = n_epochs
      best_ep_module_sum = 0
      n_z_final_module_sum = 0
      initial_sel_reg = copy.deepcopy(model_wrap.selection_reg)
      initial_ent_reg = copy.deepcopy(model_wrap.entropy_reg)
      l_initial_lr = []
      if optimizer!=None:
            for idx_group_param in range(len(optimizer.param_groups)):
                  l_initial_lr.append(copy.deepcopy(optimizer.param_groups[idx_group_param]["lr"]))
      if test_one_layer_pruning and not(is_last_module) and "opt" in arch:
            l_phases = [1,2,3]
      else:
            l_phases = [0]
      print("Phases:", l_phases)
      print("Goal sparsities:", l_goal_sparisties)
      for phase in l_phases:
            set_phase_decoder_rec(model_wrap.model, phase)
            set_phase_decoder_rec(original_module, phase)
            if phase == 1:
                  set_require_grad_rec(model_wrap.model.self_attn, True)
                  set_require_grad_rec(model_wrap.model.self_attn_layer_norm, True)
                  set_require_grad_rec(model_wrap.model.fc1, False)
                  set_require_grad_rec(model_wrap.model.fc2, False)
                  set_require_grad_rec(model_wrap.model.final_layer_norm, False)
            if phase == 2:
                  set_require_grad_rec(model_wrap.model.self_attn, False)
                  set_require_grad_rec(model_wrap.model.self_attn_layer_norm, False)
                  set_require_grad_rec(model_wrap.model.fc1, True)
                  set_require_grad_rec(model_wrap.model.fc2, False)
                  set_require_grad_rec(model_wrap.model.final_layer_norm, False)
            if phase == 3:
                  set_require_grad_rec(model_wrap.model.self_attn, False)
                  set_require_grad_rec(model_wrap.model.self_attn_layer_norm, False)
                  set_require_grad_rec(model_wrap.model.fc1, False)
                  set_require_grad_rec(model_wrap.model.fc2, True)
                  set_require_grad_rec(model_wrap.model.final_layer_norm, True)
            model_wrap.n_params_original = np.sum([np.prod(x[1].shape) if ("z" not in x[0] and x[1].requires_grad) else 0 for x in model_wrap.model.named_parameters()])
            model_wrap.n_params_original_z = np.sum([np.prod(x[1].shape) if ("z" in x[0] and x[1].requires_grad) else 0 for x in model_wrap.model.named_parameters()])
            model_wrap.to_prune_within_block = [x[0] for x in model_wrap.model.named_parameters() if x[1].requires_grad]
            for goal_sparsity_sub in l_goal_sparisties:
                  model_wrap.selection_reg = initial_sel_reg
                  model_wrap.entropy_reg = initial_ent_reg
                  for idx_group_param in range(len(optimizer.param_groups)):
                        optimizer.param_groups[idx_group_param]["lr"] = l_initial_lr[idx_group_param]
                  test_update_dataset = (goal_sparsity_sub == l_goal_sparisties[-1]) and (get_phase_decoder_rec(model_wrap.model) in [0, 1, 3]) and not(is_last_module)
                  #test_update_dataset = (test_almost_sequential == 3)
                  # if test_almost_sequential in [1, 3]:
                  #       test_update_original = False
                  # else:
                  #       test_update_original = True
                  # args_update_dataset = [model_wrap.model, model_wrap.device, original_module, loader_train, loader_val, n_train_kept, test_update_original, copy_indices_train, copy_indices_val]

                  # if test_start_sparse_gpt and goal_sparsity_sub<=goal_sparsity_discrete and goal_sparsity_sub>0 and not(retraining_of_last_block):
                  #       model_wrap.model.eval()
                  #       model_wrap.model.to(device)
                  #       print("--- Pruning using sparse-gpt ...")
                  #       prune_spargegpt_block(model_wrap.model, loader_train, device, arguments.n_train_kept, 16, goal_sparsity_sub, 0, 0, 0.01, 128, True, model_wrap.gamma)
                  #       model_wrap.model.train()
                  # elif test_start_obc and goal_sparsity_sub<=goal_sparsity_discrete and goal_sparsity_sub>0 and not(retraining_of_last_block):
                  #       model_wrap.model.eval()
                  #       model_wrap.model.to(device)
                  #       print("--- Pruning using obc ...")
                  #       prune_spargegpt_block(model_wrap.model, loader_train, device, arguments.n_train_kept, 16, goal_sparsity_sub, 0, 0, 0.01, 128, True, model_wrap.gamma)
                  #       model_wrap.model.train()
                  test_prune_discrete = test_start_sparse_gpt or test_start_obc or test_start_convex
                  if test_start_sparse_gpt:
                        type_pruning_discrete = "sparse_gpt"
                  elif test_start_obc:
                        type_pruning_discrete = "obc"
                  elif test_start_convex:
                        type_pruning_discrete = algo_pruning

                  if test_early_stopping==0:
                        loader_train.dataset.indices = [i for i in range(len(loader_train.dataset.dataset))]
                        loader_val.dataset.indices = []

                  if test_early_stopping==2:
                        test_early_stopping_used = 1
                        loader_train.dataset.indices = copy_indices_train
                        loader_val.dataset.indices = copy_indices_val

                  if test_early_stopping == 3 and not(retraining_of_last_block):
                        test_early_stopping_used = 0
                        loader_train.dataset.indices = [i for i in range(len(loader_train.dataset.dataset))]
                        loader_val.dataset.indices = []

                  if test_early_stopping == 3 and retraining_of_last_block:
                        test_early_stopping_used = 1
                        loader_train.dataset.indices = copy_indices_train
                        loader_val.dataset.indices = copy_indices_val

                  if test_prune_discrete and goal_sparsity_sub<=goal_sparsity_discrete and goal_sparsity_sub>0 and not(retraining_of_last_block):
                        model_wrap.model.eval()
                        model_wrap.model.to(device)
                        print(f"--- Pruning using {type_pruning_discrete} ...")
                        prune_block(model_wrap.model, loader_train, device, goal_sparsity_sub, 0, 0, n_parallel, True, model_wrap.gamma, end_model, arch, type_pruning_discrete, rel_damp, lambda_fisher, lambda_reconst, sparsities, sds, ind_model_wrap, n_convex, d_losses, pruning_level, n_layers, gpts, test_prune)
                        model_wrap.model.train()
                        
                  # if (not(test_update_dataset) and is_last_module) or True:
                  #       l_in_sample_loss_module = []
                  #       l_in_sample_loss_no_pen_module = []
                  #       l_validation_loss_module = []
                  #       l_in_sample_metric_module = []
                  #       l_validation_metric_module = []
                  #       l_times_epochs_module = []
                  #       l_lr_module = []
                  #       l_n_z_module = []
                  #       l_sparsity_module = []
                  #       l_sparsity_storage_module = []
                  #       l_n_params_module = []
                  #       best_model_module = model_wrap.model
                  #       best_ep_module = -1
                  #       n_z_final_module = int(model_wrap.n_params_original_z*goal_sparsity)
                  #       test_sparsity_reached_sub_module = True
                  # else:      
                  l_in_sample_loss_module, l_in_sample_loss_no_pen_module, l_validation_loss_module, l_in_sample_metric_module, l_validation_metric_module, l_times_epochs_module, l_lr_module, l_n_z_module, l_sparsity_module, l_sparsity_storage_module, l_n_params_module, best_model_module, best_ep_module, n_z_final_module, test_sparsity_reached_sub_module = train_neural_network(name_study=name_study, name_model=name_model, model_wrapper=model_wrap, dataset=dataset, optimizer=optimizer, criterion=criterion, n_epochs=n_epochs_used, batch_size_dataset=batch_size_dataset, path_save=None, test_early_stopping=test_early_stopping, trial=trial, test_save_all_models=test_save_all_models, type_decay=type_decay, gamma_lr_decay=gamma_lr_decay, T_max_cos=T_max_cos, eta_min_cos=eta_min_cos, start_lr_decay=start_lr_decay, end_lr_decay=end_lr_decay, warmup_steps=warmup_steps, type_of_task=type_of_task, test_compute_accurate_in_sample_loss=test_compute_accurate_in_sample_loss, folder_saves=folder_saves, ind_repeat=ind_repeat, patience = patience, metric_early_stopping=metric_early_stopping, period_milestones=period_milestones, goal_sparsity=goal_sparsity_sub, type_training=type_training, n_restart=n_restart, num_workers=num_workers, mode=mode_model_wrap, loss_func_and_model=(loss_func, original_module), is_last_module = is_last_module, module_training=module_training, name_dataset=name_dataset, n_train_kept=n_train_kept, n_rounds=len(l_model_wrap), current_round=ind_model_wrap, test_normalized_sgd=test_normalized_sgd, pruning_rate_cte=pruning_rate_cte, lambda_loss=lambda_loss, test_repeat_if_sparsity_not_reached=test_repeat_if_sparsity_not_reached, loss_last_block=loss_last_block, retraining_of_last_block=retraining_of_last_block, copy_indices_train=copy_indices_train, copy_indices_val=copy_indices_val, test_adaptive_lr=test_adaptive_lr, patience_adaptive_lr=patience_adaptive_lr, patience_freeze=patience_freeze, test_wait_for_pruning=test_wait_for_pruning, test_almost_sequential=test_almost_sequential, tol_ent_reg=tol_ent_reg, tol_sel_reg=tol_sel_reg, test_update_dataset=test_update_dataset, test_one_layer_pruning=test_one_layer_pruning)
                  if test_print_ram:
                        print('5. Train sub RAM memory % used:', psutil.virtual_memory()[2], flush = True)
                        print('5. Train sub final RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, flush = True)
                        print('5. Train sub final Cuda memory:', torch.cuda.memory_allocated()/10**9)
                  #model_wrap.model.load_state_dict(best_model_module.state_dict())
                  if torch.sum(list(model_wrap.model.parameters())[0] != list(best_model_module.parameters())[0]).item()!=0:
                        print("BEST MODEL AND MODEL_WRAP NOT MATCHING")
                  if test_print_ram:
                        print('6. Train sub RAM memory % used:', psutil.virtual_memory()[2], flush = True)
                        print('6. Train sub final RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, flush = True)
                        print('6. Train sub final Cuda memory:', torch.cuda.memory_allocated()/10**9)
                  l_in_sample_loss += l_in_sample_loss_module
                  l_in_sample_loss_no_pen += l_in_sample_loss_no_pen_module
                  l_validation_loss += l_validation_loss_module 
                  l_in_sample_metric += l_in_sample_metric_module
                  l_validation_metric += l_validation_metric_module
                  l_times_epochs += l_times_epochs_module
                  l_lr += l_lr_module
                  l_n_z += l_n_z_module
                  l_sparsity += l_sparsity_module #list(1-np.array(l_n_z_module)/model_wrap.n_params_original_z)
                  l_sparsity_storage += l_sparsity_storage_module #list(1-np.array(l_n_params_module)/model_wrap.n_params_original_z)
                  l_n_params += l_n_params_module
                  best_ep_module_sum += best_ep_module
                  n_z_final_module_sum += n_z_final_module
            best_ep += [best_ep_module_sum]
            n_z_final += [n_z_final_module_sum]
      set_require_grad_rec(model_wrap.model, False)
      set_phase_decoder_rec(model_wrap.model, 0)
      set_phase_decoder_rec(original_module, 0)
      if test_print_ram:
            print('7. Train sub RAM memory % used:', psutil.virtual_memory()[2], flush = True)
            print('7. Train sub final RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, flush = True)
            print('7. Train sub final Cuda memory:', torch.cuda.memory_allocated()/10**9)
      model_wrap.model.to("cpu")
      original_module.to("cpu")
      return test_sparsity_reached_sub_module

print("Import utils additive done")