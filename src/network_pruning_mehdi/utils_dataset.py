import torch
from tqdm import tqdm
import copy
import psutil

test_print_ram = False
test_print_time = False

def read_batch(batch_sgd):
      input_batch_sgd = batch_sgd[0]
      if len(batch_sgd) == 2:
            output_original = input_batch_sgd
            if len(batch_sgd[1].shape)>=2:
                  target_batch_sgd = batch_sgd[1][:,:-2]#batch_sgd[1][:,0]
                  if target_batch_sgd.shape[-1] == 1:
                        target_batch_sgd = target_batch_sgd[:,0]
                  index_seen_sgd = batch_sgd[1][:,-2]
                  old_index_seen_sgd = batch_sgd[1][:,-1]
            else:
                  target_batch_sgd = batch_sgd[1]
                  index_seen_sgd = None
                  old_index_seen_sgd = None            
      elif len(batch_sgd) == 3:
            output_original = batch_sgd[1]
            if len(batch_sgd[2].shape)>=2:
                  target_batch_sgd = batch_sgd[2][:,:-2]
                  if target_batch_sgd.shape[-1] == 1:
                        target_batch_sgd = target_batch_sgd[:,0]
                  index_seen_sgd = batch_sgd[2][:,-2]
                  old_index_seen_sgd = batch_sgd[2][:,-1]
            else:
                  target_batch_sgd = batch_sgd[2]
                  index_seen_sgd = None
                  old_index_seen_sgd = None
      return input_batch_sgd, output_original, target_batch_sgd, index_seen_sgd, old_index_seen_sgd

def update_loader_first(loader, output, output_original, current_target, index_seen_sgd, old_index_seen_sgd, n_train_kept, test_update_original = True, is_test = False):
      if not(is_test):
            current_dataset = loader.dataset
      else:
            current_dataset = loader
      if n_train_kept == -1:
            dataset = current_dataset.dataset
      else:
            dataset = current_dataset.dataset.dataset
      dataset.new_data.append(output.detach().cpu())
      if test_update_original:
            dataset.new_original_data.append(output_original.detach().cpu())
      dataset.old_indices.append(old_index_seen_sgd.cpu())
            
def update_loader_final(loader_train, loader_val, n_train_kept, test_update_original = True):
      if loader_val!=None:
            current_dataset = loader_train.dataset
      else:
            current_dataset = loader_train
      if n_train_kept == -1:
            dataset = current_dataset.dataset
      else:
            dataset = current_dataset.dataset.dataset
      if dataset.is_original and loader_val!=None:
            # loader_train.dataset.indices = list(np.arange(len(loader_train.dataset.indices)))
            # loader_val.dataset.indices = list(len(loader_train.dataset.indices)+np.arange(len(loader_val.dataset.indices)))
            dataset.is_original = False
      dataset.old_indices = torch.cat(dataset.old_indices)
      new_idx = torch.argsort(dataset.old_indices)
      dataset.data = torch.cat(dataset.new_data)[new_idx]
      # dataset.data#.share_memory_()
      # if len(dataset.new_original_data)>0:
      if test_update_original:
            dataset.data_output_original = torch.cat(dataset.new_original_data)[new_idx]
            # else:
            #       dataset.data_output_original = torch.Tensor([])
            # dataset.data_output_original#.share_memory_()
            dataset.new_original_data = []
      dataset.required_increment = torch.zeros(dataset.data.shape[0], dtype=int)#.share_memory_()
      dataset.new_data = []
      dataset.old_indices = []

def add_output_orignal(original_model, loader_train, loader_val, n_train_kept, device):
      if n_train_kept == -1:
            dataset = loader_train.dataset.dataset
      else:
            dataset = loader_train.dataset.dataset.dataset
      dataset.new_original_data = []
      dataset.old_indices_output_original = []
      with torch.no_grad():
            # Adding original output train
            for batch_sgd in tqdm(loader_train):
                  input_batch_sgd, input_batch_original_sgd, target_batch_sgd, index_seen_sgd, old_index_seen_sgd = read_batch(batch_sgd)
                  # if input_batch_sgd.dtype!= list(original_model.parameters())[0].dtype:
                  #       use_amp = True
                  # else:
                  #       use_amp = False
                  use_amp = False
                  # with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                  new_output_original = original_model(input_batch_sgd.to(device))
                  # End autocast
                  dataset.new_original_data.append(new_output_original.detach().cpu())
                  dataset.old_indices_output_original.append(old_index_seen_sgd.cpu())
            # Adding original output val                            
            for batch_sgd in tqdm(loader_val):
                  input_batch_sgd, input_batch_original_sgd, target_batch_sgd, index_seen_sgd, old_index_seen_sgd = read_batch(batch_sgd)
                  # if input_batch_sgd.dtype!= list(original_model.parameters())[0].dtype:
                  #       use_amp = True
                  # else:
                  #       use_amp = False
                  use_amp = False
                  # with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                  new_output_original = original_model(input_batch_sgd.to(device))
                  # End autocast
                  dataset.new_original_data.append(new_output_original.detach().cpu())
                  dataset.old_indices_output_original.append(old_index_seen_sgd.cpu())
            dataset.old_indices_output_original = torch.cat(dataset.old_indices_output_original)
            new_idx = torch.argsort(dataset.old_indices_output_original)
            dataset.data_output_original = torch.cat(dataset.new_original_data)[new_idx]
            dataset.new_original_data = []
            dataset.old_indices_output_original = []
            dataset.test_output_original = True

def load_dataset_in_memory(loader_train, loader_val, n_train_kept, test_update_original=True):
      for batch_sgd in tqdm(loader_train):
            input_batch_sgd, input_batch_original_sgd, target_batch_sgd, index_seen_sgd, old_index_seen_sgd = read_batch(batch_sgd)
            input_batch_sgd = input_batch_sgd.float()
            input_batch_original_sgd = input_batch_original_sgd.float()
            update_loader_first(loader_train, input_batch_sgd, input_batch_original_sgd, target_batch_sgd, index_seen_sgd, old_index_seen_sgd, n_train_kept, test_update_original)
      for batch_sgd in tqdm(loader_val):
            input_batch_sgd, input_batch_original_sgd, target_batch_sgd, index_seen_sgd, old_index_seen_sgd = read_batch(batch_sgd)
            input_batch_sgd = input_batch_sgd.float()
            input_batch_original_sgd = input_batch_original_sgd.float()
            update_loader_first(loader_val, input_batch_sgd, input_batch_original_sgd, target_batch_sgd, index_seen_sgd, old_index_seen_sgd, n_train_kept, test_update_original)
      update_loader_final(loader_train, loader_val, n_train_kept, test_update_original)

def update_dataset(model, device, original_model, loader_train, loader_val, n_train_kept, test_update_original=True, copy_indices_train=None, copy_indices_val=None, test_almost_sequential=0):
      # If loader_val==None, then loader_train is in fact loader_test
      if loader_val!=None:
            current_idx_train = copy.deepcopy(loader_train.dataset.indices)
            current_idx_val = copy.deepcopy(loader_val.dataset.indices)
            loader_train.dataset.indices = copy_indices_train
            loader_val.dataset.indices = copy_indices_val

      if model.training:
            training_mode = True
      else:
            training_mode = False
      model.eval()
      for batch_sgd in tqdm(loader_train):
            if test_print_ram:
                  print('Update dataset RAM memory % used:', psutil.virtual_memory()[2], flush = True)
                  print('Update dataset RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, flush = True)
                  print('Update dataset Cuda memory:', torch.cuda.memory_allocated()/10**9)
            input_batch_sgd, input_batch_original_sgd, target_batch_sgd, index_seen_sgd, old_index_seen_sgd = read_batch(batch_sgd)
            with torch.no_grad():
                  if input_batch_sgd.dtype!= list(model.parameters())[0].dtype:
                        input_batch_sgd = input_batch_sgd.float()
                  if input_batch_original_sgd.dtype!= list(model.parameters())[0].dtype:
                        input_batch_original_sgd = input_batch_original_sgd.float()
                  #       use_amp = True
                  # else:
                  #       use_amp = False
                  use_amp = False
                  # with torch.autocast(device_type=model_wrapper.device, dtype=torch.float16, enabled=use_amp):
                  if test_almost_sequential==3:
                        output = original_model(input_batch_original_sgd.to(device))
                  else:
                        output = model(input_batch_sgd.to(device))
                  if test_update_original:
                        output_original = original_model(input_batch_original_sgd.to(device))
                  else:
                        output_original = None 
                  # End autocast
            is_test = loader_val==None
            update_loader_first(loader_train, output, output_original, target_batch_sgd, index_seen_sgd, old_index_seen_sgd, n_train_kept, test_update_original, is_test)
      if loader_val!=None:
            for batch_sgd in tqdm(loader_val):
                  if test_print_ram:
                        print('Update dataset RAM memory % used:', psutil.virtual_memory()[2], flush = True)
                        print('Update dataset RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, flush = True)
                        print('Update dataset Cuda memory:', torch.cuda.memory_allocated()/10**9)
                  input_batch_sgd, input_batch_original_sgd, target_batch_sgd, index_seen_sgd, old_index_seen_sgd = read_batch(batch_sgd)
                  with torch.no_grad():
                        if input_batch_sgd.dtype!= list(model.parameters())[0].dtype:
                              input_batch_sgd = input_batch_sgd.float()
                        if input_batch_original_sgd.dtype!= list(model.parameters())[0].dtype:
                              input_batch_original_sgd = input_batch_original_sgd.float()
                        #       use_amp = True
                        # else:
                        #       use_amp = False
                        use_amp = False
                        # with torch.autocast(device_type=model_wrapper.device, dtype=torch.float16, enabled=use_amp):
                        if test_almost_sequential==3:
                              output = original_model(input_batch_original_sgd.to(device))
                        else:
                              output = model(input_batch_sgd.to(device))
                        if test_update_original:
                              output_original = original_model(input_batch_original_sgd.to(device))
                        else:
                              output_original = None
                        # End autocast
                  update_loader_first(loader_val, output, output_original, target_batch_sgd, index_seen_sgd, old_index_seen_sgd, n_train_kept, test_update_original)
      if test_print_ram:
            print('Update dataset before RAM memory % used:', psutil.virtual_memory()[2], flush = True)
            print('Update dataset before RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, flush = True)
            print('Update dataset before Cuda memory:', torch.cuda.memory_allocated()/10**9)
      
      update_loader_final(loader_train, loader_val, n_train_kept, test_update_original)

      if loader_val!=None:
            loader_train.dataset.indices = current_idx_train
            loader_val.dataset.indices = current_idx_val

      if test_print_ram:
            print('Update dataset final RAM memory % used:', psutil.virtual_memory()[2], flush = True)
            print('Update dataset final RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, flush = True)
            print('Update dataset final Cuda memory:', torch.cuda.memory_allocated()/10**9)
      if training_mode:
            model.train()
