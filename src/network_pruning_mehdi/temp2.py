def train_neural_network(name_study, name_model, model_wrapper, dataset, optimizer, criterion, n_epochs, batch_size_dataset, path_save, test_early_stopping, trial, test_save_all_models=False, type_decay="exponential", gamma_lr_decay=np.exp(-np.log(25)/10000), T_max_cos=10, eta_min_cos=1e-5, start_lr_decay=1e-2, end_lr_decay=1e-5, warmup_steps=100, type_of_task = "regression", test_compute_accurate_in_sample_loss = 0, folder_saves = "TSML_saves", ind_repeat=0, patience=50, metric_early_stopping="val_loss", period_milestones=25, goal_sparsity=0.0, type_training="combined", n_restart=0, num_workers=4, mode="ensemble", loss_func_and_model=("classic", None), is_last_module = False, module_training=False, name_dataset="mnist", n_train_kept = -1, n_rounds = -1, current_round = -1, test_normalized_sgd=0, pruning_rate_cte=-1, lambda_loss=1.0, test_repeat_if_sparsity_not_reached=1, loss_last_block="mce", retraining_of_last_block=False, copy_indices_train=None, copy_indices_val=None):
      test_early_stopping_used = copy.deepcopy(test_early_stopping)
      if test_early_stopping==2:
            test_early_stopping_used = 1
            loader_train.dataset.indices = copy_indices_train
            loader_val.dataset.indices = copy_indices_val

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

      if test_print_ram:
            print('1. Train RAM memory % used:', psutil.virtual_memory()[2], flush = True)
            print('1. Train final RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, flush = True)
            print('1. Train final Cuda memory:', torch.cuda.memory_allocated()/10**9)

      loader_train, loader_val, loader_test, scaler_y = dataset
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
      best_model = copy.deepcopy(model_wrapper.model)

      # BEGIN computing initial metrics
      print("Evaluating initial metrics...", flush=True)
      test_update_loader = False
      if test_early_stopping_used==0:
            val_loss, mse_val, acc_val = np.nan, np.nan, np.nan
      else:
            if module_training:
                  if is_last_module and loss_last_block=="mce":
                        if type_of_task=="regression":
                              mse_val, val_loss, _ = get_loss_metric(model_wrapper, loader_val, type_of_task, criterion, scaler_y)
                        else:
                              acc_val, val_loss, _ = get_loss_metric(model_wrapper, loader_val, type_of_task, criterion, scaler_y)
                  elif is_last_module and loss_last_block=="layer_wise":
                        # TO DO: TWO OPERATIONS HERE COULD BE REDUCED TO ONE
                        _, val_loss = get_loss_metric_module(model_wrapper, original_model, loader_val, type_of_task, criterion, scaler_y, test_update_loader=test_update_loader, name_dataset=name_dataset, n_train_kept=n_train_kept)
                        val_loss *= lambda_loss
                        acc_val = compute_acc(model_wrapper.model, loader_val, model_wrapper.device, verbose=False)
                        # if type_of_task=="regression":
                        #       mse_val, val_loss, _ = get_loss_metric(model_wrapper, loader_val, type_of_task, criterion, scaler_y)
                        # else:
                        #       try:
                        #             acc_val, val_loss, _ = get_loss_metric(model_wrapper, loader_val, type_of_task, criterion, scaler_y)
                        #       except:
                        #             import ipdb;ipdb.set_trace()
                  else:
                        acc_val, val_loss = get_loss_metric_module(model_wrapper, original_model, loader_val, type_of_task, criterion, scaler_y, test_update_loader=test_update_loader, name_dataset=name_dataset, n_train_kept=n_train_kept)
                        acc_val = np.nan
                        val_loss *= lambda_loss
            else:
                  if type_of_task=="regression":
                        mse_val, val_loss, _ = get_loss_metric(model_wrapper, loader_val, type_of_task, criterion, scaler_y)
                  else:
                        acc_val, val_loss, _ = get_loss_metric(model_wrapper, loader_val, type_of_task, criterion, scaler_y)

      if test_early_stopping_used==1:
            if (type_of_task=="regression"):
                  best_val_mse = mse_val
            if (type_of_task=="classification"):
                  best_val_loss = val_loss
                  best_val_acc = acc_val
      else:
            if module_training and (loss_last_block == "layer_wise" or not(is_last_module)):
                  _, train_loss_with_no_pen = get_loss_metric_module(model_wrapper, original_model, loader_train, type_of_task, criterion, scaler_y, test_update_loader=test_update_loader, name_dataset=name_dataset, n_train_kept=n_train_kept)
                  entropy_loss, selection_loss, l2_loss = model_wrapper.get_losses()
                  train_loss_with_pen = train_loss_with_no_pen + entropy_loss.item() + selection_loss.item() + l2_loss.item()
            else:
                  _, _, train_loss_with_pen = get_loss_metric(model_wrapper, loader_train, type_of_task, criterion, scaler_y)
            best_train_loss = train_loss_with_pen
            print("------- train_loss_with_pen:", train_loss_with_pen)
            if type_of_task == "regression":
                  best_val_mse = mse_val
            elif type_of_task == "classification":
                  best_val_acc = acc_val
      print("Done", flush=True)
      # END
      
      best_val_mse = np.inf
      best_train_loss = np.inf
      best_val_loss = np.inf
      best_val_acc = -np.inf

      
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

      for name_module in l_name_modules:
            if mode == "layer_wise" and name_module!=-1:
                  if loss_func=="layer_wise" and not(module_training):
                        handle_original = d_modules_original[name_module].register_forward_hook(get_layer_outputs(name_module, d_layer_output_original, True))
                        handle = d_modules[name_module].register_forward_hook(get_layer_outputs(name_module, d_layer_output, False))
            elif mode == "layer_wise" and name_module==-1:
                  model_wrapper.set_require_grad(True)
            sparsity = model_wrapper.get_sparsity()
            test_sparsity_reached = (sparsity>=goal_sparsity)
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
            n_mult_sel = 0
            n_mult_ent = 0
            n_epochs_no_improvement_freeze = 0
            patience_freeze = 1
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

                  if optimizer != None:
                        scheduler = initialize_scheduler(type_decay, optimizer, gamma_lr_decay, n_epochs, period_milestones, start_lr_decay, end_lr_decay, T_max_cos, eta_min_cos, warmup_steps)

                        for idx_group_param in range(len(optimizer.param_groups)):
                              optimizer.param_groups[idx_group_param]["lr"] = l_initial_lr[idx_group_param]

                  # To delete later
                  model_wrapper.step_temp = 0
                  # End
                  if "layer_wise" in model_wrapper.type_pruning or "smallest_grad" in model_wrapper.type_pruning:
                        model_wrapper.initialize_pruning()
                  
                  
                  # if name_dataset=="imagenet" and module_training and model_wrapper.ind_model_wrap==0:
                  #       n_epochs_used = 1
                  # else:
                  #       n_epochs_used = n_epochs
                  # TO DELETE LATER
                  n_epochs_used = copy.deepcopy(n_epochs)
                  # END

                  n_z = model_wrapper.get_n_z(test_grad=True, include_batchnorm=False)
                  
                  # if n_train_kept == -1:
                  #       loader_train.dataset.dataset.targets = torch.Tensor(loader_train.dataset.dataset.targets)
                  # else:
                  #       loader_train.dataset.dataset.dataset.targets = torch.Tensor(loader_train.dataset.dataset.dataset.targets)
                  
                  # # TO DELETE, SANITY CHECK
                  # if module_training and is_last_module:
                  #       n_epochs_used = 1
                  # # END
                  
                  if module_training and (n_z == 0 or n_epochs_used == 0):
                        # Transform dataset
                        if retraining_of_last_block:
                              add_text = " (Retraining)"
                        else:
                              add_text = ""
                        print(f"Round {current_round+1}/{n_rounds}"+add_text, flush = True)
                        print("Updating dataset ...", flush=True)
                        update_dataset(model_wrapper, original_model, loader_train, loader_val, n_train_kept)
                        print("Done", flush=True)
                        # End transform dataset
                        n_epochs_used = 0

                  print("Number of epochs:", n_epochs_used, flush=True)

                  # # TO DELETE, SANITY CHECK
                  # test_update_loader = False
                  # acc_val, val_loss = get_loss_metric_module(model_wrapper, original_model, loader_val, type_of_task, criterion, scaler_y, test_update_loader=test_update_loader, name_dataset=name_dataset, n_train_kept=n_train_kept)
                  # import ipdb;ipdb.set_trace()
                  # # END
                  epoch = -1
                  while epoch < n_epochs_used-1:
                        epoch += 1
                        if test_print_ram:
                              print('4. Train RAM memory % used:', psutil.virtual_memory()[2], flush = True)
                              print('4. Train final RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, flush = True)
                              print('4. Train final Cuda memory:', torch.cuda.memory_allocated()/10**9)
                        if (n_epochs_no_improvement < patience) or not(test_early_stopping_used):
                              epoch_counter+=1
                              start_epoch = time.time()
                              if test_sparsity_reached:
                                    print("n_epochs_no_improvement =", n_epochs_no_improvement)
                              elif not(phase_freeze):
                                    n_z_close_to_1 = model_wrapper.get_n_z_close_to_1(test_grad=True, include_batchnorm=False)
                                    n_z = model_wrapper.get_n_z(test_grad=True, include_batchnorm=False)
                                    ratio_z_to_1 = n_z_close_to_1/n_z
                                    if epoch != 0:
                                          if ratio_z_to_1-old_ratio<=1e-2 and np.abs(sparsity-old_sparsity)<=1e-2:
                                                n_mult_ent += 1
                                                model_wrapper.entropy_reg *= 2
                                          elif np.abs(sparsity-old_sparsity)<=1e-2:
                                                n_mult_sel += 1
                                                model_wrapper.selection_reg *= 2
                                                acc_no_sparisity_change += 1

                                    # if acc_no_sparisity_change>=3:
                                    #       acc_no_sparisity_change = 0
                                    #       model_wrapper.selection_reg *= 2
                                    old_ratio = copy.deepcopy(ratio_z_to_1)
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
                                    approx_acc_train = 0
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

                                    output = model_wrapper.model(input_batch_sgd.to(model_wrapper.device))

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

                                    if is_last_module:
                                          approx_acc_train += torch.sum(torch.argmax(torch.softmax(output, 1), 1) == y_truth).item()
                                    elif module_training:
                                          approx_acc_train = np.nan
                                    if test_print_time:
                                          print("Time 3:",time.time()-start_time_loop)
                                          start_time_loop = time.time()
                                    if test_print_ram:
                                          print('6. RAM memory % used:', psutil.virtual_memory()[2], flush = True)
                                          print('6. RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, flush = True)
                                          print('6. Cuda memory:', torch.cuda.memory_allocated()/10**9)

                                    if loss_func == "layer_wise":
                                          with torch.no_grad():
                                                output_original = original_model(input_batch_original_sgd.to(model_wrapper.device))
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
                                                loss = criterion(output, y_truth)
                                          elif module_training and (loss_last_block=="layer_wise" or not(is_last_module)):
                                                loss = torch.mean((output-output_original)**2)
                                                loss *= lambda_loss
                                          else:
                                                loss = 0
                                                for key_layer in d_layer_output:
                                                      loss += torch.mean((d_layer_output[key_layer] - d_layer_output_original[key_layer])**2)
                                                loss *= lambda_loss
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
                                          loss = criterion(output, y_truth.long())
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
                                    entropy_loss, selection_loss, l2_loss = model_wrapper.get_losses()
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
                                    loss += entropy_loss + selection_loss + l2_loss
                                    approx_loss_in_sample_with_pen += n_batch*loss.item()
                                    if test_print_time:
                                          print("Time 8:",time.time()-start_time_loop)
                                          start_time_loop = time.time()
                                    if test_print_ram:
                                          print('10. RAM memory % used:', psutil.virtual_memory()[2], flush = True)
                                          print('10. RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, flush = True)
                                          print('10. Cuda memory:', torch.cuda.memory_allocated()/10**9)

                                    loss.backward()  # Derive gradients.
                                    if test_normalized_sgd:
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
                                    try:
                                          optimizer.step()  # Update parameters based on gradients.
                                    except:
                                          import ipdb;ipdb.set_trace()
                                    # weight_z_list = list(model_wrapper.model.named_modules())
                                    # weight_z_concat = torch.cat([x[1].weight_z.view(-1) for x in weight_z_list if ("conv" in x[0] or "fc" in x[0])])
                                    # z_concat = torch.cat([x[1].z.view(-1) for x in weight_z_list if ("conv" in x[0] or "fc" in x[0])])
                                    
                                    # print(weight_z_concat[weight_z_concat<1-1e-5])
                                    # print(z_concat[weight_z_concat<1-1e-5])
                                    # import ipdb;ipdb.set_trace()
                                    # if len(weight_z_concat[weight_z_concat<1].view(-1))>0:
                                    #       print(weight_z_concat[weight_z_concat<1])
                                    # print((list(model_wrapper.model.modules()))[1])
                                    # print(weight_z_concat)


                                    #model_wrapper.compute_z()
                                    #n_z_close_to_1 = model_wrapper.get_n_z_close_to_1(test_grad=True, include_batchnorm=False)
                                    #n_weight_close_to_1 = model_wrapper.get_n_weigth_z_close_to_1(test_grad=True, include_batchnorm=False)
                                    #print(f"7. Number of z equal to 1: {n_z_close_to_1, n_weight_close_to_1}")
                                    if test_print_time:
                                          print("Time 10:",time.time()-start_time_loop)
                                          start_time_loop = time.time()
                                    if test_print_ram:
                                          print('12. RAM memory % used:', psutil.virtual_memory()[2], flush = True)
                                          print('12. RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, flush = True)
                                          print('12. Cuda memory:', torch.cuda.memory_allocated()/10**9)

                                    # To delete later
                                    # if model_wrapper.step_temp==10000:
                                    #       l_children = list(model_wrapper.model.children())
                                    #       weight_z_0 = copy.deepcopy(l_children[0].weight_z)
                                    #       weight_z_grad_0 = copy.deepcopy(l_children[0].weight_z.grad)
                                    #       weight_z_exp_avg_0 = copy.deepcopy(optimizer.state[l_children[0].weight_z]["exp_avg"])
                                    #       weight_z_exp_avg_sq_0 = copy.deepcopy(optimizer.state[l_children[0].weight_z]["exp_avg_sq"])

                                    #       weight_z_1 = copy.deepcopy(l_children[1].weight_z)
                                    #       weight_z_grad_1 = copy.deepcopy(l_children[1].weight_z.grad)
                                    #       weight_z_exp_avg_1 = copy.deepcopy(optimizer.state[l_children[1].weight_z]["exp_avg"])
                                    #       weight_z_exp_avg_sq_1 = copy.deepcopy(optimizer.state[l_children[1].weight_z]["exp_avg_sq"])
                                          
                                    #       weight_z_2 = copy.deepcopy(l_children[2].weight_z)
                                    #       weight_z_grad_2 = copy.deepcopy(l_children[2].weight_z.grad)
                                    #       weight_z_exp_avg_2 = copy.deepcopy(optimizer.state[l_children[2].weight_z]["exp_avg"])
                                    #       weight_z_exp_avg_sq_2 = copy.deepcopy(optimizer.state[l_children[2].weight_z]["exp_avg_sq"])

                                    #       l_to_keep = copy.deepcopy(torch.where(torch.sum(l_children[2].z,0)!=0)[0])

                                    #       n_z_dense = model_wrapper.get_n_z()
                                    # End
                                    # if step_temp == 200:
                                    #       import ipdb;ipdb.set_trace()
                                    #model_wrapper.compute_z()
                                    #n_z_close_to_1 = model_wrapper.get_n_z_close_to_1(test_grad=True, include_batchnorm=False)
                                    #n_weight_close_to_1 = model_wrapper.get_n_weigth_z_close_to_1(test_grad=True, include_batchnorm=False)
                                    #print(f"8. Number of z equal to 1: {n_z_close_to_1, n_weight_close_to_1}")
                                    optimizer, test_pruned = model_wrapper.prune_models()
                                    #model_wrapper.compute_z()
                                    #n_z_close_to_1 = model_wrapper.get_n_z_close_to_1(test_grad=True, include_batchnorm=False)
                                    #n_weight_close_to_1 = model_wrapper.get_n_weigth_z_close_to_1(test_grad=True, include_batchnorm=False)
                                    #print(f"9. Number of z equal to 1: {n_z_close_to_1, n_weight_close_to_1}")
                                    if test_print_time:
                                          print("Time 11:",time.time()-start_time_loop)
                                          start_time_loop = time.time()
                                    if test_print_ram:
                                          print('13. RAM memory % used:', psutil.virtual_memory()[2], flush = True)
                                          print('13. RAM Used (GB):', psutil.virtual_memory()[3]/1000000000, flush = True)
                                          print('13. Cuda memory:', torch.cuda.memory_allocated()/10**9)

                                    # if test_pruned:
                                    #       print(model_wrapper.model)

                                    # if model_wrapper.step_temp==-10:
                                    #       l_children = list(model_wrapper.model.children())
                                    #       weight_z_0_new = copy.deepcopy(l_children[0].weight_z)
                                    #       weight_z_grad_0_new = copy.deepcopy(l_children[0].weight_z.grad)
                                    #       weight_z_exp_avg_0_new = copy.deepcopy(optimizer.state[l_children[0].weight_z]["exp_avg"])
                                    #       weight_z_exp_avg_sq_0_new = copy.deepcopy(optimizer.state[l_children[0].weight_z]["exp_avg_sq"])

                                    #       weight_z_1_new = copy.deepcopy(l_children[1].weight_z)
                                    #       weight_z_grad_1_new = copy.deepcopy(l_children[1].weight_z.grad)
                                    #       weight_z_exp_avg_1_new = copy.deepcopy(optimizer.state[l_children[1].weight_z]["exp_avg"])
                                    #       weight_z_exp_avg_sq_1_new = copy.deepcopy(optimizer.state[l_children[1].weight_z]["exp_avg_sq"])
                                          
                                    #       weight_z_2_new = copy.deepcopy(l_children[2].weight_z)
                                    #       weight_z_grad_2_new = copy.deepcopy(l_children[2].weight_z.grad)
                                    #       weight_z_exp_avg_2_new = copy.deepcopy(optimizer.state[l_children[2].weight_z]["exp_avg"])
                                    #       weight_z_exp_avg_sq_2_new = copy.deepcopy(optimizer.state[l_children[2].weight_z]["exp_avg_sq"])
                                          
                                    #       n_z_sparse = model_wrapper.get_n_z()
                                    #       import ipdb;ipdb.set_trace()

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
                                    approx_acc_train = approx_acc_train/n_seen
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
                              if test_early_stopping_used==0:
                                    val_loss, mse_val, acc_val = np.nan, np.nan, np.nan
                              else:
                                    if module_training:
                                          test_update_loader = not(is_last_module) and (epoch == n_epochs_used-1)
                                          if is_last_module and loss_last_block=="mce":
                                                if type_of_task=="regression":
                                                      mse_val, val_loss, _ = get_loss_metric(model_wrapper, loader_val, type_of_task, criterion, scaler_y)
                                                else:
                                                      acc_val, val_loss, _ = get_loss_metric(model_wrapper, loader_val, type_of_task, criterion, scaler_y)
                                          elif is_last_module and loss_last_block=="layer_wise":
                                                _, val_loss = get_loss_metric_module(model_wrapper, original_model, loader_val, type_of_task, criterion, scaler_y, test_update_loader=test_update_loader, name_dataset=name_dataset, n_train_kept=n_train_kept)
                                                val_loss *= lambda_loss
                                                acc_val = compute_acc(model_wrapper.model, loader_val, model_wrapper.device, verbose=False)
                                                # if type_of_task=="regression":
                                                #       mse_val, val_loss, _ = get_loss_metric(model_wrapper, loader_val, type_of_task, criterion, scaler_y)
                                                # else:
                                                #       acc_val, val_loss, _ = get_loss_metric(model_wrapper, loader_val, type_of_task, criterion, scaler_y)
                                          else:
                                                acc_val, val_loss = get_loss_metric_module(model_wrapper, original_model, loader_val, type_of_task, criterion, scaler_y, test_update_loader=test_update_loader, name_dataset=name_dataset, n_train_kept=n_train_kept)
                                                acc_val = np.nan
                                                val_loss *= lambda_loss
                                          # # Updating dataset
                                          # if not(test_repeat_for_sparisty) and epoch==(n_epochs_used-1) and not(is_last_module):
                                          #       print("Updating dataset ...", flush=True)
                                          #       update_dataset(model_wrapper, original_model, loader_train, loader_val, n_train_kept)
                                          #       print("Done", flush=True)
                                    else:
                                          if type_of_task=="regression":
                                                mse_val, val_loss, _ = get_loss_metric(model_wrapper, loader_val, type_of_task, criterion, scaler_y)
                                          else:
                                                acc_val, val_loss, _ = get_loss_metric(model_wrapper, loader_val, type_of_task, criterion, scaler_y)
                              # Updating dataset    
                              if module_training and not(test_repeat_for_sparisty) and epoch==(n_epochs_used-1) and not(is_last_module):
                                    print("Updating dataset ...", flush=True)
                                    update_dataset(model_wrapper, original_model, loader_train, loader_val, n_train_kept)
                                    print("Done", flush=True)

                              if test_repeat_for_sparisty and epoch==(n_epochs_used-1):
                                    number_of_epochs_increase += 1
                                    n_epochs_used += n_epochs
                                    print(f"---- New number of epochs {n_epochs_used} ----")
                                    scheduler = initialize_scheduler(type_decay, optimizer, gamma_lr_decay, n_epochs, period_milestones, start_lr_decay, end_lr_decay, T_max_cos, eta_min_cos, warmup_steps)
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
                                    if phase_freeze and best_metric_freeze<current_metric:
                                          n_epochs_no_improvement_freeze += 1
                                          print("n_epochs_no_improvement_freeze =", n_epochs_no_improvement_freeze, flush = True)
                                          if n_epochs_no_improvement_freeze == patience_freeze:
                                                print("--- Unfreezing all weight_z and bias_z ---", flush = True)
                                                model_wrapper.unfreeze_all_z(d_named_parameters)
                                                phase_freeze = False
                                                n_epochs_no_improvement_freeze = 0
                                                # if n_mult_sel>=2:
                                                #       model_wrapper.selection_reg /= 2**(n_mult_sel//2)
                                                #       model_wrapper.entropy_reg /= 2**(n_mult_ent//2)
                                                #       n_mult_sel -= n_mult_sel//2
                                                #       n_mult_ent -= n_mult_ent//2
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

                              if test_compute_accurate_in_sample_loss:
                                    if type_of_task=="regression":
                                          mse_train, train_loss_with_no_pen, train_loss_with_pen = get_loss_metric(model_wrapper, loader_train, type_of_task, criterion, scaler_y)
                                    else:
                                          acc_train, train_loss_with_no_pen, train_loss_with_pen = get_loss_metric(model_wrapper, loader_train, type_of_task, criterion, scaler_y)

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

                              if mode=="layer_wise":
                                    if sparsity_total>=goal_sparsity:
                                          test_sparsity_reached = True

                              model_wrapper.test_sparsity_reached = test_sparsity_reached

                              if sparsity_becomes_reached:
                                    print("----", flush = True)
                                    print("Goal sparsity ("+str(goal_sparsity)+") reached at epoch", epoch, flush=True)
                                    print("----", flush = True)
                                    model_wrapper.freeze_all_z(d_named_parameters)
                                    if model_wrapper.type_function == "sigmoid":
                                          prune_models_external_sigmoid(model_wrapper.model, 1e-3)

                                    if test_early_stopping==2:
                                          test_early_stopping_used = 0
                                          import ipdb;ipdb.set_trace()
                                          loader_train.dataset.indices = [i for i in range(len(loader_train.dataset.dataset))]
                                          loader_val.dataset.indices = []
                                          # loader_train.dataset.indices = copy_indices_train
                                          # loader_val.dataset.indices = copy_indices_val

                              if test_early_stopping_used==1:
                                    if (type_of_task=="regression"):
                                          if (mse_val < best_val_mse) or condition_sparsity:
                                                print("--- CONDITION IMPROVEMENT (regression es) ---")
                                                best_val_mse = mse_val
                                                best_ep = epoch_counter
                                                if path_save!=None:
                                                      torch.save(model_wrapper.model.state_dict(), path_save)
                                                best_model = copy.deepcopy(model_wrapper.model)
                                                n_epochs_no_improvement = 0
                                          else:
                                                if test_sparsity_reached:
                                                      n_epochs_no_improvement += 1
                                    if (type_of_task=="classification"):
                                          if metric_early_stopping == "val_loss":
                                                condition_improvement = (val_loss < best_val_loss) or condition_sparsity
                                          elif metric_early_stopping == "val_accuracy":
                                                condition_improvement = (acc_val > best_val_acc) or condition_sparsity
                                          if condition_improvement:
                                                print("--- CONDITION IMPROVEMENT (classification es) ---")
                                                best_val_loss = val_loss
                                                best_val_acc = acc_val
                                                best_ep = epoch_counter
                                                if path_save!=None:
                                                      torch.save(model_wrapper.model.state_dict(), path_save)
                                                best_model = copy.deepcopy(model_wrapper.model)
                                                n_epochs_no_improvement = 0
                                          else:
                                                if test_sparsity_reached:
                                                      n_epochs_no_improvement += 1
                              else:
                                    if not(test_compute_accurate_in_sample_loss):
                                          train_loss_with_pen = approx_loss_in_sample_with_pen
                                          train_loss_with_no_pen = approx_loss_in_sample_no_pen
                                    # if module_training and (loss_last_block == "layer_wise" or not(is_last_module)):
                                    #       metric_in_sample = train_loss_with_no_pen
                                    # else:
                                    #       metric_in_sample = train_loss_with_pen
                                    if train_loss_with_pen < best_train_loss or condition_sparsity:
                                          print("--- CONDITION IMPROVEMENT (no es) ---")
                                          best_train_loss = train_loss_with_pen
                                          if type_of_task == "regression":
                                                best_val_mse = mse_val
                                          elif type_of_task == "classification":
                                                best_val_acc = acc_val
                                          best_ep = epoch_counter
                                          if path_save!=None:
                                                torch.save(model_wrapper.model.state_dict(), path_save)
                                          best_model = copy.deepcopy(model_wrapper.model)
                              test_model_stuck = False
                              tol_epsilon = 1e-6
                              if test_compute_accurate_in_sample_loss:
                                    print_loss_no_pen = "Exact in-sample loss with no pen"
                                    print_loss_with_pen = "Exact in-sample loss with pen"
                                    if type_of_task == "regression":
                                          if epoch >=1:
                                                test_model_stuck = abs(value_loss_with_pen-train_loss_with_pen)<=tol_epsilon and abs(value_loss_no_pen-train_loss_with_no_pen)<=tol_epsilon and abs(value_in_sample_metric - mse_train)<=tol_epsilon
                                          value_in_sample_metric = mse_train
                                          print_metric = "Exact in-sample MSE"
                                    elif type_of_task == "classification":
                                          if epoch >=1:
                                                test_model_stuck = abs(value_loss_with_pen-train_loss_with_pen)<=tol_epsilon and abs(value_loss_no_pen - train_loss_with_no_pen)<=tol_epsilon and abs(value_in_sample_metric - acc_train)<=tol_epsilon
                                          value_in_sample_metric = 100*acc_train
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
                                                test_model_stuck = abs(value_in_sample_metric-approx_acc_train)<=tol_epsilon and abs(value_loss_no_pen - approx_loss_in_sample_no_pen)<=tol_epsilon and abs(value_loss_with_pen - approx_loss_in_sample_with_pen)<=tol_epsilon
                                          value_in_sample_metric = 100*approx_acc_train
                                          print_metric = "Approx in-sample accuracy"
                                    value_loss_no_pen = approx_loss_in_sample_no_pen
                                    value_loss_with_pen = approx_loss_in_sample_with_pen

                              to_print = ""
                              if retraining_of_last_block:
                                    add_text = " (Retraining)"
                              else:
                                    add_text = ""
                              if module_training:
                                    to_print += f"Round {current_round+1}/{n_rounds}{add_text}, "
                              else:
                                    to_print += ""
                              
                              if type_of_task=="regression":
                                    if mode=="layer_wise":
                                          to_print += f'Epoch: {epoch:03d}, {print_loss_no_pen}: {value_loss_no_pen:.4f}, {print_loss_with_pen}: {value_loss_with_pen:.4f}, Validation loss: {val_loss:.4f}, {print_metric}: {value_in_sample_metric:.4f}, Validation MSE: {mse_val:.4f}, lr: {current_lr:.4f}, n_z: {n_z:.4f}, layer sparsity: {sparsity:4f}, total sparsity: {sparsity_total:4f}, sparsity storage: {sparsity_storage:4f}'
                                    elif mode=="ensemble":
                                          to_print += f'Epoch: {epoch:03d}, {print_loss_no_pen}: {value_loss_no_pen:.4f}, {print_loss_with_pen}: {value_loss_with_pen:.4f}, Validation loss: {val_loss:.4f}, {print_metric}: {value_in_sample_metric:.4f}, Validation MSE: {mse_val:.4f}, lr: {current_lr:.4f}, n_z: {n_z:.4f}, sparsity: {sparsity:4f}, sparsity storage: {sparsity_storage:4f}'
                              elif type_of_task == "classification":
                                    if mode=="layer_wise":
                                          to_print += f'Epoch: {epoch:03d}, {print_loss_no_pen}: {value_loss_no_pen:.4f}, {print_loss_with_pen}: {value_loss_with_pen:.4f}, Validation loss: {val_loss:.4f}, {print_metric}: {value_in_sample_metric:.4f}, Val Acc: {100*acc_val:.4f}, lr: {current_lr:.4f}, n_z: {n_z:.4f}, layer sparsity: {sparsity:4f}, total sparsity: {sparsity_total:4f}, sparsity storage: {sparsity_storage:4f}'
                                    elif mode=="ensemble":
                                          to_print += f'Epoch: {epoch:03d}, {print_loss_no_pen}: {value_loss_no_pen:.4f}, {print_loss_with_pen}: {value_loss_with_pen:.4f}, Validation loss: {val_loss:.4f}, {print_metric}: {value_in_sample_metric:.4f}, Val Acc: {100*acc_val:.4f}, lr: {current_lr:.4f}, n_z: {n_z:.4f}, sparsity: {sparsity:4f}, sparsity storage: {sparsity_storage:4f}'
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
                                    l_validation_metric.append(mse_val)
                                    l_in_sample_metric.append(value_in_sample_metric)
                              elif type_of_task == "classification":
                                    l_validation_metric.append(acc_val)
                                    l_in_sample_metric.append(value_in_sample_metric)
                              l_times_epochs.append(time.time()-start_epoch)
                              if np.isnan(value_loss_with_pen):
                                    print("---", flush = True)
                                    print("Loss became NaN: end of the training", flush = True)
                                    print("---", flush = True)
                                    # l_lr = l_lr + [np.nan for _ in range(max(n_epochs_used-epoch-1,0))]
                                    # l_in_sample_loss = l_in_sample_loss + [np.nan for _ in range(max(n_epochs_used-epoch-1,0))]
                                    # l_in_sample_metric = l_in_sample_metric + [np.nan for _ in range(max(n_epochs_used-epoch-1,0))]
                                    # l_validation_loss = l_validation_loss + [np.nan for _ in range(max(n_epochs_used-epoch-1,0))]
                                    # l_validation_metric = l_validation_metric + [np.nan for _ in range(max(n_epochs_used-epoch-1,0))]
                                    # l_n_z = l_n_z + [np.nan for _ in range(max(n_epochs_used-epoch-1,0))]
                                    # l_times_epochs = l_times_epochs + [np.nan for _ in range(max(n_epochs_used-epoch-1,0))]
                                    if mode=="layer_wise" and loss_func == "layer_wise" and epoch < (n_epochs_used-1):
                                          update_dataset(model_wrapper, original_model, loader_train, loader_val, n_train_kept)
                                          # for batch_sgd in tqdm(loader_train):
                                          #       input_batch_sgd, input_batch_original_sgd, target_batch_sgd, index_seen_sgd, old_index_seen_sgd = read_batch(batch_sgd)
                                          #       with torch.no_grad():
                                          #             output = model_wrapper.model(input_batch_sgd.to(model_wrapper.device))
                                          #             output_original = original_model(input_batch_sgd.to(model_wrapper.device))
                                          #       update_loader_first(loader_train, output, target_batch_sgd, index_seen_sgd, old_index_seen_sgd, n_train_kept)
                                          # acc_val, val_loss = get_loss_metric_module(model_wrapper, original_model, loader_val, type_of_task, criterion, scaler_y, test_update_loader=True, name_dataset=name_dataset, n_train_kept=n_train_kept)
                                          # update_loader_final(loader_train, loader_val, n_train_kept)

                                          # if loader_train.dataset.dataset.is_original:
                                          #       loader_train.dataset.indices = list(np.arange(len(loader_train.dataset.indices)))
                                          #       loader_val.dataset.indices = list(len(loader_train.dataset.indices)+np.arange(len(loader_val.dataset.indices)))
                                          #       loader_train.dataset.dataset.is_original = False
                                          # loader_train.dataset.dataset.data = torch.cat(loader_train.dataset.dataset.new_data)
                                          # loader_train.dataset.dataset.targets = torch.cat(loader_train.dataset.dataset.new_targets)
                                          # loader_train.dataset.dataset.required_increment = np.zeros(loader_train.dataset.dataset.data.shape[0], dtype=int)
                                          # loader_train.dataset.dataset.new_data = []
                                          # loader_train.dataset.dataset.new_targets = []
                                    break
                              if test_model_stuck:
                                    print("---", flush = True)
                                    print("Model got stuck: end of the training", flush = True)
                                    print("---", flush = True)
                                    # l_lr = l_lr + [l_lr[-1] for _ in range(max(n_epochs_used-epoch-1,0))]
                                    # l_in_sample_loss = l_in_sample_loss + [l_in_sample_loss[-1] for _ in range(max(n_epochs_used-epoch-1,0))]
                                    # l_in_sample_metric = l_in_sample_metric + [l_in_sample_metric[-1] for _ in range(max(n_epochs_used-epoch-1,0))]
                                    # l_validation_loss = l_validation_loss + [l_validation_loss[-1] for _ in range(max(n_epochs_used-epoch-1,0))]
                                    # l_validation_metric = l_validation_metric + [l_validation_metric[-1] for _ in range(max(n_epochs_used-epoch-1,0))]
                                    # l_n_z = l_n_z + [l_n_z[-1] for _ in range(max(n_epochs_used-epoch-1,0))]
                                    # l_times_epochs = l_times_epochs + [l_times_epochs[-1] for _ in range(max(n_epochs_used-epoch-1,0))]
                                    if module_training and epoch < (n_epochs_used-1):
                                          update_dataset(model_wrapper, original_model, loader_train, loader_val, n_train_kept)
                                          # for batch_sgd in tqdm(loader_train):
                                          #       input_batch_sgd, input_batch_original_sgd, target_batch_sgd, index_seen_sgd, old_index_seen_sgd = read_batch(batch_sgd)
                                          #       with torch.no_grad():
                                          #             output = model_wrapper.model(input_batch_sgd.to(model_wrapper.device))
                                          #             output_original = original_model(input_batch_sgd.to(model_wrapper.device))
                                          #       update_loader_first(loader_train, output, target_batch_sgd, index_seen_sgd, old_index_seen_sgd, n_train_kept)
                                          # acc_val, val_loss = get_loss_metric_module(model_wrapper, original_model, loader_val, type_of_task, criterion, scaler_y, test_update_loader=True, name_dataset=name_dataset, n_train_kept=n_train_kept)
                                          # update_loader_final(loader_train, loader_val, n_train_kept)

                                          # if loader_train.dataset.dataset.is_original:
                                          #       loader_train.dataset.indices = list(np.arange(len(loader_train.dataset.indices)))
                                          #       loader_val.dataset.indices = list(len(loader_train.dataset.indices)+np.arange(len(loader_val.dataset.indices)))
                                          #       loader_train.dataset.dataset.is_original = False
                                          # loader_train.dataset.dataset.data = torch.cat(loader_train.dataset.dataset.new_data)
                                          # loader_train.dataset.dataset.targets = torch.cat(loader_train.dataset.dataset.new_targets)
                                          # loader_train.dataset.dataset.required_increment = np.zeros(loader_train.dataset.dataset.data.shape[0], dtype=int)
                                          # loader_train.dataset.dataset.new_data = []
                                          # loader_train.dataset.dataset.new_targets = []
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
                              if module_training and epoch < (n_epochs_used-1):
                                    update_dataset(model_wrapper, original_model, loader_train, loader_val, n_train_kept)
                                    # for batch_sgd in tqdm(loader_train):
                                    #       input_batch_sgd, input_batch_original_sgd, target_batch_sgd, index_seen_sgd, old_index_seen_sgd = read_batch(batch_sgd)
                                    #       with torch.no_grad():
                                    #             output = model_wrapper.model(input_batch_sgd.to(model_wrapper.device))
                                    #             output_original = original_model(input_batch_sgd.to(model_wrapper.device))
                                    #       update_loader_first(loader_train, output, target_batch_sgd, index_seen_sgd, old_index_seen_sgd, n_train_kept)
                                    # acc_val, val_loss = get_loss_metric_module(model_wrapper, original_model, loader_val, type_of_task, criterion, scaler_y, test_update_loader=True, name_dataset=name_dataset, n_train_kept=n_train_kept)
                                    # update_loader_final(loader_train, loader_val, n_train_kept)
                                    
                                    # if loader_train.dataset.dataset.is_original:
                                    #       loader_train.dataset.indices = list(np.arange(len(loader_train.dataset.indices)))
                                    #       loader_val.dataset.indices = list(len(loader_train.dataset.indices)+np.arange(len(loader_val.dataset.indices)))
                                    #       loader_train.dataset.dataset.is_original = False
                                    # loader_train.dataset.dataset.data = torch.cat(loader_train.dataset.dataset.new_data)
                                    # loader_train.dataset.dataset.targets = torch.cat(loader_train.dataset.dataset.new_targets)
                                    # loader_train.dataset.dataset.required_increment = np.zeros(loader_train.dataset.dataset.data.shape[0], dtype=int)
                                    # loader_train.dataset.dataset.new_data = []
                                    # loader_train.dataset.dataset.new_targets = []
                              break
            if mode == "layer_wise" and name_module!=-1:
                  d_modules[name_module].requires_grad_(False)
            if mode == "layer_wise" and loss_func=="layer_wise" and name_module!=-1 and not(module_training):
                  handle.remove()
                  handle_original.remove()
      if epoch_counter>=0:
            n_z_final = l_n_z[epoch_counter]
      else:
            n_z_final = model_wrapper.get_n_z(test_grad=False)
      return l_in_sample_loss, l_in_sample_loss_no_pen, l_validation_loss, l_in_sample_metric, l_validation_metric, l_times_epochs, l_lr, l_n_z, l_sparsity, l_sparsity_storage, l_n_params, best_model, best_ep, n_z_final, test_sparsity_reached

