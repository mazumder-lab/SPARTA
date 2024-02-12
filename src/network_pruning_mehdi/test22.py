def epochs_training(n_epochs, n_epochs_no_improvement, patience, test_early_stopping, epoch_counter, model_wrapper, goal_sparsity, name_model, mode, type_of_task, loader_train, loader_val, name_module, criterion, d_layer_output, d_layer_output_original, type_decay, T_max_cos, eta_min_cos, scaler_y, test_save_all_models, folder_saves, name_study, trial, ind_repeat, d_modules, l_name_modules, l_n_z, metric_early_stopping, test_compute_accurate_in_sample_loss, l_lr, l_in_sample_loss, l_validation_loss, l_n_params, l_validation_metric, l_in_sample_metric, l_times_epochs, path_save, loss_func, original_model, test_sparsity_reached, sparsity, optimizer, scheduler, best_val_loss, best_val_mse, best_train_loss, best_val_acc):
      for epoch in range(n_epochs):
            if (n_epochs_no_improvement < patience) or not(test_early_stopping):
                  epoch_counter+=1
                  start_epoch = time.time()
                  if test_sparsity_reached:
                        print("n_epochs_no_improvement =", n_epochs_no_improvement)
                  else:
                        n_z_close_to_1 = model_wrapper.get_n_z_close_to_1(test_grad=True)
                        n_z = model_wrapper.get_n_z(test_grad=True)
                        print("Ratio of z equal to 1:", n_z_close_to_1/n_z, flush=True)
                        if n_z_close_to_1/n_z >= 0.95:
                              first_term = 1.05*(goal_sparsity-sparsity+1e-4)
                              if "mlpnet" in name_model:
                                    prop_reset = get_prop_reset_mnist(sparsity, first_term)
                              elif "resnet" in name_model:
                                    prop_reset = get_prop_reset_resnet20(sparsity, first_term)
                              if mode=="layer_wise":
                                    prop_reset = np.nanmax([prop_reset, 1/n_z])
                              
                              # if model_wrapper.type_pruning == "layer_wise":
                              #       optimizer.zero_grad()
                              # model_wrapper.model.train()
                              model_wrapper.compute_z()
                              n_reset = model_wrapper.reset_z(prop_reset=prop_reset)
                              print("-----", flush=True)
                              print(str(n_reset)+" z-weights have been reset", n_z_close_to_1/n_z, flush=True)
                              print("-----", flush=True)
                  loss_pred_in_sample = 0
                  # if sparsity >=0.7:
                  #       #list(model_wrapper.model.children())[-1].weight.requires_grad = False
                  #       list(model_wrapper.model.children())[-1].weight_z.requires_grad = False
                  approx_loss_in_sample_with_pen = 0
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
                        acc_batch += 1
                        # # To delete later
                        # if not(model_wrapper.dense_to_sparse):
                        #       pass
                        #       #torch.save(model_wrapper.model.state_dict(), "saves_weight_dense_temp/model_"+str(model_wrapper.step_temp))
                        # else:
                        #       weights_dense = torch.load("saves_weight_dense_temp/model_"+str(model_wrapper.step_temp))

                        #       fc1_sparse_weight = model_wrapper.model.fc1.weight
                        #       fc1_sparse_weight_z = model_wrapper.model.fc1.weight_z
                        #       fc2_sparse_weight = model_wrapper.model.fc2.weight
                        #       fc2_sparse_weight_z = model_wrapper.model.fc2.weight_z
                        #       fc3_sparse_weight = model_wrapper.model.fc3.weight
                        #       fc3_sparse_weight_z = model_wrapper.model.fc3.weight_z
                              
                        #       fc1_dense_weight = weights_dense["fc1.weight"]
                        #       fc1_dense_weight_z = weights_dense["fc1.weight_z"]
                        #       fc2_dense_weight = weights_dense["fc2.weight"]
                        #       fc2_dense_weight_z = weights_dense["fc2.weight_z"]
                        #       fc3_dense_weight = weights_dense["fc3.weight"]
                        #       fc3_dense_weight_z = weights_dense["fc3.weight_z"]

                        #       dense_z_1 = compute_z_from_tensor(fc1_dense_weight_z, 1.0)
                        #       dense_z_2 = compute_z_from_tensor(fc2_dense_weight_z, 1.0)
                        #       dense_z_3 = compute_z_from_tensor(fc3_dense_weight_z, 1.0)

                        #       to_keep_in_1 = torch.where(torch.sum(dense_z_1, 0)!=0)[0]
                        #       to_keep_out_1 = torch.where(torch.sum(dense_z_1, 1)!=0)[0]
                        #       to_keep_in_2 = torch.where(torch.sum(dense_z_2, 0)!=0)[0]
                        #       to_keep_out_2 = torch.where(torch.sum(dense_z_2, 1)!=0)[0]
                        #       to_keep_in_3 = torch.where(torch.sum(dense_z_3, 0)!=0)[0]
                        #       to_keep_out_3 = torch.where(torch.sum(dense_z_3, 1)!=0)[0]

                        #       to_keep_out_1 = torch.Tensor(np.intersect1d(to_keep_out_1, to_keep_in_2)).long()
                        #       to_keep_in_2 = torch.Tensor(np.intersect1d(to_keep_out_1, to_keep_in_2)).long()

                        #       to_keep_out_2 = torch.Tensor(np.intersect1d(to_keep_out_2, to_keep_in_3)).long()
                        #       to_keep_in_3 = torch.Tensor(np.intersect1d(to_keep_out_2, to_keep_in_3)).long()

                        #       fc1_dense_weight = fc1_dense_weight[to_keep_out_1]
                        #       fc1_dense_weight = fc1_dense_weight[:,to_keep_in_1]
                        #       fc1_dense_weight_z = fc1_dense_weight_z[to_keep_out_1]
                        #       fc1_dense_weight_z = fc1_dense_weight_z[:,to_keep_in_1]
                        #       fc2_dense_weight = fc2_dense_weight[to_keep_out_2]
                        #       fc2_dense_weight = fc2_dense_weight[:,to_keep_in_2]
                        #       fc2_dense_weight_z = fc2_dense_weight_z[to_keep_out_2]
                        #       fc2_dense_weight_z = fc2_dense_weight_z[:,to_keep_in_2]
                        #       fc3_dense_weight = fc3_dense_weight[to_keep_out_3]
                        #       fc3_dense_weight = fc3_dense_weight[:,to_keep_in_3]
                        #       fc3_dense_weight_z = fc3_dense_weight_z[to_keep_out_3]
                        #       fc3_dense_weight_z = fc3_dense_weight_z[:,to_keep_in_3]

                        #       model_wrapper.fc1_dense_weight_test = fc1_dense_weight

                        #       try:
                        #             test_1 = torch.sum(fc1_dense_weight!=fc1_sparse_weight)==0
                        #             test_2 = torch.sum(fc1_dense_weight_z!=fc1_sparse_weight_z)==0
                        #             test_3 = torch.sum(fc2_dense_weight!=fc2_sparse_weight)==0
                        #             test_4 = torch.sum(fc2_dense_weight_z!=fc2_sparse_weight_z)==0
                        #             test_5 = torch.sum(fc3_dense_weight!=fc3_sparse_weight)==0
                        #             test_6 = torch.sum(fc3_dense_weight_z!=fc3_sparse_weight_z)==0
                        #       except:
                        #             import ipdb;ipdb.set_trace()
                        #       test_complet = test_1 and test_2 and test_3 and test_4 and test_5 and test_6
                        #       if not(test_complet):
                        #             import ipdb;ipdb.set_trace()
                        model_wrapper.step_temp += 1
                        # # End
                        n_batch = batch_sgd[0].shape[0]
                        n_seen += n_batch
                        optimizer.zero_grad()
                        output = model_wrapper.model(batch_sgd[0].to(model_wrapper.device))
                        if model_wrapper.type_pruning=="layer_wise":
                              tot_layer_wise_loss = model_wrapper.compute_layer_wise_loss()
                              tot_layer_wise_loss.backward(retain_graph = True)
                              model_wrapper.save_grad_layer_wise()

                        optimizer.zero_grad()

                        y_truth = batch_sgd[1]
                        y_truth = y_truth.to(model_wrapper.device)
                        approx_acc_train += torch.sum(torch.argmax(torch.softmax(output, 1), 1) == y_truth).detach().item()
                        if loss_func == "layer_wise":
                              original_model(batch_sgd[0].to(model_wrapper.device))
                              if mode=="layer_wise" and name_module!=-1:
                                    loss = torch.mean((d_layer_output[name_module] - d_layer_output_original[name_module])**2)
                              elif name_module==-1:
                                    loss = criterion(output, y_truth)
                              else:
                                    loss = 0
                                    for key_layer in d_layer_output:
                                          loss += torch.mean((d_layer_output[key_layer] - d_layer_output_original[key_layer])**2)

                        else:
                              loss = criterion(output, y_truth)
                        loss_pred_in_sample += n_batch*loss.detach().item()
                        entropy_loss, selection_loss, l2_loss = model_wrapper.get_losses()
                        # if model_wrapper.step_temp == 446:
                        #       # if not(model_wrapper.dense_to_sparse):
                        #       #       np.save("output_dense_445.npy", output.data.numpy())
                        #       import ipdb;ipdb.set_trace()
                        loss += entropy_loss + selection_loss + l2_loss
                        approx_loss_in_sample_with_pen += n_batch*loss.detach().item()
                        loss.backward()  # Derive gradients.
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
                        optimizer.step()  # Update parameters based on gradients.

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
                        model_wrapper.compute_z()
                        optimizer, test_pruned = model_wrapper.prune_models()
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
                  # l_children = list(model_wrapper.model.children())
                  # print("first:", l_children[0].weight.shape)
                  # print("second:", l_children[1].weight.shape)
                  # print("third:", l_children[2].weight.shape)
                  #print("Cuda memory loop:", torch.cuda.memory_allocated("cuda"))
                  approx_loss_in_sample_no_pen = loss_pred_in_sample/n_seen
                  approx_loss_in_sample_with_pen = approx_loss_in_sample_with_pen/n_seen
                  if type_of_task=="classification":
                        approx_acc_train = approx_acc_train/n_seen
                  try:
                        approx_loss_in_sample_no_pen = approx_loss_in_sample_no_pen.detach().item()
                  except:
                        pass
                  optimizer.zero_grad()
                  model_wrapper.model.eval()
                  #print("Cuda memory before val:", torch.cuda.memory_allocated("cuda"))
                  if type_of_task=="regression":
                        mse_val, val_loss, _ = get_loss_metric(model_wrapper, loader_val, type_of_task, criterion, scaler_y)
                  else:
                        acc_val, val_loss, _ = get_loss_metric(model_wrapper, loader_val, type_of_task, criterion, scaler_y)

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

                  if mode == "layer_wise":
                        n_z = model_wrapper.get_n_z(test_grad=True)
                        n_z_total = model_wrapper.get_n_z(test_grad=False)
                        n_params_current = np.sum([np.prod(x[1].shape) for x in model_wrapper.model.named_parameters() if "_z" in x[0]])
                        n_params_original_z = np.sum([d_modules[x].n_weights for x in l_name_modules if x!=-1 and d_modules[x].weight.requires_grad])
                        n_params_dense_total = model_wrapper.n_params_original_z
                        #sparsity = 1 - n_z/model_wrapper.n_params_original_z
                        sparsity = 1 - n_z/n_params_original_z
                        sparsity_total = 1 - n_z_total/n_params_dense_total
                        sparsity_storage = 1 - n_params_current/n_params_dense_total
                  elif mode=="ensemble":
                        n_z = model_wrapper.get_n_z(test_grad=False)
                        n_params_current = np.sum([np.prod(x[1].shape) for x in model_wrapper.model.named_parameters() if "_z" in x[0]])
                        n_params_original_z = model_wrapper.n_params_original_z
                        sparsity = 1 - n_z/n_params_original_z
                        sparsity_storage = 1 - n_params_current/n_params_original_z

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

                  if sparsity_becomes_reached:
                        print("----", flush = True)
                        print("Goal sparsity ("+str(goal_sparsity)+") reached at epoch", epoch, flush=True)
                        print("----", flush = True)
                        #model_wrapper.l2_reg = 0.0
                        #import ipdb;ipdb.set_trace()

                  if test_early_stopping==1:
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

                  if type_of_task=="regression":
                        if mode=="layer_wise":
                              print(f'Epoch: {epoch:03d}, {print_loss_no_pen}: {value_loss_no_pen:.4f}, {print_loss_with_pen}: {value_loss_with_pen:.4f}, Validation loss: {val_loss:.4f}, {print_metric}: {value_in_sample_metric:.4f}, Validation MSE: {mse_val:.4f}, lr: {current_lr:.4f}, n_z: {n_z:.4f}, layer sparsity: {sparsity:4f}, total sparsity: {sparsity_total:4f}, sparsity storage: {sparsity_storage:4f}', flush=True)
                        elif mode=="ensemble":
                              print(f'Epoch: {epoch:03d}, {print_loss_no_pen}: {value_loss_no_pen:.4f}, {print_loss_with_pen}: {value_loss_with_pen:.4f}, Validation loss: {val_loss:.4f}, {print_metric}: {value_in_sample_metric:.4f}, Validation MSE: {mse_val:.4f}, lr: {current_lr:.4f}, n_z: {n_z:.4f}, sparsity: {sparsity:4f}, sparsity storage: {sparsity_storage:4f}', flush=True)
                  elif type_of_task == "classification":
                        if mode=="layer_wise":
                              print(f'Epoch: {epoch:03d}, {print_loss_no_pen}: {value_loss_no_pen:.4f}, {print_loss_with_pen}: {value_loss_with_pen:.4f}, Validation loss: {val_loss:.4f}, {print_metric}: {value_in_sample_metric:.4f}, Val Acc: {100*acc_val:.4f}, lr: {current_lr:.4f}, n_z: {n_z:.4f}, layer sparsity: {sparsity:4f}, total sparsity: {sparsity_total:4f}, sparsity storage: {sparsity_storage:4f}', flush=True)
                        elif mode=="ensemble":
                              print(f'Epoch: {epoch:03d}, {print_loss_no_pen}: {value_loss_no_pen:.4f}, {print_loss_with_pen}: {value_loss_with_pen:.4f}, Validation loss: {val_loss:.4f}, {print_metric}: {value_in_sample_metric:.4f}, Val Acc: {100*acc_val:.4f}, lr: {current_lr:.4f}, n_z: {n_z:.4f}, sparsity: {sparsity:4f}, sparsity storage: {sparsity_storage:4f}', flush=True)
                  l_lr.append(current_lr)
                  l_in_sample_loss.append(value_loss_with_pen)
                  l_validation_loss.append(val_loss)
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
                        # l_lr = l_lr + [np.nan for _ in range(max(n_epochs-epoch-1,0))]
                        # l_in_sample_loss = l_in_sample_loss + [np.nan for _ in range(max(n_epochs-epoch-1,0))]
                        # l_in_sample_metric = l_in_sample_metric + [np.nan for _ in range(max(n_epochs-epoch-1,0))]
                        # l_validation_loss = l_validation_loss + [np.nan for _ in range(max(n_epochs-epoch-1,0))]
                        # l_validation_metric = l_validation_metric + [np.nan for _ in range(max(n_epochs-epoch-1,0))]
                        # l_n_z = l_n_z + [np.nan for _ in range(max(n_epochs-epoch-1,0))]
                        # l_times_epochs = l_times_epochs + [np.nan for _ in range(max(n_epochs-epoch-1,0))]
                        break
                  if test_model_stuck:
                        print("---", flush = True)
                        print("Model got stuck: end of the training", flush = True)
                        print("---", flush = True)
                        # l_lr = l_lr + [l_lr[-1] for _ in range(max(n_epochs-epoch-1,0))]
                        # l_in_sample_loss = l_in_sample_loss + [l_in_sample_loss[-1] for _ in range(max(n_epochs-epoch-1,0))]
                        # l_in_sample_metric = l_in_sample_metric + [l_in_sample_metric[-1] for _ in range(max(n_epochs-epoch-1,0))]
                        # l_validation_loss = l_validation_loss + [l_validation_loss[-1] for _ in range(max(n_epochs-epoch-1,0))]
                        # l_validation_metric = l_validation_metric + [l_validation_metric[-1] for _ in range(max(n_epochs-epoch-1,0))]
                        # l_n_z = l_n_z + [l_n_z[-1] for _ in range(max(n_epochs-epoch-1,0))]
                        # l_times_epochs = l_times_epochs + [l_times_epochs[-1] for _ in range(max(n_epochs-epoch-1,0))]
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
                  break
      return best_model, best_ep, epoch_counter, test_sparsity_reached

