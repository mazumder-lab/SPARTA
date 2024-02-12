#%%
import matplotlib.pyplot as plt
l_accuracy_smooth_step = [91.3599967956543, 91.58999919891357, 91.1300003528595, 90.85999727249146, 89.50999975204468, 85.75999736785889, 74.80999827384949]
l_sparsity_smooth_step = [0.2970919802440699, 0.3986142794917794, 0.5016057230183163, 0.6072659889408134, 0.7084339217293083, 0.8024296250359904, 0.8682347382486915]

l_accuracy_discrete = [91.333333333333333, 90, 88.333333333333333, 81.333333333333333, 78.333333333333333, 53.333333333333333, 32.666666666666667]
l_sparsity_discrete = [0.5, 0.6, 0.7, 0.8, 0.825, 0.875, 0.9]

plt.figure()
plt.plot(l_sparsity_smooth_step, l_accuracy_smooth_step, label='Smooth Step')
plt.plot(l_sparsity_discrete, l_accuracy_discrete, label='Discrete')
plt.xlabel("sparisty")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Resnet20, layerwise reconstruction (smoothstep)")
plt.savefig("summary_resnet20.png")

#%%

from utils_training import *

def train_neural_network(name_study, model_wrapper_dense, model_wrapper_sparse, dataset, optimizer_dense, optimizer_sparse, criterion, n_epochs, batch_size_dataset, path_save, test_early_stopping, trial,  test_save_all_models=False, type_decay="exponential", gamma_lr_decay=np.exp(-np.log(25)/10000), T_max_cos=10, eta_min_cos=1e-5, start_lr_decay=1e-2, end_lr_decay=1e-5, warmup_steps=100, type_of_task = "regression", test_compute_accurate_in_sample_loss = 0, folder_saves = "TSML_saves", ind_repeat=0, patience=50, metric_early_stopping="val_loss", period_milestones=25, goal_sparsity=0.0, type_training="combined", n_restart=0, num_workers=4):
      loader_train, loader_val, loader_test, scaler_y = dataset
      test_sparsity_reached = False

      l_lr = []
      l_in_sample_loss = []
      l_in_sample_metric = []
      l_validation_loss = []
      l_validation_metric = []
      l_n_z = []
      l_times_epochs = []

      best_ep = 0
      best_model = copy.deepcopy(model_wrapper_dense.model)
      best_val_mse = np.inf
      best_train_loss = np.inf
      best_val_loss = np.inf
      best_val_acc = -np.inf

      n_epochs_no_improvement = 0

      # Initialize z and sparsity
      model_wrapper_dense.compute_z()
      model_wrapper_sparse.compute_z()
      sparsity = 0

      if type_training == "combined":
            n_restart = 0.5

      l_initial_lr = []
      for idx_group_param in range(len(optimizer_dense.param_groups)):
            l_initial_lr.append(optimizer_dense.param_groups[idx_group_param]["lr"])
      for idx_group_param in range(len(optimizer_sparse.param_groups)):
            l_initial_lr.append(optimizer_sparse.param_groups[idx_group_param]["lr"])

      epoch_counter = -1
      for i in range(int(2*n_restart)):

            scheduler_dense = initialize_scheduler(type_decay, optimizer_dense, gamma_lr_decay, n_epochs, period_milestones, start_lr_decay, end_lr_decay, T_max_cos, eta_min_cos, warmup_steps)
            scheduler_sparse = initialize_scheduler(type_decay, optimizer_sparse, gamma_lr_decay, n_epochs, period_milestones, start_lr_decay, end_lr_decay, T_max_cos, eta_min_cos, warmup_steps)

            for idx_group_param in range(len(optimizer_dense.param_groups)):
                  optimizer_dense.param_groups[idx_group_param]["lr"] = l_initial_lr[idx_group_param]
            for idx_group_param in range(len(optimizer_sparse.param_groups)):
                  optimizer_sparse.param_groups[idx_group_param]["lr"] = l_initial_lr[idx_group_param]

            # To delete later
            model_wrapper_dense.step_temp = 0
            model_wrapper_sparse.step_temp = 0
            # End
            for epoch in range(n_epochs):
                  if (n_epochs_no_improvement < patience) or not(test_early_stopping):
                        epoch_counter+=1
                        start_epoch = time.time()
                        if test_sparsity_reached:
                              print("n_epochs_no_improvement =", n_epochs_no_improvement)
                        else:
                              n_z_close_to_1 = model_wrapper_dense.get_n_z_close_to_1()
                              n_z = model_wrapper_dense.get_n_z()
                              print("Ratio of z equal to 1:", n_z_close_to_1/n_z, flush=True)
                              if n_z_close_to_1/n_z >= 0.95:
                                    first_term = 1.05*(goal_sparsity-sparsity+1e-4)
                                    prop_reset = get_prop_reset_mnist(sparsity, first_term)
                                    n_reset_dense = model_wrapper_dense.reset_z(prop_reset=prop_reset)
                                    n_reset_sparse = model_wrapper_sparse.reset_z(prop_reset=prop_reset)
                                    print("-----", flush=True)
                                    print(str(n_reset_dense)+" z-dense-weights have been reset", n_z_close_to_1/n_z, flush=True)
                                    print(str(n_reset_sparse)+" z-sparse-weights have been reset", n_z_close_to_1/n_z, flush=True)
                                    print("-----", flush=True)
                        loss_pred_in_sample = 0
                        # if sparsity >=0.7:
                        #       #list(model_wrapper.model.children())[-1].weight.requires_grad = False
                        #       list(model_wrapper.model.children())[-1].weight_z.requires_grad = False
                        approx_loss_in_sample_with_pen = 0
                        if type_of_task=="classification":
                              approx_acc_train = 0
                        model_wrapper_dense.model.train()
                        model_wrapper_sparse.model.train()
                        current_lr = optimizer_dense.param_groups[0]["lr"]
                        print("current lr:", current_lr)
                        n_seen = 0
                        acc_batch = 0
                        if model_wrapper_dense.type_pruning=="layer_wise":
                              model_wrapper_dense.initialize_pruning()
                              model_wrapper_sparse.initialize_pruning()
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
                              model_wrapper_dense.step_temp += 1
                              model_wrapper_sparse.step_temp += 1
                              # # End
                              n_batch = batch_sgd[0].shape[0]
                              n_seen += n_batch
                              optimizer_dense.zero_grad()
                              optimizer_sparse.zero_grad()
                              output_dense = model_wrapper_dense.model(batch_sgd[0].to(model_wrapper_dense.device))
                              output_sparse = model_wrapper_dense.model(batch_sgd[0].to(model_wrapper_dense.device))
                              if model_wrapper_dense.type_pruning=="layer_wise":
                                    tot_layer_wise_loss_dense = model_wrapper_dense.compute_layer_wise_loss()
                                    tot_layer_wise_loss_dense.backward(retain_graph = True)
                                    tot_layer_wise_loss_sparse = model_wrapper_sparse.compute_layer_wise_loss()
                                    tot_layer_wise_loss_sparse.backward(retain_graph = True)
                              model_wrapper_dense.save_grad_layer_wise()
                              model_wrapper_sparse.save_grad_layer_wise()
                              optimizer_dense.zero_grad()
                              optimizer_sparse.zero_grad()

                              y_truth = batch_sgd[1]
                              y_truth = y_truth.to(model_wrapper_dense.device)
                              approx_acc_train += torch.sum(torch.argmax(torch.softmax(output, 1), 1) == y_truth).detach().item()
                              loss = criterion(output_dense, y_truth)
                              loss_sparse = criterion(output_sparse, y_truth)
                              loss_pred_in_sample += n_batch*loss.detach().item()
                              entropy_loss, selection_loss, l2_loss = model_wrapper_dense.get_losses()
                              entropy_loss_sparse, selection_loss_sparse, l2_loss_sparse = model_wrapper_sparse.get_losses()
                              # if model_wrapper.step_temp == 446:
                              #       # if not(model_wrapper.dense_to_sparse):
                              #       #       np.save("output_dense_445.npy", output.data.numpy())
                              #       import ipdb;ipdb.set_trace()
                              loss += entropy_loss + selection_loss + l2_loss
                              approx_loss_in_sample_with_pen += n_batch*loss.detach().item()
                              loss.backward()  # Derive gradients.
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
                              model_wrapper_dense.compute_z()
                              model_wrapper_sparse.compute_z()
                              optimizer_dense, test_pruned = model_wrapper_dense.prune_models()
                              optimizer_sparse, test_pruned = model_wrapper_sparse.prune_models()

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

                        l_children = list(model_wrapper_sparse.model.children())
                        print("first:", l_children[0].weight.shape)
                        print("second:", l_children[1].weight.shape)
                        print("third:", l_children[2].weight.shape)
                        #print("Cuda memory loop:", torch.cuda.memory_allocated("cuda"))
                        approx_loss_in_sample_no_pen = loss_pred_in_sample/n_seen
                        approx_loss_in_sample_with_pen = approx_loss_in_sample_with_pen/n_seen
                        if type_of_task=="classification":
                              approx_acc_train = approx_acc_train/n_seen
                        try:
                              approx_loss_in_sample_no_pen = approx_loss_in_sample_no_pen.detach().item()
                        except:
                              pass
                        optimizer_dense.zero_grad()
                        optimizer_sparse.zero_grad()
                        model_wrapper_dense.model.eval()
                        model_wrapper_sparse.model.eval()
                        #print("Cuda memory before val:", torch.cuda.memory_allocated("cuda"))
                        acc_val, val_loss, _ = get_loss_metric(model_wrapper_dense, loader_val, type_of_task, criterion, scaler_y)
                        acc_val_sparse, val_loss_sparse, _ = get_loss_metric(model_wrapper_sparse, loader_val, type_of_task, criterion, scaler_y)

                        n_z = model_wrapper_dense.get_n_z()
                        sparsity = 1 - n_z/model_wrapper_dense.n_params_original_z

                        n_z_sparse = model_wrapper_sparse.get_n_z()
                        sparsity_sparse = 1 - n_z/model_wrapper_sparse.n_params_original_z

                        if len(l_n_z)>0:
                              old_sparsity = 1 - l_n_z[-1]/model_wrapper_dense.n_params_original_z
                              sparsity_increases_before_goal_being_reached = not(test_sparsity_reached) and (sparsity>old_sparsity)
                        else:
                              sparsity_increases_before_goal_being_reached = False

                        sparsity_becomes_reached = not(test_sparsity_reached) and (sparsity>=goal_sparsity)
                        condition_sparsity = sparsity_increases_before_goal_being_reached or sparsity_becomes_reached

                        if sparsity>=goal_sparsity:
                              test_sparsity_reached = True

                        if sparsity_becomes_reached:
                              print("----", flush = True)
                              print("Goal sparsity ("+str(goal_sparsity)+") reached at epoch", epoch, flush=True)
                              print("----", flush = True)
                              #model_wrapper.l2_reg = 0.0
                              #import ipdb;ipdb.set_trace()

                        if test_early_stopping==1:
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
                                                torch.save(model_wrapper_dense.model.state_dict(), path_save)
                                          best_model = copy.deepcopy(model_wrapper_dense.model)
                                          n_epochs_no_improvement = 0
                                    else:
                                          if test_sparsity_reached:
                                                n_epochs_no_improvement += 1
                        test_model_stuck = False
                        tol_epsilon = 1e-6
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
                              print(f'Epoch: {epoch:03d}, {print_loss_no_pen}: {value_loss_no_pen:.4f}, {print_loss_with_pen}: {value_loss_with_pen:.4f}, Validation loss: {val_loss:.4f}, {print_metric}: {value_in_sample_metric:.4f}, Validation MSE: {mse_val:.4f}, lr: {current_lr:.4f}, n_z: {n_z:.4f}, sparsity: {sparsity:4f}', flush=True)
                        elif type_of_task == "classification":
                              print(f'Epoch: {epoch:03d}, {print_loss_no_pen}: {value_loss_no_pen:.4f}, {print_loss_with_pen}: {value_loss_with_pen:.4f}, Validation loss: {val_loss:.4f}, {print_metric}: {value_in_sample_metric:.4f}, Val Acc: {100*acc_val:.4f}, lr: {current_lr:.4f}, n_z: {n_z:.4f}, sparsity: {sparsity:4f}', flush=True)
                        l_lr.append(current_lr)
                        l_in_sample_loss.append(value_loss_with_pen)
                        l_validation_loss.append(val_loss)
                        l_n_z.append(n_z)
                        if type_of_task == "classification":
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
                              scheduler_dense.step()
                              scheduler_sparse.step()
                        if trial!=None:
                              if trial.should_prune():
                                    raise optuna.exceptions.TrialPruned()
                  else:
                        print("Early stopping at epoch", epoch)
                        break

      l_in_sample_loss = np.array(l_in_sample_loss)
      l_validation_loss = np.array(l_validation_loss)
      l_in_sample_metric = np.array(l_in_sample_metric)
      l_validation_metric = np.array(l_validation_metric)
      l_times_epochs = np.array(l_times_epochs)
      l_lr = np.array(l_lr)
      l_n_z = np.array(l_n_z)

      in_sample_metric = evaluate_neural_network(best_model, loader_train, type_of_task, model_wrapper_dense.device, scaler_y=scaler_y)
      validation_metric = evaluate_neural_network(best_model, loader_val, type_of_task, model_wrapper_dense.device, scaler_y=scaler_y)
      test_metric = evaluate_neural_network(best_model, loader_test, type_of_task, model_wrapper_dense.device, scaler_y=scaler_y)
      
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