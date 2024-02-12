def prune_models_rec(self, module, name_module, optimizer, dense_to_sparse=False):
    test_pruned = False
    str_module = module.__str__().lower()
    d_children = dict(module.named_children())
    test_children = len(d_children)>0
    if test_children:
        for name_child in d_children:
            child = d_children[name_child]
            test_pruned_child = prune_models_rec(self, child, name_child, optimizer, dense_to_sparse=dense_to_sparse)
            test_pruned = (test_pruned or test_pruned_child)
    #elif not("relu" in str_module) and not("avgpool" in str_module) and not("norm" in str_module) and not("linear(" in str_module):
    elif "with_z" in str_module:
        try:
            if torch.min(module.z) == 0:
                module.weight_z.data[module.z==0] = -module.gamma
                module.weight.data[module.z==0] = 0
                test_pruned = True
            if module.test_bias:
                if torch.min(module.z_2) == 0:
                    module.bias_z.data[module.z_2==0] = -module.gamma
                    module.bias.data[module.z_2==0] = 0
                    test_pruned = True
            if dense_to_sparse:
                current_test_pruned_bias = False
                current_test_pruned = False
                if len(module.z)>0:
                    sum_rows_z = torch.sum(module.z, 1)
                    sum_columns_z = torch.sum(module.z, 0)
                    if min(torch.min(sum_rows_z).item(), torch.min(sum_columns_z).item()) == 0:
                        if "exp_avg" in optimizer.state[module.weight]:
                            test_exp_avg = True
                            old_state_exp_avg = copy.deepcopy(optimizer.state[module.weight]["exp_avg"])
                            old_state_exp_avg_z = copy.deepcopy(optimizer.state[module.weight_z]["exp_avg"])
                        else:
                            test_exp_avg = False
                        if "exp_avg_sq" in optimizer.state[module.weight]:
                            test_exp_avg_sq = True
                            old_state_exp_avg_sq = copy.deepcopy(optimizer.state[module.weight]["exp_avg_sq"])
                            old_state_exp_avg_sq_z = copy.deepcopy(optimizer.state[module.weight_z]["exp_avg_sq"])
                        else:
                            test_exp_avg_sq = False
                        if "momentum_buffer" in optimizer.state[module.weight]:
                            test_momentum_buffer = True
                            old_state_mom_buff = copy.deepcopy(optimizer.state[module.weight]["momentum_buffer"])
                            old_state_mom_buff_z = copy.deepcopy(optimizer.state[module.weight_z]["momentum_buffer"])
                        else:
                            test_momentum_buffer = False
                        current_test_pruned = True
                    if torch.min(sum_rows_z) == 0:
                        idx_keep_row = torch.where(sum_rows_z!=0)[0]
                        copy_grad = copy.deepcopy(module.weight.grad[idx_keep_row])
                        module.weight = Parameter(module.weight[idx_keep_row])
                        module.weight.grad = copy_grad
                        copy_grad = copy.deepcopy(module.weight_z.grad[idx_keep_row])
                        module.weight_z = Parameter(module.weight_z[idx_keep_row])
                        module.weight_z.grad = copy_grad
                        test_pruned = True
                        if test_exp_avg:
                            old_state_exp_avg = old_state_exp_avg[idx_keep_row]
                            old_state_exp_avg_z = old_state_exp_avg_z[idx_keep_row]
                        if test_exp_avg_sq:
                            old_state_exp_avg_sq = old_state_exp_avg_sq[idx_keep_row]
                            old_state_exp_avg_sq_z = old_state_exp_avg_sq_z[idx_keep_row]
                        if test_momentum_buffer:
                            old_state_mom_buff = old_state_mom_buff[idx_keep_row]
                            old_state_mom_buff_z = old_state_mom_buff_z[idx_keep_row]
                    if torch.min(sum_columns_z) == 0:
                        idx_keep_column = torch.where(sum_columns_z!=0)[0]
                        copy_grad = copy.deepcopy(module.weight.grad[:,idx_keep_column])
                        module.weight = Parameter(module.weight[:,idx_keep_column])
                        module.weight.grad = copy_grad
                        copy_grad = copy.deepcopy(module.weight_z.grad[:,idx_keep_column])
                        module.weight_z = Parameter(module.weight_z[:,idx_keep_column])
                        module.weight_z.grad = copy_grad
                        test_pruned = True  
                        if test_exp_avg:
                            old_state_exp_avg = old_state_exp_avg[:,idx_keep_column]
                            old_state_exp_avg_z = old_state_exp_avg_z[:,idx_keep_column]
                        if test_exp_avg_sq:
                            old_state_exp_avg_sq = old_state_exp_avg_sq[:,idx_keep_column]
                            old_state_exp_avg_sq_z = old_state_exp_avg_sq_z[:,idx_keep_column]
                        if test_momentum_buffer:
                            old_state_mom_buff = old_state_mom_buff[:,idx_keep_column]                      
                            old_state_mom_buff_z = old_state_mom_buff_z[:,idx_keep_column]
                        if name_module == "fc1":
                            self.model.idx_keep_input = self.model.idx_keep_input[idx_keep_column]
                if module.test_bias:
                    if len(module.z_2)>0:
                        sum_rows_z = torch.sum(module.z_2, 1)
                        sum_columns_z = torch.sum(module.z_2, 0)
                        if min(torch.min(sum_rows_z).item(), torch.min(sum_columns_z).item()) == 0:
                            if "exp_avg" in optimizer.state[module.bias]:
                                test_exp_avg_bias = True
                                old_state_exp_avg_bias = copy.deepcopy(optimizer.state[module.bias]["exp_avg"])
                                old_state_exp_avg_z_bias = copy.deepcopy(optimizer.state[module.bias_z]["exp_avg"])
                            else:
                                test_exp_avg_bias = False
                            if "exp_avg_sq" in optimizer.state[module.bias]:
                                test_exp_avg_sq_bias = True
                                old_state_exp_avg_sq_bias = copy.deepcopy(optimizer.state[module.bias]["exp_avg_sq"])
                                old_state_exp_avg_sq_z_bias = copy.deepcopy(optimizer.state[module.bias_z]["exp_avg_sq"])
                            else:
                                test_exp_avg_sq_bias = False
                            if "momentum_buffer" in optimizer.state[module.bias]:
                                test_momentum_buffer_bias = True
                                old_state_mom_buff_bias = copy.deepcopy(optimizer.state[module.bias]["momentum_buffer"])
                                old_state_mom_buff_z_bias = copy.deepcopy(optimizer.state[module.bias_z]["momentum_buffer"])
                            else:
                                test_momentum_buffer_bias = False
                            current_test_pruned_bias = True
                        if torch.min(sum_rows_z) == 0:
                            idx_keep_row_bias = torch.where(sum_rows_z!=0)[0]
                            copy_grad = copy.deepcopy(module.bias.grad[idx_keep_row_bias])
                            module.bias = Parameter(module.bias[idx_keep_row_bias])
                            module.bias.grad = copy_grad
                            copy_grad = copy.deepcopy(module.bias_z.grad[idx_keep_row_bias])
                            module.bias_z = Parameter(module.bias_z[idx_keep_row_bias])
                            module.bias_z.grad = copy_grad
                            test_pruned = True
                            if test_exp_avg_bias:
                                old_state_exp_avg_bias = old_state_exp_avg_bias[idx_keep_row_bias]
                                old_state_exp_avg_z_bias = old_state_exp_avg_z_bias[idx_keep_row_bias]
                            if test_exp_avg_sq_bias:
                                old_state_exp_avg_sq_bias = old_state_exp_avg_sq_bias[idx_keep_row_bias]
                                old_state_exp_avg_sq_z_bias = old_state_exp_avg_sq_z_bias[idx_keep_row_bias]
                            if test_momentum_buffer_bias:
                                old_state_mom_buff_bias = old_state_mom_buff_bias[idx_keep_row_bias]                      
                                old_state_mom_buff_z_bias = old_state_mom_buff_z_bias[idx_keep_row_bias]                      
                        if torch.min(sum_columns_z) == 0:
                            idx_keep_column_bias = torch.where(sum_rows_z!=0)[0]
                            copy_grad = copy.deepcopy(module.bias.grad[:,idx_keep_column_bias])
                            module.bias = Parameter(module.bias[:,idx_keep_column_bias])
                            module.bias.grad = copy_grad
                            copy_grad = copy.deepcopy(module.bias_z.grad[:,idx_keep_column_bias])
                            module.bias_z = Parameter(module.bias_z[:,idx_keep_column_bias])
                            module.bias_z.grad = copy_grad
                            test_pruned = True
                            if test_exp_avg_bias:
                                old_state_exp_avg_bias = old_state_exp_avg_bias[:,idx_keep_column_bias]
                                old_state_exp_avg_z_bias = old_state_exp_avg_z_bias[:,idx_keep_column_bias]
                            if test_exp_avg_sq_bias:
                                old_state_exp_avg_sq_bias = old_state_exp_avg_sq_bias[:,idx_keep_column_bias]
                                old_state_exp_avg_sq_z_bias = old_state_exp_avg_sq_z_bias[:,idx_keep_column_bias]
                            if test_momentum_buffer_bias:
                                old_state_mom_buff_bias = old_state_mom_buff_bias[:,idx_keep_column_bias]                      
                                old_state_mom_buff_z_bias = old_state_mom_buff_z_bias[:,idx_keep_column_bias]                      

                if current_test_pruned or current_test_pruned_bias:
                    optimizer_name = optimizer.__class__.__name__
                    if "momentum" in optimizer.defaults:
                        momentum = optimizer.defaults["momentum"]
                    else:
                        momentum = -1
                    copy_optimizer = initialize_optimizer(test_different_lr = self.test_different_lr, model=self.model, optimizer_name=optimizer_name, steps_per_epoch=self.steps_per_epoch, lr = optimizer.defaults["lr"], val_second_lr=self.val_second_lr, momentum=momentum, weight_decay=optimizer.defaults["weight_decay"])
                    try:
                        copy_optimizer._step_count = optimizer._step_count
                    except:
                        pass
                    if current_test_pruned:
                        if test_exp_avg:
                            copy_optimizer.state[module.weight]["exp_avg"] = old_state_exp_avg
                            copy_optimizer.state[module.weight_z]["exp_avg"] = old_state_exp_avg_z
                        if test_exp_avg_sq:
                            copy_optimizer.state[module.weight]["exp_avg_sq"] = old_state_exp_avg_sq
                            copy_optimizer.state[module.weight_z]["exp_avg_sq"] = old_state_exp_avg_sq_z
                        if test_momentum_buffer:
                            copy_optimizer.state[module.weight]["momentum_buffer"] = old_state_mom_buff
                            copy_optimizer.state[module.weight_z]["momentum_buffer"] = old_state_mom_buff_z
                    if current_test_pruned_bias:
                        if test_exp_avg_bias:
                            copy_optimizer.state[module.bias]["exp_avg"] = old_state_exp_avg_bias
                            copy_optimizer.state[module.bias_z]["exp_avg"] = old_state_exp_avg_z_bias
                        if test_exp_avg_sq:
                            copy_optimizer.state[module.bias]["exp_avg_sq"] = old_state_exp_avg_sq_bias
                            copy_optimizer.state[module.bias_z]["exp_avg_sq"] = old_state_exp_avg_sq_z_bias
                        if test_momentum_buffer:
                            copy_optimizer.state[module.bias]["momentum_buffer"] = old_state_mom_buff_bias
                            copy_optimizer.state[module.bias_z]["momentum_buffer"] = old_state_mom_buff_z_bias
        except:
            import ipdb;ipdb.set_trace()
    return optimizer, test_pruned

