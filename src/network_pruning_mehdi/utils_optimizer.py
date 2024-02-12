import torch

def initialize_optimizer(test_different_lr, model, optimizer_name, steps_per_epoch, lr, val_second_lr, momentum, weight_decay, selection_lagrangian_reg, entropy_lagrangian_reg):
      optimizer_func = getattr(torch.optim, optimizer_name)
      try:
            l_constraint = []
            if selection_lagrangian_reg!=None:
                  l_constraint.append(selection_lagrangian_reg)
            if entropy_lagrangian_reg!=None:
                  l_constraint.append(entropy_lagrangian_reg)

            if test_different_lr:
                  dict_params = dict(model.named_parameters())
                  l_params_regular_lr = []
                  l_params_modified_lr = []
                  for key in dict_params:
                        if "_z" in key:
                              l_params_modified_lr.append(dict_params[key])
                        else:
                              l_params_regular_lr.append(dict_params[key])

                  # if val_second_lr== -1:
                  #       val_second_lr = lr/steps_per_epoch
                  
                  if optimizer_name=="SGD":
                        if len(l_constraint)>0:
                              optimizer = optimizer_func([{"params":l_params_regular_lr}, {"params":l_params_modified_lr, "lr":val_second_lr}, {"params":l_constraint, "maximize":True, "lr":lr/1.0}], lr = lr, momentum=momentum, weight_decay=weight_decay)
                        else:
                              optimizer = optimizer_func([{"params":l_params_regular_lr}, {"params":l_params_modified_lr, "lr":val_second_lr}], lr = lr, momentum=momentum, weight_decay=weight_decay)
                  else:
                        if len(l_constraint)>0:
                              optimizer = optimizer_func([{"params":l_params_regular_lr}, {"params":l_params_modified_lr, "lr":val_second_lr}, {"params":l_constraint, "maximize":True, "lr":lr/1.0}], lr = lr, weight_decay=weight_decay)
                        else:
                              optimizer = optimizer_func([{"params":l_params_regular_lr}, {"params":l_params_modified_lr, "lr":val_second_lr}], lr = lr, weight_decay=weight_decay)
            else:
                  if optimizer_name=="SGD":
                        if len(l_constraint)>0:
                              optimizer = optimizer_func([{"params":model.parameters()}, {"params":l_constraint, "maximize":True, "lr":lr/1.0}], lr = lr, momentum=momentum, weight_decay=weight_decay)
                        else:
                              optimizer = optimizer_func(model.parameters(), lr = lr, momentum=momentum, weight_decay=weight_decay)
                  else:
                        if len(l_constraint)>0:
                              optimizer = optimizer_func([{"params":model.parameters()}, {"params":l_constraint, "maximize":True, "lr":lr/1.0}], lr = lr, weight_decay=weight_decay)
                        else:
                              optimizer = optimizer_func(model.parameters(), lr = lr, weight_decay=weight_decay)
      except:
            return None
      return optimizer

