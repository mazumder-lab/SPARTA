#%% Imports
import numpy as np
import argparse
import time

start_time = time.time()

parser = argparse.ArgumentParser(description='Simple soft trees training')

# Previous HYPERPARAMETERS
parser.add_argument('--arch', type=str, default='mlpnet')
parser.add_argument('--weight_decay',type=float,default=0.00003751757813)
parser.add_argument('--momentum',type=float,default=0.9)
parser.add_argument('--pretrained', type=lambda x: (str(x).lower() == 'true'), default=True)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--local_rank', default=-1, type=int, 
                        help='local rank for distributed training')
parser.add_argument('--test_distributed', default=0, type=int, 
                        help='1 if distributed processing is used for training, 0 otherwise')

# New HYPERPARAMETERS
parser.add_argument('--n_epochs', type=int, default = 300,
                    help='number of epochs for the training function')
parser.add_argument('--type_pruning', type=str, default = "magnitude",
                    help='magnitude pruning')
parser.add_argument('--timeout', type=int, default = 72*60*60,
                    help='timeout for the fine tuning')
parser.add_argument('--n_trials', type=int, default = 1,
                    help='number of trials for the fine tuning')
parser.add_argument('--n_repeat', type=int, default = 1,
                    help='number of times we repeat the experiment for a given set of hyperparameters')
parser.add_argument('--name_dataset', type=str, default = "mnist",
                    help='name of the dataset')
parser.add_argument('--lr', type=float, default = -1,
                    help='if a value is given, no fine tuning is performed for the learning rate')
parser.add_argument('--min_lr', type=float, default = 1e-5,
                    help='lower bound of the gridsearch given to Optuna for the learning rate. If lr != -1, then this parameter is ignored')
parser.add_argument('--max_lr', type=float, default = 1e-1,
                    help='upper bound of the gridsearch given to Optuna for the learning rate. If lr != -1, then this parameter is ignored')
parser.add_argument('--folder_saves', type=str, default = "Saves_net_pruning",
                    help='name of the folder where all the studies are being saved')
# parser.add_argument('--type_scaler', type=str, default = "std",
#                     help='type of scaling method (std or max)')
parser.add_argument('--type_of_task', type=str, default = "classification",
                    help='either classification or regression')
parser.add_argument('--device', type=str, default = "None",
                    help='if None, then the best possible device will be used')
parser.add_argument('--metric_best_model', type=str, default = "auc",
                    help='either auc or acc')
parser.add_argument('--seed', type=int, default = 0,
                    help='seed for the training')
parser.add_argument('--batch_size_dataset', type=int, default = 500,
                    help='batch size for the dataset')
parser.add_argument('--test_save_all_models', type=int, default = 0,
                    help='If test_save_all_models=1, we save all the models during the training (one model per epoch)')
parser.add_argument('--val_second_lr', type=float, default = -1,
                    help='if a value is given and test_diff_lr = 1, this learning rate is used for the z_i, otherwise lr/n_steps_per_epoch is used when test_diff_lr = 1')
parser.add_argument('--test_early_stopping', type=int, default = 1,
                    help='if test_early_stopping==1=1, the best model out of the n_epochs iterations is kept based on the validation loss. if test_early_stopping==1=1, the training loss is used')
parser.add_argument('--type_decay', type=str, default = "cosine",
                    help='criteria for the decay. If type_decay = "None", then no decay is applied. The other types of decays are "linear", "exponential" and "cosine"')
parser.add_argument('--gamma_lr_decay', type=float, default = 0.9,
                    help='learning rate decay for type_decay = "exponential"')
parser.add_argument('--T_max_cos', type=int, default = 300,
                    help='half-period of the cosine for type_decay = "cosine"')
parser.add_argument('--eta_min_cos', type=float, default = 1e-5,
                    help='minimum learning rate for type_decay = "cosine"')
parser.add_argument('--start_lr_decay', type=float, default = 1.0,
                    help='starting learning rate for type_decay = "linear"')
parser.add_argument('--end_lr_decay', type=float, default = 1.0,
                    help='ending learning rate for type_decay = "linear"')
parser.add_argument('--warmup_steps', type=int, default = 100,
                    help='epoch where the learning rate reaches its maximum for type_decay = "ramp"')
parser.add_argument('--path_load_weights', type=str, default = "",
                    help='if a path is provided, it is used to initialize the weights of the model')
parser.add_argument('--test_compute_accurate_in_sample_loss', type=int, default = 0,
                    help='recomputes the in-sample loss exactly')
parser.add_argument('--patience', type=int, default = 50,
                    help='patience for early stopping')
parser.add_argument('--gamma', type=float, default = 1.0,
                    help='gamma for SmoothStep')
parser.add_argument('--entropy_reg', type=float, default = 1e-1,
                    help='regularizer factor for entropy penalization in additive_soft_trees')
parser.add_argument('--selection_reg', type=float, default = 1e-2,
                    help='regularizer factor for selection penalization in additive_soft_trees')
parser.add_argument('--metric_early_stopping', type=str, default = "val_loss",
                    help='either val_loss or val_accuracy')
parser.add_argument('--l2_reg', type=float, default = 1e-3,
                    help='regularizer factor for l2 penalization in additive_soft_trees')
parser.add_argument('--period_milestones', type=int, default = 25,
                    help='period of the milestones for multi_lr scheduler')
parser.add_argument('--test_different_lr', type=int, default = 0,
                    help='if set to 1, then lr/steps_per_epoch is used for the weights corresponding to the z_i and z_ij. Otherwise, the regular learning rate is used')
parser.add_argument('--dense_to_sparse', type=int, default = 0,
                    help='if set to 1, then the weights of the model are eliminated during the training. Currently, only works with Adam, has to be set to 0 for another optimizer.')
parser.add_argument('--optimizer_name', type=str, default = "Adam",
                    help='Adam, SGD')
parser.add_argument('--goal_sparsity', type=float, default = 0.0,
                    help='the training will continue as long as goal sparsity is not reached')
parser.add_argument('--tol_z_1', type=float, default = 1.0,
                    help='tolerance to consider that a z is equal to 1 (for the restart)')
parser.add_argument('--type_training', type=str, default = "combined",
                    help="if type_training='combined', gradient descent is performed on both the weights and the zs, if type_training='alternate', we alternate between training of the weights and training of the zs")
parser.add_argument('--n_restart', type=int, default = 0,
                    help='number of times we restart the weight_z during training if type_training = "alternate"')
# parser.add_argument('--layer_wise_reg', type=float, default = 0.0,
#                     help='penalization for the layer wise reconstruction')

arguments, unknown = parser.parse_known_args()

def diff_model_wrapper(model_wrapper_dense, model_wrapper_sparse, optimizer_dense, optimizer_sparse):
    fc1_sparse_weight = model_wrapper_sparse.model.fc1.weight
    fc1_sparse_weight_z = model_wrapper_sparse.model.fc1.weight_z
    fc2_sparse_weight = model_wrapper_sparse.model.fc2.weight
    fc2_sparse_weight_z = model_wrapper_sparse.model.fc2.weight_z
    fc3_sparse_weight = model_wrapper_sparse.model.fc3.weight
    fc3_sparse_weight_z = model_wrapper_sparse.model.fc3.weight_z
    
    fc1_dense_weight = model_wrapper_dense.model.fc1.weight
    fc1_dense_weight_z = model_wrapper_dense.model.fc1.weight_z
    fc2_dense_weight = model_wrapper_dense.model.fc2.weight
    fc2_dense_weight_z = model_wrapper_dense.model.fc2.weight_z
    fc3_dense_weight = model_wrapper_dense.model.fc3.weight
    fc3_dense_weight_z = model_wrapper_dense.model.fc3.weight_z

    try:
        fc1_dense_weight_momentum = optimizer_dense.state[fc1_dense_weight]["momentum_buffer"]
        fc1_dense_weight_z_momentum = optimizer_dense.state[fc1_dense_weight_z]["momentum_buffer"]
        fc2_dense_weight_momentum = optimizer_dense.state[fc2_dense_weight]["momentum_buffer"]
        fc2_dense_weight_z_momentum = optimizer_dense.state[fc2_dense_weight_z]["momentum_buffer"]
        fc3_dense_weight_momentum = optimizer_dense.state[fc3_dense_weight]["momentum_buffer"]
        fc3_dense_weight_z_momentum = optimizer_dense.state[fc3_dense_weight_z]["momentum_buffer"]

        fc1_sparse_weight_momentum = optimizer_sparse.state[fc1_sparse_weight]["momentum_buffer"]
        fc1_sparse_weight_z_momentum = optimizer_sparse.state[fc1_sparse_weight_z]["momentum_buffer"]
        fc2_sparse_weight_momentum = optimizer_sparse.state[fc2_sparse_weight]["momentum_buffer"]
        fc2_sparse_weight_z_momentum = optimizer_sparse.state[fc2_sparse_weight_z]["momentum_buffer"]
        fc3_sparse_weight_momentum = optimizer_sparse.state[fc3_sparse_weight]["momentum_buffer"]
        fc3_sparse_weight_z_momentum = optimizer_sparse.state[fc3_sparse_weight_z]["momentum_buffer"]
    except:
        pass

    fc1_dense_z = model_wrapper_dense.model.fc1.z
    fc2_dense_z = model_wrapper_dense.model.fc2.z
    fc3_dense_z = model_wrapper_dense.model.fc3.z

    to_keep_in_1 = torch.where(torch.sum(fc1_dense_z, 0)!=0)[0]
    to_keep_out_1 = torch.where(torch.sum(fc1_dense_z, 1)!=0)[0]
    to_keep_in_2 = torch.where(torch.sum(fc2_dense_z, 0)!=0)[0]
    to_keep_out_2 = torch.where(torch.sum(fc2_dense_z, 1)!=0)[0]
    to_keep_in_3 = torch.where(torch.sum(fc3_dense_z, 0)!=0)[0]
    to_keep_out_3 = torch.where(torch.sum(fc3_dense_z, 1)!=0)[0]

    to_keep_out_1 = torch.Tensor(np.intersect1d(to_keep_out_1, to_keep_in_2)).long()
    to_keep_in_2 = torch.Tensor(np.intersect1d(to_keep_out_1, to_keep_in_2)).long()

    to_keep_out_2 = torch.Tensor(np.intersect1d(to_keep_out_2, to_keep_in_3)).long()
    to_keep_in_3 = torch.Tensor(np.intersect1d(to_keep_out_2, to_keep_in_3)).long()

    fc1_dense_weight = fc1_dense_weight[to_keep_out_1]
    fc1_dense_weight = fc1_dense_weight[:,to_keep_in_1]
    fc1_dense_weight_z = fc1_dense_weight_z[to_keep_out_1]
    fc1_dense_weight_z = fc1_dense_weight_z[:,to_keep_in_1]
    try:
        fc1_dense_weight_momentum = fc1_dense_weight_momentum[to_keep_out_1]
        fc1_dense_weight_momentum = fc1_dense_weight_momentum[:,to_keep_in_1]
        fc1_dense_weight_z_momentum = fc1_dense_weight_z_momentum[to_keep_out_1]
        fc1_dense_weight_z_momentum = fc1_dense_weight_z_momentum[:,to_keep_in_1]
    except:
        pass 

    fc2_dense_weight = fc2_dense_weight[to_keep_out_2]
    fc2_dense_weight = fc2_dense_weight[:,to_keep_in_2]
    fc2_dense_weight_z = fc2_dense_weight_z[to_keep_out_2]
    fc2_dense_weight_z = fc2_dense_weight_z[:,to_keep_in_2]
    try:
        fc2_dense_weight_momentum = fc2_dense_weight_momentum[to_keep_out_2]
        fc2_dense_weight_momentum = fc2_dense_weight_momentum[:,to_keep_in_2]
        fc2_dense_weight_z_momentum = fc2_dense_weight_z_momentum[to_keep_out_2]
        fc2_dense_weight_z_momentum = fc2_dense_weight_z_momentum[:,to_keep_in_2]
    except:
        pass

    fc3_dense_weight = fc3_dense_weight[to_keep_out_3]
    fc3_dense_weight = fc3_dense_weight[:,to_keep_in_3]
    fc3_dense_weight_z = fc3_dense_weight_z[to_keep_out_3]
    fc3_dense_weight_z = fc3_dense_weight_z[:,to_keep_in_3]
    try:
        fc3_dense_weight_momentum = fc3_dense_weight_momentum[to_keep_out_3]
        fc3_dense_weight_momentum = fc3_dense_weight_momentum[:,to_keep_in_3]
        fc3_dense_weight_z_momentum = fc3_dense_weight_z_momentum[to_keep_out_3]
        fc3_dense_weight_z_momentum = fc3_dense_weight_z_momentum[:,to_keep_in_3]
    except:
        pass
    
    try:
        test_1 = torch.sum(fc1_dense_weight!=fc1_sparse_weight)==0
        max_1 = torch.max(torch.abs(fc1_dense_weight-fc1_sparse_weight)).item()
        test_2 = torch.sum(fc1_dense_weight_z!=fc1_sparse_weight_z)==0
        max_2 = torch.max(torch.abs(fc1_dense_weight_z-fc1_sparse_weight_z)).item()
        test_3 = torch.sum(fc2_dense_weight!=fc2_sparse_weight)==0
        max_3 = torch.max(torch.abs(fc2_dense_weight-fc2_sparse_weight)).item()
        test_4 = torch.sum(fc2_dense_weight_z!=fc2_sparse_weight_z)==0
        max_4 = torch.max(torch.abs(fc2_dense_weight_z-fc2_sparse_weight_z)).item()
        test_5 = torch.sum(fc3_dense_weight!=fc3_sparse_weight)==0
        max_5 = torch.max(torch.abs(fc3_dense_weight-fc3_sparse_weight)).item()
        test_6 = torch.sum(fc3_dense_weight_z!=fc3_sparse_weight_z)==0
        max_6 = torch.max(torch.abs(fc3_dense_weight_z-fc3_sparse_weight_z)).item()

        try:
            max_7 = torch.max(torch.abs(fc1_dense_weight_momentum-fc1_sparse_weight_momentum)).item()
            max_8 = torch.max(torch.abs(fc1_dense_weight_z_momentum-fc1_sparse_weight_z_momentum)).item()
            max_9 = torch.max(torch.abs(fc2_dense_weight_momentum-fc2_sparse_weight_momentum)).item()
            max_10 = torch.max(torch.abs(fc2_dense_weight_z_momentum-fc2_sparse_weight_z_momentum)).item()
            max_11 = torch.max(torch.abs(fc3_dense_weight_momentum-fc3_sparse_weight_momentum)).item()
            max_12 = torch.max(torch.abs(fc3_dense_weight_z_momentum-fc3_sparse_weight_z_momentum)).item()
        except:
            max_7 = -1
            max_8 = -1
            max_9 = -1
            max_10 = -1
            max_11 = -1
            max_12 = -1

        max_fc1 = torch.max(torch.abs(compute_z_from_tensor(fc1_dense_weight_z, 1.0)*fc1_dense_weight-compute_z_from_tensor(fc1_sparse_weight_z, 1.0)*fc1_sparse_weight)).item()
        max_fc2 = torch.max(torch.abs(compute_z_from_tensor(fc2_dense_weight_z, 1.0)*fc2_dense_weight-compute_z_from_tensor(fc2_sparse_weight_z, 1.0)*fc2_sparse_weight)).item()
        max_fc3 = torch.max(torch.abs(compute_z_from_tensor(fc3_dense_weight_z, 1.0)*fc3_dense_weight-compute_z_from_tensor(fc3_sparse_weight_z, 1.0)*fc3_sparse_weight)).item()
        max_fc = np.max([max_fc1, max_fc2, max_fc3])
        max_momentum = np.max([max_7, max_8, max_9, max_10, max_11, max_12])
    except:
        import ipdb;ipdb.set_trace()
    test_complet = test_1 and test_2 and test_3 and test_4 and test_5 and test_6
    max_total = np.max([max_1, max_2, max_3, max_4, max_5, max_6])
    if not(test_complet):
        print("--- NOT MATCHING ---")
        print([max_1, max_2, max_3, max_4, max_5, max_6])
        print([max_fc1, max_fc2, max_fc3])
        print([max_7, max_8, max_9, max_10, max_11, max_12])
        print(" ------- ")
    # if max_total>=1e-3:
    #     import ipdb;ipdb.set_trace()
    return test_complet, max_total, max_fc, max_momentum

def make_match_wrapper(model_wrapper_dense, model_wrapper_sparse):
    fc1_sparse_weight = model_wrapper_sparse.model.fc1.weight
    fc1_sparse_weight_z = model_wrapper_sparse.model.fc1.weight_z
    fc2_sparse_weight = model_wrapper_sparse.model.fc2.weight
    fc2_sparse_weight_z = model_wrapper_sparse.model.fc2.weight_z
    fc3_sparse_weight = model_wrapper_sparse.model.fc3.weight
    fc3_sparse_weight_z = model_wrapper_sparse.model.fc3.weight_z
    
    fc1_dense_weight = model_wrapper_dense.model.fc1.weight
    fc1_dense_weight_z = model_wrapper_dense.model.fc1.weight_z
    fc2_dense_weight = model_wrapper_dense.model.fc2.weight
    fc2_dense_weight_z = model_wrapper_dense.model.fc2.weight_z
    fc3_dense_weight = model_wrapper_dense.model.fc3.weight
    fc3_dense_weight_z = model_wrapper_dense.model.fc3.weight_z

    try:
        fc1_dense_weight_momentum = optimizer_dense.state[fc1_dense_weight]["momentum_buffer"]
        fc1_dense_weight_z_momentum = optimizer_dense.state[fc1_dense_weight_z]["momentum_buffer"]
        fc2_dense_weight_momentum = optimizer_dense.state[fc2_dense_weight]["momentum_buffer"]
        fc2_dense_weight_z_momentum = optimizer_dense.state[fc2_dense_weight_z]["momentum_buffer"]
        fc3_dense_weight_momentum = optimizer_dense.state[fc3_dense_weight]["momentum_buffer"]
        fc3_dense_weight_z_momentum = optimizer_dense.state[fc3_dense_weight_z]["momentum_buffer"]

        fc1_sparse_weight_momentum = optimizer_sparse.state[fc1_sparse_weight]["momentum_buffer"]
        fc1_sparse_weight_z_momentum = optimizer_sparse.state[fc1_sparse_weight_z]["momentum_buffer"]
        fc2_sparse_weight_momentum = optimizer_sparse.state[fc2_sparse_weight]["momentum_buffer"]
        fc2_sparse_weight_z_momentum = optimizer_sparse.state[fc2_sparse_weight_z]["momentum_buffer"]
        fc3_sparse_weight_momentum = optimizer_sparse.state[fc3_sparse_weight]["momentum_buffer"]
        fc3_sparse_weight_z_momentum = optimizer_sparse.state[fc3_sparse_weight_z]["momentum_buffer"]
    except:
        pass

    try:
        fc1_dense_weight_exp_avg = optimizer_dense.state[fc1_dense_weight]["exp_avg"]
        fc1_dense_weight_z_exp_avg = optimizer_dense.state[fc1_dense_weight_z]["exp_avg"]
        fc2_dense_weight_exp_avg = optimizer_dense.state[fc2_dense_weight]["exp_avg"]
        fc2_dense_weight_z_exp_avg = optimizer_dense.state[fc2_dense_weight_z]["exp_avg"]
        fc3_dense_weight_exp_avg = optimizer_dense.state[fc3_dense_weight]["exp_avg"]
        fc3_dense_weight_z_exp_avg = optimizer_dense.state[fc3_dense_weight_z]["exp_avg"]

        fc1_sparse_weight_exp_avg = optimizer_sparse.state[fc1_sparse_weight]["exp_avg"]
        fc1_sparse_weight_z_exp_avg = optimizer_sparse.state[fc1_sparse_weight_z]["exp_avg"]
        fc2_sparse_weight_exp_avg = optimizer_sparse.state[fc2_sparse_weight]["exp_avg"]
        fc2_sparse_weight_z_exp_avg = optimizer_sparse.state[fc2_sparse_weight_z]["exp_avg"]
        fc3_sparse_weight_exp_avg = optimizer_sparse.state[fc3_sparse_weight]["exp_avg"]
        fc3_sparse_weight_z_exp_avg = optimizer_sparse.state[fc3_sparse_weight_z]["exp_avg"]

        fc1_dense_weight_exp_avg_sq = optimizer_dense.state[fc1_dense_weight]["exp_avg_sq"]
        fc1_dense_weight_z_exp_avg_sq = optimizer_dense.state[fc1_dense_weight_z]["exp_avg_sq"]
        fc2_dense_weight_exp_avg_sq = optimizer_dense.state[fc2_dense_weight]["exp_avg_sq"]
        fc2_dense_weight_z_exp_avg_sq = optimizer_dense.state[fc2_dense_weight_z]["exp_avg_sq"]
        fc3_dense_weight_exp_avg_sq = optimizer_dense.state[fc3_dense_weight]["exp_avg_sq"]
        fc3_dense_weight_z_exp_avg_sq = optimizer_dense.state[fc3_dense_weight_z]["exp_avg_sq"]

        fc1_sparse_weight_exp_avg_sq = optimizer_sparse.state[fc1_sparse_weight]["exp_avg_sq"]
        fc1_sparse_weight_z_exp_avg_sq = optimizer_sparse.state[fc1_sparse_weight_z]["exp_avg_sq"]
        fc2_sparse_weight_exp_avg_sq = optimizer_sparse.state[fc2_sparse_weight]["exp_avg_sq"]
        fc2_sparse_weight_z_exp_avg_sq = optimizer_sparse.state[fc2_sparse_weight_z]["exp_avg_sq"]
        fc3_sparse_weight_exp_avg_sq = optimizer_sparse.state[fc3_sparse_weight]["exp_avg_sq"]
        fc3_sparse_weight_z_exp_avg_sq = optimizer_sparse.state[fc3_sparse_weight_z]["exp_avg_sq"]
    except:
        import ipdb;ipdb.set_trace()

    fc1_dense_z = model_wrapper_dense.model.fc1.z
    fc2_dense_z = model_wrapper_dense.model.fc2.z
    fc3_dense_z = model_wrapper_dense.model.fc3.z

    to_keep_in_1 = torch.where(torch.sum(fc1_dense_z, 0)!=0)[0]
    to_keep_out_1 = torch.where(torch.sum(fc1_dense_z, 1)!=0)[0]
    to_keep_in_2 = torch.where(torch.sum(fc2_dense_z, 0)!=0)[0]
    to_keep_out_2 = torch.where(torch.sum(fc2_dense_z, 1)!=0)[0]
    to_keep_in_3 = torch.where(torch.sum(fc3_dense_z, 0)!=0)[0]
    to_keep_out_3 = torch.where(torch.sum(fc3_dense_z, 1)!=0)[0]

    to_keep_out_1 = torch.Tensor(np.intersect1d(to_keep_out_1, to_keep_in_2)).long()
    to_keep_in_2 = torch.Tensor(np.intersect1d(to_keep_out_1, to_keep_in_2)).long()

    to_keep_out_2 = torch.Tensor(np.intersect1d(to_keep_out_2, to_keep_in_3)).long()
    to_keep_in_3 = torch.Tensor(np.intersect1d(to_keep_out_2, to_keep_in_3)).long()

    fc1_dense_weight = fc1_dense_weight[to_keep_out_1]
    fc1_dense_weight = fc1_dense_weight[:,to_keep_in_1]
    fc1_dense_weight_z = fc1_dense_weight_z[to_keep_out_1]
    fc1_dense_weight_z = fc1_dense_weight_z[:,to_keep_in_1]

    try:
        fc1_dense_weight_momentum = fc1_dense_weight_momentum[to_keep_out_1]
        fc1_dense_weight_momentum = fc1_dense_weight_momentum[:,to_keep_in_1]
        fc1_dense_weight_z_momentum = fc1_dense_weight_z_momentum[to_keep_out_1]
        fc1_dense_weight_z_momentum = fc1_dense_weight_z_momentum[:,to_keep_in_1]
    except:
        pass 

    try:
        fc1_dense_weight_exp_avg = fc1_dense_weight_exp_avg[to_keep_out_1]
        fc1_dense_weight_exp_avg = fc1_dense_weight_exp_avg[:,to_keep_in_1]
        fc1_dense_weight_z_exp_avg = fc1_dense_weight_z_exp_avg[to_keep_out_1]
        fc1_dense_weight_z_exp_avg = fc1_dense_weight_z_exp_avg[:,to_keep_in_1]

        fc1_dense_weight_exp_avg_sq = fc1_dense_weight_exp_avg_sq[to_keep_out_1]
        fc1_dense_weight_exp_avg_sq = fc1_dense_weight_exp_avg_sq[:,to_keep_in_1]
        fc1_dense_weight_z_exp_avg_sq = fc1_dense_weight_z_exp_avg_sq[to_keep_out_1]
        fc1_dense_weight_z_exp_avg_sq = fc1_dense_weight_z_exp_avg_sq[:,to_keep_in_1]
    except:
        import ipdb;ipdb.set_trace() 

    fc2_dense_weight = fc2_dense_weight[to_keep_out_2]
    fc2_dense_weight = fc2_dense_weight[:,to_keep_in_2]
    fc2_dense_weight_z = fc2_dense_weight_z[to_keep_out_2]
    fc2_dense_weight_z = fc2_dense_weight_z[:,to_keep_in_2]

    try:
        fc2_dense_weight_momentum = fc2_dense_weight_momentum[to_keep_out_2]
        fc2_dense_weight_momentum = fc2_dense_weight_momentum[:,to_keep_in_2]
        fc2_dense_weight_z_momentum = fc2_dense_weight_z_momentum[to_keep_out_2]
        fc2_dense_weight_z_momentum = fc2_dense_weight_z_momentum[:,to_keep_in_2]
    except:
        pass

    try:
        fc2_dense_weight_exp_avg = fc2_dense_weight_exp_avg[to_keep_out_2]
        fc2_dense_weight_exp_avg = fc2_dense_weight_exp_avg[:,to_keep_in_2]
        fc2_dense_weight_z_exp_avg = fc2_dense_weight_z_exp_avg[to_keep_out_2]
        fc2_dense_weight_z_exp_avg = fc2_dense_weight_z_exp_avg[:,to_keep_in_2]

        fc2_dense_weight_exp_avg_sq = fc2_dense_weight_exp_avg_sq[to_keep_out_2]
        fc2_dense_weight_exp_avg_sq = fc2_dense_weight_exp_avg_sq[:,to_keep_in_2]
        fc2_dense_weight_z_exp_avg_sq = fc2_dense_weight_z_exp_avg_sq[to_keep_out_2]
        fc2_dense_weight_z_exp_avg_sq = fc2_dense_weight_z_exp_avg_sq[:,to_keep_in_2]
    except:
        import ipdb;ipdb.set_trace()

    fc3_dense_weight = fc3_dense_weight[to_keep_out_3]
    fc3_dense_weight = fc3_dense_weight[:,to_keep_in_3]
    fc3_dense_weight_z = fc3_dense_weight_z[to_keep_out_3]
    fc3_dense_weight_z = fc3_dense_weight_z[:,to_keep_in_3]

    try:
        fc3_dense_weight_momentum = fc3_dense_weight_momentum[to_keep_out_3]
        fc3_dense_weight_momentum = fc3_dense_weight_momentum[:,to_keep_in_3]
        fc3_dense_weight_z_momentum = fc3_dense_weight_z_momentum[to_keep_out_3]
        fc3_dense_weight_z_momentum = fc3_dense_weight_z_momentum[:,to_keep_in_3]
    except:
        pass

    try:
        fc3_dense_weight_exp_avg = fc3_dense_weight_exp_avg[to_keep_out_3]
        fc3_dense_weight_exp_avg = fc3_dense_weight_exp_avg[:,to_keep_in_3]
        fc3_dense_weight_z_exp_avg = fc3_dense_weight_z_exp_avg[to_keep_out_3]
        fc3_dense_weight_z_exp_avg = fc3_dense_weight_z_exp_avg[:,to_keep_in_3]

        fc3_dense_weight_exp_avg_sq = fc3_dense_weight_exp_avg_sq[to_keep_out_3]
        fc3_dense_weight_exp_avg_sq = fc3_dense_weight_exp_avg_sq[:,to_keep_in_3]
        fc3_dense_weight_z_exp_avg_sq = fc3_dense_weight_z_exp_avg_sq[to_keep_out_3]
        fc3_dense_weight_z_exp_avg_sq = fc3_dense_weight_z_exp_avg_sq[:,to_keep_in_3]
    except:
        import ipdb;ipdb.set_trace()

    fc1_sparse_weight.data = copy.deepcopy(fc1_dense_weight.data)
    fc1_sparse_weight_z.data = copy.deepcopy(fc1_dense_weight_z.data)
    fc2_sparse_weight.data = copy.deepcopy(fc2_dense_weight.data)
    fc2_sparse_weight_z.data = copy.deepcopy(fc2_dense_weight_z.data)
    fc3_sparse_weight.data = copy.deepcopy(fc3_dense_weight.data)
    fc3_sparse_weight_z.data = copy.deepcopy(fc3_dense_weight_z.data)

    try:
        optimizer_sparse.state[fc1_sparse_weight]["momentum_buffer"] = copy.deepcopy(fc1_dense_weight_momentum)
        optimizer_sparse.state[fc1_sparse_weight_z]["momentum_buffer"] = copy.deepcopy(fc1_dense_weight_z_momentum)
        optimizer_sparse.state[fc2_sparse_weight]["momentum_buffer"] = copy.deepcopy(fc2_dense_weight_momentum)
        optimizer_sparse.state[fc2_sparse_weight_z]["momentum_buffer"] = copy.deepcopy(fc2_dense_weight_z_momentum)
        optimizer_sparse.state[fc3_sparse_weight]["momentum_buffer"] = copy.deepcopy(fc3_dense_weight_momentum)
        optimizer_sparse.state[fc3_sparse_weight_z]["momentum_buffer"] = copy.deepcopy(fc3_dense_weight_z_momentum)
        print("successfully assgined momentum")
    except:
        pass

    try:
        optimizer_sparse.state[fc1_sparse_weight]["exp_avg"] = copy.deepcopy(fc1_dense_weight_exp_avg)
        optimizer_sparse.state[fc1_sparse_weight_z]["exp_avg"] = copy.deepcopy(fc1_dense_weight_z_exp_avg)
        optimizer_sparse.state[fc2_sparse_weight]["exp_avg"] = copy.deepcopy(fc2_dense_weight_exp_avg)
        optimizer_sparse.state[fc2_sparse_weight_z]["exp_avg"] = copy.deepcopy(fc2_dense_weight_z_exp_avg)
        optimizer_sparse.state[fc3_sparse_weight]["exp_avg"] = copy.deepcopy(fc3_dense_weight_exp_avg)
        optimizer_sparse.state[fc3_sparse_weight_z]["exp_avg"] = copy.deepcopy(fc3_dense_weight_z_exp_avg)

        optimizer_sparse.state[fc1_sparse_weight]["exp_avg_sq"] = copy.deepcopy(fc1_dense_weight_exp_avg_sq)
        optimizer_sparse.state[fc1_sparse_weight_z]["exp_avg_sq"] = copy.deepcopy(fc1_dense_weight_z_exp_avg_sq)
        optimizer_sparse.state[fc2_sparse_weight]["exp_avg_sq"] = copy.deepcopy(fc2_dense_weight_exp_avg_sq)
        optimizer_sparse.state[fc2_sparse_weight_z]["exp_avg_sq"] = copy.deepcopy(fc2_dense_weight_z_exp_avg_sq)
        optimizer_sparse.state[fc3_sparse_weight]["exp_avg_sq"] = copy.deepcopy(fc3_dense_weight_exp_avg_sq)
        optimizer_sparse.state[fc3_sparse_weight_z]["exp_avg_sq"] = copy.deepcopy(fc3_dense_weight_z_exp_avg_sq)
        print("successfully assgined exp_avg, exp_avg_sq")
    except:
        import ipdb;ipdb.set_trace()

    return None

#%%
if __name__ == '__main__':
    from previous_utils.main_utils import *
    from utils_training import *
    from utils_experiments import *
    from utils_model import *
    import signal
    from pytorch_dataset_2_0 import random_split

    signal.signal(signal.SIGVTALRM, lambda signum, frame: print("\n--- Time is over ---"))

    print('Parsed arguments:', arguments)
    # Previous hyperparameters
    arch = arguments.arch
    weight_decay = arguments.weight_decay
    momentum = arguments.momentum
    pretrained = arguments.pretrained
    # New hyperparameters
    n_epochs = arguments.n_epochs
    timeout = arguments.timeout
    n_trials = arguments.n_trials
    name_dataset = arguments.name_dataset
    learning_rate = arguments.lr
    batch_size_dataset = arguments.batch_size_dataset
    test_early_stopping = arguments.test_early_stopping
    test_save_all_models = arguments.test_save_all_models
    optimizer_name = arguments.optimizer_name
    min_lr = arguments.min_lr
    max_lr = arguments.max_lr
    type_decay = arguments.type_decay
    gamma_lr_decay = arguments.gamma_lr_decay
    T_max_cos = arguments.T_max_cos
    eta_min_cos = arguments.eta_min_cos
    start_lr_decay = arguments.start_lr_decay
    end_lr_decay = arguments.end_lr_decay
    path_load_weights = arguments.path_load_weights
    type_of_task = arguments.type_of_task
    test_compute_accurate_in_sample_loss = arguments.test_compute_accurate_in_sample_loss
    n_repeat = arguments.n_repeat
    folder_saves = arguments.folder_saves
    warmup_steps = arguments.warmup_steps
    # type_scaler = arguments.type_scaler
    patience = arguments.patience
    gamma = arguments.gamma
    entropy_reg = arguments.entropy_reg
    selection_reg = arguments.selection_reg
    l2_reg = arguments.l2_reg
    metric_early_stopping = arguments.metric_early_stopping
    device = arguments.device
    period_milestones = arguments.period_milestones
    metric_best_model = arguments.metric_best_model
    test_different_lr = arguments.test_different_lr
    dense_to_sparse = arguments.dense_to_sparse
    seed = arguments.seed
    val_second_lr = arguments.val_second_lr
    goal_sparsity = arguments.goal_sparsity
    n_restart = arguments.n_restart
    tol_z_1 = arguments.tol_z_1
    type_training = arguments.type_training
    num_workers = arguments.num_workers
    type_pruning = arguments.type_pruning
    test_distributed = arguments.test_distributed
    local_rank = arguments.local_rank

    ##Change this to path of imagenet name_dataset
    if 'IMAGENET_PATH' in os.environ:  
        IMAGENET_PATH = os.environ['IMAGENET_PATH']+"/raw"
    else:
        print('****Warning**** No IMAGENET_PATH variable')
        IMAGENET_PATH = ''
    CIFAR10_PATH = '../datasets'
    MNIST_PATH = '../datasets'

    name_dataset_paths = {'imagenet':IMAGENET_PATH,'cifar10':CIFAR10_PATH,
                    'mnist':MNIST_PATH}

    name_dataset_path = name_dataset_paths[name_dataset]

    print("Name dataset:", name_dataset)
    print("Path dataset:", name_dataset_path)
    n_params_original_z = 0
    train_val_dataset, test_dataset = get_dataset(arch, name_dataset_path)

    generator_split = torch.Generator().manual_seed(seed)
    train_dataset, validation_dataset = random_split(train_val_dataset, [0.8, 0.2], generator=generator_split)

    if device == "None":
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("--- Device =", device, "---")

    name_study = get_name_study(arch,
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
                   test_distributed)

    n_samples = len(train_dataset)

    if batch_size_dataset == -1:
        batch_size_dataset = n_samples
    
    steps_per_epoch = np.ceil(n_samples/batch_size_dataset)
    test_deja_train = False
    n_trials_done = 0
    save_study_done = None
    if folder_saves in os.listdir():
        if ("study_"+name_study) in os.listdir(folder_saves):
            if len(os.listdir(folder_saves+"/study_"+name_study))>=1:
                try:
                    with open(folder_saves+"/study_"+name_study+"/save_study.pkl", "rb") as f:
                        save_study_done = pickle.load(f)
                    n_trials_done = len(save_study_done.trials)
                except:
                    pass

    if n_trials_done>=n_trials:
        test_deja_train = True
        try:
            list_dir = os.listdir(folder_saves+"/study_"+name_study)
            trial_folders = np.array([x for x in list_dir if "trial" in x])
            path_model = folder_saves+"/study_"+name_study+"/"+trial_folders[0]
            new_path_model =  folder_saves+"/study_"+name_study+"/best_trial"
            os.rename(path_model, new_path_model)
        except:
            pass
    else:
        n_trials = n_trials - n_trials_done
        if n_trials_done>0:
            print("Continuing existing study...")
        else:
            print("New study...")
        print(str(n_trials)+" trials restants")
    if test_deja_train:
        print("Training already done for "+name_study)
    else:
        if path_load_weights!="":
            model_loaded = True
            model_state_dict = torch.load(path_load_weights, map_location=device)
        # %% Second case: the same beta for all the times
        lr = learning_rate
        print('Parsed arguments:', arguments)
        print("lr:", lr)
        if type_of_task== "classification":
            best_val_metric_best = -np.inf
        elif type_of_task=="regression":
            best_val_metric_best = np.inf
        best_val_metric_avg = 0
        for ind_repeat in range(n_repeat):
            signal.alarm(0)
            signal.alarm(24*60*60)
            print("Repeat", ind_repeat+1, "out of", n_repeat, flush=True)
            # Model initialization
            model_dense, criterion, modules_to_prune = get_model(arch, seed, pretrained=pretrained, with_z=True)
            model_sparse, criterion, modules_to_prune = get_model(arch, seed, pretrained=pretrained, with_z=True)
            print("Model intialized", flush = True)
            generator_loader = torch.Generator()
            if seed != -1:
                torch.random.manual_seed(seed)
                generator_loader = generator_loader.manual_seed(seed)

            generator_dense = torch.Generator()
            if seed != -1:
                torch.random.manual_seed(seed)
                generator_dense = generator_dense.manual_seed(seed)

            generator_sparse = torch.Generator()
            if seed != -1:
                torch.random.manual_seed(seed)
                generator_sparse = generator_sparse.manual_seed(seed)
            
            loader_train = DataLoader(train_dataset, batch_size=batch_size_dataset, shuffle=True, generator=generator_loader, num_workers=num_workers, pin_memory=True)
            loader_val = DataLoader(validation_dataset, batch_size=batch_size_dataset, num_workers=num_workers, pin_memory=True)
            loader_test = DataLoader(test_dataset, batch_size=batch_size_dataset, num_workers=num_workers, pin_memory=True)
            scaler_y = None
            #dataset = train_dataset, validation_dataset, test_dataset, scaler_y
            dataset = loader_train, loader_val, loader_test, scaler_y
            model_dense.eval()
            model_sparse.eval()
            print("Evaluating the dense model", flush = True)
            
            if n_params_original_z==0:
                n_params_original_z = np.sum([np.prod(x[1].shape) for x in  model_dense.named_parameters() if "_z" in x[0]])
            if path_load_weights!="":
                if model_loaded:
                    model_dense.load_state_dict(model_state_dict)
                print("The weights have been loaded")
            model_dense.to(device)
            model_sparse.to(device)
            optimizer_dense = initialize_optimizer(test_different_lr, model_dense, optimizer_name, steps_per_epoch, lr, val_second_lr, momentum, weight_decay)
            optimizer_sparse = initialize_optimizer(test_different_lr, model_sparse, optimizer_name, steps_per_epoch, lr, val_second_lr, momentum, weight_decay)
            if name_dataset == "mnist":
                input_channel = 784
            elif name_dataset == "cifar10" and arch == "resnet20":
                input_channel = 64
            # elif name_dataset == "imagenet" and arch == "mobilenetv1":

            model_wrapper_dense = model_wrapper(model_dense, optimizer_dense, seed, entropy_reg, selection_reg, l2_reg, device, 0, test_different_lr, steps_per_epoch, val_second_lr, momentum, weight_decay, n_params_original_z, tol_z_1, input_channel, type_pruning, generator_dense)
            model_wrapper_sparse = model_wrapper(model_sparse, optimizer_sparse, seed, entropy_reg, selection_reg, l2_reg, device, 1, test_different_lr, steps_per_epoch, val_second_lr, momentum, weight_decay, n_params_original_z, tol_z_1, input_channel, type_pruning, generator_sparse)
            model_wrapper_dense.initialize_pruning()
            model_wrapper_sparse.initialize_pruning()
            
            # Training
            time_before_training = time.time()
            #######
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
            
            sparsity_dense = 0
            sparsity_sparse = 0

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
                                    n_z_close_to_1_dense = model_wrapper_dense.get_n_z_close_to_1()
                                    n_z_close_to_1_sparse = model_wrapper_sparse.get_n_z_close_to_1()
                                    n_z_dense = model_wrapper_dense.get_n_z()
                                    n_z_sparse = model_wrapper_sparse.get_n_z()
                                    print("Ratio of dense z equal to 1:", n_z_close_to_1_dense/n_z_dense, flush=True)
                                    print("Ratio of sparse z equal to 1:", n_z_close_to_1_sparse/n_z_sparse, flush=True)
                                    if n_z_close_to_1_dense/n_z_dense >= 0.95:
                                        first_term = 1.05*(goal_sparsity-sparsity_dense+1e-4)
                                        prop_reset = get_prop_reset_mnist(sparsity_dense, first_term)
                                        
                                        test_complet, max_diff, max_diff_fc, max_diff_mom = diff_model_wrapper(model_wrapper_dense, model_wrapper_sparse, optimizer_dense, optimizer_sparse)
                                        if not(test_complet) and max_diff<=1e-5 and max_diff_fc <= 1e-5:
                                            make_match_wrapper(model_wrapper_dense, model_wrapper_sparse)
                                        
                                        # if model_wrapper_dense.step_temp == 672:
                                        #     import ipdb;ipdb.set_trace()
                                            
                                        n_reset_dense = model_wrapper_dense.reset_z(prop_reset=prop_reset)
                                        n_reset_sparse = model_wrapper_sparse.reset_z(prop_reset=prop_reset)
                                        
                                        test_complet, max_diff, max_diff_fc, max_diff_mom = diff_model_wrapper(model_wrapper_dense, model_wrapper_sparse, optimizer_dense, optimizer_sparse)
                                        if max_diff>1e-3:
                                            import ipdb;ipdb.set_trace()
                                        
                                        print("-----", flush=True)
                                        print(str(n_reset_dense)+" z-dense-weights have been reset", n_z_close_to_1_dense/n_z_dense, flush=True)
                                        print(str(n_reset_sparse)+" z-sparse-weights have been reset", n_z_close_to_1_sparse/n_z_sparse, flush=True)
                                        print("-----", flush=True)
                                loss_pred_in_sample_dense = 0
                                loss_pred_in_sample_sparse = 0
                                # if sparsity >=0.7:
                                #       #list(model_wrapper.model.children())[-1].weight.requires_grad = False
                                #       list(model_wrapper.model.children())[-1].weight_z.requires_grad = False
                                approx_loss_in_sample_with_pen_dense = 0
                                approx_loss_in_sample_with_pen_sparse = 0
                                if type_of_task=="classification":
                                    approx_acc_train_dense = 0
                                    approx_acc_train_sparse = 0
                                model_wrapper_dense.model.train()
                                model_wrapper_sparse.model.train()
                                current_lr = optimizer_dense.param_groups[0]["lr"]
                                current_lr_sparse = optimizer_sparse.param_groups[0]["lr"]
                                print("current lr dense:", current_lr)
                                print("current lr sparse:", current_lr_sparse)
                                n_seen = 0
                                acc_batch = 0
                                if model_wrapper_dense.type_pruning=="layer_wise":
                                    model_wrapper_dense.initialize_pruning()
                                    model_wrapper_sparse.initialize_pruning()
                                for batch_sgd in tqdm(loader_train):
                                    acc_batch += 1
                                    model_wrapper_dense.step_temp += 1
                                    model_wrapper_sparse.step_temp += 1
                                    # # End
                                    n_batch = batch_sgd[0].shape[0]
                                    n_seen += n_batch
                                    optimizer_dense.zero_grad()
                                    optimizer_sparse.zero_grad()
                                    output_dense = model_wrapper_dense.model(batch_sgd[0].to(model_wrapper_dense.device))
                                    output_sparse = model_wrapper_sparse.model(batch_sgd[0].to(model_wrapper_sparse.device))
                                    if torch.max(torch.abs(output_dense-output_sparse))>1e-3:
                                        print("--- DIFF ---", torch.max(torch.abs(output_dense-output_sparse)))
                                        # input_layer2 = model_wrapper_dense.model.fc1(batch_sgd[0].to(model_wrapper_dense.device).view(batch_sgd[0].shape[0], -1))
                                        # fc3_dense_z = model_wrapper_dense.model.fc3.z
                                        # l_to_keep = torch.where(torch.sum(fc3_dense_z, 0)!=0)[0]
                                        # l_to_remove = torch.where(torch.sum(fc3_dense_z, 0)==0)[0]
                                        # weight_dense_fc2 = model_wrapper_dense.model.fc2.weight[l_to_keep]
                                        # z_dense_fc2 = model_wrapper_dense.model.fc2.z[l_to_keep]
                                        # weight_sparse_fc2 = model_wrapper_sparse.model.fc2.weight
                                        # z_sparse_fc2 = model_wrapper_sparse.model.fc2.z
                                        # output_layer_dense_2 = (weight_dense_fc2*z_dense_fc2).matmul(input_layer2.T).T
                                        # output_layer_dense_2_copy = model_wrapper_dense.model.fc2(input_layer2)[:,l_to_keep]
                                        # output_layer_dense_2_copy_copy = F.linear(input_layer2, weight_dense_fc2*z_dense_fc2, model_wrapper_dense.model.fc2.bias_final)
                                        # output_layer_sparse_2 = (weight_sparse_fc2*z_sparse_fc2).matmul(input_layer2.T).T
                                        # output_layer_sparse_2_copy = model_wrapper_sparse.model.fc2(input_layer2)
                                        # output_layer_sparse_2_copy_copy = F.linear(input_layer2, weight_sparse_fc2*z_sparse_fc2, model_wrapper_sparse.model.fc2.bias_final)
                                        # import ipdb;ipdb.set_trace()
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
                                    approx_acc_train_dense += torch.sum(torch.argmax(torch.softmax(output_dense, 1), 1) == y_truth).detach().item()
                                    approx_acc_train_sparse += torch.sum(torch.argmax(torch.softmax(output_sparse, 1), 1) == y_truth).detach().item()
                                    loss_dense = criterion(output_dense, y_truth)
                                    loss_sparse = criterion(output_sparse, y_truth)
                                    loss_pred_in_sample_dense += n_batch*loss_dense.detach().item()
                                    loss_pred_in_sample_sparse += n_batch*loss_sparse.detach().item()
                                    entropy_loss_dense, selection_loss_dense, l2_loss_dense = model_wrapper_dense.get_losses()
                                    entropy_loss_sparse, selection_loss_sparse, l2_loss_sparse = model_wrapper_sparse.get_losses()
                                    # if model_wrapper.step_temp == 446:
                                    #       # if not(model_wrapper.dense_to_sparse):
                                    #       #       np.save("output_dense_445.npy", output.data.numpy())
                                    #       import ipdb;ipdb.set_trace()
                                    loss_dense += entropy_loss_dense + selection_loss_dense + l2_loss_dense
                                    loss_sparse += entropy_loss_sparse + selection_loss_sparse + l2_loss_sparse
                                    approx_loss_in_sample_with_pen_dense += n_batch*loss_dense.detach().item()
                                    approx_loss_in_sample_with_pen_sparse += n_batch*loss_sparse.detach().item()
                                    loss_dense.backward()  # Derive gradients.
                                    loss_sparse.backward()  # Derive gradients.
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
                                    test_complet, max_diff, max_diff_fc, max_diff_mom = diff_model_wrapper(model_wrapper_dense, model_wrapper_sparse, optimizer_dense, optimizer_sparse)
                                    if max_diff>1e-3:
                                        import ipdb;ipdb.set_trace()
                                    # model_wrapper_dense.model.fc1.weight.grad = torch.round(model_wrapper_dense.model.fc1.weight.grad, decimals = 5)
                                    # model_wrapper_dense.model.fc2.weight.grad = torch.round(model_wrapper_dense.model.fc2.weight.grad, decimals = 5)
                                    # model_wrapper_dense.model.fc3.weight.grad = torch.round(model_wrapper_dense.model.fc3.weight.grad, decimals = 5)
                                    # model_wrapper_sparse.model.fc1.weight.grad = torch.round(model_wrapper_sparse.model.fc1.weight.grad, decimals = 5)
                                    # model_wrapper_sparse.model.fc2.weight.grad = torch.round(model_wrapper_sparse.model.fc2.weight.grad, decimals = 5)
                                    # model_wrapper_sparse.model.fc3.weight.grad = torch.round(model_wrapper_sparse.model.fc3.weight.grad, decimals = 5)

                                    # model_wrapper_dense.model.fc1.weight_z.grad = torch.round(model_wrapper_dense.model.fc1.weight_z.grad, decimals = 5)
                                    # model_wrapper_dense.model.fc2.weight_z.grad = torch.round(model_wrapper_dense.model.fc2.weight_z.grad, decimals = 5)
                                    # model_wrapper_dense.model.fc3.weight_z.grad = torch.round(model_wrapper_dense.model.fc3.weight_z.grad, decimals = 5)
                                    # model_wrapper_sparse.model.fc1.weight_z.grad = torch.round(model_wrapper_sparse.model.fc1.weight_z.grad, decimals = 5)
                                    # model_wrapper_sparse.model.fc2.weight_z.grad = torch.round(model_wrapper_sparse.model.fc2.weight_z.grad, decimals = 5)
                                    # model_wrapper_sparse.model.fc3.weight_z.grad = torch.round(model_wrapper_sparse.model.fc3.weight_z.grad, decimals = 5)

                                    optimizer_dense.step()  # Update parameters based on gradients.
                                    optimizer_sparse.step()  # Update parameters based on gradients.
                                    
                                    test_complet, max_diff, max_diff_fc, max_diff_mom = diff_model_wrapper(model_wrapper_dense, model_wrapper_sparse, optimizer_dense, optimizer_sparse)
                                    if max_diff>1e-3:
                                        import ipdb;ipdb.set_trace()

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
                                    optimizer_dense, test_pruned_dense = model_wrapper_dense.prune_models()
                                    optimizer_sparse, test_pruned_sparse = model_wrapper_sparse.prune_models()
                                    test_complet, max_diff, max_diff_fc, max_diff_mom = diff_model_wrapper(model_wrapper_dense, model_wrapper_sparse, optimizer_dense, optimizer_sparse)
                                    if max_diff>1e-3:
                                        import ipdb;ipdb.set_trace()
                                        print("error")

                                    if not(test_complet) and max_diff<=1e-5 and max_diff_fc <= 1e-5:
                                        make_match_wrapper(model_wrapper_dense, model_wrapper_sparse)

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
                                approx_loss_in_sample_no_pen_dense = loss_pred_in_sample_dense/n_seen
                                approx_loss_in_sample_no_pen_sparse = loss_pred_in_sample_sparse/n_seen
                                approx_loss_in_sample_with_pen_dense = approx_loss_in_sample_with_pen_dense/n_seen
                                approx_loss_in_sample_with_pen_sparse = approx_loss_in_sample_with_pen_sparse/n_seen
                                if type_of_task=="classification":
                                    approx_acc_train_dense = approx_acc_train_dense/n_seen
                                    approx_acc_train_sparse = approx_acc_train_sparse/n_seen
                                try:
                                    approx_loss_in_sample_no_pen_dense = approx_loss_in_sample_no_pen_dense.detach().item()
                                    approx_loss_in_sample_no_pen_sparse = approx_loss_in_sample_no_pen_sparse.detach().item()
                                except:
                                    pass
                                optimizer_dense.zero_grad()
                                optimizer_sparse.zero_grad()
                                model_wrapper_dense.model.eval()
                                model_wrapper_sparse.model.eval()
                                #print("Cuda memory before val:", torch.cuda.memory_allocated("cuda"))
                                acc_val_dense, val_loss_dense, _ = get_loss_metric(model_wrapper_dense, loader_val, type_of_task, criterion, scaler_y)
                                acc_val_sparse, val_loss_sparse, _ = get_loss_metric(model_wrapper_sparse, loader_val, type_of_task, criterion, scaler_y)

                                n_z_dense = model_wrapper_dense.get_n_z()
                                sparsity_dense = 1 - n_z_dense/model_wrapper_dense.n_params_original_z

                                n_z_sparse = model_wrapper_sparse.get_n_z()
                                sparsity_sparse = 1 - n_z_sparse/model_wrapper_sparse.n_params_original_z

                                if len(l_n_z)>0:
                                    old_sparsity = 1 - l_n_z[-1]/model_wrapper_dense.n_params_original_z
                                    sparsity_increases_before_goal_being_reached = not(test_sparsity_reached) and (sparsity_dense>old_sparsity)
                                else:
                                    sparsity_increases_before_goal_being_reached = False

                                sparsity_becomes_reached = not(test_sparsity_reached) and (sparsity_dense>=goal_sparsity)
                                condition_sparsity = sparsity_increases_before_goal_being_reached or sparsity_becomes_reached

                                if sparsity_dense>=goal_sparsity:
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
                                                condition_improvement = (val_loss_dense < best_val_loss) or condition_sparsity
                                            elif metric_early_stopping == "val_accuracy":
                                                condition_improvement = (acc_val_dense > best_val_acc) or condition_sparsity
                                            if condition_improvement:
                                                print("--- CONDITION IMPROVEMENT (classification es) ---")
                                                best_val_loss = val_loss_dense
                                                best_val_acc = acc_val_dense
                                                best_ep = epoch_counter
                                                best_model = copy.deepcopy(model_wrapper_dense.model)
                                                n_epochs_no_improvement = 0
                                            else:
                                                if test_sparsity_reached:
                                                        n_epochs_no_improvement += 1
                                test_model_stuck = False
                                tol_epsilon = 1e-6
                                print_loss_no_pen = "Approx in-sample loss with no pen"
                                print_loss_with_pen = "Approx in-sample loss with pen"
                                if type_of_task == "classification":
                                    if epoch >=1:
                                            test_model_stuck = abs(value_in_sample_metric_dense-approx_acc_train_dense)<=tol_epsilon and abs(value_loss_no_pen_dense - approx_loss_in_sample_no_pen_dense)<=tol_epsilon and abs(value_loss_with_pen_dense - approx_loss_in_sample_with_pen_dense)<=tol_epsilon
                                    value_in_sample_metric_dense = 100*approx_acc_train_dense
                                    value_in_sample_metric_sparse = 100*approx_acc_train_sparse
                                    print_metric = "Approx in-sample accuracy"
                                value_loss_no_pen_dense = approx_loss_in_sample_no_pen_dense
                                value_loss_with_pen_dense = approx_loss_in_sample_with_pen_dense
                                value_loss_no_pen_sparse = approx_loss_in_sample_no_pen_sparse
                                value_loss_with_pen_sparse = approx_loss_in_sample_with_pen_sparse

                                if type_of_task == "classification":
                                    print(f'Epoch dense: {epoch:03d}, {print_loss_no_pen}: {value_loss_no_pen_dense:.4f}, {print_loss_with_pen}: {value_loss_with_pen_dense:.4f}, Validation loss: {val_loss_dense:.4f}, {print_metric}: {value_in_sample_metric_dense:.4f}, Val Acc: {100*acc_val_dense:.4f}, lr: {current_lr:.4f}, n_z: {n_z_dense:.4f}, sparsity: {sparsity_dense:4f}', flush=True)
                                    print(f'Epoch sparse: {epoch:03d}, {print_loss_no_pen}: {value_loss_no_pen_sparse:.4f}, {print_loss_with_pen}: {value_loss_with_pen_sparse:.4f}, Validation loss: {val_loss_sparse:.4f}, {print_metric}: {value_in_sample_metric_sparse:.4f}, Val Acc: {100*acc_val_sparse:.4f}, lr: {current_lr:.4f}, n_z: {n_z_sparse:.4f}, sparsity: {sparsity_sparse:4f}', flush=True)
                                test_complet, max_diff, max_diff_fc, max_diff_mom = diff_model_wrapper(model_wrapper_dense, model_wrapper_sparse, optimizer_dense, optimizer_sparse)
                                if max_diff>1e-3:
                                    import ipdb;ipdb.set_trace()
                                    print("error")
                                l_lr.append(current_lr)
                                l_in_sample_loss.append(value_loss_with_pen_dense)
                                l_validation_loss.append(val_loss_dense)
                                l_n_z.append(n_z_dense)
                                if type_of_task == "classification":
                                    l_validation_metric.append(acc_val_dense)
                                    l_in_sample_metric.append(value_in_sample_metric_dense)
                                l_times_epochs.append(time.time()-start_epoch)
                                if np.isnan(value_loss_with_pen_dense):
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