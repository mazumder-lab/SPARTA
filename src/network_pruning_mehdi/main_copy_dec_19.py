#%% Imports
import numpy as np
import argparse
import time

import os
import socket
# from multiprocessing import Manager

machine_name = socket.gethostname()
print("--- MACHINE NAME ---:", machine_name, flush=True)

# print("PATH IMAGENET")
# exec(open('/usr/share/modules/init/python.py').read())
# module("load /home/gridsan/groups/datasets/ImageNet/modulefile")

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
parser.add_argument('--mode', type=str, default = "ensemble",
                    help='ensemble (train all the layers at the same time) or layer_wise (only layer per layer)')
parser.add_argument('--timeout', type=int, default = 72*60*60,
                    help='timeout for the fine tuning')
parser.add_argument('--n_trials', type=int, default = 1,
                    help='number of trials for the fine tuning')
parser.add_argument('--n_repeat', type=int, default = 1,
                    help='number of times we repeat the experiment for a given set of hyperparameters')
parser.add_argument('--name_dataset', type=str, default = "mnist",
                    help='name of the dataset')
parser.add_argument('--n_train_kept', type=int, default = -1,
                    help='number of training samples kept')
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
parser.add_argument('--batch_size_dataset', type=int, default = 128,
                    help='batch size for the dataset')
parser.add_argument('--test_save_all_models', type=int, default = 0,
                    help='If test_save_all_models=1, we save all the models during the training (one model per epoch)')
parser.add_argument('--val_second_lr', type=float, default = -1,
                    help='if a value is given and test_diff_lr = 1, this learning rate is used for the z_i, otherwise lr/n_steps_per_epoch is used when test_diff_lr = 1')
parser.add_argument('--test_early_stopping', type=int, default = 1,
                    help='if test_early_stopping==1, the best model out of the n_epochs iterations is kept based on the validation loss. If test_early_stopping==0, the training loss is used. If test_early_stoppin==2, the validation loss is used to obtain the goal sparsity and the training loss to further train the model.')
parser.add_argument('--type_decay', type=str, default = "cosine",
                    help='criteria for the decay. If type_decay = "None", then no decay is applied. The other types of decays are "linear", "exponential" and "cosine"')
parser.add_argument('--gamma_lr_decay', type=float, default = 0.9,
                    help='learning rate decay for type_decay = "exponential"')
parser.add_argument('--T_max_cos', type=int, default = -1,
                    help='half-period of the cosine for type_decay = "cosine", if -1, then set to the number of epochs')
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
parser.add_argument('--entropy_reg', type=float, default = 1e-3,
                    help='regularizer factor for entropy penalization')
parser.add_argument('--selection_reg', type=float, default = 1e-3,
                    help='regularizer factor for selection penalization')
parser.add_argument('--metric_early_stopping', type=str, default = "val_loss",
                    help='either val_loss or val_accuracy')
parser.add_argument('--l2_reg', type=float, default = 1e-3,
                    help='regularizer factor for l2 penalization')
parser.add_argument('--l2_original_reg', type=float, default = 0.0,
                    help='regularizer factor for l2 penalization')
parser.add_argument('--period_milestones', type=int, default = 25,
                    help='period of the milestones for multi_lr scheduler')
parser.add_argument('--test_different_lr', type=int, default = 0,
                    help='if set to 1, then lr/steps_per_epoch is used for the weights corresponding to the z_i and z_ij. Otherwise, the regular learning rate is used')
parser.add_argument('--dense_to_sparse', type=int, default = 0,
                    help='if set to 1, then the weights of the model are eliminated during the training. Currently, only works with Adam, has to be set to 0 for another optimizer.')
parser.add_argument('--optimizer_name', type=str, default = "SGD",
                    help='Adam, SGD')
parser.add_argument('--goal_sparsity', type=float, default = 0.0,
                    help='the training will continue as long as goal sparsity is not reached')
parser.add_argument('--tol_z_1', type=float, default = 1.0,
                    help='tolerance to consider that a z is equal to 1 (for the restart)')
parser.add_argument('--type_training', type=str, default = "combined",
                    help="if type_training='combined', gradient descent is performed on both the weights and the zs, if type_training='alternate', we alternate between training of the weights and training of the zs")
parser.add_argument('--n_restart', type=int, default = 0,
                    help='number of times we restart the weight_z during training if type_training = "alternate"')
parser.add_argument('--loss_func', type=str, default = "classic",
                    help='either classic (mse/cross entropy) or layer_wise')
parser.add_argument('--type_reset', type=str, default = "ensemble",
                    help='how to select the weights to reset (either layer_wise or ensemble)')
parser.add_argument('--threshold_weights', type=int, default = 150000,
                    help='we keep adding layers together until we reach this number of weights in a block of layers')
parser.add_argument('--method_pruning', type=str, default = "schedule",
                    help='Either schedule, threshold or both: whether to use a fixed pruning schedule, a threshold or both. If both, the pruning schedule gives an upper bound on the number of weights to be pruned based on the threshold.')
parser.add_argument('--threshold_restart', type=float, default = 1e-4,
                    help='threshold on the value of the weight to decide whether it is worth pruning when method_pruning')
parser.add_argument('--test_constraint_weights', type=int, default = 0,
                    help='whether to use the Lagrangean approach in order to enfore a constraint on the percentage of small weights')
parser.add_argument('--test_one_layer_pruning', type=int, default = 0,
                    help='If set to 1, then threshold_weights is ignored. Each layer is pruned one by one (no block is created)')
parser.add_argument('--test_prop_goal_sparsity', type=int, default = 0,
                    help='Used only for layer-wise pruning. If set to 1, each block of layer is assigned a goal sparsity proportional to the number of parameters')
parser.add_argument('--test_normalized_sgd', type=int, default = 1,
                    help='If set to 1, normalized sgd is used for the z only.')
parser.add_argument('--type_function', type=str, default = "smoothstep",
                    help='Either smoothstep or sigmoid.')
parser.add_argument('--pruning_rate_cte', type=float, default = -1,
                    help='If set to -1, a handmade pruning schedule is used, otherwise pruning_rate_cte of the weights is pruned each time.')
parser.add_argument('--lambda_loss', type=float, default = 1,
                    help='Multiply the layer-wise reconstruction loss by lambda_loss')
parser.add_argument('--test_load_data_first', type=int, default = 1,
                    help='If set to 1, the data is loaded in memory before training starts')
parser.add_argument('--test_repeat_if_sparsity_not_reached', type=int, default = 1,
                    help='If set to 1, the whole pruning process is repeated for n_epochs more epochs to reach the goal sparsity')
parser.add_argument('--loss_last_block', type=str, default = "layer_wise",
                    help='Only used for layer-wise training. Either "mce" or "layer-wise" reconstruction.')
parser.add_argument('--retrain_last_block', type=int, default = 0,
                    help='If set to 1 and if loss_last_block=="layer_wise" (in the case of layer_wise training), the last block is retrained on the original (mce or mse) loss')
parser.add_argument('--test_mult_reset', type=int, default = 1,
                    help='whether to multiply the new weights by 1/z when the z are reset to approx 0.5')
parser.add_argument('--test_reset_to_orignal', type=int, default = 0,
                    help='whether to replace all the weights to their original value each time a reset is performed')
parser.add_argument('--test_start_sparse_gpt', type=int, default = 0,
                    help='wether to prune first using sparse-gpt and then fine tune the weights using layer-wise and/or actual loss function')
parser.add_argument('--test_start_convex', type=int, default = 0,
                    help='wether to prune first using sparse-gpt and then fine tune the weights using layer-wise and/or actual loss function')
parser.add_argument('--prune_bias', type=int, default = 1,
                    help='wether to prune the bias or not')
parser.add_argument('--type_compute_sparsity', type=str, default = "total",
                    help='either "prunable" or "total". If "prunable", the sparsity is computed only for the modules that are pruned. If "total", we consider the total sparsity when pruning (with the non pruned modules).')
parser.add_argument('--test_adaptive_lr', type=int, default = 1,
                    help='wether to increase the learning rate each time the validation loss does seem to make any progress for patience epochs.')
parser.add_argument('--patience_adaptive_lr', type=int, default = 5,
                    help='patience for the adaptive learning rate (if the validation loss increases slowly for patience_adaptive_lr, the learning rate is multiplied by 2).')
parser.add_argument('--patience_freeze', type=int, default = 1,
                    help='patience for the freezing of the weights')
parser.add_argument('--test_wait_for_pruning', type=int, default = 0,
                    help='In this case, the limit on the number of epochs, n_epochs, is used only after the sparsity is reached')
parser.add_argument('--test_almost_sequential', type=int, default = 3,
                    help='In this case, we do not save the original dataset and only use the dataset created from the new model')
parser.add_argument('--tol_ent_reg', type=float, default = 1e-2,
                    help='tolerance on the evolution of z to decide to multiply ent_reg by 2')
parser.add_argument('--tol_sel_reg', type=float, default = 1e-2,
                    help='tolerance on the evolution of sparsity to decide to multiply sel_reg by 2')
parser.add_argument('--goal_sparsity_discrete', type=float, default = 0.0,
                    help='goal_sparsity for sparse_gpt')
parser.add_argument('--activation_fn', type=str, default = "relu",
                    help='activation function for llm')
parser.add_argument('--n_incr_gradual_pruning', type=int, default = -1,
                    help='if n_incr_gradual_pruning==-1, no gradual pruning is used. Otherwise, we perform gradual pruning n_incr_gradual_pruning times')
parser.add_argument('--type_pruning_schedule', type=str, default = "linear",
                    help='type of pruning schedule used for gradual pruning (linear or exponential)')

# parser.add_argument('--layer_wise_reg', type=float, default = 0.0,
#                     help='penalization for the layer wise reconstruction')

arguments, unknown = parser.parse_known_args()

#%%
if __name__ == '__main__':
    from previous_utils.main_utils import *
    from utils_training import *
    from utils_experiments import *
    from utils_model import *
    # import signal
    from pytorch_dataset_2_0 import random_split

    from Sparse_GPT_utils.opt import opt_sequential
    from collections import OrderedDict

    # signal.signal(signal.SIGVTALRM, lambda signum, frame: print("\n--- Time is over ---"))

    print('Parsed arguments:', arguments, flush=True)
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
    mode = arguments.mode
    loss_func = arguments.loss_func
    type_reset = arguments.type_reset
    n_train_kept = arguments.n_train_kept
    threshold_weights = arguments.threshold_weights
    method_pruning = arguments.method_pruning
    threshold_restart = arguments.threshold_restart
    test_constraint_weights = arguments.test_constraint_weights
    test_one_layer_pruning = arguments.test_one_layer_pruning
    test_prop_goal_sparsity = arguments.test_prop_goal_sparsity
    l2_original_reg = arguments.l2_original_reg
    test_normalized_sgd = arguments.test_normalized_sgd
    type_function = arguments.type_function
    pruning_rate_cte = arguments.pruning_rate_cte
    lambda_loss = arguments.lambda_loss
    test_load_data_first = arguments.test_load_data_first
    test_repeat_if_sparsity_not_reached = arguments.test_repeat_if_sparsity_not_reached
    loss_last_block = arguments.loss_last_block
    retrain_last_block = arguments.retrain_last_block
    test_mult_reset = arguments.test_mult_reset
    test_reset_to_orignal = arguments.test_reset_to_orignal
    test_start_sparse_gpt = arguments.test_start_sparse_gpt
    prune_bias = arguments.prune_bias
    type_compute_sparsity = arguments.type_compute_sparsity
    test_adaptive_lr = arguments.test_adaptive_lr
    patience_adaptive_lr = arguments.patience_adaptive_lr
    patience_freeze = arguments.patience_freeze
    test_wait_for_pruning = arguments.test_wait_for_pruning
    test_almost_sequential = arguments.test_almost_sequential
    tol_ent_reg = arguments.tol_ent_reg
    tol_sel_reg = arguments.tol_sel_reg
    goal_sparsity_discrete = arguments.goal_sparsity_discrete
    activation_fn = arguments.activation_fn
    n_incr_gradual_pruning = arguments.n_incr_gradual_pruning
    test_start_convex = arguments.test_start_convex
    type_pruning_schedule = arguments.type_pruning_schedule

    if test_one_layer_pruning:
        threshold_weights = 1

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
                   type_pruning_schedule)
    
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    # required for the distributed training
    if test_distributed:
        import torch.distributed as dist
        # from utils_distributed import sync_dataset

        if "WORLD_SIZE" in os.environ:
            print("-- WORLD_SIZE in os.environ --", flush=True)
            arguments.world_size = int(os.environ["WORLD_SIZE"])
        else:
            print("-- WORLD_SIZE not in os.environ --", flush=True)
            arguments.world_size = 1
        arguments.distributed = arguments.world_size > 1

        if arguments.distributed:
            if 'SLURM_PROCID' in os.environ: # for slurm scheduler
                print("--- HOURAA ---", flush=True)
                arguments.rank = int(os.environ['SLURM_PROCID'])
                arguments.global_rank = int(os.environ['RANK'])
                print("DEVICE COUNT:", torch.cuda.device_count(), flush=True)
                # # TO DELETE LATER MAYEB
                # if arguments.rank in [0,2]:
                #     arguments.gpu = 0
                # else:
                #     arguments.gpu = 1
                # arguments.gpu = arguments.local_rank 
                # # END
                #arguments.gpu = arguments.rank % torch.cuda.device_count()
                arguments.gpu = arguments.rank % torch.cuda.device_count()
            elif arguments.local_rank != -1: # for torch.distributed.launch
                arguments.rank = arguments.local_rank
                arguments.gpu = arguments.local_rank
        else:
            arguments.rank = 0
            arguments.gpu=0

        if arguments.distributed:
            device = arguments.gpu
        else:
            device = torch.device(device_name)
        print('Using device ', device, flush=True)

        ngpus_per_node = torch.cuda.device_count()

        torch.backends.cudnn.benchmark = True
    else:
        if device == "None":
            device = 'cuda' if torch.cuda.is_available() else 'cpu'


    ##Change this to path of imagenet name_dataset
    if 'IMAGENET_PATH' in os.environ:  
        IMAGENET_PATH = os.environ['IMAGENET_PATH']+"/raw"
    else:
        print('****Warning**** No IMAGENET_PATH variable', flush=True)
        #IMAGENET_PATH = ''
        IMAGENET_PATH = "/run/user/62607/loopmnt4/raw"
    CIFAR10_PATH = '../datasets'
    MNIST_PATH = '../datasets'
    C4_PATH = "/home/gridsan/gafriat/Sparse_NN_shared/LLM/data/"
    WIKITEXT_PATH = "/home/gridsan/gafriat/Sparse_NN_shared/LLM/data/"
    PTB_PATH = "/home/gridsan/gafriat/Sparse_NN_shared/LLM/data/"

    name_dataset_paths = {'imagenet':IMAGENET_PATH,'cifar10':CIFAR10_PATH,
                    'mnist':MNIST_PATH, 'c4':C4_PATH, 'wikitext2':C4_PATH, 'ptb':C4_PATH}

    name_dataset_path = name_dataset_paths[name_dataset]

    print("Name dataset:", name_dataset, flush=True)
    print("Path dataset:", name_dataset_path, flush=True)
    n_params_original_z = 0
    
    if name_dataset == "mnist":
        get_item_func = get_item_mnist
    elif name_dataset == "cifar10":
        get_item_func = get_item_cifar10
    elif name_dataset == "imagenet":
        get_item_func = get_item_imagenet
    else:
        get_item_func = None

    train_val_dataset, test_dataset = get_dataset(name_dataset, name_dataset_path, n_train_kept, get_item_func, arch, seed, activation_fn, device)

    if name_dataset in ["c4", "wikitext2", "ptb"]:
        n_train_kept = -1
        (train_val_dataset, train_val_attention_mask), (test_dataset, test_attention_mask) = train_val_dataset, test_dataset
        if torch.sum(torch.abs(train_val_attention_mask-test_attention_mask)).item()!=0:
            print("--- DIFFERENCE IN ATTENTION MASK ---")
            import ipdb;ipdb.set_trace()
    if not ("deit" in arch):
        initialize_dataset(train_val_dataset, n_train_kept, name_dataset)
        initialize_dataset(test_dataset, -1, name_dataset)
    if test_almost_sequential==1:
        train_val_dataset.return_original = False
        test_dataset.return_original = False


    # if test_early_stopping==0:
    #     generator_split = torch.Generator().manual_seed(seed)
    #     train_dataset, validation_dataset = random_split(train_val_dataset, [1.0, 0.0], generator=generator_split)
    # else:

    generator_split = torch.Generator().manual_seed(seed)
    train_dataset, validation_dataset = random_split(train_val_dataset, [0.8, 0.2], generator=generator_split)
    copy_indices_train = copy.deepcopy(train_dataset.indices)
    copy_indices_val = copy.deepcopy(validation_dataset.indices)
    
    print("--- Device =", device, "---", flush=True)

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
            print("Continuing existing study...", flush=True)
        else:
            print("New study...")
        print(str(n_trials)+" trials restants", flush=True)
    if test_deja_train:
        print("Training already done for "+name_study, flush=True)
    else:
        if path_load_weights!="":
            model_loaded = True
            model_state_dict = torch.load(path_load_weights, map_location=device)
        # %% Second case: the same beta for all the times
        def objective(trial):
            try:
                global dataset, n_params_original_z, train_dataset, validation_dataset, test_dataset, n_train_kept, train_val_attention_mask, batch_size_dataset
            except:
                pass
            if learning_rate!=-1:
                lr = learning_rate
            else:
                lr = trial.suggest_float("lr", min_lr, max_lr, log = True)
            print('Parsed arguments:', arguments, flush=True)
            print("lr:", lr, flush=True)
            if type_of_task== "classification":
                best_val_metric_best = -np.inf
                if name_dataset in ["c4", "wikitext2", "ptb"]:
                    metric_name = "perplexity"
                else:
                    metric_name = "accuracy"
            elif type_of_task=="regression":
                best_val_metric_best = np.inf
                metric_name = "mse"

            best_val_metric_avg = 0

            for ind_repeat in range(n_repeat):
                # signal.alarm(0)
                # signal.alarm(24*60*60)
                print("Repeat", ind_repeat+1, "out of", n_repeat, flush=True)
                # Model initialization
                model, criterion, modules_to_prune = get_model(arch, seed, pretrained=pretrained, with_z=True, gamma=gamma, prune_bias=prune_bias, activation_fn=activation_fn)
                # if test_start_sparse_gpt:
                #     if "facebook/opt" in arch:
                #         seqlen = 2048
                #     dataloader_sparsegpt, _ = get_loaders(
                #             "c4", C4_PATH, nsamples=arguments.n_train_kept, seed=0, model=arch, seqlen=seqlen
                #             )
                #     model.eval()
                #     initialize_pruning_rec(model, type_pruning, type_function)
                #     opt_sequential(model, dataloader_sparsegpt, device, arguments.n_train_kept, -1, 1000, 16, '', False, goal_sparsity, 0, 0, 0.01, 128, True)
                #     model.train()

                if name_dataset in ["c4", "wikitext2", "ptb"]:
                    model.float()

                # if name_dataset in ["c4", "wikitext2", "ptb"]:
                #     n_params_not_included = 0
                #     n_params_not_included += np.sum([np.prod(x.shape) for x in list(model.model.decoder.embed_tokens.parameters())])
                #     n_params_not_included += np.sum([np.prod(x.shape) for x in list(model.model.decoder.embed_positions.parameters())])
                #     if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
                #         n_params_not_included += np.sum([np.prod(x.shape) for x in list(model.model.decoder.project_out.parameters())])
                #     if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
                #         n_params_not_included += np.sum([np.prod(x.shape) for x in list(model.model.decoder.project_in.parameters())])
                #     total_number_of_params = np.sum([np.prod(x[1].shape) for x in list(model.named_parameters()) if "_z" not in x[0]])
                #     n_final = (1-goal_sparsity)*total_number_of_params
                #     n_final_sub_network = n_final-n_params_not_included
                #     goal_sparsity_used = 1-n_final_sub_network/(total_number_of_params-n_params_not_included)
                goal_sparsity_used = goal_sparsity
                name_model = model.__str__().lower()
                if loss_func == "layer_wise" or l2_original_reg!=0.0:
                    original_model, criterion, modules_to_prune = get_model(arch, seed, pretrained=pretrained, with_z=False, gamma=gamma, prune_bias=prune_bias, activation_fn=activation_fn)
                                            
                    if name_dataset in ["c4", "wikitext2", "ptb"]:
                        original_model.float()

                    # if test_start_sparse_gpt:
                        # weights_sparse_gpt = original_model.state_dict()
                        # weights_sparse_gpt = list(weights_sparse_gpt.items())
                        # ind_weight = 0
                        # while ind_weight < len(weights_sparse_gpt):
                        #     name_layer = weights_sparse_gpt[ind_weight][0]
                        #     if not(("embed" in name_layer) or ("norm" in name_layer) or ("_z" in name_layer)):
                        #         new_name_layer = name_layer+"_z"
                        #         new_weights = torch.ones_like(weights_sparse_gpt[ind_weight][1])*gamma
                        #         new_weights[weights_sparse_gpt[ind_weight][1]==0] = -gamma
                        #         weights_sparse_gpt.insert(ind_weight+1, (new_name_layer, new_weights))
                        #     ind_weight+=1
                        # model.load_state_dict(OrderedDict(weights_sparse_gpt), strict=False)
                else:
                    original_model = None
                print("Model intialized", flush = True)
                print("Test distributed:",test_distributed, flush = True)
                if test_distributed:
                    if arguments.distributed:
                        print("RANK =", arguments.rank, flush = True)
                        torch.cuda.set_device(arguments.gpu)
                        #dist.init_process_group(backend='nccl', init_method='env://',
                        #                        world_size=arguments.world_size, rank=arguments.rank)
                        dist.init_process_group("nccl")
                        print("DDP initialized", flush = True)
                        # torch.backends.cudnn.benchmark = True

                        if arguments.rank!=0:
                            def print_pass(*arguments):
                                pass
                            #builtins.print = print_pass ##Uncomment to block all other ranks from printing

                        rank = dist.get_rank()
                        print(f"Start running basic DDP example on rank {rank}.", flush=True)
                        # create model and move it to GPU with id rank
                        #device_id = rank % torch.cuda.device_count()
                        if not(mode=="layer_wise") or not(loss_func=="layer_wise"):
                            model.cuda(arguments.gpu)
                        #model.to(device_id)
                        # model.to(f'cuda:{device_id}')
                        # model.to("cuda")
                        print("CURRENT GPU:", arguments.gpu, flush=True)
                        #print(arguments.local_rank, arguments.rank, arguments.gpu)
                        #print("device id", device_id)
                        if not(mode=="layer_wise") or not(loss_func=="layer_wise"):
                            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[arguments.gpu])
                        #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=None)
                        print("model DDP initialized", flush = True)
                        # model_without_ddp = model.module
                        #modules_to_prune = ["module." + x for x in modules_to_prune]
                        modules_to_prune = [x for x in modules_to_prune]
                    # else:
                        # model = model.to(device)
                        # model_without_ddp = model
                    print("Dist set up", flush = True)
                    if arguments.distributed:
                        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
                    else:
                        train_sampler = None
                    print("train_sampler set up", flush = True)

                if test_distributed:
                    if arguments.distributed:
                        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
                    else:
                        train_sampler = None
                    print("train_sampler set up", flush = True)

                generator_loader = torch.Generator()
                if seed != -1:
                    torch.random.manual_seed(seed)
                    generator_loader = generator_loader.manual_seed(seed)

                if test_distributed:
                    loader_train = DataLoader(train_dataset, batch_size=batch_size_dataset, shuffle=(train_sampler is None), generator=generator_loader, num_workers=num_workers, pin_memory=True, sampler=train_sampler)
                    loader_val = DataLoader(validation_dataset, batch_size=batch_size_dataset, num_workers=num_workers, pin_memory=True)
                    loader_test = DataLoader(test_dataset, batch_size=batch_size_dataset, num_workers=num_workers, pin_memory=True)
                    print("loader created", flush = True)
                else:
                    loader_train = DataLoader(train_dataset, batch_size=batch_size_dataset, shuffle=True, generator=generator_loader, num_workers=num_workers, pin_memory=True)
                    loader_val = DataLoader(validation_dataset, batch_size=batch_size_dataset, num_workers=num_workers, pin_memory=True)
                    loader_test = DataLoader(test_dataset, batch_size=batch_size_dataset, num_workers=num_workers, pin_memory=True)
                if test_load_data_first and not("deit" in arch):
                    print("Load data in memory first...", flush = True)
                    if test_almost_sequential==1:
                        test_update_original = False
                    else:
                        test_update_original = True
                    load_dataset_in_memory(loader_train, loader_val, n_train_kept, test_update_original)
                    print("Done!", flush = True)
                # TEST LOADER
                # loader_train.dataset.dataset.targets = np.array(loader_train.dataset.dataset.targets)
                # loader_train.dataset.dataset.is_original = False
                # train_data = copy.deepcopy(loader_train.dataset.dataset.data[loader_train.dataset.indices])
                # val_data = copy.deepcopy(loader_val.dataset.dataset.data[loader_val.dataset.indices])
                # train_targets = copy.deepcopy(loader_train.dataset.dataset.targets[loader_train.dataset.indices])
                # val_targets = copy.deepcopy(loader_val.dataset.dataset.targets[loader_val.dataset.indices])
                # for batch_sgd in tqdm(loader_train):
                #     update_loader(loader_train, batch_sgd[0], batch_sgd[1], False)
                # for batch_sgd in tqdm(loader_val):
                #     update_loader(loader_train, batch_sgd[0], batch_sgd[1], False)

                # loader_train.dataset.indices = list(np.arange(len(loader_train.dataset.indices)))
                # loader_val.dataset.indices = list(len(loader_train.dataset.indices)+np.arange(len(loader_val.dataset.indices)))
                # loader_train.dataset.dataset.is_original = False
                # loader_train.dataset.dataset.data = np.concatenate(loader_train.dataset.dataset.new_data)
                # loader_train.dataset.dataset.targets = np.concatenate(loader_train.dataset.dataset.new_targets)
                # loader_train.dataset.dataset.required_increment = np.zeros(loader_train.dataset.dataset.data.shape[0], dtype=int)
                # loader_train.dataset.dataset.new_data = []
                # loader_train.dataset.dataset.new_targets = []

                # new_train_data = []
                # new_val_data = []
                # new_train_targets = []
                # new_val_targets = []

                # for batch_sgd in tqdm(loader_train):
                #     new_train_data.append(batch_sgd[0])
                #     new_train_targets.append(batch_sgd[1])

                # for batch_sgd in tqdm(loader_val):
                #     new_val_data.append(batch_sgd[0])
                #     new_val_targets.append(batch_sgd[1])

                # new_train_data = np.concatenate(new_train_data)
                # new_val_data = np.concatenate(new_val_data)
                # new_train_targets = np.concatenate(new_train_targets)
                # new_val_targets = np.concatenate(new_val_targets)
                # # new_train_data = new_train_data.swapaxes(-1,-2).swapaxes(-2,-3)
                # # new_val_data = new_val_data.swapaxes(-1,-2).swapaxes(-2,-3)
                # for i in tqdm(range(new_val_data.shape[0])):
                #     if np.prod(np.max(np.abs(val_data - new_val_data[i]),(1,2,3)))!=0:
                #         print("ERROR")
                #         import ipdb;ipdb.set_trace()
                # import ipdb;ipdb.set_trace()
                # END

                scaler_y = None
                #dataset = train_dataset, validation_dataset, test_dataset, scaler_y
                dataset = loader_train, loader_val, loader_test, scaler_y

                # if test_distributed:
                #     if name_dataset == "imagenet" and torch.cuda.device_count()>1:
                #         print('Using DistributedDataParallel with',torch.cuda.device_count(),'GPUs', flush = True)
                #         model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[arguments.gpu])
                model.eval()
                if test_distributed and arguments.rank == 0:
                    print("Evaluating the dense model", flush = True)
                
                # if not(test_distributed):
                #     model_without_ddp = model
                # if not(test_distributed) or arguments.rank == 0 and ("resnet" in name_model):
                #     reinitialize_pruning_rec(model_without_ddp, type_pruning)
                #     model_without_ddp.to(device)
                #     start_test = time.time()
                #     dense_acc = compute_acc(model_without_ddp,loader_test,device=device)
                #     model_without_ddp.to("cpu")
                #     time_test = time.time() - start_test
                #     print('Dense test accuracy', dense_acc,' computation time : ', time_test)
                # print("Evaluation done", flush = True)
                if n_params_original_z==0:
                    n_params_original_z = np.sum([np.prod(x[1].shape) for x in model.named_parameters() if "_z" in x[0]])
                    n_params_original = np.sum([np.prod(x[1].shape) for x in  model.named_parameters() if "_z" not in x[0]])

                if path_load_weights!="":
                    if model_loaded:
                        model.load_state_dict(model_state_dict)
                    print("The weights have been loaded", flush=True)
                # model.to(device)
                
                if name_dataset == "mnist":
                    input_channel = 784
                elif name_dataset == "cifar10" and arch == "resnet20":
                    input_channel = 64
                elif name_dataset == "imagenet" and arch == "mobilenetv1":
                    input_channel = -1
                else:
                    input_channel = -1

                generator = torch.Generator(device=device_name)
                if seed != -1:
                    torch.random.manual_seed(seed)
                    generator = generator.manual_seed(seed)
                
                # Training
                time_before_training = time.time()

                if loss_func == "layer_wise":
                    l_modules_original = list(original_model.modules())
                    l_modules_original = [x for x in l_modules_original if x.__str__()[:5]=="Batch"]
                    l_modules = list(model.modules())
                    l_modules = [x for x in l_modules if x.__str__()[:5]=="Batch"]
                    # Updating the batch norm
                    original_model.to(device)
                    original_model.train()
                    if name_dataset in ["imagenet", "cifar10", "mnist"]:
                        # TO UNCOMMENT LATER
                        # dense_train_acc = compute_acc(original_model, loader_train, device=device)
                        # print('Dense train accuracy', dense_train_acc)
                        model.train()
                        for ind_batch_norm in range(len(l_modules)):
                            l_modules[ind_batch_norm].running_mean = l_modules_original[ind_batch_norm].running_mean
                            l_modules[ind_batch_norm].running_var = l_modules_original[ind_batch_norm].running_var                        

                    # if name_dataset in ["c4", "wikitext2", "ptb"]:
                    #     print("Initializing layer norm original model...", flush=True)
                    #     for batch_sgd in tqdm(loader_train):
                    #         input_batch_sgd, input_batch_original_sgd, target_batch_sgd, index_seen_sgd, old_index_seen_sgd = read_batch(batch_sgd)
                    #         original_model(input_batch_original_sgd.to(device))
                    # End

                    original_model.eval()
                    original_model.to("cpu")
                    set_require_grad_rec(original_model, False)

                # # TO DELETE
                # set_require_grad_rec(model, False)
                # l_modules_before_threshold = get_modules_layer_wise_rec(model, [])
                # l_modules = process_l_modules_w_threshold(l_modules_before_threshold, 15000)
                # l_modules_before_threshold_original = get_modules_layer_wise_rec(original_model, [])
                # l_modules_original = process_l_modules_w_threshold(l_modules_before_threshold_original, 15000)
                # model = l_modules[0]
                # set_require_grad_rec(model, True)
                # original_module = l_modules_original[0]
                # optimizer = initialize_optimizer(test_different_lr, original_module, optimizer_name, steps_per_epoch, lr, val_second_lr, momentum, weight_decay)
                # model_wrap = model_wrapper(model, optimizer, seed, entropy_reg, selection_reg, l2_reg, device, dense_to_sparse, test_different_lr, steps_per_epoch, val_second_lr, momentum, weight_decay, tol_z_1, input_channel, type_pruning, generator, type_reset)
                # is_last_module = False
                # module_training = True
                # mode_model_wrap = "ensemble"
                # l_in_sample_loss, l_in_sample_loss_no_pen, l_validation_loss, l_in_sample_metric, l_validation_metric, l_times_epochs, l_lr, l_n_z, l_sparsity, l_n_params, best_model, best_ep, n_z_final, test_sparsity_reached = train_neural_network(name_study=name_study, name_model=name_model, model_wrapper=model_wrap, dataset=dataset, optimizer=optimizer, criterion=criterion, n_epochs=n_epochs, batch_size_dataset=batch_size_dataset, path_save=None, test_early_stopping=test_early_stopping, trial=trial, test_save_all_models=test_save_all_models, type_decay=type_decay, gamma_lr_decay=gamma_lr_decay, T_max_cos=T_max_cos, eta_min_cos=eta_min_cos, start_lr_decay=start_lr_decay, end_lr_decay=end_lr_decay, warmup_steps=warmup_steps, type_of_task=type_of_task, test_compute_accurate_in_sample_loss=test_compute_accurate_in_sample_loss, folder_saves=folder_saves, ind_repeat=ind_repeat, patience = patience, metric_early_stopping=metric_early_stopping, period_milestones=period_milestones, goal_sparsity=goal_sparsity, type_training=type_training, n_restart=n_restart, num_workers=num_workers, mode=mode_model_wrap, loss_func_and_model=(loss_func, original_module), is_last_module = is_last_module, module_training=module_training)
                # # END

                if test_constraint_weights:
                    selection_lagrangian_reg = Parameter(torch.tensor(selection_reg))
                    selection_lagrangian_reg.to(device)
                    # entropy_lagrangian_reg = Parameter(torch.tensor(entropy_reg))
                    # entropy_lagrangian_reg.to(device)
                    entropy_lagrangian_reg = None
                else:
                    selection_lagrangian_reg = None
                    entropy_lagrangian_reg = None

                if loss_func == "layer_wise" and mode == "layer_wise":
                    if test_constraint_weights:
                        print("----- test_constraint_weights NOT IMPLEMENTED YET IN THIS CASE -----")
                    set_require_grad_rec(model, False)
                    if "facebook/opt" in arch:
                        l_modules_before_threshold = get_modules_layer_wise_llm(model, train_val_attention_mask)
                        l_modules_before_threshold_original = get_modules_layer_wise_llm(original_model, train_val_attention_mask)
                    elif "deit" in arch:
                        l_modules_before_threshold = get_modules_layer_wise_deit(model)
                        l_modules_before_threshold_original = get_modules_layer_wise_deit(original_model)
                    else:
                        l_modules_before_threshold = get_modules_layer_wise_rec(model, [])
                        l_modules_before_threshold_original = get_modules_layer_wise_rec(original_model, [])
                    if test_one_layer_pruning and "resnet" in arch:
                        l_modules_before_threshold, l_downsamples = process_to_one_layer(l_modules_before_threshold)
                        l_modules_before_threshold_original, l_downsamples_original = process_to_one_layer(l_modules_before_threshold_original)
                    if ("mlpnet" in name_model):
                        if test_one_layer_pruning:
                            l_modules_before_threshold = [(View_mlpnet(),0,0,0)]+l_modules_before_threshold[0:1]+[(nn.ReLU(inplace=False),0,0,0)]+l_modules_before_threshold[1:2]+[(nn.ReLU(inplace=False),0,0,0)]+l_modules_before_threshold[2:]+[(Log_softmax_mlpnet(),0,0,0)]
                            l_modules_before_threshold_original = [(View_resnet(),0,0,0)]+l_modules_before_threshold_original[0:1]+[(nn.ReLU(inplace=False),0,0,0)]+l_modules_before_threshold_original[1:2]+[(nn.ReLU(inplace=False),0,0,0)]+l_modules_before_threshold_original[2:]+[(Log_softmax_mlpnet(),0,0,0)]
                        else:
                            l_modules_before_threshold = [View_resnet()]+l_modules_before_threshold[0:1]+[nn.ReLU(inplace=False)]+l_modules_before_threshold[1:2]+[nn.ReLU(inplace=False)]+l_modules_before_threshold[2:]+[Log_softmax_mlpnet()]
                            l_modules_before_threshold_original = [View_resnet()]+l_modules_before_threshold_original[0:1]+[nn.ReLU(inplace=False)]+l_modules_before_threshold_original[1:2]+[nn.ReLU(inplace=False)]+l_modules_before_threshold_original[2:]+[Log_softmax_mlpnet()]
                    if ("resnet" in name_model):
                        if test_one_layer_pruning:
                            l_modules_before_threshold = l_modules_before_threshold[:-1]+[(View_resnet(),0,0,0)]+l_modules_before_threshold[-1:]
                            l_modules_before_threshold_original = l_modules_before_threshold_original[:-1]+[(View_resnet(),0,0,0)]+l_modules_before_threshold_original[-1:]
                        else:
                            l_modules_before_threshold = l_modules_before_threshold[:-1]+[View_resnet()]+l_modules_before_threshold[-1:]
                            l_modules_before_threshold_original = l_modules_before_threshold_original[:-1]+[View_resnet()]+l_modules_before_threshold_original[-1:]
                    if ("mobilenet" in name_model):
                        if test_one_layer_pruning:
                            l_modules_before_threshold = l_modules_before_threshold[:-1]+[(View_mobilenet(),0,0,0)]+l_modules_before_threshold[-1:]
                            l_modules_before_threshold_original = l_modules_before_threshold_original[:-1]+[(View_mobilenet(),0,0,0)]+l_modules_before_threshold_original[-1:]
                        else:
                            l_modules_before_threshold = l_modules_before_threshold[:-1]+[View_mobilenet()]+l_modules_before_threshold[-1:]
                            l_modules_before_threshold_original = l_modules_before_threshold_original[:-1]+[View_mobilenet()]+l_modules_before_threshold_original[-1:]
                    l_modules, l_modules_original = process_l_modules_w_threshold(l_modules_before_threshold, threshold_weights, test_one_layer_pruning, arch, l_modules_before_threshold_original)
                    if "facebook/opt" in arch:
                        for module in l_modules:
                            module.seqlen = model.seqlen
                        for module in l_modules_original:
                            module.seqlen = model.seqlen

                    main_model_wrapper = model_wrapper(model, None, seed, entropy_reg, selection_reg, l2_reg, device, dense_to_sparse, test_different_lr, steps_per_epoch, val_second_lr, momentum, weight_decay, tol_z_1, input_channel, type_pruning, generator, type_reset, method_pruning, threshold_restart, selection_lagrangian_reg, entropy_lagrangian_reg, l2_original_reg, original_model, type_function, test_mult_reset, test_reset_to_orignal, prune_bias, type_compute_sparsity, gamma)
                    main_model_wrapper.initialize_pruning()
                    main_model_wrapper.compute_z()

                    l_optimizers = [initialize_optimizer(test_different_lr, l_modules[ind_module], optimizer_name, steps_per_epoch, lr, val_second_lr, momentum, weight_decay, selection_lagrangian_reg, entropy_lagrangian_reg) for ind_module in range(len(l_modules))]
                    l_model_wrap = [model_wrapper(l_modules[ind_module], l_optimizers[ind_module], seed, entropy_reg, selection_reg, l2_reg, device, dense_to_sparse, test_different_lr, steps_per_epoch, val_second_lr, momentum, weight_decay, tol_z_1, input_channel, type_pruning, generator, type_reset, method_pruning, threshold_restart, selection_lagrangian_reg, entropy_lagrangian_reg, l2_original_reg, l_modules_original[ind_module], type_function, test_mult_reset, test_reset_to_orignal, prune_bias, type_compute_sparsity, gamma) for ind_module in range(len(l_modules))]
                    l_modules_params = np.array([l_model_wrap[ind_model_wrap].get_n_z(test_grad=False) for ind_model_wrap in range(len(l_model_wrap))])
                    if test_one_layer_pruning and "resnet" in arch:
                        l_optimizers_downsamples = [initialize_optimizer(test_different_lr, l_downsamples[ind_module], optimizer_name, steps_per_epoch, lr, val_second_lr, momentum, weight_decay, selection_lagrangian_reg, entropy_lagrangian_reg) for ind_module in range(len(l_downsamples))]
                        l_model_wrap_downsamples = [model_wrapper(l_downsamples[ind_module], l_optimizers_downsamples[ind_module], seed, entropy_reg, selection_reg, l2_reg, device, dense_to_sparse, test_different_lr, steps_per_epoch, val_second_lr, momentum, weight_decay, tol_z_1, input_channel, type_pruning, generator, type_reset, method_pruning, threshold_restart, selection_lagrangian_reg, entropy_lagrangian_reg, l2_original_reg, l_downsamples_original[ind_module], type_function, test_mult_reset, test_reset_to_orignal, prune_bias, type_compute_sparsity, gamma) for ind_module in range(len(l_downsamples))]
                        l_modules_downsamples_params = np.array([l_model_wrap[ind_model_wrap].get_n_z(test_grad=False) for ind_model_wrap in range(len(l_model_wrap_downsamples))])

                    if test_prop_goal_sparsity:
                        max_sparsity = 0.98
                        if test_one_layer_pruning and "resnet" in arch:
                            l_modules_both_params = np.hstack([l_modules_params,l_modules_downsamples_params])
                        else:
                            l_modules_both_params = l_modules_params
                        alpha = n_params_original_z/(np.sum(l_modules_both_params**(3/2)))*goal_sparsity_used
                        l_sparsity_both = np.sqrt(l_modules_both_params)*alpha
                        l_params_both_final = (1-l_sparsity_both)*l_modules_both_params
                        l_params_extra = (1-0.98)*l_modules_both_params-l_params_both_final
                        ind_pb = np.where(l_params_extra>0)[0]
                        ind_ok = np.where(l_params_extra<=0)[0]
                        saved_ind_pb = np.zeros_like(ind_pb)
                        saved_ind_pb = saved_ind_pb[:0]
                        while len(ind_pb)>0:
                            ind_pb = np.hstack([saved_ind_pb, ind_pb])
                            ind_ok = np.array([i for i in range(len(l_modules_both_params)) if i not in ind_pb])
                            l_sparsity_both[ind_pb] = max_sparsity
                            new_alpha = (n_params_original_z/(np.sum(l_modules_both_params[ind_ok]**(3/2))))*(goal_sparsity_used-np.sum(l_modules_both_params[ind_pb]*l_sparsity_both[ind_pb]/n_params_original_z))
                            l_sparsity_both[ind_ok] = np.sqrt(l_modules_both_params[ind_ok])*new_alpha
                            l_params_both_final = ((1-l_sparsity_both)*l_modules_both_params)
                            l_params_extra = (1-0.98)*l_modules_both_params-l_params_both_final
                            saved_ind_pb = np.hstack([saved_ind_pb, ind_pb])
                            ind_pb = np.where(l_params_extra>0)[0]
                        l_sparsity_modules = l_sparsity_both[:len(l_modules_params)]
                        if test_one_layer_pruning and "resnet" in arch:
                            l_sparsity_downsamples = l_sparsity_both[len(l_modules_params):]                        

                    l_in_sample_loss = []
                    l_in_sample_loss_no_pen = []
                    l_validation_loss = []
                    l_in_sample_metric = []
                    l_validation_metric = []
                    l_times_epochs = []
                    l_lr = []
                    l_n_z = []
                    l_sparsity = []
                    l_sparsity_storage = []
                    l_n_params = []
                    best_ep = []
                    n_z_final = []
                    test_sparsity_reached = False
                    
                    acc_downsamples = 0
                    for ind_model_wrap in range(len(l_model_wrap)):
                        print(f"ind_model_wrap: {ind_model_wrap}")
                        model_wrap = l_model_wrap[ind_model_wrap]
                        model_wrap.ind_model_wrap = ind_model_wrap
                        if test_start_convex:
                            end_model = nn.Sequential(*l_modules_original[ind_model_wrap+1:])
                        else:
                            end_model = None
                        # if name_dataset=="imagenet":
                        #     # path_imagenet_current = "/home/gridsan/gafriat/imagenet_"+str(threshold_weights)+"_"+str(ind_model_wrap)
                        #     path_imagenet_original = "/home/gridsan/gafriat/imagenet_original_"+str(threshold_weights)+"_"+str(ind_model_wrap)
                        #     if not(test_distributed) or arguments.global_rank==0:
                        #         l_dir = os.listdir(IMAGENET_PATH+"/raw_train")
                        #         # if not(os.path.exists(path_imagenet_current)):
                        #         #     os.mkdir(path_imagenet_current)
                        #         # else:
                        #         #     shutil.rmtree(path_imagenet_current)
                        #         #     os.mkdir(path_imagenet_current)
                        #         if not(os.path.exists(path_imagenet_original)):
                        #             os.mkdir(path_imagenet_original)
                        #         # import ipdb;ipdb.set_trace()
                        #         # for dir_name in l_dir:
                        #         #     if not(os.path.exists(path_imagenet_original+"/"+dir_name)):
                        #         #         os.mkdir(path_imagenet_original+"/"+dir_name)
                        #             # os.mkdir(path_imagenet_current+"/"+dir_name)
                        #     # loader_train.dataset.dataset.path_imagenet_current = path_imagenet_current
                        #     loader_train.dataset.dataset.path_imagenet_original = path_imagenet_original
                        if model_wrap.model.test_save_initial_data:
                            copy_train_dataset = copy.deepcopy(loader_train.dataset)
                            copy_validation_dataset = copy.deepcopy(loader_val.dataset)
                            copy_validation_dataset.dataset = copy_train_dataset.dataset
                            if test_distributed:
                                loader_train_downsample = DataLoader(copy_train_dataset, batch_size=batch_size_dataset, shuffle=(train_sampler is None), generator=generator_loader, num_workers=num_workers, pin_memory=True, sampler=train_sampler)
                                loader_val_downsample = DataLoader(copy_validation_dataset, batch_size=batch_size_dataset, num_workers=num_workers, pin_memory=True)
                            else:
                                loader_train_downsample = DataLoader(copy_train_dataset, batch_size=batch_size_dataset, shuffle=True, generator=generator_loader, num_workers=num_workers, pin_memory=True)
                                loader_val_downsample = DataLoader(copy_validation_dataset, batch_size=batch_size_dataset, num_workers=num_workers, pin_memory=True)
                            if model_wrap.model.test_downsample:
                                model_wrap_downsample = l_model_wrap_downsamples.pop(0)
                                optimizer_downsample = l_optimizers_downsamples.pop(0)
                                original_module_downsample = l_downsamples_original.pop(0)
                                is_last_module_downsample = False
                                dataset_downsample = loader_train_downsample, loader_val_downsample, loader_test, scaler_y
                                if test_prop_goal_sparsity:
                                    goal_sparsity_sub_module = l_sparsity_downsamples[acc_downsamples]
                                    acc_downsamples += 1
                                else:
                                    goal_sparsity_sub_module = goal_sparsity_used
                                retraining_of_last_block = False
                                train_sub_modules(model_wrap_downsample, test_distributed, arguments, device, optimizer_downsample, original_module_downsample, is_last_module_downsample, name_study, arch, dataset_downsample, criterion, n_epochs, batch_size_dataset, test_early_stopping, trial, test_save_all_models, type_decay, gamma_lr_decay, T_max_cos, eta_min_cos, start_lr_decay, end_lr_decay, warmup_steps, type_of_task, test_compute_accurate_in_sample_loss, folder_saves, ind_repeat, patience, metric_early_stopping, period_milestones, goal_sparsity_sub_module, type_training, n_restart, num_workers, loss_func, name_dataset, n_train_kept, l_model_wrap, ind_model_wrap, l_in_sample_loss, l_in_sample_loss_no_pen, l_validation_loss, l_in_sample_metric, l_validation_metric, l_times_epochs, l_lr, l_n_z, l_sparsity, l_sparsity_storage, l_n_params, best_ep, n_z_final, test_normalized_sgd, pruning_rate_cte, lambda_loss, test_repeat_if_sparsity_not_reached, loss_last_block, retraining_of_last_block, copy_indices_train, copy_indices_val, test_adaptive_lr, patience_adaptive_lr, patience_freeze, test_wait_for_pruning, test_almost_sequential, tol_ent_reg, tol_sel_reg, n_incr_gradual_pruning, goal_sparsity_discrete, test_start_sparse_gpt, test_start_convex, type_pruning_schedule, end_model)
                        optimizer = l_optimizers[ind_model_wrap]
                        original_module = l_modules_original[ind_model_wrap]
                        is_last_module = ind_model_wrap==(len(l_model_wrap)-1)
                        if test_prop_goal_sparsity:
                            goal_sparsity_sub_module = l_sparsity_modules[ind_model_wrap]
                        else:
                            goal_sparsity_sub_module = goal_sparsity_used
                        if is_last_module and name_dataset in ["c4", "wikitext2", "ptb"] and len(list(model_wrap.model.children())) <= 1:
                            goal_sparsity_sub_module = 0.0
                            n_incr_gradual_pruning_sub = -1
                        else:
                            n_incr_gradual_pruning_sub = n_incr_gradual_pruning
                        # elif test_start_sparse_gpt:
                        #     model_wrap.model.eval()
                        #     model_wrap.model.to(device)
                        #     prune_spargegpt_block(model_wrap.model, loader_train, device, arguments.n_train_kept, 16, goal_sparsity_discrete, 0, 0, 0.01, 128, True, gamma)
                        #     model_wrap.model.train()
                        # if test_start_sparse_gpt:
                        #     goal_sparsity_sub_module = 0.0
                        retraining_of_last_block = False
                        if is_last_module and batch_size_dataset == 16:
                            batch_size_dataset = 1
                            if test_distributed:
                                loader_train = DataLoader(loader_train.dataset, batch_size=batch_size_dataset, shuffle=(train_sampler is None), generator=generator_loader, num_workers=num_workers, pin_memory=True, sampler=train_sampler)
                                loader_val = DataLoader(loader_val.dataset, batch_size=batch_size_dataset, num_workers=num_workers, pin_memory=True)
                                loader_test = DataLoader(loader_test.dataset, batch_size=batch_size_dataset, num_workers=num_workers, pin_memory=True)
                                print("loader created", flush = True)
                            else:
                                loader_train = DataLoader(loader_train.dataset, batch_size=batch_size_dataset, shuffle=True, generator=generator_loader, num_workers=num_workers, pin_memory=True)
                                loader_val = DataLoader(loader_val.dataset, batch_size=batch_size_dataset, num_workers=num_workers, pin_memory=True)
                                loader_test = DataLoader(loader_test.dataset, batch_size=batch_size_dataset, num_workers=num_workers, pin_memory=True)
                            dataset = loader_train, loader_val, loader_test, scaler_y
                        # Compute H
                        if "_H" in type_pruning:
                            model_wrap.model.eval()
                            model_wrap.model.to(device)
                            gpts = get_XXt(model_wrap, loader_train)
                            model_wrap.gpts = gpts
                            model_wrap.model.train()
                        train_sub_modules(model_wrap, test_distributed, arguments, device, optimizer, original_module, is_last_module, name_study, arch, dataset, criterion, n_epochs, batch_size_dataset, test_early_stopping, trial, test_save_all_models, type_decay, gamma_lr_decay, T_max_cos, eta_min_cos, start_lr_decay, end_lr_decay, warmup_steps, type_of_task, test_compute_accurate_in_sample_loss, folder_saves, ind_repeat, patience, metric_early_stopping, period_milestones, goal_sparsity_sub_module, type_training, n_restart, num_workers, loss_func, name_dataset, n_train_kept, l_model_wrap, ind_model_wrap, l_in_sample_loss, l_in_sample_loss_no_pen, l_validation_loss, l_in_sample_metric, l_validation_metric, l_times_epochs, l_lr, l_n_z, l_sparsity, l_sparsity_storage, l_n_params, best_ep, n_z_final, test_normalized_sgd, pruning_rate_cte, lambda_loss, test_repeat_if_sparsity_not_reached, loss_last_block, retraining_of_last_block, copy_indices_train, copy_indices_val, test_adaptive_lr, patience_adaptive_lr, patience_freeze, test_wait_for_pruning, test_almost_sequential, tol_ent_reg, tol_sel_reg, n_incr_gradual_pruning_sub, goal_sparsity_discrete, test_start_sparse_gpt, test_start_convex, type_pruning_schedule, end_model)
                        if is_last_module and loss_last_block=="layer_wise" and retrain_last_block:
                            loss_last_block_final = "mce"
                            retraining_of_last_block = True
                            new_optimizer = initialize_optimizer(test_different_lr, l_modules[ind_model_wrap], optimizer_name, steps_per_epoch, lr, val_second_lr, momentum, weight_decay, selection_lagrangian_reg, entropy_lagrangian_reg)
                            model_wrap.optimizer = new_optimizer
                            test_early_stopping_last_block = test_early_stopping
                            # test_early_stopping_last_block = 1
                            # loader_train, loader_val, loader_test, scaler_y = dataset
                            # loader_train.dataset.indices = copy_indices_train
                            # loader_val.dataset.indices = copy_indices_val
                            # dataset = loader_train, loader_val, loader_test, scaler_y
                            train_sub_modules(model_wrap, test_distributed, arguments, device, new_optimizer, original_module, is_last_module, name_study, arch, dataset, criterion, n_epochs, batch_size_dataset, test_early_stopping_last_block, trial, test_save_all_models, type_decay, gamma_lr_decay, T_max_cos, eta_min_cos, start_lr_decay, end_lr_decay, warmup_steps, type_of_task, test_compute_accurate_in_sample_loss, folder_saves, ind_repeat, patience, metric_early_stopping, period_milestones, goal_sparsity_sub_module, type_training, n_restart, num_workers, loss_func, name_dataset, n_train_kept, l_model_wrap, ind_model_wrap, l_in_sample_loss, l_in_sample_loss_no_pen, l_validation_loss, l_in_sample_metric, l_validation_metric, l_times_epochs, l_lr, l_n_z, l_sparsity, l_sparsity_storage, l_n_params, best_ep, n_z_final, test_normalized_sgd, pruning_rate_cte, lambda_loss, test_repeat_if_sparsity_not_reached, loss_last_block_final, retraining_of_last_block, copy_indices_train, copy_indices_val, test_adaptive_lr, patience_adaptive_lr, patience_freeze, test_wait_for_pruning, test_almost_sequential, tol_ent_reg, tol_sel_reg, n_incr_gradual_pruning_sub, goal_sparsity_discrete, test_start_sparse_gpt, test_start_convex, type_pruning_schedule, end_model)
                        
                        if model_wrap.model.test_add_initial_data:
                            if n_train_kept == -1:
                                loader_train.dataset.dataset.data += copy.deepcopy(loader_train_downsample.dataset.dataset.data)
                                loader_train.dataset.dataset.data_output_original += copy.deepcopy(loader_train_downsample.dataset.dataset.data_output_original)
                            else:
                                loader_train.dataset.dataset.dataset.data += copy.deepcopy(loader_train_downsample.dataset.dataset.dataset.data)
                                loader_train.dataset.dataset.dataset.data_output_original += copy.deepcopy(loader_train_downsample.dataset.dataset.dataset.data_output_original)
                            # loader_val.dataset.dataset.dataset.data += copy.deepcopy(loader_val_downsample.dataset.dataset.dataset.data)
                        # set_require_grad_rec(model_wrap.model, True)
                        # if test_distributed:
                        #     model_wrap.model.cuda(arguments.gpu)
                        #     model_wrap.model = torch.nn.parallel.DistributedDataParallel(model_wrap.model, device_ids=[arguments.gpu])
                        # else:
                        #     model_wrap.model.to(device)
                        # optimizer = l_optimizers[ind_model_wrap]
                        # mode_model_wrap = "ensemble"
                        # original_module = l_modules_original[ind_model_wrap]
                        # original_module.to(device)
                        # is_last_module = ind_model_wrap==(len(l_model_wrap)-1)
                        # module_training = True
                        # l_in_sample_loss_module, l_validation_loss_module, l_in_sample_metric_module, l_validation_metric_module, l_times_epochs_module, l_lr_module, l_n_z_module, l_n_params_module, best_model_module, best_ep_module, n_z_final_module, test_sparsity_reached = train_neural_network(name_study=name_study, name_model=name_model, model_wrapper=model_wrap, dataset=dataset, optimizer=optimizer, criterion=criterion, n_epochs=n_epochs, batch_size_dataset=batch_size_dataset, path_save=None, test_early_stopping=test_early_stopping, trial=trial, test_save_all_models=test_save_all_models, type_decay=type_decay, gamma_lr_decay=gamma_lr_decay, T_max_cos=T_max_cos, eta_min_cos=eta_min_cos, start_lr_decay=start_lr_decay, end_lr_decay=end_lr_decay, warmup_steps=warmup_steps, type_of_task=type_of_task, test_compute_accurate_in_sample_loss=test_compute_accurate_in_sample_loss, folder_saves=folder_saves, ind_repeat=ind_repeat, patience = patience, metric_early_stopping=metric_early_stopping, period_milestones=period_milestones, goal_sparsity=goal_sparsity, type_training=type_training, n_restart=n_restart, num_workers=num_workers, mode=mode_model_wrap, loss_func_and_model=(loss_func, original_module), is_last_module = is_last_module, module_training=module_training, name_dataset=name_dataset, n_train_kept=n_train_kept, n_rounds=len(l_model_wrap), current_round=ind_model_wrap)
                        # model_wrap.model.load_state_dict(best_model_module.state_dict())
                        # l_in_sample_loss += l_in_sample_loss_module
                        # l_validation_loss += l_validation_loss_module
                        # l_in_sample_metric += l_in_sample_metric_module
                        # l_validation_metric += l_validation_metric_module
                        # l_times_epochs += l_times_epochs_module
                        # l_lr += l_lr_module
                        # l_n_z += l_n_z_module
                        # l_n_params += l_n_params_module
                        # best_ep += [best_ep_module]
                        # n_z_final += n_z_final_module
                        # set_require_grad_rec(model_wrap.model, False)
                        # model_wrap.model.to("cpu")
                        # original_module.to("cpu")
                    n_z_final = np.sum(n_z_final)
                    # final_sparsity = 1-n_z_final/n_params_original_z
                    final_sparsity = main_model_wrapper.get_final_sparsity()
                    if final_sparsity>=goal_sparsity_used:
                        test_sparsity_reached = True
                    else:
                        test_sparsity_reached = False
                    main_model_wrapper.model.to(device)
                    best_model = main_model_wrapper.model
                    
                    # Recreate the original dataset/loader
                    n_train_kept = arguments.n_train_kept
                    train_val_dataset, test_dataset = get_dataset(name_dataset, name_dataset_path, n_train_kept, get_item_func, arch, seed, activation_fn, device)

                    if name_dataset in ["c4", "wikitext2", "ptb"]:
                        n_train_kept = -1
                        (train_val_dataset, train_val_attention_mask), (test_dataset, test_attention_mask) = train_val_dataset, test_dataset
                        if torch.sum(torch.abs(train_val_attention_mask-test_attention_mask)).item()!=0:
                            print("--- DIFFERENCE IN ATTENTION MASK ---")
                            import ipdb;ipdb.set_trace()
                    if not ("deit" in arch):
                        initialize_dataset(train_val_dataset, n_train_kept, name_dataset)
                        initialize_dataset(test_dataset, -1, name_dataset)
                    train_val_dataset.return_original = False
                    test_dataset.return_original = False

                    generator_split = torch.Generator().manual_seed(seed)
                    train_dataset, validation_dataset = random_split(train_val_dataset, [0.8, 0.2], generator=generator_split)
                    generator_loader = torch.Generator()

                    if seed != -1:
                        torch.random.manual_seed(seed)
                        generator_loader = generator_loader.manual_seed(seed)

                    if test_distributed:
                        loader_train = DataLoader(train_dataset, batch_size=batch_size_dataset, shuffle=(train_sampler is None), generator=generator_loader, num_workers=num_workers, pin_memory=True, sampler=train_sampler)
                        loader_val = DataLoader(validation_dataset, batch_size=batch_size_dataset, num_workers=num_workers, pin_memory=True)
                        loader_test = DataLoader(test_dataset, batch_size=batch_size_dataset, num_workers=num_workers, pin_memory=True)
                        print("loader created", flush = True)
                    else:
                        loader_train = DataLoader(train_dataset, batch_size=batch_size_dataset, shuffle=True, generator=generator_loader, num_workers=num_workers, pin_memory=True)
                        loader_val = DataLoader(validation_dataset, batch_size=batch_size_dataset, num_workers=num_workers, pin_memory=True)
                        loader_test = DataLoader(test_dataset, batch_size=batch_size_dataset, num_workers=num_workers, pin_memory=True)
                    
                    if test_load_data_first and not("deit" in arch):
                        print("Load data in memory first...", flush = True)
                        if test_almost_sequential==1:
                            test_update_original = False
                        else:
                            test_update_original = True
                        load_dataset_in_memory(loader_train, loader_val, n_train_kept, test_update_original)
                        print("Done!", flush = True)

                    # Process the results
                    if name_dataset in ["c4", "wikitext2", "ptb"]:
                        model_wrap_used = l_model_wrap
                    else:
                        model_wrap_used = model_wrap
                    d_results, best_val_metric, best_model, dict_list = process_results_neural_network(l_in_sample_loss, l_in_sample_loss_no_pen, l_validation_loss, l_in_sample_metric, l_validation_metric, l_times_epochs, l_lr, l_n_z, l_sparsity, l_sparsity_storage, l_n_params, best_model, loader_train, loader_val, loader_test, type_of_task, model_wrap_used, scaler_y, metric_name, best_ep, n_z_final, test_sparsity_reached, name_dataset, n_train_kept, copy_indices_train, copy_indices_val)
                    d_results["n_params_original_z"] = int(main_model_wrapper.n_params_original_z)
                else:
                    model.to(device)
                    optimizer = initialize_optimizer(test_different_lr, model, optimizer_name, steps_per_epoch, lr, val_second_lr, momentum, weight_decay, selection_lagrangian_reg, entropy_lagrangian_reg)
                    model_wrap = model_wrapper(model, optimizer, seed, entropy_reg, selection_reg, l2_reg, device, dense_to_sparse, test_different_lr, steps_per_epoch, val_second_lr, momentum, weight_decay, tol_z_1, input_channel, type_pruning, generator, type_reset, method_pruning, threshold_restart, selection_lagrangian_reg, entropy_lagrangian_reg, l2_original_reg, original_model, type_function, test_mult_reset, test_reset_to_orignal, prune_bias, type_compute_sparsity, gamma)
                    model_wrap.initialize_pruning()
                    is_last_module = True
                    module_training = False
                    if loss_func == "layer_wise":
                        original_model.to(device)
                    retraining_of_last_block = False
                    l_in_sample_loss, l_in_sample_loss_no_pen, l_validation_loss, l_in_sample_metric, l_validation_metric, l_times_epochs, l_lr, l_n_z, l_sparsity, l_sparsity_storage, l_n_params, best_model, best_ep, n_z_final, test_sparsity_reached = train_neural_network(name_study=name_study, name_model=arch, model_wrapper=model_wrap, dataset=dataset, optimizer=optimizer, criterion=criterion, n_epochs=n_epochs, batch_size_dataset=batch_size_dataset, path_save=None, test_early_stopping=test_early_stopping, trial=trial, test_save_all_models=test_save_all_models, type_decay=type_decay, gamma_lr_decay=gamma_lr_decay, T_max_cos=T_max_cos, eta_min_cos=eta_min_cos, start_lr_decay=start_lr_decay, end_lr_decay=end_lr_decay, warmup_steps=warmup_steps, type_of_task=type_of_task, test_compute_accurate_in_sample_loss=test_compute_accurate_in_sample_loss, folder_saves=folder_saves, ind_repeat=ind_repeat, patience = patience, metric_early_stopping=metric_early_stopping, period_milestones=period_milestones, goal_sparsity=goal_sparsity_used, type_training=type_training, n_restart=n_restart, num_workers=num_workers, mode=mode, loss_func_and_model=(loss_func, original_model), is_last_module = is_last_module, module_training=module_training, name_dataset=name_dataset, n_train_kept=n_train_kept, test_normalized_sgd=test_normalized_sgd, pruning_rate_cte=pruning_rate_cte, lambda_loss=lambda_loss, test_repeat_if_sparsity_not_reached=test_repeat_if_sparsity_not_reached, loss_last_block=loss_last_block, retraining_of_last_block=retraining_of_last_block, copy_indices_train=copy_indices_train, copy_indices_val=copy_indices_val, test_adaptive_lr=test_adaptive_lr, patience_adaptive_lr=patience_adaptive_lr, patience_freeze=patience_freeze, test_wait_for_pruning=test_wait_for_pruning, test_almost_sequential=test_almost_sequential, tol_ent_reg=tol_ent_reg, tol_sel_reg=tol_sel_reg)
                    model_wrap.model.load_state_dict(best_model.state_dict())
                    # l_sparsity = 1-l_n_z/model_wrap.n_params_original_z
                    # l_sparsity_storage = 1-l_n_params/model_wrap.n_params_original_z
                    d_results, best_val_metric, best_model, dict_list = process_results_neural_network(l_in_sample_loss, l_in_sample_loss_no_pen, l_validation_loss, l_in_sample_metric, l_validation_metric, l_times_epochs, l_lr, l_n_z, l_sparsity, l_sparsity_storage, l_n_params, best_model, loader_train, loader_val, loader_test, type_of_task, model_wrap, scaler_y, metric_name, best_ep, n_z_final, test_sparsity_reached, name_dataset, n_train_kept, copy_indices_train, copy_indices_val)
                    d_results["n_params_original_z"] = int(model_wrap.n_params_original_z)
                    final_sparsity = model_wrap.get_sparsity()
                if type_of_task == "classification":
                    if metric_best_model == "acc":
                        best_val_metric = best_val_metric[0]
                    elif metric_best_model == "auc":
                        best_val_metric = best_val_metric[1]
                    else:
                        print("Error: wrong metric to pick the best model, either acc or auc", flush=True)
                d_results["sparsity"] = final_sparsity
                time_training = time.time()-time_before_training
                d_results["time_training"] = time_training
                best_val_metric_avg += best_val_metric/n_repeat
                if ((best_val_metric>=best_val_metric_best) and (type_of_task=="classification")) or ((best_val_metric<=best_val_metric_best) and (type_of_task=="regression")):
                    best_val_metric_best = best_val_metric
                best_model.to(device)

                #Saving of the results
                dict_params = vars(arguments)
                dict_params["lr"] = lr
                
                save_results(dict_list=dict_list, best_model=best_model, d_results=d_results, name_study=name_study, dict_params=dict_params, trial=trial, folder_saves=folder_saves, ind_repeat=ind_repeat)
            delete_models(name_study, folder_saves, n_repeat, type_of_task)
            return best_val_metric_avg

        if save_study_done!=None:
            conduct_fine_tuning(objective, name_study, timeout, n_trials, save_study_done, folder_saves=folder_saves, type_of_task=type_of_task)
        else:
            conduct_fine_tuning(objective, name_study, timeout, n_trials, folder_saves=folder_saves, type_of_task=type_of_task)

    delete_models(name_study, folder_saves, n_repeat, type_of_task, delete_pass=True)

    try:
        list_dir = os.listdir(folder_saves+"/study_"+name_study)
        trial_folders = np.array([x for x in list_dir if "trial" in x])
        path_model = folder_saves+"/study_"+name_study+"/"+trial_folders[0]
        new_path_model =  folder_saves+"/study_"+name_study+"/best_trial"
        os.rename(path_model, new_path_model)
    except:
        pass

    if test_save_all_models:
        print("Gathering the models", flush=True)
        for ind_repeat in range(n_repeat):
            gather_list_models(name_study, ind_repeat, folder_saves, device)
        print("Done", flush=True)

    in_sample_metric_avg = 0
    validation_metric_avg = 0
    test_metric_avg = 0

    sparsity_avg = np.array([])

    for ind_repeat in range(n_repeat):
        dict_params, in_sample_metric, validation_metric, test_metric, best_ep, time_training, n_z, n_params, sparsity = read_results(name_study, ind_repeat, type_of_task =type_of_task, folder_saves=folder_saves)
        in_sample_metric_avg += in_sample_metric/n_repeat
        validation_metric_avg += validation_metric/n_repeat
        test_metric_avg += test_metric/n_repeat
        sparsity_avg = np.hstack([sparsity_avg, sparsity])

    if n_params_original_z==0:
        model, criterion, modules_to_prune = get_model(arch, seed, pretrained=pretrained, with_z=True, gamma=gamma, prune_bias=prune_bias, activation_fn=activation_fn)
        n_params_original_z = np.sum([np.prod(x[1].shape) for x in  model.named_parameters() if "_z" in x[0]])
        n_params_original = np.sum([np.prod(x[1].shape) for x in  model.named_parameters() if "_z" not in x[0]])

    sparsity_avg = np.nanmean(sparsity_avg)

    print("Best params =", dict_params, flush=True)
    print("Best epoch = "+ str(best_ep)+"/"+str(n_epochs-1), flush=True)
    print("  ", flush=True)
    if type_of_task=="regression":
        print("In-sample mse =", in_sample_metric_avg, flush=True)
        print("Validation mse =", validation_metric_avg, flush=True)
        print("Out-of-sample mse =", test_metric_avg, flush=True)
    elif type_of_task=="classification":
        if name_dataset in ["c4", "wikitext2", "ptb"]:
            evaluation_metric = "perplexity"
        else:
            evaluation_metric = "accuracy"
            in_sample_metric_avg[0]*=100
            validation_metric_avg[0]*=100
            test_metric_avg[0]*=100
        print(f"In-sample {evaluation_metric} =", in_sample_metric_avg[0],"%", flush=True)
        print(f"Validation {evaluation_metric} =", validation_metric_avg[0],"%", flush=True)
        print(f"Out-of-sample {evaluation_metric} =", test_metric_avg[0], "%", flush=True)
        print("In-sample auc =", in_sample_metric_avg[1], flush=True)
        print("Validation auc =", validation_metric_avg[1], flush=True)
        print("Out-of-sample auc =", test_metric_avg[1], flush=True)
    print("Sparsity =", sparsity_avg, flush=True)
    print("Training time = ", time_training, flush=True)
    print("TOTAL TIME = ", time.time()-start_time, flush=True)
# %%
