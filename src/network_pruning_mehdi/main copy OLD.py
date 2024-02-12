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
parser.add_argument('--loss_func', type=str, default = "classic",
                    help='either classic (mse/cross entropy) or layer_wise')
parser.add_argument('--type_reset', type=str, default = "layer_wise",
                    help='how to select the weights to reset (either layer_wise or ensemble)')
# parser.add_argument('--layer_wise_reg', type=float, default = 0.0,
#                     help='penalization for the layer wise reconstruction')

arguments, unknown = parser.parse_known_args()

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
    mode = arguments.mode
    loss_func = arguments.loss_func
    type_reset = arguments.type_reset

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    # required for the distributed training
    if test_distributed:
        import torch.distributed as dist
        if "WORLD_SIZE" in os.environ:
            print("-- WORLD_SIZE in os.environ --")
            arguments.world_size = int(os.environ["WORLD_SIZE"])
        else:
            print("-- WORLD_SIZE not in os.environ --")
            arguments.world_size = 1
        arguments.distributed = arguments.world_size > 1

        if arguments.distributed:
            if 'SLURM_PROCID' in os.environ: # for slurm scheduler
                print("--- HOURAA ---")
                arguments.rank = int(os.environ['SLURM_PROCID'])
                # TO DELETE LATER MAYEB
                if arguments.rank in [0,2]:
                    arguments.gpu = 0
                else:
                    arguments.gpu = 1
                arguments.gpu = arguments.local_rank 
                # END
                #arguments.gpu = arguments.rank % torch.cuda.device_count()
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
        print('Using device ', device)

        ngpus_per_node = torch.cuda.device_count()

        torch.backends.cudnn.benchmark = True
    else:
        if device == "None":
            device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
    
    train_val_dataset, test_dataset = get_dataset(name_dataset, name_dataset_path)
    generator_split = torch.Generator().manual_seed(seed)
    train_dataset, validation_dataset = random_split(train_val_dataset, [0.8, 0.2], generator=generator_split)
    
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
                   test_distributed,
                   mode,
                   loss_func,
                   type_reset)

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
        def objective(trial):
            try:
                global dataset, n_params_original_z
            except:
                pass
            if learning_rate!=-1:
                lr = learning_rate
            else:
                lr = trial.suggest_float("lr", min_lr, max_lr, log = True)
            print('Parsed arguments:', arguments)
            print("lr:", lr)
            if type_of_task== "classification":
                best_val_metric_best = -np.inf
                metric_name = "accuracy"
            elif type_of_task=="regression":
                best_val_metric_best = np.inf
                metric_name = "mse"

            best_val_metric_avg = 0                    

            for ind_repeat in range(n_repeat):
                signal.alarm(0)
                signal.alarm(24*60*60)
                print("Repeat", ind_repeat+1, "out of", n_repeat, flush=True)
                # Model initialization
                model, criterion, modules_to_prune = get_model(arch, seed, pretrained=pretrained, with_z=True)
                name_model = model.__str__().lower()
                if loss_func == "layer_wise":
                    original_model, criterion, modules_to_prune = get_model(arch, seed, pretrained=pretrained, with_z=False)
                else:
                    original_model = None
                print("Model intialized", flush = True)
                print("Test distributed:",test_distributed, flush = True)
                if test_distributed:
                    if arguments.distributed:
                        print("RANK =", arguments.rank, flush = True)
                        dist.init_process_group(backend='nccl', init_method='env://',
                                                world_size=arguments.world_size, rank=arguments.rank)
                        print("DDP initialized", flush = True)
                        torch.backends.cudnn.benchmark = True

                        if arguments.rank!=0:
                            def print_pass(*arguments):
                                pass
                            #builtins.print = print_pass ##Uncomment to block all other ranks from printing

                        torch.cuda.set_device(arguments.gpu)
                        model.cuda(arguments.gpu)
                        #model.to(f'cuda:{arguments.gpu}')
                        print("CURRENT GPU:", arguments.gpu)
                        print(arguments.rank, arguments.gpu)
                        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[arguments.gpu])
                        print("model DDP initialized", flush = True)
                        model_without_ddp = model.module
                        #modules_to_prune = ["module." + x for x in modules_to_prune]
                        modules_to_prune = [x for x in modules_to_prune]
                    else:
                        # model = model.to(device)
                        model_without_ddp = model
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
                scaler_y = None
                #dataset = train_dataset, validation_dataset, test_dataset, scaler_y
                dataset = loader_train, loader_val, loader_test, scaler_y

                if test_distributed:
                    if name_dataset == "imagenet" and torch.cuda.device_count()>1:
                        print('Using DistributedDataParallel with',torch.cuda.device_count(),'GPUs', flush = True)
                        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[arguments.gpu])
                model.eval()
                if test_distributed and arguments.rank == 0:
                    print("Evaluating the dense model", flush = True)
                
                # if not(test_distributed):
                #     model_without_ddp = model
                # if not(test_distributed) or arguments.rank == 0:
                #     start_test = time.time()
                #     dense_acc = compute_acc(model_without_ddp,loader_test,device=device)
                #     time_test = time.time() - start_test
                #     print('Dense test accuracy', dense_acc,' computation time : ', time_test)
                # print("Evaluation done", flush = True)
                if n_params_original_z==0:
                    n_params_original_z = np.sum([np.prod(x[1].shape) for x in  model.named_parameters() if "_z" in x[0]])
                if path_load_weights!="":
                    if model_loaded:
                        model.load_state_dict(model_state_dict)
                    print("The weights have been loaded")
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
                    if "resnet" in name_model:
                        original_model.to(device)
                        original_model.train()
                        #dense_train_acc = compute_acc(original_model, loader_train, device=device)
                        model.train()
                        for ind_batch_norm in range(len(l_modules)):
                            l_modules[ind_batch_norm].running_mean = l_modules_original[ind_batch_norm].running_mean
                            l_modules[ind_batch_norm].running_var = l_modules_original[ind_batch_norm].running_var                        
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
                # l_in_sample_loss, l_validation_loss, l_in_sample_metric, l_validation_metric, l_times_epochs, l_lr, l_n_z, l_sparsity, l_n_params, best_model, best_ep, n_z_final, test_sparsity_reached = train_neural_network(name_study=name_study, name_model=name_model, model_wrapper=model_wrap, dataset=dataset, optimizer=optimizer, criterion=criterion, n_epochs=n_epochs, batch_size_dataset=batch_size_dataset, path_save=None, test_early_stopping=test_early_stopping, trial=trial, test_save_all_models=test_save_all_models, type_decay=type_decay, gamma_lr_decay=gamma_lr_decay, T_max_cos=T_max_cos, eta_min_cos=eta_min_cos, start_lr_decay=start_lr_decay, end_lr_decay=end_lr_decay, warmup_steps=warmup_steps, type_of_task=type_of_task, test_compute_accurate_in_sample_loss=test_compute_accurate_in_sample_loss, folder_saves=folder_saves, ind_repeat=ind_repeat, patience = patience, metric_early_stopping=metric_early_stopping, period_milestones=period_milestones, goal_sparsity=goal_sparsity, type_training=type_training, n_restart=n_restart, num_workers=num_workers, mode=mode_model_wrap, loss_func_and_model=(loss_func, original_module), is_last_module = is_last_module, module_training=module_training)
                # # END

                if loss_func == "layer_wise" and mode == "layer_wise":
                    set_require_grad_rec(model, False)
                    l_modules_before_threshold = get_modules_layer_wise_rec(model, [])
                    l_modules_before_threshold_original = get_modules_layer_wise_rec(original_model, [])
                    if "resnet" in name_model:
                        l_modules_before_threshold = l_modules_before_threshold[:-1]+[View()]+l_modules_before_threshold[-1:]
                        l_modules_before_threshold_original = l_modules_before_threshold_original[:-1]+[View()]+l_modules_before_threshold_original[-1:]
                    threshold_weight = 15000
                    l_modules = process_l_modules_w_threshold(l_modules_before_threshold, threshold_weight)
                    l_modules_original = process_l_modules_w_threshold(l_modules_before_threshold_original, threshold_weight)
                    l_optimizers = [initialize_optimizer(test_different_lr, l_modules[ind_module], optimizer_name, steps_per_epoch, lr, val_second_lr, momentum, weight_decay) for ind_module in range(len(l_modules))]
                    l_model_wrap = [model_wrapper(l_modules[ind_module], l_optimizers[ind_module], seed, entropy_reg, selection_reg, l2_reg, device, dense_to_sparse, test_different_lr, steps_per_epoch, val_second_lr, momentum, weight_decay, tol_z_1, input_channel, type_pruning, generator, type_reset) for ind_module in range(len(l_modules))]
                    main_model_wrapper = model_wrapper(model, None, seed, entropy_reg, selection_reg, l2_reg, device, dense_to_sparse, test_different_lr, steps_per_epoch, val_second_lr, momentum, weight_decay, tol_z_1, input_channel, type_pruning, generator, type_reset)
                    main_model_wrapper.initialize_pruning()

                    l_in_sample_loss = []
                    l_validation_loss = []
                    l_in_sample_metric = []
                    l_validation_metric = []
                    l_times_epochs = []
                    l_lr = []
                    l_n_z = []
                    l_n_params = []
                    best_ep = []
                    n_z_final = 0
                    test_sparsity_reached = False

                    for ind_model_wrap in range(len(l_model_wrap)):
                        model_wrap = l_model_wrap[ind_model_wrap]
                        set_require_grad_rec(model_wrap.model, True)
                        model_wrap.model.to(device)
                        optimizer = l_optimizers[ind_model_wrap]
                        mode_model_wrap = "ensemble"
                        original_module = l_modules_original[ind_model_wrap]
                        original_module.to(device)
                        is_last_module = ind_model_wrap==(len(l_model_wrap)-1)
                        module_training = True
                        l_in_sample_loss_module, l_validation_loss_module, l_in_sample_metric_module, l_validation_metric_module, l_times_epochs_module, l_lr_module, l_n_z_module, l_n_params_module, best_model_module, best_ep_module, n_z_final_module, test_sparsity_reached = train_neural_network(name_study=name_study, name_model=name_model, model_wrapper=model_wrap, dataset=dataset, optimizer=optimizer, criterion=criterion, n_epochs=n_epochs, batch_size_dataset=batch_size_dataset, path_save=None, test_early_stopping=test_early_stopping, trial=trial, test_save_all_models=test_save_all_models, type_decay=type_decay, gamma_lr_decay=gamma_lr_decay, T_max_cos=T_max_cos, eta_min_cos=eta_min_cos, start_lr_decay=start_lr_decay, end_lr_decay=end_lr_decay, warmup_steps=warmup_steps, type_of_task=type_of_task, test_compute_accurate_in_sample_loss=test_compute_accurate_in_sample_loss, folder_saves=folder_saves, ind_repeat=ind_repeat, patience = patience, metric_early_stopping=metric_early_stopping, period_milestones=period_milestones, goal_sparsity=goal_sparsity, type_training=type_training, n_restart=n_restart, num_workers=num_workers, mode=mode_model_wrap, loss_func_and_model=(loss_func, original_module), is_last_module = is_last_module, module_training=module_training)
                        model_wrap.model.load_state_dict(best_model_module.state_dict())
                        l_in_sample_loss += l_in_sample_loss_module
                        l_validation_loss += l_validation_loss_module
                        l_in_sample_metric += l_in_sample_metric_module
                        l_validation_metric += l_validation_metric_module
                        l_times_epochs += l_times_epochs_module
                        l_lr += l_lr_module
                        l_n_z += l_n_z_module
                        l_n_params += l_n_params_module
                        best_ep += [best_ep_module]
                        n_z_final += n_z_final_module
                        set_require_grad_rec(model_wrap.model, False)
                        model_wrap.model.to("cpu")
                        original_module.to("cpu")
                    final_sparsity = 1-n_z_final/n_params_original_z
                    test_sparsity_reached = final_sparsity>=goal_sparsity
                    best_model = main_model_wrapper.model
                    d_results, best_val_metric, best_model, dict_list = process_results_neural_network(l_in_sample_loss, l_validation_loss, l_in_sample_metric, l_validation_metric, l_times_epochs, l_lr, l_n_z, l_n_params, best_model, loader_train, loader_val, loader_test, type_of_task, model_wrap, scaler_y, metric_name, best_ep, n_z_final, test_sparsity_reached)
                else:
                    model.to(device)
                    optimizer = initialize_optimizer(test_different_lr, model, optimizer_name, steps_per_epoch, lr, val_second_lr, momentum, weight_decay)
                    model_wrap = model_wrapper(model, optimizer, seed, entropy_reg, selection_reg, l2_reg, device, dense_to_sparse, test_different_lr, steps_per_epoch, val_second_lr, momentum, weight_decay, tol_z_1, input_channel, type_pruning, generator, type_reset)
                    model_wrap.initialize_pruning()
                    is_last_module = True
                    module_training = False
                    if loss_func == "layer_wise":
                        original_model.to(device)
                    l_in_sample_loss, l_validation_loss, l_in_sample_metric, l_validation_metric, l_times_epochs, l_lr, l_n_z, l_sparsity, l_n_params, best_model, best_ep, n_z_final, test_sparsity_reached = train_neural_network(name_study=name_study, name_model=name_model, model_wrapper=model_wrap, dataset=dataset, optimizer=optimizer, criterion=criterion, n_epochs=n_epochs, batch_size_dataset=batch_size_dataset, path_save=None, test_early_stopping=test_early_stopping, trial=trial, test_save_all_models=test_save_all_models, type_decay=type_decay, gamma_lr_decay=gamma_lr_decay, T_max_cos=T_max_cos, eta_min_cos=eta_min_cos, start_lr_decay=start_lr_decay, end_lr_decay=end_lr_decay, warmup_steps=warmup_steps, type_of_task=type_of_task, test_compute_accurate_in_sample_loss=test_compute_accurate_in_sample_loss, folder_saves=folder_saves, ind_repeat=ind_repeat, patience = patience, metric_early_stopping=metric_early_stopping, period_milestones=period_milestones, goal_sparsity=goal_sparsity, type_training=type_training, n_restart=n_restart, num_workers=num_workers, mode=mode, loss_func_and_model=(loss_func, original_model), is_last_module = is_last_module, module_training=module_training)

                    # Recreate the original dataset/loader
                    train_val_dataset, test_dataset = get_dataset(name_dataset, name_dataset_path)
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

                    # Process the results
                    d_results, best_val_metric, best_model, dict_list = process_results_neural_network(l_in_sample_loss, l_validation_loss, l_in_sample_metric, l_validation_metric, l_times_epochs, l_lr, l_n_z, l_n_params, best_model, loader_train, loader_val, loader_test, type_of_task, model_wrap, scaler_y, metric_name, best_ep, n_z_final, test_sparsity_reached)
                if type_of_task == "classification":
                    if metric_best_model == "acc":
                        best_val_metric = best_val_metric[0]
                    elif metric_best_model == "auc":
                        best_val_metric = best_val_metric[1]
                    else:
                        print("Error: wrong metric to pick the best model, either acc or auc")
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
        print("Gathering the models")
        for ind_repeat in range(n_repeat):
            gather_list_models(name_study, ind_repeat, folder_saves, device)
        print("Done")

    in_sample_metric_avg = 0
    validation_metric_avg = 0
    test_metric_avg = 0
    n_params_used = 0

    for ind_repeat in range(n_repeat):
        dict_params, in_sample_metric, validation_metric, test_metric, best_ep, time_training, n_z, n_params, sparsity = read_results(name_study, ind_repeat, type_of_task =type_of_task, folder_saves=folder_saves)
        in_sample_metric_avg += in_sample_metric/n_repeat
        validation_metric_avg += validation_metric/n_repeat
        test_metric_avg += test_metric/n_repeat
        n_params_used += n_z/n_repeat

        if n_params_original_z==0:
            model, criterion, modules_to_prune = get_model(arch, seed, pretrained=pretrained, with_z=False)
            n_params_original_z = np.sum([np.prod(x.shape) for x in  model.parameters()])

    
    sparsity = 1 - n_params_used/n_params_original_z

    print("Best params =", dict_params)
    print("Best epoch = "+ str(best_ep)+"/"+str(n_epochs-1))
    print("  ")
    if type_of_task=="regression":
        print("In-sample mse =", in_sample_metric_avg)
        print("Validation mse =", validation_metric_avg)
        print("Out-of-sample mse =", test_metric_avg)
    elif type_of_task=="classification":
        print("In-sample accuracy =", 100*in_sample_metric_avg[0],"%")
        print("Validation accuracy =", 100*validation_metric_avg[0],"%")
        print("Out-of-sample accuracy =", 100*test_metric_avg[0], "%")
        print("In-sample auc =", in_sample_metric_avg[1])
        print("Validation auc =", validation_metric_avg[1])
        print("Out-of-sample auc =", test_metric_avg[1])
    print("Sparsity =", sparsity)
    print("Training time = ", time_training)
    print("TOTAL TIME = ", time.time()-start_time)
# %%
