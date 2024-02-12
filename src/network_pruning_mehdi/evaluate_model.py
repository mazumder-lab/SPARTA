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
parser.add_argument('--name_dataset', type=str, default = "mnist",
                    help='name of the dataset')
parser.add_argument('--n_train_kept', type=int, default = -1,
                    help='number of training samples kept')
parser.add_argument('--path_model', type=str, default = "",
                    help='path of the saved model to evaluate')
parser.add_argument('--type_of_task', type=str, default = "classification",
                    help='either classification or regression')
parser.add_argument('--device', type=str, default = "None",
                    help='if None, then the best possible device will be used')
parser.add_argument('--seed', type=int, default = 0,
                    help='seed for the training')
parser.add_argument('--batch_size_dataset', type=int, default = 128,
                    help='batch size for the dataset')
parser.add_argument('--gamma', type=float, default = 1.0,
                    help='gamma for SmoothStep')
parser.add_argument('--n_repeat', type=int, default = 1,
                    help='number of times we repeat the experiment for a given set of hyperparameters')
parser.add_argument('--pretrained', type=lambda x: (str(x).lower() == 'true'), default=True)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--test_load_data_first', type=int, default = 1,
                    help='If set to 1, the data is loaded in memory before training starts')
parser.add_argument('--test_one_layer_pruning', type=int, default = 0,
                    help='If set to 1, then threshold_weights is ignored. Each layer is pruned one by one (no block is created)')
parser.add_argument('--threshold_weights', type=int, default = 150000,
                    help='we keep adding layers together until we reach this number of weights in a block of layers')
parser.add_argument('--type_function', type=str, default = "smoothstep",
                    help='Either smoothstep or sigmoid.')

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

    # signal.signal(signal.SIGVTA-1M, lambda signum, frame: print("\n--- Time is over ---"))

    print('Parsed arguments:', arguments, flush=True)
    # Previous hyperparameters
    arch = arguments.arch
    name_dataset = arguments.name_dataset
    n_train_kept = arguments.n_train_kept
    path_model = arguments.path_model
    type_of_task = arguments.type_of_task
    device = arguments.device
    seed = arguments.seed
    batch_size_dataset = arguments.batch_size_dataset
    gamma = arguments.gamma
    n_repeat = arguments.n_repeat
    pretrained = arguments.pretrained
    num_workers = arguments.num_workers
    test_load_data_first = arguments.test_load_data_first
    test_one_layer_pruning = arguments.test_one_layer_pruning
    threshold_weights = arguments.threshold_weights
    type_function = arguments.type_function

    device_name = "cuda" if torch.cuda.is_available() else "cpu"

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

    train_val_dataset, test_dataset = get_dataset(name_dataset, name_dataset_path, n_train_kept, get_item_func, arch, seed)
    
    if name_dataset in ["c4", "wikitext2", "ptb"]:
        n_train_kept = -1
        (train_val_dataset, train_val_attention_mask), (test_dataset, test_attention_mask) = train_val_dataset, test_dataset
        if torch.sum(torch.abs(train_val_attention_mask-test_attention_mask)).item()!=0:
            print("--- DIFFERENCE IN ATTENTION MASK ---")
            import ipdb;ipdb.set_trace()
    initialize_dataset(train_val_dataset, n_train_kept, name_dataset)
    initialize_dataset(test_dataset, -1, name_dataset)
    train_val_dataset.return_original = False
    test_dataset.return_original = False


    generator_split = torch.Generator().manual_seed(seed)
    train_dataset, validation_dataset = random_split(train_val_dataset, [0.8, 0.2], generator=generator_split)
    copy_indices_train = copy.deepcopy(train_dataset.indices)
    copy_indices_val = copy.deepcopy(validation_dataset.indices)
    
    print("--- Device =", device, "---", flush=True)

    n_samples = len(train_dataset)

    if batch_size_dataset == -1:
        batch_size_dataset = n_samples
    
    test_deja_train = False
    n_trials_done = 0
    save_study_done = None

    model, criterion, modules_to_prune = get_model(arch, seed, pretrained=pretrained, with_z=True, gamma=gamma, prune_bias=False)
    name_model = model.__str__().lower()

    if name_dataset in ["c4", "wikitext2", "ptb"]:
        model.float()

    #path_model = "/home/gridsan/gafriat/projects/network_pruning/Saves_test_opt_nov_27/study_opt-125m_lr_0.001_adpt_5_0.001_1_c4_128_norm_SGD_smst_n_epochs_200_aft_pru_npb_lw_loss_lw_1500000_res_ens_gam_1_sel_0.001_ent_0.001_l2_0.001_wd_1e-05_mom_0.9_cte_sch_0.1_auc_0.5_prun_seed_50_diff_lr_ll_100_tr_llb_lw_rt/best_trial/repeat_0/model"
    #path_model = "/home/gridsan/gafriat/projects/network_pruning/Saves_opt_dec_18/study_opt-125m_sgpt_lr_0.05_1_c4_128_norm_SGD_smst_n_ep_200_af_pru_frz_10_npb_es_p_10_val_loss_lw_loss_lw_res_ens_te_0.001_gam_1_sel_0.001_ent_0.001_l2_0.001_wd_1e-05_mom_0.9_cte_sch_0.1_auc_0.5_prun_gpe_10_seed_50_ll_100_tr_llb_lw_rt/best_trial/repeat_0/model"
    path_model = "/home/gridsan/gafriat/projects/network_pruning/Saves_opt_dec_19/study_opt-125m_sgpt_lr_0.001_adpt_5_1_c4_128_norm_SGD_smst_n_ep_200_af_pru_frz_10_npb_es_p_10_val_loss_lw_loss_lw_one_l_res_ens_te_0.001_gam_1_sel_0.001_ent_0.001_l2_0.001_wd_1e-05_mom_0.9_cte_sch_0.1_auc_0.5_prun_seed_50_ll_100_tr_llb_lw_rt/best_trial/repeat_0/model"
    
    new_weights = torch.load(path_model)
    import ipdb;ipdb.set_trace()
    # l_weights = list(new_weights.items())
    # new_l_weights = []
    # for i in range(len(l_weights)):
    #     weights_layer = l_weights[i]
    #     name_weights = weights_layer[0]
    #     values_weights = weights_layer[1]
    #     if not("_z" in name_weights):
    #         new_l_weights.append((name_weights,values_weights))
    #     else:
    #         old_weights_layer = new_l_weights.pop()
    #         mask = compute_z_from_tensor(values_weights, 1.0, "smoothstep")
    #         new_l_weights.append((old_weights_layer[0],old_weights_layer[1]*mask))
    # new_weights = OrderedDict(new_l_weights)
    # model.float()
    model.load_state_dict(new_weights)
    
    generator_loader = torch.Generator()
    if seed != -1:
        torch.random.manual_seed(seed)
        generator_loader = generator_loader.manual_seed(seed)

    loader_train = DataLoader(train_dataset, batch_size=batch_size_dataset, shuffle=True, generator=generator_loader, num_workers=num_workers, pin_memory=True)
    loader_val = DataLoader(validation_dataset, batch_size=batch_size_dataset, num_workers=num_workers, pin_memory=True)
    loader_test = DataLoader(test_dataset, batch_size=batch_size_dataset, num_workers=num_workers, pin_memory=True)
    if test_load_data_first:
        print("Load data in memory first...", flush = True)
        test_update_original = False
        load_dataset_in_memory(loader_train, loader_val, n_train_kept, test_update_original)
        print("Done!", flush = True)
    scaler_y = None
    dataset = loader_train, loader_val, loader_test, scaler_y

    model.eval()
    
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

    selection_lagrangian_reg = None
    entropy_lagrangian_reg = None

    set_require_grad_rec(model, False)
    if "facebook/opt" in arch:
        l_modules_before_threshold = get_modules_layer_wise_llm(model, train_val_attention_mask)
    else:
        l_modules_before_threshold = get_modules_layer_wise_rec(model, [])
    if test_one_layer_pruning:
        l_modules_before_threshold, l_downsamples = process_to_one_layer(l_modules_before_threshold)
    if ("mlpnet" in name_model):
        if test_one_layer_pruning:
            l_modules_before_threshold = [(View_mlpnet(),0,0,0)]+l_modules_before_threshold[0:1]+[(nn.ReLU(inplace=False),0,0,0)]+l_modules_before_threshold[1:2]+[(nn.ReLU(inplace=False),0,0,0)]+l_modules_before_threshold[2:]+[(Log_softmax_mlpnet(),0,0,0)]
        else:
            l_modules_before_threshold = [View_resnet()]+l_modules_before_threshold[0:1]+[nn.ReLU(inplace=False)]+l_modules_before_threshold[1:2]+[nn.ReLU(inplace=False)]+l_modules_before_threshold[2:]+[Log_softmax_mlpnet()]
    if ("resnet" in name_model):
        if test_one_layer_pruning:
            l_modules_before_threshold = l_modules_before_threshold[:-1]+[(View_resnet(),0,0,0)]+l_modules_before_threshold[-1:]
        else:
            l_modules_before_threshold = l_modules_before_threshold[:-1]+[View_resnet()]+l_modules_before_threshold[-1:]
    if ("mobilenet" in name_model):
        if test_one_layer_pruning:
            l_modules_before_threshold = l_modules_before_threshold[:-1]+[(View_mobilenet(),0,0,0)]+l_modules_before_threshold[-1:]
        else:
            l_modules_before_threshold = l_modules_before_threshold[:-1]+[View_mobilenet()]+l_modules_before_threshold[-1:]
    l_modules = process_l_modules_w_threshold(l_modules_before_threshold, threshold_weights, test_one_layer_pruning)
    if "facebook/opt" in arch:
        for module in l_modules:
            module.seqlen = model.seqlen

    main_model_wrapper = model_wrapper(model, None, seed, 0, 0, 0, device, 0, 0, None, -1, None, None, None, input_channel, "schedule", generator, "magnitude", None, None, selection_lagrangian_reg, entropy_lagrangian_reg, 0, None, type_function, 0, 0, 0, 0, gamma)
    main_model_wrapper.initialize_pruning()
    main_model_wrapper.compute_z()

    l_optimizers = [None for ind_module in range(len(l_modules))]
    l_model_wrap = [model_wrapper(l_modules[ind_module], l_optimizers[ind_module], seed, 0, 0, 0, device, 0, 0, None, -1, None, None, None, input_channel, "schedule", generator, "magnitude", None, None, selection_lagrangian_reg, entropy_lagrangian_reg, 0, None, type_function, 0, 0, 0, 0, gamma) for ind_module in range(len(l_modules))]
    l_modules_params = np.array([l_model_wrap[ind_model_wrap].get_n_z(test_grad=False) for ind_model_wrap in range(len(l_model_wrap))])
    if test_one_layer_pruning:
        l_optimizers_downsamples = [initialize_optimizer(0, l_downsamples[ind_module], None, None, -1, -1, None, None, selection_lagrangian_reg, entropy_lagrangian_reg) for ind_module in range(len(l_downsamples))]
        l_model_wrap_downsamples = [model_wrapper(l_downsamples[ind_module], l_optimizers_downsamples[ind_module], seed, 0, 0, 0, device, 0, 0, None, -1, None, None, None, input_channel, "schedule", generator, "magnitude", None, None, selection_lagrangian_reg, entropy_lagrangian_reg, 0, None, type_function, 0, 0, 0, 0, gamma) for ind_module in range(len(l_downsamples))]
        l_modules_downsamples_params = np.array([l_model_wrap[ind_model_wrap].get_n_z(test_grad=False) for ind_model_wrap in range(len(l_model_wrap_downsamples))])

    main_model_wrapper.model.to(device)
    best_model = main_model_wrapper.model

    in_sample_metric, validation_metric, test_metric = evaluate_llm(l_model_wrap, loader_train, loader_val, loader_test, n_train_kept, l_model_wrap[0].device, copy_indices_train, copy_indices_val, test_almost_sequential)
    print("Train metric:",in_sample_metric)
    print("Val metric:",validation_metric)
    print("Test metric:",test_metric)
    import ipdb;ipdb.set_trace()
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
        model, criterion, modules_to_prune = get_model(arch, seed, pretrained=pretrained, with_z=True, gamma=gamma, prune_bias=0)
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
