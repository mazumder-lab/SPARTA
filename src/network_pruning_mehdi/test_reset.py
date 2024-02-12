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

# New HYPERPARAMETERS
parser.add_argument('--n_epochs', type=int, default = 1000,
                    help='number of epochs for the training function')
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
parser.add_argument('--type_decay', type=str, default = "None",
                    help='criteria for the decay. If type_decay = "None", then no decay is applied. The other types of decays are "linear", "exponential" and "cosine"')
parser.add_argument('--gamma_lr_decay', type=float, default = 0.9,
                    help='learning rate decay for type_decay = "exponential"')
parser.add_argument('--T_max_cos', type=int, default = 5,
                    help='half-period of the cosine for type_decay = "cosine"')
parser.add_argument('--eta_min_cos', type=float, default = 1.0,
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
parser.add_argument('--entropy_reg', type=float, default = 0.1,
                    help='regularizer factor for entropy penalization in additive_soft_trees')
parser.add_argument('--selection_reg', type=float, default = 0.001,
                    help='regularizer factor for selection penalization in additive_soft_trees')
parser.add_argument('--metric_early_stopping', type=str, default = "val_loss",
                    help='either val_loss or val_accuracy')
parser.add_argument('--l2_reg', type=float, default = 0.0,
                    help='regularizer factor for l2 penalization in additive_soft_trees')
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

arguments, unknown = parser.parse_known_args()

#%%
from previous_utils.main_utils import get_dataset, get_model
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

#%%
##Change this to path of imagenet name_dataset
if 'IMAGENET_PATH' in os.environ:  
    IMAGENET_PATH = os.environ['IMAGENET_PATH']
else:
    print('****Warning**** No IMAGENET_PATH variable')
    IMAGENET_PATH = ''
CIFAR10_PATH = '../datasets'
MNIST_PATH = '../datasets'

name_dataset_paths = {'imagenet':IMAGENET_PATH,'cifar10':CIFAR10_PATH,
                'mnist':MNIST_PATH}

name_dataset_path = name_dataset_paths[name_dataset]

n_params_original_z = 0
train_val_dataset, test_dataset = get_dataset(arch,name_dataset_path)
loader = DataLoader(train_val_dataset, batch_size=500)

generator_split = torch.Generator().manual_seed(seed)
train_dataset, validation_dataset = random_split(train_val_dataset, [0.8, 0.2], generator=generator_split)

scaler_y = None
dataset = train_dataset, validation_dataset, test_dataset, scaler_y

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
                type_training)

n_samples = len(train_dataset)

if batch_size_dataset == -1:
    batch_size_dataset = n_samples

steps_per_epoch = np.ceil(n_samples/batch_size_dataset)
test_deja_train = False
n_trials_done = 0
save_study_done = None

lr = learning_rate
print('Parsed arguments:', arguments)
print("lr:", lr)
model_simple, criterion, modules_to_prune = get_model(arch, seed, pretrained=pretrained, with_z=False)
model, criterion, modules_to_prune = get_model(arch, seed, pretrained=pretrained, with_z=True)
loaded_weights = torch.load("/Users/afriatg/Desktop/Research/network_pruning/Saves_combined_from_scratch/study_mlpnet_lr_0.005_500_mnist_Adam_n_epochs_300_cosine_T_max_300_min_lr_1e-05_es_patience_50_val_loss_gamma_1_sel_reg_0.01_ent_reg_0.1_l2_reg_0.001_wd_3.752e-5_mom_0.9_scratch_auc_goal_s_0.98/best_trial/repeat_0/model")
model_wrap = model_wrapper(model, seed, entropy_reg, selection_reg, l2_reg, device, dense_to_sparse, test_different_lr, steps_per_epoch, val_second_lr, momentum, weight_decay, n_params_original_z, tol_z_1)
model_wrap.reset_z(True)
# # %%
# l_children = list(model_wrap.model.children())
# l_children_simple = list(model_simple.children())
# # %%
# for i in range(3):
#     print(torch.max(torch.abs(l_children_simple[i].weight - l_children[i].z*l_children[i].weight)))
# # %%
# model.eval()
# model_simple.eval()
# model(train_dataset.dataset.data.float()) - model_simple(train_dataset.dataset.data.float())
# # %%
# input_x = train_dataset.dataset.data.float()
# input_x = input_x.view(input_x.shape[0], -1)
# x = l_children[0](input_x)
# # x = F.relu(l_children[1](x))
# # x = l_children[2](x)
# # x = F.log_softmax(x, dim=1)
# # %%
# x_simple = l_children_simple[0](input_x)
# # x_simple = F.relu(l_children_simple[1](x_simple))
# # x_simple = l_children_simple[2](x_simple)
# # x_simple = F.log_softmax(x_simple, dim=1)

# # %%
# torch.max(torch.abs(x-x_simple))
# # %%
# (l_children_simple[0].weight).mm(input_x.T).T-x_simple
# # %%
# (l_children[0].weight*l_children[0].z).mm(input_x.T).T-x
# # %%
# #test_dataset.data = test_dataset.data.float()
# #evaluate_neural_network(model_simple, test_dataset, type_of_task, "cpu", scaler_y=scaler_y)
# # %%
# #train_val_dataset.data = train_val_dataset.data.float()
# #evaluate_neural_network(model_simple, train_val_dataset, type_of_task, "cpu", scaler_y=scaler_y)
# # %%
# from previous_utils.main_utils import compute_acc
# model_simple, criterion, modules_to_prune = get_model(arch, seed, pretrained=pretrained, with_z=False)
# train_val_dataset, test_dataset = get_dataset(arch,name_dataset_path)
# generator_split = torch.Generator().manual_seed(seed)
# train_dataset, validation_dataset = random_split(train_val_dataset, [0.8, 0.2], generator=generator_split)
# train_dataset = train_dataset.dataset
# validation_dataset = validation_dataset.dataset
# #print(compute_acc(model, DataLoader(train_dataset, batch_size=500), verbose=True))
# print(evaluate_neural_network(model_simple, DataLoader(test_dataset, batch_size=500), type_of_task, "cpu", scaler_y=scaler_y))
# # %%
# model_simple.enable_dropout
# # %%
# train_val_dataset.data.shape
# # %%
# firt_el = list(DataLoader(train_val_dataset, batch_size=500, shuffle=False))[0][0]
# # %%
# train_val_dataset.data[:500, None,:,:]
# # %%
# for name,param in model_simple.named_parameters():
#     print(name, param.mean())
# # %%
# model, criterion, modules_to_prune = get_model(arch, seed, pretrained=pretrained, with_z=True)
# loaded_weights = torch.load("/Users/afriatg/Desktop/Research/network_pruning/Saves_combined_pretrained/study_mlpnet_lr_0.005_500_mnist_Adam_n_epochs_300_cosine_T_max_300_min_lr_1e-05_es_patience_50_val_loss_gamma_1_sel_reg_0.01_ent_reg_0.1_l2_reg_0.001_wd_3.752e-5_mom_0.9_auc_goal_s_0.98/best_trial/repeat_0/model", map_location="cpu")
# model.load_state_dict(loaded_weights)
# # %%
# l_children = list(model.children())
# n_0 = 0
# n_tot = 0
# for i in range(3):
#     l_children[i].compute_z()
#     l_children[i].weight.data[l_children[i].z==0] = 0
#     n_0+=torch.sum((l_children[i].z==0).float())
#     n_tot+=np.prod(l_children[i].z.shape)
# # %%
# n_0/n_tot
# # %%
# evaluate_neural_network(model, DataLoader(test_dataset, batch_size=500), type_of_task, "cpu", scaler_y=scaler_y)

# # %%
from previous_utils.main_utils import compute_acc
model_simple, criterion, modules_to_prune = get_model(arch, seed, pretrained=pretrained, with_z=False)
model, criterion, modules_to_prune = get_model(arch, seed, pretrained=pretrained, with_z=True)
loaded_weights = torch.load("/Users/afriatg/Desktop/Research/network_pruning/Saves_combined_pretrained/study_mlpnet_lr_0.005_500_mnist_Adam_n_epochs_300_cosine_T_max_300_min_lr_1e-05_es_patience_50_val_loss_gamma_1_sel_reg_0.01_ent_reg_0.1_l2_reg_0.001_wd_3.752e-5_mom_0.9_auc_goal_s_0.98/best_trial/repeat_0/model", map_location="cpu")
model_simple.load_state_dict(loaded_weights, strict=False)
model.load_state_dict(loaded_weights)
# %%
print(compute_acc(model, DataLoader(test_dataset, batch_size=500), verbose=False))
print(compute_acc(model_simple, DataLoader(test_dataset, batch_size=500), verbose=False))
#%%
l_children_simple = list(model_simple.children())
l_children = list(model.children())
n_0 = 0
n_tot = 0
for i in range(3):
    child_simple = l_children_simple[i]
    child = l_children[i]
    child.compute_z()
    print(child.z[((child.z>0.0)*(child.z<1.0)).bool()])
    child_simple.weight.data[child.z == 0] = 0
    n_0+=torch.sum((child_simple.weight==0).float()).item()
    n_tot+=np.prod(child_simple.weight.shape)
# %%
print("Total number of weights that could be pruned:", n_tot)
print("Total number of pruned weights:", n_0)
print("Sparsity:", n_0/n_tot)
# %%
print(compute_acc(model, DataLoader(test_dataset, batch_size=500), verbose=False))
print(compute_acc(model_simple, DataLoader(test_dataset, batch_size=500), verbose=False))
#%%
l_children_simple = list(model_simple.children())
n_z_to_delete = 0
for k in range(3):
    child_simple = l_children_simple[k]
    weight_data = child_simple.weight.data
    n_row, n_col = weight_data.shape
    l_row_to_delete = []
    for i in range(n_row):
        current_row = weight_data[i,:]
        if torch.sum(current_row!=0)==0:
            n_z_to_delete+=n_col
            l_row_to_delete.append(i)
    for j in range(n_col):
        current_col = weight_data[:,j]
        if torch.sum(current_col!=0)==0:
            #print(current_col)
            n_z_to_delete+=n_row-len(l_row_to_delete)

print("n_z to delete:", n_z_to_delete)
# %%
from previous_utils.main_utils import compute_acc, get_model, get_dataset, model_factory
from torch.utils.data import DataLoader
from utils_training import evaluate_neural_network

if 'IMAGENET_PATH' in os.environ:  
    IMAGENET_PATH = os.environ['IMAGENET_PATH']
else:
    print('****Warning**** No IMAGENET_PATH variable')
    IMAGENET_PATH = ''
CIFAR10_PATH = '../datasets'
MNIST_PATH = '../datasets'

name_dataset_paths = {'imagenet':IMAGENET_PATH,'cifar10':CIFAR10_PATH,
                'mnist':MNIST_PATH}
name_dataset = "cifar10"
name_dataset_path = name_dataset_paths[name_dataset]

model_simple, criterion, modules_to_prune = get_model("resnet20", 0, pretrained=True, with_z=False)
train_val_dataset, test_dataset = get_dataset("resnet20",name_dataset_path)
#%%
model_simple.train()
compute_acc(model_simple, DataLoader(train_val_dataset, batch_size=128), 'cpu')
#%%
model_simple.eval()
compute_acc(model_simple, DataLoader(test_dataset, batch_size=128), 'cpu')
#%%
model_simple,train_dataset,test_dataset,criterion,modules_to_prune = model_factory('resnet20','datasets/')
model_simple.eval()
#%%
model_simple, criterion, modules_to_prune = get_model("mlpnet", 0, pretrained=True, with_z=False)
#%%
#print(compute_acc(model_simple, DataLoader(test_dataset, batch_size=500), verbose=True))
print(evaluate_neural_network(model_simple, DataLoader(test_dataset, batch_size=500), "classification", "cpu"))

# %%
n_tot = 0
for param in model_simple.named_parameters():
    if "_z" in param[0]:
        n_tot+=np.prod(param[1].shape)
    if np.prod(param[1].shape)== (270906-270896):
        print(param[0])

param
# %%
def compute_n_z_rec2(module):
    n_z = 0
    name_module = module.__str__().lower()
    d_children = dict(module.named_children())
    test_children = len(d_children)>0
    if test_children:
        for name_child in d_children:
            child = d_children[name_child]
            n_z += compute_n_z_rec2(child)
    elif not("relu" in name_module) and not("avgpool" in name_module) and not("norm" in name_module):
        try:
            if module.test_bias:
                print(name_module)
                return np.prod(module.z.shape, dtype=float).item()+np.prod(module.z_2.shape, dtype=float).item()
            else:
                return np.prod(module.z.shape, dtype=float).item()
        except:
            print(name_module)
    return n_z


from utils_model import compute_n_z_rec
compute_z_rec(model_simple)
compute_n_z_rec2(model_simple)
# %%
# %%
compute_n_z_rec(model_simple)
# %%
model_simple
# %%
