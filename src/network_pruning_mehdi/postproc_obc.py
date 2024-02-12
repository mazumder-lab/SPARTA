import argparse

import torch
import torch.nn as nn

#from OBC_utils.datautils import *
#from OBC_utils.datautils2 import *
from database_obc import *
#from OBC_utils.modelutils import *
#from OBC_utils.quant import *
from torch.utils.data import DataLoader

from previous_utils.main_utils import get_model, get_dataset
from utils_training import get_item_mnist, get_item_imagenet, get_item_cifar10, initialize_dataset, load_dataset_in_memory
from pytorch_dataset_2_0 import random_split
# from utils_dataset import read_batch


parser = argparse.ArgumentParser()

parser.add_argument('arch', type=str)
parser.add_argument('name_dataset', type=str)
#parser.add_argument('load', type=str)
parser.add_argument('target', type=float)
parser.add_argument('--database', choices=['', 'mixed', '4block', 'unstr', '4block_8w8a'], default='')
parser.add_argument('--prefix', type=str, default='Saves_OBC')
parser.add_argument('--num_workers', type=int, default=0)

parser.add_argument('--datapath', type=str, default='')
parser.add_argument('--seed', type=int, default=0)

parser.add_argument('--batch_size_dataset', type=int, default=-1)
parser.add_argument('--nrounds', type=int, default=-1)
parser.add_argument('--noaug', action='store_true')

parser.add_argument('--skip-firstlast', action='store_true')

parser.add_argument('--bnt', action='store_true')
parser.add_argument('--bnt-batches', type=int, default=100)
parser.add_argument('--lintune', action='store_true')
parser.add_argument('--lintune-loss', action='store_true')
parser.add_argument('--lintune-epochs', type=int, default=100)
parser.add_argument('--lintune-lr', type=float, default=1e-4)
parser.add_argument('--gap', action='store_true')
parser.add_argument('--gap-epochs', type=int, default=100)
parser.add_argument('--gap-lr', type=float, default=1e-5)
parser.add_argument('--finetune', action='store_true')
parser.add_argument('--finetune-mse', action='store_true')
parser.add_argument('--finetune-epochs', type=int, default=2)
parser.add_argument('--finetune-lr', type=float, default=1e-5)
parser.add_argument('--statcorr', action='store_true')
parser.add_argument('--statcorr-samples', type=int, default=-1)
parser.add_argument('--save', type=str)

parser.add_argument('--prune_bias', type=int, default = 0,
                    help='wether to prune the bias or not')
parser.add_argument('--pretrained', type=lambda x: (str(x).lower() == 'true'), default=True)
parser.add_argument('--gamma', type=float, default = 1.0,
                help='gamma for SmoothStep')
parser.add_argument('--activation_fn', type=str, default = "relu",
                help='activation function for llm')
parser.add_argument('--n_train_kept', type=int, default = -1,
                help='number of training samples kept')
parser.add_argument('--test_almost_sequential', type=int, default = 3,
                help='If test_almost_sequential==1, we do not save the original dataset and only use the dataset created from the new model. If test_almost_sequential==2, we use this strategy only for pruning and use the original dataset for the retraining phases. If test_almost_sequential==3, we use fully parallel pruning and retraining.')

parser.add_argument('--dp', action='store_true')
parser.add_argument('--rel_damp', type=float, default = 1e-2,
                help='rel_damp*torch.diag(H).mean() is added to the hessian for more stability.')
parser.add_argument('--lambda_fisher', type=float, default = 1e4,
                    help='lambda_fisher*WF is used in the objective function (with WF the wood-fisher approximation of the Hessian)')
parser.add_argument('--lambda_reconst', type=float, default = 1.0,
                    help='lambda_reconst*H is used in the objective function (with H the Hessian of the layer-wise reconstruction loss)')
parser.add_argument('--n_convex', type=int, default = -1,
                    help='number of convex combinations to try per layer (if set to -1, only the combination (lambda_reconst, lambda_fisher) is tried. If n_convex!=-1, then (lambda_reconst, lambda_fisher) is ignored and a list of n_convex pairs is created.')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = parser.parse_args()

name_dataset = args.name_dataset
seed = args.seed
n_train_kept = args.n_train_kept
n_convex = args.n_convex

if n_convex==-1:
    args.prefix += f"_{args.lambda_reconst}_{args.lambda_fisher}_{args.rel_damp}"
else:
    args.prefix += f"_{n_convex}_{args.rel_damp}"

path_file = '%s_%s_%dx_spdy' % (args.arch, args.database, int(args.target * 100))
path_file = os.path.join(args.prefix, "sparsity_levels", path_file)
if args.dp:
    path_file = path_file.replace('spdy', 'dp')

args.load = path_file+ '.txt'

print("Folder saves:", args.prefix)


#dataloader, testloader = get_loaders(
#    args.name_dataset, path=args.datapath,
#    batchsize=args.batch_size_dataset, workers=args.workers,
#    nsamples=args.n_train_kept, seed=args.seed,
#    noaug=args.noaug
#)
#get_model, test, run = get_functions(args.arch)

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

#model, loader_train, loader_test = model_factory(args.arch, dset_path, True, seed, args.n_train_kept, batch_size=args.batch_size_dataset)
model, criterion, modules_to_prune = get_model(args.arch, seed, pretrained=args.pretrained, with_z=False, gamma=args.gamma, prune_bias=args.prune_bias, activation_fn=args.activation_fn)

test_update_test_vit = True

if name_dataset == "mnist":
    get_item_func = get_item_mnist
elif name_dataset == "cifar10":
    get_item_func = get_item_cifar10
elif name_dataset == "imagenet":
    get_item_func = get_item_imagenet
else:
    get_item_func = None

test_pass_first_modules = False
train_val_dataset, test_dataset = get_dataset(name_dataset, name_dataset_path, n_train_kept, get_item_func, args.arch, seed, args.activation_fn, device, args.test_almost_sequential, test_update_test_vit, test_pass_first_modules)

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
generator_loader = torch.Generator()

if seed != -1:
    torch.random.manual_seed(seed)
    generator_loader = generator_loader.manual_seed(seed)

loader_train = DataLoader(train_dataset, batch_size=args.batch_size_dataset, shuffle=True, generator=generator_loader, num_workers=args.num_workers, pin_memory=True)
loader_val = DataLoader(validation_dataset, batch_size=args.batch_size_dataset, num_workers=args.num_workers, pin_memory=True)
loader_test = DataLoader(test_dataset, batch_size=args.batch_size_dataset, num_workers=args.num_workers, pin_memory=True)
loader_train.dataset.indices = [i for i in range(len(loader_train.dataset.dataset))]
loader_val.dataset.indices = []

print("Load data in memory first...", flush = True)
test_update_original = False
load_dataset_in_memory(loader_train, loader_val, n_train_kept, test_update_original)
print("Done!", flush = True)

#modelp = get_model()
modelp = copy.deepcopy(model)
modelp.to(device)
modelp.eval()

if args.load.endswith('.pth'):
    tmp = torch.load(args.load)
    if any('scale' in k for k in tmp):
        add_actquant(modelp)
    if args.skip_firstlast:
        for l in firstlast_names(args.arch):
            if any('scale' in k for k in tmp):
                tmp[l + '.quantizer.scale'][:] = 0
                l += '.module'
            l += '.weight'
            tmp[l] = modelp.state_dict()[l]
    modelp.load_state_dict(tmp)
modelp = modelp.to(device)

if args.database != '':
    if args.database == 'mixed':
        print('Stitching ...')
        db = QuantNMDatabase(args.arch, prefix=args.prefix)
    if args.database in ['4block', 'unstr', '4block_8w8a']:
        db = SparsityDatabase(args.database, args.arch, prefix=args.prefix, dev='cpu')
    if args.database in ['mixed', '4block_8w8a']:
        add_actquant(modelp)
    modelp = modelp.to('cpu')
    layersp = find_layers(modelp)
    # New code:
    if "deit" in args.arch:
        new_layersp = {}
        for key in layersp:
            if not("embed" in key) and not("head" in key) and ("blocks" in key):
                new_layersp[key]=layersp[key]
    layersp = new_layersp
    # End new code

    with open(args.load, 'r') as f:
        config = {}
        for l in f.readlines():
            level, name = l.strip().split(' ')
            config[name] = level 
    db.stitch(layersp, config)
    modelp = modelp.to(device)
    layersp = find_layers(modelp)
    # New code:
    if "deit" in args.arch:
        new_layersp = {}
        for key in layersp:
            if not("embed" in key) and not("head" in key) and ("blocks" in key):
                new_layersp[key]=layersp[key]
    layersp = new_layersp
    # End new code

    if args.save:
        torch.save(modelp.state_dict(), args.save)
        exit()


if args.bnt:
    print('Batchnorm tuning ...')

    loss = 0
    for batch in tqdm(loader_train):
        loss += run(modelp, batch, loss=True)
    print(loss / args.n_train_kept)

    batchnorms = find_layers(modelp, [nn.BatchNorm2d])
    for bn in batchnorms.values():
        bn.reset_running_stats()
        bn.momentum = .1
    modelp.train()
    with torch.no_grad():
        i = 0
        while i < args.bnt_batches:
            for batch in tqdm(loader_train):
                if i == args.bnt_batches:
                    break
                # print('%03d' % i)
                run(modelp, batch)
                i += 1
    modelp.eval()

    loss = 0
    for batch in loader_train:
        loss += run(modelp, batch, loss=True)
    print(loss / args.n_train_kept)

if args.lintune:
    print('Linear tuning ...')
    modeld = get_model()
    params = []
    for n, p in modelp.named_parameters():
        if len(p.shape) == 1:
            params.append(p)
        else:
            p.requires_grad = False
    optim = torch.optim.Adam(params, lr=args.lintune_lr)
    criterion = nn.MSELoss()
    for i in range(args.lintune_epochs):
        cumloss = 0
        for batch in loader_train:
            if args.lintune_loss:
                loss = run(modelp, batch, loss=True)
            else:
                with torch.no_grad():
                    y = run(modeld, batch)
                loss = criterion(run(modelp, batch), y)
            cumloss += loss.item()
            loss.backward()
            optim.step()
            optim.zero_grad()
        print('%02d %.4f' % (i, cumloss / len(loader_train)))

if args.gap:
    modeld = get_model()
    layersp = find_layers(modelp) 
    # New code:
    if "deit" in args.arch:
        new_layersp = {}
        for key in layersp:
            if not("embed" in key) and not("head" in key) and ("blocks" in key):
                new_layersp[key]=layersp[key]
    layersp = new_layersp
    # End new code

    layersd = find_layers(modeld)
    # New code:
    if "deit" in args.arch:
        new_layersd = {}
        for key in layersd:
            if not("embed" in key) and not("head" in key) and ("blocks" in key):
                new_layersd[key]=layersd[key]
    layersd = new_layersd
    # End new code

    masks = {n: l.weight.data == 0 for n, l in layersp.items()}

    def cache_output(name, outputs):
        def tmp(layer, inp, out):
            outputs[name] = out
        return tmp
    outputsp = {}
    handlesp = []
    for name in layersp:
        handlesp.append(
            layersp[name].register_forward_hook(cache_output(name, outputsp))
        )
    outputsd = {}
    handlesd = [] 
    for name in layersd:
        handlesd.append(
            layersd[name].register_forward_hook(cache_output(name, outputsd))
        )

    criterion = nn.MSELoss(reduction='sum')
    optim = torch.optim.Adam(modelp.parameters(), lr=args.gap_lr)

    for i in range(args.gap_epochs):
        cumloss = 0
        for batch in loader_train:
            with torch.no_grad():
                run(modeld, batch) 
            run(modelp, batch)
            loss = 0
            for name in outputsd:
                norm = torch.norm(outputsd[name].data).item() ** 2
                loss += criterion(outputsp[name], outputsd[name].data) / norm
            cumloss += loss.item()
            loss.backward()
            optim.step()
            optim.zero_grad()
            for name, mask in masks.items():
                layersp[name].weight.data[mask] = 0
        print('%05d: %.6f' % (i, cumloss / len(loader_train)))

    for h in handlesp:
        h.remove()
    for h in handlesd:
        h.remove()

if args.finetune:
    print('Finetuning ...')
    modeld = get_model()
    masks = {n: p == 0 for n, p in modelp.named_parameters()}
    optim = torch.optim.Adam(modelp.parameters(), lr=args.finetune_lr)
    criterion = nn.MSELoss()
    for i in range(args.finetune_epochs):
        cumloss = 0
        for batch in loader_train:
            if args.finetune_mse:
                with torch.no_grad():
                    y = run(modeld, batch)
                loss = criterion(run(modelp, batch), y)
            else:
                loss = run(modelp, batch, loss=True)
            cumloss += loss.item()
            loss.backward()
            optim.step()
            optim.zero_grad()
            for n, p in modelp.named_parameters():
                p.data[masks[n]] = 0
        print('%02d %.4f' % (i, cumloss / len(loader_train)))

if args.statcorr:
    print('Stat correction ...')

    if args.statcorr_samples == -1:
        args.statcorr_samples = args.n_train_kept
    trainloader, testloader = get_loaders(
        args.name_dataset, batchsize=args.statcorr_samples, noaug=True
    )
    batch = next(iter(trainloader))

    modeld = get_model()
    layersd = find_layers(modeld, layers=[nn.BatchNorm2d, nn.LayerNorm])
    layersp = find_layers(modelp, layers=[nn.BatchNorm2d, nn.LayerNorm])

    REDUCE = {
        2: [0],
        3: [0, 1],
        4: [0, 2, 3]
    }

    meansd = {}
    stdsd = {}
    def hookd(name):
        def tmp(layer, inp, out):
            red = REDUCE[len(out.shape)]
            meansd[name] = torch.mean(out.data, red, keepdim=True)
            stdsd[name] = torch.std(out.data, red, keepdim=True)
        return tmp
    meansp = {}
    stdsp = {}
    def hookp(name):
        def tmp(layer, inp, out):
            red = REDUCE[len(out.shape)]
            meansp[name] = torch.mean(out.data, red, keepdim=True)
            stdsp[name] = torch.std(out.data, red, keepdim=True)
            out.data -= meansp[name]
            out.data *= stdsd[name] / (stdsp[name] + 1e-9)
            out.data += meansd[name]
        return tmp
    handles = []
    for name in layersd:
        handles.append(layersd[name].register_forward_hook(hookd(name)))
    with torch.no_grad():
        run(modeld, batch)
    for h in handles:
        h.remove()
    handles = []
    for name in layersp:
        handles.append(layersp[name].register_forward_hook(hookp(name)))
    with torch.no_grad():
        run(modelp, batch)
    for h in handles:
        h.remove()

    def hook(name):
        def tmp(layer, inp, out):
            out.data -= meansp[name]
            out.data *= stdsd[name] / (stdsp[name] + 1e-9)
            out.data += meansd[name]
        return tmp
    for name in layersp:
        layersp[name].register_forward_hook(hook(name))

#TO DELETE:
# l_acc = []
# l_targets = [0.5, 0.6, 0.7, 0.8, 0.9]#, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98]
# for target in l_targets:
#     modelp.load_state_dict(torch.load(f"Saves_OBC_0.0_1.0_0.01/models_unstr/mlpnet_{int(10000*target)}.pth"))
#     acc = test(modelp, loader_test)
#     l_acc.append(acc)
# for i in range(len(l_targets)):
#     print(l_targets[i], round(l_acc[i],2))
# import ipdb; ipdb.set_trace()
# END

acc = test(modelp, loader_test)
name = '%s' % (args.load)
new_name = args.load.replace("sparsity_levels", "results")
name_save_model = '%s' % (new_name.replace(".txt",".pth"))

if not(os.path.exists(name.split("/")[0]+"/results")):
    os.mkdir(name.split("/")[0]+"/results")

print("Accuracy:", acc)

with open(new_name, 'w') as f:
    f.write('%.2f\n' % (acc))

torch.save(modelp.state_dict(), name_save_model)