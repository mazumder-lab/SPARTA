import torch
import torch.nn as nn
import os
import copy

import numpy as np
from OBC_utils.modelutils import *
from torch.utils.data import Dataset, DataLoader, Subset
from previous_utils.main_utils import get_model, get_dataset
from utils_training import get_item_mnist, get_item_imagenet, get_item_cifar10, initialize_dataset, load_dataset_in_memory
from pytorch_dataset_2_0 import random_split
from utils_dataset import read_batch

def get_flops(layers, model, sample, run):
    flops = {}
    def record_flops(name):
        def tmp(layer, inp, out):
            inp = inp[0]
            if isinstance(layer, nn.Conv2d):
                flops[name] = inp.shape[2] * inp.shape[3]
                flops[name] *= layer.weight.numel()
                stride = list(layer.stride)
                flops[name] //= stride[0] * stride[1] 
            if isinstance(layer, nn.Linear):
                flops[name] = layer.weight.numel()
        return tmp
    handles = []
    for name, layer in layers.items():
        if hasattr(layer, 'module'):
            layer.module.register_forward_hook(record_flops(name))
        else:
            layer.register_forward_hook(record_flops(name))
    with torch.no_grad():
        run(model, sample)
    for h in handles:
        h.remove()
    return flops

def get_sparsity(layers, model, sample, run):
    sparsity = {}
    def record_sparsity(name):
        def tmp(layer, inp, out):
            inp = inp[0]
            if isinstance(layer, nn.Conv2d):
                sparsity[name] = layer.weight.numel()
            if isinstance(layer, nn.Linear):
                sparsity[name] = layer.weight.numel()
        return tmp
    handles = []
    for name, layer in layers.items():
        if hasattr(layer, 'module'):
            layer.module.register_forward_hook(record_sparsity(name))
        else:
            layer.register_forward_hook(record_sparsity(name))
    with torch.no_grad():
        run(model, sample)
    for h in handles:
        h.remove()
    return sparsity

def load_errors(sds, path, norm=False):
    errors = {}
    with open(path, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            name = lines[i].strip()
            errors[name] = {}
            i += 1
            for _ in range(len(sds)):
                err, level = lines[i].strip().split(' ')
                errors[name][level] = float(err)
                i += 1
    if norm:
        for name in errors:
            norm = max(errors[name].values())
            if norm > 0:
                for level in errors[name]:
                    errors[name][level] /= norm
    return errors

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SparsityDatabase:

    def __init__(self, sparsetype, model, prefix='', dev=device):
        self.sds = {}
        path = os.path.join(prefix, 'models_' + sparsetype)
        for f in os.listdir(path):
            if not (f.startswith(model + '_') and f.endswith('.pth')):
                continue
            sparsity = '0.' + f.split('.')[0].split('_')[-1]
            self.sds[sparsity] = torch.load(os.path.join(path, f), map_location=dev)
        self.sparsetype = sparsetype
        self.model = model
        self.prefix = prefix

    def load(self, layers, name, config='', sd=None):
        if not sd:
            sd = self.sds[config]
        if '8w8a' in self.sparsetype:
            layers[name].module.weight.data = sd[name + '.module.weight']
            layers[name].quantizer.maxq.data = sd[name + '.quantizer.maxq']
            layers[name].quantizer.scale.data = sd[name + '.quantizer.scale']
            layers[name].quantizer.zero.data = sd[name + '.quantizer.zero']
        else:
            layers[name].weight.data = sd[name + '.weight']

    def stitch(self, layers, config):
        for name, layer in layers.items():
            self.load(layers, name, config[name])

    def load_errors(self, name):
        path = os.path.join(
            self.prefix, 'scores/%s_%s_%s.txt' % (self.model, self.sparsetype, name)
        )
        return load_errors(self.sds, path, norm=name == 'squared')

    def get_params(self, layers):
        res = {}
        for name in layers:
            res[name] = {}
            for sparsity in self.sds:
                res[name][sparsity] = torch.sum(
                    (self.sds[sparsity][name + '.weight'] != 0).float()
                ).item()
        return res

    def get_flops(self, layers, model, sample, run):
        flops = get_flops(layers, model, sample, run)
        res = {}
        for name in layers:
            res[name] = {}
            for sparsity in self.sds:
                res[name][sparsity] = flops[name] * torch.mean(
                    (self.sds[sparsity][name + '.weight'] != 0).float()
                ).item()
        return res
    
    def get_sparsity(self, layers, model, sample, run):
        sparsitys = get_sparsity(layers, model, sample, run)
        res = {}
        for name in layers:
            res[name] = {}
            for sparsity in self.sds:
                res[name][sparsity] = sparsitys[name] * torch.mean(
                    (self.sds[sparsity][name + '.weight'] != 0).float()
                ).item()
        return res

    def get_timingsq(self):
        timings = {}
        with open('timings/%sq.txt' % self.model, 'r') as f:
            lines = f.readlines()
            baselinetime = float(lines[0])
            i = 1
            while i < len(lines):
                name = lines[i].strip()
                timings[name] = {}
                i += 1
                for _ in range(len(self.sds)):
                    time, level = lines[i].strip().split(' ')
                    timings[name][level] = float(time)
                    i += 1
        return baselinetime, timings


class QuantNMDatabase:

    def __init__(self, model, prefix=''):
        self.sds = {}
        for path in ['models_quant', 'models_nm_quant']:
            for f in os.listdir(os.path.join(prefix, path)):
                if not (f.startswith(model + '_') and f.endswith('.pth')):
                    continue
                config = '_'.join(f.split('.')[0].split('_')[1:])
                self.sds[config] = torch.load(os.path.join(prefix, path, f), map_location=device)
        self.model = model
        self.prefix = prefix

    def load(self, layers, name, config='', sd=None):
        if not sd:
            sd = self.sds[config]
        layers[name].module.weight.data = sd[name + '.module.weight']
        layers[name].quantizer.maxq.data = sd[name + '.quantizer.maxq']
        layers[name].quantizer.scale.data = sd[name + '.quantizer.scale']
        layers[name].quantizer.zero.data = sd[name + '.quantizer.zero']

    def stitch(self, layers, config):
        for name, layer in layers.items():
            self.load(layers, name, config[name])

    def load_errors(self, name):
        path = os.path.join(self.prefix, 'scores/%s_mixed_%s.txt' % (self.model, name))
        return load_errors(self.sds, path, norm=name == 'squared')

    def get_bits(self, layers):
        res = {}
        for name, layer in layers.items():
            paramcount = layer.module.weight.numel()
            res[name] = {
                # '24_4w4a': paramcount * 5,
                # '24_8w8a': paramcount * 9, 
                '24_4w4a': paramcount * 4,
                '24_8w8a': paramcount * 8, 
                   '4w4a': paramcount * 4,
                   '8w8a': paramcount * 8
            }
        return res

    def get_bops(self, layers, model, sample, run):
        flops = get_flops(layers, model, sample, run)
        res = {}
        for name, layer in layers.items():
            res[name] = {
                '24_4w4a': flops[name] * 32 // 2 // 8,
                '24_8w8a': flops[name] * 32 // 2 // 4,
                   '4w4a': flops[name] * 32 // 8,
                   '8w8a': flops[name] * 32 // 4 
            }
            if (layers[name].module.weight.numel() // layers[name].module.weight.shape[0]) % 4 != 0:
                res[name]['24_4w4a'] *= 2
                res[name]['24_8w8a'] *= 2
        return res


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('arch', type=str)
    parser.add_argument('name_dataset', type=str)
    parser.add_argument('database', choices=['mixed', '4block', 'unstr', '4block_8w8a'])
    parser.add_argument('mode', choices=['loss', 'squared', 'spdy', 'stitch', 'eval'])
    parser.add_argument('--prefix', type=str, default='Saves_OBC')
    parser.add_argument('--profile', type=str, default='')
    parser.add_argument('--score_path', type=str, default='scores')
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--datapath', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size_dataset', type=int, default=-1)
    parser.add_argument('--nrounds', type=int, default=-1)

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

    parser.add_argument('--rel_damp', type=float, default = 1e-2,
                    help='rel_damp*torch.diag(H).mean() is added to the hessian for more stability.')
    parser.add_argument('--lambda_fisher', type=float, default = 1e4,
                        help='lambda_fisher*WF is used in the objective function (with WF the wood-fisher approximation of the Hessian)')
    parser.add_argument('--lambda_reconst', type=float, default = 1.0,
                        help='lambda_reconst*H is used in the objective function (with H the Hessian of the layer-wise reconstruction loss)')
    parser.add_argument('--n_convex', type=int, default = -1,
                        help='number of convex combinations to try per layer (if set to -1, only the combination (lambda_reconst, lambda_fisher) is tried. If n_convex!=-1, then (lambda_reconst, lambda_fisher) is ignored and a list of n_convex pairs is created.')

    args = parser.parse_args()
    name_dataset = args.name_dataset
    seed = args.seed
    n_train_kept = args.n_train_kept
    n_convex = args.n_convex

    if n_convex==-1:
        args.prefix += f"_{args.lambda_reconst}_{args.lambda_fisher}_{args.rel_damp}"
    else:
        args.prefix += f"_{n_convex}_{args.rel_damp}"

    print("Folder saves:", args.prefix, flush=True)

    if not(os.path.exists(args.prefix+"/scores")):
        os.mkdir(args.prefix+"/scores")

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

    #model, loader_train, loader_test = model_factory(args.arch, dset_path, True, seed, args.nsamples, batch_size=args.batch_size_dataset)
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
            print("--- DIFFERENCE IN ATTENTION MASK ---", flush=True)
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

    #get_model, test, run = get_functions(args.arch)
    #dataloader, testloader = get_loaders(
    #    args.name_dataset, path=args.datapath,
    #    batch_size_dataset=args.batch_size_dataset, workers=args.workers,
    #    nsamples=args.nsamples, seed=seed,
    #    noaug=args.mode == 'loss'
    #)
    
    if args.nrounds == -1:
        args.nrounds = 1 if 'yolo' in args.arch or 'bert' in args.arch else 10
        if args.mode == 'loss':
            args.nrounds = 1

    filepath = os.path.join(args.prefix, args.score_path, '%s_%s_%s.txt' % (args.arch, args.database, args.mode))

    #modelp = get_model()
    modelp = copy.deepcopy(model)
    modelp.to(device)
    modelp.eval()

    if args.database == 'mixed':
        db = QuantNMDatabase(args.arch, prefix=args.prefix)
    if args.database in ['4block', 'unstr', '4block_8w8a']:
        db = SparsityDatabase(args.database, args.arch, prefix=args.prefix)
    if args.database in ['mixed', '4block_8w8a']:
        add_actquant(modelp)
    layersp = find_layers(modelp)

    #for i in range(layersp['fc'].weight.shape[0]):
    #    print(i)
    #    W = layersp['fc'].weight.data
    #    thresh = torch.sort(torch.abs(W[i, :]), descending=True)[0][9]
    #    W[i, torch.abs(W[i, :]) < thresh] = 0
    #    print(torch.mean((W[i, :] == 0).float()))
    #test(modelp, loader_test)
    #exit()

    #config = {n: '0.0000' for n in layersp}
    #config['fc'] = '0.9797' # '0.9900'
    #db.stitch(layersp, config)
    #with torch.no_grad():
    #    print(run(modelp, next(iter(loader_train)), loss=True) / args.nsamples)
    #test(modelp, loader_test)
    #exit()

    if args.mode == 'stitch':
        with open(args.profile, 'r') as f:
            config = {}
            for l in f.readlines():
                level, name = l.strip().split(' ')
                config[name] = '24_8w8a' # level
            db.stitch(layersp, config)
            test(modelp, loader_test)
        exit()

    if args.mode == 'eval':
        for s in sorted(db.sds):
            db.stitch(layersp, {n: s for n in layersp})
            print(s, flush=True)
            test(modelp, loader_test)
        exit()

    if args.mode == 'spdy':
        layersp = find_layers(modelp)
        tmp = (np.arange(len(db.sds)) / (len(db.sds) - 1)) ** 2
        print(len(db.sds), flush=True)
        print(len(tmp), flush=True)
        with open(filepath, 'w') as f:
            for layer in layersp:
                print(layer, flush=True)
                f.write(layer + '\n')
                for i, name in enumerate(sorted(db.sds)):
                    f.write('%.6f %s\n' % (tmp[i], name))
        exit()

    if args.mode == 'squared':
        modeld = get_model()
        layersd = find_layers(modeld)

        errs = {n: {} for n in layersp}
        def accumerrs(name):
            def tmp(layer, inp, out):
                errs[name]['dense'] = errs[name].get('dense', 0) + torch.sum(out.data ** 2).item()
                for config in sorted(db.sds):
                    db.load(layersp, name, config)
                    errs[name][config] = errs[name].get(config, 0) + torch.sum((layersp[name](inp[0].data) - out.data) ** 2).item()
            return tmp
        for name in layersd:
            layersd[name].register_forward_hook(accumerrs(name))

        with torch.no_grad():
            for _ in range(args.nrounds):
                for batch_sgd in tqdm(loader_train):
                    input_batch_sgd, input_batch_original_sgd, target_batch_sgd, index_seen_sgd, old_index_seen_sgd = read_batch(batch_sgd)
                    run(modeld, (input_batch_sgd, target_batch_sgd))

        with open(filepath, 'w') as f:
            for name in errs:
                f.write(name + '\n') 
                for config in sorted(errs[name]):
                    if config != 'dense':
                        f.write('%.6f %s\n' % (errs[name][config] / errs[name]['dense'], config))
        exit()

    if args.mode == 'loss':
        sd = modelp.state_dict()
        errs = {n: {} for n in layersp}
        baseloss = 0

        for i_round in range(args.nrounds):
            print(i_round, flush=True)
            for batch_sgd in tqdm(loader_train):
                input_batch_sgd, input_batch_original_sgd, target_batch_sgd, index_seen_sgd, old_index_seen_sgd = read_batch(batch_sgd)
                with torch.no_grad():
                    baseloss += run(modelp, (input_batch_sgd, target_batch_sgd), loss=True)
                    for name in layersp:
                        print(name, flush=True)
                        for config in sorted(db.sds):
                            db.load(layersp, name, config)
                            errs[name][config] = errs[name].get(config, 0) + run(modelp, (input_batch_sgd, target_batch_sgd), loss=True)
                        db.load(layersp, name, sd=sd)
        baseloss /= len(loader_train) * args.nrounds
        for name in errs:
            for config in errs[name]:
                errs[name][config] /= len(loader_train) * args.nrounds

        with open(filepath, 'w') as f:
            for name in errs:
                f.write(name + '\n') 
                for config in sorted(errs[name]):
                    f.write('%+.6f %s\n' % (errs[name][config] - baseloss, config))
        exit()
