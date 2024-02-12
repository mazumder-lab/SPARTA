import torch
import sys
import numpy as np
import os
import random
import torchvision.datasets as datasets
# from new_cifar10 import CIFAR10, CIFAR100
# from new_imagnet import ImageFolder
# from new_mnist import MNIST
import torchvision.transforms as transforms

#from torchvision.models import resnet50 as torch_resnet50
from new_resnet50 import resnet50 as torch_resnet50
from models.resnet_cifar10 import resnet20
from models.wideresnet_cifar import Wide_ResNet
from models.mlpnet import MlpNet
from models.mobilenet import mobilenet
from collections import OrderedDict
import json
import torch.distributed as dist
from utils_model import Conv2d_with_z, Linear_with_z
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
import shutil
from utils_training import read_batch, Pos_embed, update_dataset, initialize_dataset

#from CHITA_opt.L0_card_const import Heuristic_CD_PP,Active_IHTCDLS_PP,Heuristic_LS,Heuristic_LSBlock,evaluate_obj
from torch.utils.data import Dataset
import copy
from torch.utils.data import DataLoader

from models.resnet_mehdi import ResNet18, get_train_and_test_dataloader

class Dataset_LLM(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
        self.return_original = True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        """
        Args:
        index (int): Index

        Returns:
        tuple: (image, target) where target is index of the target class.
        """
        new_index = index-self.required_increment[index]
        data_point, target = self.data[new_index], self.targets[new_index]

        target = torch.cat([target, torch.tensor([new_index, index])]).long()
        #target = torch.Tensor([target, new_index, index]).long()

        if not(self.is_original) and self.return_original:
            sample_original = self.data_output_original[new_index]
        
        if not(self.is_original) and self.return_original:
            return data_point, sample_original, target
        else:
            return data_point, target

def sync_weights(model, rank, world_size):
    for param in model.parameters():
        if rank == 0:
            # Rank 0 is sending it's own weight
            # to all it's siblings (1 to world_size)
            for sibling in range(1, world_size):
                dist.send(param.data, dst=sibling)
        else:
            # Siblings must recieve the parameters
            dist.recv(param.data, src=0)

def sync_mask(pruner, rank, world_size):
    if rank == 0:
        # Rank 0 is sending it's own weight
        # to all it's siblings (1 to world_size)
        for sibling in range(1, world_size):
            dist.send(pruner.mask.data, dst=sibling)
    else:
        # Siblings must recieve the parameters
        dist.recv(pruner.mask.data, src=0)

def flatten_tensor_list(tensors):
    flattened = []
    for tensor in tensors:
        flattened.append(tensor.view(-1))
    return torch.cat(flattened, 0)

def print_parameters(model):
    for name, param in model.named_parameters(): 
        print(name, param.shape)

def load_model(path, model):
    tmp = torch.load(path, map_location='cpu')
    if 'state_dict' in tmp:
        tmp = tmp['state_dict']
    if 'model' in tmp:
        tmp = tmp['model']
    for k in list(tmp.keys()):
        if 'module.' in k:
            tmp[k.replace('module.', '')] = tmp[k]
            del tmp[k]
    model.load_state_dict(tmp)

def imagenet_get_datasets(data_dir, get_item_func):

    train_dir = os.path.join(data_dir, 'raw_train')
    test_dir = os.path.join(data_dir, 'raw_val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    # train_transform = transforms.Compose([
    #     transforms.RandomResizedCrop(224),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     normalize,
    # ])

    train_transform = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip()
    ]
    
    train_transform += [
        transforms.ToTensor(),
        normalize,
    ]
    train_transform = transforms.Compose(train_transform)

    class CustomImageFolder(datasets.ImageFolder):
        def __getitem__(self, index: int):
            return get_item_func(self, index)

    # train_dataset = datasets.ImageFolder(train_dir, train_transform)
    train_dataset = CustomImageFolder(train_dir, train_transform)

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # test_dataset = datasets.ImageFolder(test_dir, test_transform)
    test_dataset = CustomImageFolder(test_dir, test_transform)

    return train_dataset, test_dataset

@torch.no_grad()
def get_pvec(model, params):
    state_dict = model.state_dict()
    return torch.cat([
        state_dict[p].reshape(-1) for p in params
    ])

@torch.no_grad()
def get_sparsity(model, params):
    pvec = get_pvec(model,params)
    return (pvec == 0).float().mean()

@torch.no_grad()
def get_blocklist(model,params,block_size):
    i_w = 0
    block_list = [0]
    state_dict = model.state_dict()
    for p in params:
        param_size = np.prod(state_dict[p].shape)
        if param_size <block_size:
            block_list.append(i_w+param_size)
        else:
            num_block = int(param_size/block_size)
            block_subdiag = list(range(i_w,i_w+param_size+1,int(param_size/num_block))) 
            block_subdiag[-1] = i_w+param_size
            block_list += block_subdiag   
        i_w += param_size
    return block_list

@torch.no_grad()
def set_pvec(w, model, params, device, nhwc=False):
    state_dict = model.state_dict()
    i = 0
    for p in params:
        count = state_dict[p].numel()
        if type(w) ==  torch.Tensor :
            state_dict[p] = w[i:(i + count)].reshape(state_dict[p].shape)
        else:
            state_dict[p] = torch.Tensor(w[i:(i + count)]).to(device).reshape(state_dict[p].shape)
        i += count
    model.load_state_dict(state_dict)

@torch.no_grad()
def get_gvec(model, params):
    named_parameters = dict(model.named_parameters())
    return torch.cat([
        named_parameters[p].grad.reshape(-1) for p in params
    ])

@torch.no_grad()
def get_gvec1(model, params):
    named_parameters = dict(model.named_parameters())
    return torch.cat([
        named_parameters[p].grad_sample.reshape(named_parameters[p].grad_sample.shape[0],-1) for p in params
    ],dim=1)

@torch.no_grad()
def get_gps_vec(model, params):
    named_parameters = dict(model.named_parameters())
    return torch.cat([
        named_parameters[p].grad_sample.reshape(named_parameters[p].grad_sample.shape[0],-1) for p in params
    ],dim=1)

@torch.no_grad()
def apply_mask(mask, model, params,device):
    state_dict = model.state_dict()
    i = 0
    for p in params:
        param = state_dict[p]
        count = param.numel()
        state_dict[p] *= mask[i:(i + count)].to(device).reshape(param.shape).float()
        i += count
    model.load_state_dict(state_dict)
    
@torch.no_grad()
def zero_grads(model):
    for p in model.parameters():
        p.grad = None

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def compute_acc(model,dataloader,device='cpu',verbose=False):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    i = 0
    with torch.no_grad():
        for batch_sgd in tqdm(dataloader):
            input_batch_sgd, input_batch_original_sgd, target_batch_sgd, index_seen_sgd, old_index_seen_sgd = read_batch(batch_sgd)
            i+=1
            images, labels = input_batch_sgd.to(device), target_batch_sgd.to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            if len(labels.shape)>=2:
                labels = labels[:,0]
            correct += (predicted == labels).sum().item()
            if verbose and i%10 == 0:
                print(total, correct)

            del images,labels,outputs

    return 100 * correct / total

def compute_loss(model,criterion,dataloader,device='cpu',verbose=False):
    avg_loss = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    i = 0
    with torch.no_grad():
        for data in dataloader:
            i+=1
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            images=images
            labels=labels
            # calculate outputs by running images through the network
            outputs = model(images)
            loss = criterion(outputs, labels).item()
            avg_loss += loss
            if verbose and i%100 ==0:
                print('computing loss', i)

            del images,labels,outputs

    return avg_loss / i

def generate_schedule(num_stages, base_level,sparsity_level_selection,schedule):
    repeat=1
    if num_stages == 1:
        return [sparsity_level_selection]
    if schedule == 'exp':
        sparsity_multiplier = (sparsity_level_selection - base_level)*np.power(2, num_stages-1)/(np.power(2, num_stages-1) - 1)
        l =[base_level + sparsity_multiplier*((np.power(2, stage) - 1)/np.power(2, stage)) for stage in range(num_stages)]
        return [x for x in l for _ in range(repeat)]
    elif schedule == 'poly':
        l= [sparsity_level_selection + (base_level-sparsity_level_selection)*np.power(1 - (stage/(num_stages-1)), 3) for stage in range(num_stages)]
        return [x for x in l for _ in range(repeat)]
    elif schedule == 'const':
        return [sparsity_level_selection for stage in range(num_stages)]
    elif schedule == 'linear':
        return [base_level + stage*(sparsity_level_selection - base_level)/(num_stages-1) for stage in range(num_stages)]
    elif schedule == 'MFAC':
        sparsity_multiplier = ((1. - sparsity_level_selection) / (1. - base_level)) ** (1./num_stages)
        return [1. - ((1. - base_level) * (sparsity_multiplier**(stage+1))) for stage in range(num_stages)]

def mnist_get_datasets(data_dir, get_item_func):
    # same used in hessian repo!
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    class CustomMNIST(datasets.MNIST):
        def __getitem__(self, index: int):
            return get_item_func(self, index)
    
    train_dataset = CustomMNIST(root=data_dir, train=True,
                                   download=True, transform=train_transform)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = CustomMNIST(root=data_dir, train=False,
                                  transform=test_transform)

    return train_dataset, test_dataset

def model_factory(arch,dset_path,pretrained=True):
    if arch == 'mlpnet':
        model = MlpNet(args=None,dataset='mnist',with_z = False)
        train_dataset,test_dataset = mnist_get_datasets(dset_path)
        criterion = torch.nn.functional.nll_loss

        state_trained = torch.load('checkpoints/mnist_25_epoch_93.97.ckpt',map_location=torch.device('cpu'))['model_state_dict']
        new_state_trained = OrderedDict()
        for k in state_trained:
            if 'mask' in k:
                continue
            new_state_trained[k.split('.')[1]+'.'+k.split('.')[3]] = state_trained[k]
        if pretrained:
            model.load_state_dict(new_state_trained, strict=False)
        modules_to_prune = []
        for name, param in model.named_parameters():
            #print("name is {} and shape of param is {} \n".format(name, param.shape))
            layer_name,param_name = '.'.join(name.split('.')[:-1]),name.split('.')[-1]
            if param_name == 'bias':
                    continue
            if 'conv' in layer_name or 'fc' in layer_name:
                modules_to_prune.append(name)
        return model,train_dataset,test_dataset,criterion,modules_to_prune
    elif arch == 'resnet20':
        state_trained = torch.load('checkpoints/resnet20_cifar10.pth.tar',map_location=torch.device('cpu'))['state_dict']
        new_state_trained = OrderedDict()
        for k in state_trained:
            new_state_trained[k[7:]] = state_trained[k]

        model = resnet20(with_z = False)
        if pretrained:
            model.load_state_dict(new_state_trained)

        test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        train_random_transforms=True

        if train_random_transforms:
            train_transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

        train_dataset = datasets.CIFAR10(root=dset_path, train=True, download=True,transform=train_transform)
        test_dataset = datasets.CIFAR10(root=dset_path, train=False, download=True,transform=test_transform)

        criterion = torch.nn.functional.cross_entropy

        modules_to_prune = []
        for name, param in model.named_parameters():
            #print("name is {} and shape of param is {} \n".format(name, param.shape))
            layer_name,param_name = '.'.join(name.split('.')[:-1]),name.split('.')[-1]
            if param_name == 'bias':
                    continue
            if 'conv' in layer_name or 'fc' in layer_name:
                modules_to_prune.append(name)

        return model,train_dataset,test_dataset,criterion,modules_to_prune
    
    elif arch == 'mobilenetv1':
        model = mobilenet()

        train_dataset,test_dataset = imagenet_get_datasets(dset_path)

        criterion = torch.nn.functional.cross_entropy

        modules_to_prune = []
        for name, layer in model.named_modules():
            if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear) or isinstance(layer, Conv2d_with_z) or isinstance(layer, Linear_with_z):
                modules_to_prune.append(name+'.weight')

        if pretrained:
            path = 'checkpoints/MobileNetV1-Dense-STR.pth'
            state_trained = torch.load(path,map_location=torch.device('cpu'))['state_dict']
            new_state_trained = model.state_dict()
            for k in state_trained:
                key = k[7:]
                if key in new_state_trained:
                    new_state_trained[key] = state_trained[k].view(new_state_trained[key].size())
                else:
                    print('Missing key',key)
            model.load_state_dict(new_state_trained,strict=False)

        return model,train_dataset,test_dataset,criterion,modules_to_prune
    elif arch == 'resnet50':
        model = torch_resnet50(weights=None)
        train_dataset,test_dataset = imagenet_get_datasets(dset_path)
        criterion = torch.nn.functional.cross_entropy

        modules_to_prune = []
        for name, layer in model.named_modules():
            if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear) or isinstance(layer, Conv2d_with_z) or isinstance(layer, Linear_with_z):
                modules_to_prune.append(name+'.weight')
        print('Pruning modeules',modules_to_prune)
        if pretrained:
            
            path = 'checkpoints/ResNet50-Dense.pth'
            #path = 'checkpoints/resnet50-19c8e357.pth'
            
            state_trained = torch.load(path,map_location=torch.device('cpu'))['state_dict']
            new_state_trained = model.state_dict()
            for k in state_trained:
                key = k[7:]
                if key in new_state_trained:
                    new_state_trained[key] = state_trained[k].view(new_state_trained[key].size())
                else:
                    print('Missing key',key)
            model.load_state_dict(new_state_trained,strict=False)
            
            #model.load_state_dict(torch.load(path))

        return model,train_dataset,test_dataset,criterion,modules_to_prune

def pass_data_through_first_block_opt(model, dataset, device, n_train_kept):

    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(device) 
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(device)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(device) 
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(device) 
    layers[0] = layers[0].to(device)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (dataset.shape[0], model.seqlen, model.config.hidden_size), dtype=dtype, device="cpu"
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp.cpu()
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i_batch in range(len(dataset)):
        # import ipdb;ipdb.set_trace()
        try:
            model(dataset[[i_batch]].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    attention_mask = cache['attention_mask']
    return inps, attention_mask

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    
# New functions
def get_dataset(name_dataset, dset_path, n_train_kept, get_item_func, arch=None, seed=0, activation_fn="relu", device="cpu", test_almost_sequential=0, test_update_test_vit=0, test_pass_first_modules=True, further_subsampling=-1):
    if name_dataset == 'mnist':
        train_dataset, test_dataset = mnist_get_datasets(dset_path, get_item_func)
        if n_train_kept!=-1:
            targets_train = np.array(train_dataset.targets)
            indices_train = np.arange(len(targets_train))
            indices_to_remove, indices_train_subsample, targets_to_remove, targets_train_subsample = train_test_split(indices_train, targets_train, test_size = n_train_kept, stratify = targets_train, random_state=0)
            # Reorder the elements in the dataset
            # train_dataset.data = np.array(train_dataset.data)
            train_dataset.targets = np.array(train_dataset.targets)
            train_dataset.data = train_dataset.data[np.concatenate([indices_train_subsample, indices_to_remove])]
            train_dataset.targets = train_dataset.targets[np.concatenate([indices_train_subsample, indices_to_remove])]
            train_dataset = torch.utils.data.Subset(train_dataset, np.arange(n_train_kept))

    elif name_dataset == 'cifar10':
        test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        train_random_transforms=True

        if train_random_transforms:
            train_transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

        class CustomCIFAR10(datasets.CIFAR10):
            def __getitem__(self, index: int):
                return (self, index)

        train_dataset = CustomCIFAR10(root=dset_path, train=True, download=True, transform=train_transform)
        test_dataset = CustomCIFAR10(root=dset_path, train=False, download=True, transform=test_transform)

        if n_train_kept!=-1:
            targets_train = np.array(train_dataset.targets)
            indices_train = np.arange(len(targets_train))
            indices_to_remove, indices_train_subsample, targets_to_remove, targets_train_subsample = train_test_split(indices_train, targets_train, test_size = n_train_kept, stratify = targets_train, random_state=0)
            # Reorder the elements in the dataset
            # train_dataset.data = np.array(train_dataset.data)
            train_dataset.targets = np.array(train_dataset.targets)
            train_dataset.data = train_dataset.data[np.concatenate([indices_train_subsample, indices_to_remove])]
            train_dataset.targets = train_dataset.targets[np.concatenate([indices_train_subsample, indices_to_remove])]
            train_dataset = torch.utils.data.Subset(train_dataset, np.arange(n_train_kept))

    elif name_dataset == 'cifar100':

        train_loader, test_loader = get_train_and_test_dataloader(
            dataset=name_dataset,
            batch_size=128,
            get_item_func=get_item_func
        )
        train_dataset = train_loader.dataset
        test_dataset = test_loader.dataset
        
        if n_train_kept!=-1:
            targets_train = np.array(train_dataset.targets)
            indices_train = np.arange(len(targets_train))
            indices_to_remove, indices_train_subsample, targets_to_remove, targets_train_subsample = train_test_split(indices_train, targets_train, test_size = n_train_kept, stratify = targets_train, random_state=0)
            # Reorder the elements in the dataset
            # train_dataset.data = np.array(train_dataset.data)
            train_dataset.targets = np.array(train_dataset.targets)
            train_dataset.data = train_dataset.data[np.concatenate([indices_train_subsample, indices_to_remove])]
            train_dataset.targets = train_dataset.targets[np.concatenate([indices_train_subsample, indices_to_remove])]
            train_dataset = torch.utils.data.Subset(train_dataset, np.arange(n_train_kept))
    

    elif name_dataset == 'imagenet':
        train_dataset, test_dataset = imagenet_get_datasets(dset_path, get_item_func)
        if n_train_kept!=-1:
            targets_train = np.array(train_dataset.targets)
            indices_train = np.arange(len(targets_train))
            indices_to_remove, indices_train_subsample, targets_to_remove, targets_train_subsample = train_test_split(indices_train, targets_train, test_size = n_train_kept, stratify = targets_train, random_state=seed)
            if further_subsampling!=-1:
                stratify_on = targets_train_subsample%further_subsampling
                indices_to_remove2, indices_train_final, _, _ = train_test_split(indices_train_subsample, targets_train[indices_train_subsample], test_size = further_subsampling, stratify = stratify_on, random_state=seed)
                indices_to_remove = np.hstack([indices_to_remove, indices_to_remove2])
                indices_train_subsample = indices_train_final
                n_train_kept = further_subsampling

            # Reorder the elements in the dataset
            train_dataset.imgs = np.array(train_dataset.imgs)
            train_dataset.samples = np.array(train_dataset.samples)
            train_dataset.targets = np.array(train_dataset.targets)
            train_dataset.imgs = train_dataset.imgs[np.concatenate([indices_train_subsample, indices_to_remove])]
            train_dataset.samples = train_dataset.samples[np.concatenate([indices_train_subsample, indices_to_remove])]
            train_dataset.targets = train_dataset.targets[np.concatenate([indices_train_subsample, indices_to_remove])]
            # End change of order
            path_imagenet = "../../imagenet_subsample"
            if not(os.path.exists(path_imagenet)):
                os.mkdir(path_imagenet)
            print("Saving imagenet training set...")
            for ind_sample in tqdm(range(n_train_kept)):#indices_train_subsample):
                old_path_img = train_dataset.imgs[ind_sample][0]
                name_img = old_path_img.split("/")[-1]
                new_path_img = path_imagenet+"/"+name_img
                # if name_img == "n02109047_1300.JPEG":
                #     import ipdb;ipdb.set_trace()
                if not(os.path.exists(new_path_img)):
                    shutil.copyfile(old_path_img, new_path_img)
                # test_loaded = False
                # while not(test_loaded):
                #     try:
                #         train_dataset.loader(new_path_img)
                #         test_loaded = True
                #     except:
                #         shutil.copyfile(old_path_img, new_path_img)

                train_dataset.imgs[ind_sample] = (new_path_img, train_dataset.imgs[ind_sample][1])
                train_dataset.samples[ind_sample] = (new_path_img, train_dataset.samples[ind_sample][1])

            print("Saving imagenet testing set...")
            for ind_sample in tqdm(range(len(test_dataset))):#indices_train_subsample):
                old_path_img = test_dataset.imgs[ind_sample][0]
                name_img = old_path_img.split("/")[-1]
                # if name_img == "ILSVRC2012_val_00015966.JPEG":
                #     import ipdb;ipdb.set_trace()
                new_path_img = path_imagenet+"/"+name_img
                if not(os.path.exists(new_path_img)):
                    shutil.copyfile(old_path_img, new_path_img)
                test_dataset.imgs[ind_sample] = (new_path_img, test_dataset.imgs[ind_sample][1])
                test_dataset.samples[ind_sample] = (new_path_img, test_dataset.samples[ind_sample][1])

            # train_dataset = torch.utils.data.Subset(train_dataset, indices_train_subsample)
            train_dataset = torch.utils.data.Subset(train_dataset, np.arange(n_train_kept))
        if "deit" in arch and test_pass_first_modules:
            #new_path_imagenet_test = arch+"_imagenet_test.npy"
            new_path_imagenet_test = arch+"_imagenet_test.pt"
            model, _, _ = get_model(arch, seed=seed, pretrained=True, with_z=False, gamma=1.0, prune_bias=False, activation_fn=activation_fn)
            model.eval()
            model.to(device)
            first_modules = nn.Sequential(*[model.patch_embed, Pos_embed(model), model.patch_drop, model.norm_pre])
            initialize_dataset(train_dataset, n_train_kept, name_dataset)
            initialize_dataset(test_dataset, -1, name_dataset)

            loader_train = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
            loader_test = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
            update_dataset(first_modules, device, first_modules, loader_train, None, n_train_kept, False, None, None, test_almost_sequential)
            if test_update_test_vit and new_path_imagenet_test in os.listdir(".."):
                # with open("../"+new_path_imagenet_test, 'rb') as f:
                #     data_numpy = np.load(f)
                # loader_test.dataset.data = torch.Tensor(data_numpy)
                loader_test.dataset.data = torch.load('../'+new_path_imagenet_test)
            elif test_update_test_vit:
                update_dataset(first_modules, device, first_modules, loader_test, None, -1, False, None, None, test_almost_sequential)
                # with open("../"+new_path_imagenet_test, 'wb') as f:
                #     np.save(f, loader_test.dataset.data.numpy())
                torch.save(loader_test.dataset.data, '../'+new_path_imagenet_test)

            current_dataset = loader_train
            if n_train_kept == -1:
                dataset = current_dataset.dataset
            else:
                dataset = current_dataset.dataset.dataset
            dataset.is_original = False
            loader_test.dataset.is_original = False

            # update_dataset(first_modules, device, None, loader_test, None, -1, False, None, None)
            train_dataset = loader_train.dataset
            test_dataset = loader_test.dataset
            del model, first_modules
    
    elif name_dataset in ["c4", "wikitext2", "ptb"]:
        if "facebook/opt" in arch:
            seqlen = 2048
        train_dataset, test_dataset = get_loaders(
                name_dataset, dset_path, nsamples=n_train_kept, seed=seed, model=arch, seqlen=seqlen
            )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        X_train_data = torch.cat([train_dataset[i][0] for i in range(len(train_dataset))])
        y_train_data = copy.deepcopy(X_train_data)
        model, _, _ = get_model(arch, seed=seed, pretrained=True, with_z=False, gamma=1.0, prune_bias=False, activation_fn=activation_fn)
        model.float()
        model.eval()
        X_train_data, train_attention_mask = pass_data_through_first_block_opt(model, X_train_data, device, n_train_kept)
        train_dataset = Dataset_LLM(X_train_data, y_train_data)
        test_dataset = test_dataset.input_ids
        n_samples_test = test_dataset.numel() // seqlen
        test_dataset = test_dataset[:,:n_samples_test*seqlen]
        test_dataset = test_dataset.reshape((n_samples_test, test_dataset.numel()//n_samples_test))
        y_test_data = copy.deepcopy(test_dataset)
        test_dataset, test_attention_mask = pass_data_through_first_block_opt(model, test_dataset, device, n_train_kept)
        test_dataset = Dataset_LLM(test_dataset, y_test_data)
        del model
        return (train_dataset, train_attention_mask), (test_dataset, test_attention_mask)
    else:
        print("NO DATASET FOUND FOR", name_dataset, dset_path)

    return train_dataset, test_dataset

def get_model(arch, seed, pretrained=True, with_z = True, gamma=1.0, prune_bias=True, activation_fn="relu"):
    if arch == 'mlpnet':
        torch.random.manual_seed(seed)
        model = MlpNet(args=None,dataset='mnist', with_z=with_z, gamma=gamma, prune_bias=prune_bias)

        criterion = torch.nn.functional.nll_loss
        if pretrained:
            state_trained = torch.load('checkpoints/mnist_25_epoch_93.97.ckpt',map_location=torch.device('cpu'))['model_state_dict']
            new_state_trained = OrderedDict()
            for k in state_trained:
                if 'mask' in k:
                    continue
                new_state_trained[k.split('.')[1]+'.'+k.split('.')[3]] = state_trained[k]
            model.load_state_dict(new_state_trained, strict=False)
        modules_to_prune = []
        for name, param in model.named_parameters():
            #print("name is {} and shape of param is {} \n".format(name, param.shape))
            layer_name,param_name = '.'.join(name.split('.')[:-1]),name.split('.')[-1]
            if param_name == 'bias':
                    continue
            if 'conv' in layer_name or 'fc' in layer_name:
                modules_to_prune.append(name)

    elif arch == 'resnet20':
        torch.random.manual_seed(seed)
        model = resnet20(gamma=gamma, with_z=with_z, prune_bias=prune_bias)
        if pretrained:
            state_trained = torch.load('checkpoints/resnet20_cifar10.pth.tar',map_location=torch.device('cpu'))['state_dict']
            new_state_trained = OrderedDict()
            for k in state_trained:
                new_state_trained[k[7:]] = state_trained[k]
            model.load_state_dict(new_state_trained, strict=False)

        criterion = torch.nn.functional.cross_entropy

        modules_to_prune = []
        for name, param in model.named_parameters():
            #print("name is {} and shape of param is {} \n".format(name, param.shape))
            layer_name,param_name = '.'.join(name.split('.')[:-1]),name.split('.')[-1]
            if param_name == 'bias':
                    continue
            if 'conv' in layer_name or 'fc' in layer_name:
                modules_to_prune.append(name)
    
    elif arch == 'resnet18':
        torch.random.manual_seed(seed)
        model = ResNet18(with_z=with_z, gamma=gamma, prune_bias=prune_bias)
        from opacus.validators import ModuleValidator
        model.train()
        model = ModuleValidator.fix(model.to("cpu"))
        if pretrained:
            state_trained = torch.load('checkpoints/lsr=01train_resnet_gn.pt', map_location=torch.device('cpu'))
            model.load_state_dict(state_trained, strict=False)

        criterion = torch.nn.functional.cross_entropy

        modules_to_prune = []
        for name, param in model.named_parameters():
            #print("name is {} and shape of param is {} \n".format(name, param.shape))
            layer_name,param_name = '.'.join(name.split('.')[:-1]),name.split('.')[-1]
            if param_name == 'bias':
                    continue
            if 'conv' in layer_name or 'fc' in layer_name:
                modules_to_prune.append(name)
    
    elif arch == 'mobilenetv1':
        torch.random.manual_seed(seed)
        model = mobilenet(gamma=gamma, with_z=with_z, prune_bias=prune_bias)

        criterion = torch.nn.functional.cross_entropy

        modules_to_prune = []
        for name, layer in model.named_modules():
            if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear) or isinstance(layer, Conv2d_with_z) or isinstance(layer, Linear_with_z):
                modules_to_prune.append(name+'.weight')


        if pretrained:
            path = 'checkpoints/MobileNetV1-Dense-STR.pth'
            state_trained = torch.load(path,map_location=torch.device('cpu'))['state_dict']
            new_state_trained = model.state_dict()
            for k in state_trained:
                key = k[7:]
                if key in new_state_trained:
                    new_state_trained[key] = state_trained[k].view(new_state_trained[key].size())
                else:
                    print('Missing key',key)
            model.load_state_dict(new_state_trained, strict=False)
    
    elif arch == 'resnet50':
        torch.random.manual_seed(seed)
        model = torch_resnet50(weights=None, with_z=with_z, gamma=gamma, prune_bias=prune_bias)

        criterion = torch.nn.functional.cross_entropy

        modules_to_prune = []
        for name, layer in model.named_modules():
            if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear) or isinstance(layer, Conv2d_with_z) or isinstance(layer, Linear_with_z):
                modules_to_prune.append(name+'.weight')
        print('Pruning modeules',modules_to_prune)
        if pretrained:
            
            path = 'checkpoints/ResNet50-Dense.pth'
            #path = 'checkpoints/resnet50-19c8e357.pth'
            
            state_trained = torch.load(path,map_location=torch.device('cpu'))['state_dict']
            new_state_trained = model.state_dict()
            for k in state_trained:
                key = k[7:]
                if key in new_state_trained:
                    new_state_trained[key] = state_trained[k].view(new_state_trained[key].size())
                else:
                    print('Missing key',key)
            model.load_state_dict(new_state_trained, strict=False)

    elif "facebook/opt" in arch:
        torch.random.manual_seed(seed)
        model = get_opt(arch, "/home/gridsan/gafriat/Sparse_NN_shared/LLM/model/", cached=True, with_z=with_z, gamma=gamma, prune_bias=prune_bias, activation_fn=activation_fn)
        
        #import ipdb;ipdb.set_trace()
        model.config.use_cache = False

        criterion = torch.nn.functional.cross_entropy

        modules_to_prune = -1

    elif arch == 'deit_tiny_patch16_224':
        torch.random.manual_seed(seed)
        model = deit_tiny_patch16_224(pretrained=False, with_z=with_z, gamma=gamma)
        if pretrained:
            state_trained = torch.load('checkpoints/deit_tiny_patch16_224-a1311bcf.pth',map_location=torch.device('cpu'))
            model.load_state_dict(state_trained['model'], strict=False)

        criterion = torch.nn.functional.cross_entropy

        modules_to_prune = []

    elif arch == 'deit_small_patch16_224':
        torch.random.manual_seed(seed)
        model = deit_small_patch16_224(pretrained=False, with_z=with_z, gamma=gamma)
        if pretrained:
            state_trained = torch.load('checkpoints/deit_small_patch16_224-cd65a155.pth',map_location=torch.device('cpu'))
            model.load_state_dict(state_trained['model'], strict=False)

        criterion = torch.nn.functional.cross_entropy

        modules_to_prune = []

    elif arch == 'deit_base_patch16_224':
        torch.random.manual_seed(seed)
        model = deit_base_patch16_224(pretrained=False, with_z=with_z, gamma=gamma)
        if pretrained:
            state_trained = torch.load('checkpoints/deit_base_patch16_224-b5f2ef4d.pth',map_location=torch.device('cpu'))
            model.load_state_dict(state_trained['model'], strict=False)

        criterion = torch.nn.functional.cross_entropy

        modules_to_prune = []
    
    return model,criterion,modules_to_prune



