import os
import sys
sys.path.append('yolov5')

import numpy as np
import torch
from opacus.validators import ModuleValidator

from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from models.mobilenet import mobilenet
from torchvision.models import resnet50 as torch_resnet50
from collections import OrderedDict
from models.resnet_cifar10 import resnet20
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from resnet_mehdi import ResNet18
from dataset_utils_mehdi import get_train_and_test_dataloader

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def random_subset_old(data, nsamples, seed):
    set_seed(seed)
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    return Subset(data, idx[:nsamples])

def random_subset(train_dataset, nsamples, seed):
    if nsamples!=-1:
        targets_train = np.array(train_dataset.targets)
        indices_train = np.arange(len(targets_train))
        indices_to_remove, indices_train_subsample, targets_to_remove, targets_train_subsample = train_test_split(indices_train, targets_train, test_size = nsamples, stratify = targets_train, random_state=0)
        train_dataset = torch.utils.data.Subset(train_dataset, indices_train_subsample)
    return train_dataset

def imagenet_get_datasets_old(data_dir):

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

    train_dataset = datasets.ImageFolder(train_dir, train_transform)

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    test_dataset = datasets.ImageFolder(test_dir, test_transform)

    return train_dataset, test_dataset

def imagenet_get_datasets(data_dir):

    train_dir = os.path.join(data_dir, 'raw_train')
    test_dir = os.path.join(data_dir, 'raw_val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    train_transform = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip()
    ]
    
    train_transform += [
        transforms.ToTensor(),
        normalize,
    ]
    train_transform = transforms.Compose(train_transform)

    train_dataset = datasets.ImageFolder(train_dir, train_transform)

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    test_dataset = datasets.ImageFolder(test_dir, test_transform)

    return train_dataset, test_dataset

def compute_acc(model,dataloader,device='cpu',verbose=False):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    i = 0
    with torch.no_grad():
        for data in tqdm(dataloader):
            i+=1
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            images=images
            labels=labels
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if verbose and i%10 == 0:
                print(total,correct)

            del images,labels,outputs

    return 100 * correct / total

def model_factory(arch, dset_path, pretrained=True, seed = 0, nsamples=-1, initialize_bn = True, batch_size = 128, name_dataset=None):
    
    if arch == 'resnet50':
        model = torch_resnet50()
        train_dataset,test_dataset = imagenet_get_datasets(dset_path)

        if pretrained:
            path = '../network_pruning/checkpoints/ResNet50-Dense.pth'
            
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
        
        if nsamples!=-1:
            train_dataset = random_subset(train_dataset, nsamples, seed)
        data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8,pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=8,pin_memory=True)

        if initialize_bn:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.train()
            compute_acc(model, data_loader, device)
            model.eval()

        return model,data_loader,test_loader
    
    elif arch == 'mobilenetv1':
        model = mobilenet()
        train_dataset,test_dataset = imagenet_get_datasets(dset_path)

        if pretrained:
            path = '../network_pruning/checkpoints/MobileNetV1-Dense-STR.pth'
            state_trained = torch.load(path,map_location=torch.device('cpu'))['state_dict']
            new_state_trained = model.state_dict()
            for k in state_trained:
                key = k[7:]
                if key in new_state_trained:
                    new_state_trained[key] = state_trained[k].view(new_state_trained[key].size())
                else:
                    print('Missing key',key)
            model.load_state_dict(new_state_trained,strict=False)
        if nsamples!=-1:
            train_dataset = random_subset(train_dataset, nsamples, seed)
        data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=8,pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True,num_workers=8,pin_memory=True)

        if initialize_bn:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.train()
            compute_acc(model, data_loader, device)
            model.eval()

        return model,data_loader,test_loader

    elif arch == 'resnet20':
        state_trained = torch.load('../network_pruning/checkpoints/resnet20_cifar10.pth.tar',map_location=torch.device('cpu'))['state_dict']
        new_state_trained = OrderedDict()
        for k in state_trained:
            new_state_trained[k[7:]] = state_trained[k]

        model = resnet20()
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

        if nsamples!=-1:
            train_dataset = random_subset(train_dataset, nsamples, seed)
        data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=8,pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True,num_workers=8,pin_memory=True)
        
        if initialize_bn:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.train()
            compute_acc(model, data_loader, device)
            model.eval()

    elif arch == 'resnet18':
        state_trained = torch.load('lsr=01train_resnet_gn.pt',map_location=torch.device('cpu'))
        model = ResNet18(num_classes=100)
        model.train()
        model = ModuleValidator.fix(model.to("cpu"))
        if pretrained:
            model.load_state_dict(state_trained)

        if name_dataset=="cifar10":
            model.linear = torch.nn.Linear(
                in_features=model.linear.in_features,
                out_features=10,
                bias=model.linear.bias is not None,
            )

        data_loader, test_loader = get_train_and_test_dataloader(
            dataset=name_dataset,
            batch_size=batch_size,
        )

        # if nsamples!=-1:
        #     train_dataset = random_subset(train_dataset, nsamples, seed)
        # data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=8,pin_memory=True)
        # test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True,num_workers=8,pin_memory=True)
        
        if initialize_bn:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.train()
            compute_acc(model, data_loader, device)
            model.eval()
        

        return model,data_loader,test_loader
