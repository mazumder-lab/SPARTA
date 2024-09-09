import os

import torch
import torch.cuda
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from medmnist import OrganAMNIST


class SqueezeTargetTransform:
    def __call__(self, target):
        return target.squeeze()


def get_train_and_test_dataloader(
    dataset="cifar10",
    batch_size=1000,
    shuffle=True,
    test_batch_size=128,
):
    if dataset == "cifar100":
        print("==> Preparing CIFAR 100 data..")
        normalize = transforms.Normalize(
            (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
            (0.2673342858792401, 0.2564384629170883, 0.27615047132568404),
        )
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )

        trainset = torchvision.datasets.CIFAR100(
            root="../datasets", train=True, download=True, transform=transform_train
        )

        cifar100_training_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=shuffle, num_workers=2
        )

        testset = torchvision.datasets.CIFAR100(
            root="../datasets", train=False, download=True, transform=transform_test
        )
        cifar100_test_loader = torch.utils.data.DataLoader(
            testset, batch_size=test_batch_size, shuffle=False, num_workers=1
        )
        return cifar100_training_loader, cifar100_test_loader

    elif dataset == "cifar10":
        print("==> Preparing CIFAR 10 data..")
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )

        trainset = torchvision.datasets.CIFAR10(
            root="../datasets", train=True, download=True, transform=transform_train
        )

        cifar10_training_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=shuffle, num_workers=2
        )

        testset = torchvision.datasets.CIFAR10(
            root="../datasets", train=False, download=True, transform=transform_test
        )
        cifar10_test_loader = torch.utils.data.DataLoader(
            testset, batch_size=test_batch_size, shuffle=False, num_workers=1
        )
        return cifar10_training_loader, cifar10_test_loader
    
    
    elif dataset == "OrganAMNIST":
        print("==> Preparing OrganAMNIST data..")
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Repeat the single channel 3 times
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalization for 3 channels
        ])

        train_dataset = OrganAMNIST(root="../datasets", split='train', transform=data_transform, target_transform=SqueezeTargetTransform(), download=True)
        test_dataset = OrganAMNIST(root="../datasets", split='test', transform=data_transform, target_transform=SqueezeTargetTransform(), download=True)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False)
        
        return train_loader, test_loader



    else:
        raise Exception("Unknown dataset, exiting..")
