import os

import torch
import torch.cuda
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def get_train_and_test_dataloader(
    dataset="cifar10",
    batch_size=1000,
    shuffle=True,
    test_batch_size=128,
):
    if dataset == "imagenet32":
        IMAGENET32_PATH = "/home/gridsan/mmakni/BoxImageNet32/raw"
        train_dir = os.path.join(IMAGENET32_PATH, "train")
        val_dir = os.path.join(IMAGENET32_PATH, "val")

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
        train_dataset = datasets.ImageFolder(train_dir, train_transform)

        test_transform = transforms.Compose([transforms.ToTensor(), normalize])
        val_dataset = datasets.ImageFolder(val_dir, test_transform)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            num_workers=1,
            batch_size=batch_size,
            pin_memory=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )
        return train_loader, val_loader

    elif dataset == "cifar100":
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
    else:
        raise Exception("Unknown dataset, exiting..")


def collate_fn(tokenizer, batch, device=None):
    out_tensor = torch.zeros(batch.size()[0], 3, 224, 224)
    for i in range(batch.size()[0]):
        img = batch[i, :, :, :].to("cpu")
        out_tensor[i, :, :, :] = tokenizer(img, return_tensors="pt")["pixel_values"]
    if device is not None:
        out_tensor = out_tensor.to(device)
    return out_tensor
