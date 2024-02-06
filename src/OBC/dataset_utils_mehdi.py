import torch
import torch.cuda
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DistributedSampler

from conf import settings

# from transformers import ViTFeatureExtractor


def get_training_dataloader(
    mean,
    std,
    batch_size=16,
    num_workers=1,
    world_size=1,
    rank=0,
    shuffle=True,
    ViT=False,
):
    """return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """
    if ViT:
        transform_train = transforms.Compose(
            [
                # transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(15),
                transforms.ToTensor(),
            ]
        )
    else:
        transform_train = transforms.Compose(
            [
                # transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    # cifar100_training = CIFAR100Train(path, transform=transform_train)
    cifar100_training = torchvision.datasets.CIFAR100(
        root="../../datasets", train=True, download=True, transform=transform_train
    )
    if world_size > 1:  # Create a parallel sampler for multi-gpu case
        sampler = DistributedSampler(cifar100_training, num_replicas=world_size, rank=rank, shuffle=shuffle)
        cifar100_training_loader = torch.utils.data.DataLoader(
            cifar100_training,
            shuffle=False,
            sampler=sampler,
            num_workers=num_workers,
            batch_size=batch_size,
            pin_memory=False,
        )
    else:
        cifar100_training_loader = torch.utils.data.DataLoader(
            cifar100_training,
            shuffle=shuffle,
            num_workers=num_workers,
            batch_size=batch_size,
        )

    return cifar100_training_loader


def get_test_dataloader(mean, std, batch_size=16, num_workers=1, shuffle=False, ViT=False):
    """return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """
    if ViT:
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
    else:
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    # cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar100_test = torchvision.datasets.CIFAR100(
        root="../../datasets", train=False, download=True, transform=transform_test
    )
    cifar100_test_loader = torch.utils.data.DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size
    )

    return cifar100_test_loader


def get_train_and_test_dataloader(
    dataset="cifar10",
    batch_size=1000,
    world_size=1,
    rank=0,
    shuffle=True,
    ViT=False,
    use_dp=False,
    test_batch_size=128,
):
    if dataset == "cifar100":
        print("==> Preparing CIFAR 100 data..")
        cifar100_training_loader = get_training_dataloader(
            settings.CIFAR100_TRAIN_MEAN,
            settings.CIFAR100_TRAIN_STD,
            num_workers=1,
            batch_size=batch_size,
            world_size=world_size,
            rank=rank,
            ViT=ViT,
        )

        cifar100_test_loader = get_test_dataloader(
            settings.CIFAR100_TRAIN_MEAN,
            settings.CIFAR100_TRAIN_STD,
            num_workers=1,
            batch_size=test_batch_size,
            ViT=ViT,
        )
        return cifar100_training_loader, cifar100_test_loader
    elif dataset == "cifar10":
        print("==> Preparing CIFAR 10 data..")
        if ViT:
            transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
        else:
            transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )
            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )

        # In DP, do not use data augmentation, it is already a strong regularizer (see cvpr2021)
        if use_dp:
            transform_train = transform_test
        trainset = torchvision.datasets.CIFAR10(
            root="../../datasets", train=True, download=True, transform=transform_train
        )

        if world_size > 1:  # Create a parallel sampler for multi-gpu case
            sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank, shuffle=shuffle)
            cifar10_training_loader = torch.utils.data.DataLoader(
                trainset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=1,
                sampler=sampler,
                pin_memory=False,
            )
        else:
            cifar10_training_loader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, shuffle=shuffle, num_workers=2
            )

        testset = torchvision.datasets.CIFAR10(
            root="../../datasets", train=False, download=True, transform=transform_test
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