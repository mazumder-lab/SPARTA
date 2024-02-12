"""ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils_model import Conv2d_with_z, Linear_with_z, conv3x3


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, with_z=True, gamma=1.0, prune_bias=False):
        super(BasicBlock, self).__init__()
        self.gamma=gamma
        if with_z:
            self.conv1 = Conv2d_with_z(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, gamma=gamma, prune_bias=prune_bias)
        else:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if with_z:
            self.conv2 = Conv2d_with_z(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, gamma=gamma, prune_bias=prune_bias)
        else:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            if with_z:
                self.shortcut = nn.Sequential(
                    Conv2d_with_z(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                        gamma=gamma, 
                        prune_bias=prune_bias
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, in_planes, planes, stride=1):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(self.expansion * planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion * planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(
#                     in_planes,
#                     self.expansion * planes,
#                     kernel_size=1,
#                     stride=stride,
#                     bias=False,
#                 ),
#                 nn.BatchNorm2d(self.expansion * planes),
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, with_z=True, gamma=1.0, prune_bias=False):
        super(ResNet, self).__init__()
        self.gamma = gamma
        self.in_planes = 64
        if with_z:
            self.conv1 = Conv2d_with_z(3, 64, kernel_size=3, stride=1, padding=1, bias=False, gamma=gamma, prune_bias=prune_bias)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, with_z=with_z, gamma=gamma, prune_bias=prune_bias)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, with_z=with_z, gamma=gamma, prune_bias=prune_bias)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, with_z=with_z, gamma=gamma, prune_bias=prune_bias)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, with_z=with_z, gamma=gamma, prune_bias=prune_bias)
        if with_z:
            self.linear = Linear_with_z(512 * block.expansion, num_classes, gamma=gamma, prune_bias=prune_bias)
        else:
            self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, with_z, gamma, prune_bias):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, with_z=with_z, gamma=gamma, prune_bias=prune_bias))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes=100, with_z=True, prune_bias = False, gamma=1.0):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, with_z=with_z, gamma=gamma, prune_bias=prune_bias)


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


# def ResNet50(num_classes=100):
#     return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


# def ResNet101():
#     return ResNet(Bottleneck, [3, 4, 23, 3])


# def ResNet152():
#     return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(len(net.get_parameter_groups("hsn")))
    # print (list(net.named_parameters()))
    print(y.size())

import torch
import torch.cuda
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DistributedSampler

# from transformers import ViTFeatureExtractor


def get_training_dataloader(
    mean,
    std,
    get_item_func,
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
    class CustomCIFAR100(torchvision.datasets.CIFAR100):
        def __getitem__(self, index: int):
            return get_item_func(self, index)

    # cifar100_training = CIFAR100Train(path, transform=transform_train)
    cifar100_training = CustomCIFAR100(
        root="../datasets", train=True, download=True, transform=transform_train
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


def get_test_dataloader(mean, std, get_item_func, batch_size=16, num_workers=1, shuffle=False, ViT=False):
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

    class CustomCIFAR100(torchvision.datasets.CIFAR100):
        def __getitem__(self, index: int):
            return get_item_func(self, index)

    # cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar100_test = CustomCIFAR100(
        root="../datasets", train=False, download=True, transform=transform_test
    )
    cifar100_test_loader = torch.utils.data.DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size
    )

    return cifar100_test_loader


def get_train_and_test_dataloader(
    dataset="cifar100",
    batch_size=1000,
    world_size=1,
    rank=0,
    shuffle=True,
    ViT=False,
    use_dp=False,
    test_batch_size=128,
    get_item_func=None
):
    CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

    if dataset == "cifar100":
        print("==> Preparing CIFAR 100 data..")
        cifar100_training_loader = get_training_dataloader(
            CIFAR100_TRAIN_MEAN,
            CIFAR100_TRAIN_STD,
            num_workers=1,
            batch_size=batch_size,
            world_size=world_size,
            rank=rank,
            ViT=ViT,
            get_item_func=get_item_func
        )

        cifar100_test_loader = get_test_dataloader(
            CIFAR100_TRAIN_MEAN,
            CIFAR100_TRAIN_STD,
            num_workers=1,
            batch_size=test_batch_size,
            ViT=ViT,
            get_item_func=get_item_func
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
            root="../datasets", train=True, download=True, transform=transform_train
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