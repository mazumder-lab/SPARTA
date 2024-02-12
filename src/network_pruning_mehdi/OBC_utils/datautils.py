import os
import sys
sys.path.append('yolov5')

import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split


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
        # # Reorder the elements in the dataset
        # train_dataset.imgs = np.array(train_dataset.imgs)
        # train_dataset.samples = np.array(train_dataset.samples)
        # train_dataset.targets = np.array(train_dataset.targets)
        # train_dataset.imgs = train_dataset.imgs[np.concatenate([indices_train_subsample, indices_to_remove])]
        # train_dataset.samples = train_dataset.samples[np.concatenate([indices_train_subsample, indices_to_remove])]
        # train_dataset.targets = train_dataset.targets[np.concatenate([indices_train_subsample, indices_to_remove])]
        # # End change of order
        train_dataset = torch.utils.data.Subset(train_dataset, indices_train_subsample)
    return train_dataset


_IMAGENET_RGB_MEANS = (0.485, 0.456, 0.406)
_IMAGENET_RGB_STDS = (0.229, 0.224, 0.225)

def get_imagenet_old(path, noaug=False):
    img_size = 224  # standard
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_RGB_MEANS, std=_IMAGENET_RGB_STDS),
    ])
    non_rand_resize_scale = 256.0 / 224.0  # standard
    test_transform = transforms.Compose([
        transforms.Resize(round(non_rand_resize_scale * img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_RGB_MEANS, std=_IMAGENET_RGB_STDS),
    ])

    train_dir = os.path.join(os.path.expanduser(path), 'raw_train')
    test_dir = os.path.join(os.path.expanduser(path), 'raw_val')

    if noaug:
        train_dataset = datasets.ImageFolder(train_dir, test_transform)
    else:
        train_dataset = datasets.ImageFolder(train_dir, train_transform)
    test_dataset = datasets.ImageFolder(test_dir, test_transform)

    return train_dataset, test_dataset


def get_imagenet(data_dir):

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


class YOLOv5Wrapper(Dataset):
    def __init__(self, original):
        self.original = original
    def __len__(self):
        return len(self.original)
    def __getitem__(self, idx):
        tmp = list(self.original[idx])
        tmp[0] = tmp[0].float() / 255
        return tmp

def get_coco(path, batchsize):
    from yolov5.utils.datasets import LoadImagesAndLabels
    train_data = LoadImagesAndLabels(
        os.path.join(path, 'images/calib'), batch_size=batchsize
    )
    train_data = YOLOv5Wrapper(train_data)
    train_data.collate_fn = LoadImagesAndLabels.collate_fn
    test_data = LoadImagesAndLabels(
        os.path.join(path, 'images/val2017'), batch_size=batchsize, pad=.5
    )
    test_data = YOLOv5Wrapper(test_data)
    test_data.collate_fn = LoadImagesAndLabels.collate_fn
    return train_data, test_data


DEFAULT_PATHS = {
    'imagenet': [
        '../imagenet'
    ],
    'coco': [
        '../coco'
    ]
}

def get_loaders(
    name, path='', batchsize=-1, workers=8, nsamples=1024, seed=0,
    noaug=False
):
    if name == 'squad':
        if batchsize == -1:
            batchsize = 16
        import bertsquad
        set_seed(seed)
        return bertsquad.get_dataloader(batchsize, nsamples), None

    if not path:
        for path in DEFAULT_PATHS[name]:
            if os.path.exists(path):
                break

    if name == 'imagenet':
        if batchsize == -1:
            batchsize = 128
        train_data, test_data = get_imagenet(path, noaug=noaug)
        train_data = random_subset(train_data, nsamples, seed)
    if name == 'coco':
        if batchsize == -1:
            batchsize = 16
        train_data, test_data = get_coco(path, batchsize)

    collate_fn = train_data.collate_fn if hasattr(train_data, 'collate_fn') else None
    trainloader = DataLoader(
        train_data, batch_size=batchsize, num_workers=workers, pin_memory=True, shuffle=True,
        collate_fn=collate_fn
    )
    collate_fn = test_data.collate_fn if hasattr(test_data, 'collate_fn') else None
    testloader = DataLoader(
        test_data, batch_size=batchsize, num_workers=workers, pin_memory=True, shuffle=False,
        collate_fn=collate_fn
    )

    return trainloader, testloader
