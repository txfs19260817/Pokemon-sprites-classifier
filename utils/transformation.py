import random

import torch
import torchvision.transforms as transforms

size = [224, 224]
mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)
kernels = (1, 3)

transform = {
    'train': transforms.Compose([
        transforms.Resize(size),  # Resizing the image as the VGG only take 224 x 244 as input size
        transforms.RandomApply(torch.nn.ModuleList([
            transforms.GaussianBlur(random.choice(kernels)),
        ]), p=0.1),
        # transforms.RandomHorizontalFlip(), # Flip the data horizontally
        # TODO if it is needed, add the random crop
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]),
    'test': transforms.Compose([
        transforms.Resize(size),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]),
}
