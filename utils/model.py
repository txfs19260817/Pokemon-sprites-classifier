import torch.nn as nn
from torchvision import models

supported_models = [
    "alexnet",
    "mobilenetv2",
    "mobilenetv3",
    "resnet18",
    "shufflenetv2",
    "shufflenetv2small"
]


def training_model(model_name, num_classes, pretrained=True):
    if 'alex' in model_name.lower():
        return alexnet(num_classes, pretrained)
    if 'mobilenetv2' in model_name.lower():
        return mobilenetv2(num_classes, pretrained)
    if 'mobilenetv3' in model_name.lower():
        return mobilenetv3(num_classes, pretrained)
    if 'resnet' in model_name.lower():
        return resnet(num_classes, pretrained)
    if 'shufflenetv2' in model_name.lower():
        small = True if 'small' in model_name.lower() else False
        return shufflenetv2(num_classes, pretrained, small)
    raise ModuleNotFoundError("the model name: " + model_name + " is invalid")


def alexnet(num_classes, pretrained=True):
    net = models.alexnet(pretrained=pretrained)
    net.classifier[6] = nn.Linear(4096, num_classes)
    return net


def mobilenetv2(num_classes, pretrained=True):
    net = models.mobilenet_v2(pretrained=pretrained)
    net.classifier[1] = nn.Linear(1280, num_classes)
    return net


def mobilenetv3(num_classes, pretrained=True):
    net = models.mobilenet_v3_small(pretrained=pretrained)
    net.classifier[-1] = nn.Linear(1024, num_classes)
    return net


def resnet(num_classes, pretrained=True):
    net = models.resnet18(pretrained=pretrained)
    net.fc = nn.Linear(512, num_classes)
    return net


def shufflenetv2(num_classes, pretrained=True, small=False):
    net = models.shufflenet_v2_x0_5(pretrained) if small else models.shufflenet_v2_x1_0(pretrained)
    net.fc = nn.Linear(1024, num_classes)
    return net
