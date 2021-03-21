import torch.nn as nn
from torchvision import models


supported_models = [
    "alexnet",
    "mobilenetv2",
    "resnet18"
]


def training_model(model_name, num_classes, pretrained=True):
    if 'alex' in model_name.lower():
        return alexnet(num_classes, pretrained)
    if 'mobilenetv2' in model_name.lower():
        return mobilenetv2(num_classes, pretrained)
    if 'resnet' in model_name.lower():
        return resnet(num_classes, pretrained)
    raise ModuleNotFoundError("the model name: " + model_name + " is invalid")


def alexnet(num_classes, pretrained=True):
    net = models.alexnet(pretrained=pretrained)
    net.classifier[6] = nn.Linear(4096, num_classes)
    return net


def mobilenetv2(num_classes, pretrained=True):
    net = models.mobilenet_v2(pretrained=True)
    net.classifier[1] = nn.Linear(1280, num_classes)
    return net


def resnet(num_classes, pretrained=True):
    net = models.resnet18(pretrained=True)
    net.fc = nn.Linear(512, num_classes)
    return net
