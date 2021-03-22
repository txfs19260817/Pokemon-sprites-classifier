import argparse
import os

import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm

from utils.labeling import generate_label_csv
from utils.model import training_model, supported_models
from utils.transformation import transform


def train(args):
    train_path, val_path = os.path.join(args.dataset_root_path, "train"), os.path.join(args.dataset_root_path, "test")
    trainset = torchvision.datasets.ImageFolder(root=train_path, transform=transform['train'])
    valset = torchvision.datasets.ImageFolder(root=val_path, transform=transform['test'])
    trainloader = torch.utils.data.DataLoader(trainset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    valloader = torch.utils.data.DataLoader(valset, 1, shuffle=False, num_workers=1)

    classes = trainset.classes
    iters = len(trainloader)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weight_path = args.arch + '.pth'

    model = training_model(args.arch, len(classes), pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.epochs)

    pbar = tqdm(range(args.epochs))
    for ei, epoch in enumerate(pbar):  # loop over the dataset multiple times
        running_loss = 0.0
        model.train()
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step(epoch + i / iters)

            # stat
            running_loss += loss.item()

        pbar.set_description('loss: %.5f, lr = %.5f' % (running_loss, optimizer.param_groups[0]['lr']))
        pbar.refresh()  # to show immediately the update
        if ei % (args.epochs // 10) == args.epochs // 10 - 1:
            validation(valloader, model, device, classes)

    torch.save(model.state_dict(), weight_path)
    print('Finished training, and weight was saved in ' + weight_path)
    generate_label_csv(classes)
    print('Generated label.csv')


def validation(testloader, model, device, classes):
    print('\nStart validation on the training set')
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if classes[labels[0]] != classes[predicted[0]]:
                print("expected: ", classes[labels[0]], "predicted: ", classes[predicted[0]])

    print('Accuracy of the network on images: %.2f %%' % (100 * correct / total))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Pokemon species classifier.')
    parser.add_argument('-d', '--dataset-root-path', metavar='DIR',
                        help='root path to dataset (default: ./dataset)', default="dataset")
    parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('-e', '--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('-j', '--num-workers', type=int, default=2, metavar='N',
                        help='number of workers to sample data (default: 2)')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                        metavar='LR', help='initial learning rate (default: 0.0001)', dest='lr')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='mobilenetv2',
                        choices=supported_models,
                        help='model architecture: ' +
                             ' | '.join(supported_models) +
                             ' (default: mobilenetv2)')
    args = parser.parse_args()
    train(args)
