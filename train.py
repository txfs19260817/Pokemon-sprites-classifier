import os
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm

import utils
from model import training_model
from transformation import transform

batch_size = 32
num_workers = 2
epochs = 200


def train(dataset_root_path, model_name):
    train_path, val_path = os.path.join(dataset_root_path, "train"), os.path.join(dataset_root_path, "test")
    dataset = torchvision.datasets.ImageFolder(root=train_path, transform=transform['train'])
    val_percent = 0.05
    val_amount = int(dataset.__len__() * val_percent)
    trainset, valset = torch.utils.data.random_split(dataset, [dataset.__len__() - val_amount, val_amount])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=1)

    classes = dataset.classes
    iters = len(trainloader)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = training_model(model_name, len(classes), pretrained=True)
    model = model.to(device)
    weight_path = model_name + '.pth'

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, epochs)

    pbar = tqdm(range(epochs))
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

            # print statistics
            running_loss += loss.item()

        pbar.set_description('loss: %.5f, lr = %.5f' % (running_loss, optimizer.param_groups[0]['lr']))
        pbar.refresh()  # to show immediately the update
        if ei % (epochs // 10) == epochs // 10 - 1:
            validation(valloader, model, device, classes)

    torch.save(model.state_dict(), weight_path)
    print('Finished training, and weight was saved in ' + weight_path)
    utils.generate_label_csv(classes)
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

    print('Accuracy of the network on images: %d %%' % (100 * correct / total))


if __name__ == '__main__':
    train('dataset', 'mobilenetv2')
