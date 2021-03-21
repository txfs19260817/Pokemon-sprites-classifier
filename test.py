import os
import torch
import torchvision
from PIL import Image
from model import training_model
from transformation import transform


def test(dataset_root_path, model_name, image_path):
    train_path = os.path.join(dataset_root_path, "train")
    dataset = torchvision.datasets.ImageFolder(root=train_path, transform=transform['train'])
    classes = dataset.classes
    print(classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = training_model(model_name, len(classes), pretrained=True)
    model = model.to(device)
    model.eval()
    weight_path = model_name+'.pth'
    model.load_state_dict(torch.load(weight_path))

    img = Image.open(image_path)
    img = transform['train'](img).unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        output = model(img)
    _, predicted = torch.max(output.data, 1)
    print('Result: ', predicted.data, classes[predicted[0]])


if __name__ == '__main__':
    test('dataset', 'mobilenetv2',
         "dataset/train/amoonguss/amoonguss-1616210225.png")
