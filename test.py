import argparse
import os
import torch
import torchvision
from PIL import Image

from utils.model import training_model, supported_models
from utils.transformation import transform


def test(args):
    train_path = os.path.join(args.dataset_root_path, "train")
    dataset = torchvision.datasets.ImageFolder(root=train_path, transform=transform['test'])
    classes = dataset.classes
    print(classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weight_path = args.arch + '.pth'

    model = training_model(args.arch, len(classes), pretrained=True)
    model = model.to(device)
    model.load_state_dict(torch.load(weight_path))
    model.eval()

    for image_path in args.filenames:
        img = Image.open(image_path)
        img = transform['test'](img).unsqueeze(0)
        img = img.to(device)

        with torch.no_grad():
            output = model(img)
        _, predicted = torch.max(output.data, 1)
        print('Result: ', predicted.data, classes[predicted[0]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test the trained Pokemon species classifier.')
    parser.add_argument('filenames', metavar='FILE', type=str, nargs='+',
                        help='images to be tested')
    parser.add_argument('-d', '--dataset-root-path', metavar='DIR',
                        help='root path to dataset (default: ./dataset)', default="dataset")
    parser.add_argument('-a', '--arch', metavar='ARCH', default='mobilenetv2',
                        choices=supported_models,
                        help='model architecture: ' +
                             ' | '.join(supported_models) +
                             ' (default: mobilenetv2)')
    args = parser.parse_args()
    test(args)

# filenames = [
#     'dataset/train/incineroar/incineroar-1616205113.png',
#     'dataset/train/zacian/zacian-1616210152.png',
# ]
