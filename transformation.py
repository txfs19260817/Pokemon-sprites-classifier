import torchvision.transforms as transforms

size = [224, 224]
mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)

transform = {
    'train': transforms.Compose([
        transforms.Resize(size),  # Resizing the image as the VGG only take 224 x 244 as input size
        # transforms.RandomHorizontalFlip(), # Flip the data horizontally
        # TODO if it is needed, add the random crop
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]),
    'test': transforms.Compose([
        transforms.Resize(size),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=mean, std=std)
    ])
}
