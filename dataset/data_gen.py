import os
import shutil
from pathlib import Path


# generate dataset from individual images for torchvision.datasets.ImageFolder
def generate_dataset(root):
    image_paths = os.listdir(root)
    images = [i for i in image_paths if '.png' in i]
    names = sorted(list(set([i.split(".")[0].split("-")[0] for i in images])))
    # mkdir
    for n in names:
        Path(os.path.join(root, n)).mkdir(exist_ok=True)  # python 3.5 above
    for p in image_paths:
        shutil.move(os.path.join(root, p), os.path.join(
            root, p.split(".")[0].split("-")[0]))


if __name__ == '__main__':
    generate_dataset('train')
