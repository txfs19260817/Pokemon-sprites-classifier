import argparse
import random

from PIL import Image

from utils.labeling import resize_and_crop


def labeling(image_path):
    image = Image.open(image_path).convert('RGB')
    crop_list = resize_and_crop(image)
    for i, img in enumerate(crop_list):
        img.save('{}-{}.png'.format(i, random.randint(1, 10000000000)))


if __name__ == '__main__':
    print("[INFO] Please ensure that the input image(s) should be SCREENSHOT rather than photograph.")
    parser = argparse.ArgumentParser(description='A labeling tool helps resize and crop a team preview screenshot '
                                                 'and output 6 sprite thumbnails.')
    parser.add_argument('filenames', metavar='FILE', type=str, nargs='+', help='images to be processed')
    args = parser.parse_args()
    for p in args.filenames:
        labeling(p)
