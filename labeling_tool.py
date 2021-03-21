import argparse
import random

from PIL import Image

# crop parameters
expected_size = (1280, 720)
begin = (85, 20)
box = (75, 75)
offset_x, offset_y = 588, 186


def labeling(image_path):
    image = Image.open(image_path).convert('RGB')
    if image.width != 1280 and image.height != 720:
        image = image.resize(expected_size)
    crop_list = [
        image.crop((begin[0], begin[1], begin[0] + box[0], begin[1] + box[1])),
        image.crop((begin[0] + offset_x, begin[1], begin[0] + offset_x + box[0], begin[1] + box[1])),
        image.crop((begin[0], begin[1] + offset_y, begin[0] + box[0], begin[1] + offset_y + box[1])),
        image.crop((begin[0] + offset_x, begin[1] + offset_y, begin[0] + offset_x + box[0], begin[1] + offset_y + box[1])),
        image.crop((begin[0], begin[1] + offset_y * 2, begin[0] + box[0], begin[1] + offset_y * 2 + box[1])),
        image.crop((begin[0] + offset_x, begin[1] + offset_y * 2, begin[0] + offset_x + box[0], begin[1] + offset_y * 2 + box[1])),
    ]
    for i, img in enumerate(crop_list):
        img.save('{}-{}.png'.format(i, random.randint(1, 10000000000)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A labeling tool helps resize and crop a team preview screenshot '
                                                 'into 6 sprite thumbnails.')
    parser.add_argument('filenames', metavar='FILE', type=str, nargs='+', help='images to be processed')
    args = parser.parse_args()
    for p in args.filenames:
        labeling(p)
