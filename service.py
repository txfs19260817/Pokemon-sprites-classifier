import argparse
import io
import logging

import numpy as np
import torch
from PIL import Image
from flask import Flask, jsonify, request

from utils.labeling import label_csv2dict
from utils.model import supported_models, training_model
from utils.transformation import transform

# args
parser = argparse.ArgumentParser(description="Pokemon dot sprites classifier inference service")
parser.add_argument('-a', '--arch', metavar='ARCH', default='mobilenetv2',
                    choices=supported_models,
                    help='model architecture: ' +
                         ' | '.join(supported_models) +
                         ' (default: mobilenetv2)')
parser.add_argument('-l', '--label', metavar='FILE', type=str, default='label.csv',
                    help='path to label.csv')
args = parser.parse_args()

# app settings
app = Flask(__name__)
logger = logging.getLogger(__name__)
inference_types = ('single', 'team')

# model settings
class_index = label_csv2dict(args.label)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weight_path = args.arch + '.pth'
model = training_model(args.arch, len(class_index), pretrained=False)
model = model.to(device)
model.load_state_dict(torch.load(weight_path, map_location=device))
model.eval()

# crop parameters
expected_size = (1280, 720)
begin = (85, 20)
box = (75, 75)
offset_x, offset_y = 588, 186


def resize_and_crop(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
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
    return crop_list


def get_team_preview_prediction(image_bytes):
    crop_list = resize_and_crop(image_bytes)
    tensor_list = [transform['test'](i) for i in crop_list]
    tensor = torch.stack(tensor_list)
    tensor = tensor.to(device)
    with torch.no_grad():
        outputs = model(tensor)
    _, predicted_indices = torch.max(outputs.data, 1)
    predicted_classes = [class_index[i.item()] for i in predicted_indices]
    logger.info('Result: ' + ' '.join(predicted_classes))
    return predicted_classes


def get_single_prediction(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    tensor = transform['test'](image).unsqueeze(0)
    tensor = tensor.to(device)
    with torch.no_grad():
        output = model(tensor)
    _, predicted_idx = torch.max(output.data, 1)
    predicted_class = class_index[predicted_idx.item()]
    logger.info('Result: ' + predicted_class)
    return predicted_class


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file, inference_type = request.files['file'], request.files['type']
        if inference_type not in inference_types:
            inference_type = 'team'
        img_bytes = file.read()
        if inference_type == 'single':
            class_name = get_single_prediction(image_bytes=img_bytes)
            return jsonify({'name': class_name})
        if inference_type == 'team':
            classes_name = get_team_preview_prediction(image_bytes=img_bytes)
            return jsonify({'names': classes_name})


if __name__ == '__main__':
    app.run()
