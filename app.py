import argparse
import io
import json
import logging

import torch
from PIL import Image
from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from utils.labeling import label_csv2dict, resize_and_crop
from utils.model import training_model
from utils.transformation import transform


# load config
with open('configs/config.json', 'r') as f:
    args = json.load(f)

# app settings
app = Flask(__name__)
inference_types = ('single', 'team')

# log settings
gunicorn_logger = logging.getLogger('gunicorn.error')
app.logger.handlers = gunicorn_logger.handlers
app.logger.setLevel(gunicorn_logger.level)

# limiter
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["100 per day"]
)

# model settings
class_index = label_csv2dict(args['label'])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weight_path = args['arch'] + '.pth'
model = training_model(args['arch'], len(class_index), pretrained=False)
model = model.to(device)
model.load_state_dict(torch.load(weight_path, map_location=device))
model.eval()


def get_team_preview_prediction(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    crop_list = resize_and_crop(image)
    tensor_list = [transform['test'](i) for i in crop_list]
    tensor = torch.stack(tensor_list)
    tensor = tensor.to(device)
    with torch.no_grad():
        outputs = model(tensor)
    _, predicted_indices = torch.max(outputs.data, 1)
    predicted_classes = [class_index[i.item()] for i in predicted_indices]
    return predicted_classes


def get_single_prediction(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    tensor = transform['test'](image).unsqueeze(0)
    tensor = tensor.to(device)
    with torch.no_grad():
        output = model(tensor)
    _, predicted_idx = torch.max(output.data, 1)
    predicted_class = class_index[predicted_idx.item()]
    return predicted_class


@app.route('/advapi/predict', methods=['GET', 'POST'])
@limiter.limit("2/minute", override_defaults=False)
def predict():
    if request.method == 'GET':
        app.logger.warning('/predict is visited in GET method')
        return 'The model is up and running. Send a POST request'
    if request.method == 'POST':
        file, inference_type = request.files['file'], request.files['type']
        if inference_type not in inference_types:
            inference_type = 'team'
        img_bytes = file.read()
        if inference_type == 'single':
            class_name = get_single_prediction(image_bytes=img_bytes)
            app.logger.info('Result: ' + class_name)
            return jsonify({'name': class_name})
        if inference_type == 'team':
            classes_name = get_team_preview_prediction(image_bytes=img_bytes)
            app.logger.info('Result: ' + ' '.join(classes_name))
            return jsonify({'names': classes_name})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', action='store_true', help='debug mode')
    console_args = parser.parse_args()
    if console_args.debug:
        app.run(debug=console_args.debug)
    else:
        app.run(ssl_context=(args.ssl.certfile, args.ssl.keyfile))
