import argparse

import numpy as np
import onnx
import onnxruntime
import torch
import torch.onnx
from PIL import Image

from utils.model import training_model, supported_models
from utils.transformation import transform
from utils.labeling import label_csv2dict


def torch2onnx(model_name, image_path, label_path):
    with open(label_path, 'r') as f:
        num_of_classes = len(f.readlines()) - 1

    device = torch.device('cpu')
    weight_path = model_name + '.pth'
    onnx_path = model_name + '.onnx'

    # Load pretrained model weights and set the model to inference mode
    model = training_model(model_name, num_of_classes, pretrained=False)
    model = model.to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    # Input to the model
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    torch_out = model(x)

    # Export the model
    torch.onnx.export(model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      onnx_path,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}})

    # check the ONNX model with ONNX’s API
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # verify the model’s output with ONNX Runtime
    ort_session = onnxruntime.InferenceSession(onnx_path)

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

    img = Image.open(image_path)
    img_y = transform['test'](img).unsqueeze(0)

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_y)}
    ort_outs = ort_session.run(None, ort_inputs)
    output = ort_outs[0]
    prediction = np.argmax(output, 1)[0]
    print("Prediction: ", prediction, label_csv2dict(label_path)[prediction])


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export a PyTorch model to an ONNX model.')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='mobilenetv2',
                        choices=supported_models,
                        help='model architecture: ' +
                             ' | '.join(supported_models) +
                             ' (default: mobilenetv2)')
    parser.add_argument('-i', '--image-path', metavar='DIR',
                        help='path to an image for test', required=True)
    parser.add_argument('-l', '--label-path', metavar='FILE', type=str, default='./dataset/label.csv',
                        help='path to label.csv (default: ./dataset/label.csv)')
    args = parser.parse_args()
    torch2onnx(args.arch, args.image_path, args.label_path)
