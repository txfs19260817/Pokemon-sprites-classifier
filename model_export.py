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


def get_model(model_name, label_path):
    label_dict = label_csv2dict(label_path)
    num_of_classes = len(label_dict)

    device = torch.device('cpu')
    weight_path = model_name + '.pth'

    # Load pretrained model weights and set the model to inference mode
    model = training_model(model_name, num_of_classes, pretrained=False)
    model = model.to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    return model, label_dict


def torch2torch_script(model, model_name):
    torch_script_path = model_name + '.pt'
    example = torch.rand(1, 3, 224, 224)
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save(torch_script_path)
    print('a Torch Script has been exported to: ' + torch_script_path)


def torch2onnx(model, model_name, image_path, label_dict):
    onnx_path = model_name + '.onnx'

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
    print('an ONNX model has been exported to: ' + onnx_path)

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
    print("Prediction: ", prediction, label_dict[prediction])


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export a PyTorch model to an ONNX model or a Torch Script.')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='shufflenetv2',
                        choices=supported_models,
                        help='model architecture: ' +
                             ' | '.join(supported_models) +
                             ' (default: shufflenetv2)')
    parser.add_argument('-i', '--image-path', metavar='DIR', type=str,
                        help='path to an image for test (required by ONNX export)')
    parser.add_argument('-l', '--label-path', metavar='FILE', type=str, default='./dataset/label.csv',
                        help='path to label.csv (default: ./dataset/label.csv)')
    parser.add_argument('-t', '--type', type=str, default='torchscript',
                        choices=['onnx', 'torchscript'], help='choose which format to convert (default: torchscript)')
    args = parser.parse_args()
    
    model, label_dict = get_model(args.arch, args.label_path)
    if args.type == 'onnx':
        assert isinstance(args.image_path, str) and len(args.image_path) > 0, "an image is required for ONNX model exporting"
        torch2onnx(model, args.arch, args.image_path, label_dict)
    elif args.type == 'torchscript':
        torch2torch_script(model, args.arch)
