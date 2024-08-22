import argparse
from src.models.DICOMMulticlassClassification import MulticlassClassificationCNN
import torch
from models.export_model import (
    export_model_pytorch_trace,
    export_model_onnx
)
import os

FORMATS = ['torchscript', 'onnx']




def export(opt):

    weights, image_size, format, num_classes = (
        opt.weights,
        opt.imgsz,
        opt.format.strip(),
        opt.num_clasess
    )

    assert format in FORMATS, f"Invalid format : {format}. Valid formats: {FORMATS}"

    if weights:
        try:
            model = MulticlassClassificationCNN(num_classes=num_classes)
            model.load_state_dict(torch.load(weights))

        except Exception as e:
            raise Exception(f'Problem loading the Model: {e}')

    else:
        raise Exception(f'Weights path is required')
    
    im = torch.rand(1, 1, image_size, image_size)

    weights_name = os.path.basename(weights)

    if format == "torchscript":
        export_model_pytorch_trace(model=model,
                                   im=im,
                                   torchscript_file_path=os.path.splitext(weights_name)[0] + f'.{format}')
    elif format == "onnx":
        export_model_onnx(model=model,
                          im=im,
                          onnx_file_path=os.path.splitext(weights_name)[0] + f'.{format}')
    else:
        raise Exception(f'Invalid format: {format} - Format not supported right now')

    


def parse_opt(known=False):
    """Parses command-line arguments for YOLOv5 training, validation, and testing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=None, help="initial weights path")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=224, help="train, val image size (pixels)")
    parser.add_argument("--format", "--fmt", "--format-model", type=str, default='torchscript', help="Format of exporting")
    parser.add_argument("--num-classes", default="3", help="num of classes", type=int)
    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(parameters):
    export(parameters)


def run(**kwargs):
    """
    Executes YOLOv5 training with given options, overriding with any kwargs provided.

    Example: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    """
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
