import argparse
from models.DICOMMulticlassClassification import MulticlassClassificationCNN
import torch
from models.export_model import (
    export_model_pytorch_trace,
    export_model_onnx
)
import os
from typing import Union, Any

FORMATS = ['torchscript', 'onnx']




def export(opt: argparse.Namespace) -> None:
    """
    Exports a trained model in either TorchScript or ONNX format.

    Args:
        opt (argparse.Namespace): Parsed command-line arguments containing:
            - weights (str): Path to the model weights file.
            - imgsz (int): Image size for the dummy input tensor.
            - format (str): Format to export the model ('torchscript' or 'onnx').
            - num_classes (int): Number of output classes for the model.

    Raises:
        Exception: If there is an issue loading the model or if an invalid format is provided.
    """

    weights, image_size, format, num_classes = (
        opt.weights,
        opt.imgsz,
        opt.format.strip(),
        opt.num_classes
    )

    assert format in FORMATS, f"Invalid format : {format}. Valid formats: {FORMATS}"

    if not weights:
        raise Exception("Weights path is required")

    try:
        # Load the model
        model = MulticlassClassificationCNN(num_classes=num_classes)
        model.load_state_dict(torch.load(weights))
    except Exception as e:
        raise Exception(f"Problem loading the Model: {e}")
    
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
    """
    Parses command-line arguments for model exporting.

    Args:
        known (bool, optional): Whether to return known arguments only (used for testing). Defaults to False.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=None, help="initial weights path")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=224, help="train, val image size (pixels)")
    parser.add_argument("--format", "--fmt", "--format-model", type=str, default='torchscript', help="Format of exporting")
    parser.add_argument("--num-classes", default="3", help="num of classes", type=int)
    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(parameters: argparse.Namespace) -> None:
    """
    Main function to initiate model export with the given parameters.

    Args:
        parameters (argparse.Namespace): Command-line arguments for exporting the model.
    """
    export(parameters)


def run(**kwargs: Union[str, int]) -> argparse.Namespace:
    """
    Executes the model export with given options, overriding any command-line arguments with the provided keyword arguments.

    Args:
        **kwargs (Union[str, int]): Additional options to override default command-line arguments.

    Returns:
        argparse.Namespace: The final set of options used for model export.
    """
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
