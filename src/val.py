import argparse
from utils.data import (
    read_yaml,
    download_public_google_drive_file,
    decompress_file,
    define_val_transformation,
    define_dataloader
)
from utils.utils import check_directory_contents
from data.DICOMDataset import MulticlassDICOMDataset
from models.validation_model import validate_model
from utils.schemas import MulticlassDataModel
from pathlib import Path
import sys
import os
from typing import Union


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] 
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


def val(opt: argparse.Namespace) -> None:
    """
    Validates a multiclass classification model using the provided configuration options.

    Args:
        opt (argparse.Namespace): Parsed command-line arguments containing:
            - batch_size (int): Batch size for validation.
            - weights (str): Path to the model weights file.
            - data (str): Path to the dataset configuration file (YAML).
            - save_dir_data (str): Directory path for saving or accessing extracted data.
            - imgsz (int): Image size for the input images.
            - model_type (str): Type of the model (e.g., 'pytorch', 'onnx').
    
    Returns:
        None
    """

    batch_size, weights, data, save_dir_data, image_size, model_type = (
        opt.batch_size,
        opt.weights,
        opt.data,
        opt.save_dir_data,
        opt.imgsz,
        opt.model_type
    )

    data_cfg = read_yaml(data)
    data_model = MulticlassDataModel(**data_cfg)

    dataset_extracted = os.path.join(save_dir_data,
                                     data_model.folder)

    directory_exists = check_directory_contents(dataset_extracted)
    
    if not directory_exists:

        destination_file = os.path.join(save_dir_data, 'tmp_dataset.' + data_model.extension)

        download_public_google_drive_file(data_model.id_file,
                                          destination_file)
        
        decompress_file(destination_file,
                        save_dir_data)
    
    
    transformations_validation = define_val_transformation(image_size)

    validation_dataset = MulticlassDICOMDataset(os.path.join(dataset_extracted, 'val'),
                                            data_model.classes,
                                            transform=transformations_validation)
    
    validation_dataloader = define_dataloader(validation_dataset, batch_size, shuffle=False)

   
    validate_model(
        validation_dataloader,
        weights,
        classes = data_model.classes,
        num_classes = len(data_model.classes),
        model_type = model_type
        )


def parse_opt(known: bool = False) -> argparse.Namespace:
    """
    Parses command-line arguments for model validation.

    Args:
        known (bool, optional): Whether to return known arguments only (used for testing). Defaults to False.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=None, help="initial weights path")
    parser.add_argument("--data", type=str, default=None, help="dataset.yaml path")
    parser.add_argument("--batch-size", type=int, default=16, help="total batch size for all GPUs, -1 for autobatch")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=224, help="train, val image size (pixels)")
    parser.add_argument("--device", default="cuda", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--save-dir-data", default=ROOT / "data/raw", help=" TODO ")
    parser.add_argument("--model-type", default="pytorch", help=" TODO ")


    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(parameters: argparse.Namespace) -> None:
    """
    Main function to initiate the validation process using the provided parameters.

    Args:
        parameters (argparse.Namespace): Command-line arguments for the validation process.
    """
    val(parameters)


def run(**kwargs: Union[str, int]) -> argparse.Namespace:
    """
    Executes the validation process with the given options, overriding any command-line arguments with the provided keyword arguments.

    Args:
        **kwargs (Union[str, int]): Additional options to override default command-line arguments.

    Example:
        import val; val.run(data='dataset.yaml', imgsz=320, weights='model_weights.pth')

    Returns:
        argparse.Namespace: The final set of options used for validation.
    """
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)