import argparse
from utils.data import (
    read_yaml,
    download_public_google_drive_file,
    decompress_file,
    define_train_transformation,
    define_val_transformation,
    define_dataloader
)
from utils.losses import define_CrossEntropy_loss
from utils.optimizers import define_adam_optimizer
from utils.utils import check_directory_contents
from data.DICOMDataset import MulticlassDICOMDataset
from models.DICOMMulticlassClassification import MulticlassClassificationCNN
from models.train_model import train_model
from utils.schemas import MulticlassDataModel
from pathlib import Path
import sys
import os
import torch
from typing import Union


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] 
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

def train(opt: argparse.Namespace) -> None:
    """
    Trains a multiclass classification model using the provided configuration options.

    Args:
        opt (argparse.Namespace): Parsed command-line arguments containing:
            - epochs (int): Number of training epochs.
            - batch_size (int): Batch size for training.
            - weights (str): Path to the initial model weights (if any).
            - data (str): Path to the dataset configuration file (YAML).
            - save_dir_data (str): Directory path for saving or accessing extracted data.
            - imgsz (int): Image size for the input images.
            - patience (int): EarlyStopping patience (epochs without improvement).
    """
    
    epochs, batch_size, weights, data, save_dir_data, image_size, patience = (
        opt.epochs,
        opt.batch_size,
        opt.weights,
        opt.data,
        opt.save_dir_data,
        opt.imgsz,
        opt.patience
    )
    
    data_cfg = read_yaml(data)
    data_model = MulticlassDataModel(**data_cfg)

    dataset_extracted = os.path.join(save_dir_data,
                                     data_model.folder)

    directory_exists = check_directory_contents(dataset_extracted)
    print(f"directory_exists: {directory_exists}")
    
    if not directory_exists:

        destination_file = os.path.join(save_dir_data, 'tmp_dataset.' + data_model.extension)

        download_public_google_drive_file(data_model.id_file,
                                          destination_file)
        
        decompress_file(destination_file,
                        save_dir_data)
    
    
    transformations_training = define_train_transformation(image_size)
    transformations_validation = define_val_transformation(image_size)

    training_dataset = MulticlassDICOMDataset(os.path.join(dataset_extracted, 'train'),
                                          data_model.classes,
                                          transform=transformations_training)
    validation_dataset = MulticlassDICOMDataset(os.path.join(dataset_extracted, 'val'),
                                            data_model.classes,
                                            transform=transformations_validation)
    
    training_dataloader = define_dataloader(training_dataset, batch_size)
    validation_dataloader = define_dataloader(validation_dataset, batch_size, shuffle=False)

    model = MulticlassClassificationCNN(num_classes=len(data_model.classes))
    if weights:
        model.load_state_dict(torch.load(weights))
    
    criterion = define_CrossEntropy_loss()
    params = {
        'lr':1e-4
    }
    optimizer = define_adam_optimizer(model.parameters(), **params)

    train_model(epochs,
                model,
                training_dataloader,
                validation_dataloader,
                criterion,
                optimizer,
                patience=patience)

def main(parameters: argparse.Namespace) -> None:
    """
    Main function to start the training process using the given parameters.

    Args:
        parameters (argparse.Namespace): Command-line arguments for the training process.
    """
    train(parameters)


def parse_opt(known: bool = False) -> argparse.Namespace:
    """
    Parses command-line arguments for model training.

    Args:
        known (bool, optional): Whether to return known arguments only (used for testing). Defaults to False.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=None, help="initial weights path")
    parser.add_argument("--data", type=str, default=None, help="dataset.yaml path")
    parser.add_argument("--epochs", type=int, default=100, help="total training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="total batch size for all GPUs, -1 for autobatch")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=224, help="train, val image size (pixels)")
    parser.add_argument("--patience", type=int, default=10, help="EarlyStopping patience (epochs without improvement)")
    parser.add_argument("--device", default="cuda", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--save-dir-data", default=ROOT / "data/raw", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")

    return parser.parse_known_args()[0] if known else parser.parse_args()


def run(**kwargs: Union[str, int]) -> argparse.Namespace:
    """
    Executes the training process with the given options, overriding any command-line arguments with the provided keyword arguments.

    Args:
        **kwargs (Union[str, int]): Additional options to override default command-line arguments.

    Example:
        import train; train.run(data='dataset.yaml', imgsz=320, weights='model_weights.pth')

    Returns:
        argparse.Namespace: The final set of options used for training.
    """
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

