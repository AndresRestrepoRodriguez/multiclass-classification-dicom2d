from utils.model import read_pytorch_model_eval
from models.predict_model import predict_model
import argparse
from typing import Union, Any


def predict(opt: argparse.Namespace) -> None:
    """
    Loads a trained model and predicts the output for a given image.

    Args:
        opt (argparse.Namespace): Parsed command-line arguments containing:
            - weights (str): Path to the model weights file.
            - image_path (str): Path to the input image for prediction.
            - imgsz (int): Image size for resizing the input image.
            - device (str): Device to run the model on (e.g., 'cuda' or 'cpu').
            - num_classes (int): Number of output classes for the model.

    Returns:
        None
    """
    weights, image_path, image_size, device, num_clasess = (
        opt.weights,
        opt.image_path,
        opt.imgsz,
        opt.device,
        opt.num_classes
    )

    model = read_pytorch_model_eval(weights=weights,
                                    num_classes=num_clasess)

    prediction = predict_model(image=image_path,
                               model=model,
                               img_size=image_size)
    
    print(f"prediction: {prediction}")


def main(parameters: argparse.Namespace) -> None:
    """
    Main function to initiate the model prediction with the provided parameters.

    Args:
        parameters (argparse.Namespace): Command-line arguments for the prediction process.
    """
    predict(parameters)


def parse_opt(known: bool = False) -> argparse.Namespace:
    """
    Parses command-line arguments for model prediction.

    Args:
        known (bool, optional): Whether to return known arguments only (used for testing). Defaults to False.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=None, help="initial weights path")
    parser.add_argument("--image-path", type=str, default=None, help="image path")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=224, help="train, val image size (pixels)")
    parser.add_argument("--device", default="cuda", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--num-classes", default="3", help="num of classes", type=int)

    return parser.parse_known_args()[0] if known else parser.parse_args()


def run(**kwargs: Union[str, int]) -> argparse.Namespace:
    """
    Executes model prediction with given options, overriding any command-line arguments with the provided keyword arguments.

    Args:
        **kwargs (Union[str, int]): Additional options to override default command-line arguments.

    Example:
        import train; train.run(image_path='path_to_image.dcm', weights='model_weights.pth')

    Returns:
        argparse.Namespace: The final set of options used for prediction.
    """
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)