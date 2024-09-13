from src.models.DICOMMulticlassClassification import MulticlassClassificationCNN
import torch


def read_pytorch_model_eval(weights: str, num_classes: int, device: str = 'cpu') -> torch.nn.Module:
    """
    Loads and prepares a PyTorch model for evaluation.

    This function initializes a multiclass classification CNN model, loads its weights 
    from the specified file, and sets it to evaluation mode.

    Args:
        weights (str): Path to the saved PyTorch model weights (in `.pth` format).
        num_classes (int): The number of output classes for the classification model.
        device (str, optional): The device to load the model onto (e.g., 'cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        torch.nn.Module: The loaded model ready for evaluation.
    """
    model = MulticlassClassificationCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(weights, map_location=torch.device(device)))
    model.eval()

    return model


def read_torchcript_model_eval(weights: str, device: str = 'cpu') -> torch.jit.ScriptModule:
    """
    Loads and prepares a TorchScript model for evaluation.

    This function loads a pre-trained TorchScript model from a file and sets it to evaluation mode.

    Args:
        weights (str): Path to the saved TorchScript model file (in `.pt` or `.ts` format).
        device (str, optional): The device to load the model onto (e.g., 'cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        torch.jit.ScriptModule: The loaded TorchScript model ready for evaluation.
    """
    model = torch.jit.load(weights)
    model.to(device)
    model.eval()

    return model