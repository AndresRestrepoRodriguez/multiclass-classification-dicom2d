import torch
from models.DICOMMulticlassClassification import MulticlassClassificationCNN
from sklearn.metrics import ConfusionMatrixDisplay
from typing import List
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
from utils import metrics as metrics_citadel


def validate_model(
    data_loader: torch.utils.data.DataLoader, 
    model_path: str, 
    classes: List[str], 
    num_classes: int, 
    model_type: str = 'pytorch'
) -> None:
    """
    Validates a multiclass classification model on the given dataset.

    Args:
        data_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        model_path (str): Path to the saved model file (either PyTorch, TorchScript, or ONNX).
        classes (List[str]): List of class names corresponding to the model's output.
        num_classes (int): Number of output classes for the classification task.
        model_type (str, optional): Type of the model to load ('pytorch', 'torchscript', 'onnx'). Defaults to 'pytorch'.

    Raises:
        ValueError: If an invalid model type is specified.
        Exception: If there is an issue loading the model.

    Returns:
        None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        if model_type == 'pytorch':
            model = MulticlassClassificationCNN(num_classes=num_classes)
            model.load_state_dict(torch.load(model_path))
            model.to(device)
            model.eval()

        elif model_type == 'torchscript':
            model = torch.jit.load(model_path)
            model.to(device)
            model.eval()

        elif model_type == 'onnx':
            ort_session = ort.InferenceSession(model_path)

            def model(x):
                # Preparing input to the ONNX model
                inputs = {ort_session.get_inputs()[0].name: x.numpy()}
                # Running inference
                outputs = ort_session.run(None, inputs)
                # Converting output from ONNX to tensor for consistency with PyTorch models
                return torch.from_numpy(outputs[0])
        else:
            raise ValueError("Invalid model type specified.")
    except Exception as e:
        raise Exception(f'Problem loading the Model: {e}')

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in data_loader:

            # Adjust device handling for ONNX as it needs CPU numpy arrays
            if model_type == 'onnx':
                images = images.to('cpu')
                outputs = model(images)
            else:
                images = images.to(device)
                outputs = model(images)
            
            predicted = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(outputs.cpu().numpy())
    

    # Convert lists to numpy arrays
    np_all_labels = np.array(all_labels)
    np_all_predictions = np.array(all_predictions)

    # Generate a report (custom metrics function)
    metrics_citadel.generate_report(np_all_labels, np_all_predictions)

    