import torch
from models.DICOMMulticlassClassification import MulticlassClassificationCNN
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    precision_recall_curve,
    auc,
    fbeta_score,
    classification_report

)
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt


def validate_model(data_loader, model_path, classes, num_classes, model_type='pytorch'):
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
            all_predictions.extend(predicted.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='macro')
    recall = recall_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro')
    cm = confusion_matrix(all_labels, all_predictions)

    # Display Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot().figure_.savefig('confusion_matrix.png')

    # Classification Report
    print(f"Classification Report:\n{classification_report(all_labels, all_predictions, target_names=classes)}")

    print(f"Confusion Matrix: {cm}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1 Score (macro): {f1:.4f}")

    

    