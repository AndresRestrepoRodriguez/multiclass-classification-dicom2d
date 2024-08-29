import torch
from models.DICOMMulticlassClassification import MulticlassClassificationCNN
from sklearn.metrics import ConfusionMatrixDisplay

import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
from utils import metrics as metrics_citadel


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

            #sf = torch.nn.Softmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(outputs.cpu().numpy())
            #all_predictions.extend(sf(outputs).cpu().numpy())
    

    np_all_labels = np.array(all_labels)
    np_all_predictions = np.array(all_predictions)


    confusion_matrix = metrics_citadel.confusionmatrix(np_all_labels, np_all_predictions)
    specificity = metrics_citadel.specificity(np_all_labels, np_all_predictions)
    accuracy = metrics_citadel.accuracy(np_all_labels, np_all_predictions)
    f1_score = metrics_citadel.f1score(np_all_labels, np_all_predictions)
    f05_score = metrics_citadel.f05score(np_all_labels, np_all_predictions)
    f2score = metrics_citadel.f2score(np_all_labels, np_all_predictions)
    precision = metrics_citadel.precision(np_all_labels, np_all_predictions)
    recall = metrics_citadel.recall(np_all_labels, np_all_predictions)

    #Per class metrics

    per_class_specificity = metrics_citadel.per_class_specificity(np_all_labels, np_all_predictions)
    per_class_accuracy = metrics_citadel.per_class_accuracy(np_all_labels, np_all_predictions)
    per_class_precision = metrics_citadel.per_class_precision(np_all_labels, np_all_predictions)
    per_class_recall = metrics_citadel.per_class_recall(np_all_labels, np_all_predictions)
    per_class_f1score = metrics_citadel.per_class_f1score(np_all_labels, np_all_predictions)
    per_class_f2score = metrics_citadel.per_class_f2score(np_all_labels, np_all_predictions)
    per_class_f05score = metrics_citadel.per_class_f05score(np_all_labels, np_all_predictions)


    #cm = confusion_matrix(all_labels, all_predictions)

    # Display Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=classes)
    disp.plot().figure_.savefig('confusion_matrix.png')

    print(f"Confusion Matrix: {confusion_matrix}")
    print(f"Validation Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_score}")
    print(f"specificity: {specificity}")
    print(f"f05_score: {f05_score}")
    print(f"f2score: {f2score}")

    print("Per class -------------------------------")
    print(f"per_class_specificity: {per_class_specificity}")
    print(f"per_class_accuracy: {per_class_accuracy}")
    print(f"per_class_precision: {per_class_precision}")
    print(f"per_class_recall: {per_class_recall}")
    print(f"per_class_f1score: {per_class_f1score}")
    print(f"per_class_f2score: {per_class_f2score}")
    print(f"per_class_f05score: {per_class_f05score}")




    

    