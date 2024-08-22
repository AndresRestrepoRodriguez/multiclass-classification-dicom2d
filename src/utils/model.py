from src.models.DICOMMulticlassClassification import MulticlassClassificationCNN
import torch


def read_pytorch_model_eval(weights: str, num_classes:int, device='cpu'):
    model = MulticlassClassificationCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(weights,  map_location=torch.device(device)))
    model.eval()

    return model


def read_torchcript_model_eval(weights: str, device='cpu'):
    model = torch.jit.load(weights)
    model.to(device)
    model.eval()

    return model