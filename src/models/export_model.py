import torch
import torch.nn as nn
import torch.nn.functional as F
import json


# Define the function that exports the model to both formats
def export_model_pytorch(model, torchscript_file_path):
    """
    Exports a PyTorch model to both TorchScript and ONNX formats.

    Args:
        model (torch.nn.Module): The PyTorch model to export.
        input_shape (tuple): The shape of the dummy input tensor for ONNX export.
        torchscript_file_path (str): File path where the TorchScript model will be saved.
        onnx_file_path (str): File path where the ONNX model will be saved.
    """
    # Ensure the model is in evaluation mode
    model.eval()

    # Export to TorchScript
    scripted_model = torch.jit.script(model)
    scripted_model.save(torchscript_file_path)
    print(f"TorchScript model saved to {torchscript_file_path}")


# Define the function that exports the model to both formats
def export_model_pytorch_trace(model, im, torchscript_file_path):
    """
    Exports a PyTorch model to both TorchScript and ONNX formats.

    Args:
        model (torch.nn.Module): The PyTorch model to export.
        input_shape (tuple): The shape of the dummy input tensor for ONNX export.
        torchscript_file_path (str): File path where the TorchScript model will be saved.
        onnx_file_path (str): File path where the ONNX model will be saved.
    """
    # Ensure the model is in evaluation mode
    model.eval()

    # Export to TorchScript
    #scripted_model = torch.jit.script(model)
    ts = torch.jit.trace(model, im, strict=False)
    d = {"shape": im.shape}
    extra_files = {"config.txt": json.dumps(d)}  # torch._C.ExtraFilesMap()
    #scripted_model.save(torchscript_file_path)
    ts.save(str(torchscript_file_path), _extra_files=extra_files)
    print(f"TorchScript model saved to {torchscript_file_path}")


# Define the function that exports the model to both formats
def export_model_onnx(model, im, onnx_file_path):
    """
    Exports a PyTorch model to both TorchScript and ONNX formats.

    Args:
        model (torch.nn.Module): The PyTorch model to export.
        input_shape (tuple): The shape of the dummy input tensor for ONNX export.
        torchscript_file_path (str): File path where the TorchScript model will be saved.
        onnx_file_path (str): File path where the ONNX model will be saved.
    """
    # Ensure the model is in evaluation mode
    model.eval()

    # Export the model to ONNX
    torch.onnx.export(model, im, onnx_file_path, export_params=True, opset_version=12,
                      do_constant_folding=True, input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    print(f"ONNX model saved to {onnx_file_path}")
