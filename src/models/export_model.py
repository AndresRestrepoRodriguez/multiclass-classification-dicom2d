import torch
import torch.nn as nn
import torch.nn.functional as F
import json


# Define the function that exports the model to both formats
def export_model_pytorch(model: nn.Module, torchscript_file_path: str) -> None:
    """
    Exports a PyTorch model to TorchScript format.

    Args:
        model (torch.nn.Module): The PyTorch model to export.
        torchscript_file_path (str): File path where the TorchScript model will be saved.

    Returns:
        None
    """
    # Ensure the model is in evaluation mode
    model.eval()

    # Export to TorchScript
    scripted_model = torch.jit.script(model)
    scripted_model.save(torchscript_file_path)
    print(f"TorchScript model saved to {torchscript_file_path}")


# Define the function that exports the model to both formats
def export_model_pytorch_trace(model: nn.Module, im: torch.Tensor, torchscript_file_path: str) -> None:
    """
    Exports a PyTorch model to TorchScript format using tracing.

    Args:
        model (torch.nn.Module): The PyTorch model to export.
        im (torch.Tensor): An example input tensor to trace the model.
        torchscript_file_path (str): File path where the TorchScript model will be saved.

    Returns:
        None
    """
    # Ensure the model is in evaluation mode
    model.eval()

    # Export to TorchScript
    ts = torch.jit.trace(model, im, strict=False)
    d = {"shape": im.shape}
    extra_files = {"config.txt": json.dumps(d)}  # torch._C.ExtraFilesMap()
    ts.save(str(torchscript_file_path), _extra_files=extra_files)
    print(f"TorchScript model saved to {torchscript_file_path}")


# Define the function that exports the model to both formats
def export_model_onnx(model: nn.Module, im: torch.Tensor, onnx_file_path: str) -> None:
    """
    Exports a PyTorch model to ONNX format.

    Args:
        model (torch.nn.Module): The PyTorch model to export.
        im (torch.Tensor): An example input tensor for the ONNX export.
        onnx_file_path (str): File path where the ONNX model will be saved.

    Returns:
        None
    """
    # Ensure the model is in evaluation mode
    model.eval()

    # Export the model to ONNX
    torch.onnx.export(model,
                      im,
                      onnx_file_path,
                      export_params=True,
                      opset_version=12,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    print(f"ONNX model saved to {onnx_file_path}")
