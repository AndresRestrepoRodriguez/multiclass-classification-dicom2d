import torch
import base64
import numpy as np
import onnxruntime as ort
import io
from src.utils.data import (
    read_dicom_image,
    define_val_transformation,
    normalize_image,
    normalize_ct_int16
)


def predict_model(image: str, model, image_base64=False, img_size=224, device='cpu'):

    if image_base64:
        dicom_data = base64.b64decode(image)
        dicom_file = io.BytesIO(dicom_data)
    else:
        dicom_file = image
    dicom_image_array = read_dicom_image(dicom_file)
    transformation = define_val_transformation(img_size)
    dicom_image_transformed = transformation(dicom_image_array)
    dicom_image_transformed = dicom_image_transformed.to(device)

    #model.eval()
    with torch.no_grad():
        output = model(dicom_image_transformed.unsqueeze(0)).squeeze()
        #predicted = torch.sigmoid(output).round()
    return output.item()


def predict_model_citadel(image_data, model, img_size=224, device='cpu'):
    #imgarr = np.asarray(image_data)
    #print(f"type in api: {imgarr.dtype}")
    np_array = np.asarray(image_data)
    print(np_array)
    img_array = normalize_ct_int16(np_array)
    #img_array = np.asarray(image_data)
    transformation = define_val_transformation(img_size)
    dicom_image_transformed = transformation(img_array)
    dicom_image_transformed = dicom_image_transformed.to(device)
    with torch.no_grad():
        output = model(dicom_image_transformed.unsqueeze(0)).squeeze()
    outputs = output.cpu().numpy()
    return outputs.tolist()

