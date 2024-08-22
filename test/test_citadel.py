import requests
import base64
import glob as glob
import os
import pydicom
import numpy as np
from PIL import Image
import torch
import pandas as pd


images_brain = glob.glob(os.path.join('data/val/brain', '*.dcm'))
images_chest = glob.glob(os.path.join('data/val/chest', '*.dcm'))
tmp_image_path = 'data/tmp_png_dicom.png'

results = []

def dicom_to_png(dicom_file_path, png_file_path):
    # Read the DICOM file
    dicom_image = pydicom.dcmread(dicom_file_path, force=True)
    
    # Get the pixel array from the DICOM file
    image_array = dicom_image.pixel_array
    
    # Normalize the pixel array to the range 0-255
    #image_array_normalized = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array)) * 255
    #image_array_normalized = image_array_normalized.astype(np.uint8)
    
    # Create an Image object from the pixel array
    image = Image.fromarray(image_array)
    
    # Save the image as a PNG file
    image.save(png_file_path)
    print(f"Converted {dicom_file_path} to {png_file_path}")


for image in images_brain:
    ground_true_class = 0
    label = 'brain'
    dicom_to_png(image, tmp_image_path)

    with open(tmp_image_path, 'rb') as f:
        encoded_image = base64.b64encode(f.read()).decode('utf-8')

    data = {'image': encoded_image}
    response = requests.post('http://127.0.0.1:5000/predict', json=data)
    prob = response.json().get('predictions')
    class_prob = torch.sigmoid(torch.tensor(1-prob)).round().item()
    
    results.append([image, prob, class_prob, ground_true_class, label])


for image in images_chest:
    ground_true_class = 1
    label = 'chest'
    dicom_to_png(image, tmp_image_path)

    with open(tmp_image_path, 'rb') as f:
        encoded_image = base64.b64encode(f.read()).decode('utf-8')

    data = {'image': encoded_image}
    response = requests.post('http://127.0.0.1:5000/predict', json=data)
    prob = response.json().get('predictions')
    class_prob = torch.sigmoid(torch.tensor(prob)).round().item()
    results.append([image, prob, class_prob, ground_true_class, label])


df = pd.DataFrame(results, columns=['path', 'prob', 'pred_class', 'true_class', 'label'])
df.to_csv('docker_results_hoang_prob_0.csv', index=False) 

    