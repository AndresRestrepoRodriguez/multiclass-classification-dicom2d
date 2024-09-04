import requests
import base64
import glob as glob
import os
import pydicom
import numpy as np
from PIL import Image
import torch
import pandas as pd
import metrics as metrics_citadel


images_brain = glob.glob(os.path.join('data/val/brain', '*.dcm'))
images_chest = glob.glob(os.path.join('data/val/chest', '*.dcm'))
images_abdomen = glob.glob(os.path.join('data/val/abdomen', '*.dcm'))
tmp_image_path = 'data/tmp_png_dicom.png'


y_true = []
y_pred = []
y_score = []

def dicom_to_png(dicom_file_path, png_file_path):
    # Read the DICOM file
    dicom_image = pydicom.dcmread(dicom_file_path, force=True)
    
    # Get the pixel array from the DICOM file
    image_array = dicom_image.pixel_array


    #print(f"type in testing: {image_array.dtype}")
    
    # Normalize the pixel array to the range 0-255
    #image_array_normalized = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array)) * 255
    #image_array_normalized = image_array_normalized.astype(np.uint8)
    
    # Create an Image object from the pixel array
    image = Image.fromarray(image_array)
    
    # Save the image as a PNG file
    image.save(png_file_path)
    #print(f"Converted {dicom_file_path} to {png_file_path}")


for image in images_brain:
    ground_true_class = 1
    label = 'brain'
    dicom_to_png(image, tmp_image_path)

    with open(tmp_image_path, 'rb') as f:
        encoded_image = base64.b64encode(f.read()).decode('utf-8')

    data = {'image': encoded_image}
    response = requests.post('http://127.0.0.1:5000/predict', json=data)
    prob = response.json().get('predictions')
    prob = np.float32(prob)
    #print(prob)
    #print(type(prob))
    #class_prob = torch.sigmoid(torch.tensor(1-prob)).item()
    y_pred.append(prob)
    y_true.append(ground_true_class)
    y_score.append(prob)


for image in images_chest:
    ground_true_class = 2
    label = 'chest'
    dicom_to_png(image, tmp_image_path)

    with open(tmp_image_path, 'rb') as f:
        encoded_image = base64.b64encode(f.read()).decode('utf-8')

    data = {'image': encoded_image}
    response = requests.post('http://127.0.0.1:5000/predict', json=data)
    prob = response.json().get('predictions')
    prob = np.float32(prob)
    #class_prob = torch.sigmoid(torch.tensor(prob)).item()
    #print(prob)
    #print(type(prob))
    y_pred.append(prob)
    y_true.append(ground_true_class)
    y_score.append(prob)


for image in images_abdomen:
    ground_true_class = 0
    label = 'abdomen'
    dicom_to_png(image, tmp_image_path)

    with open(tmp_image_path, 'rb') as f:
        encoded_image = base64.b64encode(f.read()).decode('utf-8')

    data = {'image': encoded_image}
    response = requests.post('http://127.0.0.1:5000/predict', json=data)
    prob = response.json().get('predictions')
    #prob = np.float32(prob)
    #class_prob = torch.sigmoid(torch.tensor(prob)).item()
    #print(prob)
    #print(type(prob))
    y_pred.append(prob)
    y_true.append(ground_true_class)
    y_score.append(prob)


y_true_np = np.array(y_true)
y_pred_np = np.array(y_pred)
y_score_np = np.array(y_score)

confusion_matrix = metrics_citadel.confusionmatrix(y_true_np, y_pred_np)
specificity = metrics_citadel.specificity(y_true_np, y_pred_np)
accuracy = metrics_citadel.accuracy(y_true_np, y_pred_np)
f1_score = metrics_citadel.f1score(y_true_np, y_pred_np)
f05_score = metrics_citadel.f05score(y_true_np, y_pred_np)
f2score = metrics_citadel.f2score(y_true_np, y_pred_np)
precision = metrics_citadel.precision(y_true_np, y_pred_np)
recall = metrics_citadel.recall(y_true_np, y_pred_np)

#Per class metrics

per_class_specificity = metrics_citadel.per_class_specificity(y_true_np, y_pred_np)
per_class_accuracy = metrics_citadel.per_class_accuracy(y_true_np, y_pred_np)
per_class_precision = metrics_citadel.per_class_precision(y_true_np, y_pred_np)
per_class_recall = metrics_citadel.per_class_recall(y_true_np, y_pred_np)
per_class_f1score = metrics_citadel.per_class_f1score(y_true_np, y_pred_np)
per_class_f2score = metrics_citadel.per_class_f2score(y_true_np, y_pred_np)
per_class_f05score = metrics_citadel.per_class_f05score(y_true_np, y_pred_np)


#cm = confusion_matrix(all_labels, all_predictions)

# Display Confusion Matrix
#disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=classes)
#disp.plot().figure_.savefig('confusion_matrix.png')

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

    