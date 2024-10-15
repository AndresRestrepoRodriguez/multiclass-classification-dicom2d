# DICOM 2D Image Multiclass Classification with PyTorch

This repository contains a pipeline for multiclass classification of DICOM 2D medical images using PyTorch. The code is designed to handle the preprocessing, training, validation, and model exporting for image classification tasks where the input images are in DICOM format.

## Table of Contents
- [Introduction](#introduction)
- [DICOM 2D Images](#dicom-2d-images)
- [Multiclass Image Classification](#binary-image-classification)
- [How to Use](#how-to-use)
  - [Dataset Preparation](#dataset-preparation)
  - [Configuration File](#configuration-file)
  - [Training, Validation, and Exporting the Model](#training-validation-and-exporting-the-model)
- [Jupyter Notebook Usage](#jupyter-notebook-usage)

## Introduction

In this repository, we provide a streamlined approach to classify medical images stored in the DICOM format into two categories. PyTorch is used to build a neural network model for binary classification, where you can train, validate, and export your trained model efficiently.

## DICOM 2D Images

DICOM (Digital Imaging and Communications in Medicine) is a standard format for storing medical imaging data. In this project, we work with DICOM files, which store 2D medical images such as X-rays, MRIs, or CT scans. DICOM images are widely used in medical environments because they contain both the image data and metadata (e.g., patient information, imaging parameters).

To handle DICOM images, we convert them into 2D image tensors that can be fed into the PyTorch models for binary classification tasks.

## Multiclass Image Classification

Multiclass image classification extends beyond binary classification by categorizing images into three or more distinct classes. In the context of medical image analysis, this could involve classifying conditions such as "normal," "benign," and "malignant," or identifying different types of abnormalities, diseases, or anatomical structures. The model learns to distinguish between multiple categories by extracting relevant features from the input images, such as DICOM images, and assigning them to one of the predefined classes. This approach is crucial in medical diagnostics, where multiple outcomes or diagnoses are possible based on the visual information in the image.

## How to Use

### Dataset Preparation

Before starting the training process, you need to prepare and organize your dataset. The images should be arranged in the following structure:

```plaintext
/dataset
│
├── /train
│   ├── /class_0
│   │   └── image1.dcm, image2.dcm, ...
│   ├── /class_1
│   │   └── image1.dcm, image2.dcm, ...
│   ├── /class_2
│       └── image1.dcm, image2.dcm, ...
│
└── /val
    ├── /class_0
    │   └── image1.dcm, image2.dcm, ...
    ├── /class_1
    │   └── image1.dcm, image2.dcm, ...
    ├── /class_2
        └── image1.dcm, image2.dcm, ...
```

- `train/`: This folder contains the training images organized into subfolders where each subfolder corresponds each class.
- `val/`: This folder contains the validation images organized similarly into subfolders.

### Upload and Share Dataset on Google Drive

Once the dataset is organized, compress the dataset folder into a `.rar` file and upload it to Google Drive. After uploading, set the sharing option to **"Anyone with the link"**. This will make the dataset accessible for use in training.

### Configuration File

After uploading the dataset to Google Drive, you need to create a configuration file (e.g., `dataset_config.ymal`) that will store important information such as the download link, dataset paths, and class names.

Example of a configuration file (`dataset_config.yaml`):

```yaml
source: google-drive
id_file: 1GioXTPCM9oMJmBuqX-3emdu6MPn
task: multiclass-classification
classes: ['bodyfront', 'bodyside', 'otherview']
extension: rar
folder: dataset-multiclass-dicom_v2-int16
```

Make sure to replace the information with the actual details from the dataset. The configuration file should be located into the data/raw folder. There is located the previous example.

### Training, Validation, and Exporting the Model

After setting up the dataset and configuration file, you can call the Python script to start the training, validation, and model exporting process. Below is an example command:

Training

```bash
python train.py --data dataset_config.yaml 
```

- `--epochs`: (int) Number of training epochs. Default 100.
- `--batch-size`: (int) Batch size for training. Default 16.
- `--weights`: (str) Path to the initial model weights (if any). 
- `--data`: (str) Path to the dataset configuration file (YAML).
- `--save_dir_data`: (str) Directory path for saving or accessing extracted data. Default data/raw/.
- `--imgsz`: (int) Image size for the input images. Default 224.
- `--patience`: (int) EarlyStopping patience (epochs without improvement). Default 10.
- `--device`: (str) Default cuda.

You can modify the script parameters to fit your specific use case.

Validation

```bash
python val.py --data dataset_config.yaml --weights /path/model.pt
```

- `--batch-size`: (int) Batch size for validation. Default 16.
- `--weights`: (str) Path to the model weights or model path. 
- `--data`: (str) Path to the dataset configuration file (YAML).
- `--save_dir_data`: (str) Directory path for saving or accessing extracted data. Default data/raw/.
- `--imgsz`: (int) Image size for the input images. Default 224.
- `--model-type`: (str) Type of the model to validate. Default pytorch. Possible values: pytorch, torchscript or onnx.
- `--device`: (str) Default cuda.

You can modify the script parameters to fit your specific use case.

Exporting

```bash
python exporting.py --data dataset_config.yaml --weights /path/model.pt
```

- `--weights`: (str) Path to the model weights or model path. 
- `--num-classes`: (int) Number of classes of the model.
- `--format`: (str) Final format. Default torchscript. Options: torchscript and onnx.
- `--imgsz`: (int) Image size for the input images. Default 224.

You can modify the script parameters to fit your specific use case.

## Jupyter Notebook Usage

In addition to the Python script, a Jupyter notebook is provided in this repository for ease of use, experimentation, and quick model iterations.

To use the notebook:

1. Open the provided notebook located in Notebooks folder (Sandbox_Torch_DICOM_2D_Multiclass.ipynb) in your Jupyter environment or Colab.
2. Update the paths for your dataset and configuration file.
3. Run through the cells to:
   - Setting up.
   - Train the model.
   - Validate the model.
   - Export the trained model to a file format such as `.torchscript` or `.onnx`.

The notebook is structured to walk you through each step of the process interactively.

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

