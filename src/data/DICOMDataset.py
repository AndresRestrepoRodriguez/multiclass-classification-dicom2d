from torch.utils.data import Dataset
import os
import pydicom
import numpy as np
from utils.data import normalize_image


class MulticlassDICOMDataset(Dataset):
    """
        Args:
            directory (str): Path to the main directory containing subdirectories for each class.
            classes (list): List of class names corresponding to subdirectory names.
            transform (callable, optional): Optional transform to be applied on a sample.
    """
    
    def __init__(self, directory, classes, transform=None):
        self.directory = directory
        self.transform = transform
        self.filenames = []
        self.labels = []

        # Load all images and labels from the directory structure
        for label, subdir in enumerate(classes):
            subdir_path = os.path.join(directory, subdir)
            for filename in os.listdir(subdir_path):
                if filename.endswith('.dcm') or filename.endswith('.DCM'):
                    self.filenames.append(os.path.join(subdir_path, filename))
                    self.labels.append(label)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        ds = pydicom.dcmread(self.filenames[idx])
        image = ds.pixel_array
        image = normalize_image(image)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    



