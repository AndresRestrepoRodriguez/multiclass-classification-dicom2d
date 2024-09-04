import os
import zipfile
import tarfile
import rarfile
from pathlib import Path
import torchvision.transforms as transforms
import yaml
import requests
from torch.utils.data import Dataset, DataLoader

import pydicom
import numpy as np
from bs4 import BeautifulSoup



def download_public_google_drive_file(file_id, destination):
    """
    Download a publicly shared file from Google Drive using its file ID and save it to a local file.

    Args:
    file_id (str): Google Drive shared file ID.
    destination (str): The local path to save the downloaded file.

    Returns:
    None
    """
    # Construct the URL to download the file
    download_url = f"https://drive.google.com/uc?id={file_id}&export=download"

    # Make the request and download the file
    with requests.get(download_url, stream=True) as response:
        response.raise_for_status()
        with open(destination, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
    print(f"File has been downloaded successfully and saved to {destination}")


def download_file_from_google_drive(file_id, destination):
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)
    print(f"token: {token}")
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)

def get_confirm_token(response):
    # We use BeautifulSoup to parse the HTML content
    soup = BeautifulSoup(response.content, "html.parser")
    # Find the input tag that has the name attribute 'confirm' and extract its value
    tag = soup.find('input', {'name': 'confirm'})
    if tag:
        return tag['value']
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def decompress_file(filepath, extract_to):
    """
    Decompress a file based on its extension.

    Args:
    filepath (str): The path to the compressed file.
    extract_to (str): The directory to decompress the file into.

    Returns:
    None
    """
    Path(extract_to).mkdir(parents=True, exist_ok=True)

    if filepath.endswith('.zip'):
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            print("Extracted all files from the zip archive.")
    elif filepath.endswith('.tar.gz') or filepath.endswith('.tgz'):
        with tarfile.open(filepath, 'r:gz') as tar_ref:
            tar_ref.extractall(path=extract_to)
            print("Extracted all files from the tar.gz archive.")
    elif filepath.endswith('.rar'):
        with rarfile.RarFile(filepath, 'r') as rar_ref:
            rar_ref.extractall(extract_to)
            print("Extracted all files from the rar archive.")
    else:
        print(f"No extraction performed for '{filepath}'.")

    # Optionally, remove the compressed file after extraction
    os.remove(filepath)
    print(f"Removed compressed file '{filepath}'.")


def define_train_transformation(img_size: int):

    training_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
        transforms.ColorJitter(hue=.05, saturation=.05),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
    ])

    return training_transform


def define_val_transformation(img_size: int):

    validation_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
    ])

    return validation_transform


def define_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)



def read_yaml(filepath: str):
    """
    Reads a YAML file, parses it, and populates a Pydantic model.

    Args:
        filepath (str): Path to the YAML file.
        model (BaseModel): Pydantic model class to be populated.

    Returns:
        BaseModel: An instance of the provided Pydantic model filled with YAML data.
    """
    try:
        with open(filepath, 'r') as file:
            data = yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: The file {filepath} does not exist.")
        return None
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    return data


def read_dicom_image(image_file):
    ds = pydicom.dcmread(image_file, force=True)
    image = ds.pixel_array.astype(np.float32) / 255.0
    return image


def normalize_image(image):
    # Check the data type
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    elif image.dtype == np.uint16:
        return image.astype(np.float32) / 65535.0
    elif image.dtype == np.int16:
        # Shift to positive range and normalize
        image_shifted = image.astype(np.float32) + 32768
        return image_shifted / 65535.0
    else:
        raise ValueError("Unsupported data type")