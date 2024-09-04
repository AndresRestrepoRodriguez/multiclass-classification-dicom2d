import pydicom
import numpy as np
import glob as glob
import os

def read_dicom_to_int16(file_path):
    """
    Reads a DICOM image and converts it to an int16 NumPy array.
    
    Parameters:
    - file_path: Path to the DICOM file.
    
    Returns:
    - image_array: Image as a NumPy array with int16 data type.
    - dicom_ds: Original pydicom Dataset.
    """
    # Read the DICOM file
    dicom_ds = pydicom.dcmread(file_path)
    
    # Get the pixel data as a NumPy array and convert to int16
    image_array = dicom_ds.pixel_array.astype(np.int16)
    
    return image_array, dicom_ds

def save_dicom_as_int16(file_path, image_array, original_dicom_ds):
    """
    Saves an int16 NumPy array as a DICOM image.
    
    Parameters:
    - file_path: Path where the DICOM file will be saved.
    - image_array: The NumPy array to save as a DICOM image.
    - original_dicom_ds: The original pydicom Dataset to copy metadata from.
    """
    # Create a copy of the original dataset to modify
    new_dicom_ds = original_dicom_ds.copy()

    # Set the pixel data to the new image array
    new_dicom_ds.PixelData = image_array.tobytes()
    new_dicom_ds.Rows, new_dicom_ds.Columns = image_array.shape
    
    # Adjust metadata to reflect the new data type
    new_dicom_ds.BitsAllocated = 16
    new_dicom_ds.BitsStored = 16
    new_dicom_ds.HighBit = 15
    new_dicom_ds.PixelRepresentation = 1  # 1 means signed integers (int16)

    # Save the modified DICOM file
    new_dicom_ds.save_as(file_path)

paths_src = ["/mnt/c/Users/USER/OneDrive - British Standards Institution/Documents/User Stories/multiclass dicom/v3/dataset-multiclass-dicom_v2/train/bodyfront",
         "/mnt/c/Users/USER/OneDrive - British Standards Institution/Documents/User Stories/multiclass dicom/v3/dataset-multiclass-dicom_v2/val/bodyfront",
         "/mnt/c/Users/USER/OneDrive - British Standards Institution/Documents/User Stories/multiclass dicom/v3/dataset-multiclass-dicom_v2/train/bodyside",
         "/mnt/c/Users/USER/OneDrive - British Standards Institution/Documents/User Stories/multiclass dicom/v3/dataset-multiclass-dicom_v2/val/bodyside",
         "/mnt/c/Users/USER/OneDrive - British Standards Institution/Documents/User Stories/multiclass dicom/v3/dataset-multiclass-dicom_v2/train/otherview",
         "/mnt/c/Users/USER/OneDrive - British Standards Institution/Documents/User Stories/multiclass dicom/v3/dataset-multiclass-dicom_v2/val/otherview"]

paths_dest = ["/mnt/c/Users/USER/OneDrive - British Standards Institution/Documents/User Stories/multiclass dicom/v3/dataset-multiclass-dicom_v2-int16/train/bodyfront",
         "/mnt/c/Users/USER/OneDrive - British Standards Institution/Documents/User Stories/multiclass dicom/v3/dataset-multiclass-dicom_v2-int16/val/bodyfront",
         "/mnt/c/Users/USER/OneDrive - British Standards Institution/Documents/User Stories/multiclass dicom/v3/dataset-multiclass-dicom_v2-int16/train/bodyside",
         "/mnt/c/Users/USER/OneDrive - British Standards Institution/Documents/User Stories/multiclass dicom/v3/dataset-multiclass-dicom_v2-int16/val/bodyside",
         "/mnt/c/Users/USER/OneDrive - British Standards Institution/Documents/User Stories/multiclass dicom/v3/dataset-multiclass-dicom_v2-int16/train/otherview",
         "/mnt/c/Users/USER/OneDrive - British Standards Institution/Documents/User Stories/multiclass dicom/v3/dataset-multiclass-dicom_v2-int16/val/otherview"]


for path_src, path_dest in zip(paths_src, paths_dest):
    print(path_src)
    print(path_dest)
    images = glob.glob(os.path.join(path_src, '*.dcm'))
    for image in images:
        name_image = os.path.basename(image)
        dest_path = os.path.join(path_dest, name_image)
        # Read and convert the DICOM image to int16
        image_array, dicom_ds = read_dicom_to_int16(image)

        # Save the image array as a new DICOM file with int16 data type
        save_dicom_as_int16(dest_path, image_array, dicom_ds)



