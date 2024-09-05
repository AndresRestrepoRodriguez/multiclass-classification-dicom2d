import requests
import base64
import glob as glob
import os
import pydicom


"""images_bodyfront_train = glob.glob(os.path.join('data/dataset/bodyfront/train', '*.dcm'))
images_bodyfront_val = glob.glob(os.path.join('data/dataset/bodyfront/val', '*.dcm'))
images_bodyside_train = glob.glob(os.path.join('data/dataset/bodyside/train', '*.dcm'))
images_bodyside_val = glob.glob(os.path.join('data/dataset/bodyside/val', '*.dcm'))
images_otherview_train = glob.glob(os.path.join('data/dataset/otherview/train', '*.dcm'))
images_otherview_val = glob.glob(os.path.join('data/dataset/otherview/val', '*.dcm'))"""


testing_path = glob.glob(os.path.join('data/val/otherview', '*.dcm'))

ct = 0
for image in testing_path:
    # Read the DICOM file
    dicom_image = pydicom.dcmread(image, force=True)
    
    # Get the pixel array from the DICOM file
    image_array = dicom_image.pixel_array
    print(image_array.dtype)

    modality = dicom_image['Modality'].value
    print(modality)
