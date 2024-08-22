import requests
import base64
import glob as glob
import os


images = glob.glob(os.path.join('data/', '*.dcm'))

for image in images:
    with open(image, 'rb') as f:
        encoded_image = base64.b64encode(f.read()).decode('utf-8')

    data = {'image': encoded_image}
    response = requests.post('http://127.0.0.1:5000/predict', json=data)
    print(response.json())