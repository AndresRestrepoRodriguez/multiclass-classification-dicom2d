import base64
import io
from flask import Flask, jsonify, request
from src.utils.model import read_pytorch_model_eval, read_torchcript_model_eval
from src.models.predict_model import predict_model_citadel, predict_model, predict_model_docker
from src.utils.data import (
    read_yaml
)
from src.utils.schemas import MulticlassPytorchModel
import os
from PIL import Image
import numpy as np


app = Flask(__name__)

model_path = 'models/docker_multiclass_dicom2d.torchscript'
model = read_torchcript_model_eval(weights=model_path)


@app.route('/predict', methods=['POST'])
def predict():
    global model
    data = request.get_json()
    img_data = Image.open(io.BytesIO(base64.b64decode(data['image'])))
    

    predictions = predict_model_docker(image_data=img_data,
                                       model=model)
    
    return jsonify({'predictions': predictions})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
    