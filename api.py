import base64
import io
from flask import Flask, jsonify, request
from src.utils.model import read_pytorch_model_eval, read_torchcript_model_eval
from src.models.predict_model import predict_model_citadel, predict_model
from src.utils.data import (
    read_yaml
)
from src.utils.schemas import BinaryPytorchModel
import os
from PIL import Image
import numpy as np


app = Flask(__name__)
MODELS_DIRECTORY = 'models'
MODEL_CONFIG_PATH = 'models/model_torch.yaml'

#Download model
model_cfg = read_yaml(MODEL_CONFIG_PATH)
model_structure = BinaryPytorchModel(**model_cfg)

print(model_structure)

destination_file = os.path.join(MODELS_DIRECTORY, 'tmp_model.' + model_structure.extension)

model_path = 'models/best_model_brain_chest.torchscript'
model = read_torchcript_model_eval(weights=model_path)


@app.route('/predict', methods=['POST'])
def predict():
    global model
    data = request.get_json()
    img_data = Image.open(io.BytesIO(base64.b64decode(data['image'])))
    

    prediction = predict_model_citadel(image_data=img_data,
                                       model=model)
    
    return jsonify({'predictions': prediction})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
    