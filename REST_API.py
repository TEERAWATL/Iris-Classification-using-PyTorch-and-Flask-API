#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import torch
import torch.nn as nn
import joblib

app = Flask(__name__)
api = Api(app)

# Define the neural network layers
input_layer = nn.Linear(4, 10)
hidden_layer = nn.Linear(10, 10)
output_layer = nn.Linear(10, 3)

# Load the trained model and label encoder
model = nn.Sequential(input_layer, nn.ReLU(), hidden_layer, nn.ReLU(), output_layer)
model.load_state_dict(torch.load('iris_model_pytorch.pt'))
model.eval()

encoder = joblib.load('label_encoder_pytorch.pkl')

class Predict(Resource):
    def post(self):
        input_data = request.get_json(force=True)
        input_features = np.array(input_data['features']).reshape(1, -1)
        input_features_torch = torch.tensor(input_features, dtype=torch.float32)

        with torch.no_grad():
            outputs = model(input_features_torch)
            _, predicted = torch.max(outputs.data, 1)

        label = encoder.inverse_transform(predicted.numpy())

        return jsonify({'class': label.tolist()})

api.add_resource(Predict, '/predict')

if __name__ == '__main__':
    app.run(debug=True)

