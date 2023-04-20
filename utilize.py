#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib
import pickle

def load_data():
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, encoder

def load_model(model_path):
    input_layer = nn.Linear(4, 10)
    hidden_layer = nn.Linear(10, 10)
    output_layer = nn.Linear(10, 3)

    model = nn.Sequential(input_layer, nn.ReLU(), hidden_layer, nn.ReLU(), output_layer)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def evaluate_model(model, encoder, X_test, y_test):
    X_test_torch = torch.tensor(X_test, dtype=torch.float32)
    
    with torch.no_grad():
        y_pred = model(X_test_torch)
        _, y_pred_classes = torch.max(y_pred.data, 1)

    y_pred_classes = y_pred_classes.numpy()
    accuracy = accuracy_score(y_test, y_pred_classes)
    conf_matrix = confusion_matrix(y_test, y_pred_classes)

    return accuracy, conf_matrix

if __name__ == '__main__':
    X_train, X_test, y_train, y_test, encoder = load_data()
    model = load_model('iris_model_pytorch.pt')
    accuracy, conf_matrix = evaluate_model(model, encoder, X_test, y_test)
    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix:")
    print(conf_matrix)

