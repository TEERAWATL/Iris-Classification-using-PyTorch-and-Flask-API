#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets
import joblib
import pickle

iris = datasets.load_iris()
X, y = iris.data, iris.target

encoder = LabelEncoder()
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_torch = torch.tensor(X_train, dtype=torch.float32)
y_train_torch = torch.tensor(y_train, dtype=torch.long)

# Define the neural network layers
input_layer = nn.Linear(4, 10)
hidden_layer = nn.Linear(10, 10)
output_layer = nn.Linear(10, 3)

# Create the model, loss function, and optimizer
model = nn.Sequential(input_layer, nn.ReLU(), hidden_layer, nn.ReLU(), output_layer)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 50
batch_size = 10

for epoch in range(num_epochs):
    for i in range(0, len(X_train_torch), batch_size):
        X_batch = X_train_torch[i:i+batch_size]
        y_batch = y_train_torch[i:i+batch_size]

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

# Save the trained model and label encoder
torch.save(model.state_dict(), 'iris_model_pytorch.pt')
joblib.dump(encoder, 'label_encoder_pytorch.pkl')

# Save the test data
with open('test_data_pytorch.pkl', 'wb') as f:
    pickle.dump((X_test, y_test), f)

