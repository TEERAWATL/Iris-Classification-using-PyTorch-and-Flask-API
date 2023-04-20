#!/usr/bin/env python
# coding: utf-8

# In[4]:


import requests

url = "http://127.0.0.1:5000/predict"
data = {"features": [5.1, 3.5, 1.4, 0.2]}  # Replace with your own feature values

response = requests.post(url, json=data)
print(response.json())

