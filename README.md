# Iris-Classification-using-PyTorch-and-Flask-API

This project demonstrates how to train a simple deep learning model using PyTorch to classify iris flowers into three species based on their sepal and petal measurements. The model is then exposed through a REST API using Flask.

## Installation
1. Clone the repository
2. Navigate to your project's root directory using the cd command cd path/to/your/project
3. Create a new conda environment and activate it

conda create -n my_project_env python=3.8

conda activate my_project_env

4. Install the required packages

pip install -r requirements.txt

## Usage

1. Train the model

python train_model.py

2. Evaluate the model

python utilize.py

3. Run the Flask REST API

python REST_API.py

4. Test the REST API
'''
import requests

url = "http://127.0.0.1:5000/predict"
data = {"features": [5.1, 3.5, 1.4, 0.2]}  # Replace with your own feature values

response = requests.post(url, json=data)
print(response.json())
'''
This script sends a POST request to the /predict endpoint with a sample iris flower's measurements and prints out the predicted class.

## To send a request and receive a response using Postman, follow these steps:

1. Download and install Postman from the official website if you haven't already.

2. Open Postman and create a new request by clicking the + button in the top left corner of the window.

3. Set the HTTP method to POST using the dropdown menu next to the URL bar.

4. Enter the URL http://127.0.0.1:5000/predict in the URL bar.

5. In the request settings tabs, below the URL bar, click on the Body tab.

6. Select the raw option and choose JSON as the data format from the dropdown menu to the right.

7. In the text area that appears, enter the JSON payload containing the iris flower's measurements. 

### For example:
### Ex1
POST:

{
  "features": [5.1, 3.5, 1.4, 0.2]
}

output:

{
  "class": [
    0
  ]
}

### Ex1
POST:

{
  "features": [6.0, 2.9, 4.5, 1.5]
}

output:

{
  "class": [
    1
  ]
}

### Ex2
POST:

{
  "features": [6.3, 3.3, 6.0, 2.5]
}

output:

{
  "class": [
    2
  ]
}


