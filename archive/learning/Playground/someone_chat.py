#!/usr/bin/env python3
#
# someone_chat.py

import requests

# Define the API endpoint
url = "http://ai.sr:5000/predict"

max_length = 8192
temperature = .7
num_return_sequences = 1

while True:
    message = input("Enter a message: ")
    # Define the data to be sent to the API
    data = {
        "input_text": message,
        "max_length": max_length,
        "temperature": temperature,
        "num_return_sequences": num_return_sequences
    }

    # Send a POST request to the API and get the response
    response = requests.post(url, json=data)

    if response.status_code == 200:
        print("Generated Text: ", response.text)
    else:
        print(f"Request failed with status code {response.status_code}")
