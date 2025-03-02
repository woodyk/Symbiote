#!/usr/bin/env python3
#
# donna.py

import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Symbiote:
    def __init__(self, model_path):
        self.model = self.load_h5_model(model_path)
        self.tokenizer = Tokenizer(num_words=10000)  # Set to appropriate number of words

    def load_h5_model(self, model_path):
        return load_model(model_path)

    def prepare_input(self, input_text):
        sequences = self.tokenizer.texts_to_sequences([input_text])
        padded_input = pad_sequences(sequences, maxlen=100)  # Replace 100 with your max length
        return padded_input

    def generate_response(self, input_text):
        processed_input = self.prepare_input(input_text)
        response = self.model.predict(processed_input)
        response_text = ' '.join([str(r) for r in response[0]])  # Adjust as needed
        return response_text

    def get_layers(self):
        # Get the number of layers
        num_layers = len(model.layers)
        print(f"Number of Layers: {num_layers}")

        # View layer names
        print("Layer Names: ")
        for layer in model.layers:
            print(layer.name)

        model.summary()

    def chat(self):
        print("Chatbot is running. Type 'quit' to exit.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'quit':
                break
            response = self.generate_response(user_input)
            print("Bot:", response)

# Usage
if __name__ == "__main__":
    model_path = '~/.symbiote/models/donna.h5'  # Replace with the correct path to your model
    model_path = os.path.expanduser(model_path)
    symbiote = Symbiote(model_path)
    symbiote.chat()

