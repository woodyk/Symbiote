#!/usr/bin/env python3
#
# tt.py

import re
import os
import time
import requests

API_KEY = os.getenv("HUGGINGFACE_API_KEY") 

class huggingBot:
    def __init__(self):
        self.api_url = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
        self.headers = {"Authorization": f"Bearer {API_KEY}"}

    def query(self, payload):
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        return response.json()

    def get_response(self, user_input):
        output = self.query({"inputs": user_input})
        return output[0]['generated_text']

    def strip_to_punctuation(self, text):
        match = re.search(r"[^.,!?;:]*$", text)
        if match:
            # Subtract match length from total length to find the index of the last punctuation
            return text[:len(text) - len(match.group(0))]
        return text  # Return the original text if no punctuation is found

    def trim(self, user_input, prev_text):
        if prev_text:
            uilen = len(user_input)
            trimmed = prev_text[uilen:]
        else:
            trimmed = user_input

        return trimmed

    def run(self, user_input):
        print("---")
        user_input = self.get_response(user_input)
        print(user_input)
        prev_text = user_input 
        #prev_text = False 
        while True:
            user_input = self.get_response(user_input)
            if user_input == prev_text or len(user_input) >= 5000:
                break

            prev_text = user_input
            time.sleep(0.2)

        print(user_input)
        print("---")
        return user_input

if __name__ == "__main__":
    chatbot = huggingBot()
    chatbot.run("Write a script that counts from 1 to 10 using javascript.")

