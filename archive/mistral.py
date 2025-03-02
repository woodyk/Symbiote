#!/usr/bin/env python3
#
# mistral2.py

import os
import requests
import json

def call_mistral_api(api_key, api_url, request_body):
    headers = {
        'x-api-key': api_key,
        'Content-Type': 'application/json'
    }
    response = requests.post(api_url, headers=headers, data=json.dumps(request_body))
    return response.json()

def generate_response(request_body):
    # Replace with your actual Mistral AI API key
    api_key = os.getenv("MISTRAL_AI_API_KEY") 
    api_url = "https://api.mistral.ai/v1/chat/completions"

    response_data = call_mistral_api(api_key, api_url, request_body)
    return response_data["choices"][0]["message"]["content"].strip()

def chatbot(user_input):
    request_body = {
        "model": "mistral-small-latest",
        "messages": [{"role": "user", "content": user_input}],
        "temperature": 0.7,
        "max_tokens": 100,
    }

    response = generate_response(request_body)
    return response

if __name__ == "__main__":
    print("Welcome to the chatbot! (Type 'quit' to exit)")

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() == "quit":
            break

        bot_response = chatbot(user_input)
        print(f"Bot: {bot_response}")

