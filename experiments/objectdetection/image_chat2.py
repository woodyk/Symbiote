#!/usr/bin/env python3
#
# image_chat2.py

import requests
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

# Load the processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

# Ensure the model is configured for CPU
model = model.to('cpu')

def load_image(image_path_or_url):
    if image_path_or_url.startswith('http://') or image_path_or_url.startswith('https://'):
        response = requests.get(image_path_or_url, stream=True)
        image = Image.open(response.raw).convert('RGB')
    else:
        image = Image.open(image_path_or_url).convert('RGB')
    return image

def ask_question(image, question):
    inputs = processor(image, question, return_tensors="pt")
    out = model.generate(**inputs)
    answer = processor.decode(out[0], skip_special_tokens=True)
    return answer

def main():
    print("Welcome to the Image VQA Chatbot!")
    image_path_or_url = input("Please provide an image path or URL: ")
    
    try:
        image = load_image(image_path_or_url)
        print("Image loaded and ready for discussion.")
        
        while True:
            question = input("Ask a question about the image (or type 'exit' to quit): ")
            if question.lower() == 'exit':
                break
            answer = ask_question(image, question)
            print("Chatbot:", answer)
    
    except Exception as e:
        print(f"Failed to load image: {e}")

if __name__ == "__main__":
    main()

