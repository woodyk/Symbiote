#!/usr/bin/env python3
#
# image_chat.py

import torch
from PIL import Image
import requests
from io import BytesIO
from transformers import AutoModel, AutoTokenizer

# Load the model and tokenizer
model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2', trust_remote_code=True, torch_dtype=torch.bfloat16)

# Adjust this based on your hardware setup
model = model.to(device='cpu', dtype=torch.bfloat16)  # For Nvidia GPUs support BF16 (like A100, H100, RTX3090)
# model = model.to(device='cuda', dtype=torch.float16)  # Uncomment for GPUs without BF16 support
# model = model.to(device='mps', dtype=torch.float16)   # Uncomment for Mac with MPS support

tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2', trust_remote_code=True)
model.eval()

def load_image(image_path_or_url):
    if image_path_or_url.startswith('http://') or image_path_or_url.startswith('https://'):
        response = requests.get(image_path_or_url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_path_or_url).convert('RGB')
    return image

def ask_question(image, question):
    msgs = [{'role': 'user', 'content': question}]
    
    res, context, _ = model.chat(
        image=image,
        msgs=msgs,
        context=None,
        tokenizer=tokenizer,
        sampling=True,
        temperature=0.7
    )
    return res

def main():
    print("Welcome to the Image Chatbot!")
    image_path_or_url = input("Please provide an image path or URL: ")
    
    try:
        image = load_image(image_path_or_url)
        print("Image loaded and ready for discussion.")
        
        while True:
            question = input("Ask a question about the image (or type 'exit' to quit): ")
            if question.lower() == 'exit':
                break
            response = ask_question(image, question)
            print("Chatbot:", response)
    
    except Exception as e:
        print(f"Failed to load image: {e}")

if __name__ == "__main__":
    main()

