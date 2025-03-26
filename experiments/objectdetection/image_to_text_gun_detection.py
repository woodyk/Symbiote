#!/usr/bin/env python3
#
# aaaa.py

import requests
from io import BytesIO
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Function to load an image from a URL
def load_image_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image

# Function to generate image caption
def generate_caption(image):
    # Initialize the processor and model
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    # Preprocess the image and generate the caption
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)

    # Decode the output caption
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# Main function
def main():
    url = "https://www.lvcriminaldefense.com/wp-content/uploads/2015/03/concealed-carry.jpg"
    url = "https://img.freepik.com/free-photo/young-serious-blonde-handsome-man-holds-gun-isolated-white-background-with-copy-space_141793-65630.jpg"
    image = load_image_from_url(url)

    caption = generate_caption(image)
    print("Generated Caption:", caption)

if __name__ == "__main__":
    main()

