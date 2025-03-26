#!/usr/bin/env python3
#
# suspicious_detection.py

from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Define the weapon-related classifications
weapon_classes = [
    "firearm",
    "knife",
    "no weapon"
]

# URL of the image to classify
url = "https://www.alreporter.com/wp-content/uploads/2019/04/AdobeStock_142035778-e1556623470812.jpeg"
url = "https://www.offgridweb.com/wp-content/uploads/2016/02/The-Gray-Man-crowd.jpg"
url = "https://cardinidefense.com/cdn/shop/files/ConcealedCarryTaylor_7_2048x.jpg?v=1701457246"
url = "https://hidinghilda.com/cdn/shop/files/sling-leather-backpack-concealed-carry-purse-for-women-lady-conceal-bags-34150608470060_540x_39768824-dc54-47e1-9b4f-097a1c785c98.webp?v=1724440883&width=1946"
url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcThH_0dwUHjtCPdB_8ApHSjW0BzrDXs39062w&s"
url = "https://static.vecteezy.com/system/resources/previews/002/635/601/large_2x/business-people-walking-and-talk-to-each-other-photo.jpg"

# Open the image
image = Image.open(requests.get(url, stream=True).raw)

# Process the image and the weapon classifications
inputs = processor(text=weapon_classes, images=image, return_tensors="pt", padding=True)

# Get the model outputs
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image # image-text similarity score
probs = logits_per_image.softmax(dim=1) # get label probabilities

# Display the probabilities for each classification
for i, weapon_class in enumerate(weapon_classes):
    print(f"{weapon_class}: {probs[0][i].item():.4f}")

