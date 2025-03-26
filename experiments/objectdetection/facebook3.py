#!/usr/bin/env python3
#
# facebook3.py

from transformers import DetrImageProcessor, DetrForObjectDetection, logging
import torch
import json
from PIL import Image, ImageDraw, ImageFont
import requests
import warnings
import random

warnings.simplefilter(action='ignore', category=FutureWarning)

# Load the image from a URL
url = "https://img.huffingtonpost.com/asset/604a9b78250000ab0084d7d8.jpeg?cache=VzlnZffmkh&ops=1200_630"
image = Image.open(requests.get(url, stream=True).raw)

# Initialize the image processor and model
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

# Preprocess the image and get model outputs
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# Convert outputs to COCO API format and filter results
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.1)[0]

# Extract detected objects and their details
objects = []
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    detected_object = {
        "object": model.config.id2label[label.item()],
        "confidence": round(score.item(), 3),
        "location": box
    }
    objects.append(detected_object)

print(json.dumps(objects, indent=4))

# Draw bounding boxes and labels on the image
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()  # Using default font; you can specify a path to a .ttf file if available

for obj in objects:
    label = obj['object']
    confidence = obj['confidence']
    box = obj['location']
    
    # Choose a random color for each box
    color = tuple(random.choices(range(256), k=3))
    
    # Draw the bounding box
    draw.rectangle(box, outline=color, width=3)
    
    # Draw the label and confidence score above the bounding box
    label_text = f"{label} ({confidence})"
    text_bbox = draw.textbbox((0, 0), label_text, font=font)
    text_size = (text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1])
    
    text_location = (box[0], box[1] - text_size[1]) if box[1] - text_size[1] > 0 else (box[0], box[1])
    text_box_coords = [text_location[0], text_location[1], text_location[0] + text_size[0], text_location[1] + text_size[1]]
    
    draw.rectangle(text_box_coords, fill=color)
    draw.text(text_location, label_text, fill="white", font=font)

# Show the image with bounding boxes and labels
image.show()

