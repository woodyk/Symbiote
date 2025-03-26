#!/usr/bin/env python3
#
# abc.py

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import requests
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# Initialize processor and model
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

# Load the image
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
url = "https://www.shutterstock.com/shutterstock/photos/1690131103/display_1500/stock-photo-st-jo-texas-usa-march-close-up-side-view-of-a-man-wearing-an-open-carry-handgun-in-1690131103.jpg"
url = "https://americanhandgunner.com/wp-content/uploads/2022/08/AHJA07-BIG-043.jpg"
url = "https://lailluminator.com/wp-content/uploads/2022/06/concealed-carry-2-1536x865.jpg"
url = "https://i.guim.co.uk/img/media/34cab48b42513abcb7f149a7dc79205d612cac52/0_154_4624_2775/master/4624.jpg?width=1200&height=900&quality=85&auto=format&fit=crop&s=5a962722f982f214feea98a40a9c0f8a"
image = Image.open(requests.get(url, stream=True).raw)

# Define the text prompts for detection
texts = [["gun", "pistol", "shotgun", "rifle", "firearm"]]

# Prepare inputs for the model
inputs = processor(text=texts, images=image, return_tensors="pt")
outputs = model(**inputs)

# Target image sizes (height, width) to rescale box predictions [batch_size, 2]
target_sizes = torch.Tensor([image.size[::-1]])
# Convert outputs (bounding boxes and class logits) to COCO API
results = processor.post_process_object_detection(outputs=outputs, threshold=0.1, target_sizes=target_sizes)

# Draw bounding boxes and labels on the image
i = 0  # Retrieve predictions for the first image for the corresponding text queries
text = texts[i]
boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

# Convert to a draw object
draw = ImageDraw.Draw(image, "RGBA")

# Define font (optional, depending on availability)
try:
    font = ImageFont.truetype("arial.ttf", 15)
except IOError:
    font = ImageFont.load_default()

for box, score, label in zip(boxes, scores, labels):
    box = [round(i, 2) for i in box.tolist()]
    x_min, y_min, x_max, y_max = box

    # Draw a transparent rectangle
    draw.rectangle([x_min, y_min, x_max, y_max], outline=(255, 0, 0, 255), fill=(255, 0, 0, 128))

    # Draw the label with confidence score
    label_text = f"{text[label]}: {round(score.item(), 3)}"
    text_bbox = draw.textbbox((x_min, y_min), label_text, font=font)
    text_location = [x_min, y_min - (text_bbox[3] - text_bbox[1])]
    draw.rectangle([text_location[0], text_location[1], text_location[0] + (text_bbox[2] - text_bbox[0]), text_location[1] + (text_bbox[3] - text_bbox[1])], fill=(255, 0, 0, 255))
    draw.text(text_location, label_text, fill=(255, 255, 255, 255), font=font)

# Display the image with highlighted boxes
image.show()

# Optionally, save the image with boxes
# image.save("output_with_boxes.png")

