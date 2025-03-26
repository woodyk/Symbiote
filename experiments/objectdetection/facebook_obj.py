#!/usr/bin/env python3
#
# facebook_obj.py

from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image, ImageDraw, ImageFont
import requests
import json

# Load the image from the URL
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Initialize the processor and model
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

# Process the image and run object detection
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# Post-process the outputs to get bounding boxes with a confidence threshold
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

# Prepare the JSON document
detections = []
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    label_name = model.config.id2label[label.item()]
    detections.append({
        "label_name": label_name,
        "confidence": round(score.item(), 3),
        "location": box
    })

# Print the JSON document with indent=4
detections_json = json.dumps(detections, indent=4)
print(detections_json)

# Draw bounding boxes on the image
draw = ImageDraw.Draw(image)
for detection in detections:
    box = detection["location"]
    label_name = detection["label_name"]
    confidence = detection["confidence"]

    # Draw a transparent rectangle with a colored outline
    draw.rectangle(box, outline="red", width=3)
    draw.text((box[0], box[1] - 10), f"{label_name}: {confidence}", fill="red")

# Display the image with bounding boxes
image.show()

