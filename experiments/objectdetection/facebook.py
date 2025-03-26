#!/usr/bin/env python3
#
# facebook.py

from transformers import DetrImageProcessor, DetrForObjectDetection, logging
import torch
import json
from PIL import Image
import requests
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

url = "https://thumbs.dreamstime.com/z/people-walking-along-busy-city-street-way-to-work-johannesburg-south-africa-february-122086536.jpg"
url = "https://img.huffingtonpost.com/asset/604a9b78250000ab0084d7d8.jpeg?cache=VzlnZffmkh&ops=1200_630"
image = Image.open(requests.get(url, stream=True).raw)

#pic = "/Users/kato/Pictures/face_test_image.jpg"
#image = Image.open(pic)

# you can specify the revision tag if you don't want the timm dependency
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# convert outputs (bounding boxes and class logits) to COCO API
# let's only keep detections with score > 0.9
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.1)[0]

objects = []
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    object = {
            "object": model.config.id2label[label.item()],
            "confidence": round(score.item(), 3),
            "location": box
            }
    objects.append(object)

print(json.dumps(objects, indent=4))
