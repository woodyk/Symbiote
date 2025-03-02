#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: ObjectDetection.py
# Author: Wadih Khairallah
# Description: 
# Created: 2024-11-27 06:25:36
#!/usr/bin/env python3
#
# ObjectDetection.py

import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image, ImageDraw, ImageFont
import requests
import json
import random
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class ObjectDetection:
    def __init__(self, model_name="facebook/detr-resnet-50", revision="no_timm"):
        self.processor = DetrImageProcessor.from_pretrained(model_name, revision=revision)
        self.model = DetrForObjectDetection.from_pretrained(model_name, revision=revision)

    def load_image(self, image_source):
        """
        Loads an image from a URL or a local file path.
        """
        if image_source.startswith(('http://', 'https://')):
            image = Image.open(requests.get(image_source, stream=True).raw)
        else:
            image = Image.open(image_source)
        return image

    def detect_objects(self, image, threshold=0.1):
        """
        Detects objects in the given image using the loaded model.
        """
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]

        objects = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            detected_object = {
                "object": self.model.config.id2label[label.item()],
                "confidence": round(score.item(), 3),
                "location": box
            }
            objects.append(detected_object)
        
        return objects

    def draw_boxes(self, image, objects):
        """
        Draws bounding boxes and labels on the image.
        """
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()  # You can specify a path to a .ttf file if available

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

        return image

    def process_image(self, image_source, threshold=0.1):
        """
        Processes the image, detects objects, and returns the image with boxes and the detection JSON.
        """
        image = self.load_image(image_source)
        objects = self.detect_objects(image, threshold=threshold)
        labeled_image = self.draw_boxes(image, objects)
        return labeled_image, objects


if __name__ == "__main__":
    # Initialize the object detection
    detector = ObjectDetection()

    # Process an image from a URL
    image_source = "https://img.huffingtonpost.com/asset/604a9b78250000ab0084d7d8.jpeg?cache=VzlnZffmkh&ops=1200_630"
    image_source = "https://images.saymedia-content.com/.image/t_share/MTc0MzUzNzE2MzQ0NzkyNzEw/how-to-clean-a-messy-room-quickly.jpg"
    labeled_image, detected_objects = detector.process_image(image_source, threshold=0.3)

    # Display the labeled image
    labeled_image.show()

    # Print the JSON output
    print(json.dumps(detected_objects, indent=4))
