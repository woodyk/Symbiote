#!/usr/bin/env python3
#
# image_watcher.py

import time
import io
import os
import torch
import pyautogui
import json
import pytesseract
from PIL import Image, ImageDraw
from transformers import DetrImageProcessor, DetrForObjectDetection

# Initialize the processor and model
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

def take_screenshot():
    # Capture the screen and save it to an in-memory BytesIO object
    screenshot = pyautogui.screenshot()
    image_data = io.BytesIO()
    screenshot.save(image_data, format='PNG')
    image_data.seek(0)
    return image_data

def detect_and_display_objects(image_data):
    try:
        with Image.open(image_data) as image:
            # Ensure the image is in RGB mode
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Perform text extraction (OCR)
            extracted_text = pytesseract.image_to_string(image)
            if extracted_text.strip():
                print("\n--- Extracted Text ---")
                print(extracted_text)

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
            if detections:
                detections_json = json.dumps(detections, indent=4)
                print("\n--- Detected Objects ---")
                print(detections_json)

                # Draw bounding boxes on the image if any objects are detected
                draw = ImageDraw.Draw(image)
                for detection in detections:
                    box = detection["location"]
                    label_name = detection["label_name"]
                    confidence = detection["confidence"]

                    # Draw a transparent rectangle with a colored outline
                    draw.rectangle(box, outline="red", width=3)
                    draw.text((box[0], box[1] - 10), f"{label_name}: {confidence}", fill="red")

                # Display the image with bounding boxes
                #image.show()

    except Exception as e:
        print(f"Failed to process screenshot: {e}")

def monitor_screen(interval=5):
    while True:
        # Take a screenshot
        screenshot_data = take_screenshot()
        detect_and_display_objects(screenshot_data)

        # Close the in-memory image data to free up resources
        screenshot_data.close()

        # Wait for the specified interval before taking the next screenshot
        time.sleep(interval)

if __name__ == "__main__":
    print("Starting screen monitoring with DETR and OCR...")
    # Start monitoring the screen, taking screenshots every 5 seconds
    monitor_screen(interval=5)

