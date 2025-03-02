#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: ImageAnalysis.py
# Author: Wadih Khairallah
# Description: 
# Created: 2024-11-27 06:41:26

import os
import sys
import base64
import ollama
import json
import requests
from deepface import DeepFace
from PIL import Image, ImageDraw, ImageFont
import webbrowser
import tempfile
from bs4 import BeautifulSoup
from io import BytesIO
import pytesseract
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
import time

class ImageAnalyzer:

    def __init__(self, detection=False, extract_text=False, backend='opencv'):
        self.detection = detection
        self.extract_text = extract_text
        self.backend = backend
        if self.detection:
            self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
            self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

    def load_image(self, image_source):
        try:
            if image_source.startswith(('http://', 'https://')):
                img = Image.open(requests.get(image_source, stream=True).raw)
            else:
                img = Image.open(image_source)
            return img
        except Exception as e:
            print(f"Error opening path: {e}")
            return None

    def analyze_faces(self, image_path):
        try:
            analysis_results = DeepFace.analyze(
                img_path=image_path,
                actions=['emotion', 'age', 'gender', 'race'],
                enforce_detection=True,
                detector_backend=self.backend,
                silent=True
            )
            return analysis_results
        except Exception as e:
            print(f"An error occurred while analyzing the photo: {e}")
            return None

    def draw_boxes(self, image, analysis_results):
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        for i, result in enumerate(analysis_results):
            region = result['region']
            box = (region['x'], region['y'], region['x'] + region['w'], region['y'] + region['h'])

            draw.rectangle(box, outline="red", width=3)

            label_text = f"Person {i+1}\nGender: {result['dominant_gender']}, Age: {result['age']}"
            text_bbox = draw.textbbox((0, 0), label_text, font=font)
            text_size = (text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1])
            text_location = (box[0], box[1] - text_size[1] - 10 if box[1] - text_size[1] - 10 > 0 else box[1])

            background_box = [text_location[0] - 5, text_location[1] - 5, text_location[0] + text_size[0] + 5, text_location[1] + text_size[1] + 5]
            draw.rectangle(background_box, fill=(0, 0, 0, 128))

            draw.text(text_location, label_text, fill="white", font=font)

        return image

    def zero_shot_object_detection(self, image):
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt")
            outputs = self.model(**inputs)
            target_sizes = torch.tensor([image.size[::-1]])
            results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

            detections = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                label_name = self.model.config.id2label[label.item()]
                detections.append({
                    "label_name": label_name,
                    "confidence": round(score.item(), 3),
                    "location": box
                })

            draw = ImageDraw.Draw(image)
            for detection in detections:
                box = detection["location"]
                label_name = detection["label_name"]
                confidence = detection["confidence"]
                draw.rectangle(box, outline="red", width=3)
                draw.text((box[0], box[1] - 10), f"{label_name}: {confidence}", fill="red")

            return detections

        except Exception as e:
            print(f"Object detection failed: {e}")
            return []

    def extract_text_from_image(self, image):
        try:
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            print(f"Text extraction failed: {e}")
            return ""

    def analyze_images(self, path, mode='html', images_per_row=3):
        def save_image_as_jpg(image):
            with tempfile.NamedTemporaryFile(suffix='.jpg', dir='/tmp', delete=False) as tmp_file:
                image.save(tmp_file.name, format='JPEG')
                return tmp_file.name

        images = []
        img_sources = []
        detections = []

        if os.path.isfile(path):
            img = self.load_image(path)
            if img:
                images.append(img)
                img_sources.append(path)
        elif path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')):
            img = self.load_image(path)
            if img:
                images.append(img)
                img_sources.append(path)
        else:
            response = requests.get(path)
            if response.status_code != 200:
                raise Exception(f"Failed to fetch the webpage: {response.status_code}")

            soup = BeautifulSoup(response.content, 'html.parser')
            img_tags = soup.find_all('img')

            for img_tag in img_tags:
                img_url = img_tag.get('src')
                if img_url and not img_url.startswith(('http://', 'https://')):
                    img_url = requests.compat.urljoin(path, img_url)

                try:
                    img = self.load_image(img_url)
                    if img:
                        images.append(img)
                        img_sources.append(img_url)
                except Exception as e:
                    print(f"Failed to open image: {img_url} - {e}")

        for img in images:
            result = {"image_source": img_sources[images.index(img)]}

            if self.detection:
                detected_objects = self.zero_shot_object_detection(img)
                result["detections"] = detected_objects
                image_description = self.describe_image(img)
                result["description"] = image_description

                if any(d['label_name'] == 'person' for d in detected_objects):
                    temp_image_path = save_image_as_jpg(img)
                    deepface_analysis = self.analyze_faces(temp_image_path)
                    result["deepface_analysis"] = deepface_analysis

                    if deepface_analysis:
                        img = self.draw_boxes(img, deepface_analysis)

                    if temp_image_path and os.path.exists(temp_image_path):
                        os.remove(temp_image_path)


            if self.extract_text:
                extracted_text = self.extract_text_from_image(img)
                result["extracted_text"] = extracted_text

            detections.append(result)

        if mode == 'html':
            html_content = """
            <html>
            <head>
            <style>
                body { background-color: #121212; color: #FFFFFF; font-family: 'Courier New', monospace; }
                .container { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; padding: 20px; }
                .card { background-color: #1E1E1E; border-radius: 10px; padding: 20px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); text-align: center; }
                .card img { max-width: 100%; height: auto; border-radius: 10px; display: block; margin-left: auto; margin-right: auto; }
                .card a { color: #BB86FC; text-decoration: none; }
                .card a:hover { text-decoration: underline; }
                h1 { text-align: center; color: #ff8c00; } /* Hacker orange color */
                .json-content { background-color: #000000; color: #FFFFFF; padding: 10px; border-radius: 10px; display: none; font-size: 12px; text-align: left; }
                .expand-button { cursor: pointer; color: #BB86FC; margin-top: 10px; }
            </style>
            <script>
                function toggleContent(id) {
                    var content = document.getElementById(id);
                    if (content.style.display === "none") {
                        content.style.display = "block";
                    } else {
                        content.style.display = "none";
                    }
                }
            </script>
            </head>
            <body>
            <h1>Extracted Images</h1>
            <div class="container">
            """

            for i, result in enumerate(detections):
                json_content = f"""
                <div class="expand-button" onclick="toggleContent('json{i}')">▼ Detections JSON</div>
                <div class="json-content" id="json{i}">
                    <pre>{json.dumps(result, indent=4)}</pre>
                </div>
                """

                html_content += f"""
                <div class="card">
                    <a href="{result['image_source']}" target="_blank">
                        <img src="{result['image_source']}" alt="Image" />
                    </a>
                    {json_content}
                </div>
                """

            html_content += """
            </div>
            </body>
            </html>
            """

            try:
                with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html', dir='/tmp') as f:
                    f.write(html_content)
                    temp_file_path = f.name

                webbrowser.open(f'file://{os.path.realpath(temp_file_path)}')

                time.sleep(2)

            finally:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

        return detections

    def render_human_readable(self, data):
        rendered_output = ""

        for idx, entry in enumerate(data):
            rendered_output += f"### Image {idx + 1}\n"
            rendered_output += f"- **Image Source:** [{entry['image_source']}]({entry['image_source']})\n"

            # Render Detected Objects
            if 'detections' in entry and entry['detections']:
                rendered_output += "- **Detected Objects:**\n"
                for obj in entry['detections']:
                    rendered_output += f"  - **Label:** {obj['label_name']}, **Confidence:** {obj['confidence']}, **Location:** {obj['location']}\n"
            else:
                rendered_output += "- **Detected Objects:**\n  - No objects detected.\n"

            # Render DeepFace Analysis
            if 'deepface_analysis' in entry and entry['deepface_analysis']:
                rendered_output += "- **DeepFace Analysis:**\n"
                for idx, face in enumerate(entry['deepface_analysis']):
                    rendered_output += f"  - **Face {idx + 1}:**\n"
                    rendered_output += f"    - **Age:** {face.get('age', 'N/A')}\n"
                    rendered_output += f"    - **Gender:** {face.get('dominant_gender', 'N/A')} ({max(face.get('gender', {}).values(), default='N/A')}%)\n"
                    rendered_output += f"    - **Dominant Race:** {face.get('dominant_race', 'N/A')} ({max(face.get('race', {}).values(), default='N/A')}%)\n"
                    rendered_output += f"    - **Dominant Emotion:** {face.get('dominant_emotion', 'N/A')} ({max(face.get('emotion', {}).values(), default='N/A')}%)\n"
            else:
                rendered_output += "- **DeepFace Analysis:**\n  - No face detected.\n"

            # Render Extracted Text
            extracted_text = entry.get('extracted_text', '')
            if extracted_text:
                rendered_output += f"- **Extracted Text:**\n  - \"{extracted_text}\"\n"
            else:
                rendered_output += "- **Extracted Text:**\n  - No text extracted.\n"


            # Render Description
            extracted_description = entry.get('description', '')
            if extracted_description:
                rendered_output +=f"- **Description:**\n  - \"{extracted_description}\"\n"
            else:
                rendered_output +="- **Description:**\n  - \" No description extracted.\n"

            rendered_output += "\n"  # Add a newline between image entries for readability

        return rendered_output

    def image_to_base64(self, img):
        try:
            # Convert image to RGB if it is in P mode or any other incompatible mode
            if img.mode in ("P", "RGBA", "LA", "1"):
                img = img.convert("RGB")

            buffered = BytesIO()
            img.save(buffered, format="JPEG")  # Saving the image to the buffer in JPEG format
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")  # Converting to base64
            return img_str
        except Exception as e:
            print(f"Failed to convert image to base64: {e}")
            return None

    def describe_image(self, img):
        try:
            encoded_image = self.image_to_base64(img)
            if not encoded_image:
                return None

            # Call the Ollama API with the LLaVA model
            try:
                response = ollama.generate(
                    model='minicpm-v',
                    prompt="Please describe the content of the image.",
                    images=[encoded_image]
                )

                # Extract the generated text from the response
                detected_objects = response['response']
                return detected_objects
            except Exception as e:
                print(f"Error processing image: {e}")
                return None
        except Exception as e:
            print(f"Failed to describe image: {e}")
            return None

# Example usage
if __name__ == "__main__":
    extractor = ImageAnalyzer(detection=True, extract_text=True, backend='mtcnn')
    path = 'https://www.google.com/search?num=10&newwindow=1&sca_esv=019edcabf1e12258&sca_upv=1&q=text&udm=2&fbs=AEQNm0A2upiO_GHeTz6R89sNEjTHXSUfB8x3gweQ77S5CBNH1qkT9yo9p8LiN9Ph7QzgeH30iP61xKdhHkWU-Ava8l8nz9PkGePjGF8Xy5hRILCU_i0WJtOhvdQtLcJTjKD65-XLl9jT4l6cw86Jk_hwDRHHDS8CHxrc85H5U7_MQxCP3a_KWAcZPuVhBaxiP-PKqs6uCp1N4n6aTDbog6fOLFt_TfQI4A&sa=X&ved=2ahUKEwiW87bNo5uIAxU5r4QIHSz0CFoQtKgLegQIExAB&biw=1728&bih=958&dpr=2#imgrc=YrwwLQ0P_highM&imgdii=OzNsao3_-WQiwM'
    path = 'https://www.google.com/search?num=10&newwindow=1&sca_esv=019edcabf1e12258&sca_upv=1&q=faces&udm=2&fbs=AEQNm0DfIOrc-JVK7JSIXTRypzh1d0Xgc-Qx-MWluXtsBh8oLqezDGS-GBRoVhGkujFN9L89G8OpQ_4dZf-PrNf_f2qsBtQmtLqsm6yxAHTedliasJBB9FQSsYwZnyPqqc5ldCsnkGqLbFClqXddPoG99GG8FOnDSp-qlZzIkKjMKkP9CcmcXuDZxiWgaR1I6cT030OuyH9GGaehPhQtpHoYO4976gklqg&sa=X&ved=2ahUKEwiIz7qOpZuIAxXen4QIHRucJ2gQtKgLegQIERAB&biw=1728&bih=958&dpr=2'
    #results = extractor.analyze_images(path, mode='html')
    results = extractor.analyze_images(path, mode='none')
    human_readable = extractor.render_human_readable(results)
    print(json.dumps(results, indent=4))
    print(human_readable)


"""
# ImageAnalyzer Class Library

## Overview

The `ImageAnalyzer` class is a powerful tool designed for extracting and analyzing various features from images. It supports object detection, text extraction, face analysis, and image description generation using advanced models like DeepFace and LLaVA. The library can process images from files, URLs, and even entire webpages, generating both human-readable and JSON-formatted outputs.

## Features

- **Object Detection**: Detects objects within images using a zero-shot object detection model.
- **Text Extraction**: Extracts textual content from images using OCR (Optical Character Recognition).
- **Face Analysis**: Analyzes faces in images to determine attributes like age, gender, race, and dominant emotions using DeepFace.
- **Image Description**: Generates a natural language description of the image using the LLaVA model from `ollama`.
- **HTML Report Generation**: Creates a comprehensive HTML report with expandable JSON data for each analyzed image.

## Installation

Before using the `ImageAnalyzer` class, ensure you have installed the required dependencies:

```bash
pip install deepface ollama pytesseract transformers torch
```

You may also need to install additional system dependencies for `pytesseract` and `DeepFace` backends.

## Usage

### 1. Initialization

You can initialize the `ImageAnalyzer` class by specifying whether to enable object detection and text extraction.

```python
from image_extractor import ImageAnalyzer

extractor = ImageAnalyzer(detection=True, extract_text=True, backend='mtcnn')
```

### 2. Analyzing Images

Use the `analyze_images` method to analyze images from a file path, direct URL, or webpage. The results can be returned as an HTML report or a JSON object.

```python
path = 'https://example.com/image.jpg'
results = extractor.analyze_images(path, mode='html')
```

### 3. Rendering Human-Readable Results

You can convert the JSON results into a more readable format using the `render_human_readable` method.

```python
human_readable = extractor.render_human_readable(results)
print(human_readable)
```

### 4. Example Usage

Here's a complete example that initializes the class, analyzes images from a URL, and prints both the JSON results and a human-readable summary.

```python
from image_extractor import ImageAnalyzer

extractor = ImageAnalyzer(detection=True, extract_text=True, backend='mtcnn')
path = 'https://example.com/image.jpg'
results = extractor.analyze_images(path, mode='html')
human_readable = extractor.render_human_readable(results)

print(json.dumps(results, indent=4))
print(human_readable)
```

## Advanced Features

### Object Detection

The `ImageAnalyzer` can detect objects within an image using a zero-shot object detection model. Detected objects are highlighted with bounding boxes in the image.

### DeepFace Integration

When faces are detected, the `ImageAnalyzer` uses DeepFace to analyze and extract details about age, gender, race, and dominant emotions. These details are drawn directly onto the image.

### Image Description with LLaVA

The `describe_image` method calls the LLaVA model from `ollama` to generate a natural language description of the image, which is then included in the analysis results.

## Conclusion

The `ImageAnalyzer` class is a versatile tool for image analysis, providing comprehensive insights into image content through object detection, text extraction, face analysis, and more. Whether generating detailed reports or integrating with larger systems, this class is designed to be flexible and powerful.
"""
