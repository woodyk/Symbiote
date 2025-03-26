#!/usr/bin/env python3
#
# image_analysis.py

import time
import os
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

class ImageExtractor:

    def __init__(self, detection=False, extract_text=False, backend='opencv'):
        self.detection = detection
        self.extract_text = extract_text
        self.backend = backend
        if self.detection:
            self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
            self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

    def load_image(self, image_source):
        if image_source.startswith(('http://', 'https://')):
            img = Image.open(requests.get(image_source, stream=True).raw)
        else:
            img = Image.open(image_source)
        return img

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
        texts = []
        deepface_results = []

        if os.path.isfile(path):
            img = self.load_image(path)
            images.append(img)
            img_sources.append(path)
        elif path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')):
            img = self.load_image(path)
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
                    images.append(img)
                    img_sources.append(img_url)
                except Exception as e:
                    print(f"Failed to open image: {img_url} - {e}")

        for img in images:
            result = {"image_source": img_sources[images.index(img)]}

            if self.detection:
                detected_objects = self.zero_shot_object_detection(img)
                result["detections"] = detected_objects

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
                .text-content, .detection-content, .deepface-content { background-color: #000000; color: #FFFFFF; padding: 10px; border-radius: 10px; display: none; font-size: 12px; text-align: left; }
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
                detection_content = ""
                text_content = ""
                deepface_content = ""

                if "detections" in result and result["detections"]:
                    detection_content = f"""
                    <div class="expand-button" onclick="toggleContent('detection{i}')">▼ Show Detected Objects</div>
                    <div class="detection-content" id="detection{i}">
                        <pre>"""
                    for obj in result["detections"]:
                        detection_content += f"{obj['label_name']}: Confidence {obj['confidence']}\n"
                    detection_content += "</pre></div>"

                if "extracted_text" in result and result["extracted_text"]:
                    text_content = f"""
                    <div class="expand-button" onclick="toggleContent('text{i}')">▼ Show Text</div>
                    <div class="text-content" id="text{i}">
                        <pre>{result['extracted_text']}</pre>
                    </div>
                    """

                if "deepface_analysis" in result and result["deepface_analysis"]:
                    deepface_content = f"""
                    <div class="expand-button" onclick="toggleContent('deepface{i}')">▼ Show DeepFace Analysis</div>
                    <div class="deepface-content" id="deepface{i}">
                        <pre>{json.dumps(result['deepface_analysis'], indent=4)}</pre>
                    </div>
                    """

                html_content += f"""
                <div class="card">
                    <a href="{result['image_source']}" target="_blank">
                        <img src="{result['image_source']}" alt="Image" />
                    </a>
                    {detection_content}
                    {text_content}
                    {deepface_content}
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

# Example usage
extractor = ImageExtractor(detection=True, extract_text=True, backend='mtcnn')
path = 'https://www.google.com/search?num=10&newwindow=1&sca_esv=1ebd42d001b6190b&sca_upv=1&q=frogs&udm=2&fbs=AEQNm0A-5VTqs5rweptgTqb6m-Eb3TvVcv4l7eCyod9RtZW9874wvsYjTfpwMQKGHqKPG-IB7j9flyfH28tJSLVuVdcT1tesPpIhTR_8sOQ3FQrQWiVTfWhoIplDgGh5JzUv9F4u3riMB636EHR41DrkNY_uSRk347tLZsVeJqqyuWPTyXrtg-EYkFQYZqw6rWM1khGHS26HrYFGhj2Qe1uCS-2MrLbBw&sa=X&ved=2ahUKEwjiodXZkpuIAxX3g4QIHYP4ChoQtKgLegQIHhAB&biw=1728&bih=958&dpr=2'
path = 'https://www.google.com/search?num=10&newwindow=1&sca_esv=2564783a7bea0270&sca_upv=1&q=random+objects&udm=2&fbs=AEQNm0DfIOrc-JVK7JSIXTRypzh1d0Xgc-Qx-MWluXtsBh8oLqezDGS-GBRoVhGkujFN9L89G8OpQ_4dZf-PrNf_f2qsBtQmtLqsm6yxAHTedliasJBB9FQSsYwZnyPqqc5ldCsnkGqLbFClqXddPoG99GG8FOnDSp-qlZzIkKjMKkP9CcmcXuDZxiWgaR1I6cT030OuyH9GGaehPhQtpHoYO4976gklqg&sa=X&ved=2ahUKEwjczI-ko5uIAxVRRDABHY4OHqoQtKgLegQIDhAB&biw=1728&bih=958&dpr=2'
path = 'https://www.google.com/search?q=text&num=10&newwindow=1&sca_esv=2564783a7bea0270&sca_upv=1&udm=2&biw=1728&bih=958&ei=mvnQZtWhCKeEwbkPus9Z&ved=0ahUKEwjV1oClo5uIAxUnQjABHbpnFgAQ4dUDCBE&uact=5&oq=text&gs_lp=Egxnd3Mtd2l6LXNlcnAiBHRleHQyDRAAGIAEGLEDGEMYigUyChAAGIAEGEMYigUyChAAGIAEGEMYigUyChAAGIAEGEMYigUyCxAAGIAEGLEDGIMBMgsQABiABBixAxiDATIIEAAYgAQYsQMyCBAAGIAEGLEDMggQABiABBixAzILEAAYgAQYsQMYgwFIugpQgAZYsAlwA3gAkAEAmAFOoAGrAqoBATS4AQPIAQD4AQGYAgegAsECwgIEEAAYA8ICBRAAGIAEwgIOEAAYgAQYsQMYgwEYigWYAwCIBgGSBwE3oAenEw&sclient=gws-wiz-serp'
path = 'https://s3-us-west-2.amazonaws.com/courses-images/wp-content/uploads/sites/2483/2018/02/28185128/scholarly_text.jpg'
path = 'https://www.google.com/search?num=10&newwindow=1&sca_esv=019edcabf1e12258&sca_upv=1&q=text&udm=2&fbs=AEQNm0A2upiO_GHeTz6R89sNEjTHXSUfB8x3gweQ77S5CBNH1qkT9yo9p8LiN9Ph7QzgeH30iP61xKdhHkWU-Ava8l8nz9PkGePjGF8Xy5hRILCU_i0WJtOhvdQtLcJTjKD65-XLl9jT4l6cw86Jk_hwDRHHDS8CHxrc85H5U7_MQxCP3a_KWAcZPuVhBaxiP-PKqs6uCp1N4n6aTDbog6fOLFt_TfQI4A&sa=X&ved=2ahUKEwiW87bNo5uIAxU5r4QIHSz0CFoQtKgLegQIExAB&biw=1728&bih=958&dpr=2#imgrc=YrwwLQ0P_highM&imgdii=OzNsao3_-WQiwM'
path = 'https://www.google.com/search?num=10&newwindow=1&sca_esv=019edcabf1e12258&sca_upv=1&q=faces&udm=2&fbs=AEQNm0DfIOrc-JVK7JSIXTRypzh1d0Xgc-Qx-MWluXtsBh8oLqezDGS-GBRoVhGkujFN9L89G8OpQ_4dZf-PrNf_f2qsBtQmtLqsm6yxAHTedliasJBB9FQSsYwZnyPqqc5ldCsnkGqLbFClqXddPoG99GG8FOnDSp-qlZzIkKjMKkP9CcmcXuDZxiWgaR1I6cT030OuyH9GGaehPhQtpHoYO4976gklqg&sa=X&ved=2ahUKEwiIz7qOpZuIAxXen4QIHRucJ2gQtKgLegQIERAB&biw=1728&bih=958&dpr=2'
results = extractor.analyze_images(path, mode='html')

print(json.dumps(results, indent=4))


