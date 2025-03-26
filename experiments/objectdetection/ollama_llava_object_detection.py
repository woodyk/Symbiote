#!/usr/bin/env python3
#
# ollama_llava_object_detection.py

from PIL import Image
import json
import io
import os
import sys
import ollama
import base64
from openai import OpenAI
import cv2
from deepface import DeepFace

class FaceAnalyzer:
    def __init__(self, backend='opencv'):
        """
        Initializes the FaceAnalyzer with the specified backend.

        :param backend: The backend to use for face detection and analysis (default: 'opencv').
                        Other options include 'mtcnn', 'dlib', 'ssd', etc.
        """
        self.backend = backend

    def detect_face(self, photo_path):
        """
        Checks if a face is present in the photo using OpenCV's face detector.

        :param photo_path: Path to the photo file.
        :return: True if a face is detected, False otherwise.
        """
        try:
            # Load the image
            img = cv2.imread(photo_path)
            if img is None:
                print(f"Could not read image {photo_path}. Skipping...")
                return False

            # Convert image to grayscale for face detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Load OpenCV's default face detector
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

            # Detect faces in the image
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Return True if at least one face is detected
            return len(faces) > 0
        except Exception as e:
            print(f"An error occurred while detecting faces in the photo {photo_path}: {e}")
            return False

    def analyze_photo(self, photo_path):
        """
        Analyzes the photo at the given path and returns the analysis result.

        :param photo_path: Path to the photo file to be analyzed.
        :return: A list containing the analysis of the photo.
        """
        try:
            # Analyze the image at the given path with the specified backend
            result = DeepFace.analyze(
                img_path=photo_path,
                actions=['emotion', 'age', 'gender', 'race'],
                enforce_detection=True,  # Enforce face detection to reduce misclassifications
                detector_backend=self.backend,  # Using specified backend
                silent=True
            )
            return result

        except Exception as e:
            print(f"An error occurred while analyzing the photo: {e}")
            return None

    def find_matching_faces(self, source_photo_path, directory):
        """
        Crawls through the given directory to find photos with faces that match the one
        in the source photo.

        :param source_photo_path: Path to the source photo file.
        :param directory: Directory to crawl for matching faces.
        :return: A list of file paths that match the face in the source photo.
        """
        matches = []
        try:
            # Analyze the face in the source photo
            source_result = self.analyze_photo(source_photo_path)
            if source_result is None:
                raise Exception("Failed to analyze source photo.")

            # Supported image file extensions
            supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp', '.ppm', '.pgm', '.pbm', '.pnm')

            # Crawl through the directory and compare faces
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith(supported_extensions):
                        target_photo_path = os.path.join(root, file)

                        # Check if a face exists in the photo before running full analysis
                        if not self.detect_face(target_photo_path):
                            print(f"No face detected in {target_photo_path}. Skipping...")
                            continue

                        try:
                            # Verify if the face in the current photo matches the source photo
                            verification_result = DeepFace.verify(
                                img1_path=source_photo_path,
                                img2_path=target_photo_path,
                                detector_backend=self.backend,
                                silent=True
                            )
                            if verification_result["verified"]:
                                matches.append(target_photo_path)
                                print(f"Match found: {target_photo_path}")
                        except Exception as e:
                            print(f"Could not verify {target_photo_path}: {e}")
        except Exception as e:
            print(f"An error occurred during the matching process: {e}")

        return matches


def convert_to_png_base64(image_path):
    # Open the image
    try:
        with Image.open(image_path) as img:
            # Convert to PNG format if not already in JPEG, JPG, or PNG
            if img.format not in ['JPEG', 'JPG', 'PNG']:
                with io.BytesIO() as output:
                    img.save(output, format="PNG")
                    png_data = output.getvalue()
            else:
                with io.BytesIO() as output:
                    img.save(output, format=img.format)
                    png_data = output.getvalue()

        # Encode the PNG image to base64
        png_base64 = base64.b64encode(png_data).decode('utf-8')
        return png_base64
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def detect_objects_llava(image_path, promptnum):
    encoded_image = convert_to_png_base64(image_path)

    if encoded_image is None:
        sys.exit(1)

    # Prepare the prompt for object detection
    prompt = {}
    prompt['data0'] = """You are an advanced image analysis AI designed for investigative profiling. Your task is to analyze a given image and extract detailed information that could be useful in an investigation. You will generate a detailed description containing elements present in the image, focusing on details that are relevant for identifying individuals, objects, and environmental context.

Your analysis should cover the following areas:

1. **People and Identifying Features**:
   - Identify any people in the image, including details such as race, gender, estimated age, and posture (e.g., standing, sitting).
   - Describe any visible clothing, including color, type (e.g., shirt, jacket), and any distinctive patterns or logos.
   - Note any accessories, such as sunglasses, hats, jewelry, watches, or bags, including their color and style.
   - Include specific identifiers such as facial hair (beard, mustache), tattoos, scars, or other distinguishing marks.

2. **Vehicles and Transportation**:
   - Identify any vehicles present in the image, noting details such as the type of vehicle (e.g., car, motorcycle, bicycle), color, make, model, and any visible license plates.
   - Describe the vehicle's location in the image (e.g., foreground, background) and whether it is partially or fully visible.

3. **Objects and Items**:
   - List all significant objects found in the image, such as electronics (e.g., smartphones, laptops), weapons, tools, or other items that could be of interest in an investigation.
   - Include details such as the object's color, brand, and condition (e.g., new, damaged).

4. **Buildings and Environmental Context**:
   - Describe any buildings or structures visible in the image, including their type (e.g., house, office, warehouse), architectural style, and materials.
   - Note any signage, addresses, or visible security features like cameras or fences.
   - If indoors, describe the setting (e.g., living room, office space) and notable furnishings or decorations.

5. **Nature and Surroundings**:
   - Identify natural elements like vegetation (e.g., trees, plants), water bodies (e.g., rivers, lakes), and sky conditions (e.g., clear, cloudy).
   - Describe the overall location, such as urban, rural, beach, or forest, and any environmental conditions (e.g., rain, snow).

6. **Potential Evidence**:
   - Highlight any objects or elements that could be used as evidence in an investigation, such as discarded items, fingerprints, footprints, or other trace evidence.
   - Note any digital devices that may contain data (e.g., phones, laptops) and their status (e.g., powered on, off).

Instructions:
Focus on details that are useful for investigation purposes, such as identifiable features, potential evidence, and environmental context.
"""

    prompt['data1'] = """Create a dynamic YAML document only containing the objects and features found in the image. Add a "description" key to the YAML document that contains a general description of the image.  If a person or persons are detected you must provide a description of each of the person or persons, age, race, emotions, attire, position and other details. If a person or persons are found in the image attempt to identify the following details in the YAML output.
Identity
Gender:
Options: male, female, other
Race:
Options: asian, black, white, hispanic, mixed, other
Age Group:
Options: child, teen, adult, elderly
Approximate Age:
Options: A specific number (e.g., 25)
Emotions
Primary Emotion:
Options: happy, sad, angry, fearful, surprised, neutral
Emotion Confidence:
Options: A percentage (e.g., 85%)
Facial Features
Hair Color:
Options: black, brown, blonde, red, gray, other
Hair Length:
Options: short, medium, long
Hair Style:
Options: straight, curly, wavy, bald, other
Facial Hair:
Beard:
Options: none, stubble, full
Mustache:
Options: none, light, full
Glasses:
Wearing:
Options: true, false
Type:
Options: reading, sunglasses
Color:
Options: black, brown, other
Tattoos or Scars:
Visible:
Options: true, false
Description:
Options: Description of the location and appearance of tattoos or scars (e.g., left arm, dragon tattoo)
Attire
Top:
Type:
Options: t-shirt, shirt, jacket, coat, hoodie, other
Color:
Options: red, blue, black, white, green, other
Pattern:
Options: solid, striped, checked, patterned
Logo:
Present:
Options: true, false
Description:
Options: Description of the logo (e.g., Nike swoosh)
Bottom:
Type:
Options: jeans, shorts, trousers, skirt, other
Color:
Options: blue, black, gray, white, other
Length:
Options: full, above knee, below knee
Footwear:
Type:
Options: sneakers, shoes, boots, sandals, other
Color:
Options: white, black, brown, other
Accessories:
Type:
Options: watch, belt, bracelet, necklace, earrings, other
Color:
Options: gold, silver, black, other
Position:
Options: left wrist, right wrist, neck, ear, other
Position and Posture
Position in Image:
Options: foreground, background, middle
Posture:
Options: standing, sitting, lying down, running, walking
Facing Direction:
Options: forward, backward, left, right
Interaction with Objects:
Options: Description of interactions (e.g., holding phone, holding bag, leaning on wall)
Contextual Details
Location Type:
Options: indoor, outdoor
Specific Location:
Options: street, park, beach, office, restaurant, home, other
Activity:
Options: shopping, exercising, working, eating, other
Accompanied By:
Options: alone, with others, with pet
Other Identifiers
Carrying:
Options: bag, backpack, briefcase, shopping bags, other
Digital Devices:
Type:
Options: smartphone, tablet, laptop, camera, other
Condition:
Options: new, used, damaged
Notable Items:
Options: Description of items (e.g., umbrella, book, coffee cup)
Security Features:
Wearing Mask:
Options: true, false
Wearing Hat:
Options: true, false
Carrying Weapon:
Options: true, false
Visible Identification:
Options: Description of visible ID (e.g., badge, passport, ID card)
"""

    prompt['data2'] = """Create a dynamic JSON document only containing the objects and features found in the image.  If a person or persons are identified in the image then provide details such as emotion, race, age, attire and other identifying features of the person or persons. The key "description" which contains a natural language description of the image will always be provided.  You will respond with the JSON only."""

    prompt['data3'] = """List all the object found in the image in a JSON document.  Only return the JSON document.  Provide as much detail as possible."""

    prompt['data4'] = """Return a JSON document describing all person or persons found in the image and details about their location based off the background.  If no persons are found then return None."""

    prompt['data0'] = """List and describe all objects found in the image. """
    prompt['data1'] = """List and describe in detail any persons found in the image.  Identify age, race, emotion attire and any other related details to the persons found."""
    prompt['data2'] = """List all background elements and attempt to identify a location where the image was taken."""
    prompt['data3'] = """List all color groups found in the image and their percentage used in the image."""

    wprompt = 'data' + str(promptnum)

    # Call the Ollama API with the LLaVA model
    stream = ollama.generate(
        model='llava:13b',
        prompt=prompt[wprompt],
        stream=True,
        options= { 'num_ctx': 8192 },
        images=[encoded_image]
    )

    # Extract the generated text from the response
    detected_objects = str()
    for chunk in stream:
        #value = chunk['message']['content']
        value = chunk['response']
        print(value, end='', flush=True)
        detected_objects += value

    return detected_objects

def process_openai(encoded_image):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    prompt = """Analyze the image and extract all the details about each object found.  Be very descriptive of the objects and their details.  Identify any person or persons in the image and list all physical traits you can about the persons."""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded_image}"
                    }
                }
            ]
        }
    ]

    # Make the API call
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        stream=True,
    )

    # Extract and return the response
    returned = ""
    for chunk in response:
        value = chunk.choices[0].delta.content
        if value is not None:
            print(value, end='', flush=True)
            returned += value 

    return returned 

# Example usage
image_path = "/Users/kato/Pictures/F4AdDnjWEAAVBxr.jpeg"
image_path = "/Users/kato/Pictures/IMG_5265.jpg"
image_path = "/Users/kato/Pictures/IMG_4451.jpg"
image_path = "/Users/kato/Pictures/DALL·E 2024-04-30 17.18.31 - Create an anime-style image of a character holding a giant knife, standing in front of a city surrounded by a giant rope serving as city walls. The kn.webp"
image_path = "/Users/kato/Pictures/IMG_5264.heic"
image_path = "/Users/kato/Pictures/gravnet11-DALL·E 2024-01-18 23.12.48 - Create a visualization showing the outcome of a GravNet simulation with a focus on fractal designs. The image should illustrate whether the simulation.png"
image_path = "/Users/kato/Pictures/pathfinder1.webp"
image_path = "/Users/kato/Pictures/keys2.jpg"
image_path = "/Users/kato/Pictures/face_test_image.jpg"

objects = ""

analyzer = FaceAnalyzer(backend='mtcnn')
analysis_result = analyzer.analyze_photo(image_path)

if analysis_result:
    objects += json.dumps(analysis_result, indent=4)
    print(objects)

objects += process_openai(convert_to_png_base64(image_path)) 

for i in range(4):
    print(f"\nRun {i}\n\n")
    objects += detect_objects_llava(image_path, i)

prompt = f"""Analyze the details provided and  merge them into a single JSON document summarizing the text.  Ensure the JSON document contains representative details to all the following descriptions provided.   Do not provide comments such as // in the output.

DETAILS:
{objects}

Return only the updated JSON document.
"""
stream = ollama.generate(
        model='mistral-nemo',
        stream=True,
        prompt=prompt,
        options= { 'num_ctx': 16000 },
    ) 
print()


for chunk in stream:
    print(chunk['response'], end='', flush=True)
