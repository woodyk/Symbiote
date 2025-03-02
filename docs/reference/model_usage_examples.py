#!/usr/bin/env python3
#
# model_usage_examples.py

from transformers import DetrImageProcessor, DetrForObjectDetection
from transformers import pipeline, Conversation
from diffusers import StableDiffusionImg2ImgPipeline
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas
import requests
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
import requests
import json
import os
import sys
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Suppress warnings

from transformers.utils import logging
logging.set_verbosity_error()

#sys.stderr = open(os.devnull, 'w')

image = "https://images.saymedia-content.com/.image/t_share/MTc0MzUzNzE2MzQ0NzkyNzEw/how-to-clean-a-messy-room-quickly.jpg"
audio = ""

def text_emotion_detection(text_input):
    print("\n--- Textual Emotion Detection ---")

    # Initialize the emotion detection pipeline
    emotion_classifier = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions")

    # Perform emotion detection on the input text
    result = emotion_classifier(text_input)

    # Convert the result to a JSON string for better readability
    result_json = json.dumps(result, indent=4)

    # Print the JSON document
    print(f"Emotion Detection Results:\n{result_json}")

def visual_emotion_detection(image_input):
    print("\n--- Emotion Detection ---")

    # Check if the image_input is a URL or a local file path
    if image_input.startswith("http://") or image_input.startswith("https://"):
        # Load the image from the web URL
        image = Image.open(requests.get(image_input, stream=True).raw).convert("RGB")
    elif os.path.isfile(image_input):
        # Load the image from the local file
        image = Image.open(image_input).convert("RGB")
    else:
        raise ValueError("The provided image_input is neither a valid URL nor a valid file path.")

    # Initialize the emotion detection pipeline
    emotion_classifier = pipeline("image-classification", model="dima806/facial_emotions_image_detection")

    # Perform emotion detection on the image
    result = emotion_classifier(image)

    # Convert the result to a JSON string for better readability
    result_json = json.dumps(result, indent=4)

    # Print the JSON document
    print(f"Emotion Detection Results:\n{result_json}")

    # Display the image alongside detected emotions
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title('Detected Emotions')
    plt.axis('off')

    # Show emotion labels on the image (if applicable)
    for idx, emotion in enumerate(result):
        plt.text(10, 30 + idx * 30, f"{emotion['label']}: {round(emotion['score'] * 100, 2)}%", color="white", fontsize=12, bbox=dict(facecolor='black', alpha=0.7))

    plt.show()

def timeseries_prediction(csv_file=None):
    print("\n--- Time Series Prediction ---")
    
    if csv_file and os.path.isfile(csv_file):
        try:
            # Load the CSV file
            df = pd.read_csv(csv_file)
            
            # Ensure the CSV has the required columns
            if 'timestamp' not in df.columns or 'price' not in df.columns:
                raise ValueError("CSV file must contain 'timestamp' and 'price' columns.")
            
            # Convert the 'timestamp' to a numerical value (e.g., seconds since epoch)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s').map(pd.Timestamp.timestamp)
            
            # Extract features and target variables
            X = df['timestamp'].values.reshape(-1, 1)  # Reshape for sklearn
            y = df['price'].values
            
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return
    else:
        # Generate synthetic data if no CSV file is provided
        np.random.seed(42)
        time_stamps = np.arange(1609459200, 1609459200 + 3600 * 100, 60)  # Unix epoch timestamps every minute for 100 hours
        
        # Generate synthetic stock prices with a simple pattern and noise
        prices = np.sin(np.linspace(0, 20, len(time_stamps))) * 20 + 100 + np.random.normal(0, 2, len(time_stamps))
        
        # Use the generated timestamps and prices
        X = time_stamps.reshape(-1, 1)
        y = prices
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Fit a simple linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions for the test set
    y_pred = model.predict(X_test)
    
    # Prepare the prediction for the next 10 minutes
    last_timestamp = int(X_test[-1][0])  # Convert to standard Python int
    future_timestamps = np.array([last_timestamp + 60 * i for i in range(1, 11)]).reshape(-1, 1)
    future_predictions = model.predict(future_timestamps)
    
    # Convert future_timestamps to standard Python int
    future_timestamps = future_timestamps.flatten().astype(int)
    
    # Print future predictions
    future_predictions_info = {
        "last_known_time": last_timestamp,
        "last_known_price": float(y_test[-1]),  # Convert to float
        "future_timestamps": future_timestamps.tolist(),  # Convert to list of standard ints
        "predicted_prices": future_predictions.tolist()  # Convert to list of floats
    }
    
    future_predictions_json = json.dumps(future_predictions_info, indent=4)
    print(future_predictions_json)
    
    # Plot actual vs predicted prices for the test set
    plt.figure(figsize=(10, 6))
    plt.plot(X_test.flatten(), y_test, label='Actual Prices', color='blue')
    plt.plot(X_test.flatten(), y_pred, label='Predicted Prices', color='orange')
    plt.title('Actual vs Predicted Stock Prices')
    plt.xlabel('Timestamp (Unix Epoch)')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    
    # Plot predictions for the next 10 minutes
    plt.figure(figsize=(10, 6))
    plt.plot(future_timestamps, future_predictions, label='Future Predictions', color='red')
    plt.title('Future Price Predictions for the Next 10 Minutes')
    plt.xlabel('Timestamp (Unix Epoch)')
    plt.ylabel('Predicted Price')
    plt.legend()
    plt.show()

def anomaly_detection():
    print("\n--- Anomaly Detection ---")

    # Generate a dataset with clear anomalies
    normal_data = np.array([
        [1, 1], [2, 2], [2, 1], [1, 2], [3, 3],
        [4, 4], [5, 5], [5, 4], [4, 5], [3, 4]
    ])
    
    # Add clear anomalies
    anomalies = np.array([
        [10, 10],  # Anomaly far from the main cluster
        [-10, -10], # Another anomaly far from the main cluster
        [8, 1],    # Anomaly outside the main cluster but closer
    ])
    
    # Combine normal data with anomalies
    data = np.concatenate([normal_data, anomalies], axis=0)

    # Fit the Isolation Forest model
    model = IsolationForest(contamination=0.2, random_state=42)
    model.fit(data)

    # Predict anomalies
    predictions = model.predict(data)
    
    # -1 indicates an anomaly, 1 indicates a normal point
    is_anomaly = predictions == -1
    
    # Prepare the JSON document with useful details
    anomaly_info = {
        "total_points": len(data),
        "anomalies_detected": int(np.sum(is_anomaly)),
        "anomaly_indices": np.where(is_anomaly)[0].tolist(),
        "anomaly_points": data[is_anomaly].tolist()
    }
    
    # Convert the JSON document to a pretty-printed string
    anomaly_info_json = json.dumps(anomaly_info, indent=4)
    
    # Print the JSON document
    print(anomaly_info_json)
    
    # Visualize the data points
    plt.figure(figsize=(8, 8))
    plt.scatter(data[:, 0], data[:, 1], color='blue', label='Normal Points')
    plt.scatter(data[is_anomaly][:, 0], data[is_anomaly][:, 1], color='red', label='Anomalies')
    plt.title('Anomaly Detection')
    plt.legend()
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

def timeseries_anomaly_detection(csv_file=None):
    print("\n--- Time Series Anomaly Detection ---")
    
    # If a CSV file is provided, load the data from the file
    if csv_file and os.path.isfile(csv_file):
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)
            
            # Ensure the CSV has the required columns
            if 'timestamp' not in df.columns or 'value' not in df.columns:
                raise ValueError("CSV file must contain 'timestamp' and 'value' columns.")
            
            # Convert the 'timestamp' column to numpy array
            time_stamps = df['timestamp'].values
            
            # Extract the 'value' column as the time series data
            values = df['value'].values
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return
    else:
        # Generate synthetic data if no CSV file is provided
        np.random.seed(42)
        time_stamps = np.arange(1609459200, 1609459200 + 3600 * 100, 3600)  # Unix epoch timestamps for 100 hours
        
        # Generate normal time series data
        normal_values = np.sin(np.linspace(0, 20, len(time_stamps))) + np.random.normal(0, 0.1, len(time_stamps))
        
        # Introduce anomalies in the time series data
        anomalies = np.array([5, -3, 4])
        anomaly_indices = [20, 50, 80]  # Points in time where anomalies occur
        values = normal_values.copy()
        values[anomaly_indices] = anomalies

    # Fit the Isolation Forest model on the values (ignoring time for simplicity)
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(values.reshape(-1, 1))

    # Predict anomalies
    predictions = model.predict(values.reshape(-1, 1))
    
    # -1 indicates an anomaly, 1 indicates a normal point
    is_anomaly = predictions == -1
    
    # Prepare the JSON document with useful details
    anomaly_info = {
        "total_points": len(time_stamps),
        "anomalies_detected": int(np.sum(is_anomaly)),
        "anomaly_indices": np.where(is_anomaly)[0].tolist(),
        "anomaly_timestamps": time_stamps[is_anomaly].tolist(),
        "anomaly_values": values[is_anomaly].tolist()
    }
    
    # Convert the JSON document to a pretty-printed string
    anomaly_info_json = json.dumps(anomaly_info, indent=4)
    
    # Print the JSON document
    print(anomaly_info_json)
    
    # Plot the time series data with anomalies highlighted
    plt.figure(figsize=(10, 6))
    plt.plot(time_stamps, values, label='Time Series Data', color='blue')
    plt.scatter(time_stamps[is_anomaly], values[is_anomaly], color='red', label='Anomalies')
    plt.title('Time Series Anomaly Detection')
    plt.xlabel('Timestamp (Unix Epoch)')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

def audio_classification():
    print("\n--- Audio Classification ---")
    print("Input: (Audio file path) Not demonstrated in this script.")
    print("Output: Audio Classification is not directly demonstrated due to lack of audio input in this script.")
    # Example assumes availability of an audio file or use of a suitable model
    # audio_classifier = pipeline("audio-classification", model="superb/wav2vec2-base-superb-ks")
    # result = audio_classifier("path_to_audio_file.wav")

def automatic_speech_recognition():
    print("\n--- Automatic Speech Recognition ---")
    print("Input: (Audio file path) Not demonstrated in this script.")
    print("Output: Automatic Speech Recognition is not directly demonstrated due to lack of audio input in this script.")
    # speech_recognizer = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")
    # result = speech_recognizer("path_to_audio_file.wav")

def conversational():
    print("\n--- Conversational ---")
    input_text = "Hello, how are you?"
    conversation = Conversation(input_text)
    conversational_pipeline = pipeline("conversational", model="microsoft/DialoGPT-medium")
    result = conversational_pipeline([conversation])
    print(f"Input: {input_text}")
    print(f"Output:")
    print(result)

def depth_estimation(image_input):
    print("\n--- Depth Estimation ---")
    
    # Check if the image_input is a URL or a local file path
    if image_input.startswith("http://") or image_input.startswith("https://"):
        # Load the image from the web URL
        image = Image.open(requests.get(image_input, stream=True).raw)
    elif os.path.isfile(image_input):
        # Load the image from the local file
        image = Image.open(image_input)
    else:
        raise ValueError("The provided image_input is neither a valid URL nor a valid file path.")
    
    # Initialize the depth estimation pipeline
    depth_estimator = pipeline("depth-estimation", model="Intel/dpt-large")
    
    # Perform depth estimation
    result = depth_estimator(image)
    
    # Check the structure of the result and extract the depth map
    depth_map = result['predicted_depth'].squeeze()

    # Convert the depth map to a NumPy array if it's a tensor
    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.cpu().numpy()
    
    # Normalize the depth map to the range [0, 1] for visualization
    depth_map_normalized = depth_map / np.max(depth_map)
    
    # Get key points: center and corners of the image
    height, width = depth_map.shape
    key_points = {
        "center": (width // 2, height // 2),
        "top_left": (0, 0),
        "top_right": (width - 1, 0),
        "bottom_left": (0, height - 1),
        "bottom_right": (width - 1, height - 1)
    }
    
    # Extract depth values at key points
    key_points_depth = {name: float(depth_map[y, x]) for name, (x, y) in key_points.items()}
    
    # Prepare the JSON document with additional useful information
    depth_info = {
        "image_size": {"width": width, "height": height},
        "depth_map_shape": depth_map.shape,
        "max_depth_value": float(np.max(depth_map)),
        "min_depth_value": float(np.min(depth_map)),
        "depth_at_key_points": key_points_depth
    }
    
    # Convert the JSON document to a pretty-printed string
    depth_info_json = json.dumps(depth_info, indent=4)
    
    # Print the JSON document
    print(depth_info_json)
    
    # Optionally, display the depth map using matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(depth_map_normalized, cmap='plasma')
    for name, (x, y) in key_points.items():
        plt.text(x, y, name, color="white", fontsize=12, ha='center', va='center', 
                 bbox=dict(facecolor='black', alpha=0.5, lw=0))
    plt.colorbar(label='Depth')
    plt.title('Depth Estimation with Key Points')
    plt.axis('off')
    plt.show()

def document_question_answering():
    print("\n--- Document Question Answering ---")
    #document_qa = pipeline("document-question-answering", model="impira/layoutlmv2")
    #result = document_qa(image)
    #print(f"Output:")
    #print(json.dumps(result, indent=4))

def feature_extraction():
    print("\n--- Feature Extraction ---")
    input_text = "Extract features from this text."
    
    # Initialize the feature extraction pipeline
    feature_extractor = pipeline("feature-extraction", model="bert-base-uncased")
    
    # Perform feature extraction
    result = feature_extractor(input_text)
    
    # Convert the result to a numpy array for easier analysis
    features = np.array(result)
    
    # Prepare the JSON document with relevant details
    feature_info = {
        "input_text": input_text,
        "output_type": str(type(result)),
        "feature_shape": features.shape,
        "summary_statistics": {
            "mean": float(np.mean(features)),
            "std_dev": float(np.std(features))
        },
    }
    
    # Convert the JSON document to a pretty-printed string
    feature_info_json = json.dumps(feature_info, indent=4)
    
    # Print the JSON document
    print(feature_info_json)

def fill_mask():
    print("\n--- Fill Mask ---")
    input_text = "The capital of France is [MASK]."
    fill_masker = pipeline("fill-mask", model="bert-base-uncased")
    result = fill_masker(input_text)
    print(f"Input: {input_text}")
    print(f"Output:")
    print(json.dumps(result, indent=4))

def image_classification():
    print("\n--- Image Classification ---")
    image_classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
    result = image_classifier(image)
    print(f"Output:")
    print(json.dumps(result, indent=4))

def image_feature_extraction(image_input):
    print("\n--- Image Feature Extraction ---")
    
    # Check if the image_input is a URL or a local file path
    if image_input.startswith("http://") or image_input.startswith("https://"):
        # Load the image from the web URL
        image = Image.open(requests.get(image_input, stream=True).raw)
    elif os.path.isfile(image_input):
        # Load the image from the local file
        image = Image.open(image_input)
    else:
        raise ValueError("The provided image_input is neither a valid URL nor a valid file path.")
    
    # Initialize the image feature extraction pipeline
    image_feature_extractor = pipeline("image-feature-extraction", model="google/vit-base-patch16-224")
    
    # Perform image feature extraction
    result = image_feature_extractor(image)
    
    # Convert the result to a numpy array for easier analysis
    features = np.array(result)
    
    # Output a summary of the extracted features
    feature_shape = features.shape
    mean_feature_value = np.mean(features)
    std_feature_value = np.std(features)
    sample_features = features.flatten()[:10]  # Take the first 10 values as a sample

    # Prepare the JSON document with relevant details
    feature_info = {
        "input_image_shape": image.size,
        "output_type": str(type(result)),
        "feature_shape": feature_shape,
        "summary_statistics": {
            "mean": float(mean_feature_value),
            "std_dev": float(std_feature_value)
        },
    }
    
    # Convert the JSON document to a pretty-printed string
    feature_info_json = json.dumps(feature_info, indent=4)
    
    # Print the JSON document
    print(feature_info_json)

def image_segmentation(image_input):
    print("\n--- Image Segmentation ---")

    # Check if the image_input is a URL or a local file path
    if image_input.startswith("http://") or image_input.startswith("https://"):
        # Load the image from the web URL
        image = Image.open(requests.get(image_input, stream=True).raw)
    elif os.path.isfile(image_input):
        # Load the image from the local file
        image = Image.open(image_input)
    else:
        raise ValueError("The provided image_input is neither a valid URL nor a valid file path.")

    # Initialize the image segmentation pipeline
    image_segmenter = pipeline("image-segmentation", model="facebook/detr-resnet-50-panoptic")

    # Perform image segmentation
    result = image_segmenter(image)

    # Prepare the drawing context for the image
    segmented_image = image.copy().convert("RGBA")
    mask_overlay = Image.new("RGBA", segmented_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(mask_overlay)

    # Initialize list for JSON document
    segment_info = []

    # Loop over detected segments and collect information
    for obj in result:
        label = obj['label']
        score = obj['score']
        mask = obj['mask']

        # Convert mask to grayscale image if necessary
        if isinstance(mask, np.ndarray):
            mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
        else:
            mask_image = mask.convert('L')  # Ensure mask is in grayscale mode

        # Apply a random color to the mask
        color = tuple(np.random.randint(0, 256, size=3)) + (150,)  # RGB + Alpha

        # Apply the colored mask to the mask overlay
        mask_overlay.paste(ImageOps.colorize(mask_image, black="black", white=color), (0, 0), mask_image)

        # Append the information to the list for JSON document
        segment_info.append({
            "label": label,
            "score": round(score, 3)
        })

    # Combine the original image with the mask overlay
    segmented_image = Image.alpha_composite(segmented_image, mask_overlay)

    # Convert the JSON document to a pretty-printed string
    segment_info_json = json.dumps(segment_info, indent=4)

    # Print the JSON document
    print(segment_info_json)

    # Display the segmented image
    plt.figure(figsize=(10, 10))
    plt.imshow(segmented_image)
    plt.axis('off')
    plt.title('Segmented Image')
    plt.show()

def image_to_image(image_input, prompt="A fantasy landscape"):
    print("\n--- Image to Image ---")
    
    # Check if the image_input is a URL or a local file path
    if image_input.startswith("http://") or image_input.startswith("https://"):
        # Load the image from the web URL
        image = Image.open(requests.get(image_input, stream=True).raw).convert("RGB")
    elif os.path.isfile(image_input):
        # Load the image from the local file
        image = Image.open(image_input).convert("RGB")
    else:
        raise ValueError("The provided image_input is neither a valid URL nor a valid file path.")
    
    # Initialize the Stable Diffusion Image-to-Image pipeline
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
    pipe = pipe.to("cpu")  # Assuming you have a CUDA-compatible GPU

    # Perform image-to-image transformation
    output_image = pipe(prompt=prompt, init_image=image, strength=0.75, guidance_scale=7.5).images[0]
    
    # Display the input and output images side by side
    plt.figure(figsize=(12, 6))
    
    # Display input image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Input Image')
    plt.axis('off')
    
    # Display output image
    plt.subplot(1, 2, 2)
    plt.imshow(output_image)
    plt.title('Output Image')
    plt.axis('off')
    
    # Show both images
    plt.show()
    
    # Prepare the JSON document with output details
    output_info = {
        "input_image_size": image.size,
        "output_image_size": output_image.size,
        "output_image_mode": output_image.mode,
        "prompt": prompt
    }
    
    # Convert the JSON document to a pretty-printed string
    output_info_json = json.dumps(output_info, indent=4)
    
    # Print the JSON document
    print(output_info_json)

def image_to_text():
    print("\n--- Image to Text ---")
    image_to_text_pipeline = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
    result = image_to_text_pipeline(image)
    print(f"Output:")
    print(json.dumps(result, indent=4))

def crop_detected_objects(image_input):
    print("\n--- Crop Detected Objects ---")
    
    # Check if the image_input is a URL or a local file path
    if image_input.startswith("http://") or image_input.startswith("https://"):
        # Load the image from the web URL
        image = Image.open(requests.get(image_input, stream=True).raw).convert("RGB")
    elif os.path.isfile(image_input):
        # Load the image from the local file
        image = Image.open(image_input).convert("RGB")
    else:
        raise ValueError("The provided image_input is neither a valid URL nor a valid file path.") 

    # Initialize the DETR model and processor
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt")

    # Perform object detection
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process the results to get bounding boxes
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

    # Prepare to draw bounding boxes and labels
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()  # Use default font for labels
    cropped_images = []
    
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        # Get the bounding box coordinates and label name
        box = [round(i, 2) for i in box.tolist()]
        label_name = model.config.id2label[label.item()]
        
        # Draw bounding box and label on the original image
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], box[1] - 10), label_name, fill="red", font=font)

        # Crop the image according to the bounding box
        cropped_img = image.crop(box)
        
        # Draw the label on the cropped image
        draw_cropped = ImageDraw.Draw(cropped_img)
        draw_cropped.text((10, 10), label_name, fill="red", font=font)
        
        cropped_images.append(cropped_img)

    # Display the original image with bounding boxes and labels
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title('Original Image with Bounding Boxes and Labels')
    plt.axis('off')
    plt.show()

    # Display each cropped image with its label
    for idx, cropped_img in enumerate(cropped_images):
        plt.figure(figsize=(4, 4))
        plt.imshow(cropped_img)
        plt.title(f'Cropped Image {idx + 1}: {label_name}')
        plt.axis('off')
        plt.show()

def mask_generation(image_input):
    print("\n--- Mask Generation ---")
    
    # Check if the image_input is a URL or a local file path
    if image_input.startswith("http://") or image_input.startswith("https://"):
        # Load the image from the web URL
        image = Image.open(requests.get(image_input, stream=True).raw)
    elif os.path.isfile(image_input):
        # Load the image from the local file
        image = Image.open(image_input)
    else:
        raise ValueError("The provided image_input is neither a valid URL nor a valid file path.")
    
    # Initialize the mask generation pipeline with the SamModel
    mask_generator = pipeline("mask-generation", model="facebook/sam-vit-base")

    # Perform mask generation
    result = mask_generator(image)

    # Process the 'masks' output
    if 'masks' in result and isinstance(result['masks'], list):
        masks = result['masks']
        segmented_image = image.copy().convert("RGBA")
        mask_overlay = Image.new("RGBA", segmented_image.size, (0, 0, 0, 0))

        # Initialize list for JSON document
        mask_info = []

        # Process each mask
        for idx, mask_array in enumerate(masks):
            # Convert the boolean mask to a grayscale image
            mask_image = Image.fromarray(mask_array.astype(np.uint8) * 255, mode='L')

            # Apply a random color to the mask
            color = tuple(np.random.randint(0, 256, size=3)) + (150,)  # RGB + Alpha
            
            # Apply the colored mask to the mask overlay
            mask_overlay.paste(ImageOps.colorize(mask_image, black="black", white=color), (0, 0), mask_image)

            # Append information to the JSON list
            mask_info.append({
                "mask_index": idx,
                "description": f"Mask {idx + 1} with random color"
            })

        # Combine the original image with the mask overlay
        segmented_image = Image.alpha_composite(segmented_image, mask_overlay)

        # Convert the JSON document to a pretty-printed string
        mask_info_json = json.dumps(mask_info, indent=4)

        # Print the JSON document
        print(mask_info_json)

        # Display the segmented image with the masks overlaid
        plt.figure(figsize=(10, 10))
        plt.imshow(segmented_image)
        plt.axis('off')
        plt.title('Mask Generation')
        plt.show()

    else:
        print("Unexpected output format from mask-generation pipeline:", result)

from transformers import pipeline
import json

def ner():
    print("\n--- Named Entity Recognition (NER) ---")
    
    # Simulate a real-world financial news article
    input_text = ("Apple is looking at buying a U.K. startup for $1 billion. "
                  "The acquisition could happen as soon as next month, according to sources close to the deal. "
                  "This move follows a similar acquisition by Google last year, when they acquired a U.S.-based AI company for $500 million.")
    
    # Initialize the NER pipeline with a more descriptive model
    ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
    
    # Perform NER on the input text
    result = ner_pipeline(input_text)
    
    # Process the result to organize entities by their type
    entities = {}
    for entity in result:
        entity_type = entity['entity']
        entity_text = entity['word']
        
        # Add the entity to the corresponding list in the dictionary
        if entity_type not in entities:
            entities[entity_type] = []
        entities[entity_type].append(entity_text)

    # Print the input text
    print(f"Input: {input_text}")

    # Print the output in a well-formatted JSON structure
    print("Extracted Entities:")
    print(json.dumps(entities, indent=4))

    # Real-world application: Summarize key findings
    print("\n\t--- Summary ---")
    companies = entities.get('B-ORG', []) + entities.get('I-ORG', [])
    locations = entities.get('B-LOC', []) + entities.get('I-LOC', [])
    dates = entities.get('B-DATE', []) + entities.get('I-DATE', [])
    monetary_values = entities.get('B-MISC', []) + entities.get('I-MISC', [])

    if companies:
        print(f"\tCompanies mentioned: {', '.join(companies)}")
    if locations:
        print(f"\tLocations mentioned: {', '.join(locations)}")
    if dates:
        print(f"\tRelevant dates: {', '.join(dates)}")
    if monetary_values:
        print(f"\tMonetary values mentioned: {', '.join(monetary_values)}")

def object_detection():
    print("\n--- Object Detection ---")
    object_detector = pipeline("object-detection", model="facebook/detr-resnet-50")
    result = object_detector(image)
    print(f"Output:")
    print(json.dumps(result, indent=4))

def question_answering():
    print("\n--- Question Answering ---")
    context = "BERT is a transformer model developed by Google."
    question = "Who developed BERT?"
    question_answerer = pipeline("question-answering", model="bert-base-uncased")
    result = question_answerer(question=question, context=context)
    print(f"Input Context: {context}")
    print(f"Input Question: {question}")
    print(f"Output:")
    print(json.dumps(result, indent=4))

def sentiment_analysis():
    print("\n--- Sentiment Analysis ---")
    input_text = "I love this product! It's amazing."
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    result = sentiment_pipeline(input_text)[0]
    print(f"Input: {input_text}")
    print(f"Output:")
    print(json.dumps(result, indent=4))

def summarization():
    print("\n--- Summarization ---")
    input_text = ("Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural "
                  "intelligence displayed by humans and animals.")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    result = summarizer(input_text)
    print(f"Input: {input_text}")
    print(f"Output:")
    print(json.dumps(result, indent=4))

def table_question_answering():
    print("\n--- Table Question Answering ---")
    print("Input: (Table file path) Not demonstrated in this script.")
    print("Output: Table Question Answering is not directly demonstrated due to lack of table input in this script.")
    # table_qa = pipeline("table-question-answering", model="google/tapas-base-finetuned-wtq")
    # result = table_qa(table="path_to_table_file.csv", query="What is the total revenue?")

def text_generation():
    print("\n--- Text Generation ---")
    input_text = "Artificial intelligence will"
    text_generator = pipeline("text-generation", model="gpt2")
    result = text_generator(input_text, max_length=50)
    print(f"Input: {input_text}")
    print(f"Output:")
    print(json.dumps(result, indent=4))

def text_to_audio():
    print("\n--- Text to Audio ---")
    print("Input: (Text) Not demonstrated in this script.")
    print("Output: Text to Audio is not directly demonstrated due to lack of audio output in this script.")
    # text_to_audio_pipeline = pipeline("text-to-audio", model="facebook/mms-lid-voxpopuli")
    # result = text_to_audio_pipeline("Text to audio example.")

def text_to_speech():
    print("\n--- Text to Speech ---")
    print("Input: (Text) Not demonstrated in this script.")
    print("Output: Text to Speech is not directly demonstrated due to lack of audio output in this script.")
    # text_to_speech_pipeline = pipeline("text-to-speech", model="espnet/kan-bayashi_ljspeech_vits")
    # result = text_to_speech_pipeline("Text to speech example.")

def text2text_generation():
    print("\n--- Text to Text Generation ---")
    input_text = "Translate English to German: The house is wonderful."
    text2text_generator = pipeline("text2text-generation", model="t5-small")
    result = text2text_generator(input_text)
    print(f"Input: {input_text}")
    print(f"Output:")
    print(json.dumps(result, indent=4))

def token_classification():
    print("\n--- Token Classification: Legal Document Analysis ---")
    
    # Example text from a legal document or contract
    input_text = "Apple is looking at buying U.K. startup for $1 billion by the end of 2023."
    
    # Use a model fine-tuned for token classification (NER)
    token_classifier = pipeline("token-classification", model="dbmdz/bert-large-cased-finetuned-conll03-english")
    
    # Perform token classification on the input text
    result = token_classifier(input_text)
    
    # Process the results to group tokens by their entity type
    entities = {}
    for token in result:
        entity_type = token['entity']
        entity_word = token['word']
        
        if entity_type not in entities:
            entities[entity_type] = []
        entities[entity_type].append(entity_word)

    # Print the input text
    print(f"Input: {input_text}")
    
    # Print the output in a well-formatted JSON structure
    print("Classified Tokens:")
    print(json.dumps(entities, indent=4))

    # Real-world application: Summarize key findings
    print("\n\t--- Summary ---")
    organizations = entities.get('B-ORG', []) + entities.get('I-ORG', [])
    locations = entities.get('B-LOC', []) + entities.get('I-LOC', [])
    monetary_values = entities.get('B-MISC', []) + entities.get('I-MISC', [])
    dates = entities.get('B-DATE', []) + entities.get('I-DATE', [])

    if organizations:
        print(f"\tOrganizations mentioned: {' '.join(organizations)}")
    if locations:
        print(f"\tLocations mentioned: {' '.join(locations)}")
    if monetary_values:
        print(f"\tMonetary values mentioned: {' '.join(monetary_values)}")
    if dates:
        print(f"\tDates mentioned: {' '.join(dates)}")

def translation():
    print("\n--- Translation ---")
    input_text = "Artificial intelligence is the future of technology."
    translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
    result = translator(input_text)
    print(f"Input: {input_text}")
    print(f"Output:")
    print(json.dumps(result, indent=4))

def video_classification():
    print("\n--- Video Classification ---")
    print("Input: (Video file path) Not demonstrated in this script.")
    print("Output: Video Classification is not directly demonstrated due to lack of video input in this script.")
    # video_classifier = pipeline("video-classification", model="mcg-nju/videomae-base")
    # result = video_classifier("path_to_video_file.mp4")

def visual_question_answering():
    print("\n--- Visual Question Answering ---")
    vqa_pipeline = pipeline("visual-question-answering", model="dandelin/vilt-b32-finetuned-vqa")
    result = vqa_pipeline(image=image, question="Describe all objects in the image.")
    print(f"Output:")
    print(json.dumps(result, indent=4))

def zero_shot_audio_classification():
    print("\n--- Zero-Shot Audio Classification ---")
    print("Input: (Audio file path) Not demonstrated in this script.")
    print("Output: Zero-Shot Audio Classification is not directly demonstrated due to lack of audio input in this script.")
    # zero_shot_audio_classifier = pipeline("zero-shot-audio-classification", model="facebook/wav2vec2-large-960h-lv60-self")
    # result = zero_shot_audio_classifier("path_to_audio_file.wav", candidate_labels=["speech", "music", "noise"])

def zero_shot_classification():
    print("\n--- Zero-Shot Classification ---")
    input_text = "This is a tutorial on NLP."
    zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    result = zero_shot_classifier(input_text, candidate_labels=["education", "politics", "sports"])
    print(f"Input: {input_text}")
    print(f"Output:")
    print(json.dumps(result, indent=4))

def zero_shot_image_classification(image_input):
    print("\n--- Zero-Shot Image Classification ---")
    
    # Check if the image_input is a URL or a local file path
    if image_input.startswith("http://") or image_input.startswith("https://"):
        # Load the image from the web URL
        image = Image.open(requests.get(image_input, stream=True).raw)
    elif os.path.isfile(image_input):
        # Load the image from the local file
        image = Image.open(image_input)
    else:
        raise ValueError("The provided image_input is neither a valid URL nor a valid file path.")
    
    # Initialize the zero-shot image classification pipeline
    zero_shot_image_classifier = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch16")
    
    # Define broad candidate labels
    candidate_labels = [
        "animal", "vehicle", "person", "furniture", "electronics", 
        "building", "outdoor", "food", "clothing", "landscape", 
        "plant", "tool", "toy", "sport", "art", "music", 
        "book", "appliance", "sign", "instrument"
    ]
    
    # Perform zero-shot image classification
    result = zero_shot_image_classifier(image, candidate_labels=candidate_labels)
    
    # Print the output in a well-formatted JSON structure
    print("Output:")
    print(json.dumps(result, indent=4))

def zero_shot_object_detection(image_path):
    print("\n--- Zero-Shot Object Detection ---")
    # Check if the image_path is a URL or a local file path
    if image_path.startswith("http://") or image_path.startswith("https://"):
        # Load the image from the web URL
        image = Image.open(requests.get(image_path, stream=True).raw)
    elif os.path.isfile(image_path):
        # Load the image from the local file
        image = Image.open(image_path)
    else:
        raise ValueError("The provided image_path is neither a valid URL nor a valid file path.")

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

if __name__ == "__main__":
    text_emotion_detection("I am so happy and excited about this new project!")
    visual_emotion_detection(image)
    timeseries_prediction()
    timeseries_anomaly_detection()
    anomaly_detection()
    visual_question_answering()
    depth_estimation(image)
    token_classification()
    mask_generation(image)
    crop_detected_objects(image)
    zero_shot_object_detection(image)
    audio_classification()
    automatic_speech_recognition()
    conversational()
    document_question_answering()
    feature_extraction()
    fill_mask()
    image_classification()
    image_feature_extraction(image)
    image_segmentation(image)
    # image_to_image is a large model for this example
    #image_to_image(image)
    image_to_text()
    ner()
    object_detection()
    question_answering()
    sentiment_analysis()
    summarization()
    table_question_answering()
    text_generation()
    text_to_audio()
    text_to_speech()
    text2text_generation()
    translation()
    video_classification()
    zero_shot_audio_classification()
    zero_shot_classification()
    zero_shot_image_classification(image)

