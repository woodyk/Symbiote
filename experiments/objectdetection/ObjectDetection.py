#!/usr/bin/env python3
#
# ObjectDetection.py

# object_detector.py

import cv2

class ObjectDetector:
    def __init__(self, model_path=None, config_path=None, classes_path=None):
        """
        Initializes the ObjectDetector with the specified model, configuration, and class labels.

        :param model_path: Path to the pre-trained model file (e.g., 'MobileNetSSD_deploy.caffemodel').
        :param config_path: Path to the model's configuration file (e.g., 'MobileNetSSD_deploy.prototxt').
        :param classes_path: Path to a file containing the class labels (optional).
        """
        if model_path and config_path:
            self.net = cv2.dnn.readNetFromCaffe(config_path, model_path)
        else:
            raise ValueError("You must provide both model_path and config_path.")

        if classes_path:
            with open(classes_path, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
        else:
            # Default classes for MobileNet SSD
            self.classes = ["background", "aeroplane", "bicycle", "bird", "boat",
                            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                            "sofa", "train", "tvmonitor"]

    def detect_objects(self, photo_path):
        """
        Detects objects in the given photo using the pre-trained model.

        :param photo_path: Path to the photo file.
        :return: A list of detected objects and their associated confidence scores.
        """
        detected_objects = []
        try:
            # Load the image
            image = cv2.imread(photo_path)
            if image is None:
                print(f"Could not read image {photo_path}. Skipping...")
                return detected_objects

            (h, w) = image.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
            self.net.setInput(blob)
            detections = self.net.forward()

            # Loop over the detections and add detected objects to the list
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.2:  # Consider detections above a certain confidence threshold
                    idx = int(detections[0, 0, i, 1])
                    label = self.classes[idx]
                    detected_objects.append((label, confidence))
                    print(f"Detected {label} with confidence {confidence:.2f}")

        except Exception as e:
            print(f"An error occurred while detecting objects in the photo {photo_path}: {e}")

        return detected_objects

if __name__ == "__main__":
    # Initialize the ObjectDetector with the required model and config files
    object_detector = ObjectDetector(
        model_path="models/MobileNetSSD_deploy.caffemodel",
        config_path="models/MobileNetSSD_deploy.prototxt",
        classes_path=None  # Using default MobileNet SSD classes
    )

    # Specify the path to the image you want to analyze
    photo_path = "/Users/kato/Pictures/keys.jpg"

    # Detect objects in the image
    detected_objects = object_detector.detect_objects(photo_path)

    # Output the results
    if detected_objects:
        print("Objects detected in the image:")
        for obj, confidence in detected_objects:
            print(f"Object: {obj}, Confidence: {confidence:.2f}")
    else:
        print("No objects detected.")
