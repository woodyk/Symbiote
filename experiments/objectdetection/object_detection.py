#!/usr/bin/env python3
#
# object_detection.py

from ultralytics import YOLO

def identify_objects_in_image(image_path):
    # Load the YOLOv5 model
    model = YOLO('models/yolov5l.pt')

    # Run the model on the image
    results = model(image_path)

    # Extract the labels of detected objects
    objects_detected = []
    for result in results:
        for obj in result.boxes:
            objects_detected.append(obj.cls)

    # Get the names of the objects from their class IDs
    object_names = [model.names[int(cls_id)] for cls_id in objects_detected]

    # Remove duplicates by converting to a set
    unique_objects = list(set(object_names))

    return unique_objects

if __name__ == '__main__':
    image_path = '/Users/kato/Pictures/face_test_image.jpg'
    image_path = '/Users/kato/Pictures/Tux.png'
    image_path = '/Users/kato/Pictures/pathfinder1.webp'
    image_path = '/Users/kato/Pictures/gun_armed.jpg'
    detected_objects = identify_objects_in_image(image_path)
    print("Objects detected:", detected_objects)
