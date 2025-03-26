#!/usr/bin/env python3
#
# exif_extract.py

import exif
from PIL import Image
import json
import io
from pillow_heif import register_heif_opener, HeifImagePlugin

# Register HEIF/HEIC opener with Pillow
register_heif_opener()

def get_exif_data(image_path):
    try:
        # Open the image file
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()

        # Try to extract EXIF data using the exif library
        try:
            exif_image = exif.Image(image_data)
            if exif_image.has_exif:
                return {k: str(v) for k, v in exif_image.get_all().items()}
        except Exception as e:
            print(f"Error with exif library: {e}")
            # If exif library fails, we'll try with Pillow

        # If exif library didn't work, try with Pillow
        image = Image.open(io.BytesIO(image_data))

        # Check if the image is HEIC/HEIF
        if isinstance(image, HeifImagePlugin.HeifImageFile):
            print("HEIC/HEIF image detected")
            metadata = image.info
            # Extract EXIF data if available
            exif_data = metadata.get('exif', None)
            if exif_data:
                from PIL.ExifTags import TAGS
                exif_dict = {}
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    exif_dict[tag] = str(value)
                metadata['exif'] = exif_dict
            return {k: str(v) if not isinstance(v, dict) else v for k, v in metadata.items()}

        # For other formats, try to get EXIF data
        if hasattr(image, '_getexif'):
            exif_data = image._getexif()
            if exif_data:
                from PIL.ExifTags import TAGS
                exif_dict = {TAGS.get(key, key): str(value) for key, value in exif_data.items()}
                return exif_dict

        # If no EXIF data found, return basic image info
        info = {
            "format": image.format,
            "mode": image.mode,
            "size": image.size
        }
        return info

    except Exception as e:
        return {"error": f"Error processing image: {str(e)}"}

# Function to handle JSON serialization errors
def json_safe_dumps(obj, **kwargs):
    def default(o):
        if isinstance(o, (tuple, list)):
            return [default(i) for i in o]
        return str(o)
    return json.dumps(obj, default=default, **kwargs)

# Example usage
image_path = '/Users/kato/Pictures/IMG_4451.HEIC'
exif_data = get_exif_data(image_path)

# Print the dictionary as a JSON document with an indent of 4
print(json_safe_dumps(exif_data, indent=4, ensure_ascii=False))
