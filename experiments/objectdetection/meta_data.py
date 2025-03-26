#!/usr/bin/env python3
#
# meta_data.py

import os
import json
import magic
import exifread
from PyPDF2 import PdfReader
from mutagen import File as MutagenFile
from PIL import Image
from PIL.ExifTags import TAGS
import docx

def extract_metadata(file_path):
    metadata = {}

    # Basic file stats
    try:
        stat_info = os.stat(file_path)
        metadata['stat'] = {
            'size': stat_info.st_size,
            'created': stat_info.st_ctime,
            'modified': stat_info.st_mtime,
            'accessed': stat_info.st_atime,
        }
    except Exception as e:
        metadata['stat'] = f"Error extracting stat data: {str(e)}"

    # File type detection
    try:
        mime = magic.Magic(mime=True)
        file_type = mime.from_file(file_path)
        metadata['file_type'] = file_type
    except Exception as e:
        metadata['file_type'] = f"Error detecting file type: {str(e)}"

    # EXIF metadata for images
    if metadata['file_type'].startswith('image'):
        try:
            with open(file_path, 'rb') as f:
                tags = exifread.process_file(f)
                metadata['exif'] = {tag: str(value) for tag, value in tags.items()}
        except Exception as e:
            metadata['exif'] = f"Error extracting EXIF data: {str(e)}"

        # Additional image metadata using Pillow
        try:
            image = Image.open(file_path)
            exif_data = image._getexif()
            if exif_data:
                image_metadata = {TAGS.get(tag): value for tag, value in exif_data.items() if tag in TAGS}
                metadata['image_metadata'] = image_metadata
        except Exception as e:
            metadata['image_metadata'] = f"Error extracting image data with Pillow: {str(e)}"

    # PDF metadata using PdfReader
    if metadata['file_type'] == 'application/pdf':
        try:
            reader = PdfReader(file_path)
            pdf_metadata = reader.metadata
            metadata['pdf_metadata'] = {key: str(value) for key, value in pdf_metadata.items()}
        except Exception as e:
            metadata['pdf_metadata'] = f"Error extracting PDF metadata: {str(e)}"

    # Audio file metadata
    if metadata['file_type'].startswith('audio'):
        try:
            audio = MutagenFile(file_path)
            audio_metadata = {k: str(v) for k, v in audio.items()}
            metadata['audio_metadata'] = audio_metadata
        except Exception as e:
            metadata['audio_metadata'] = f"Error extracting audio metadata: {str(e)}"

    # Word document metadata
    if metadata['file_type'] == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        try:
            doc = docx.Document(file_path)
            core_props = doc.core_properties
            metadata['word_metadata'] = {
                'title': core_props.title,
                'author': core_props.author,
                'last_modified_by': core_props.last_modified_by,
                'created': core_props.created,
                'modified': core_props.modified,
                'subject': core_props.subject,
                'keywords': core_props.keywords,
                'category': core_props.category,
                'comments': core_props.comments,
                'content_status': core_props.content_status,
                'version': core_props.version
            }
        except Exception as e:
            metadata['word_metadata'] = f"Error extracting Word document metadata: {str(e)}"

    return metadata

# Example usage
file_path = '/Users/kato/Documents/Wadih_Resume.pdf'  # Replace with your file path
file_path = '/Users/kato/Documents/Mathematics of Love and Intelligence.docx'
metadata = extract_metadata(file_path)
print(metadata)

