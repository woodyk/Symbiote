#!/usr/bin/env python3
#
# image_extractor.py

import requests
from bs4 import BeautifulSoup
from io import BytesIO
from PIL import Image

def extract_and_open_images(url):
    # Send a request to the URL
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code != 200:
        raise Exception(f"Failed to fetch the webpage: {response.status_code}")
    
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find all image tags
    img_tags = soup.find_all('img')
    
    # Loop through all the image tags and open them
    for img_tag in img_tags:
        img_url = img_tag.get('src')
        
        # Handle relative URLs
        if img_url and not img_url.startswith(('http://', 'https://')):
            img_url = requests.compat.urljoin(url, img_url)
        
        try:
            # Get the image content
            img_response = requests.get(img_url)
            
            # Check the content type to ensure it's an image
            content_type = img_response.headers.get('Content-Type')
            if 'image' in content_type:
                img = Image.open(BytesIO(img_response.content))
                img.show()
            else:
                print(f"Skipped non-image content at {img_url}")
        except Exception as e:
            print(f"Failed to open image: {img_url} - {e}")


# Example usage
url = 'https://www.google.com/search?num=10&newwindow=1&sca_esv=1ebd42d001b6190b&sca_upv=1&q=frogs&udm=2&fbs=AEQNm0A-5VTqs5rweptgTqb6m-Eb3TvVcv4l7eCyod9RtZW9874wvsYjTfpwMQKGHqKPG-IB7j9flyfH28tJSLVuVdcT1tesPpIhTR_8sOQ3FQrQWiVTfWhoIplDgGh5JzUv9F4u3riMB636EHR41DrkNY_uSRk347tLZsVeJqqyuWPTyXrtg-EYkFQYZqw6rWM1khGHS26HrYFGhj2QeE1uCS-2MrLbBw&sa=X&ved=2ahUKEwjiodXZkpuIAxX3g4QIHYP4ChoQtKgLegQIHhAB&biw=1728&bih=958&dpr=2'
extract_and_open_images(url)

