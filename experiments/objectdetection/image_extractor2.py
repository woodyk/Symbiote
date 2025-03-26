#!/usr/bin/env python3
#
# image_extractor2.py

import requests
from bs4 import BeautifulSoup
from io import BytesIO
from PIL import Image
import webbrowser
import tempfile
import os
import time

def extract_and_display_images(url, mode='merged', images_per_row=3):
    # Send a request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code != 200:
        raise Exception(f"Failed to fetch the webpage: {response.status_code}")

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all image tags
    img_tags = soup.find_all('img')
    images = []
    img_sources = []

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
                images.append(img)
                img_sources.append(img_url)
            else:
                print(f"Skipped non-image content at {img_url}")
        except Exception as e:
            print(f"Failed to open image: {img_url} - {e}")

    if not images:
        print("No images found.")
        return

    if mode == 'merged':
        # Determine the size of the resulting image
        max_width = max(img.width for img in images)
        max_height = max(img.height for img in images)
        total_width = images_per_row * max_width
        total_height = (len(images) + images_per_row - 1) // images_per_row * max_height

        # Create a new blank image
        combined_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))

        # Paste each image into the combined image
        for index, img in enumerate(images):
            x = (index % images_per_row) * max_width
            y = (index // images_per_row) * max_height
            combined_image.paste(img, (x, y))

        # Display the combined image
        combined_image.show()

    elif mode == 'individual':
        # Show each image individually
        for img in images:
            img.show()

    elif mode == 'html':
        # Create a dark-themed HTML page with a table of images
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
        </style>
        </head>
        <body>
        <h1>Extracted Images</h1>
        <div class="container">
        """

        for img_src in img_sources:
            html_content += f"""
            <div class="card">
                <a href="{img_src}" target="_blank">
                    <img src="{img_src}" alt="Image" />
                </a>
            </div>
            """

        html_content += """
        </div>
        </body>
        </html>
        """

        # Write the HTML content to a temporary file and open it in the default browser
        try:
            with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html', dir='/tmp') as f:
                f.write(html_content)
                temp_file_path = f.name

            webbrowser.open(f'file://{os.path.realpath(temp_file_path)}')

            time.sleep(2)

        finally:
            # Remove the temporary file after use
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

# Example usage:
url = 'https://www.google.com/search?num=10&newwindow=1&sca_esv=1ebd42d001b6190b&sca_upv=1&q=frogs&udm=2&fbs=AEQNm0A-5VTqs5rweptgTqb6m-Eb3TvVcv4l7eCyod9RtZW9874wvsYjTfpwMQKGHqKPG-IB7j9flyfH28tJSLVuVdcT1tesPpIhTR_8sOQ3FQrQWiVTfWhoIplDgGh5JzUv9F4u3riMB636EHR41DrkNY_uSRk347tLZsVeJqqyuWPTyXrtg-EYkFQYZqw6rWM1khGHS26HrYFGhj2QeE1uCS-2MrLbBw&sa=X&ved=2ahUKEwjiodXZkpuIAxX3g4QIHYP4ChoQtKgLegQIHhAB&biw=1728&bih=958&dpr=2'
url = 'https://trinket.io/python'
extract_and_display_images(url, mode='merged')
#extract_and_display_images(url, mode='individual')
extract_and_display_images(url, mode='html')

