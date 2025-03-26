#!/usr/bin/env python3
#
# w3m_browse.py

import subprocess
import os
import tempfile

def browse(website_url: str) -> str:
    """
    Opens an interactive terminal web browser using w3m to browse the given URL
    and returns the contents of the last viewed webpage.

    :param website_url: The URL of the website to browse.
    :return: The text content of the last viewed webpage.
    """
    # Create a temporary file to store the last viewed page's URL
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        last_page_url_file = tmp_file.name
    
    try:
        # Open w3m with the provided URL
        subprocess.run(['w3m', website_url])
        
        # After the interactive session ends, extract the last viewed page's content
        last_page_url = ""
        with open(last_page_url_file, 'r') as file:
            last_page_url = file.read().strip()
        
        # If there is no last page URL, return an empty string
        if not last_page_url:
            return ""
        
        # Use w3m to dump the content of the last page
        result = subprocess.run(['w3m', '-dump', last_page_url], capture_output=True, text=True)
        
        if result.returncode == 0:
            return result.stdout
        else:
            raise Exception(f"w3m returned a non-zero exit code when dumping the page: {result.returncode}")
    except FileNotFoundError:
        print("Error: w3m is not installed on your system. Please install it and try again.")
        return ""
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""
    finally:
        # Clean up the temporary file
        if os.path.exists(last_page_url_file):
            os.remove(last_page_url_file)



content = browse('https://google.com')
print(content)
