#!/usr/bin/env python3
#
# webcrawler.py

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

def is_valid_url(url):
    """
    Check if the URL is valid.
    """
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)

def web_crawler(url, max_depth=1, depth=0):
    """
    A web crawler function that fetches and returns the text content of a webpage.
    It also follows links within the page up to a specified depth.
    This version uses Selenium to evaluate JavaScript and scroll to the bottom of the page.
    
    :param url: The URL to start crawling from.
    :param max_depth: The maximum depth to crawl to.
    :param depth: The current depth of crawling.
    :return: A list of strings, each containing the text content of a paragraph on the page.
    """
    if depth > max_depth:
        return []
    
    if not is_valid_url(url):
        print(f"Invalid URL: {url}")
        return []
    
    try:
        # Set up the WebDriver (e.g., for Chrome)
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        
        # Scroll to the bottom of the page
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'p')))
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
        
        # Extract the content of the page
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        text_content = []
        for paragraph in soup.find_all('p'):
            text_content.append(paragraph.get_text(strip=True))
        
        # Follow links within the page
        for link in soup.find_all('a'):
            href = link.get('href')
            if href and not href.startswith('#'):
                full_url = urljoin(url, href)
                text_content.extend(web_crawler(full_url, max_depth, depth + 1))
        
        # Close the WebDriver
        driver.quit()
        
        return text_content
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return []

# Example usage
text = web_crawler('https://openai.com', max_depth=0)
print(text)
