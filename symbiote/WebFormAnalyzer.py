#!/usr/bin/env python3
#
# WebFormAnalyzer.py

import logging
import time
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import (
    WebDriverException,
    TimeoutException,
    NoSuchElementException,
)
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.chrome.options import Options as ChromeOptions

# Configure logging
logging.basicConfig(level=logging.INFO)

def init_driver(headless=True, browser='firefox', page_load_timeout=30):
    """
    Initialize Selenium WebDriver with configurable browser and timeout.

    :param headless: Run browser in headless mode (default: True).
    :param browser: Browser to use ('firefox' or 'chrome', default: 'firefox').
    :param page_load_timeout: Page load timeout in seconds (default: 30).
    :return: Initialized WebDriver instance.
    """
    try:
        if browser.lower() == 'firefox':
            options = FirefoxOptions()
            if headless:
                options.add_argument('--headless')
            driver = webdriver.Firefox(options=options)
        elif browser.lower() == 'chrome':
            options = ChromeOptions()
            if headless:
                options.add_argument('--headless')
            driver = webdriver.Chrome(options=options)
        else:
            raise ValueError("Browser must be 'firefox' or 'chrome'")
        driver.set_page_load_timeout(page_load_timeout)
        return driver
    except WebDriverException as e:
        logging.error(f"Error initializing WebDriver: {e}")
        return None


def analyze_site_features(url, headless=True, browser='firefox'):
    """
    Analyze a website for input features like forms and search bars.

    :param url: The URL of the website to analyze.
    :param headless: Run browser in headless mode (default: True).
    :param browser: Browser to use ('firefox' or 'chrome', default: 'firefox').
    :return: A list of dictionaries containing form details.
    """
    driver = init_driver(headless=headless, browser=browser)
    if driver is None:
        logging.error("WebDriver initialization failed.")
        return []

    try:
        logging.info(f"Accessing URL: {url}")
        try:
            driver.get(url)
        except TimeoutException:
            logging.error(f"Timeout while loading {url}")
            driver.quit()
            return []

        # Wait for dynamic content to load
        time.sleep(2)

        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')

        forms = soup.find_all('form')
        features = []

        for form in forms:
            action = form.get('action')
            method = form.get('method', 'GET').upper()
            inputs = []

            for input_tag in form.find_all(['input', 'textarea', 'select']):
                input_type = input_tag.get('type', 'text')
                input_name = input_tag.get('name')

                # For select elements, get the name and options
                if input_tag.name == 'select':
                    options = [option.get('value') or option.text for option in input_tag.find_all('option')]
                    inputs.append({
                        'type': 'select',
                        'name': input_name,
                        'options': options
                    })
                else:
                    inputs.append({
                        'type': input_type,
                        'name': input_name
                    })

            form_details = {
                'action': urljoin(url, action) if action else url,
                'method': method,
                'inputs': inputs
            }
            features.append(form_details)

        driver.quit()
        return features

    except Exception as e:
        logging.error(f"Error analyzing site features: {e}")
        driver.quit()
        return []


# Example usage
if __name__ == "__main__":
    # Replace with the URL you want to analyze
    site_url = 'https://www.exploit-db.com/google-hacking-database'

    features = analyze_site_features(site_url, headless=True, browser='firefox')
    print(f"Site Features of {site_url}:")
    for form in features:
        print(f"Form action: {form['action']}")
        print(f"Method: {form['method']}")
        print("Inputs:")
        for input_field in form['inputs']:
            print(f"  - Name: {input_field.get('name')}, Type: {input_field.get('type')}")
            if input_field.get('type') == 'select':
                print(f"    Options: {input_field.get('options')}")
        print("-" * 40)

