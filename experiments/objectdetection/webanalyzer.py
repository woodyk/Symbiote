#!/usr/bin/env python3
#
# webanalyzer.py

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
    Analyze a website for input features, including GET/POST methods, form validation,
    dynamic JavaScript interactions, file uploads, and AJAX requests.
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
            enctype = form.get('enctype', 'application/x-www-form-urlencoded')  # To handle multipart forms
            inputs = []

            for input_tag in form.find_all(['input', 'textarea', 'select', 'button']):
                input_type = input_tag.get('type', 'text')
                input_name = input_tag.get('name')
                input_value = input_tag.get('value', '')

                # Handle select elements
                if input_tag.name == 'select':
                    options = []
                    selected_value = None
                    for option in input_tag.find_all('option'):
                        option_value = option.get('value') or option.text
                        if 'selected' in option.attrs:
                            selected_value = option_value
                        options.append(option_value)
                    inputs.append({
                        'type': 'select',
                        'name': input_name,
                        'options': options,
                        'selected': selected_value
                    })

                # Handle radio buttons and checkboxes
                elif input_type in ['radio', 'checkbox']:
                    is_checked = 'checked' in input_tag.attrs
                    inputs.append({
                        'type': input_type,
                        'name': input_name,
                        'value': input_value,
                        'checked': is_checked
                    })

                # Handle hidden inputs
                elif input_type == 'hidden':
                    inputs.append({
                        'type': input_type,
                        'name': input_name,
                        'value': input_value
                    })

                # Handle buttons and submits
                elif input_tag.name == 'button' or input_type in ['submit', 'button']:
                    inputs.append({
                        'type': input_type,
                        'name': input_name,
                        'value': input_value
                    })

                # Handle other input types
                else:
                    # Capture additional validation attributes
                    input_pattern = input_tag.get('pattern', None)
                    input_maxlength = input_tag.get('maxlength', None)
                    input_required = 'required' in input_tag.attrs

                    inputs.append({
                        'type': input_type,
                        'name': input_name,
                        'value': input_value,
                        'pattern': input_pattern,
                        'maxlength': input_maxlength,
                        'required': input_required
                    })

            form_details = {
                'action': urljoin(url, action) if action else url,
                'method': method,
                'enctype': enctype,
                'inputs': inputs
            }
            features.append(form_details)

        # Check for non-form inputs in anchor tags (e.g., query strings in links)
        anchors = soup.find_all('a', href=True)
        for anchor in anchors:
            href = anchor.get('href')
            if '?' in href:  # Check for query strings
                parsed_url = urljoin(url, href)
                inputs.append({
                    'type': 'link',
                    'url': parsed_url,
                    'parameters': href.split('?')[1]  # Extract parameters if any
                })

        # Simulate interaction and detect JavaScript events and dynamic inputs
        try:
            # Simulate form interaction (optional for dynamic content)
            driver.execute_script("document.querySelectorAll('form').forEach(form => form.submit())")
            time.sleep(1)  # Wait for the form submission
        except NoSuchElementException:
            logging.warning(f"No forms found to submit on {url}")

        # Detect if the site uses jQuery for AJAX requests
        is_using_jquery = driver.execute_script("return typeof jQuery != 'undefined';")
        if is_using_jquery:
            logging.info("Site is using jQuery for AJAX requests.")

        # Capture any AJAX requests during the form interaction
        requests = driver.execute_script("return window.performance.getEntriesByType('resource');")
        for request in requests:
            if request['initiatorType'] == 'xmlhttprequest':
                features.append({
                    'type': 'ajax_request',
                    'url': request['name']
                })

        driver.quit()
        return features

    except Exception as e:
        logging.error(f"Error analyzing site features: {e}")
        driver.quit()
        return []


# Example usage
if __name__ == "__main__":
    # Replace with the URL you want to analyze
    site_url = 'https://en.wikipedia.org/'
    site_url = 'https://google.com'

    features = analyze_site_features(site_url, headless=True, browser='firefox')
    print(f"Site Features of {site_url}:")
    for form in features:
        # Check if this is an AJAX request or regular form data
        if 'type' in form and form['type'] == 'ajax_request':
            print(f"AJAX Request URL: {form['url']}")
        else:
            print(f"Form action: {form['action']}")
            print(f"Method: {form['method']}")
            print(f"Enctype: {form.get('enctype')}")
            print("Inputs:")
            for input_field in form['inputs']:
                print(f"  - Name: {input_field.get('name')}, Type: {input_field.get('type')}")
                if input_field.get('type') == 'select':
                    print(f"    Options: {input_field.get('options')}, Selected: {input_field.get('selected')}")
                elif input_field.get('type') in ['radio', 'checkbox']:
                    print(f"    Value: {input_field.get('value')}, Checked: {input_field.get('checked')}")
                else:
                    print(f"    Value: {input_field.get('value')}, Pattern: {input_field.get('pattern')}, "
                          f"Maxlength: {input_field.get('maxlength')}, Required: {input_field.get('required')}")
        print("-" * 40)

