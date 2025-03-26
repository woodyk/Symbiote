#!/usr/bin/env python3
#
# structure_extraction.py

import json
import re
import pandas as pd
import yaml
from bs4 import BeautifulSoup
import requests
from io import StringIO

def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup.get_text()

def extract_json_blocks(text):
    json_blocks = []
    start_idx = 0
    while True:
        start_idx = text.find('{', start_idx)
        if start_idx == -1:
            break
        stack = []
        for i in range(start_idx, len(text)):
            if text[i] == '{':
                stack.append('{')
            elif text[i] == '}':
                stack.pop()
                if not stack:
                    try:
                        json_blocks.append(json.loads(text[start_idx:i + 1]))
                    except json.JSONDecodeError:
                        pass
                    start_idx = i + 1
                    break
        else:
            break
    return json_blocks

def extract_yaml_blocks(text):
    yaml_pattern = re.compile(r'(^|\n)---\n(.*?)(\n|\Z)', re.DOTALL)
    yaml_blocks = []
    matches = yaml_pattern.findall(text)
    for match in matches:
        try:
            yaml_blocks.append(yaml.safe_load(match[1]))
        except yaml.YAMLError:
            pass
    return yaml_blocks

def extract_csv_blocks(text):
    csv_pattern = re.compile(r'(?:^|\n)((?:[^,\n]+,)+[^,\n]+(\n|$))', re.DOTALL)
    csv_blocks = []
    matches = csv_pattern.findall(text)
    for match in matches:
        match = match[0].strip()
        if ',' in match and '\n' in match:
            try:
                csv_blocks.append(pd.read_csv(StringIO(match)))
            except pd.errors.ParserError:
                pass
    return csv_blocks

def extract_markdown_blocks(text):
    markdown_pattern = re.compile(r'(?:^|\n)(#.*?)(\n|$)', re.DOTALL)
    return markdown_pattern.findall(text)

def extract_code_blocks(text):
    code_pattern = re.compile(r'(?:^|\n)(def .*?:|class .*?:|function .*?{)(.*?)(?=\n|$)', re.DOTALL)
    return [match[0] + match[1] for match in code_pattern.findall(text)]

def identify_data_structures(text):
    structures = {
        'json': extract_json_blocks(text),
        'yaml': extract_yaml_blocks(text),
        'csv': extract_csv_blocks(text),
        'markdown': extract_markdown_blocks(text),
        'code': extract_code_blocks(text),
        'unstructured': []
    }

    # Remove identified structures from the text and classify the remainder as unstructured
    for key, blocks in structures.items():
        for block in blocks:
            text = text.replace(str(block), '')

    # Remaining text after extraction is considered unstructured
    unstructured_text = text.strip()
    if unstructured_text:
        structures['unstructured'].append(unstructured_text)

    return structures

def process_input(input_data):
    if isinstance(input_data, str):
        if re.match(r'^https?:\/\/', input_data):
            text_data = extract_text_from_url(input_data)
        else:
            text_data = input_data
    elif isinstance(input_data, bytes):
        text_data = input_data.decode('utf-8')
    elif hasattr(input_data, 'read'):
        text_data = input_data.read()
    else:
        raise ValueError("Unsupported input type")

    return identify_data_structures(text_data)

def main(input_data):
    structured_data = process_input(input_data)
    return structured_data

if __name__ == "__main__":
    # Example usage
    input_string = '''
    {
        "name": "John",
        "age": 30
    }

    ---
    name: John
    age: 30

    # This is a markdown header

    def hello_world():
        print("Hello, world!")

    This is some unstructured text.
    '''

    result = main(input_string)
    print(json.dumps(result, indent=2, default=str))

