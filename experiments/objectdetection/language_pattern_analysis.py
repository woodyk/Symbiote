#!/usr/bin/env python3
#
# language_pattern_analysis.py

import requests
from bs4 import BeautifulSoup
import nltk
import re
import json
from nltk import pos_tag, word_tokenize
from nltk.tokenize import sent_tokenize

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Predefined patterns
patterns = [
    # Simple Sentence (S + V)
    (r'^(NN|NNS|NNP|NNPS|PRP) (VB|VBD|VBG|VBN|VBP|VBZ)$', "Simple Sentence (S + V)"),
    # Add all other patterns here as defined earlier
    # ...
]

def extract_text_from_url(url):
    """Extracts human-readable text from the given URL."""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract text from <p> tags (common for article content)
    paragraphs = soup.find_all('p')
    text = ' '.join([para.get_text() for para in paragraphs])

    return text

def generate_regex_for_sentence(sentence):
    """Generates a regex pattern for the given sentence and matches it with predefined patterns."""
    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)
    tag_string = " ".join([pos for word, pos in tagged])

    for pattern, description in patterns:
        if re.match(pattern, tag_string):
            return pattern, description

    custom_pattern = r'^' + " ".join(f'({pos})' for pos in tag_string.split()) + r'$'
    return custom_pattern, "OTHER"

def process_text(text):
    """Processes the text and generates a dictionary of unique patterns found in the sentences."""
    sentences = sent_tokenize(text)
    pattern_dict = {}

    for sentence in sentences:
        pattern, description = generate_regex_for_sentence(sentence)
        if pattern in pattern_dict:
            pattern_dict[pattern]['count'] += 1
        else:
            pattern_dict[pattern] = {'description': description, 'count': 1}

    return pattern_dict

def main(url):
    # Step 1: Extract text from URL
    text = extract_text_from_url(url)

    # Step 2: Process the text to create patterns
    pattern_dict = process_text(text)

    # Step 3: Print out the patterns as a JSON list pretty printed with indent=4
    print(json.dumps(pattern_dict, indent=4))

    # Step 4: Summarize the duplicate patterns
    total_duplicates = sum(value['count'] - 1 for value in pattern_dict.values() if value['count'] > 1)
    print(f"Total duplicate patterns found: {total_duplicates}")

# Example usage
if __name__ == "__main__":
    url = "https://en.wikipedia.org/wiki/Florida"  # Replace with the actual URL you want to process
    main(url)

