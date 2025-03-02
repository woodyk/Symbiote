#!/usr/bin/env python3
#
# indexer_openai2.py

import os
import sys
import json
import gzip
from collections import Counter
import mimetypes
from datetime import datetime
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk

# Download necessary NLTK data
nltk.download('stopwords')

# Prepare the stop words set and stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def is_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            file.read(1024)
        return True
    except UnicodeDecodeError:
        return False
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return False

def get_file_attributes(file_path):
    try:
        stats = os.stat(file_path)
        return {
            'size': stats.st_size,
            'modified_date': datetime.fromtimestamp(stats.st_mtime).isoformat()
        }
    except Exception as e:
        print(f"Error getting attributes for {file_path}: {e}")
        return {}

def calculate_term_frequencies(content):
    words = content.split()
    filtered_words = [stemmer.stem(word.lower()) for word in words if word.lower() not in stop_words and word.isalpha()]
    frequencies = Counter(filtered_words)
    total_words = sum(frequencies.values())
    return {word: count / total_words for word, count in frequencies.items()}

def index_text_file(file_path):
    print(f"Indexing {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
        term_frequencies = calculate_term_frequencies(content)
        file_attributes = get_file_attributes(file_path)
        return {
            'path': file_path,
            'term_frequencies': term_frequencies,
            'attributes': file_attributes
        }
    except Exception as e:
        print(f"Error indexing {file_path}: {e}")
        return None

def index_directory(directory_path, output_file):
    with gzip.open(output_file, 'wt', encoding='utf-8') as jsonl_file:  # Using GZIP for compression
        for root, _, files in os.walk(directory_path):
            for name in files:
                file_path = os.path.join(root, name)
                if is_text_file(file_path):
                    index_entry = index_text_file(file_path)
                    if index_entry:
                        jsonl_file.write(json.dumps(index_entry) + '\n')

if __name__ == '__main__':
    directory_path = sys.argv[1] 
    output_file = 'bayesian_index.jsonl.gz'  # Updated to indicate GZIP compression
    index_directory(directory_path, output_file)

