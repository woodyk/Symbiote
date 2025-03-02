#!/usr/bin/env python3
#
# combined.py

import os
import sys
import json
import gzip
from collections import Counter
from datetime import datetime
import argparse
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# Ensure necessary NLTK data is available
nltk.download('stopwords', quiet=True)

class BayesianFileIndexer:
    def __init__(self, index_file='bayesian_index.jsonl.gz'):
        self.index_file = index_file
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

    def is_text_file(self, file_path):
        try:
            # Simple heuristic to filter non-text files
            return file_path.endswith('.txt')
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return False

    def calculate_term_frequencies(self, content):
        words = content.split()
        filtered_words = [self.stemmer.stem(word.lower()) for word in words if word.lower() not in self.stop_words and word.isalpha()]
        frequencies = Counter(filtered_words)
        total_words = sum(frequencies.values())
        return {word: count / total_words for word, count in frequencies.items()}

    def index_text_file(self, file_path):
        print(f"Indexing {file_path}...")
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
            term_frequencies = self.calculate_term_frequencies(content)
            file_attributes = {
                'size': os.path.getsize(file_path),
                'modified_date': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
            }
            return {
                'path': file_path,
                'term_frequencies': term_frequencies,
                'attributes': file_attributes
            }
        except Exception as e:
            print(f"Error indexing {file_path}: {e}")
            return None

    def index_directory(self, directory_path):
        indexed_files = []
        for root, _, files in os.walk(directory_path):
            for name in files:
                file_path = os.path.join(root, name)
                if self.is_text_file(file_path):
                    index_entry = self.index_text_file(file_path)
                    if index_entry:
                        indexed_files.append(index_entry)
        with gzip.open(self.index_file, 'wt', encoding='utf-8') as jsonl_file:
            for entry in indexed_files:
                jsonl_file.write(json.dumps(entry) + '\n')
        print(f"Indexing complete. Indexed {len(indexed_files)} files.")

    def load_index(self):
        with gzip.open(self.index_file, 'rt', encoding='utf-8') as file:
            return [json.loads(line) for line in file]

    def calculate_match_score(self, file_index, search_terms):
        score = 0
        for term in search_terms:
            score += file_index['term_frequencies'].get(term, 0)
        return score

    def search_index(self, search_terms, top_n=10):
        index = self.load_index()
        search_terms = [self.stemmer.stem(term.lower()) for term in search_terms.split() if term.lower() not in self.stop_words]
        scored_files = [(file, self.calculate_match_score(file, search_terms)) for file in index]

        # Filter results to only include those with a score greater than zero
        filtered_scored_files = filter(lambda x: x[1] > 0, scored_files)

        # Sort the filtered results based on match score, from highest to lowest
        top_matches = sorted(filtered_scored_files, key=lambda x: x[1], reverse=True)[:top_n]

        if top_matches:
            for file, score in top_matches:
                print(f"Match: {file['path']} (Score: {score})")
        else:
            print("No matches found with a score greater than zero.")

def main():
    parser = argparse.ArgumentParser(description="Index or search text files using Bayesian techniques.")
    parser.add_argument("directory_or_query", help="Directory to index or search query, depending on the mode.", type=str)
    parser.add_argument("-m", "--mode", help="Operation mode: 'index' or 'search'.", default="search", type=str)
    parser.add_argument("-n", "--top_n", help="Number of top matches to return for search mode.", default=10, type=int)

    args = parser.parse_args()

    indexer = BayesianFileIndexer()
    if args.mode == "index":
        indexer.index_directory(args.directory_or_query)
    elif args.mode == "search":
        indexer.search_index(args.directory_or_query, args.top_n)
    else:
        print("Invalid mode. Use 'index' or 'search'.")

if __name__ == '__main__':
    main()

