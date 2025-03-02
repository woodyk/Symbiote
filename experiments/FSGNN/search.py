#!/usr/bin/env python3
#
# search.py

import json
import gzip
import argparse
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk

# Ensure necessary NLTK data is available
nltk.download('stopwords', quiet=True)

class BayesianSearcher:
    def __init__(self, index_file='bayesian_index.jsonl.gz'):
        self.index_file = index_file
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

    def load_index(self):
        """Load the indexed data from a GZIP-compressed JSONL file."""
        with gzip.open(self.index_file, 'rt', encoding='utf-8') as file:
            # Ensure all term frequencies keys are treated case insensitively
            return [self.process_index_entry(json.loads(line)) for line in file]

    def process_index_entry(self, entry):
        """ Process each index entry to ensure term frequencies are case-insensitive """
        entry['term_frequencies'] = {self.stemmer.stem(key.lower()): value for key, value in entry['term_frequencies'].items()}
        return entry

    def calculate_match_score(self, file_index, search_terms):
        """Calculate how well the file matches the search terms."""
        score = 0
        for term in search_terms:
            # Access term frequencies in a case-insensitive manner
            score += file_index['term_frequencies'].get(term, 0)
        return score

    def search_index(self, search_terms, top_n=10):
        """Search the index and return the top N matches."""
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
    parser = argparse.ArgumentParser(description="Search indexed text files using Bayesian techniques.")
    parser.add_argument("search_query", help="Search terms as a single string.", type=str)
    parser.add_argument("-i", "--index_file", help="Path to the index file.", default="bayesian_index.jsonl.gz", type=str)
    parser.add_argument("-n", "--top_n", help="Number of top matches to return.", default=10, type=int)

    args = parser.parse_args()

    searcher = BayesianSearcher(index_file=args.index_file)
    searcher.search_index(args.search_query, args.top_n)

if __name__ == '__main__':
    main()

