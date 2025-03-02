#!/usr/bin/env python3
#
# search_openai.py

import json
import gzip
import argparse  # Import argparse for command line argument parsing

def load_index(index_file):
    """Load the index from a GZIP-compressed JSONL file."""
    with gzip.open(index_file, 'rt', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

def calculate_match_score(file_index, search_terms):
    """Calculate the match score for a file based on search terms."""
    score = 0
    for term in search_terms:
        score += file_index['term_frequencies'].get(term, 0)
    return score

def search_index(index, search_terms, top_n=10):
    """Search the index for matching files and return the top N matches."""
    search_terms = search_terms.lower().split()
    scored_files = [(file, calculate_match_score(file, search_terms)) for file in index]
    top_matches = sorted(scored_files, key=lambda x: x[1], reverse=True)[:top_n]
    return top_matches

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Search the index for matching files.")
    parser.add_argument("search_query", help="Search terms as a single string.", type=str)
    parser.add_argument("-i", "--index_file", help="Path to the index file.", default="bayesian_index.jsonl.gz", type=str)
    parser.add_argument("-n", "--top_n", help="Number of top matches to return.", default=10, type=int)
    
    # Parse command line arguments
    args = parser.parse_args()

    # Load the index and search
    index = load_index(args.index_file)
    top_matches = search_index(index, args.search_query, args.top_n)
    
    # Print the top matches
    for file, score in top_matches:
        print(f"Match: {file['path']} (Score: {score})")

if __name__ == '__main__':
    main()

