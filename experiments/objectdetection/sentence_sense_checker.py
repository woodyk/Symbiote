#!/usr/bin/env python3
#
# sentence_sense_checker.py

import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from nltk.tokenize import sent_tokenize
from nltk import pos_tag, word_tokenize
import nltk
import language_tool_python

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Initialize the LanguageTool object
tool = language_tool_python.LanguageTool('en-US')

def extract_text_from_url(url):
    """Extracts human-readable text from the given URL."""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract text from <p> tags (common for article content)
    paragraphs = soup.find_all('p')
    text = ' '.join([para.get_text() for para in paragraphs])

    return text

def sentence_validity_score(sentence):
    """Returns a score between 0 and 1 for how grammatically correct the sentence is."""
    # Check for grammatical errors using LanguageTool
    matches = tool.check(sentence)
    grammar_errors = len(matches)

    # Basic grammar score: reduce score based on the number of errors
    grammar_score = max(0, 1 - (grammar_errors / (grammar_errors + 1)))

    # Check for semantic coherence using TextBlob
    blob = TextBlob(sentence)
    coherence_score = 1 - blob.sentiment.subjectivity  # Subjectivity closer to 0 indicates better coherence

    # Combine grammar and coherence scores (weighted equally)
    final_score = (grammar_score + coherence_score) / 2
    return max(0, min(final_score, 1))  # Ensure the score is between 0 and 1

def process_text(text):
    """Processes the text and evaluates the grammatical correctness of each sentence."""
    sentences = sent_tokenize(text)
    sentence_scores = {}

    for sentence in sentences:
        score = sentence_validity_score(sentence)
        sentence_scores[sentence] = score

    return sentence_scores

def main(url):
    # Step 1: Extract text from URL
    text = extract_text_from_url(url)

    # Step 2: Process the text to evaluate sentences
    sentence_scores = process_text(text)

    # Step 3: Print out the sentence scores
    for sentence, score in sentence_scores.items():
        print(f"Sentence: {sentence}\nScore: {score:.2f}\n")

# Example usage
if __name__ == "__main__":
    url = "https://en.wikipedia.org/wiki/Florida"  # Replace with the actual URL you want to process
    main(url)

