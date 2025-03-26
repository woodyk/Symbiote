#!/usr/bin/env python3
#
# deceptive_sentence_patterns.py

import re

def analyze_deception_patterns(sentences):
    # Define patterns based on deceptive language features
    deception_patterns = {
        # Existing Patterns
        "overemphasizing_truthfulness": re.compile(r"\b(honestly|to be honest|believe me|i swear|let me be clear|trust me)\b", re.IGNORECASE),
        "non_contracted_denials": re.compile(r"\b(i did not|he does not|she did not|they did not|it is not)\b", re.IGNORECASE),
        "hedging_statements": re.compile(r"\b(as far as i know|to the best of my knowledge|i believe|maybe|possibly|likely|probably)\b", re.IGNORECASE),
        "avoidance_of_pronouns": re.compile(r"\b(the document was|the item was|the task was)\b", re.IGNORECASE),
        "excessive_detail": re.compile(r"\b(first|then|after that|next)\b", re.IGNORECASE),
        "euphemisms": re.compile(r"\b(take|borrow|misplace|involved|accidentally)\b", re.IGNORECASE),
        "repeated_question": re.compile(r"\b(did i|do you mean)\b", re.IGNORECASE),
        "defensive_responses": re.compile(r"\b(why would you|what do you mean|how could you)\b", re.IGNORECASE),
        "verbal_fillers": re.compile(r"\b(um|uh|you know|like)\b", re.IGNORECASE),

        # New Patterns
        "certainty_words": re.compile(r"\b(always|never|absolutely|definitely|certainly)\b", re.IGNORECASE),
        "lack_of_specificity": re.compile(r"\b(something|stuff|things|someone|somebody|somewhere)\b", re.IGNORECASE),
        "chronological_storytelling": re.compile(r"\b(first|second|third|after that|then)\b", re.IGNORECASE),
    }

    # Analyze each sentence
    results = []
    for sentence in sentences:
        sentence_result = {}
        for pattern_name, pattern in deception_patterns.items():
            if pattern.search(sentence):
                sentence_result[pattern_name] = True
            else:
                sentence_result[pattern_name] = False
        results.append((sentence, sentence_result))

    return results

# Example usage
sentences = [
    "Honestly, I didn't take the money.",
    "To the best of my knowledge, the document was misplaced.",
    "First, I went to the store, then I came back home.",
    "Why would you ask me that?",
    "Um, I believe that's correct.",
    "I always tell the truth, never a lie.",
    "Someone did something wrong."
]

analysis_results = analyze_deception_patterns(sentences)
for sentence, result in analysis_results:
    print(f"Sentence: {sentence}")
    print("Deceptive Patterns Detected:", result)
    print("\n")

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load English model
nlp = spacy.load("en_core_web_sm")

def detect_unexpected_details(text, threshold=0.3):
    # Split the text into sentences
    sentences = [sent.text for sent in nlp(text).sents]

    # Compute TF-IDF matrix for the sentences
    vectorizer = TfidfVectorizer().fit_transform(sentences)
    vectors = vectorizer.toarray()

    # Calculate cosine similarity between sentences
    cosine_matrix = cosine_similarity(vectors)

    # Calculate the relevance of each sentence (mean similarity to others)
    relevance_scores = cosine_matrix.mean(axis=1)

    # Flag sentences that are below a relevance threshold
    unexpected_details = [sentences[i] for i in range(len(sentences)) if relevance_scores[i] < threshold]

    return unexpected_details

# Example usage
text = """
I went to the store to buy groceries. The store was very busy with people shopping for the holidays.
I saw a man wearing a red hat, which reminded me of Santa Claus. The hat had a small logo of a baseball team.
The store had a big sale on canned goods, which I took advantage of.
"""

unexpected_details = detect_unexpected_details(text)
for detail in unexpected_details:
    print(f"Unexpected Detail Detected: {detail}")

