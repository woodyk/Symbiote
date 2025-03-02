#!/usr/bin/env python3
#
# code_extraction.py

import re
from collections import Counter
from pygments.lexers import guess_lexer, get_all_lexers
from bs4 import BeautifulSoup

def extract_code_blocks(text):
    # Extract Markdown-style code blocks
    code_blocks = re.findall(r'```(.*?)```', text, re.DOTALL)

    # Extract HTML-style code blocks
    soup = BeautifulSoup(text, 'html.parser')
    for tag in soup.find_all(['code', 'pre', 'script']):
        code_blocks.append(tag.get_text())

    return code_blocks

def score_code_block(block):
    # Get a list of all programming language keywords
    keywords = [item[2] for item in get_all_lexers() if item[1]]
    keywords = [item for sublist in keywords for item in sublist]

    # Tokenize the block and count the occurrences of each token
    tokens = re.findall(r'\b\w+\b', block)
    token_counts = Counter(tokens)

    # Score the block by summing the counts of tokens that are keywords
    score = sum(count for token, count in token_counts.items() if token in keywords)
    return score

def identify_language(block):
    try:
        lexer = guess_lexer(block)
        return lexer.name
    except Exception:
        return "Unknown"

def process_text(text):
    code_blocks = extract_code_blocks(text)
    for block in code_blocks:
        score = score_code_block(block)
        language = identify_language(block)
        print(f"Code block:\n{block}\nScore: {score}\nLanguage: {language}\n")

