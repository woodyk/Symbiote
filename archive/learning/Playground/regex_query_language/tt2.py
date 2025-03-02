#!/usr/bin/env python3
#
# tt2.py

import re

def lucene_like_to_regex(query):
    # Replace field:term to term
    single_term_regex = re.sub(r'\S+:(\S+)', r'\1', query)

    # Escape special regex characters, but leave our syntax elements
    escaped = re.sub(r'([\\.+^$[\]{}=!<>|:,\-])', r'\\\1', single_term_regex)

    # Restore escaped spaces (i.e., '\ ' to ' ')
    escaped = re.sub(r'\\ ', ' ', escaped)

    # Process grouping parentheses and quoted strings
    groups_and_quotes = re.sub(r'([()])', r'\\\1', escaped)
    groups_and_quotes = re.sub(r'"(.*?)"', r'\1', groups_and_quotes)

    # Convert wildcard queries to regex
    wildcard_regex = groups_and_quotes.replace('?', '.').replace('*', '.*')

    # Convert TO (range) queries to regex
    range_regex = re.sub(r'\[(\d+)\sTO\s(\d+)\]', lambda m: f"[{m.group(1)}-{m.group(2)}]", wildcard_regex)

    # Convert AND, OR and NOT queries to regex
    # AND operator is a bit tricky. We use positive lookaheads to emulate AND behavior in regex
    and_operator_regex = re.sub(r'(\S+)\sAND\s(\S+)', r'(?=.*\1)(?=.*\2)', range_regex)
    or_operator_regex = and_operator_regex.replace(' OR ', '|')
    not_operator_regex = or_operator_regex.replace(' NOT ', '^(?!.*')

    # Closing parentheses for each NOT operator
    final_regex = not_operator_regex.replace(' ', ').*')

    return final_regex

# Test the function
query = 'wadih mike'
print(lucene_like_to_regex(query))

sentence = "Hello from mike and wadih."
if re.search(lucene_like_to_regex(query), sentence):
    print("OK")




