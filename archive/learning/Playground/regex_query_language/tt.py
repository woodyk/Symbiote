#!/usr/bin/env python3
#
# tt.py

import re

def lucene_to_regex(lucene_query):
    # Convert Lucene single-term queries to regex
    single_term_regex = re.sub(r'(\S+):(\S+)', r'\2', lucene_query)

    # Convert Lucene wildcard queries to regex
    wildcard_regex = single_term_regex.replace('?', '.').replace('*', '.*')

    # Convert quoted exact phrases to regex
    quoted_regex = re.sub(r'"(.*)"', lambda match: match.group(1).replace(' ', '\s'), wildcard_regex)

    # Convert Lucene TO (range) queries to regex
    range_regex = re.sub(r'\[(\d+) TO (\d+)\]',
                         lambda match: "[" + match.group(1) + "-" + match.group(2) + "]",
                         quoted_regex)

    # Convert Lucene AND queries to regex (using lookahead)
    and_operator_regex = re.sub(r'(\S+) AND (\S+)', r'(?=.*\1)(?=.*\2)', range_regex)

    # Convert Lucene OR queries to regex
    bool_operator_regex = and_operator_regex.replace(' OR ', '|')


    result = quoted_regex

    return result

# Test the function
lucene_query = 'field:"term1 OR term2" AND [3 TO 6]'
print(lucene_query)
print(lucene_to_regex(lucene_query))

