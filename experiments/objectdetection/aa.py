#!/usr/bin/env python3
#
# aa.py

import re

def build_datetime_patterns():
    """Build and return a dictionary of datetime regex patterns."""
    datetime_patterns = {
        # Date Patterns
        "numeric_date": r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        "year_first_date": r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
        "textual_date": r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b',
        "ordinal_date": r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*,?\s+\d{4}\b',
        "weekday_date": r'\b(?:Sun|Mon|Tue|Wed|Thu|Fri|Sat)[a-z]*,\s+\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b',

        # Time Patterns
        "twelve_hour_time": r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b',
        "twenty_four_hour_time": r'\b\d{1,2}:\d{2}(:\d{2})?\b',
        "time_with_timezone": r'\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\s*[A-Z]{2,4}\b',
        "iso_time_with_timezone": r'\b\d{2}:\d{2}:\d{2}[-+]\d{2}:\d{2}\b',

        # Combined Date and Time
        "datetime_numeric": r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s+\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\b',
        "datetime_iso": r'\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?(?:Z|[-+]\d{2}:\d{2})?\b',
    }
    return datetime_patterns

def extract_datetime(text):
    """Extract datetime information from the text based on predefined patterns."""
    datetime_patterns = build_datetime_patterns()
    extracted_datetimes = {}

    # Iterate over each pattern and apply it to the text
    for label, pattern in datetime_patterns.items():
        regex = re.compile(pattern)
        matches = regex.findall(text)
        if matches:
            # Store non-empty matches in the dictionary
            extracted_datetimes[label] = matches

    return extracted_datetimes

def extract_pii(text):
    """Extract PII information including datetime patterns from the text."""
    extracted_pii = {}

    # Extract datetime patterns
    datetime_matches = extract_datetime(text)
    
    # Add the datetime matches to the PII extraction results
    extracted_pii.update(datetime_matches)
    
    return extracted_pii

# Example usage
text_to_scan = """
    The event took place on 2021-08-15 at 14:30 PM. 
    Another date could be 01/02/2020 or July 4th, 2023.
    Meeting is scheduled for 02:30 PM PST or 14:00 UTC.
    """
pii_results = extract_pii(text_to_scan)
print(pii_results)

