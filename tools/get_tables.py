#!/usr/bin/env python3
#
# get_tables.py

import requests
from bs4 import BeautifulSoup
from io import StringIO
import pandas as pd
import json
import re


def clean_value(value):
    """
    Clean a single value by removing non-printable characters, unicode escapes, and converting types.
    Additionally, strips leading/trailing whitespace for string values.
    """
    if pd.isna(value):  # Handle NaN
        return None
    if isinstance(value, str):
        value = value.strip()  # Remove surrounding whitespace
        value = re.sub(r'[^\x20-\x7E]', '', value)  # Remove non-printable characters
        if value.isdigit():
            return int(value)
        try:
            return float(value) if '.' in value else int(value)
        except ValueError:
            pass
        return value  # Return clean string
    elif isinstance(value, (int, float)):  # Return numbers as-is
        return value
    return str(value).strip()  # Convert other types (e.g., objects) to strings and strip them


def clean_column_name(column_name):
    """
    Clean column names by converting to lowercase and replacing spaces with underscores.
    """
    return re.sub(r'\s+', '_', str(column_name).strip().lower())


def extract_tables_to_json(url):
    """
    Extract and clean tables from a webpage and convert them into JSON objects.
    """
    try:
        # Step 1: Fetch the webpage content
        response = requests.get(url)
        response.raise_for_status()
        html_content = response.text

        # Step 2: Parse the HTML
        soup = BeautifulSoup(html_content, 'html.parser')

        # Step 3: Find all tables
        tables = soup.find_all('table')
        if not tables:
            return {"error": "No tables found on the webpage."}

        json_tables = {}

        # Step 4: Process each table
        for i, table in enumerate(tables, start=1):
            # Convert the table to a DataFrame
            table_html = StringIO(str(table))
            try:
                df = pd.read_html(table_html)[0]
            except ValueError:
                continue  # Skip if the table cannot be parsed

            # Clean column names
            df.columns = [clean_column_name(col) for col in df.columns]

            # Clean the data using map across columns
            for column in df.columns:
                df[column] = df[column].map(clean_value)

            # Convert the DataFrame to a JSON object
            json_table = df.to_dict(orient='records')
            json_tables[f"table_{i}"] = json_table

        return json_tables

    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

def main():
    # Example usage
    url = "https://en.wikipedia.org/wiki/International_Bank_Account_Number"  # Replace with the desired URL
    url = "https://en.wikipedia.org/wiki/International_Bank_Account_Number"  # Replace with the desired URL
    json_result = extract_tables_to_json(url)
    print(type(json_result))

    #print(json.dumps(json_result, indent=4))
    # Save to a file or display the result
    #with open("tables.json", "w") as f:
    #    json.dump(json_result, f, indent=4)

if __name__ == "__main__":
    main()

