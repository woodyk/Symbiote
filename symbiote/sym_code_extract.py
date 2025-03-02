#!/usr/bin/env python3
#
# sym_code_extract.py
"""
=======================================================
PROJECT PROMPT SUMMARY
=======================================================

Purpose and Context:
---------------------
This project centers on identifying and extracting code from various inputs such as files, strings, and websites.
The goal is to provide a modular and extensible framework that can:
  - Determine whether input contains code or is purely code.
  - Extract visible code snippets from web pages.
  - Handle plain text files, source code repositories, and HTML-based content.

Structure and Setup:
---------------------
1. **Core Classes and Methods**:
    - `CodeIdentifier`: Central class responsible for:
      - Identifying whether an input is a code file.
      - Extracting code blocks from text, files, or web pages.
      - Classifying input using heuristics and tools like Pygments.
    - Methods include:
      - `analyze_file`: Determines if a file contains or is purely code and extracts code.
      - `analyze_url`: Handles both plain text and HTML-based websites, detecting visible or full code.
      - `analyze_string`: Processes raw text to identify and extract embedded code blocks.

2. **Tools and Libraries**:
    - **Pygments**:
      - Used to guess file types and determine if content qualifies as source code.
      - Ensures compatibility with diverse programming languages.
    - **BeautifulSoup (bs4)**:
      - Extracts visible human-readable content from HTML pages.
      - Strips tags to isolate readable code snippets.
    - **Regex**:
      - Used extensively for detecting code patterns in text or mixed-content files.
    - **Requests**:
      - Handles web requests to fetch content from URLs.

Methodologies and Conventions:
-------------------------------
- **Modular Approach**:
  - Each analysis (file, string, URL) is encapsulated in its own method for maintainability.
- **Default Behaviors**:
  - URL analysis defaults to "visible" mode, extracting only human-readable snippets unless otherwise specified.
- **Content-Type Awareness**:
  - Dynamically adjusts behavior based on MIME types (e.g., `text/plain` for raw files, `text/html` for web pages).
- **Edge Case Handling**:
  - Ensures robustness when encountering non-code files, unsupported formats, or unreachable URLs.

Functionality Implemented:
---------------------------
- Unified interface to process strings, files, or URLs and classify their content.
- Extraction of code blocks from fenced Markdown, HTML `<pre>/<code>` tags, or raw content.
- URL support with options to extract visible code or the full source of a website.
- Command-line interface for seamless interaction with different input types.

Patterns and Preferences:
--------------------------
- **Clear Separation of Concerns**:
  - Detection (`is_code_file` and `has_code`) is decoupled from extraction.
- **Graceful Handling**:
  - Returns meaningful outputs and error messages for invalid or unsupported inputs.
- **Comprehensive Analysis**:
  - Handles both human-readable snippets and full code extractions based on mode.

Extension and Maintenance Guidelines:
--------------------------------------
1. **Adding Support for New Formats**:
    - Extend `extract_code_blocks` or introduce new methods for specific formats.
    - Example: Add patterns for LaTeX or other specialized document types.
2. **Improving Heuristics**:
    - Enhance `is_code_file` with additional logic for ambiguous inputs.
3. **Integrations**:
    - Add options to directly integrate with version control systems or cloud-based repositories.
4. **Command-Line Usability**:
    - Expand CLI options to support batch processing, logging, or exporting results.
5. **Error Handling**:
    - Continue to refine edge-case detection (e.g., malformed HTML or inaccessible URLs).

Alignment with Personal Style:
-------------------------------
- Emphasis on modularity, clarity, and extensibility.
- Robust handling of diverse input types.
- Intuitive defaults with flexibility for advanced use cases.
- Focused on practical functionality and integration into larger workflows.

=======================================================
"""


import argparse
import os
import re
import requests
from bs4 import BeautifulSoup
from pygments.lexers import guess_lexer
from pygments.util import ClassNotFound


class CodeIdentifier:
    def __init__(self):
        self.code_block_patterns = [
            r'```(\w*\n)?(.*?)```',  # Fenced code blocks
            r'~~~(\w*\n)?(.*?)~~~',  # Alternative fenced blocks
            r'\'\'\'(\w*\n)?(.*?)\'\'\'',  # Triple single quotes
        ]

    def is_code_file(self, text):
        """
        Determines if the entire content is purely source code.
        """
        if not text.strip():
            return False  # Empty content is not source code

        try:
            lexer = guess_lexer(text)
            return True  # Pygments identifies it as code
        except ClassNotFound:
            return self.has_code(text)  # Fallback to heuristic check

    def has_code(self, text):
        """
        Determines if the text contains identifiable code blocks.
        """
        code_blocks = self.extract_code_blocks(text)
        return len(code_blocks) > 0

    def extract_code_blocks(self, text):
        """
        Extracts code blocks from the input text.
        """
        code_blocks = []

        # Match fenced code blocks
        for pattern in self.code_block_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                code_blocks.append(match[1] if len(match) > 1 else match[0])

        return code_blocks

    def analyze_file(self, file_path):
        """
        Analyzes a file to determine if it is a code file or contains code.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()

        is_code = self.is_code_file(content)
        has_code_blocks = self.has_code(content)

        # Harmonize flags: if it's a code file, it must contain code
        if is_code:
            has_code_blocks = True

        # If the file is purely code, consider the entire content as a block
        extracted_blocks = []
        if is_code:
            extracted_blocks = [content]
        elif has_code_blocks:
            extracted_blocks = self.extract_code_blocks(content)

        return {
            "is_code_file": is_code,
            "has_code": has_code_blocks,
            "code_blocks": extracted_blocks
        }

    def analyze_url(self, url, mode="visible"):
        """
        Fetches and analyzes content from a URL to determine if it contains code.

        Modes:
        - "visible": Extracts human-readable code snippets.
        - "source": Extracts the raw source code of the website.
        """
        try:
            response = requests.get(url)
            if response.status_code != 200:
                raise Exception(f"Failed to fetch URL: {url} (HTTP {response.status_code})")

            content_type = response.headers.get('Content-Type', '').lower()

            # If the content is plain text, analyze directly
            if 'text/plain' in content_type or url.endswith(('.py', '.txt')):
                content = response.text
                is_code = self.is_code_file(content)

                # For plain text code files, treat the entire content as a single block
                return {
                    "mode": "visible",
                    "is_code_file": is_code,
                    "has_code": is_code,  # If it's a code file, it contains code by definition
                    "code_blocks": [content] if is_code else []
                }

            # If the content is HTML, process it for visible snippets
            elif 'text/html' in content_type:
                content = response.text
                soup = BeautifulSoup(content, 'html.parser')
                plain_text = soup.get_text(separator="\n")  # Extract visible text
                has_code_blocks = self.has_code(plain_text)

                return {
                    "mode": "visible",
                    "has_code": has_code_blocks,
                    "code_blocks": self.extract_code_blocks(plain_text) if has_code_blocks else []
                }

            else:
                raise Exception("Unsupported Content-Type for URL analysis.")

        except Exception as e:
            return {"error": str(e)}


    def analyze_string(self, text):
        """
        Analyzes a string to determine if it is purely code or contains code blocks.
        """
        is_code = self.is_code_file(text)
        has_code_blocks = self.has_code(text)

        # Harmonize flags
        if is_code:
            has_code_blocks = True

        return {
            "is_code_file": is_code,
            "has_code": has_code_blocks,
            "code_blocks": self.extract_code_blocks(text) if has_code_blocks else []
        }

    def analyze(self, input_data, mode="visible"):
        """
        Unified interface for analyzing input (string, file, or URL).
        """
        if isinstance(input_data, str):
            # Check if input is a file path
            if os.path.isfile(input_data):
                return self.analyze_file(input_data)
            elif input_data.startswith("http://") or input_data.startswith("https://"):
                return self.analyze_url(input_data, mode=mode)
            else:
                return self.analyze_string(input_data)
        else:
            raise TypeError("Input data must be a string representing a file path, URL, or text content.")


def main():
    parser = argparse.ArgumentParser(description="Identify and extract code from a file or website.")
    parser.add_argument("input", help="Path to a file, a URL, or a text string")
    parser.add_argument(
        "-e", "--extract",
        action="store_true",
        help="Extract code blocks from the input"
    )
    parser.add_argument(
        "--mode",
        choices=["visible", "source"],
        default="visible",
        help="Mode of analysis for URLs: 'visible' for snippets, 'source' for raw website code (default: 'visible')"
    )

    args = parser.parse_args()
    input_data = args.input
    extract_code = args.extract
    mode = args.mode

    # Initialize the CodeIdentifier and analyze the input
    identifier = CodeIdentifier()

    try:
        result = identifier.analyze(input_data, mode=mode)

        if "error" in result:
            print(f"Error: {result['error']}")
            return

        print(f"Analyzing input: {input_data}")
        print(f"Mode: {result.get('mode', 'N/A')}")
        print(f"Is code file: {result.get('is_code_file', 'N/A')}")
        print(f"Contains code: {result.get('has_code', 'N/A')}")

        if mode == "source" and "source_code" in result:
            print("\nExtracted Source Code:")
            print(result["source_code"])
        elif extract_code and result.get("has_code"):
            print("\nExtracted Code Blocks:")
            for idx, block in enumerate(result["code_blocks"], start=1):
                print(f"\nBlock {idx}:\n{'-' * 20}\n{block.strip()}\n{'-' * 20}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()

