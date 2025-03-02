#!/usr/bin/env python3
#
# sym_codeextract.py

import re
import os
import uuid
from collections import Counter
from bs4 import BeautifulSoup

import pygments
from pygments.lexers import Python3Lexer, guess_lexer_for_filename, get_lexer_by_name, guess_lexer, get_all_lexers
from pygments.styles import get_all_styles
from pygments import highlight
from pygments.style import Style
from pygments.token import Token
from pygments.formatters import Terminal256Formatter

pygment_styles = [
        'murphy',
        'monokai',
        'native',
        'fruity',
        'rrt',
        'rainbow_dash',
        'stata-dark',
        'inkpot',
        'zenburn',
        'gruvbox-dark',
        'dracula',
        'one-dark',
        'nord',
        'github-dark'
    ]

class CodeBlockIdentifier:
    def __init__(self, text=None):
        self.text = text
        # Regex to match fenced code blocks in Markdown or similar formats
        self.block_match = r'```(\w*\n)?(.*?)```|```(\w*\n)?(.*?)```|~~~(\w*\n)?(.*?)~~~|\'\'\'(\w*\n)?(.*?)\'\'\''
        self.syntax_style = 'monokai'

    def extract_code_blocks(self, text):
        """
        Extracts code blocks from text using regex and BeautifulSoup for HTML.
        """
        code_blocks = re.findall(self.block_match, text, re.DOTALL)
        # Flatten the tuple results from regex and remove empty matches
        code_blocks = [item for sublist in code_blocks for item in sublist if item]

        # Additional extraction from HTML if BeautifulSoup is initialized
        if BeautifulSoup:
            soup = BeautifulSoup(text, 'html.parser')
            for tag in soup.find_all(['code', 'pre', 'script']):
                code_blocks.append(tag.get_text())

        return code_blocks

    def identify_language(self, code_block):
        """
        Identifies the programming language of a code block using Pygments.
        """
        try:
            lexer = guess_lexer(code_block)
            return lexer.aliases[0]  # Return the primary alias of the lexer
        except Exception as e:
            return "unknown"

    def highlight_code(self, code_block, language='auto'):
        """
        Syntax highlights a code block. If language is 'auto', attempts to guess.
        """
        if language == 'auto':
            try:
                lexer = guess_lexer(code_block)
            except Exception:
                lexer = get_lexer_by_name('text')
        else:
            lexer = get_lexer_by_name(language)

        formatter = Terminal256Formatter(style=self.syntax_style)
        return highlight(code_block, lexer, formatter)

    # Additional methods like lint_file, write_tmp_code, score_code_block, etc., can be added here.

    def lint_file(self, file_name):
        # Read the file contents
        with open(file_name, 'r') as file:
            code = file.read()

        # Guess the lexer based on the filename and the code
        try:
            lexer = guess_lexer_for_filename(file_name, code)
        except ClassNotFound:
            print(f'Could not determine the language of the file: {file_name}')
            return None

        # Choose the linter command based on the lexer name
        if 'python' in lexer.name.lower():
            command = ['pylint', file_name]
        elif 'javascript' in lexer.name.lower():
            command = ['eslint', file_name]
        elif 'java' in lexer.name.lower():
            command = ['checkstyle', file_name]
        elif 'c++' in lexer.name.lower() or 'c' in lexer.name.lower():
            command = ['cppcheck', file_name]
        elif 'shell' in lexer.name.lower():
            command = ['shellcheck', file_name]
        elif 'php' in lexer.name.lower():
            command = ['php', '-l', file_name]
        elif 'ruby' in lexer.name.lower():
            command = ['ruby', '-c', file_name]
        else:
            print(f'Unsupported language: {lexer.name}')
            return None

        # Run the linter command
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Return the linter output
        return process.stdout.decode()

    def extract_html_code_blocks(self, *args, **kwargs):
        if 'text' in kwargs:
            self.text = kwargs['text']

        code_blocks = re.findall(self.block_match, self.text, re.DOTALL)
        soup = BeautifulSoup(self.text, 'html.parser')
        for tag in soup.find_all(['code', 'pre', 'script']):
            code_blocks.append(tag.get_text())

        if len(code_blocks) > 0:
            return code_blocks
        else:
            return None

    def extract_markdown_code_blocks(self, *args, **kwargs):
        if 'text' in kwargs:
            self.text = kwargs['text']

        code_blocks = re.findall(self.block_match, self.text, re.DOTALL)
        new_blocks = []
        for code in code_blocks:
            code = list(code)
            mdlang = None
            if len(code[1]) > 0:
                mdlang = code[0].rstrip("\n")
                code[0] = mdlang

            filtered = [item for item in code if item]
            new_blocks.append(filtered)

        if len(new_blocks) > 0:
            return new_blocks
        else:
            return None

    def write_tmp_code(self, code_block, file_extension, file_name='/tmp/testcode'):
        unique_filename = f"{file_name}_{str(uuid.uuid4())}.{file_extension}"
        with open(unique_filename, 'w') as file:
            file.write(code_block)
        os.chmod(unique_filename, 0o755)
        return unique_filename

    def score_code_block(self, block, file_name=False):
        ''' add counts of code artifacts to the scoring eg. higher number of import, or if else statements can increase the score '''
        # Define weights for each scoring component
        weights = {
            'keyword_score': 0.2,
            'pattern_score': 0.2,
            'comment_score': 0.2,
            'code_percentage_score': 0.2,
            'filename_score': 0.1,
            'lexer_score': 0.1
        }

        # Initialize scores
        scores = {
            'keyword_score': 0,
            'pattern_score': 0,
            'comment_score': 0,
            'code_percentage_score': 0,
            'filename_score': 0,
            'lexer_score': 0
        }

        # Get all keywords from all lexers
        keywords = [item[2] for item in get_all_lexers() if item[1]]
        keywords = [item for sublist in keywords for item in sublist]

        # Find all tokens in the block
        tokens = re.findall(r'\b\w+\b', text)
        token_counts = Counter(tokens)

        # Calculate the score based on the presence of keywords
        scores['keyword_score'] = sum(count for token, count in token_counts.items() if token in keywords) / len(tokens)

        # Calculate the score based on the presence of import statements, function or class definitions, and common programming keywords
        patterns = [r'\bimport\b', r'\bdef\b', r'\bclass\b', r'\bif\b', r'\belse\b', r'\bfor\b', r'\bwhile\b', r'\breturn\b', r'\bin\b', r'\btry\b', r'\bexcept\b']
        scores['pattern_score'] = sum(1 for pattern in patterns if re.search(pattern, text)) / len(tokens)

        # Calculate the score based on the presence of comment lines
        comment_patterns = [r'//', r'#', r'"""', r'/*', r"'''"]
        scores['comment_score'] = sum(1 for pattern in comment_patterns if re.search(pattern, text)) / len(tokens)

        # Calculate the percentage of code vs other text
        code_percentage = len(re.findall(r'\b\w+\b', text)) / len(text.split())
        scores['code_percentage_score'] = code_percentage

        # Use Pygments to guess the language
        try:
            guess_lexer(text)
            scores['lexer_score'] = 1
        except:
            pass

        # Use Pygments to get a lexer for the filename
        if filename:
            try:
                get_lexer_for_filename(file_name)
                scores['filename_score'] = 1
            except:
                pass

        # Calculate the final score as the weighted sum of the scores
        final_score = sum(scores[key] * weights[key] for key in scores)

        return final_score

    def identify_language(self, block, lang=None):
        print(lang)
        try:
            if lang is not None:
                # If 'lang' is provided, use it to get the lexer
                lexer = get_lexer_by_name(lang, stripall=True)
            else:
                # Try to guess the lexer
                lexer = guess_lexer(block)
            language = lexer.name
            # Extract the file extension from the lexer's filenames attribute
            file_extension = lexer.filenames[0].split('.')[-1]
            return language, file_extension
        except Exception as e:
            # Handle other exceptions gracefully, e.g., print an error message
            return None, None

    def process_text(self):
        code_blocks = self.extract_markdown_code_blocks()
        if code_blocks is None:
            code_blocks = self.extract_html_code_blocks()

        code_files = []
        for code_block in code_blocks:
            if len(code_block) > 1:
                language, extension = self.identify_language(code_block[1], lang=code_block[0])
                code = code_block[1]
            else:
                language, extension = self.identify_language(code_block[0])
                code = code_block[0]

            file_path = self.write_tmp_code(code, extension)
            code_files.append(file_path)

        return code_files

    def syntax_highlighter(self, *args, **kwargs):
        if 'text' in kwargs:
            self.text = kwargs['text']

        # Create a Terminal256Formatter instance for formatting the highlighted output
        formatter = Terminal256Formatter(style=self.syntax_style)
        lexer = Python3Lexer()

        # Strip and save \n from original content
        slash_ns = ''
        slash_n_pattern = r'(\n|\n+)$'
        match = re.search(slash_n_pattern, self.text)
        if match:
            slash_ns = match.group(1)

        highlighted_text = highlight(self.text, lexer, formatter)
        highlighted_text = re.sub(slash_n_pattern, slash_ns, highlighted_text)

        return highlighted_text

# Example usage
if __name__ == "__main__":
    identifier = CodeBlockIdentifier()
    text = """Here is some Python code:
    ```python
    def hello_world():
        print("Hello, world!")
    ```
    And here is some HTML:
    ```html
    <div>Hello, world!</div>
    ```
    """
    code_blocks = identifier.extract_code_blocks(text)
    for block in code_blocks:
        language = identifier.identify_language(block)
        highlighted = identifier.highlight_code(block, language)
        print(f"Language: {language}\nHighlighted Code:\n{highlighted}")
