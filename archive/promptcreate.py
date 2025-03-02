#!/usr/bin/env python3
#
# promptcreate.py

import os
import json
import uuid
import ast
import re

from bs4 import BeautifulSoup
from typing import List
from pathlib import Path
from pygments.lexers import guess_lexer_for_filename
from pygments.util import ClassNotFound

class PromptGenerator:
    def __init__(self, path: str, num_prompts: int = 3):
        self.path = Path(path)
        self.num_prompts = num_prompts
        if not self.path.is_dir() and not self.path.is_file():
            raise ValueError(f"{self.path} is not a valid directory or file.")
    
    def create_prompt(self):
        if self.path.is_dir():
            files = self.path.glob('**/*')
        elif self.path.is_file():
            files = [self.path]
        
        data = []
        for file in files:
            language = self.guess_language(file).lower()
            if file.suffix in ['.css', '.php', '.html', '.py', '.js', '.java', '.cpp']:  # add more file types if needed
                with open(file, 'r') as f:
                    code = f.read()
                    prompts = self.generate_prompts(code, self.num_prompts)
                    for prompt in prompts:
                        data.append({
                            'uuid': str(uuid.uuid4()) + "***",
                            'prompt': prompt,
                            'code': code
                        })

                    components = self.extract_components(code, language)
                    for component in components:
                        prompts = self.generate_prompts(component, self.num_prompts)
                        for prompt in prompts:
                            data.append({
                                'uuid': str(uuid.uuid4()),
                                'prompt': prompt, 
                                'code': component
                            })
                        
        
        print(json.dumps(data, indent=4))

        '''
        with open('dataset.jsonl', 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        '''

    def generate_prompts(self, code: str, num_prompts: int) -> List[str]:
        # Implement this method to generate prompts based on the code
        pre_prompt = f'Take the following code snippet and provide {num_prompts} NLM prompts that would describe the creation of this function in unique ways. Each prompt will be encapulated in tripple quotes """ separated by a new line. Do not have the prompts ask to describe but request the creation of the function.\n---\n'
        pre_prompt = f'NLM prompts that would describe the creation of this function in unique ways. The prompts should be formed in a way that asks for the function to be created as it might be a description in a code comment. The prompt should not require extensive definition of the function it self but a generalized way to create this type of function.  The prompts should be no longer than 25 words.\n---\n'

        prompts = []
        while num_prompts > 0:
            num_prompts -= 1
            prompts.append("this is a prompt")

        return prompts 

    def extract_components(self, code: str, language: str) -> List[str]:
        components = []

        if language == 'python':
            components = re.findall(r'\bdef\b\s+\w+\(.*?\):.*?(?=\bdef\b|\Z)', code, re.DOTALL)
        elif language == 'perl':
            components = re.findall(r'\bsub\b\s+\w+\s*{.*?}(?=\bsub\b|\Z)', code, re.DOTALL)
        elif language == 'ruby':
            components = re.findall(r'\bdef\b\s+\w+\s*(\(.*?\))?.*?end(?=\bdef\b|\Z)', code, re.DOTALL)
        elif language == 'php':
            components = re.findall(r'\bfunction\b\s+\w+\(.*?\)\s*{.*?}(?=\bfunction\b|\Z)', code, re.DOTALL)
        elif language in ['c', 'cpp']:
            components = re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\s+\w+\(.*?\)\s*{.*?}(?=[A-Za-z_][A-Za-z0-9_]*\s+\w+\(|\Z)', code, re.DOTALL)
        elif language == 'bash':
            components = re.findall(r'\bfunction\b\s+\w+\s*\(.*?\)\s*{.*?}(?=\bfunction\b|\Z)', code, re.DOTALL)
        elif language == 'go':
            components = re.findall(r'\bfunc\b\s+\(.*?\)\s*\w+\(.*?\)\s*{.*?}(?=\bfunc\b|\Z)', code, re.DOTALL)
        elif language == 'css':
            components = re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\s*{.*?}(?=[A-Za-z_][A-Za-z0-9_]*\s*{|\Z)', code, re.DOTALL)
        elif language == 'html':
            components = re.findall(r'<\w+.*?>.*?</\w+>(?=<\w+|\Z)', code, re.DOTALL)
        elif language == 'java':
            components = re.findall(r"(public|protected|private|static|\s) +[\w\<\>\[\]]+\s+(\w+) *\([^\)]*\) *(\{?|[^;])", code, re.DOTALL)
        elif language == 'javascript':
            components = re.findall(r"(function\s+\w+\s*\(.*?\)\s*\{[^}]*\})", code, re.DOTALL)
        else:
            print(f"Unsupported language: {language}")
            return "unknown"

        print(components)

        return components

    def generate_prompt(self, component: str) -> str:
        # Implement this method to generate a prompt for a component
        pass

    def guess_language(self, filename):
        try:
            with open(filename, 'r') as f:
                code = f.read()
            lexer = guess_lexer_for_filename(filename, code)
            return lexer.name
        except ClassNotFound:
            return "Unknown language"
        except Exception as e:
            return str(e)

pgen = PromptGenerator('./')
pgen.create_prompt()
