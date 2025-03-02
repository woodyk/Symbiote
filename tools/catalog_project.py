#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: catalog_project.py
# Author: Wadih Khairallah
# Description: Directory summarizer for organization and analysis
# Created: 2024-11-30

import os
import json
import mimetypes
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

# Constants
MAX_CHAR_LIMIT = 10000  # Maximum characters per file summary
DEFAULT_IGNORE = [".git", ".gitignore", ".DS_Store"]  # Default ignored files and directories
SUMMARY_CHAR_LIMIT = 200  # Maximum characters for text summary

def load_gitignore(base_path):
    """Load patterns from .gitignore to exclude from processing."""
    gitignore_path = os.path.join(base_path, ".gitignore")
    if os.path.exists(gitignore_path):
        with open(gitignore_path, "r", encoding="utf-8", errors="ignore") as f:
            patterns = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        return patterns
    return []

def is_ignored(path, patterns):
    """Check if the given path matches any of the ignore patterns."""
    for pattern in patterns:
        if pattern in path:
            return True
    return False

def truncate_content(content, limit=MAX_CHAR_LIMIT):
    """Truncate content to the specified character limit."""
    if len(content) <= limit:
        return content
    start = content[:limit // 3]
    middle = content[len(content) // 2 - limit // 6:len(content) // 2 + limit // 6]
    end = content[-limit // 3:]
    return start + "\n...\n" + middle + "\n...\n" + end

def extract_python_summary(file_path):
    """Extract functions, classes, and docstrings from a Python file."""
    summary = []
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith("def ") or stripped.startswith("class ") or stripped.startswith('"""'):
                    summary.append(stripped)
                if len(summary) > MAX_CHAR_LIMIT // 80:  # Limit number of lines
                    break
    except Exception as e:
        return f"Error extracting summary: {e}"
    return "\n".join(summary)

def summarize_with_sumy(filepath, char_limit=SUMMARY_CHAR_LIMIT):
    """Summarize text files using the Sumy library."""
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read(char_limit)
            parser = PlaintextParser.from_string(content, Tokenizer("english"))
            summarizer = LsaSummarizer()
            summary = summarizer(parser.document, 2)  # Two sentences
            return " ".join(str(sentence) for sentence in summary)
    except Exception as e:
        return f"Error summarizing file with Sumy: {e}"

def summarize_file(file_path, mime_type):
    """Generate a summary of the file's content based on type."""
    try:
        if mime_type.startswith("text/"):
            if file_path.endswith(".py"):
                summary = extract_python_summary(file_path)
            else:
                summary = summarize_with_sumy(file_path)
        else:
            summary = f"File type: {mime_type}, Name: {os.path.basename(file_path)}"
        return truncate_content(summary, MAX_CHAR_LIMIT)
    except Exception as e:
        return f"Error summarizing file {file_path}: {e}"

def crawl_directory(base_path, depth=2, ignore_patterns=None):
    """Crawl the directory up to the specified depth, summarizing files."""
    if ignore_patterns is None:
        ignore_patterns = DEFAULT_IGNORE + load_gitignore(base_path)

    results = []

    def crawl(current_path, current_depth):
        if current_depth > depth:
            return
        try:
            for entry in os.listdir(current_path):
                full_path = os.path.join(current_path, entry)
                if is_ignored(entry, ignore_patterns):
                    continue

                if os.path.isfile(full_path):
                    mime_type, _ = mimetypes.guess_type(full_path)
                    mime_type = mime_type or "unknown/unknown"
                    file_data = {
                        "path": full_path,
                        "name": entry,
                        "mime_type": mime_type,
                        "summary": summarize_file(full_path, mime_type),
                        "size": os.path.getsize(full_path),
                    }
                    results.append(file_data)
                elif os.path.isdir(full_path):
                    crawl(full_path, current_depth + 1)
        except Exception as e:
            results.append({"error": f"Error accessing {current_path}: {e}"})

    crawl(base_path, 0)
    return results

def write_report(results, output_file="summary_report.json"):
    """Write the collected summaries to a JSON file for further analysis."""
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        print(f"Summary report written to {output_file}")
    except Exception as e:
        print(f"Error writing report: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Directory summarizer for organization.")
    parser.add_argument("path", help="Base path to crawl")
    parser.add_argument("--depth", type=int, default=2, help="Depth of directory traversal")
    parser.add_argument("--output", default="summary_report.json", help="Output file for the summary")
    args = parser.parse_args()

    print(f"Crawling directory: {args.path} with depth: {args.depth}")
    summaries = crawl_directory(args.path, depth=args.depth)
    write_report(summaries, output_file=args.output)

