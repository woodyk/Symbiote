#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: stream_print.py
# Author: Wadih Khairallah
# Description: 
# Created: 2024-11-27 14:04:06
# Modified: 2024-11-28 04:45:50

from rich.console import Console
from rich.highlighter import ReprHighlighter
from rich.highlighter import RegexHighlighter
import textwrap
import re
import sys

class StreamWrap:
    """
    A rich-enhanced stream wrapper for testing buffer management,
    word wrapping, incremental processing logic, and real-time highlighting.
    """

    def __init__(self, width=20, indent=0, subsequent_indent=0, highlighter=None):
        """
        Initializes the StreamWrap instance.

        :param width: Target width for text wrapping.
        :param indent: Number of spaces for the first line's indent.
        :param subsequent_indent: Number of spaces for subsequent lines' indent.
        :param highlighter: Optional `rich.highlighter` instance for applying text highlights.
        """
        self.console = Console()
        self.target_width = (self.console.width * (75 / 100)) - int(indent)
        self.chunk_buffer = ""
        self.wrapper = textwrap.TextWrapper(
            width=self.target_width,
            initial_indent=" " * indent,
            subsequent_indent=" " * subsequent_indent,
            break_long_words=False,
            break_on_hyphens=False,
        )
        self.highlighter = highlighter or ReprHighlighter()
        self.linecheck = False
        self.linecounter = 0

    def _render_output(self, text):
        """
        Renders the text to the console using `rich` for dynamic highlighting.

        :param text: The text to render.
        """
        highlighted_text = self.highlighter(text)
        #self.console.print(highlighted_text, end="")

    def _process_buffer(self):
        """
        Processes the buffer, wraps text, and renders complete lines.
        Retains incomplete lines in the buffer for further processing.
        """
        # Wrap the chunk buffer
        wrapped_lines = self.wrapper.wrap(self.chunk_buffer)

        # Debugging: Log the buffer content and wrapped lines
        #print(f"[DEBUG] Buffer content: {self.chunk_buffer}")
        #print(f"[DEBUG] Wrapped lines: {wrapped_lines}")

        # Print all complete lines
        for line in wrapped_lines[:-1]:  # Exclude the last (incomplete) line
            self._render_output(line + "\n")  # Render with a newline for proper line breaks

        # Retain only the unprocessed portion of the buffer
        if wrapped_lines:
            processed_text = "".join(line.lstrip() for line in wrapped_lines[:-1])
            self.chunk_buffer = self.chunk_buffer[len(processed_text):].lstrip()
        else:
            self.chunk_buffer = ""

    def __call__(self, chunk):
        """
        Handles incremental streaming of text into the buffer and processes it.

        :param text: Input text to stream.
        """
        chunk_len = len(chunk)
        self.chunk_buffer += chunk
        wrapped = self.wrapper.wrap(self.chunk_buffer)
        wrapped_len = len(wrapped) - 1
        wrapped_data = wrapped.pop()
        if wrapped_len > 1:
            print("\n")

        if re.search(r"^\s", wrapped_data) and self.linecheck is False:
            print(wrapped_data, end="", flush=True)
            self.linecheck = True
        elif chunk == " ":
            print(chunk, end="", flush=True)
        else:
            print(wrapped_data[-chunk_len:], end="", flush=True)

    def finalize(self):
        """
        Processes any remaining text in the buffer and clears it.
        """
        if self.chunk_buffer:
            wrapped_lines = self.wrapper.wrap(self.chunk_buffer)
            for line in wrapped_lines:
                self._render_output(line + "\n")
        self.chunk_buffer = ""


# Debugging and Testing
if __name__ == "__main__":
    # Define a custom highlighter using RegexHighlighter
    class MyHighlighter(RegexHighlighter):
        base_style = "bold magenta"
        highlights = [r"\b(test|validate|functionality|wrapping|processing)\b"]

    # Create a StreamWrap instance with custom highlighting
    stream_wrap = StreamWrap(width=50, indent=0, subsequent_indent=0, highlighter=MyHighlighter())

    test_input = "This is a test to validate the StreamWrap functionality.  The goal is to check proper word wrapping and incremental processing."

    # Simulating character-by-character streaming
    import time
    for char in test_input:
        stream_wrap(char)
        time.sleep(0.1)  # Simulate streaming delay

    # Finalize to print any remaining text
    stream_wrap.finalize()


"""
# Intelligent Debugging and Refactoring Directive

## Purpose
This script is being debugged and/or refactored using an iterative, analysis-driven process with a Large Language Model (LLM). The following behaviors and methodologies should guide the LLM in assisting with this project:

## Methodology
1. **Incremental Testing and Analysis**:
    - Whenever an issue is identified, analyze the smallest functional components of the code.
    - Use internal Python analysis capabilities to validate functionality, logic, and output.
    - Truncate test outputs to only include what is necessary to confirm functionality, avoiding unnecessary clutter in responses.

2. **Iterative Fix and Validation Cycle**:
    - Propose and implement fixes incrementally, addressing one issue at a time.
    - Return modified code snippets as needed to address specific problems or inconsistencies.
    - After providing a fix, run a test to confirm the issue is resolved before moving to the next issue.

3. **In-Code Annotations**:
    - Document debugging notes and reasoning directly in the code to maintain a historical record for future debugging sessions.
    - Include pertinent details on why changes were made and how they affect overall functionality.

4. **Merge-Ready Solutions**:
    - Ensure all proposed solutions are designed to merge seamlessly with the existing codebase.
    - Provide clean, optimized chunks of code with explicit instructions for integrating them into the project.

5. **Refinement Until Finalization**:
    - Continue iterative analysis and debugging until the implementation works as intended, is efficient, and is robust.

## Instructions to the LLM
1. Whenever you receive this script for debugging or enhancement:
    - Begin by identifying issues or testing the provided functionality using Python analysis.
    - Incrementally address each issue with precise fixes and verify the results before proceeding.
    - Return concise, actionable code snippets that are ready for integration.

2. If additional context is required, prompt the user to clarify the problem, desired behavior, or test case specifics.

3. Always maintain notes in the code that explain why changes were made and how they address identified issues.

## Goal
The ultimate goal of this process is to resolve all issues in the code incrementally and to provide a clean, fully functional, and refactored implementation by the end of the debugging session.

"""

