#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: SteamWrap.py
# Author: Wadih Khairallah
# Description: 
# Created: 2024-11-28 02:26:17
from rich.console import Console
from rich.highlighter import RegexHighlighter
import textwrap

from rich.console import Console
from rich.highlighter import RegexHighlighter
import textwrap


class StreamWrap:
    """
    A rich-enhanced implementation of a stream wrapper for testing buffer management,
    word wrapping, incremental processing logic, and text highlighting.
    """

    def __init__(self, width=75, indent=0, subsequent_indent=0, highlighter=None):
        """
        Initializes the StreamWrap instance.

        :param width: Target width for text wrapping.
        :param indent: Number of spaces for the first line's indent.
        :param subsequent_indent: Number of spaces for subsequent lines' indent.
        :param highlighter: A `rich.highlighter` instance for applying text highlights.
        """
        self.console = Console()
        self.target_width = width
        self.indent = " " * indent
        self.subsequent_indent = " " * subsequent_indent
        self.highlighter = highlighter or RegexHighlighter()
        self.buffer = ""

    def _flush_buffer(self):
        """
        Flushes the buffer, wraps text, applies highlighting, and prints complete lines.
        Retains incomplete lines for further processing.
        """
        wrapped_lines = textwrap.wrap(
            self.buffer,
            width=self.target_width,
            initial_indent=self.indent,
            subsequent_indent=self.subsequent_indent,
            break_long_words=False,
            break_on_hyphens=False,
        )

        for line in wrapped_lines[:-1]:  # Print all complete lines
            highlighted_line = self.highlighter(line)
            self.console.print(highlighted_line)

        # Retain the last (incomplete) line
        self.buffer = wrapped_lines[-1] if wrapped_lines else ""

    def __call__(self, text):
        """
        Handles incremental streaming of text into the buffer and processes it.

        :param text: Input text to stream.
        """
        for char in text:
            self.buffer += char
            if char == "\n":
                self._flush_buffer()
                self.buffer = ""

    def finalize(self):
        """
        Flushes any remaining text in the buffer and clears the state.
        """
        if self.buffer.strip():
            self._flush_buffer()
        self.buffer = ""


def main():
    """
    Test script for StreamWrap.
    """
    # Define a custom highlighter using RegexHighlighter
    class MyHighlighter(RegexHighlighter):
        base_style = "bold magenta"
        highlights = [r"\b(test|stream|highlight|wrap|buffer|functionality)\b"]

    # Create an instance of StreamWrap with custom highlighting
    stream_wrap = StreamWrap(
        width=50,
        indent=4,
        subsequent_indent=4,
        highlighter=MyHighlighter(),
    )

    test_input = (
        "This is a test to validate the StreamWrap functionality. "
        "The goal is to check proper word wrapping, incremental processing, "
        "and text highlighting with a rich-enhanced implementation."
    )

    # Simulating character-by-character streaming
    import time  # Added to simulate streaming effect
    for char in test_input:
        stream_wrap(char)
        time.sleep(0.05)  # Adjust delay for better visual streaming effect

    # Finalize to print any remaining text
    stream_wrap.finalize()


if __name__ == "__main__":
    main()

