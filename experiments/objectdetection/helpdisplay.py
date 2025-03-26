#!/usr/bin/env python3
#
# helpdisplay.py

from rich.console import Console
from rich.table import Table

# Define the command list as before
command_list = {
    "help::": "This help output.",
    "convo::": "Load, create conversation.",
    "role::": "Load built in system roles.",
    "clear::": "Clear the screen.",
    "flush::": "Flush the current conversation from memory.",
    "tokens::": "Token usage summary.",
    "save::": "Save self.symbiote_settings and backup the ANNGL",
    "exit::": "Exit symbiote the symbiote CLI",
    "setting::": "View, change, or add settings for symbiote.",
    "maxtoken::": "Change maxtoken setting.",
    "model::": "Change the AI model being used.",
    "cd::": "Change working directory.",
    "pwd::": "Show current working directory.",
    "file::": "Load a file for submission.",
    "webvuln::": "Run and summarize a web vulnerability scan on a given URL.",
    "deception::": "Run deception analysis on the given text",
    "fake_news::": "Run fake news analysis on the given text",
    "yt_transcript::": "Download the transcripts from youtube url for processing.",
    "image_extract::": "Extract images from a given URL and display them.",
    "analyze_image::": "Analyze an image or images from a website or file.",
    "w3m::|browser::": "Open a URL in w3m terminal web browser.",
    "nuclei::": "Run a nuclei scan on a given domain and analyze the results.",
    "qr::": "Generate a QR code from the given text.",
    "extract::": "Extract data features for a given file or directory and summarize.",
    "links::": "Extract links from the given text.",
    "code::": "Extract code and write files.",
    "get::": "Get remote data based on uri http, ftp, ssh, etc...",
    "crawl::": "Crawl remote data based on uri http, ftp, ssh, etc...",
    "tree::": "Load a directory tree for submission.",
    "shell::": "Load the symbiote bash shell.",
    "clipboard::": "Load clipboard contents into symbiote.",
    "ls::": "Load ls output for submission.",
    "search::": "Search index for specific data.",
    "history::": "Show discussion history.",
    "train::": "Train AI model on given data in a file or directory.",
    "structure::": "Data structure builder.",
    "exec::": "Execute a local cli command and learn from the execution fo the command.",
    "fine-tune::": "Fine-tune a model on a given data in a file or directory.",
    "image::": "Render an image from the provided text.",
    "replay::": "Replay the current conversation to the current model.",
    "prompter::": "Create prompts matched to datasets.",
    "purge::": "Purge the last response given. eg. thumbs down",
    "note::": "Create a note that is tracked in a separate conversation",
    "index::": "Index files into Elasticsearch.",
    "whisper::": "Process audio file to text using whipser.",
    "define::": "Request definition on keyword or terms.",
    "theme::": "Change the theme for the symbiote cli.",
    "view::": "View a file",
    "scroll::": "Scroll through the text of a given file a file",
    "dork::": "Run a google search on your search term.",
    "wiki::": "Run a wikipedia search on your search term.",
    "headlines::|news::": "Get headlines from major news agencies.",
    "agents::": "Run an iterative agents request specifying the number of iterations.",
    "mail::": "Load e-mail messages from gmail.",
}

def symhelp():
    # Create a Console object
    console = Console()

    # Create a table with two columns: "Command" and "Description"
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")

    # Sort the commands and add rows to the table
    for cmd, desc in sorted(command_list.items()):
        table.add_row(cmd, desc)

    # Display the table using the Console object
    console.print(table)

# Example usage of the symhelp function
symhelp()

