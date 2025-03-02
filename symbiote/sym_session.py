#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: sym_session.py
# Author: my name here
# Description: 
# Created: 2024-11-22 19:01:02
# Modified: 2024-11-30 04:50:53

import time
import sys
import os
import re
import json
import base64
import importlib
import queue
import threading
import subprocess
import tempfile
import platform
from collections import Counter
from io import BytesIO
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

# Local imports (explicit, no wildcard)
from symbiote.sym_imports import (
        print, sp,
        box, inspect, Align,
        SQUARE, Columns, Console,
        Group, Highlighter, Live,
        Markdown, escape, Padding,
        Panel, Rule, Syntax,
        Table, Text, Theme,
        Tree, pretty, inspect
    )
from symbiote.sym_highlighter import SymHighlighter
sym_theme = SymHighlighter()
console = Console(highlighter=sym_theme, theme=sym_theme.theme)
print = console.print
log = console.log

log("Loading symbiote roles.")
from symbiote.sym_roles import Roles
log("Loading symbiote speech.")
from symbiote.sym_speech import SymSpeech
log("Loading symbiote utils.")
from symbiote.sym_utils import (
        is_url, is_image, extract_metadata,
        extract_text, clean_path
    )
log("Loading symbiote theme_manager.")
from symbiote.theme_manager import ThemeManager
log("Loading symbiote module_inspector.")
from symbiote.sym_module_inspector import ModuleInspector as Inspect
log("Loading symbiote memory_store.")
from symbiote.sym_memory_store import SymMemoryStore
log(f"Loading symbiote sym_toolbar.")
import symbiote.sym_toolbar as sym_toolbar
log(f"Loading symbiote sym_code_extract.")
from symbiote.sym_code_extract import CodeIdentifier

# Third-party imports
log(f"Loading third party modules.")
import requests
import clipboard
import qrcode
import psutil
from halo import Halo
from PIL import Image, ImageDraw, ImageColor
from bs4 import BeautifulSoup
import pgeocode

# Set up pgeocode (moved initialization to the script's main logic or class constructor)
nomi = pgeocode.Nominatim('us')

log("Loading inquirerpy.")
# InquirerPy imports
from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.validator import PathValidator
from InquirerPy.prompts.filepath import FilePathCompleter

log("Loading prompt_toolkit.")
# Application and session management
from prompt_toolkit import Application, print_formatted_text 
from prompt_toolkit.document import Document
from prompt_toolkit.history import InMemoryHistory, FileHistory
from prompt_toolkit.shortcuts import (
        PromptSession, print_container, prompt, 
        input_dialog, yes_no_dialog, progress_dialog,
        message_dialog
    )
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.completion import Completion, WordCompleter
from prompt_toolkit.styles import Style
from prompt_toolkit.layout import Layout, HSplit
from prompt_toolkit.widgets import (
        Label, SearchToolbar, Dialog,
        TextArea, Frame, Box, Button
    )
from prompt_toolkit.layout.containers import (
        Window, VSplit, Float,
        FloatContainer
    )
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.cursor_shapes import CursorShape, ModalCursorShapeConfig
from prompt_toolkit.application.current import get_app
from prompt_toolkit.formatted_text import ANSI, HTML
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.lexers import PygmentsLexer
from pygments.lexers.python import PythonLexer

system = platform.system()

models = [
        "groq:llama-3.1-70b-versatile",
        "groq:mixtral-8x7b-32768",
        ] 

log("Loading ollama.")
from ollama import Client
olclient = Client(host='http://localhost:11434')
try:
    model_obj = olclient.list()
    for mod in model_obj['models']:
        models.append("ollama:" + mod.model)
except Exception as e:
    log(f"Failed to load ollama models: {e}")

log("Loading openai.")
import openai
oaiclient = openai.OpenAI()
try:
    response = oaiclient.models.list()
    for model in response:
        models.append("openai:" + model.id)
except Exception as e:
    log(e)
    pass

log("Loading groq.")
from groq import Groq
try:
    grclient = Groq()
except Exception as e:
    log(e)

command_list = {
        "help::": "This help output.",
        "convo::": "Load, create conversation.",
        "roles::": "Load built in system roles.",
        "clear::": "Clear the screen.",
        "flush::": "Flush the current conversation from memory.",
        "save::": "Save self.settings and backup the ANNGL",
        "exit::": "Exit symbiote the symbiote CLI",
        "settings::": "View, change, or add settings for symbiote.",
        "model::": "Change the AI model being used.",
        "cd::": "Change working directory.",
        "file::": "Load a file for submission.",
        "memory::": "CRUD access to symbiote memory",
        "search::": "Search symbiote memory for information",
        "memget::": "Pull contents from symbiote memory for analysis.",
        "reload::": "Reload running python modules.",
        "weather::": "Display the current weater.",
        "inspect::": "Realtime python object inspector for current running session.",
        "vscan::": "Run and summarize a web vulnerability scan on a given URL.",
        "deception::": "Run deception analysis on the given text",
        "geo::": "Check the geo location of an IP or a domains IP.",
        "fake_news::": "Run fake news analysis on the given text",
        "yt_transcript::": "Download the transcripts from youtube url for processing.",
        "image_extract::": "Extract images from a given URL and display them.",
        "analyze_image::": "Analyze an image or images from a website or file.",
        "w3m::|browser::": "Open a URL in w3m terminal web browser.",
        "qr::": "Generate a QR code from the given text.",
        "extract::": "Extract data features for a given file or directory and summarize.",
        "code::": "Extract code and write files.",
        "get::": "Get remote data based on uri http, ftp, ssh, etc.",
        "getip::": "Get your IP address information.",
        "crawl::": "Crawl remote data based on uri http, ftp, ssh, etc.",
        "shell::": "Work in shell mode and execute commands.",
        "clipboard::": "Load clipboard contents into symbiote.",
        "history::": "Show discussion history.",
        "$": "Execute a local cli command and learn from the execution fo the command.",
        "image::": "Render an image from the provided text.",
        "note::": "Create a note that is tracked in a separate conversation",
        "index::": "Index files into Elasticsearch.",
        "define::": "Request definition on keyword or terms.",
        "theme::": "Change the theme for the symbiote cli.",
        "view::": "View a file",
        "scroll::": "Scroll through the text of a given file a file",
        "google::": "Run a google search on your search term.",
        "wiki::": "Run a wikipedia search on your search term.",
        "headlines::|news::": "Get headlines from major news agencies.",
        "mail::": "Load e-mail messages from gmail.",
    }

audio_triggers = {
        'speech_off': [r'keyword speech off', 'settings:speech:0:'],
        'speech_on': [r'keyword speech on', 'settings:speech:1:'],
        'interactive': [r'keyword interactive mode', 'settings:listen:0:'],
        'settings': [r'keyword show setting', 'settings::'],
        'file': [r'keyword open file', 'file::'],
        'shell': [r'keyword (open shell|openshell)', 'shell::'],
        'role': [r'keyword change (role|roll)', 'roles::'],
        'conversation': [r'keyword change conversation', 'convo::'],
        'model': [r'keyword change model', 'model::'],
        'get': [r'keyword get website', 'get::'],
        'whisper': [r'keyword whisper', 'whisper::'],
        'crawl': [r'keyword crawl website', 'crawl::'],
        'clipboard_url': [r'keyword get clipboard [url|\S+site]', 'clipboard:get:'],
        'clipboard': [r'keyword get clipboard', 'clipboard::'],
        'exit': [r'keyword exit now', 'exit::'],
        'help': [r'keyword (get|show) help', 'help::'],
        'extract': [r'keyword extract data', 'extract::'],
        'summary': [r'keyword summarize data', 'summary::'],
        'keyword': [r'keyword (get|show) keyword', 'keywords::'],
        'history': [r'keyword (get|show) history', 'history::'],
        'perifious': [r'(i cast|icast) periph', 'perifious::'],
        'scroll': [r'keyword scroll file', 'scroll::'],
    }


# Default settings for openai and symbiote module.
homedir = os.getenv('HOME')
symbiote_settings = {
        "model": "",
        "max_tokens": 512,
        "location": '',
        "stream": True,
        "user": "smallroom",
        "conversation": '',
        "vi_mode": False,
        "speech": False,
        "listen": False,
        "debug": False,
        "elasticsearch": "http://dockera.vm.sr:9200",
        "elasticsearch_index": "symbiote",
        "symbiote_path": os.path.join(homedir, ".symbiote"),
        "perifious": False,
        "role": "DEFAULT",
        "notes": os.path.join(homedir, ".symbiote") + "/notes.jsonl",
        "theme": 'default',
        "imap_username": '',
        "imap_password": '',
        "think": False,
        "markdown": True,
        "config_file": f"{homedir}/.symbiote/config"
    }

class SymSession():
    def __init__(self, *args, **kwargs):
        self.config_file = symbiote_settings['config_file']
        settings = self.load_settings()
        if settings is None:
            self.settings = symbiote_settings 
            settings = symbiote_settings

        self.settings = settings 
        self.conversation_history = []
        self.estimated_tokens = self.estimate_token_count(json.dumps(self.conversation_history))
        self.audio_triggers = audio_triggers
        self.spinner = Halo(text='Processing ', spinner='dots')
        self.shell_mode = False
        self.command_register = []

        # initialize memory manager
        self.memory = SymMemoryStore(save_path="/tmp/symbiote_mem.json")

        if 'debug' in kwargs:
            self.settings['debug'] = kwargs['debug']
            
        if 'working_directory' in kwargs:
            self.working_directory = kwargs['working_directory']
        else:
            self.working_directory = os.getcwd()

        # Set symbiote home path parameters
        symbiote_dir = os.path.expanduser(self.settings['symbiote_path'])
        if not os.path.exists(symbiote_dir):
            os.mkdir(symbiote_dir)

        if 'stream' in kwargs:
            self.settings['stream'] = kwargs['stream']

        # Get hash for current settings
        self.default_hash = hash(json.dumps(self.settings, sort_keys=True)) 

        # Set the conversations directory
        self.conversations_dir = os.path.join(symbiote_dir, "conversations")
        if not os.path.exists(self.conversations_dir):
            os.mkdir(self.conversations_dir)

        # Set the default conversation
        if self.settings['conversation'] == '/dev/null':
            self.conversations_file = self.settings['conversation']
            self.convo_file = self.conversations_file
        else:
            self.conversations_file = os.path.join(self.conversations_dir, self.settings['conversation'])
            self.convo_file = os.path.basename(self.conversations_file)

        # Set symbiote shell history file
        history_file = os.path.join(symbiote_dir, "symbiote_shell_history")
        if not os.path.exists(history_file):
            open(history_file, 'a').close()

        self.history = FileHistory(history_file)

        # Load the default conversation
        #self.current_conversation = self.load_conversation(self.conversations_file)
        self.current_conversation = []

        # Init the shell theme manager
        self.theme_manager = ThemeManager()
        self.prompt_style = self.theme_manager.get_theme(self.settings['theme'])

        self.command_list = command_list

        # Command completion for the prompt
        commands = []
        for command in self.command_list:
            commands.append(command)

        self.command_completer = WordCompleter(commands)

        # Get location details
        self.geo = nomi.query_postal_code(self.settings['location'])

        if 'suppress' in kwargs:
            self.suppress = kwargs['suppress']
        else:
            self.suppress = False

    def display_settings(self):
        table = Table(show_header=False, box=None, expand=True, header_style="")
        table.add_column("Key", style="gold1", ratio=2, no_wrap=True, justify="right")
        table.add_column("Value", style="", ratio=4, justify="left")

        for key, val in sorted(self.settings.items()):
            table.add_row(key, str(val))

        print(table)

    def display_help(self, short=True):
        if short:
            # Create a table with three columns for commands only
            table = Table(show_header=False, box=None, expand=True, header_style="")
            table.add_column("Command 1", style="gold1", ratio=1, no_wrap=True, justify="left")
            table.add_column("Command 2", style="gold1", ratio=1, no_wrap=True, justify="left")
            table.add_column("Command 3", style="gold1", ratio=1, no_wrap=True, justify="left")

            # Prepare rows for three-column layout
            commands = sorted(self.command_list.keys())
            for i in range(0, len(commands), 3):
                row = commands[i:i + 3]  # Get the next three commands
                while len(row) < 3:  # Ensure the row has exactly three items
                    row.append("")
                table.add_row(*row)
        else:
            # Create a two-column table with commands and descriptions
            table = Table(show_header=False, box=None, expand=True, header_style="")
            table.add_column("Command", style="gold1", ratio=2, no_wrap=True, justify="right")
            table.add_column("Description", style="", ratio=6, justify="left")

            # Add rows with command descriptions
            for cmd, desc in sorted(self.command_list.items()):
                table.add_row(cmd, desc)

        print(table)
        """
        # Panel with instructions
        panel = Panel(
            table,
            title="Command Help",
            subtitle="Press 'q' or 'Escape' to exit",
            height=console.height,
        )

        # Key bindings for prompt_toolkit
        help_keys = KeyBindings()

        @help_keys.add(Keys.Any)  # Catch any key not explicitly handled
        def _(event):
            event.app.exit()

        help_session = PromptSession(key_bindings=help_keys)
        with Live(panel, console=console, screen=True, refresh_per_second=10) as live:
            user_input = help_session.prompt(key_bindings=help_keys)
            return
        """

    def display_convo(self, convo=False):
        conversations = sorted(self.get_conversations(self.conversations_dir))
        if convo:
            selected_file = convo
        else:
            if not conversations:
                return
            conversations.insert(0, Choice("notes", name="Open notes conversation."))
            conversations.insert(0, Choice("clear", name="Clear conversation."))
            conversations.insert(0, Choice("export", name="Export conversation."))
            convesations.insert(0, Choice("new", name="Create new conversation."))

            selected_file = self.list_selector("Select a conversation:", conversations)

        if selected_file == None:
            return

        if selected_file == "new":
            selected_file = self.text_prompt("File name:")
        elif selected_file == "notes":
            selected_file = self.settings['notes']
        elif selected_file == "clear":
            clear_file = self.list_selector("Select a conversation:", conversations)

            clear_file = os.path.join(self.conversations_dir, clear_file)

            try:
                with open(clear_file, 'w') as file:
                    pass
            except:
                log(f"Unable to clear {clear_file}")

            if self.settings['conversation'] == os.path.basename(clear_file):
                self.current_conversation = self.sym.load_conversation(clear_file)

            log(f"Conversation cleared: {clear_file}")

            return
        elif selected_file == "export":
            export_file = self.list_selector("Select a conversation:", conversations)

            file_name = os.path.join(self.conversations_dir, export_file)
            self.sym.export_conversation(file_name)

            return
        
        if selected_file == "null": 
            self.conversations_file = '/dev/null'
            self.settings['conversation'] = self.conversations_file
            self.current_conversation = []
            self.convo_file = self.conversations_file
        else:
            self.settings['conversation'] = selected_file
            self.conversations_file = os.path.join(self.conversations_dir, selected_file)
            self.current_conversation = self.sym.load_conversation(self.conversations_file)
            self.convo_file = os.path.basename(self.conversations_file)

        log(f"Loaded conversation: {selected_file}")

        return

    def selectModel(self, *args):
        model_list = models
        print(f"Current Model: {self.settings['model']}")
        try:
            model_name = args[0]
            if model_name in model_list:
                selected_model = args[0]
            else:
                log(f"No such model: {model_name}")
                return None
        except:
            selected_model = self.list_selector("Select a model:", sorted(model_list))

        self.settings['model'] = selected_model
        self.save_settings()

        return None

    def console(self, *args, **kwargs):
        #history = InMemoryHistory() 
        if 'prompt_only' in kwargs:
            self.prompt_only = kwargs['prompt_only']
        else:
            self.prompt_only = False

        if 'suppress' in kwargs:
            self.suppress = kwargs['suppress']
        else:
            self.suppress = False

        if 'user_input' in kwargs:
            user_input = kwargs['user_input']

        if 'working_directory' in kwargs:
            self.working_directory = kwargs['working_directory']
            self.previous_directory = self.working_directory
            os.chdir(self.working_directory)

        self.toolbar_message = "none" 
        kb = KeyBindings()
        self.live_render_buffer = ""

        # handle control-c
        @kb.add('c-c')
        def handle_inturrupt(event):
            log("Control-C detected.")
            sys.exit(1)

        def get_toolbar():
            return self.toolbar_message

        def update_toolbar():
            while True:
                time.sleep(2)
                tb_settings = { 
                    "Time": datetime.now().strftime("%H:%M:%S"),
                    "Model": self.settings['model'],
                    "Role": self.settings['role'],
                    "Shell": self.shell_mode
                }
                tb_functions = {
                    "CPU": lambda: f"{psutil.cpu_percent()}%",
                    "Memory": lambda: f"{psutil.virtual_memory().percent}%",
                }
                try:
                    self.toolbar_message = sym_toolbar.render_dashboard(
                        settings=tb_settings,
                        functions=tb_functions,
                        max_lines=8,
                        tail_lines=self.live_render_buffer,
                    )
                    app = get_app()
                    app.invalidate()
                except Exception as e:
                    log(f"Failure to pull toolbar message: {e}")

        def get_prompt():
            if self.shell_mode is True:
                prompt_content = "shell mode> "
            else:
                prompt_content = f"{self.settings['role'].lower()}> "
            
            prompt = f"{prompt_content}"
            return prompt 

        try:
            self.ps = PromptSession(
                key_bindings=kb,
                vi_mode=self.settings['vi_mode'],
                history=self.history,
                style=self.prompt_style,
                enable_system_prompt=True,
                bottom_toolbar=get_toolbar,
                enable_suspend=True,
                enable_open_in_editor=False,
                mouse_support=True,
                )

            toolbar_updater = threading.Thread(target=update_toolbar, daemon=True)
            toolbar_updater.start()
        except Exception as e:
            log(f"Failure to start PromptSession: {e}")
            return None


        while True:
            response = str()
            user_input = str()
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            current_date = now.strftime("%m/%d/%Y")

            print("\n")
            print("\n")

            # Chack for a change in settings and write them
            check_settings = hash(json.dumps(self.settings, sort_keys=True)) 

            self.estimated_tokens = self.estimate_token_count(json.dumps(self.conversation_history))

            if self.settings['listen']:
                if not hasattr(self, 'symspeech'):
                    speech = SymSpeech(settings=self.settings)
                    speechQueue = SymSpeech.start_keyword_listen()

                user_input = self.speech.keyword_listen()

            # Get the current path
            current_path = os.getcwd()
            # Get the home directory
            home_dir = os.path.expanduser('~')

            # Replace the home directory with ~
            if current_path.startswith(home_dir):
                current_path = '~' + current_path[len(home_dir):]

            def process_user_input(user_input=None):
                def not_empty(user_input):
                    if (user_input is None
                        or user_input.startswith("\n")
                        or user_input == ""):
                        return False 
                    return True 

                if not_empty(user_input):
                    user_input = self.process_commands(user_input)

                response = ""
                if not_empty(user_input):
                    response = self.send_message(user_input)
                    print()

                    if self.shell_mode is True:
                        if response.startswith("`") and response.endswith("`"):
                            response = response[1:-1]

                        print(response)
                        confirm = self.yes_no_prompt("Execute command?")
                        if confirm is True:
                            output = self.exec_command(response)
                            print(f"\n{output}\n")

                return response

            try:
                user_input = self.ps.prompt(
                    message=get_prompt(),
                    multiline=True,
                    default=user_input,
                    cursor=CursorShape.BLOCK,
                    rprompt=f"{current_date} {current_time}",
                    vi_mode=self.settings['vi_mode'],
                    completer=self.command_completer,
                    complete_while_typing=False,
                    refresh_interval=0.5,
                )

                if re.search(r'^exit::', user_input):
                    self.save_settings()
                    sys.exit(0)

                user_input_processing = threading.Thread(
                        target=process_user_input,
                        args=(user_input,),
                        daemon=True
                    )
                user_input_processing.start()
            except KeyboardInterrupt:
                break
            
            if check_settings != self.default_hash:
                self.save_settings()
                self.default_hash = check_settings

            # Prompt waits here for processing to complete
            app = get_app()
            app.invalidate()
            while user_input_processing.is_alive():
                time.sleep(0.1)


    def writeHistory(self, role, text):
        hist_entry = {
                #"epoch": time.time(),
                "role": role,
                "content": text 
                }
        self.conversation_history.append(hist_entry)


    def send_message(self, user_input):
        if self.settings['model'] is None:
            log(f"No model selected.  Issue model::")
            return None

        assistant_prompt = "symbiote> "
        if self.settings['think'] is True:
            self.think(user_input)

        current_role = self.settings['role']
        available_roles = Roles.get_roles()

        self.geo = nomi.query_postal_code(self.settings['location'])
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        current_date = now.strftime("%m/%d/%Y")

        suffix = f"time:{current_time},"
        suffix += f"date:{current_date},"
        suffix += f"operating system:{system},"
        for key, val in self.geo.items():
            if key == "accuracy":
                continue

            suffix += f"{key}: {val},"

        #system_prompt = f"{available_roles[current_role]}\n{suffix.strip()}"
        #system_prompt = available_roles[current_role available_roles[current_role] 
    
        #self.writeHistory('system', system_prompt)
        self.writeHistory('user', user_input)

        '''
        self.estimated_tokens = self.estimate_token_count(json.dumps(self.conversation_history))
        num_ctx = self.estimated_tokens + 8192
        if self.estimated_tokens > self.settings['max_tokens']:
            self.conversation_history = self.truncate_history(self.conversation_history, self.settings['max_tokens'])
            self.estimated_tokens = self.estimate_token_count(json.dumps(self.conversation_history))
            num_ctx = self.estimated_tokens + 8192
        '''
        num_ctx = self.settings['max_tokens']

        """ Rule of thumb character count for efficent queries """
        message = self.conversation_history
        tlen = self.estimate_token_count(json.dumps(message))
        if int(tlen) >= 100000:
            log(f"High token estimate {tlen}: concider flush::")

        if self.shell_mode is True:
            message = []
            current_role = "SHELL"
            self.suppress = True
            message.append({"role": "system", "content": available_roles[current_role]})
            message.append({"role": "user", "content": user_input})

        response = str() 
        streaming = self.settings['stream']
        markdown = self.settings['markdown']
        if self.shell_mode is True:
            streaming = False
            markdown = False

        if len(message) == 0:
            log(f"The user input is empty.  Possibly your input was too large.")
            return None

        if self.settings['model'].startswith("openai"):
            model_name = self.settings['model'].split(":")
            model = model_name[1]
            
            # Remove system messages from the history if using o1 models
            # o1 models do not allow role type of system
            # also streaming must be turned off
            if model.startswith("o1-"):
                streaming = False
                for message in self.conversation_history:
                    if message['role'] == 'system':
                        self.conversation_history.remove(message)

            # OpenAI Chat Completion
            try:
                stream = oaiclient.chat.completions.create(
                        model = model,
                        messages = message,
                        stream = streaming,
                        )
            except Exception as e:
                log(e)
                return None

            if self.suppress is True:
                response = stream.choices[0].message.content
            else:
                if streaming:
                    for chunk in stream:
                        if chunk.choices[0].delta.content is not None:
                            response += chunk.choices[0].delta.content
                            if markdown is True:
                                print(chunk.choices[0].delta.content, end="")
                            else:
                                print(chunk.choices[0].delta.content, end='')
                else:
                    response = stream.choices[0].message.content
                    if markdown is True:
                        print(Markdown(response))
                    else:
                        print(response)

        elif self.settings['model'].startswith("ollama"):
            # Ollama Chat Completion
            available_functions = {
                    "test": "function_to_call",
                    }

            model_name = self.settings['model'].split(":")
            model = model_name[1] + ":" + model_name[2]

            try:
                stream = olclient.chat(
                        model = model,
                        messages = message,
                        stream = streaming,
                        #format = "json",
                        options = { "num_ctx": num_ctx },
                        tools=[],
                        )

            except Exception as e:
                log(e)
                return None

            if self.suppress is True:
                response = stream['message']['content']
            else:
                if streaming:
                    for chunk in stream:
                        response += chunk['message']['content']
                        if markdown is True:
                            print(chunk['message']['content'], end="")
                        else:
                            print(chunk['message']['content'], end='', highlight=False, style="grey89")
                else:
                    response = stream['message']['content']
                    if markdown is True:
                        print(Markdown(response))
                    else:
                        print(response)

            """
            if stream.message.tool_calls:
            # There may be multiple tool calls in the response
                for tool in stream.message.tool_calls:
                    # Ensure the function is available, and then call it
                    if function_to_call := available_functions.get(tool.function.name):
                        print('Calling function:', tool.function.name)
                        print('Arguments:', tool.function.arguments)
                        print('Function output:', function_to_call(**tool.function.arguments))
                    else:
                      print('Function', tool.function.name, 'not found')
            """
        elif self.settings['model'].startswith("groq"):
            model_name = self.settings['model'].split(":")
            model = model_name[1]

            try:
                stream = grclient.chat.completions.create(
                        model = model,
                        messages = message,
                        stream = stream,
                        )
            except Exception as e:
                log(e)
                return None

            if self.suppress is True:
                response = stream.choices[0].message.content
            else:
                if streaming:
                    for chunk in stream:
                        if chunk.choices[0].delta.content is not None:
                            response += chunk.choices[0].delta.content
                            if markdown is True:
                                print(chunk.choices[0].delta.content, end="")
                            else:
                                print(chunk.choices[0].delta.content, end='')
                else:
                    response = stream.choices[0].message.content
                    if markdown is True:
                        print(Markdown(response))
                    else:
                        print(response)

        self.writeHistory('assistant', response)

        if self.settings['speech'] and self.suppress is False:
            speech = Speech()
            speech_thread = threading.Thread(target=speech.say, args=(response,))
            speech_thread.start()

        self.suppress = False

        return response

    def truncate_history(self, history, size):
        tokens = self.estimate_token_count(json.dumps(history))
        while tokens > size:
            history.pop(0)
            tokens = self.estimate_token_count(json.dumps(history))

        return history

    def process_commands(self, user_input):
        def _registerCommand(command=None):
            if command is not None: 
                self.command_register.append(command)

        def _isCommand(command=None):
            is_command = None
            command_pattern = r"(^|\b| )(?P<command_name>(\$|\w+)):(?P<content>.*?):($|\b| )"
            for match in re.finditer(command_pattern, input_text):
                command_name = match.group("command_name")

                check = f"{command_name}::"
                if check in self.command_register:
                    is_command = True
                else:
                    is_command = None
                    log(f"Error command invalid: {check}")
                    return None

        def _checkCommand(input_text=None):
            if input_text is None:
                return None

            input_text = input_text.strip() 
            command_pattern = r"(^|\b| )(?P<command_name>(\$|\w+)):(?P<content>.*?):($|\b| )"

            # Lists to store the results
            commands = []
            surrounding_texts = []
            last_end = 0

            # Find all command matches
            for match in re.finditer(command_pattern, input_text):
                # Extract command name and content
                command_name = match.group("command_name")

                # Checking if this is a valid command.
                '''
                check = f"{command_name}::"
                if check not in self.command_register:
                    log(f"Error invalid command: {check}")
                    return None
                '''

                if match.group("content"):
                    content = match.group("content")
                else:
                    content = None

                # Append command info
                commands.append({
                    "command_name": command_name,
                    "content": content,
                    "start": match.start(),
                    "end": match.end()
                })

                # Capture surrounding text before each command
                start, end = match.span()
                if start > last_end:
                    surrounding_texts.append(input_text[last_end:start].strip())

                last_end = end

            # Append any remaining text after the last command
            if last_end < len(input_text):
                surrounding_texts.append(input_text[last_end:].strip())

            # If there is no surrounding text, return None
            if not any(surrounding_texts):
                return None
            else:
                return True

            # Process each command and replace it in the input_text
            offset = 0  # Track character offset for replacement
            for command in commands:
                command_name = command["command_name"]
                content = command["content"]

                # Attempt to call the function dynamically
                try:
                    func = command_functions.get(f"command_{command_name}")
                    if callable(func):
                        log(f"Calling function: {command_name} with content: {content}")
                        replacement_content = func(content)
                        replacement = f"\n```\n{replacement_content}\n```\n"

                        start = command["start"] + offset
                        end = command["end"] + offset
                        input_text = input_text[:start] + replacement + input_text[end:]

                        offset += len(replacement) - (end - start)
                    else:
                        log(f"No function found for command: {command_name}")
                except Exception as e:
                    log(f"Error calling function {command_name}: {e}")

            return intput_text


        # Audio keyword triggers
        for keyword in self.audio_triggers:
            if re.search(self.audio_triggers[keyword][0], user_input):
                user_input = self.audio_triggers[keyword][1]
                break

        _registerCommand("test::")
        if user_input.startswith('test::'):
            test_text = """
            Send funds to support@example.org
            Visit https://example.com for more info.
            Check the config at /etc/config/settings.ini or C:\\Windows\\System32\\drivers\\etc\\hosts.
            Here is some JSON: {"key": "value"} or an array: [1, 2, 3].
            IPv4: 192.168.1.1, IPv6: fe80::1ff:fe23:4567:890a, MAC: 00:1A:2B:3C:4D:5E.
            Timestamp: 2023-11-18T12:34:56Z, Hex: 0x1A2B3C, Env: $HOME or %APPDATA%.
            UUID: 550e8400-e29b-41d4-a716-446655440000
            """

            print(Panel(test_text, title="test"))
            time.sleep(5)
            print()
            print("[red]hello world[/red]")
            print(test_text)
            time.sleep(5)
            with console.capture() as capture:
                console.print(test_text)

            ansii_text = capture.get()
            for i in range(50):
                test_text += "Test the length\n"
            self.display_pager(test_text)
            self.layout_pager(test_text)
            return None

        _registerCommand("perifious::")
        if re.search(r'^perifious::', user_input):
            speech = Speech(debug=self.settings['debug'])
            speech.say('Your wish is my command!')
            if self.settings['perifious']:
                user_input = 'settings:perifious:0:'
            else:
                user_input = 'settings:perifious:1:'


        _registerCommand("shell::")
        if re.search(r'^shell::', user_input):
            if self.shell_mode is False:
                self.shell_mode = True
                self.ps.style = self.theme_manager.get_theme("liveshell")
            else:
                self.shell_mode = False
                self.ps.style = self.theme_manager.get_theme(self.settings["theme"])
            return None


        _registerCommand("help::")
        help_pattern = r"^help::|^help:(.*):"
        match = re.search(help_pattern, user_input)
        if match:
            short = True
            if match.group(1):
                short = False 
            self.display_help(short)
            return None


        _registerCommand("clear::")
        _registerCommand("reset::")
        if re.search(r"^clear::|^reset::", user_input):
            os.system('reset')
            return None


        _registerCommand("save::")
        if re.search(r"^save::", user_input):
            self.save_settings()
            return None


        _registerCommand("exit::")
        if re.search(r'^exit::', user_input):
            self.save_settings()
            sys.exit(0)


        # Trigger introspect::
        _registerCommand("introspect::")
        introspect_pattern = r"introspect::"
        match = re.search(introspect_pattern, user_input)
        if match:
            try:
                pass
            except Exception as e:
                print("An error occurred:", str(e))

            return None

        # Trigger to read clipboard contents
        _registerCommand("clipboard::")
        clipboard_pattern = r'clipboard::|clipboard:(.*):'
        match = re.search(clipboard_pattern, user_input)
        if match:
            import symbiote.sym_crawler as webcrawler
            contents = clipboard.pshort()
            if match.group(1):
                sub_command = match.group(1).strip()
                if sub_command == 'get':
                    if re.search(r'^https?://\S+', contents):
                        log(f"Fetching content from: {contents}")
                        crawler = webcrawler.WebCrawler(browser='chrome')
                        pages = crawler.pull_website_content(
                                url, search_term=None,
                                crawl=False, depth=None
                            )
                        crawler.close()
                        website_content = str()
                        if pages:
                            for md5, page in pages.items():
                                website_content += page['content']
                        else:
                            log(f"Unable to fetch data for {url}")

                        user_input = user_input[:match.start()] + website_content + user_input[match.end():]
            return user_input


        # Trigger to reload modules
        _registerCommand("reload::")
        reload_pattern = r"reload::|reload:(.*):"
        match = re.search(reload_pattern, user_input)
        if match:
            if match.group(1):
                module = match.group(1)
                if module in list(sys.modules.keys()):
                    name = syss.modules.get(module)
                    log(f"Reloading {module}")
                    importlib.reload(module)
                else:
                    log(f"No such module: {module}")
            else:
                for module_name in list(sys.modules.keys()):
                    if module_name.startswith("symbiote."):
                        del module_name
                        module = sys.modules.get(module_name)
                        log(f"Reloading {module}")
                        importlib.reload(module)
            return None


        # Trigger to choose role
        _registerCommand("roles::")
        _registerCommand("role::")
        role_pattern = r'^roles?::|roles?:(.*):'
        match = re.search(role_pattern, user_input)
        if match:
            available_roles = Roles.get_roles()
            if match.group(1):
                selected_role = match.group(1).strip()
                selected_role = selected_role.upper()
            else:
                if not available_roles:
                    return None
                role_list = []
                for role_name in available_roles:
                    role_list.append(role_name)
                print(f"Current Role: {self.settings['role']}")
                selected_role = self.list_selector("Select a role:", sorted(role_list))
                if selected_role is None:
                    return None
            if selected_role in available_roles:
                self.settings['role'] = selected_role 
                self.save_settings()
            else:
                log(f"No such role: {selected_role}")
            return None


        # Trigger to display settings
        _registerCommand("settings::")
        setting_pattern = r'^settings::|settings:(.*):(.*):'
        match = re.search(setting_pattern, user_input)

        if match:
            # Extract setting key and value
            setting = match.group(1).lower() if match.group(1) else None
            set_value = match.group(2) if match.group(2) else None

            if setting:
                if setting in self.settings:
                    # Get the current type of the setting
                    current_type = type(self.settings[setting])

                    # Handle boolean settings
                    if isinstance(self.settings[setting], bool):
                        set_value = str(set_value).lower() not in ("false", "0", "off")

                    # Special handling for the "location" setting
                    if setting == "location":
                        try:
                            # Attempt to query postal code first
                            prior = set_value.lower()
                            self.geo = nomi.query_postal_code(set_value)
                            set_value = str(self.geo['postal_code']).lower()

                            # If no change, try querying location details
                            if set_value == prior:
                                tmp = nomi.query_location(set_value, top_k=1)
                                set_value = tmp['postal_code'].iloc[0]
                                self.geo = nomi.query_postal_code(set_value)
                        except Exception as e:
                            log(f"Error processing location: {e}")
                            return None

                    # Update the setting and save
                    self.settings[setting] = current_type(set_value)
                    self.save_settings()
                    return None

            # Display settings if no valid setting key was found
            self.display_settings()
            return None


        # Trigger for changing gpt model 
        _registerCommand("model::")
        model_pattern = r'^model::|model:(.*):'
        match = re.search(model_pattern, user_input)
        if match:
            if match.group(1):
                model_name = match.group(1).strip()
                self.selectModel(model_name)
            else:
                self.selectModel()
            return None 


        # Trigger for changing the conversation file
        _registerCommand("convo::")
        convo_pattern = r'^convo::|convo:(.*):'
        match = re.search(convo_pattern, user_input)
        if match:
            if match.group(1):
                convo_name = match.group(1).strip()
                self.display_convo(convo_name) 
            else:
                self.display_convo()
            return None 


        # Trigger for changing working directory in chat
        _registerCommand("cd::")
        cd_pattern = r'^cd::|cd:(.*):'
        match = re.search(cd_pattern, user_input)
        if match:
            if match.group(1):
                requested_directory = match.group(1).strip()
            else:
                requested_directory = '~'

            if requested_directory == '-':
                requested_directory = self.previous_directory

            requested_directory = os.path.abspath(os.path.expanduser(requested_directory))
            if os.path.exists(requested_directory):
                self.previous_directory = self.working_directory
                self.working_directory = requested_directory 
                os.chdir(self.working_directory)
            else:
                log(f"Directory does not exit: {requested_directory}")
                return None

            return None 


        # Trigger to list verbal keywords prompts.
        _registerCommand("keywords::")
        keywords_pattern = r'^keywords::'
        match = re.search(keywords_pattern, user_input)
        if match:
            for keyword in self.audio_triggers:
                if keyword == 'perifious':
                    continue
                print(f'trigger: {self.audio_triggers[keyword][0]}')

            return None 

        # Trigger for extract:: processing.
        _registerCommand("extract::")
        summary_pattern = r'^extract::|^extract:(.*):(.*):|^extract:(.*):'
        match = re.search(summary_pattern, user_input)
        if match:
            if match.group(1):
                file_path = match.group(1)
                screenshot_pattern = r'^screenshot$'
                if re.search(screenshot_pattern, file_path):
                    file_path = capture_screen()
                    index = True

            if match.group(2):
                reindex = match.group(2)
                if reindex.lower() == ("1" or "true"):
                    reindex = True
                else:
                    reindex = False

            if file_path is None:
                file_path = self.file_selector()

            file_path = os.path.expanduser(file_path)

            if os.path.isfile(file_path):
                metadata = extract_metadata(file_path)
                if metadata:
                    content = extract_text(file_path)

                if content:
                    metadata["contents"] = content

                self.memory.create("extract_command", metadata)
            else:
                log(f"Invalid file: {file_path}")

            log(f"File details stored: key: extract_command")
            return None 


        # Trigger to flush current running conversation from memory.
        _registerCommand("flush::")
        flush_pattern = r'^flush::'
        match = re.search(flush_pattern, user_input)
        if match:
            self.flush_history()
            return None 

        # Trigger for history::. Show the history of the messages.
        _registerCommand("history::")
        history_pattern = r'^history::|^history:(.*):'
        match = re.search(history_pattern, user_input)
        if match:
            if match.group(1):
                history_length = int(match.group(1))
                print(history_length)
                time.sleep(4)
            else:
                history_length = False 
           
            history = str()
            for line in self.conversation_history:
                history += f"role: {line['role']}\n{line['content']}\n"
                print(f"role: {line['role']}")
                print(Markdown(line['content']))
                print()

            return None


        # Trigger for code:: extraction from provided text
        _registerCommand("code::")
        code_pattern = r'code::|code:(.*):'
        match = re.search(code_pattern, user_input)
        if match:
            log(f"Disabled...")
            return None
            codeRun = False
            if match.group(1):
                text = match.group(1)
                codeidentify = CodeIdentifier.analyze(text)
                print(codeidentify)
                return
            else:
                # process the last conversation message for code to extract
                last_message = self.conversation_history[-1]
                print(last_message['content'])
                codeidentify = codeextract.CodeBlockIdentifier(last_message['content'])

            files = codeidentify.process_text()
            for file in files:
                print(file)

            if codeRun:
                pass

            return files


        # Trigger for note:: taking.
        _registerCommand("note::")
        note_pattern = r'^note::|^note:([\s\S]*?):'
        match = re.search(note_pattern, user_input)
        if match:
            if match.group(1):
                user_input = match.group(1)

            self.save_conversation(user_input, self.settings['notes'])

            return None


        # Trigger menu for cli theme change
        _registerCommand("theme::")
        theme_pattern = r'theme::|theme:(.*):'
        match = re.search(theme_pattern, user_input)
        if match:
            if match.group(1):
                theme_name = match.group(1)
                prompt_style = self.theme_manager.get_theme(theme_name)
            else:
                theme_name, prompt_style = self.theme_manager.select_theme() 

            self.ps.style = prompt_style
            self.settings['theme'] = theme_name
            self.save_settings()

            return None 


        # trigger terminal image rendering view:: 
        _registerCommand("view::")
        view_pattern = r'view::|^view:(.*):|^view:(https?:\/\/\S+):'
        match = re.search(view_pattern, user_input)
        file_path = None

        if match:
            if match.group(1):
                file_path = match.group(1)
            else:
                file_path = self.file_selector('File name:')
                #file_path = self.file_browser()

            file_path = clean_path(file_path)
            if is_image(file_path):
                content = imageToAscii(file_path)
                p(content)
                return None
            elif is_url(file_path):
                import symbiote.sym_crawler as webcrawler
                crawler = webcrawler.WebCrawler(browser='chrome')
                pages = crawler.pull_website_content(
                        file_path, search_term=None,
                        crawl=False, depth=None
                    )
                crawler.close()
                if pages:
                    content = ""   
                    css = ""
                    script = ""
                    link_list = []
                    for md5, item in pages.items():
                        content += item['content']
                        link_list = link_list + item['links']
                        css += "\n".join(item['css'])
                        script += "\n".join(item['scripts'])
                    links = "\n".join(link_list)
                else:
                    log(f"No content gathered for {url}")
                    return None
            else:
                content = extract_text(file_path)
                print(content)

            return None


        # Trigger image analysis and reporting analyse_image::
        _registerCommand("analyze_image::")
        analyze_image_pattern = r'^analyze_image::|^analyze_image:(.*):'
        match = re.search(analyze_image_pattern, user_input)

        if match:
            if match.group(1):
                image_path = match.group(1)
            else:
                #image_path = self.file_selector('Image path:')
                image_path = self.file_browser()
                image_path = os.path.expanduser(image_path)
                image_path = os.path.abspath(image_path)

            self.spinner.start()
            import symbiote.ImageAnalysis as ia
            extractor = ia.ImageAnalyzer(detection=True, extract_text=True, backend='mtcnn') 
            results = extractor.analyze_images(image_path, mode='none')
            self.spinner.succeed('Completed')
            human_readable = extractor.render_human_readable(results)
            print(human_readable)

            content = f"Analyze the following details collected about the image or images and summarize the details.\n{human_readable}\n"

            return content


        # Trigger to find files by search find::
        _registerCommand("find::")
        find_pattern = r'^find::|^find:(.*):'
        match = re.search(find_pattern, user_input)
        if match:
            if match.group(1):
                pattern = match.group(1)
                result = self.find_files(pattern)
                return None

            result = self.find_files()   

            return None


        # Trigger to init scrolling
        _registerCommand("scroll::")
        scroll_pattern = r'scroll::|scroll:(.*):'
        match = re.search(scroll_pattern, user_input)
        if match:
            file_path = None
            if match.group(1):
                file_path = match.group(1)

            #file_path = self.file_selector("File name:")
            file_path = self.file_browser()
            print(file_path)

            if file_path is None:
                return None

            file_path = os.path.expanduser(file_path)
            absolute_path = os.path.abspath(file_path)

            scroll_content(absolute_path)

            return None


        # Trigger for wikipedia search wiki::
        _registerCommand("wiki::")
        wiki_pattern = r'wiki:(.*):'
        match = re.search(wiki_pattern, user_input)
        if match:
            if match.group(1):
                import symbiote.Wikipedia as wikipedia
                wiki = wikipedia.WikipediaSearch()
                search_term = match.group(1)
                results = wiki.search(search_term, 5)
                results_str = str()
                for result in results:
                    results_str += result['text']

                print(Panel(Text(results_str[:8000]), title=f"Wikipedia: {search_term}"))
                content = f"wikipedia search\n"
                content += '\n```\n{}\n```\n'.format(results_str)
                user_input = user_input[:match.start()] + content + user_input[match.end():]

                return user_input
            else:
                log("No search term provided.")
                return None


        # Trigger for headline analysis
        _registerCommand("news::")
        _registerCommand("headlines::")
        news_pattern = r'\bnews::|\bheadlines::'
        match = re.search(news_pattern, user_input)
        if match:
            import symbiote.headlines as hl
            gh = hl.getHeadlines()
            result = gh.scrape()
            print(Panel(Text(result), title=f"News Headlines"))
            content = f"Consolidate and summarize the following.\n"
            content += '\n```\n{}\n```\n'.format(result)
            user_input = user_input[:match.start()] + content + user_input[match.end():]

            return user_input


        # Trigger for google search or dorking
        _registerCommand("google::")
        google_pattern = r'google:(.*):'
        match = re.search(google_pattern, user_input)
        if match:
            from symbiote.sym_google import GoogleSearch
            if match.group(1):
                search_term = match.group(1)
            else:
                search_term = get_display_text("Search term>")

            if not search_term:
                log("No search term provided.")
                return None
            else:
                search = GoogleSearch()
                search_results = search.google_search(search_term)
                search.display_google_search_results(search_results)
                self.memory.create(
                        "google_command",
                        search_results
                    )


                log(f"Results saved to memory key:google_")
                return None

        # Trigger for define::
        _registerCommand("define::")
        define_pattern = r'define:(.*):'
        match = re.search(define_pattern, user_input)
        if match:
            if match.group(1):
                term = match.group(1)
                user_input = f"Provide detailed definition for the word {term}" 

                return user_input


        # Trigger for imap mail checker mail::
        _registerCommand("mail::")
        mail_pattern = r'mail::'
        match = re.search(mail_pattern, user_input)
        if match:
            import symbiote.GetEmail as mail
            mail_checker = mail.MailChecker(
                    username=self.settings['imap_username'],
                    password=self.settings['imap_password'],
                    mail_type='imap',
                    days=2,
                    unread=False,
                    #model=self.settings['model'],
                    model=None,
                    )
            email = mail_checker.check_mail()

            content = [] 
            for key, val in enumerate(email):
                content.append(f"Message: {key}")
                content.append(f"\tDate: {val['date']}")
                content.append(f"\tFrom: {val['from']}")
                content.append(f"\tTo: {val['to']}")
                content.append(f"\tSubject: {val['subject'][:30]}")
                #content.append(f"\tBody Size: {len(val['body'])} chars")
                content.append("")
                #content.append(f"\tBody:\n\t\t{val['body'][:]}\n")

            content = "\n".join(content)

            if email is None:
                log(f"No emails to analyze.")
                return None

            review = json.dumps(email)
            prompt = f"Create a report from the following in a nice format."
            review = f"{prompt}\n```json\n{review}\n```\n"
            print(Panel(Text(content), title=f"Emails: {self.settings['imap_username']}"))
            print()
            user_input = user_input[:match.start()] + review + user_input[match.end():]

            return user_input


        # Trigger for w3m web browser functionality browser::
        _registerCommand("w3m::")
        _registerCommand("browser::")
        browser_pattern = r'w3m:(.*):|browser:(.*):'
        match = re.search(browser_pattern, user_input)
        if match:
            if match.group(1):
                url = match.group(1)
                self.open_w3m(url)
            else:
                self.open_w3m()

            return None


        # Trigger to extract images from a url image_extract::
        _registerCommand("image_extract::")
        image_extract_pattern = r'^image_extract:(.*):'
        match = re.search(image_extract_pattern, user_input)
        if match:
            if match.group(1):
                url = match.group(1)
                self.url_image_extract(url)
            else:
                log('No url specified.')

            return None


        # Trigger for qr code generation qr::
        _registerCommand("qr::")
        qr_pattern = r"qr:(.*):"
        match = re.search(qr_pattern, user_input)
        if match:
            if match.group(1):
                content = match.group(1)
                self.generate_qr_terminal(content)
            else:
                log("No content provided for the qr.")

            return None

        
        # Trigger for weather::
        _registerCommand("weather::")
        weather_pattern = r"weather::|weather:(.*):"
        match = re.search(weather_pattern, user_input)
        if match:
            if match.group(1):
                location = match.group(1)
            else:
                location = self.settings['location']

            if location is None:
                log("No location set in settings::")
                return None

            result = self.get_weather(location)

            if result is None:
                log(f"Unable to get weather for {location}")
                return None

            weather = json.dumps(result['current_condition'], indent=4) 
            self.memory.create("weather_command", weather)

            print(Panel(Text(weather), title=f"Weather: {location}"))
            if _checkCommand(user_input) is None:
                log(f"Results written to history")
                self.writeHistory("user", weather)
                return None

            content = f"Analyze the following weather details and provide a well formatted weather report for {location}.\n```json\n{weather}```"
            user_input = user_input[:match.start()] + content + user_input[match.end():]

            return user_input

        
        # Trigger for inspect:: command to inspect running python objects.
        _registerCommand("inspect::")
        inspect_pattern = r'inspect::|inspect:(.*):'
        match = re.search(inspect_pattern, user_input)
        if match:
            if match.group(1):
                obj = match.group(1)
            else:
                all_objs = list({**globals(), **locals()}.keys())
                obj = self.list_selector("Objects:", all_objs)

            if obj:
                inspector = Inspect(obj)
                report = inspector.generate_report(output='render')
                content = json.dumps(report, indent=4)
                self.memory.create("inspect_command", content)

                if _checkCommand(user_input) is None:
                    log(f"Results written to history")
                    self.writeHistory("user", content)
                    return None

                del inspector 
            else:
                log(f"No object selected.")

            return None


        # Trigger for memget:: management
        _registerCommand("memget::")
        memget_pattern = r"memget::|memget:(.*):"
        match = re.search(memget_pattern, user_input)
        if match:
            json_dat = None
            if match.group(1):
                getobj = match.group(1)
            else:
                getobj = self.text_prompt("Object>") 

            if getobj is None or getobj == "":
                log(f"Empty object requeted.")
                return None

            results = self.memory.read(getobj)

            if isinstance(results, str):
                json_data = results
            else:
                json_data = json.dumps(results, indent=4)

            if json_data:
                print(Panel(Text(json_data[:5000]), title=f"Content: {getobj}"))

                if _checkCommand(user_input) is None:
                    log(f"Results written to history")
                    self.writeHistory("user", json_data)
                    return None

                content = f"\n```\n{json_data}\n```"
                user_input = user_input[:match.start()] + content + user_input[match.end():]

                return user_input

            return None


        # Trigger for memory:: management
        _registerCommand("memory::")
        memory_pattern = r"^memory::|^memory:(.*):"
        match = re.search(memory_pattern, user_input)
        if match:
            if match.group(1):
                info = match.group(1)

            inspect(self.memory)
            return None

        
        # Trigger for search:: on memory 
        _registerCommand("search::")
        search_pattern = r"search::|search:(.*):"
        match = re.search(search_pattern, user_input)
        if match:
            if match.group(1):
                search_term = match.group(1)
            else:
                search_term = self.text_prompt("Search term|regex>") 

            text_result = self.get_search_results(search_term)

            if text_result:
                self.memory.create("search_command", text_result)
                if _checkCommand(user_input) is None:
                    log(f"Results written to history")
                    self.writeHistory("user", text_result)
                    return None
                else:
                    content = f"```json\n{text_result}\n```"
                    user_input = user_input[:match.start()] + content + user_input[match.end():]
                    return user_input

            return None

        
        # Trigger for toolbar::
        _registerCommand("toolbar::")
        if user_input.startswith("toolbar:"):
            content = user_data.split(":")
            if len(content) > 1:
                self.live_render_buffer = content[1]

            return None

        
        # Trigger for file:: processing.
        _registerCommand("file::")
        file_pattern = r'file::|file:(.*):'
        match = re.search(file_pattern, user_input)
        if match: 
            file_path = None
            content = None
            metadata = None
            if match.group(1):
                file_path = match.group(1)
            else:
                #file_path = self.file_browser()
                file_path = self.file_selector('File name:')

            if file_path is None:
                log(f"No such file or directory: {file_path}")
                return None 
            
            file_path = os.path.abspath(os.path.expanduser(file_path))

            if os.path.isfile(file_path):
                content = extract_text(file_path)
                metadata = extract_metadata(file_path)
                ci = CodeIdentifier()
                code_check = ci.analyze(content)
                metadata["is_code_file"] = code_check["is_code_file"]
                metadata["has_code"] = code_check["has_code"]

                if metadata:
                    self.memory.create("file_command_metadata", metadata)
                    metadata["readable_strings"] = str(len(metadata["readable_strings"]))
                    self.display_object(metadata)
                    log(f"memory key created: file_command_metadataa")

                if content:
                    self.memory.create("file_command_content", content)
                    log(f"memory key created: file_command_content")

                if _checkCommand(user_input) is None:
                    #print(Panel(displayed, title=f"File: {file_path}"))
                    return None

                file_content = f'\n```file name: {file_path}\n{content}\n```\n'
                user_input = user_input[:match.start()] + file_content + user_input[match.end():]
                return user_input

            elif os.path.isdir(file_path):
                log(f"Directory crawling is temporarily disabled due to memory consumption.")
                return None
                '''
                content = extractDirText(file_path)
                if content is None:
                    log(f"No content found in directory: {file_path}")
                    return None
                print(Panel(display, title=f"Content: {file_path}"))
                user_input = user_input[:match.start()] + dir_content + user_input[match.end():]
                '''
            return None

        
        # Trigger image:: execution for AI image generation
        _registerCommand("image::")
        image_pattern = r'^image:([\s\S]*?):'
        match = re.search(image_pattern, user_input)
        if match:
            if match.group(1):
                query = match.group(1)
                self.flux_image_generator(query)
            else:
                log(f"No image description provided.")

            return None

        
        # Trigger system execution of a command
        _registerCommand("$")
        exec_pattern = r'\$:(.*):'
        match = re.search(exec_pattern, user_input)
        if match:
            if match.group(1):
                command = match.group(1)
                result = self.exec_command(command)

                if result:
                    self.memory.create("exec_command", result)

                    if _checkCommand(user_input) is None:
                        log(f"Results written to history")
                        self.writeHistory('user', result)
                        print(Panel(Text(result), title=f"Command: {command}"))
                        return None

                    print(Panel(Text(result), title=f"Command: {command}"))
                    content = f"\n```{command}\n{result}\n```\n"
                    user_input = user_input[:match.start()] + content + user_input[match.end():]
                    return user_input
            else:
                log(f"No commands specified.")
                return None

        
        # Trigger for getip::
        _registerCommand("getip::")
        getip_pattern = r'getip::'
        match = re.search(getip_pattern, user_input)
        if match:
            ipinfo = self.get_network_data()
            report = self.iface_report(ipinfo)
            self.memory.create("getip_command", report)

            print(Panel(Text(report), title=f"Network Info:"))

            content = str()
            content = f"\n```network_info\n{ipinfo}\n```"
            
            if _checkCommand(user_input) is None:
                log(f"Results written to history")
                self.writeHistory('user', content)
                return None

            user_input = user_input[:match.start()] + content + user_input[match.end():]

            return user_input

        
        # Trigger for get:URL processing. Load website content into user_input for model consumption.
        _registerCommand("get::")
        get_pattern = r'get::|get:(https?://\S+):'
        match = re.search(get_pattern, user_input)
        if match:
            crawl = False
            website_content = str() 
            if match.group(1):
                url = match.group(1)
            else:
                url = self.text_prompt("URL to load:")
            if url is None:
                log(f"No URL given: {url}")
                return None 

            log(f"Fetching {url}.")
            import symbiote.sym_crawler as webcrawler
            crawler = webcrawler.WebCrawler(browser='chrome')
            pages = crawler.pull_website_content(url, search_term=None, crawl=crawl, depth=None)
            crawler.close()

            if pages:
                content = str()
                css = str()
                script = str()
                links = str()
                link_list = []
                for md5, item in pages.items():
                    content += item['content']
                    link_list = link_list + item['links']
                    css += "\n".join(item['css'])
                    script += "\n".join(item['scripts'])

                links += "\n".join(link_list)
            else:
                log(f"No content gathered for {url}")
                return None

            self.memory.create("get_command",
                               {"content": content,
                                "links": link_list,
                                "css": css,
                                "scripts": script
                                })
            content = self.clean_text(content)
            css = self.clean_text(css)
            script = self.clean_text(script)
            #self.web_data_stats(self.memory.read("get_command"))
            self.display_object(self.memory.read("get_command"))
            """
            print(Panel(Text(f"Content: {url}")))
            print(Text(content[:1000]))
            print(Panel(Text(f"footer info)")))
            """
            '''
            if css:
                print(Panel(Text(css[:1000]), title=f"CSS: {url}"))
            if script:
                print(Panel(Text(script[:1000]), title=f"Scripts: {url}"))
            if len(links) > 0:
                print(Panel(Text(links), title=f"Links: {url}"))
            '''
            if _checkCommand(user_input) is None:
                log(f"Results written to history")
                self.writeHistory("user", f"URL: {url}\n" + content)
                return None
            else:
                content += f"```URL: {url}\n{content}\n```"
                user_input = user_input[:match.start()] + content + user_input[match.end():]
                return user_input

            return None 

        
        # Trigger for fake news analysis fake_news::
        _registerCommand("fake_news::")
        fake_news_pattern = r'\bfake_news::|\bfake_news:(.*):'
        match = re.search(fake_news_pattern, user_input)
        if match:
            data = None
            if match.group(1):
                data = match.group(1)
            else:
                data = self.text_prompt("URL or text to analyze:")

            if data is None:
                log(f"No data provided.")
                return None

            self.spinner.start()
            import symbiote.FakeNewsAnalysis as fake_news
            detector = fake_news.FakeNewsDetector()
            if is_url(data):
                text = detector.download_text_from_url(data)
            else:
                text = data

            result = detector.analyze_text(text)
            self.spinner.succeed('Completed')

            if result:
                output = json.dumps(result, indent=4)
                print(Panel(Text(output), title=f"FakeNew Analysis: {data}"))
                user_input = f"The following results are from a fake news analyzer.  Analyze the following json document and provide a summary and report of the findings.\n{result}"
            else:
                log(f"No results for {data}")
                return None
            
            return user_input

        
        # Trigger for downloading youtube transcripts yt_transcript::
        _registerCommand("yt_transcript::")
        yt_transcript_pattern = r'yt_transcript::|yt_transcript:(.*):'
        match = re.search(yt_transcript_pattern, user_input)
        if match:
            if match.group(1):
                yt_url = match.group(1)
            else:
                yt_url = self.text_prompt("Youtube URL:")

            if yt_url == None:
                log(f"No transciprts url set.")
                return None

            log(f"Fetching youtube transcript from: {yt_url}")

            import symbiote.YoutubeUtility as ytutil
            yt = ytutil.YouTubeUtility(yt_url)
            transcript = yt.get_transcript()
            if transcript:
                print(Panel(Text(transcript), title=f"Transcripts: {yt_url}"))
                user_input = user_input[:match.start()] + transcript + user_input[match.end():]
            else:
                log(f"No transcripts found.")
                return None

            return user_input

        
        # Trigger web vulnerability scan vscan::
        _registerCommand("vscan::")
        vscan_pattern = r'vscan::|vscan:(.*):'
        match = re.search(vscan_pattern, user_input)
        if match:
            if match.group(1):
                url = match.group(1)
            else:
                url = self.text_prompt("URL to scan:")

            if is_url(url):
                import symbiote.WebVulnerabilityScan as web_vuln
                scanner = web_vuln.SecurityScanner(headless=True, browser='chrome')
                scanner.scan(url)
                report = scanner.generate_report()
                self.memory.create("vscan_command", report)
                print(Panel(Text(report), title=f"Report: {url}"))
                
                if _checkCommand(user_input) is None:
                    log(f"Results written to history")
                    self.writeHistory("user", report)
                    return None

                user_input = user_input[:match.start()] + report + user_input[match.end():]
                return user_input

        
        # Trigger for textual deception analysis deception::
        _registerCommand("deception::")
        deception_pattern = r'deception::|deception:(.*):'
        match = re.search(deception_pattern, user_input)
        analysis_src = None
        if match:
            if match.group(1):
                analysis_src = match.group(1)
            else:
                analysis_src = self.text_prompt("Text or URL:")

            if analysis_src == None:
                log("No content to analyze.")
                return None
            
            self.spinner.start()
            import symbiote.DeceptionDetection as deception
            detector = deception.DeceptionDetector()
            results = detector.analyze_text(analysis_src)
            self.spinner.succeed('Completed')

            if results:
                self.memory.create("deception_command", results)
                content = json.dumps(results, indent=4)
                print(Panel(Text(content), title=f"Deception Analysis Summary"))

                if _checkCommand(user_input) is None:
                    log(f"Results written to history")
                    self.writeHistory("user", content) 
                    return None

                user_input = user_input[:match.start()] + report + user_input[match.end():]
                return user_input
            else:
                log("No results returned.")
                return None

        
        # Trigger for crawl:URL processing. Load website content into user_input for model consumption.
        _registerCommand("crawl::")
        crawl_pattern = r'crawl::|crawl:(https?://\S+):'
        match = re.search(crawl_pattern, user_input)
        if match:
            crawl = True
            website_content = str() 
            if match.group(1):
                url = match.group(1)
            else:
                url = self.text_prompt("URL to load:")
            
            if url == None:
                log(f"No URL specified.")
                return None 

            crawler = webcrawler.WebCrawler(browser='chrome')
            self.spinner.start()
            pages = crawler.pull_website_content(url, search_term=None, crawl=crawl, depth=None)
            crawler.close()
            self.spinner.succeed('Completed')
            for md5, page in pages.items():
                content += page['content']
            print(Panel(Text(content), title=f"Content: {url}"))
            user_input = user_input[:match.start()] + content + user_input[match.end():]
            print()
            return user_input 

        
        # Catchall for stray command entries. 
        catchall_pattern = r"^.*::?$"
        if re.search(catchall_pattern, user_input):
            log(f"Unknown command: {user_input}")
            return None

        return user_input


    def save_settings(self):
        try:
            with open(self.config_file, "w") as file:
                json.dump(self.settings, file, indent=4, sort_keys=True)
        except Exception as e:
            log(f"Error Writing: {e}")

        return None

    
    def load_settings(self):
        try:
            with open(self.config_file, "r") as file:
                settings = json.load(file)
        except Exception as e:
            log(f"Error Reading: {e}")
            return None
        print(settings)
        return settings

    def create_dialog(self, title, text):
        message_dialog(
            title=title,
            text=text).run()

    def file_selector(self, message, start_path='./'):
        result = inquirer.filepath(
                message=message,
                default=start_path,
                #validate=PathValidator(is_file=True, message="Input is not a file"),
                wrap_lines=True,
                mandatory=False,
            ).execute()
        return result

    def list_selector(self, message, selection):
        result = inquirer.select(
                message=message,
                choices=selection,
                mandatory=False).execute()
        return result 

    def text_prompt(self, message):
        result = inquirer.text(
                message=message,
                mandatory=False,
            ).execute()
        return result

    def yes_no_prompt(self, question="Continue?"):
        answer = inquirer.confirm(message=question, default=False).execute()
        return answer
            
    def find_files(self, pattern=None):
        # Recursively get a list of all files from the current directory
        all_files = []
        for root, dirs, files in os.walk('.'):
            # Filter out hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            # Filter out hidden files
            files = [f for f in files if not f.startswith('.')]
            # Collect full paths of remaining files
            for f in files:
                full_path = os.path.join(root, f)
                all_files.append(full_path)

        if pattern is None:
            # Prompt user for a pattern (regex)
            pattern = prompt("Enter a pattern (regex) to search for files: ")

        try:
            # Filter files based on the regex pattern
            matching_files = []
            for file in all_files:
                if re.search(pattern, file):
                    matching_files.append(file)

            if len(matching_files) > 0:
                selected_file = self.list_selector("Matching files:", sorted(matching_files))
                return selected_file
            else:
                log(f"No matching file found for: {pattern}")
                return None

        except re.error:
            log("Invalid regex pattern!")

    def load_conversation(self):
        return
        self.conversations_file = conversations_file
        data = []

        if os.path.exists(self.conversations_file):
            try:
                with open(conversations_file, 'r') as file:
                    for line in file:
                        data.append(json.loads(line))

            except Exception as e:
                log("Error: opening %s: %s" % (conversations_file, e))
                sys.exit(10)

        return data

    def save_conversation(self, role, text):
        ''' Save conversation output to loaded conversation file '''
        json_conv = {
                "epoch": time.time(),
                "role": role,
                "content": text 
                }

        jsonl_string = json.dumps(json_conv)

        with open(self.conversations_file, 'a+') as file:
            #json.dump(data, file, indent=2)
            file.write(jsonl_string + "\n")

    def estimate_token_count(self, text):
        """
        1,000 tokens	 4,000 characters	    Short queries or summaries, single-topic prompts, or brief Q&A.
        2,400 tokens	 9,600 characters	    Small documents, single-page summaries, basic explanations.
        8,000 tokens 	 32,000 characters	    Medium-length documents, detailed Q&A, or multi-paragraph context.
        16,000 tokens	 64,000 characters	    Comprehensive analyses, lengthy articles, or multiple sections.
        32,000 tokens	 128,000 characters	    Long documents, in-depth reports, or multiple complex topics.
        64,000 tokens	 256,000 characters	    Book chapters, extended research papers, large data collections.
        128,000 tokens	 512,000 characters	    Entire books, extensive document collections, or archival text.
        256,000 tokens	 1,024,000 characters	Very large datasets, multi-book volumes, or thorough archive analysis.
        512,000 tokens	 2,048,000 characters	Collections of books or structured data, such as encyclopedias.
        1,000,000 tokens 4,000,000 characters	Massive corpora, extensive archives, or enterprise-scale datasets.
        On average a token = 4 characters
        """
        token_count = len(text)
        return round(token_count)

    def clean_text(self, text):
        # Remove leading and trailing whitespace
        text = text.strip()


        # Remove any non-ASCII characters (optional, based on your needs)
        text = re.sub(r'[^\x00-\x7F]+', '', text)

        # Normalize dashes and remove unnecessary punctuation
        #text = re.sub(r'[–—]', '-', text)  # Normalize dashes
        #text = re.sub(r'[“”]', '"', text)  # Normalize quotes
        #text = re.sub(r"[‘’]", "'", text)  # Normalize apostrophes

        # Optionally remove or replace other special characters
        # You can customize this step according to your needs
        # For example, remove non-alphanumeric except common punctuation
        #text = re.sub(r'[^\w\s.,!?\'"-]', '', text)

        # Further replace double punctuation (optional)
        text = re.sub(r'\.{2,}', '.', text)  # Replace multiple dots with a single dot

        # Replace multiple spaces with a single space
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n+', '\n', text)

        #text = text.lower()

        return text

    def image_to_base64(self, image_path):
        # Open the image
        try:
            with Image.open(image_path) as img:
                # Convert to PNG format if not already in JPEG, JPG, or PNG
                if img.format not in ['JPEG', 'JPG', 'PNG']:
                    with BytesIO() as output:
                        img.save(output, format="PNG")
                        png_data = output.getvalue()
                else:
                    with BytesIO() as output:
                        img.save(output, format=img.format)
                        png_data = output.getvalue()

            # Encode the PNG image to base64
            png_base64 = base64.b64encode(png_data).decode('utf-8')
        except Exception as e:
            log(f"Error processing image: {e}")
            return None

    def describe_image(self, image_path):
        try:
            encoded_image = self.image_to_base64(image_path)
        except:
            return None

        # Call the Ollama API with the LLaVA model
        try:
            response = ollama.generate(
                model='minicpm-v',
                prompt=prompt,
                images=[encoded_image]
            )

            # Extract the generated text from the response
            detected_objects = response['response']
            return detected_objects
        except Exception as e:
            log(f"Error processing image: {e}")
            return None

    def python_tool(self, code):
        # Start the Python subprocess in interactive mode
        python_interpreter = sys.executable 
        child = pexpect.spawn(python_interpreter, ['-i'], encoding='utf-8')

        # Wait for the initial prompt
        child.expect('>>> ')

        # Initialize a string to store the interaction log
        interaction_log = str()

        # Split the code into lines
        lines = code.strip().split('\n')

        # Iterate through each line of code and send it to the subprocess
        for i, line in enumerate(lines):
            try:
                # Send the line to the Python interpreter
                child.sendline(line)

                # Log the input line
                interaction_log += f">>> {line}\n"

                # Expect and capture all outputs until the next primary prompt
                while True:
                    index = child.expect([r'>>> ', r'\.\.\. ', pexpect.EOF])
                    interaction_log += child.before.strip() + "\n"
                    if index == 0:  # '>>> ' prompt indicates primary prompt
                        print(child.before.strip())
                        break
                    elif index == 1:  # '... ' prompt indicates continuation
                        print(child.before.strip())
                        # If this is the last line, send an empty line to execute the block
                        if i == len(lines) - 1:
                            child.sendline('')
                    elif index == 2:  # EOF indicates the end of the output
                        print(child.before.strip())
                        return interaction_log.strip()
            except pexpect.EOF:
                print(child.before.strip())
                interaction_log += child.before.strip() + "\n"
                break
            except pexpect.ExceptionPexpect as e:
                log(f"Error executing line '{line}': {e}")
                interaction_log += f"Error executing line '{line}': {e}\n"
                break
       
        # Close the child process
        child.close()

        return interaction_log.strip()

    def flux_image_generator(self, query_text):
        self.spinner.start()
        # Get the API key from environment variables
        api_key = os.getenv("HUGGINGFACE_API_KEY")

        # Set up the API URL and headers
        API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
        headers = {"Authorization": f"Bearer {api_key}"}

        def query(payload):
            response = requests.post(API_URL, headers=headers, json=payload)
            return response.content

        image_bytes = query({
            "inputs": query_text,
        })

        # You can access the image with PIL.Image for example
        image = Image.open(BytesIO(image_bytes))
        self.spinner.succeed('Completed')

        # Open the image for viewing
        image.show()

    def generate_qr(self,
        text: str,
        center_color: str = "#00FF00",
        outer_color: str = "#0000FF",
        back_color: str = "black",
        dot_size: int = 10,
        border_size: int = 10
    ):
        # Create a QR code object
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,  # Higher error correction
            box_size=dot_size,
            border=4,
        )

        # Add data to the QR code
        qr.add_data(text)
        qr.make(fit=True)

        # Get the QR code matrix
        qr_matrix = qr.get_matrix()
        qr_size = len(qr_matrix)

        # Create a new image with a background color
        img_size = qr_size * dot_size + 2 * border_size
        img = Image.new("RGBA", (img_size, img_size), back_color)
        draw = ImageDraw.Draw(img)

        # Calculate the gradient
        def getGradientColor(x, y, center_x, center_y, inner_color, outer_color, max_dist):
            # Distance from the center
            dist = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
            ratio = min(dist / max_dist, 1)

            # Interpolating the color
            r1, g1, b1 = inner_color
            r2, g2, b2 = outer_color

            r = int(r1 + (r2 - r1) * ratio)
            g = int(g1 + (g2 - g1) * ratio)
            b = int(b1 + (b2 - b1) * ratio)

            return (r, g, b)

        # Center of the QR code
        center_x = img_size // 2
        center_y = img_size // 2
        max_dist = ((center_x) ** 2 + (center_y) ** 2) ** 0.5

        # Convert hex colors to RGB
        center_color_rgb = ImageColor.getrgb(center_color)
        outer_color_rgb = ImageColor.getrgb(outer_color)

        # Draw the QR code with gradient dots
        for y in range(qr_size):
            for x in range(qr_size):
                if qr_matrix[y][x]:
                    x1 = x * dot_size + border_size
                    y1 = y * dot_size + border_size
                    fill_color = getGradientColor(x1, y1, center_x, center_y, center_color_rgb, outer_color_rgb, max_dist)
                    draw.ellipse([x1, y1, x1 + dot_size, y1 + dot_size], fill=fill_color)

        # Display the image
        img.show()

    def generate_qr_terminal(self,
        text: str,
        center_color: str = "#00FF00",
        outer_color: str = "#0000FF",
        back_color: str = "#000000",
    ):
        log(f"QR requested for : {text}")

        # Generate QR code matrix
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=1,
            border=4,  # Ensure QR standard border
        )
        qr.add_data(text)
        qr.make(fit=True)
        qr_matrix = qr.get_matrix()

        # Dimensions of the QR code
        qr_size = len(qr_matrix)

        # Convert hex colors to RGB
        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip("#")
            return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

        center_rgb = hex_to_rgb(center_color)
        outer_rgb = hex_to_rgb(outer_color)

        # Interpolate between colors
        def interpolate_color(x, y, center_x, center_y, max_dist):
            dist = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
            ratio = min(dist / max_dist, 1)

            r = int(center_rgb[0] + ratio * (outer_rgb[0] - center_rgb[0]))
            g = int(center_rgb[1] + ratio * (outer_rgb[1] - center_rgb[1]))
            b = int(center_rgb[2] + ratio * (outer_rgb[2] - center_rgb[2]))

            return f"rgb({r},{g},{b})"

        # Calculate center and max distance
        center_x = qr_size // 2
        center_y = qr_size // 2
        max_dist = ((center_x) ** 2 + (center_y) ** 2) ** 0.5

        # Prepare to center the QR code in the terminal
        terminal_width = console.size.width
        qr_width = qr_size * 2  # Each QR column rendered takes 2 spaces in terminal
        padding = max((terminal_width - qr_width) // 2, 0)

        # Render QR code as a square
        qr_render = []
        for y in range(0, qr_size, 2):  # Process two rows at a time
            line = Text(" " * padding)  # Add left padding to center the QR code
            for x in range(qr_size):
                upper = qr_matrix[y][x]
                lower = qr_matrix[y + 1][x] if y + 1 < qr_size else 0

                if upper and lower:
                    color = interpolate_color(x, y, center_x, center_y, max_dist)
                    line.append("█", style=f"{color} on {color}")
                elif upper:
                    color = interpolate_color(x, y, center_x, center_y, max_dist)
                    line.append("▀", style=f"{color} on {back_color}")
                elif lower:
                    color = interpolate_color(x, y + 1, center_x, center_y, max_dist)
                    line.append("▄", style=f"{color} on {back_color}")
                else:
                    line.append(" ", style=f"{back_color} on {back_color}")
            console.print(line)
            qr_render.append(line)

        """
        # Example Usage
        generate_qr_terminal(
            "https://example.com",
            center_color="#00FF00",
            outer_color="#0000FF",
            back_color="#000000",
        )

            qr_render.append(line)
        """

        return qr_render



    def open_w3m(self, website_url='https://google.com'):
        try:
            subprocess.run(['w3m', website_url])
        except FileNotFoundError:
            log("Error: w3m is not installed on your system. Please install it and try again.")
        except Exception as e:
            log(f"An error occurred: {e}")

    def url_image_extract(self, url, mode='merged', images_per_row=3):
        # Send a request to the URL
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code != 200:
            raise Exception(f"Failed to fetch the webpage: {response.status_code}")

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all image tags
        img_tags = soup.find_all('img')
        images = []
        img_sources = []

        # Loop through all the image tags and open them
        for img_tag in img_tags:
            img_url = img_tag.get('src')

            # Handle relative URLs
            if img_url and not img_url.startswith(('http://', 'https://')):
                img_url = requests.compat.urljoin(url, img_url)

            try:
                # Get the image content
                img_response = requests.get(img_url)

                # Check the content type to ensure it's an image
                content_type = img_response.headers.get('Content-Type')
                if 'image' in content_type:
                    img = Image.open(BytesIO(img_response.content))
                    images.append(img)
                    img_sources.append(img_url)
                else:
                    log(f"Skipped non-image content at {img_url}")
            except Exception as e:
                log(f"Failed to open image: {img_url} - {e}")

        if not images:
            log("No images found.")
            return

        if mode == 'merged':
            # Determine the size of the resulting image
            max_width = max(img.width for img in images)
            max_height = max(img.height for img in images)
            total_width = images_per_row * max_width
            total_height = (len(images) + images_per_row - 1) // images_per_row * max_height

            # Create a new blank image
            combined_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))

            # Paste each image into the combined image
            for index, img in enumerate(images):
                x = (index % images_per_row) * max_width
                y = (index // images_per_row) * max_height
                combined_image.paste(img, (x, y))

            # Display the combined image
            combined_image.show()

        elif mode == 'individual':
            # Show each image individually
            for img in images:
                img.show()

    def file_browser(self):
        """Terminal-based file browser using prompt_toolkit to navigate and select files."""
        current_path = Path.home()  # Start in the user's home directory
        #current_path = Path.cwd()
        files = []
        selected_index = 0  # Track the selected file/folder index
        scroll_offset = 0  # Track the starting point of the visible list
        show_hidden = False  # Initialize hidden files visibility

        terminal_height = int(os.get_terminal_size().lines)
        max_display_lines = terminal_height - 4  # Reduce by 2 for header and footer lines

        def update_file_list():
            """Update the list of files in the current directory, with '..' as the first entry to go up."""
            nonlocal files, selected_index, scroll_offset
            # List current directory contents and insert '..' at the top for navigating up
            all_files = [".."] + sorted(current_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))

            # Filter out hidden files if `show_hidden` is False
            files = [f for f in all_files if isinstance(f, str) or show_hidden or not f.name.startswith('.')]

            selected_index = 0
            scroll_offset = 0

        def get_display_text():
            """Display text for the current directory contents with the selected item highlighted."""
            text = []
            visible_files = files[scroll_offset:scroll_offset + max_display_lines]
            for i, f in enumerate(visible_files):
                real_index = scroll_offset + i
                prefix = "> " if real_index == selected_index else "  "
                
                # Use only the file or directory name for display
                display_name = f if isinstance(f, str) else f.name
                display_name += "/" if isinstance(f, Path) and f.is_dir() else ""
                
                line = f"{prefix}{display_name}"
                text.append((f"{'yellow' if real_index == selected_index else 'white'}", line))
                text.append(('', '\n'))
            return text

        # Initialize file list with the home directory contents
        update_file_list()

        # Key bindings
        kb = KeyBindings()

        @kb.add("up")
        def move_up(event):
            nonlocal selected_index, scroll_offset
            selected_index = (selected_index - 1) % len(files)
            # Scroll up if the selection goes above the visible area
            if selected_index < scroll_offset:
                scroll_offset = max(0, scroll_offset - 1)

        @kb.add("down")
        def move_down(event):
            nonlocal selected_index, scroll_offset
            selected_index = (selected_index + 1) % len(files)
            # Scroll down if the selection goes beyond the visible area
            if selected_index >= scroll_offset + max_display_lines:
                scroll_offset = min(len(files) - max_display_lines, scroll_offset + 1)

        @kb.add("enter")
        def enter_directory(event):
            nonlocal current_path
            selected_file = files[selected_index]

            if selected_file == "..":
                # Move up to the parent directory
                current_path = current_path.parent
                update_file_list()
            elif isinstance(selected_file, Path) and selected_file.is_dir():
                # Enter the selected directory
                current_path = selected_file
                update_file_list()
            elif isinstance(selected_file, Path) and selected_file.is_file():
                # Select the file and exit
                event.app.exit(result=str(selected_file))  # Return the file path as a string

        @kb.add("escape")
        def cancel_selection(event):
            event.app.exit(result=None)  # Exit with None if canceled

        @kb.add("c-h")
        def toggle_hidden(event):
            """Toggle the visibility of hidden files."""
            nonlocal show_hidden
            show_hidden = not show_hidden
            update_file_list()

        # Layout with footer for shortcut hint
        header_window = Frame(Window(FormattedTextControl(lambda: f"Current Directory: {current_path}"), height=1))
        file_list_window = Window(content=FormattedTextControl(get_display_text), wrap_lines=False, height=max_display_lines)
        footer_window = Window(content=FormattedTextControl(lambda: "Press Ctrl-H to show/hide hidden files. Escape to exit."), height=1, style="grey")

        layout = Layout(HSplit([
            header_window, 
            file_list_window,
            footer_window
        ]))

        # Application
        app = Application(layout=layout, key_bindings=kb, full_screen=True, refresh_interval=0.1)

        # Run the application and return the selected file path (or None if canceled)
        return app.run()

    def flush_history(self):
        self.conversation_history = []
        self.current_conversation = []
        return

    def get_conversations(self, path):
        files = os.listdir(path)
        if not files:
            log("No conversations found.")

        conversations = []
        for file in files:
            if re.search(r'\S+.jsonl$', file):
                conversations.append(file)

        return conversations

    def display_pager(self, text):
        search_field = SearchToolbar()
        
        output_field = TextArea(
                text,
                lexer=PygmentsLexer(PythonLexer),
                scrollbar=False,
                search_field=search_field,
                read_only=True,
                )

        pager_kb = KeyBindings()

        @pager_kb.add("escape")
        @pager_kb.add("q")
        def _(event):
            event.app.exit()

        framing = Frame(
                output_field,
                title="Content Viewer",
                )

        layout = Layout(framing, focused_element=output_field)

        app = Application(
                layout=layout,
                key_bindings=pager_kb,
                mouse_support=True,
                full_screen=True
                )

        return app.run()

    def get_weather(self, location="33004"):
        text = None 
        try:
            # Format the URL for wttr.in with the specified location
            url = f'https://wttr.in/{location}?format=j1'
            response = requests.get(url)

            if response.status_code == 200:
                text = json.loads(response.text)
                return text 
            else:
                log(f"Failed to get weather data. Status code: {response.status_code}")
        except Exception as e:
            log(f"An error occurred: {e}")

        return text

    def get_network_data(self):
        import netifaces
        import dns.resolver
        network_info = {
            "interfaces": {},
            "dns_resolvers": [],
            "default_gateways": {},
            "external_ip": "", 
        }

        try:
            # Format the URL for wttr.in with the specified location
            url = f'https://api.ipify.org'
            response = requests.get(url)

            if response.status_code == 200:
                network_info['external_ip'] = response.text
            else:
                log(f"Failed to get weather data. Status code: {response.status_code}")
        except Exception as e:
            log(f"An error occurred: {e}")

        # Gather information about each network interface
        for interface in netifaces.interfaces():
            interface_info = {}
            addresses = netifaces.ifaddresses(interface)
            
            # IPv4 information
            if netifaces.AF_INET in addresses:
                ipv4_info = addresses[netifaces.AF_INET][0]
                interface_info["ipv4"] = {
                    "address": ipv4_info.get("addr"),
                    "netmask": ipv4_info.get("netmask"),
                    "broadcast": ipv4_info.get("broadcast")
                }
            
            # IPv6 information
            if netifaces.AF_INET6 in addresses:
                ipv6_info = addresses[netifaces.AF_INET6][0]
                interface_info["ipv6"] = {
                    "address": ipv6_info.get("addr"),
                    "netmask": ipv6_info.get("netmask"),
                    "broadcast": ipv6_info.get("broadcast")
                }
            
            # MAC address (hardware address)
            if netifaces.AF_LINK in addresses:
                mac_info = addresses[netifaces.AF_LINK][0]
                interface_info["mac"] = mac_info.get("addr")
            
            # Add interface information to the main dictionary
            network_info["interfaces"][interface] = interface_info

        # Collect DNS resolvers
        try:
            with open('/etc/resolv.conf', 'r') as file:
                for line in file:
                    if line.startswith('nameserver'):
                        network_info["dns_resolvers"].append(line.split()[1])
        except FileNotFoundError:
            network_info["dns_resolvers"] = ["Could not read /etc/resolv.conf"]

        # Collect default gateway information
        gateways = netifaces.gateways()
        for family, gateway_info in gateways.get("default", {}).items():
            if family == netifaces.AF_INET:
                network_info["default_gateways"]["ipv4"] = {
                    "gateway": gateway_info[0],
                    "interface": gateway_info[1]
                }
            elif family == netifaces.AF_INET6:
                network_info["default_gateways"]["ipv6"] = {
                    "gateway": gateway_info[0],
                    "interface": gateway_info[1]
                }

        return network_info

    def iface_report(self, network_info):
        report = []
        report.append("Network Information\n" + "="*20)

        # External IP
        report.append("\nExternal IP:")
        if network_info["external_ip"]:
            report.append(f"  - {network_info['external_ip']}")
        else:
            report.append("  None")

        # DNS Resolvers
        report.append("\nDNS Resolvers:")
        if network_info["dns_resolvers"]:
            for resolver in network_info["dns_resolvers"]:
                report.append(f"  - {resolver}")
        else:
            report.append("  None")

        # Default Gateways
        report.append("\nDefault Gateways:")
        gateway_devs = []
        ipv4_gateway = network_info["default_gateways"].get("ipv4")
        if ipv4_gateway:
            report.append(f"  Gateway: {ipv4_gateway['gateway']}")
            report.append(f"  Interface: {ipv4_gateway['interface']}")
            gateway_devs.append(ipv4_gateway['interface'])

        ipv6_gateway = network_info["default_gateways"].get("ipv6")
        if ipv6_gateway:
            report.append(f"  Gateway: {ipv6_gateway['gateway']}")
            report.append(f"  Interface: {ipv6_gateway['interface']}")
            gateway_devs.append(ipv6_gateway['interface'])

        # Interface information
        report.append(f"\nInterfaces:")
        for interface, details in network_info["interfaces"].items():
            primary = "" 
            if interface in gateway_devs:
                primary = " *primary" 

            mac = details.get("mac", None)
            # IPv4 Info
            ipv4_info = details.get("ipv4")
            if ipv4_info:
                report.append(f"  {interface}:{primary}")
                report.append(f"    Address: {ipv4_info.get('address')}")
                report.append(f"    Netmask: {ipv4_info.get('netmask')}")
                report.append(f"    Broadcast: {ipv4_info.get('broadcast')}")
                if mac is not None:
                    report.append(f"    Mac: {mac}")

                continue

            # IPv6 Info
            ipv6_info = details.get("ipv6")
            if ipv6_info:
                report.append(f"  {interface}:{primary}")
                report.append(f"    Address: {ipv6_info.get('address')}")
                report.append(f"    Netmask: {ipv6_info.get('netmask')}")
                report.append(f"    Broadcast: {ipv6_info.get('broadcast')}")
                if mac is not None:
                    report.append(f"    Mac: {mac}")

                continue

        # Join the report list into a single string
        return "\n".join(report)

    def get_search_results(self, search_term=None):
        search_pattern = ""
        if search_term.startswith("/") and search_term.endswith("/"):
            search_pattern = rf"{search_term[1:-1]}" 
            try:
                search_for = re.compile(search_pattern)
            except Exception as e:
                log(f"Invalid regex pattern: {e}")
                return None
        elif search_term is None:
            log(f"Variable search_term is empty.")
            return None
        else:
            search_for = search_term

        results = self.memory.search(search_for)

        json_results = []
        if results:
            snips_get = {} 
            parent = []
            for idx, result in enumerate(results):
                if re.search(r"\[\d+\]$", result['key']):
                    new_key = re.sub(r"\[\d+\]", "", result['key'])
                    if new_key in snips_get:
                        results[snips_get[new_key]]['snippets'] = results[snips_get[new_key]]['snippets'] + result['snippets']
                    else:
                        new_parent = re.sub(r'\.[^.]*$', '', result['key'])
                        index = len(results)
                        snips_get[new_key] = index
                        results.append({
                            "key": new_key,
                            "parent": new_parent,
                            "type": "list",
                            "snippets": [],
                        })
                    continue

                result_entry = {
                    "search_term": search_term,
                    "key": result['key'],
                    "parent": result['parent'],
                    "type": result['type'],
                    "snippets": []  # This will hold each highlighted snippet
                }

                header_text = Text.from_markup(
                    f"[bold bright_cyan]Key:[/bold bright_cyan] {result['key']}\n"
                    f"[bold bright_cyan]Parent:[/bold bright_cyan] {result['parent']}\n"
                    f"[bold bright_cyan]Type:[/bold bright_cyan] {result['type']}()\n"
                    f"[bold bright_cyan]Matched:[/bold bright_cyan] {search_term}"
                    )
                parent.append(header_text)
                print(header_text)

                # Loop through each snippet in the result
                for idx, snip in enumerate(result['snippets']):
                    parent.append(Text())
                    snip = escape(snip.strip())
                    if len(snip) > 0:
                        if isinstance(search_for, re.Pattern):
                            highlighted_text = re.sub(
                                search_pattern,
                                f"[bold bright_green]\\g<0>[/bold bright_green]",
                                snip,
                                flags=re.IGNORECASE
                            )
                        else:
                            highlighted_text = re.sub(
                                re.escape(search_for),
                                f"[bold bright_green]{search_term}[/bold bright_green]",
                                snip,
                                flags=re.IGNORECASE
                            )

                        snippet = Text.from_markup(f"{highlighted_text.strip()}") 
                        parent.append(Padding(snippet, (1, 0, 1, 4), style="on grey15"))

                        result_entry["snippets"].append(snip)

                json_results.append(result_entry)
                parent.append(Text())
                #parent.append(print(Panel(, title=f"Key: {result['key']}", padding=(1, 2))))
            panel_group = Group(*parent)
            print(Panel(panel_group))
        else:
            log(f"No results for: {search_term}")
            return None

        return escape(json.dumps(json_results, indent=4)) 

    def exec_command(self, command):
        output = ""
        try:
            process = subprocess.Popen(command, shell=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            output = stdout + stderr
        except KeyboardInterrupt:
            log("\nCommand interrupted by Control-C.", flush=True)
            if process:
                process.terminate()
                process.wait()
        except subprocess.CalledProcessError as e:
            output = e.stdout + e.stderr if e.stdout or e.stderr else "Command exited with a status other than 0."

        return output.strip()

    def web_data_stats(self, data):
        content = data.get("content", "")
        console.clear()
        
        words = content.split()
        word_count = len(words)
        char_count = len(content)
        line_count = content.count('\n') + 1

        links = data.get("links", [])
        links_list = []
        for link in links:
            if re.search(r'^https?://\S+', link):
                domain = link.split("/")[2]
                links_list.append(domain)
        domains = set(links_list)

        css_files = data.get("css", [])
        css_size = sum(len(css) for css in css_files)

        scripts = data.get("scripts", [])
        script_size = sum(len(script) for script in scripts)

        coll = []
        header = Panel(
            Text("Dashboard: Website Analysis", justify="center", style="cyan"),
            style=""
        )
        print(header)
        coll.append(header)

        content_table = Table.grid(expand=True )
        content_table.add_column(justify="left", no_wrap=True)
        content_table.add_row("[bold]Words:[/bold] ", f"{word_count}")
        content_table.add_row("[bold]Characters:[/bold] ", f"{char_count}")
        content_table.add_row("[bold]Lines:[/bold] ", f"{line_count}")

        content_panel = Panel(content_table, title="[bold magenta]Content Stats[/bold magenta]")
        print(content_panel)
        coll.append(content_panel)

        links_table = Table.grid(expand=True)
        links_table.add_column(justify="left")
        links_table.add_row("[bold]Total Links:[/bold] ", f"{len(links)}")
        links_table.add_row("[bold]Unique Domains:[/bold] ", ", ".join(domains) or "None")

        links_panel = Panel(links_table, title="[bold yellow]Links Stats[/bold yellow]")
        print(links_panel)
        coll.append(links_panel)

        css_table = Table.grid(expand=True)
        css_table.add_column(justify="left")
        css_table.add_row("[bold]CSS Files:[/bold] ", f"{len(css_files)}")
        css_table.add_row("[bold]Total Size:[/bold] ", f"{css_size / 1024:.2f} KB")

        css_panel = Panel(css_table, title="[bold green]CSS Stats[/bold green]")
        print(css_panel)
        coll.append(css_panel)

        scripts_table = Table.grid(expand=True)
        scripts_table.add_column(justify="left")
        scripts_table.add_row("[bold]JS Files:[/bold] ", f"{len(scripts)}")
        scripts_table.add_row("[bold]Total Size:[/bold] ", f"{script_size / 1024:.2f} KB")

        scripts_panel = Panel(scripts_table, title="[bold blue]Scripts Stats[/bold blue]")
        print(scripts_panel)
        coll.append(scripts_panel)

        layout = Table.grid(expand=True, padding=(0, 0))
        layout.add_column(ratio=1)
        layout.add_column(ratio=2)
        layout.add_column(ratio=1)
        footer = Panel(
            Text("Web Stats", justify="center", style=""),
            style=""
        )
        layout.add_row(header, header, header)  # Header spans all columns
        layout.add_row(content_panel, links_panel, css_panel)  # Main panels
        layout.add_row(scripts_panel, footer, footer)  # Footer spans columns
        
        container = Panel(layout)
        coll.append(footer)
        print(layout)

    def display_metadata(self, metadata):
        def render_hashes(hashes):
            table = Table(show_header=False, header_style="", expand=True)
            table.add_column("Algorithm", justify="right")
            table.add_column("Value", justify="left")
            for key, value in hashes.items():
                table.add_row(key, value)
            return table

        def render_urls(urls):
            table = Table(show_header=False, header_style="", expand=True)
            table.add_column("URL", justify="left")
            for url in urls:
                table.add_row(url)
            return table

        def render_exif(exif):
            table = Table(show_header=False, header_style="", expand=True)
            table.add_column("Key", ratio=1, justify="right")
            table.add_column("Value", ratio=4, justify="left")
            for key, value in exif.items():
                table.add_row(key, str(value))
            return table

        table = Table(show_header=False, box=None, expand=True, header_style="bold", title="File Information")
        table.add_column("Key", justify="right", style="gold1")
        table.add_column("Value", justify="left")

        for key, value in metadata.items():
            if key == "hashes":
                value = render_hashes(value)
            elif key == "embedded_urls":
                value = render_urls(value)
            elif key == "exif":
                value = render_exif(value)
            elif key == "readable_strings":
                value = str(len(value))
            elif key == "is_code":
                value = str(value)
            else:
                value = str(value) if value is not None else "None"

            table.add_row(key, value)

        terminal_width = console.width
        table_width = max(len(line) for line in str(table).split("\n"))
        padding = (terminal_width - table_width) // 2

        console.print(" " * padding, table)

    def display_object(self, data):
        """
        Render a table for the given dictionary or list.
        """
        def render_dict(d):
            """
            Render a dictionary as a table.
            """
            table = Table(show_header=False, expand=True)
            table.add_column("Key", ratio=1, justify="right", style="gold1")
            table.add_column("Value", ratio=4, justify="left")

            for key, value in d.items():
                value = render_value(value)
                table.add_row(key, value)
            return table

        def render_list(lst):
            """
            Render a list as a table.
            """
            table = Table(show_header=False, expand=True)
            table.add_column("Index", ratio=1, justify="right", style="gold1")
            table.add_column("Value", ratio=4, justify="left")

            for idx, value in enumerate(lst):
                value = render_value(value)
                table.add_row(str(idx), value)
            return table

        def render_value(value):
            """
            Render a value, handling nested dictionaries, lists, and other types.
            """
            if isinstance(value, dict):
                return render_dict(value)
            elif isinstance(value, list):
                truncated_list = value[:20]
                return render_list(truncated_list)
                #return render_list(value)
            elif isinstance(value, str):
                return value[:1000] + "..." if len(value) > 1000 else value
            elif value is None:
                return "None"
            else:
                return str(value)

        # Decide what to render based on the type of the input data
        if isinstance(data, dict):
            table = render_dict(data)
        elif isinstance(data, list):
            table = render_list(data)
        else:
            raise TypeError("Input must be a dictionary or a list.")

        # Center the table in the terminal
        terminal_width = console.width
        table_width = max(len(line) for line in str(table).split("\n"))
        padding = (terminal_width - table_width) // 2
        table.box = None

        # Print the table
        console.print(" " * padding, table)
