#!/usr/bin/env python3
#
# sym_monitor.py

import io
import sys
import re
import threading
import time
import subprocess
import clipboard
import platform
import pyaudio
import speech_recognition as sr
from pynput.keyboard import Listener
from symbiote.sym_speech import SymSpeech

if platform.system() == 'Linux':
    from evdev import InputDevice, categorize, ecodes

class SymMonitor:
    def __init__(self, schat, debug=False):
        self.schat = schat
        self.debug = debug
        self.chat_is_active = False
        self.lastlog = ""
        self.command = ["terminator", "-e"]
        self.totallog = ""
        self.previousclip = ""
        self.clipboard_content = clipboard.paste()

        # Activation key combinations to monitor for.
        self.keys_activate = {
                "clipboard": r'Key\.ctrl\]',
                "append": r'Key\.ctrl=',
                "keyboard": r':help::|Key\.ctrlh'
                }

        self.speech = SymSpeech(monitor=True, debug=self.debug)
        self.speech.start_keyword_listen()

        #schat.chat(user_input="role:HELP_ROLE:", suppress=True, run=True)

    def linux_find_keyboard_device(self):
        ''' Find keyboard device for linux '''
        with open("/proc/bus/input/devices") as f:
            lines = f.readlines()

        for line in lines:
            if "keyboard" in line.lower():
                next_line = lines[lines.index(line) + 1]
                event_line = ""
                for f in lines[lines.index(next_line):]:
                    if f.strip().startswith("H: Handlers"):
                        event_line = f

                event = event_line.strip().split("event")[1]
                event = event.split(" ")[0]
                return "/dev/input/event" + event

    def linux_read_keys(self):
        key_dev = self.linux_find_keyboard_device()
        dev = InputDevice(key_dev)

        for event in dev.read_loop():
            if event.type == ecodes.EV_KEY:
                print(categorized(event))

    def pull_clipboard(self):
        # Pull clipboard contents
        self.clipboard_content = clipboard.paste()
        if self.clipboard_content != self.previousclip:
            self.previousclip = self.clipboard_content

    def scrub_keys(self, log):
        self.key_mapping = {
                ":help::": '',
                "Key\.enter": '\n',
                "Key\.space": ' ',
                "Key\.tab": '\t',
                "Key\.ctrlh": '',
                "Key\.ctrl=": '',
                "Key\.ctrl]": '',
                "Key\.ctrl": '',
                "Key\.shift_r": '',
                "Key\.shift": '',
                "Key\.down": '',
                "Key\.up": '',
                "Key\.left": '',
                "Key\.right": '',
                "Key\.cmdr": '',
                "Key\.cmd": '',
                "Key\.esc": '',
                "Key\.caps_lock": '',
                "Key\.alt_r": '',
                "Key\.alt": ''
            }

        for key in self.key_mapping:
           log = re.sub(key, self.key_mapping[key], log)

        self.lastlog = ""

        return log 

    def on_press(self, key):
        if self.chat_is_active:
            return

        try:
            pressed = re.sub('\'', "", str(key))

        except AttributeError:
            print("Unable to capture key.")

        if self.debug:
            print(pressed, end='')

        if re.search(r'^Key\.backspace', pressed):
            self.lastlog = self.lastlog[:-1]
            return

        self.lastlog += pressed

        if re.search(self.keys_activate['clipboard'], self.lastlog):
            self.pull_clipboard()
            content = self.scrub_keys(self.clipboard_content)
            if re.search(r'^http:\/\/.*|^https:\/\/.*', content):
                content = f'get:{content}:'
                if self.debug:
                    print(content)

            self.schat.chat(user_input=content, suppress=True, run=True)
            content = ""
            self.mon_launch_window(content)
        elif re.search(self.keys_activate['append'], self.lastlog):
            self.lastlog = self.scrub_keys(self.lastlog)
            content = self.scrub_keys(self.lastlog)
            self.schat.chat(user_input=content, suppress=True, run=True)
        elif re.search(self.keys_activate['keyboard'], self.lastlog):
            content = self.scrub_keys(self.lastlog)
            self.mon_launch_window(content)
            
    def mon_launch_window(self, content):
        self.chat_is_active = True
        issue_command = f'symbiote -q "{content}"'

        self.command.append(issue_command)

        process = subprocess.Popen(self.command, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        while process.poll() is None:
            time.sleep(1)

        self.command.pop()
        self.chat_is_active = False

    def start(self):
        listener = Listener(on_press=self.on_press)
        self.t = threading.Thread(target=listener.start)
        self.t.start()

