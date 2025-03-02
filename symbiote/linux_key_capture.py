#!/usr/bin/env python3
from evdev import InputDevice, categorize, ecodes

key_map = {
    'KEY_ESC': ('\x1b', '\x1b'),
    'KEY_1': ('1', '!'),
    'KEY_2': ('2', '@'),
    'KEY_3': ('3', '#'),
    'KEY_4': ('4', '$'),
    'KEY_5': ('5', '%'),
    'KEY_6': ('6', '^'),
    'KEY_7': ('7', '&'),
    'KEY_8': ('8', '*'),
    'KEY_9': ('9', '('),
    'KEY_0': ('0', ')'),
    'KEY_MINUS': ('-', '_'),
    'KEY_EQUAL': ('=', '+'),
    'KEY_BACKSPACE': ('\b \b', '\b \b'),
    'KEY_TAB': ('\t', '\t'),
    'KEY_Q': ('q', 'Q'),
    'KEY_W': ('w', 'W'),
    'KEY_E': ('e', 'E'),
    'KEY_R': ('r', 'R'),
    'KEY_T': ('t', 'T'),
    'KEY_Y': ('y', 'Y'),
    'KEY_U': ('u', 'U'),
    'KEY_I': ('i', 'I'),
    'KEY_O': ('o', 'O'),
    'KEY_P': ('p', 'P'),
    'KEY_LEFTBRACE': ('[', '{'),
    'KEY_RIGHTBRACE': (']', '}'),
    'KEY_ENTER': ('\n', '\n'),
    'KEY_A': ('a', 'A'),
    'KEY_S': ('s', 'S'),
    'KEY_D': ('d', 'D'),
    'KEY_F': ('f', 'F'),
    'KEY_G': ('g', 'G'),
    'KEY_H': ('h', 'H'),
    'KEY_J': ('j', 'J'),
    'KEY_K': ('k', 'K'),
    'KEY_L': ('l', 'L'),
    'KEY_SEMICOLON': (';', ':'),
    'KEY_APOSTROPHE': ("'", '"'),
    'KEY_GRAVE': ('`', '~'),
    'KEY_BACKSLASH': ('\\', '|'),
    'KEY_Z': ('z', 'Z'),
    'KEY_X': ('x', 'X'),
    'KEY_C': ('c', 'C'),
    'KEY_V': ('v', 'V'),
    'KEY_B': ('b', 'B'),
    'KEY_N': ('n', 'N'),
    'KEY_M': ('m', 'M'),
    'KEY_COMMA': (',', '<'),
    'KEY_DOT': ('.', '>'),
    'KEY_SLASH': ('/', '?'),
    'KEY_SPACE': (' ', ' '),
}

modifier_keys = {
    'KEY_LEFTSHIFT': 'KEY_LEFTSHIFT',
    'KEY_RIGHTSHIFT': 'KEY_RIGHTSHIFT',
    'KEY_LEFTCTRL': 'KEY_LEFTCTRL',
    'KEY_RIGHTCTRL': 'KEY_RIGHTCTRL',
    'KEY_LEFTALT': 'KEY_LEFTALT',
    'KEY_RIGHTALT': 'KEY_RIGHTALT',
    'KEY_LEFTMETA': 'KEY_LEFTMETA',  # Left "Windows" or "Super" key
    'KEY_RIGHTMETA': 'KEY_RIGHTMETA',  # Right "Windows" or "Super" key
    'KEY_CAPSLOCK': 'KEY_CAPSLOCK',
    'KEY_NUMLOCK': 'KEY_NUMLOCK',
    'KEY_SCROLLLOCK': 'KEY_SCROLLLOCK',
}

def find_keyboard_device():
    with open("/proc/bus/input/devices") as f:
        lines = f.readlines()

    for line in lines:
        if "keyboard" in line.lower():
            next_line = lines[lines.index(line) + 1]
            event_line = ""
            for f in lines[lines.index(next_line):]:
                if f.strip().startswith("H: Handlers"):
                    event_line = f
                    break

            event = event_line.strip().split("event")[1]
            event = event.split(" ")[0]
            return "/dev/input/event" + event

key_dev = find_keyboard_device()
dev = InputDevice(key_dev)
shift_pressed = False
caps_pressed = False

for event in dev.read_loop():
    if event.type == ecodes.EV_KEY:
        key_event = categorize(event)
        if key_event.keystate == key_event.key_down:
            if key_event.keycode == 'KEY_CAPSLOCK':
                if caps_pressed:
                    caps_pressed = False
                else:
                    caps_pressed = True
            elif key_event.keycode in ['KEY_LEFTSHIFT', 'KEY_RIGHTSHIFT']:
                shift_pressed = True
            elif key_event.keycode in key_map:
                char_lower, char_upper = key_map[key_event.keycode]
                if shift_pressed or caps_pressed:
                    print(char_upper, end="", flush=True)
                else:
                    print(char_lower, end="", flush=True)
            elif key_event.keycode in modifier_keys:
                print(modifier_keys[key_event.keycode], end="", flush=True)
            else:
                print(key_event.keycode, end="", flush=True)

        elif key_event.keystate == key_event.key_up:
            if key_event.keycode in ['KEY_LEFTSHIFT', 'KEY_RIGHTSHIFT']:
                shift_pressed = False
