#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: sym_highlighter.py
# Author: Wadih Khairallah
# Description: 
# Created: 2024-11-26 23:30:12
# Modified: 2024-11-26 23:38:51
#!/usr/bin/env python3
#
# sym_highlighter.py

import re
from rich.console import Console
from rich.highlighter import RegexHighlighter
from rich.theme import Theme
from rich.color import Color, ANSI_COLOR_NAMES
from typing import Dict, Union

"""
ANSI_COLOR_NAMES = {
    "black": 0,
    "red": 1,
    "green": 2,
    "yellow": 3,
    "blue": 4,
    "magenta": 5,
    "cyan": 6,
    "white": 7,
    "bright_black": 8,
    "bright_red": 9,
    "bright_green": 10,
    "bright_yellow": 11,
    "bright_blue": 12,
    "bright_magenta": 13,
    "bright_cyan": 14,
    "bright_white": 15,
    "grey0": 16,
    "gray0": 16,
    "navy_blue": 17,
    "dark_blue": 18,
    "blue3": 20,
    "blue1": 21,
    "dark_green": 22,
    "deep_sky_blue4": 25,
    "dodger_blue3": 26,
    "dodger_blue2": 27,
    "green4": 28,
    "spring_green4": 29,
    "turquoise4": 30,
    "deep_sky_blue3": 32,
    "dodger_blue1": 33,
    "green3": 40,
    "spring_green3": 41,
    "dark_cyan": 36,
    "light_sea_green": 37,
    "deep_sky_blue2": 38,
    "deep_sky_blue1": 39,
    "spring_green2": 47,
    "cyan3": 43,
    "dark_turquoise": 44,
    "turquoise2": 45,
    "green1": 46,
    "spring_green1": 48,
    "medium_spring_green": 49,
    "cyan2": 50,
    "cyan1": 51,
    "dark_red": 88,
    "deep_pink4": 125,
    "purple4": 55,
    "purple3": 56,
    "blue_violet": 57,
    "orange4": 94,
    "grey37": 59,
    "gray37": 59,
    "medium_purple4": 60,
    "slate_blue3": 62,
    "royal_blue1": 63,
    "chartreuse4": 64,
    "dark_sea_green4": 71,
    "pale_turquoise4": 66,
    "steel_blue": 67,
    "steel_blue3": 68,
    "cornflower_blue": 69,
    "chartreuse3": 76,
    "cadet_blue": 73,
    "sky_blue3": 74,
    "steel_blue1": 81,
    "pale_green3": 114,
    "sea_green3": 78,
    "aquamarine3": 79,
    "medium_turquoise": 80,
    "chartreuse2": 112,
    "sea_green2": 83,
    "sea_green1": 85,
    "aquamarine1": 122,
    "dark_slate_gray2": 87,
    "dark_magenta": 91,
    "dark_violet": 128,
    "purple": 129,
    "light_pink4": 95,
    "plum4": 96,
    "medium_purple3": 98,
    "slate_blue1": 99,
    "yellow4": 106,
    "wheat4": 101,
    "grey53": 102,
    "gray53": 102,
    "light_slate_grey": 103,
    "light_slate_gray": 103,
    "medium_purple": 104,
    "light_slate_blue": 105,
    "dark_olive_green3": 149,
    "dark_sea_green": 108,
    "light_sky_blue3": 110,
    "sky_blue2": 111,
    "dark_sea_green3": 150,
    "dark_slate_gray3": 116,
    "sky_blue1": 117,
    "chartreuse1": 118,
    "light_green": 120,
    "pale_green1": 156,
    "dark_slate_gray1": 123,
    "red3": 160,
    "medium_violet_red": 126,
    "magenta3": 164,
    "dark_orange3": 166,
    "indian_red": 167,
    "hot_pink3": 168,
    "medium_orchid3": 133,
    "medium_orchid": 134,
    "medium_purple2": 140,
    "dark_goldenrod": 136,
    "light_salmon3": 173,
    "rosy_brown": 138,
    "grey63": 139,
    "gray63": 139,
    "medium_purple1": 141,
    "gold3": 178,
    "dark_khaki": 143,
    "navajo_white3": 144,
    "grey69": 145,
    "gray69": 145,
    "light_steel_blue3": 146,
    "light_steel_blue": 147,
    "yellow3": 184,
    "dark_sea_green2": 157,
    "light_cyan3": 152,
    "light_sky_blue1": 153,
    "green_yellow": 154,
    "dark_olive_green2": 155,
    "dark_sea_green1": 193,
    "pale_turquoise1": 159,
    "deep_pink3": 162,
    "magenta2": 200,
    "hot_pink2": 169,
    "orchid": 170,
    "medium_orchid1": 207,
    "orange3": 172,
    "light_pink3": 174,
    "pink3": 175,
    "plum3": 176,
    "violet": 177,
    "light_goldenrod3": 179,
    "tan": 180,
    "misty_rose3": 181,
    "thistle3": 182,
    "plum2": 183,
    "khaki3": 185,
    "light_goldenrod2": 222,
    "light_yellow3": 187,
    "grey84": 188,
    "gray84": 188,
    "light_steel_blue1": 189,
    "yellow2": 190,
    "dark_olive_green1": 192,
    "honeydew2": 194,
    "light_cyan1": 195,
    "red1": 196,
    "deep_pink2": 197,
    "deep_pink1": 199,
    "magenta1": 201,
    "orange_red1": 202,
    "indian_red1": 204,
    "hot_pink": 206,
    "dark_orange": 208,
    "salmon1": 209,
    "light_coral": 210,
    "pale_violet_red1": 211,
    "orchid2": 212,
    "orchid1": 213,
    "orange1": 214,
    "sandy_brown": 215,
    "light_salmon1": 216,
    "light_pink1": 217,
    "pink1": 218,
    "plum1": 219,
    "gold1": 220,
    "navajo_white1": 223,
    "misty_rose1": 224,
    "thistle1": 225,
    "yellow1": 226,
    "light_goldenrod1": 227,
    "khaki1": 228,
    "wheat1": 229,
    "cornsilk1": 230,
    "grey100": 231,
    "gray100": 231,
    "grey3": 232,
    "gray3": 232,
    "grey7": 233,
    "gray7": 233,
    "grey11": 234,
    "gray11": 234,
    "grey15": 235,
    "gray15": 235,
    "grey19": 236,
    "gray19": 236,
    "grey23": 237,
    "gray23": 237,
    "grey27": 238,
    "gray27": 238,
    "grey30": 239,
    "gray30": 239,
    "grey35": 240,
    "gray35": 240,
    "grey39": 241,
    "gray39": 241,
    "grey42": 242,
    "gray42": 242,
    "grey46": 243,
    "gray46": 243,
    "grey50": 244,
    "gray50": 244,
    "grey54": 245,
    "gray54": 245,
    "grey58": 246,
    "gray58": 246,
    "grey62": 247,
    "gray62": 247,
    "grey66": 248,
    "gray66": 248,
    "grey70": 249,
    "gray70": 249,
    "grey74": 250,
    "gray74": 250,
    "grey78": 251,
    "gray78": 251,
    "grey82": 252,
    "gray82": 252,
    "grey85": 253,
    "gray85": 253,
    "grey89": 254,
    "gray89": 254,
    "grey93": 255,
    "gray93": 255,
}
"""

class SymHighlighter(RegexHighlighter):
    # Console default highlighter theme
    DEFAULT_STYLES = {
        "email": "yellow",
        "url": "green",
        "unix_path": "cyan",
        "windows_path": "#1e90ff",
        "ini": (255, 99, 71),
        "json": "#FFA500",
        "ipv4": "chartreuse2",
        "ipv6": "deep_pink3",
        "mac": "#9ACD32",
        "time": "gold1",
        "timestamp": "gold1",
        "date": "gold1",
        "week": "gold1",
        "month": "gold1",
        "datetime": "gold1",
        "hex_number": "#FF4500",
        "env_var": "magenta",
        "uuid": "sky_blue3",
        "phone_number": "turquoise4",
        "debug": "black",
        "analyze": "cyan",
        "mlink": "green1",
        "mref": "light_sky_blue1",
    }  

    PHONE_PATTERNS = [
        r"\b(?P<phone_number>\+?\d{1,3}[-.\s]?\(?\d{1,3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b",
    ]

    MONTH_PATTERNS = [
        r"\b(?P<month>(?i:January|February|March|April|May|June|July|August|September|October|November|December))\b",
        r"\b(?P<month>(?i:Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec))\b",
    ]

    RENDERING_PATTERNS = [
        r"\b\[(?P<mlink>.*)\]\b",
        r"\b\((?P<mref>^#.*)\)\b",
        r"\b\*\*(?P<mbold>.*)\*\*\b",
    ]

    WEEK_PATTERNS = [
        r"(?P<week>(?i:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday))",
        r"(?P<week>\b(?i:Mon|Tue(s)|Wed|Thu(rs)|Fri|Sat|Sun)\b)",
    ]

    DATETIME_PATTERNS = [
        r"(?P<datetime>\b\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?\b)",  # ISO-8601
        r"(?P<datetime>\d{4}(/|-)(?:0[1-9]|1[0-2])(/|-)(?:0[1-9]|[12][0-9]|3[01])\b)",
        r"(?P<datetime>(?:[01][0-9]|2[0-3])(:)[0-5][0-9](:)(?:[0-5][0-9]|60)\b)",
        r"(?P<datetime>(?:[01][0-9]|2[0-3])(:)[0-5][0-9]\b)",
        r"(?P<datetime>\b(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}\b))",  # Standard datetime
        r"(?P<datetime>\b\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}\b)",  # Special datetime
        r"(?P<datetime>(?:[01][0-9]|2[0-3])(:)[0-5][0-9]\b)",
        r"(?P<datetime>(?:(\d{2}){2}\d{4} (\d{2}:){3}))",
    ]

    DATE_PATTERNS = [
        r"\b(?P<date>\d{4}[-/]\d{1,2}[-/]\d{1,2})\b",
        r"\b(?P<date>\d{1,2}[-/]\d{1,2}[-/]\d{4})\b",
        r"\b(?P<date>(?i:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{1,2}(st|nd|rd|th)\s\d{4})\b",
        r"\b(?P<date>(?i:Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{1,2}(st|nd|rd|th)\s\d{4})\b",
    ]

    TIME_PATTERNS = [
        r"\b(?P<time>\d{2}(:|\.)\d{2}((:|\.)\d{2,4}|))\b",
        r"\b(?P<time>(\d{2}(\.|:)|)\d{1,2}(?i:am|pm))\b",
    ]

    IPV6_PATTERNS = [
        r"\b(?P<ipv6>(?:[0-9a-fA-F]{1,4}|)(:)([0-9a-fA-F]{1,4}))\b",
        r"\b(?P<ipv6>(?:[0-9a-fA-F]{1,4}))?!\.(\d{1,3}\.){3}",
        r"\b(?P<ipv6>(?:[0-9a-fA-F]{1,4}))?!.*::.*::",
        r"(?P<ipv6>(?:([0-9a-fA-F]{4}(:)[0-9a-fA-F]{3}(::))))",
        r"\b(?P<ipv6>(?:[0-9a-fA-F]{4}(::)))",
        r"(?P<ipv6>(?:(::|:)(([0-9a-fA-F]{4}+)|\d\b|([0-9a-fA-F]{3})(:)[0-9a-fA-F](::))))",
    ]

    IPV4_PATTERNS = [
        r"\b(?P<ipv4>(?:(\d{1,3}\.){3}\d{1,3}(\/\d{1,2}\b|\/|)))",  # IPv4 pattern
    ]

    MACADDRESS_PATTERNS = [
        #r"\b(?P<mac>([0-9a-fA-F]{2}[:-]){5}[0-9a-fA-F]{2})\b",
        #r"\b(?P<mac>[0-9a-fA-F]{12})\b",
        #r"\b(?P<mac>([0-9a-fA-F]{4}\.){2}[0-9a-fA-F]{4})\b",
        r"(?P<mac>(?:(\s|^)([0-9a-fA-F]{2}(\.|\-|:)){5}[0-9a-fA-F]{2}))",
        r"(?P<mac>(?:(\s|^)[0-9a-fA-F]{12}))",
    ]

    MISC_PATTERNS = [
        r"(?P<unix_path>(?:[ \t\n]|^)/(?:[a-zA-Z0-9_.-]+/)*[a-zA-Z0-9_.-]+)",
        r"(?P<windows_path>([a-zA-Z]:\\|\\\\)[\w\\.-]+)",  # Windows file paths
        r"(?P<email>[\w.-]+@([\w-]+\.)+[\w-]+)",  # Email addresses
        r"\b(?P<email>(\w+\.|)\w+@([\w-]+\.)+[\w-]+)",  # Email addresses
        r"(?P<url>([a-zA-Z]+):\/\/[a-zA-Z0-9\-._~:/?#[\]@!$&'()*+,;=%]+)",
        r"(?P<ini>\[\w+\])",                     # INI sections
        r"(?P<json>{.*?}|\[.*?\])",              # JSON-like objects
        r"(?P<hex_number>\b0x[0-9a-fA-F]+\b)",   # Hexadecimal numbers
        r"(?P<env_var>\$[\w]+|%[\w]+%)",         # Environment variables
        r"(?P<uuid>\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b)",  # UUIDs
    ]

    ANALYZE_PATTERNS = [
        r"(?P<analyze>\b[A-Za-z0-9]{10,20}\b)",
        r"\b(?P<analyze>(?=[A-Za-z0-9]*[A-Za-z])(?=[A-Za-z0-9]*\d)[A-Za-z0-9]{6,20}\b)",
        r"\b(?P<analyze>\d{8,20}\b)",
        r"\b(?P<analyze>[A-Z]{8,20}\b)",
        r"\b(?P<analyze>(?:[A-Za-z0-9]{2}[,:|.-]([A-Za-z0-9]{2}|)){4,20})\b",
    ]

    """
    PHONE_PATTERNS = [
        r"(?P<phone_number>\+?\d{1,3}(-| |)?\(?\d{3}\)?(-| |)?\d{3}(-| |)?\d{4})",  # E.164 and variations
        r"(?P<phone_number>\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4})",  # Standard US formats
        r"(?P<phone_number>(\+|)\d{1,3}(-)\d{1,4}(-)\d{1,4}(-)\d{1,9})",  # International with optional separators
        r"(?P<phone_number>(\+|)\d{10})",  # Compact 10-digit
    ]

    DATETIME_PATTERNS = [
        r"(?P<datetime>\b\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?\b)",  # ISO-8601
        r"(?P<datetime>(?i:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday))",
        r"(?P<datetime>\b(?i:Mon|Tue(s)|Wed|Thu(rs)|Fri|Sat|Sun)\b)",
        r"(?P<datetime>(?i:January|February|March|April|May|June|July|August|September|October|November|December))",
        r'(?P<datetime>(?i:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec))',
        r"(?P<datetime>\d{4}(/|-)(?:0[1-9]|1[0-2])(/|-)(?:0[1-9]|[12][0-9]|3[01])\b)",
        r"(?P<datetime>(?:[01][0-9]|2[0-3])(:)[0-5][0-9](:)(?:[0-5][0-9]|60)\b)",
        r"(?P<datetime>(?:[01][0-9]|2[0-3])(:)[0-5][0-9]\b)",
        r"(?P<datetime>\b(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}\b))",  # Standard datetime
        r"(?P<datetime>\b\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}\b)",  # Special datetime
        r"(?P<datetime>(?:[01][0-9]|2[0-3])(:)[0-5][0-9]\b)",
        r"(?P<datetime>(?:(\d{2}){2}\d{4} (\d{2}:){3}))",
    ]

    NETWORK_PATTERNS = [
        r"\b(?P<ipv4>(?:(\d{1,3}\.){3}\d{1,3}(\/\d{1,2}\b|\/|)))",  # IPv4 pattern
        r"\b(?P<ipv6>(?:[0-9a-fA-F]{1,4}|)(:)([0-9a-fA-F]{1,4}))\b",
        r"\b(?P<ipv6>(?:[0-9a-fA-F]{1,4}))?!\.(\d{1,3}\.){3}",
        r"\b(?P<ipv6>(?:[0-9a-fA-F]{1,4}))?!.*::.*::",
        r"(?P<ipv6>(?:([0-9a-fA-F]{4}(:)[0-9a-fA-F]{3}(::))))",
        r"\b(?P<ipv6>(?:[0-9a-fA-F]{4}(::)))",
        r"(?P<ipv6>(?:(::|:)(([0-9a-fA-F]{4}+)|\d\b|([0-9a-fA-F]{3})(:)[0-9a-fA-F](::))))",
        r"(?P<mac>(?:(\s|^)([0-9a-fA-F]{2}(\.|\-|:)){5}[0-9a-fA-F]{2}))",
        r"(?P<mac>(?:(\s|^)[0-9a-fA-F]{12}))",
    ]

    MISC_PATTERNS = [
        r"(?P<unix_path>(?:[ \t\n]|^)/(?:[a-zA-Z0-9_.-]+/)*[a-zA-Z0-9_.-]+)",
        r"(?P<windows_path>([a-zA-Z]:\\|\\\\)[\w\\.-]+)",  # Windows file paths
        r"(?P<email>[\w.-]+@([\w-]+\.)+[\w-]+)",  # Email addresses
        r"(?P<url>([a-zA-Z]+):\/\/[a-zA-Z0-9\-._~:/?#[\]@!$&'()*+,;=%]+)",
        r"(?P<ini>\[\w+\])",                     # INI sections
        r"(?P<json>{.*?}|\[.*?\])",              # JSON-like objects
        r"(?P<hex_number>\b0x[0-9a-fA-F]+\b)",   # Hexadecimal numbers
        r"(?P<env_var>\$[\w]+|%[\w]+%)",         # Environment variables
        r"(?P<uuid>\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b)",  # UUIDs
    ]
    highlights = NETWORK_PATTERNS + PHONE_PATTERNS + DATETIME_PATTERNS + MISC_PATTERNS
    """

    highlights = RENDERING_PATTERNS + ANALYZE_PATTERNS + MONTH_PATTERNS + TIME_PATTERNS + MACADDRESS_PATTERNS + DATE_PATTERNS + WEEK_PATTERNS + IPV6_PATTERNS + IPV4_PATTERNS + PHONE_PATTERNS + DATETIME_PATTERNS + MISC_PATTERNS


    def __init__(self, styles: Dict[str, Union[str, tuple]] = None):
        """
        Initialize the highlighter with user-defined styles.

        Args:
            custtom_styles (Dict[str, Union[str, tuple]]): A dictionary mapping pattern names to colors.
                Colors can be strings (e.g., "red", "blue", "#RRGGBB") or RGB tuples.
        """
        if styles is None:
            styles = self.DEFAULT_STYLES

        self._validate_highlights()
        self.styles = self._convert_styles(styles)
        self.base_style = "highlight."
        super().__init__()

    def _validate_highlights(self):
        """Validate all regex patterns in the highlights list."""
        for pattern in self.highlights:
            try:
                re.compile(pattern)  # Attempt to compile the regex
            except re.error as e:
                raise ValueError(f"Invalid regex pattern: {pattern}\nError: {e}")

    def _convert_styles(self, styles: Dict[str, Union[str, tuple]]) -> Dict[str, str]:
        """Convert user-provided styles into rich-compatible color styles."""
        converted_styles = {}
        for key, color in styles.items():
            if isinstance(color, tuple):  # RGB tuple
                converted_styles[key] = Color.from_rgb(*color).name
            elif isinstance(color, str):  # Hex or named color
                if color.startswith("#"):  # Hex color
                    converted_styles[key] = Color.parse(color).name
                elif color in ANSI_COLOR_NAMES:  # Named color
                    converted_styles[key] = color
                else:
                    raise ValueError(f"Unsupported color name for '{key}': {color}. Must be a valid ANSI color.")
            else:
                raise ValueError(f"Unsupported color format for '{key}': {color}. Use named colors, hex, or RGB tuples.")
        return converted_styles

    def get_console(self) -> Console:
        """Return a rich.Console instance with the highlighter and custom theme."""
        return Console(highlighter=self, theme=self.theme)

    def debug_patterns(self, test_text: str):
        """Test all patterns against a given text and show matched groups."""
        for idx, pattern in enumerate(self.highlights, start=1):
            print(f"Pattern {idx}: {pattern}")
            matches = re.finditer(pattern, test_text)
            for match in matches:
                print(f"  Match: {match.groupdict()}\n")

    @property
    def theme(self) -> Theme:
        """Generate a rich Theme from the user-defined styles."""
        return Theme({f"highlight.{key}": value for key, value in self.styles.items()})

def main():
    # Test text
    test_text = """
    The server at 192.168.1.1/24 manages local network traffic, while the router gateway is set to 10.0.0.1. A public web server is accessible at 203.0.113.45, and an alternative testing server uses 198.51.100.27/32. For internal systems, we use a small subnet like 172.16.0.0/16. Occasionally, a device might have a static IP of 192.0.2.10, and legacy systems still refer to older IPs like 127.0.0.1 for loopback or 8.8.8.8 for Google's DNS. A customer mentioned their IP being 169.254.1.5, which falls under the link-local range. Lastly, our firewall monitors traffic from 123.123.123.123, a public IP on a different network.

    2001:db8:3333:4444:5555:6666:7777:8888
2001:db8:3333:4444:CCCC:DDDD:EEEE:FFFF
:: (implies all 8 segments are zero)
2001:db8:: (implies that the last six segments are zero)
::1234:5678
::2
  ::0bff:db8:1:1
       ::0bff:db8:1:1
  2001:db8::ff34:5678
  2001:db8::1:5678
2001:0db8:0001:0000:0000:0ab9:C0A8:0102 (This can be compressed to eliminate leading zeros, as follows: 2001:db8:1::ab9:C0A8:102 )
    Send funds to support@example.org
       AA:BB:CC:DD:EE:FF
aa:bb:cc:dd:ee:ff
Aa:Bb:Cc:Dd:Ee:Ff
AABBCCDDEEFF
        aabbccddeeff   
AaBbCcDdEeFf
AA-BB-CC-DD-EE-FF
aa-bb-cc-dd-ee-ff
Aa-Bb-Cc-Dd-Ee-Ff
AA.BB.CC.DD.EE.FF
aa.bb.cc.dd.ee.ff
Aa.Bb.Cc.Dd.Ee.Ff
IPv6: fe80::1ff:fe23:4567:890a
    IPv6: fe80::1ff:fe23:4567:890a
    Give me a call at 1234567890
    Visit https://example.com for more info.
    Check the config at /etc/config/settings.ini or C:\\Windows\\System32\\drivers\\etc\\hosts.
    My phone number is 942 282 1445 or 954 224-3454 or (282) 445-4983
    +1 (203) 553-3294 and this 1-849-933-9938 
    Here is some JSON: {"key": "value"} or an array: [1, 2, 3].
    IPv4: 192.168.1.1, IPv6: fe80::1ff:fe23:4567:890a, MAC: 00:1A:2B:3C:4D:5E.
    Timestamp: 2023-11-18T12:34:56Z, Hex: 0x1A2B3C, Env: $HOME or %APPDATA%.
    UUID: 550e8400-e29b-41d4-a716-446655440000
     https://localhost/test.html
    ssh://localhost:808/test
    11/19/2024 01:21:23
     11/19/2024 01:21:23
    12.03.24 
    Jan March May July dec 
    mon monday tues fri sunday
    IPv6: fe80::1ff:fe23:4567:890a
    Short: fe80::
    Dual: 2001:db8::192.168.1.1
    Invalid: 123::abc::456
    """
    print("Testing default")
    # test load default style highlights
    try:
        # Create a highlighter and console
        colors = SymHighlighter()
        console = colors.get_console()
        console.print(test_text)
    except ValueError as e:
        print(f"Error: {e}")

    #colors.debug_patterns(test_text)
    import sys
    sys.exit()

    print("Testing configured styles")
    # Define styles for highlighting
    test_styles = {
        "email": "yellow",
        "url": "green",
        "unix_path": "cyan",
        "windows_path": "#1E90FF",  # Dodger Blue
        "ini": (255, 99, 71),  # Tomato (RGB tuple)
        "json": "#FFA500",  # Orange
        "ipv4": "blue",  # Incorrect name will now raise an error
        "ipv6": "plum4",  # Adjusted color name
        "mac": "#9ACD32",  # YellowGreen
        "timestamp": "gold1",
        "hex_number": "#FF4500",  # OrangeRed
        "env_var": "magenta",
        "uuid": "gold1",  # Adjusted color name
    }

    try:
        # Create a highlighter and console
        custom = SymHighlighter(test_styles)
        console = Console(highlighter=custom, theme=custom.theme)

        console.print(test_text)
        del console
    except ValueError as e:
        print(f"Error: {e}")

    print("Testing for failure in styles")
    # Check failure of color types
    bad_styles = {
            "ipv4": "pink", # does not exist
    }

    try:
        # Create a highlighter and console
        custom = SymHighlighter(bad_styles)
        console = Console(highlighter=custom, theme=custom.theme)

        console.print(test_text)
        del console
    except ValueError as e:
        print(f"Error: {e}")

    '''
    highlighter = SymHighlighter()
    con = highlighter.get_console()

    con.print(test_text)

    highlighter.debug_patterns(test_text)
    '''


if __name__ == "__main__":
    main()

import re

# Mapping of strftime directives to regex patterns
STRFTIME_TO_REGEX = {
    "%A": r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)",
    "%a": r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)",
    "%B": r"(?:January|February|March|April|May|June|July|August|September|October|November|December)",
    "%b": r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)",
    "%C": r"\d{2}",  # Century
    "%c": r".+",  # Locale's date and time representation (complex)
    "%D": r"\d{2}/\d{2}/\d{2}",  # Equivalent to %m/%d/%y
    "%d": r"(?:0[1-9]|[12][0-9]|3[01])",  # Day of the month (01-31)
    "%e": r"(?: [1-9]|[12][0-9]|3[01])",  # Day of the month (1-31, space-padded)
    "%F": r"\d{4}-\d{2}-\d{2}",  # Equivalent to %Y-%m-%d
    "%G": r"\d{4}",  # ISO 8601 year with century
    "%g": r"\d{2}",  # ISO 8601 year without century
    "%H": r"(?:[01][0-9]|2[0-3])",  # Hour (00-23)
    "%h": r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)",  # Same as %b
    "%I": r"(?:0[1-9]|1[0-2])",  # Hour (01-12)
    "%j": r"(?:00[1-9]|0[1-9][0-9]|[12][0-9]{2}|3[0-6][0-6])",  # Day of year (001-366)
    "%k": r"(?: [0-9]|1[0-9]|2[0-3])",  # Hour (0-23, space-padded)
    "%l": r"(?: [1-9]|1[0-2])",  # Hour (1-12, space-padded)
    "%M": r"[0-5][0-9]",  # Minute (00-59)
    "%m": r"(?:0[1-9]|1[0-2])",  # Month (01-12)
    "%n": r"\n",  # Newline
    "%p": r"(?:AM|PM|am|pm)",  # Locale's AM/PM
    "%R": r"(?:[01][0-9]|2[0-3]):[0-5][0-9]",  # Equivalent to %H:%M
    "%r": r"(?:0[1-9]|1[0-2]):[0-5][0-9]:[0-5][0-9] (?:AM|PM|am|pm)",  # Equivalent to %I:%M:%S %p
    "%S": r"(?:[0-5][0-9]|60)",  # Second (00-60)
    "%s": r"\d+",  # Seconds since the Epoch
    "%T": r"(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]",  # Equivalent to %H:%M:%S
    "%t": r"\t",  # Tab
    "%U": r"(?:[0-4][0-9]|5[0-3])",  # Week number (Sunday-starting, 00-53)
    "%u": r"[1-7]",  # Weekday (1=Monday, 7=Sunday)
    "%V": r"(?:0[1-9]|[1-4][0-9]|5[0-3])",  # ISO 8601 week number (01-53)
    "%v": r"(?: [1-9]|[12][0-9]|3[01])-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{4}",  # Equivalent to %e-%b-%Y
    "%W": r"(?:[0-4][0-9]|5[0-3])",  # Week number (Monday-starting, 00-53)
    "%w": r"[0-6]",  # Weekday (0=Sunday, 6=Saturday)
    "%X": r".+",  # Locale's time representation
    "%x": r".+",  # Locale's date representation
    "%Y": r"\d{4}",  # Year with century
    "%y": r"\d{2}",  # Year without century
    "%Z": r"[A-Za-z]+",  # Timezone name
    "%z": r"[+-]\d{4}",  # Timezone offset
    "%+": r".+",  # Locale's date and time representation
    "%%": r"%",  # Literal %
}
"""
class StrftimeRegex():
    def strftime_to_regex(patterns):
        new_pattern_list = []
        for pattern in patterns:
            # Match and process <datetime></datetime> tags
            def replace_strftime(match):
                inner_pattern = match.group(1)
                for strftime_directive, regex_equivalent in STRFTIME_TO_REGEX.items():
                    inner_pattern = inner_pattern.replace(strftime_directive, regex_equivalent)

                convert = f"(?P<datetime>{inner_pattern})"
                if re.compile(convert):
                    return convert

            # Replace all <datetime>...</datetime> tags with regex equivalents
            updated_pattern = re.sub(r"<datetime>(.*?)</datetime>", replace_strftime, pattern)
            new_pattern_list.append(updated_pattern)
        
        return new_pattern_list

    # Example usage
    patterns = [
        r"(?P<strftime>\b<datetime>%Y(/|.|-)%m(/|.|-)%d</datetime>\b)",
        r"(?P<strftime>\b<datetime>%H(:|.)%M(:|.)%S</datetime>\b)",
        r"(?P<strftime>\b<datetime>%H(:|.)%M</datetime>\b)",
        r"(?P<strftime>\b<datetime>%H(:|.)%M</datetime>\b)",
    ]

    converted_patterns = strftime_to_regex(patterns)
    for converted in converted_patterns:
        print(f"r\"{converted}\",")
"""
