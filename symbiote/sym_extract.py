#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: sym_extract.py
# Author: Wadih Khairallah
# Description: 
# Created: 2024-11-27 20:38:27
#!/usr/bin/env python3
#
# sym_extract.py

import re
import sys
import json

from rich.console import Console
from rich.highlighter import RegexHighlighter
from rich.theme import Theme
from rich.color import Color, ANSI_COLOR_NAMES
from typing import Dict, Union


PHONE_PATTERNS = [
    r"\b(?P<phone_number>\+?\d{1,3}[-.\s]?\(?\d{1,3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b",
]

MONTH_PATTERNS = [
    r"\b(?P<month>(?i:January|February|March|April|May|June|July|August|September|October|November|December))\b",
    r"\b(?P<month>(?i:Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec))\b",
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
TERM_PATTERNS = [ ]
"""
TERM_PATTERNS = [
    r"\b(?P<protocol>(?i:dns|http|https|ftp|ssh|tcp|ip|udp|ssl|smtp|telnet|ipv4|ipv6|icmp|arp|pop3|imap|smb|nfs|dhcp|tftp|ldap|snmp|sctp|bgp|ospf|rsvp|rtp|rdp|lldp|sip|pim|mpls|gre|ppp|pptp|l2tp|ipsec|nat|stp|rip|eigrp|http2|spdy|quic|sctp|ntp|kerberos|radius|gopher|mqtt|coap|amqp))\b",
    r"\b(?P<language>(?i:python|javascript|java|c\+\+|c#|ruby|perl|php|swift|kotlin|go|rust|typescript|scala|r|matlab|bash|shell|powershell|html|css|sql|dart|elixir|erlang|haskell|clojure|f#|visual\s*basic|fortran|cobol|assembly|lisp|prolog|vhdl|verilog|sas|groovy|smalltalk|tcl|awk|ada|pascal|delphi|nim|julia|crystal|objective-c|postscript|apl|scratch|logo|abap|pl/sql))\b",
    r"\b(?P<file_format>(?i:txt|pdf|doc|docx|xls|xlsx|csv|json|xml|yaml|html|md|ppt|pptx|png|jpg|jpeg|gif|bmp|tiff|svg|mp3|wav|flac|aac|ogg|mp4|mkv|avi|mov|wmv))\b",
    r"\b(?P<os>(?i:windows|linux|macos|ios|android|ubuntu|debian|redhat|centos|fedora|arch|kali|alpine|unix|solaris|bsd|freebsd))\b",
    r"\b(?P<web_tech>(?i:react|angular|vue|svelte|next\.js|nuxt\.js|node\.js|django|flask|express|spring|rails|laravel|webpack|gulp|grunt|babel|eslint|graphql|rest|soap|ajax))\b",
    r"\b(?P<cloud_devops>(?i:aws|azure|gcp|digitalocean|heroku|jenkins|travis|circleci|gitlab-ci|github-actions|docker|kubernetes|helm|podman|rancher|prometheus|grafana|nagios|zabbix|ansible|terraform|puppet|chef))\b",
    r"\b(?P<database>(?i:mysql|postgresql|sqlite|oracle|mssql|mongodb|cassandra|couchdb|redis|dynamodb|influxdb|timescale|opentsdb|neo4j|janusgraph|dgraph))\b",
    r"\b(?P<cybersecurity>(?i:firewall|vpn|antivirus|malware|ransomware|phishing|zero-day|nmap|metasploit|wireshark|kali|burp|nessus|ossec|tls|ssl|ipsec|oauth|saml|fido2|nist|iso27001|cobit))\b",
    r"\b(?P<hardware>(?i:cpu|gpu|ram|ssd|hdd|motherboard|psu|intel|amd|nvidia|asus|dell|lenovo|hp|keyboard|mouse|monitor|printer|router))\b",
    r"\b(?P<networking>(?i:router|switch|modem|firewall|access point|lan|wan|vpn|dns|dhcp|nat|subnet))\b",
    r"\b(?P<ml_ai>(?i:tensorflow|keras|pytorch|sklearn|xgboost|lightgbm|cnn|rnn|gan|transformer|bert|gpt|svm|mlflow|kubeflow|airflow))\b",
    r"\b(?P<version_control>(?i:git|svn|mercurial|cvs|bitbucket|github|gitlab))\b",
    r"\b(?P<mobile_dev>(?i:swift|kotlin|objective-c|flutter|react-native|xamarin|cordova|ionic))\b",
    r"\b(?P<blockchain>(?i:bitcoin|ethereum|dogecoin|litecoin|solana|polkadot|hyperledger|cosmos|nft|smart-contract|dapp|defi))\b",
    r"\b(?P<game_dev>(?i:unity|unreal|godot|cryengine|c#|c\+\+|lua|python|panda3d|monogame))\b" 
]
"""

IPV4_PATTERNS = [
    r"\b(?P<ipv4>(?:(\d{1,3}\.){3}\d{1,3}(\/\d{1,2}\b|\/|)))",  # IPv4 pattern
]

MACADDRESS_PATTERNS = [
    r"\b(?P<mac>([0-9a-fA-F]{2}[:-]){5}[0-9a-fA-F]{2})\b",
    r"\b(?P<mac>[0-9a-fA-F]{12})\b",
    r"\b(?P<mac>([0-9a-fA-F]{4}\.){2}[0-9a-fA-F]{4})",
]

MISC_PATTERNS = [
    r"(?P<unix_path>(?:[ \t\n]|^)/(?:[a-zA-Z0-9_.-]+/)*[a-zA-Z0-9_.-]+)",
    r"(?P<windows_path>([a-zA-Z]:\\|\\\\)[\w\\.-]+)",  # Windows file paths
    r"(?P<email>[\w.-]+@([\w-]+\.)+[\w-]+)",  # Email addresses
    r"(?P<url>([a-zA-Z]+):\/\/[a-zA-Z0-9\-._~:/?#[\]@!$&'()*+,;=%]+)",
    r"(?P<ini>\[\w+\])",                     # INI sections
    #r"(?P<json>{.*?}|\[.*?\])",              # JSON-like objects
    r"(?P<hex_number>\b0x[0-9a-fA-F]+\b)",   # Hexadecimal numbers
    r"(?P<env_var>\$[\w]+|%[\w]+%)",         # Environment variables
    r"(?P<uuid>\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b)",  # UUIDs
]

DOCUMENT_ANALYSIS_PATTERNS = [
    # Interrogative context with surrounding words
    r"(?P<interrogative_context>(?:\b\w+\b\s+){2,5}\b(how|why|what|when|where|who)\b(?:\s+\b\w+\b){2,5})",
    
    # Reporting statements with surrounding context
    r"(?P<reporting_statement>(?:\b\w+\b\s+){2,5}\b\w+(?:\s\w+)*\s+(said|stated|reported|mentioned|claimed|explained|noted|suggested)\b(?:\s+\".*?\"|\s+\b.*?\b[.!?]))",
    
    # Entity names after relational prepositions with meaningful context
    r"(?P<entity_name>(?:\b\w+\b\s+){2,5}\b(by|from|to|with|about|for)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)(?:\s+\b\w+\b){2,5})",
    
    # All-caps phrases with contextual information
    r"(?P<all_caps_phrase>(?:\b\w+\b\s+){2,5}\b[A-Z]{2,}(?:\s+[A-Z]{2,})*\b(?:\s+\b\w+\b){2,5})",
    
    # Quoted text with nearby context
    r"(?P<quoted_text>(?:\b\w+\b\s+){2,5}\".*?\"(?:\s+\b\w+\b){2,5})",
    
    # Action contexts with surrounding context
    r"(?P<action_context>(?:\b\w+\b\s+){2,5}\b\w+(?:\s+\w+)*\s+(completed|developed|initiated|threatened|investigated)\s+\b.*?[.!?](?:\s+\b\w+\b){2,5})",
    
    # Possessive context with nearby context
    r"(?P<possessive_context>(?:\b\w+\b\s+){2,5}(\b[A-Z][a-z]+(?:'s|’s)\s+\w+)|\b(owned by|reported by|handled by)\s+([A-Z][a-z]+)(?:\s+\b\w+\b){2,5})",
 
    # Proper nouns with meaningful surroundings
    r"(?P<proper_noun>(?:\b\w+\b\s+){2,5}\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b(?:\s+\b\w+\b){2,5})",
    
    # Adjectives indicating criticality or importance with nearby context
    r"(?P<adjective_context>(?:\b\w+\b\s+){2,5}\b(critical|urgent|significant|dangerous|illegal|fraudulent)\b\s+\w+(?:\s+\b\w+\b){2,5})",
    
    # Question context with sufficient surrounding words
    r"(?P<question_context>(?:\b\w+\b\s+){2,5}\b(who|what|why|how|where|when)\b(?:\s+\b\w+\b){2,5}[?])",
    
    # Numbers with measurement terms and context
    r"(?P<number_context>(?:\b\w+\b\s+){2,5}\b\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:\s+(units|percent|completion|days|weeks))?\b(?:\s+\b\w+\b){2,5})",
    
    # Threat-related phrases with meaningful context
    r"(?P<threat_context>(?:\b\w+\b\s+){2,5}(threat|risk|danger|exploit)(?:\s+\b\w+\b){2,5}[.!?])",
]

ANALYZE_PATTERNS = [
    r"(?P<analyze>\b[A-Za-z0-9]{20,100}\b)",    
    r"\b(?P<analyze>(?=[A-Za-z0-9]*[A-Za-z])(?=[A-Za-z0-9]*\d)[A-Za-z0-9]{6,20}\b)",
    r"\b(?P<analyze>\d{8,20}\b)",
    r"\b(?P<analyze>[A-Z]{8,20}\b)",
    r"\b(?P<analyze>(?:[A-Za-z0-9]{2}[,:|.-]([A-Za-z0-9]{2}|)){4,20})\b",
]

PATTERNS = DOCUMENT_ANALYSIS_PATTERNS + TERM_PATTERNS + ANALYZE_PATTERNS + MONTH_PATTERNS + TIME_PATTERNS + MACADDRESS_PATTERNS + DATE_PATTERNS + WEEK_PATTERNS + IPV6_PATTERNS + IPV4_PATTERNS + PHONE_PATTERNS + DATETIME_PATTERNS + MISC_PATTERNS


def extract_pii(content):
    results = extract_and_stitch_matches(content)
    summary = summarize(content)
    results['summary'] = summary
    return results


def extract_and_stitch_matches(text, patterns=PATTERNS):
    """
    Extract labeled matches using regex patterns and stitch contiguous matches
    into full objects. Newline characters act as absolute boundaries.
    Deduplicates entries before returning results.

    Args:
        text (str): The input text to analyze.
        patterns (List[str]): List of regex patterns with labeled groups.

    Returns:
        Dict[str, List[str]]: Extracted and stitched matches grouped by labels.
    """
    import re

    # Sanitize Text Input
    text = ''.join(char for char in text if char.isprintable())
    text = re.sub(r'\n+', ' ', text)  # Replace multiple newlines with a space
    text = re.sub(r' +', ' ', text)   # Replace multiple spaces with a single space

    # Dictionary to hold matches grouped by label
    matches_by_label = {}

    compiled_patterns = []
    errors = []
    # Verify patterns before continuing
    for pattern in patterns:
        try:
            compiled_patterns.append(re.compile(pattern))
        except re.error as e:
            errors.append((pattern, str(e)))

    if errors:
        for error in errors:
            print(error)
        return

    # Iterate through all patterns
    for pattern in patterns:
        regex = re.compile(pattern)
        for match in regex.finditer(text):
            for label, value in match.groupdict().items():
                if value:
                    if label not in matches_by_label:
                        matches_by_label[label] = []

                    # Avoid duplicate entries
                    if not any(existing["value"] == value for existing in matches_by_label[label]):
                        matches_by_label[label].append({
                            "value": value,
                            "start": match.start(),
                            "end": match.end()
                        })

    # Stitch contiguous matches for each label
    stitched_results = {}
    for label, matches in matches_by_label.items():
        if not matches:
            continue

        # Sort matches by start position
        matches = sorted(matches, key=lambda m: m["start"])
        stitched = []
        current_match = matches[0]

        for i in range(1, len(matches)):
            next_match = matches[i]
            # Get the intervening text between current and next match
            intervening_text = text[current_match["end"]:next_match["start"]]

            # Check for newline in the intervening text or non-contiguity
            if "\n" in intervening_text or current_match["end"] < next_match["start"]:
                # Add the current match as a separate entity
                stitched.append(current_match)
                current_match = next_match
            else:
                # Extend the current match to include the next match
                current_match["value"] += intervening_text + next_match["value"]
                current_match["end"] = next_match["end"]

        # Add the final match
        stitched.append(current_match)

        # Store stitched values, ensuring no duplicates
        stitched_results[label] = list(set(m["value"].strip() for m in stitched))

    return stitched_results

def summarize(content):
    from spacy.lang.en.stop_words import STOP_WORDS
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.lsa import LsaSummarizer
    tokenizer = Tokenizer("english")
    parser = PlaintextParser.from_string(content, tokenizer)
    stop_words = list(STOP_WORDS)
    summarizer = LsaSummarizer()
    summarizer.stop_words = stop_words
    summary_list = summarizer(parser.document, 10)
    summary = " ".join(str(sentence) for sentence in summary_list)

    return summary


content = """
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
One Apr 4th 1922 at 12pm or 12:30 pm or 10:20am
Here is some JSON: {"key": "value"} or an array: [1, 2, 3].
IPv4: 192.168.1.1, IPv6: fe80::1ff:fe23:4567:890a, MAC: 00:1A:2B:3C:4D:5E.
Timestamp: 2023-11-18T12:34:56Z, Hex: 0x1A2B3C, Env: $HOME or %APPDATA%.
UUID: 550e8400-e29b-41d4-a716-446655440000
 https://localhost/test.html
   July 23rd 2023
ssh://localhost:808/test
11/19/2024 01:21:23
 11/19/2024 01:21:23
   2024/8/29
12.03.24 
Jan March May July dec 
mon monday tues fri sunday
IPv6: fe80::1ff:fe23:4567:890a
Short: fe80::
Dual: 2001:db8::192.168.1.1
Invalid: 123::abc::456
"""
if len(sys.argv) > 1:
    file = sys.argv[1]
    with open(file, 'r') as fh:
        content = fh.read()

# Extract and stitch matches
pii_data = extract_pii(content)
print(json.dumps(pii_data, indent=4))
