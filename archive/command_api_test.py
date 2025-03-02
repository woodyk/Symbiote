#!/usr/bin/env python3
#
# tt.py

import os
import sys
import time
import symbiote.chat as sym 

symbiote = sym.symchat(working_directory=os.getcwd(), debug=False, output=False)

query = "help::"
response = symbiote.process_input(user_input=query)

print(response) # truncated_conversation
