#!/usr/bin/env bash
#
# File: pii.sh
# Author: Wadih Khairallah
# Description: 
# Created: 2025-03-03 14:09:21
# Modified: 2025-03-03 14:39:34

# EMAIL
find ./ -type f -exec cat {} + | LC_ALL=C grep -Eo '[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}' | sort -u

# URL
find ./ -type f -exec cat {} + | LC_ALL=C grep -Eo '(https?://|www\.)[a-zA-Z0-9._-]+(\.[a-zA-Z]{2,})+(/[a-zA-Z0-9._?=%&/-]*)?' | sort -u

# IPV4
find ./ -type f -exec cat {} + | LC_ALL=C grep -Eo '\b((25[0-5]|2[0-4][0-9]|1?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|1?[0-9][0-9]?)\b' | sort -u

# IPV6
#find ./ -type f -exec cat {} + | LC_ALL=C grep -Eo '([0-9a-fA-F]{1,4}:){1,7}[0-9a-fA-F]{1,4}' | sort -u

# MAC
find ./ -type f -exec cat {} + | LC_ALL=C grep -Eio '([0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}' | sort -u

# CC
find ./ -type f -exec cat {} + | LC_ALL=C grep -Eo '\b[3456][0-9]{3}[- ]?[0-9]{4}[- ]?[0-9]{4}[- ]?[0-9]{4}\b' | sort -u

# SS
find ./ -type f -exec cat {} + | LC_ALL=C grep -Eo '\b[0-9]{3}-[0-9]{2}-[0-9]{4}\b' | sort -u

# PHONE
find ./ -type f -exec cat {} + | LC_ALL=C grep -Eo '\b([0-9]{3}-[0-9]{3}-[0-9]{4}|\([0-9]{3}\) [0-9]{3}-[0-9]{4})\b' | sort -u

# DATES
find ./ -type f -exec cat {} + | LC_ALL=C grep -Eo '\b(0[1-9]|1[0-2])[/-](0[1-9]|[12][0-9]|3[01])[/-](19|20)[0-9]{2}\b' | sort -u

