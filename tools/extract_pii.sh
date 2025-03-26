#!/usr/bin/env bash
#
# File: extract_pii.sh
# Author: Wadih Khairallah
# Description: Extracts email addresses, website URLs, and phone numbers from files and prints to terminal.
# Created: 2025-03-03 14:12:00

# Set the directory to scan (default: current directory)
DIR=${1:-.}

echo "🔍 Scanning files in: $DIR"

# Find and extract unique email addresses
echo -e "\n📧 Unique Email Addresses:"
find "$DIR" -type f -exec cat {} + | LC_ALL=C grep -Eo '[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}' | sort -u

# Find and extract unique website URLs (http and https only)
echo -e "\n🌐 Unique Website URLs:"
find "$DIR" -type f -exec cat {} + | LC_ALL=C grep -Eo 'https?://[a-zA-Z0-9./?=_-]+' | sort -u

# Find and extract unique phone numbers (various formats)
echo -e "\n📞 Unique Phone Numbers:"
find "$DIR" -type f -exec cat {} + | LC_ALL=C grep -Eo '\+?[0-9]{1,4}?[-. (]?[0-9]{2,4}[-. )]?[0-9]{2,4}[-. ]?[0-9]{2,9}' | sort -u
