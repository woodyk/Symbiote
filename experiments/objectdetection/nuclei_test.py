#!/usr/bin/env python3
#
# nuclei_test.py

import subprocess

def run_nuclei_scan(domain: str) -> str:
    # Check if nuclei is installed
    try:
        subprocess.run(['nuclei', '-version'], capture_output=True, text=True, check=True)
    except FileNotFoundError:
        print("Error: nuclei is not installed or not in the system's PATH.")
        return None
    except subprocess.CalledProcessError:
        print("Error: failed to check nuclei version.")
        return None

    # Run the nuclei scan
    try:
        # Construct the command to run nuclei with the target domain
        command = ['nuclei', '-u', domain, '-c', '100', '-j']

        # Run the command using subprocess and capture the output
        result = subprocess.run(command, capture_output=True, text=True)

        # Check if the command was successful (non-zero return code means error)
        if result.returncode != 0:
            print(f"Error: Nuclei scan failed with code {result.returncode}.")
            print(result.stderr)
            return None

        # Return the output if successful
        return result.stdout

    except Exception as e:
        # In case any other unexpected error occurs, print the error
        print(f"An unexpected error occurred: {str(e)}")
        return None

# Example usage:
domain_to_scan = "smallroom.com"
scan_result = run_nuclei_scan(domain_to_scan)
if scan_result:
    print(scan_result)

