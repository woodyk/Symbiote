#!/usr/bin/env python3
#
# sym_user_crawl.py
"""
Project Overview:
This script is designed for querying and checking the availability of usernames
across a wide array of websites and platforms. It leverages a JSON-based configuration
file (`sites.json` by default) that contains information about site-specific error
handling, patterns, and expected responses. This modular and extensible approach
ensures scalability and adaptability for various platforms and use cases.

Session Context:
The session focused on enhancing the functionality of a username availability checker,
optimizing it for modularity and user interaction. Key improvements included the
integration of progress tracking, a human-readable output format, and a dynamic 
mechanism for switching between raw and formatted output. The script was refined 
to support both command-line and programmatic usage, ensuring flexibility for 
interactive and automated workflows.

Key Features and Functionality:
1. **Username Availability Checker**:
   - Accepts a target username and checks its existence across platforms defined 
     in the JSON configuration file.
   - Handles error messages, status codes, and regex-based validation specific 
     to each platform.
   - Supports custom headers for sites requiring additional metadata.

2. **Dynamic Output Formats**:
   - JSON-style output for programmatic usage.
   - Human-readable columnar output using the `--columns` switch for enhanced
     readability in the terminal.

3. **Progress Tracking**:
   - Tracks and displays the number of completed site checks versus total checks.
   - Provides real-time updates during the execution.

4. **Asynchronous Execution**:
   - Allows programmatic queuing of username checks.
   - Supports querying the status of checks without blocking, facilitating 
     long-running processes.

5. **Error Handling and Resilience**:
   - Accounts for unreachable or non-responsive sites.
   - Categorizes results into clear states such as "Does not exist" or 
     specific error messages (e.g., 404 errors, name resolution failures).

Methodologies and Best Practices:
- Modular Design: The script is structured with functions for discrete tasks, 
  ensuring clarity and reusability.
- Use of External Libraries: `requests` for HTTP communication and `tabulate` 
  for formatted terminal output.
- Configurable Defaults: Default JSON file paths and CLI switches provide 
  flexibility without compromising ease of use.
- Extensibility: The design allows for straightforward additions of new 
  platforms or configuration tweaks.

Guidelines for Extending or Modifying:
- To add a new platform, update the JSON configuration file with the appropriate 
  URL patterns, error handling logic, and headers (if needed).
- For additional output formats, integrate new renderers in the output 
  functions (e.g., export to CSV, HTML).
- To refine progress tracking, consider integrating a progress bar library 
  such as `tqdm`.

Personal Style Alignment:
- The script reflects a balance between automation and user control, favoring 
  clean, structured outputs and logical workflows.
- Outputs are designed to be both human-readable and machine-parseable, 
  maintaining flexibility across use cases.
- The implementation prioritizes error handling and clear reporting, ensuring 
  robustness in diverse environments.

Reusable Prompt for Future Interactions:
- "Enhance the username availability checker by [specific feature or goal]. 
   Maintain its modular design, ensure compatibility with existing CLI switches, 
   and prioritize user feedback through dynamic outputs and progress updates."
"""

import requests
import json
import threading
import queue
import time
import argparse


class UsernameChecker:
    """
    A class to check username existence across multiple websites asynchronously.
    """

    def __init__(self, json_file="sites.json"):
        """
        Initializes the UsernameChecker instance.

        Parameters:
        - json_file: Path to the JSON file containing website configurations.
        """
        self.json_file = json_file
        self.requests_queue = queue.Queue()  # Queue to hold username checks
        self.results = {}  # Store results for each username
        self.lock = threading.Lock()  # Ensure thread-safe updates
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()

    def _load_site_data(self):
        """
        Loads the site data from the JSON file.
        """
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                site_data = json.load(f)
            if not isinstance(site_data, dict):
                raise ValueError("JSON data must be a dictionary.")
            return site_data
        except Exception as e:
            raise Exception(f"Error loading JSON file: {str(e)}")

    def _process_queue(self):
        """
        Continuously processes usernames from the queue in the background.
        """
        while True:
            try:
                username = self.requests_queue.get()
                if username is None:
                    break  # Exit signal for the worker thread

                # Process the username
                site_data = self._load_site_data()
                self._check_username(username, site_data)
                self.requests_queue.task_done()

            except Exception as e:
                print(f"Error processing queue: {str(e)}")

    def _check_username(self, username, site_data):
        """
        Checks the existence of a username across multiple sites.

        Parameters:
        - username: The username to check.
        - site_data: The loaded site configuration data.
        """
        results = {}
        total_sites = len(site_data)
        completed_sites = 0

        for site, data in site_data.items():
            url_template = data.get("url")
            if not url_template:
                results[site] = "Skipped: Missing URL template"
                completed_sites += 1
                continue

            url = url_template.format(username)
            error_type = data.get("errorType", "status_code")
            error_msgs = data.get("errorMsg", [])
            headers = data.get("headers", {})

            try:
                response = requests.get(url, headers=headers, timeout=10)
                response_content = response.text

                if error_type == "status_code":
                    results[site] = "Exists" if response.status_code == 200 else "Does not exist"

                elif error_type == "message":
                    if any(error_msg in response_content for error_msg in error_msgs):
                        results[site] = "Does not exist"
                    else:
                        results[site] = "Exists"
                else:
                    results[site] = "Unknown error type"

            except requests.exceptions.RequestException as e:
                results[site] = f"Error: {str(e)}"

            completed_sites += 1

            # Update progress in a thread-safe manner
            with self.lock:
                self.results[username] = {
                    "status": "In Progress" if completed_sites < total_sites else "Completed",
                    "completed_sites": completed_sites,
                    "total_sites": total_sites,
                    "results": results,
                }

    def add_username_check(self, username):
        """
        Adds a username to the queue for checking.

        Parameters:
        - username: The username to check.
        """
        with self.lock:
            if username not in self.results:
                self.results[username] = {
                    "status": "Queued",
                    "completed_sites": 0,
                    "total_sites": 0,
                    "results": {}
                }
        self.requests_queue.put(username)

    def get_status(self, username):
        """
        Returns the current status of the username check.

        Parameters:
        - username: The username to query.

        Returns:
        - Status dictionary containing progress and results.
        """
        with self.lock:
            return self.results.get(username, {"status": "Not Found"})

    def stop(self):
        """
        Stops the background worker thread.
        """
        self.requests_queue.put(None)
        self.worker_thread.join()

import argparse
from tabulate import tabulate

def print_human_readable(results):
    """
    Print results in a human-readable column format.
    """
    table_data = []
    for site, result in results.items():
        table_data.append([site, result])

    headers = ["Site", "Status"]
    #print(tabulate(table_data, headers, tablefmt="grid"))
    #print(tabulate(table_data, headers, tablefmt="grid"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check username availability across multiple sites.")
    parser.add_argument("--json", type=str, help="Path to JSON configuration file", default="sites.json")
    parser.add_argument("--columns", action="store_true", help="Display results in a human-readable column format")
    parser.add_argument("username", type=str, help="Username to check")

    args = parser.parse_args()

    checker = UsernameChecker(args.json)
    checker.add_username_check(args.username)

    while True:
        status = checker.get_status(args.username)

        if args.columns:
            # Print human-readable table for progress
            print(f"\nStatus for {args.username} (Progress: {status['completed_sites']}/{status['total_sites']}):")
            print_human_readable(status["results"])
        else:
            # Default JSON-style output
            print(f"Status for {args.username}: {status}")

        if status["status"] == "Completed":
            if args.columns:
                print("\nFinal Results:")
                print_human_readable(status["results"])
            else:
                print(f"Final Results: {status['results']}")
            break

        time.sleep(2)

def main():
    """
    Entry point for command-line usage.
    """
    parser = argparse.ArgumentParser(description="Check username existence on multiple websites.")
    parser.add_argument("username", type=str, help="Username to check.")
    parser.add_argument(
        "--json",
        type=str,
        default="sites.json",
        help="Path to the JSON file containing website configuration. Default: 'sites.json'",
    )
    args = parser.parse_args()

    try:
        checker = UsernameChecker(args.json)
        checker.add_username_check(args.username)

        # Check status periodically
        while True:
            status = checker.get_status(args.username)
            print(type(status))
            print(f"Status for {args.username}: {status}")
            if status["status"] == "Completed":
                break
            time.sleep(2)

        # Stop the worker thread gracefully
        checker.stop()

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()

