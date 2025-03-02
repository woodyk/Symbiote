#!/usr/bin/env python3
#
# googleSearch.py

import os
import requests
import urllib
from bs4 import BeautifulSoup
from rich.console import Console
console = Console()
print = console.print
log = console.log
log("Loading symbiote googleSearch.")

class googleSearch:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY") 
        self.cse_id = os.getenv("GOOGLE_API_CX")

    def fetch_links(self, query, num_results=10):
        #query = urllib.parse.quote(query)
        search_url = f"https://www.googleapis.com/customsearch/v1?key={self.api_key}&cx={self.cse_id}&q={query}&num={num_results}"
        try:
            response = requests.get(search_url, timeout=5)
            response.raise_for_status()
            search_results = response.json()
            links = []
            for item in search_results.get('items', []):
                #log(f"{item['snippet']}\n\n")
                links.append(item['link'])

            return links
        except Exception as e:
            log(f"Failed to fetch search results: {e}")
            return None

    def fetch_text_from_urls(self, urls):
        all_text = str() 
        links = list()
        for url in urls:
            log(f"Fetching: {url}")
            try:
                response = requests.get(url, timeout=5)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                for link in soup.find_all('a', href=True):
                    links.append(link.get('href'))

                text = soup.get_text()
                all_text += text + " "
            except Exception as e:
                log(f"Failed to fetch page content from {url}: {e}")
                continue

        return all_text.strip()

if __name__ == "__main__":
    # Usage example
    search_engine = googleSearch()
    links = search_engine.fetch_links("site:w3schools.com python")
    all_text = search_engine.fetch_text_from_urls(links)
    log(all_text)
