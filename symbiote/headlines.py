#!/usr/bin/env python3
#
# headlines.py

from rich.console import Console
console = Console()
print = console.print
log = console.log
log("Loading symbiote headlines.")

import requests
from bs4 import BeautifulSoup

class getHeadlines:
    def __init__(self):
        # Define news sites
        self.news_sites = {
            'BBC': 'https://www.bbc.com/news',
            'CNN': 'https://www.cnn.com',
            'Fox News': 'https://www.foxnews.com',
            'The Guardian': 'https://www.theguardian.com',
            'MSNBC': 'https://msnbc.com',
            'NPR': 'https://www.npr.org',
            'Aljazeera': 'https://www.aljazeera.com',
        }

    def scrape(self):
        # Scrape headlines
        headlines = {}
        for name, url in self.news_sites.items():
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # This is a simplified selector, and may need to be adjusted per site
            news_headlines = soup.select('h3, h2')
            headlines[name] = [headline.text.strip() for headline in news_headlines[:10]]  # Top 10 headlines

        headline_str = ""
        # Example of printing out headlines from each site
        for site, headlines_list in headlines.items():
            headline_str += f"Top headlines from {site}:\n"
            for headline in headlines_list:
                headline_str += f"- {headline}\n"

        return headline_str

if __name__ == "__main__":
    gh = getHeadlines()
    result = gh.scrape()
    log(result)

