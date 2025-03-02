#!/usr/bin/env python3
#
# sym_google.py 

import os
import requests
from urllib.parse import quote
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from rich.console import Console
from rich.panel import Panel

console = Console()
print = console.print
log = console.log
log("Loading symbiote googleSearch.")

class GoogleSearch:
    def __init__(self):
        """
        Initializes the googleSearch object with API credentials.
        """
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.cse_id = os.getenv("GOOGLE_API_CX")
        self.session = requests.Session()

    def google_search(self, query, num_results=10):
        """
        Fetches search result links from Google Custom Search Engine.

        Args:
            query (str): The search query.
            num_results (int): Number of results to fetch.

        Returns:
            list: List of links from the search results.
        """
        query_encoded = quote(query)
        search_url = f"https://www.googleapis.com/customsearch/v1?key={self.api_key}&cx={self.cse_id}&q={query_encoded}&num={num_results}"
        try:
            response = self.session.get(search_url, timeout=5)
            response.raise_for_status()
            search_results = response.json()
            return search_results
        except Exception as e:
            log(f"Failed to fetch search results: {e}")
            return []

    def display_google_search_results(self, response):
        """
        Display Google Custom Search results in the terminal using rich.

        Args:
            response (dict): JSON response from the Google Custom Search API.
        """
        # Extract search metadata
        search_info = response.get('searchInformation', {})
        total_results = search_info.get('formattedTotalResults', 'N/A')
        search_time = search_info.get('formattedSearchTime', 'N/A')
        
        panels = []

        # Create a summary panel
        summary = f"[bold cyan]Google Search Results[/bold cyan]\n"
        summary += f"[green]Total Results:[/green] {total_results}\n"
        summary += f"[green]Search Time:[/green] {search_time} seconds"
        print(Panel(summary, title="Summary", title_align="left", expand=False))

        # Display each search result in its own panel
        items = response.get('items', [])
        for index, item in enumerate(items, start=1):
            title = item.get('title', 'No Title')
            link = item.get('link', 'No Link')
            domain = item.get('displayLink', 'No Domain')
            snippet = item.get('snippet', 'No Snippet')

            # Format the result details
            content = (
                f"[bold]{title}[/bold]\n"
                f"[green]{link}[/green]\n"
                f"[yellow]Domain:[/yellow] {domain}\n\n"
                f"{snippet}"
                f""
            )

            print(Panel(content, title=f"Result {index}", title_align="left", expand=True))
        return



    def fetch_text_from_url(self, url):
        """
        Fetches and cleans text content from a single URL.

        Args:
            url (str): URL to fetch content from.

        Returns:
            dict: A dictionary containing the URL, cleaned text, and links.
        """
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove script and style elements
            for element in soup(['script', 'style']):
                element.decompose()

            # Extract and clean text
            text = soup.get_text(separator=' ')
            cleaned_text = ' '.join(text.split())  # Normalize whitespace

            # Extract links
            links = [link.get('href') for link in soup.find_all('a', href=True)]

            return {
                "url": url,
                "text": cleaned_text,
                "links": links
            }
        except Exception as e:
            log(f"Failed to fetch page content from {url}: {e}")
            return {
                "url": url,
                "text": "",
                "links": []
            }

    def fetch_text_from_urls(self, urls):
        """
        Fetches text content from multiple URLs in parallel.

        Args:
            urls (list): List of URLs to fetch content from.

        Returns:
            list: A list of dictionaries containing text and links for each URL.
        """
        results = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(self.fetch_text_from_url, urls))
        return results

    def close(self):
        """
        Closes the requests session.
        """
        self.session.close()


if __name__ == "__main__":
    # Usage example
    search_engine = GoogleSearch()
    query = "site:w3schools.com python"
    search_results = search_engine.google_search(query)
    rich_render = search_engine.display_google_search_results(search_results)
    """
    if links:
        results = search_engine.fetch_text_from_urls(links)
        for result in results:
            log(f"URL: {result['url']}")
            log(f"Extracted Text: {result['text'][:500]}...")  # Print first 500 characters of text
            log(f"Extracted Links: {result['links'][:5]}")    # Print first 5 links
    else:
        log("No links found.")

    """
    search_engine.close()

