#!/usr/bin/env python3
#
# WebCrawlerSearch.py

import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin

class WebCrawler:
    def __init__(self, google_api_key=None, google_cse_id=None, bing_api_key=None, scroll_pause_time=2):
        self.scroll_pause_time = scroll_pause_time
        self.visited_urls = set()
        self.google_api_key = google_api_key
        self.google_cse_id = google_cse_id
        self.bing_api_key = bing_api_key

        # Set up Chrome options to ensure JavaScript is enabled and headless mode is used
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--enable-javascript")  # Enable JavaScript
        self.driver = webdriver.Chrome(service=Service(), options=chrome_options)

    def _scroll_to_bottom(self):
        """Scrolls down to the bottom of the page to load all dynamic content."""
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        while True:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(self.scroll_pause_time)
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

    def _get_page_data(self, url):
        """Loads a webpage using Selenium, scrolls to the bottom, and returns the page source."""
        try:
            self.driver.get(url)
            time.sleep(2)  # Wait for initial page load
            self._scroll_to_bottom()  # Scroll to ensure all content is loaded
            page_source = self.driver.page_source
            return page_source
        except Exception as e:
            print(f"Failed to get page data for {url}: {str(e)}")
            return None

    def _clean_text(self, html):
        """Extracts and cleans text from HTML, removing unnecessary whitespace, CSS, and script tags."""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()  # Completely remove these tags
        
        text = soup.get_text(separator=' ')  # Get text and separate by space for better readability
        cleaned_text = ' '.join(text.split())  # Remove extra whitespace
        return cleaned_text

    def _get_links_from_page(self, url, html):
        """Extracts all the links from the given HTML page."""
        soup = BeautifulSoup(html, 'html.parser')
        links = set()
        for link in soup.find_all('a', href=True):
            abs_url = urljoin(url, link['href'])
            if urlparse(abs_url).scheme in ['http', 'https']:
                links.add(abs_url)
        return links

    def _crawl_page(self, url, current_depth, depth):
        """Crawls a page and collects all text and links."""
        if current_depth > depth or url in self.visited_urls:
            return
        
        print(f"Crawling: {url}")
        self.visited_urls.add(url)
        
        page_data = self._get_page_data(url)
        if page_data is None:
            return
        
        links = self._get_links_from_page(url, page_data)
        collected_text = self._clean_text(page_data)
        
        if current_depth < depth:
            for link in links:
                additional_text = self._crawl_page(link, current_depth + 1, depth)
                if additional_text:
                    collected_text += " " + additional_text

        return collected_text

    def crawl(self, url, depth=1):
        """Initiates crawling from the given URL and returns the clean text."""
        return self._crawl_page(url, current_depth=1, depth=depth)

    def get_text_from_url(self, url):
        """Pulls all the text from a single URL and returns the cleaned result."""
        page_data = self._get_page_data(url)
        if page_data:
            return self._clean_text(page_data)
        return None

    def _search_google_api(self, query, max_results):
        """Searches Google using the Google Search API and returns a list of result links."""
        if not self.google_api_key or not self.google_cse_id:
            raise ValueError("Google API key and CSE ID must be set to use Google Search API.")
        
        url = f"https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.google_api_key,
            "cx": self.google_cse_id,
            "q": query,
            "num": max_results
        }

        response = requests.get(url, params=params)
        if response.status_code != 200:
            raise Exception(f"Error fetching search results from Google API: {response.status_code}")
        
        results = response.json().get("items", [])
        links = [result["link"] for result in results]
        return links

    def _search_bing_api(self, query, max_results):
        """Searches Bing using the Bing Search API and returns a list of result links."""
        if not self.bing_api_key:
            raise ValueError("Bing API key must be set to use Bing Search API.")
        
        url = f"https://api.bing.microsoft.com/v7.0/search"
        headers = {
            "Ocp-Apim-Subscription-Key": self.bing_api_key
        }
        params = {
            "q": query,
            "count": max_results
        }

        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            raise Exception(f"Error fetching search results from Bing API: {response.status_code}")
        
        results = response.json().get("webPages", {}).get("value", [])
        links = [result["url"] for result in results]
        return links

    def _search_duckduckgo(self, query, max_results):
        """Searches DuckDuckGo and returns a list of result links."""
        self.driver.get(f"https://duckduckgo.com/?q={query}")
        time.sleep(2)
        self._scroll_to_bottom()  # Ensure all search results are loaded
        results = self.driver.find_elements(By.CSS_SELECTOR, 'a.result__url')
        links = [result.get_attribute('href') for result in results[:max_results]]
        return links

    def search_and_extract(self, query, search_engine='google', max_results=10):
        """Performs a search on the given engine, pulls text from each result, and returns the cleaned results."""
        if search_engine == 'google':
            search_results = self._search_google_api(query, max_results)
        elif search_engine == 'bing':
            search_results = self._search_bing_api(query, max_results)
        elif search_engine == 'duckduckgo':
            search_results = self._search_duckduckgo(query, max_results)
        else:
            raise ValueError("Supported search engines are 'google', 'bing', and 'duckduckgo'.")
        
        collected_text = ""
        for link in search_results:
            page_text = self.get_text_from_url(link)
            if page_text:
                collected_text += " " + page_text

        return collected_text.strip()

    def close(self):
        """Shuts down the Selenium driver."""
        self.driver.quit()

# Example usage:
# Replace with your Google API Key, CSE ID, and Bing API Key
google_api_key = "YOUR_GOOGLE_API_KEY"
google_cse_id = "YOUR_GOOGLE_CSE_ID"
bing_api_key = "YOUR_BING_API_KEY"

crawler = WebCrawler(google_api_key=google_api_key, google_cse_id=google_cse_id, bing_api_key=bing_api_key)

# Search using DuckDuckGo and extract text from top 5 results
search_text_ddg = crawler.search_and_extract("Python programming", search_engine='duckduckgo', max_results=5)
print("Search Text (DuckDuckGo):", search_text_ddg)

# Search using Bing API and extract text from top 5 results
#search_text_bing = crawler.search_and_extract("Python programming", search_engine='bing', max_results=5)

# Search using Google API and extract text from top 5 results
#search_text_google = crawler.search_and_extract("Python programming", search_engine='google', max_results=5)

# Get text from a single URL
page_text = crawler.get_text_from_url("https://openai.com")
print("Page Text:", page_text)

# Crawl a URL up to a certain depth
crawled_text = crawler.crawl("https://books.toscrape.com", depth=1)
print("Crawled Text:", crawled_text)

crawler.close()

#print("Search Text (Bing):", search_text_bing)
#print("Search Text (Google):", search_text_google)

