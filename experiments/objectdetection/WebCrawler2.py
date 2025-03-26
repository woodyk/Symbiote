#!/usr/bin/env python3
#
# WebCrawler2.py

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin

class WebCrawler:
    def __init__(self, depth=1, scroll_pause_time=2):
        self.depth = depth
        self.scroll_pause_time = scroll_pause_time
        self.visited_urls = set()
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
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

    def _get_links_from_page(self, url, html):
        """Extracts all the links from the given HTML page."""
        soup = BeautifulSoup(html, 'html.parser')
        links = set()
        for link in soup.find_all('a', href=True):
            abs_url = urljoin(url, link['href'])
            if urlparse(abs_url).scheme in ['http', 'https']:
                links.add(abs_url)
        return links

    def _crawl_page(self, url, current_depth):
        """Crawls a page and collects all text and links."""
        if current_depth > self.depth or url in self.visited_urls:
            return
        
        print(f"Crawling: {url}")
        self.visited_urls.add(url)
        
        page_data = self._get_page_data(url)
        if page_data is None:
            return
        
        links = self._get_links_from_page(url, page_data)
        
        if current_depth < self.depth:
            for link in links:
                self._crawl_page(link, current_depth + 1)

        return page_data

    def crawl(self, url):
        """Initiates crawling from the given URL."""
        return self._crawl_page(url, current_depth=1)

    def _search_google(self, query):
        """Searches Google and returns a list of result links."""
        self.driver.get(f"https://www.google.com/search?q={query}")
        time.sleep(2)  # Wait for search results
        self._scroll_to_bottom()  # Ensure all search results are loaded
        results = self.driver.find_elements(By.CSS_SELECTOR, 'a')
        links = [result.get_attribute('href') for result in results if result.get_attribute('href') and 'url?q=' in result.get_attribute('href')]
        return links

    def _search_duckduckgo(self, query):
        """Searches DuckDuckGo and returns a list of result links."""
        self.driver.get(f"https://duckduckgo.com/?q={query}")
        time.sleep(2)
        self._scroll_to_bottom()  # Ensure all search results are loaded
        results = self.driver.find_elements(By.CSS_SELECTOR, 'a.result__url')
        links = [result.get_attribute('href') for result in results]
        return links

    def search_and_crawl(self, query, search_engine='google'):
        """Performs a search on the given engine and crawls the result links."""
        if search_engine == 'google':
            search_results = self._search_google(query)
        elif search_engine == 'duckduckgo':
            search_results = self._search_duckduckgo(query)
        else:
            raise ValueError("Search engine not supported: use 'google' or 'duckduckgo'")
        
        for link in search_results:
            self.crawl(link)

    def close(self):
        """Shuts down the Selenium driver."""
        self.driver.quit()

# Example usage:
crawler = WebCrawler(depth=2)
crawler.search_and_crawl("Python programming", search_engine='google')
crawler.close()

