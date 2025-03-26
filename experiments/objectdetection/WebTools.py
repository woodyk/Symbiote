#!/usr/bin/env python3
#
# WebTools.py

from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import hashlib
import re
import time
import os
import urllib.parse

class WebCrawler:
    def __init__(self, browser="firefox"):
        self.visited_urls = set()
        self.pages = {}  # Store page details in a dictionary
        self.match_count = 0  # Count of matched pages
        self.crawl_count = 0  # Count of crawled pages
        self.discarded_count = 0 # Count discarded pages
        self.browser = browser
        self.base_url = None 

        # Set up the WebDriver and make it run headlessly
        if self.browser.lower() == "chrome":
            options = webdriver.ChromeOptions()
            options.headless = True
            options.add_argument("--headless")
            self.driver = webdriver.Chrome(service=webdriver.chrome.service.Service(ChromeDriverManager().install()), options=options)
        elif self.browser.lower() == "firefox" or self.browser.lower() == "gecko":
            options = webdriver.FirefoxOptions()
            options.headless = True
            options.add_argument("--headless")
            self.driver = webdriver.Firefox(service=webdriver.firefox.service.Service(GeckoDriverManager().install(), log_path='/dev/null'), options=options)
        else:
            print(f"Unsupported browser: {self.browser}")
            return ""

    def scroll_down_page(self, pause_time=2, max_scrolls=10):
        """Scroll down the webpage progressively to load all dynamic content."""
        last_height = self.driver.execute_script("return document.body.scrollHeight")

        for _ in range(max_scrolls):
            # Scroll down to the bottom
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            
            # Wait for new content to load
            time.sleep(pause_time)

            # Calculate new scroll height and compare with the last scroll height
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                # If no new content is loaded, break the loop
                break
            last_height = new_height

    def pull_website_content(self, url, search_term=None, crawl=False, depth=None):
        if self.base_url is None:
            self.base_url = url

        self.search_term = search_term
        self.crawl = crawl

        try: 
            self.driver.get(url)
            self.scroll_down_page()  # Ensure full scroll before scraping
        except Exception as e:
            print(f"Error fetching the website content: {e}")
            return ""
        
        # Parse the fully scrolled page content with BeautifulSoup
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')

        # Remove all script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get the text content
        text = soup.get_text()
        text = re.sub(r'\n+', r'\n', text)
        text = re.sub(r'\s+', r' ', text)

        # Compute the md5 sum of the page content
        md5_sum = hashlib.md5(text.encode()).hexdigest()

        # If the md5 sum already exists in the pages dictionary, discard the page
        if md5_sum in self.pages:
            self.discarded_count += 1
            return ""

        # Check if the search_term is in the page content
        matched = False
        if self.search_term:
            search_variations = [self.search_term.lower(), self.search_term.upper(), self.search_term.capitalize()]
            if any(re.search(variation, text) for variation in search_variations):
                matched = True
                self.match_count += 1

        # Store the page details in the pages dictionary
        self.pages[md5_sum] = {
            'url': url,
            'content_type': self.driver.execute_script("return document.contentType"),
            'content': text,
            'matched': matched,
        }

        # Display a progress update
        self.crawl_count += 1
        progress = f"\x1b[2KCount: {self.crawl_count} Discarded: {self.discarded_count} Matches: {self.match_count} URL: {url}"
        print(progress, end='\r')

        # If crawl option is set to True, find all links and recursively pull content
        if self.crawl and (depth is None or depth > 0):
            links = soup.find_all('a')
            for link in links:
                href = link.get('href')
                absolute_url = urljoin(url, href)
                if absolute_url.startswith(self.base_url) and absolute_url not in self.visited_urls:  # Stay url sticky
                    self.visited_urls.add(absolute_url)
                    self.pull_website_content(absolute_url, search_term=self.search_term, crawl=True, depth=None if depth is None else depth - 1)

        return self.pages


class DuckDuckGoSearch:
    def __init__(self, browser="firefox"):
        self.browser = browser
        if self.browser.lower() == "chrome":
            options = webdriver.ChromeOptions()
            options.headless = True
            options.add_argument("--headless")
            self.driver = webdriver.Chrome(service=webdriver.chrome.service.Service(ChromeDriverManager().install()), options=options)
        elif self.browser.lower() == "firefox" or self.browser.lower() == "gecko":
            options = webdriver.FirefoxOptions()
            options.headless = True
            options.add_argument("--headless")
            self.driver = webdriver.Firefox(service=webdriver.firefox.service.Service(GeckoDriverManager().install(), log_path='/dev/null'), options=options)
        else:
            print(f"Unsupported browser: {self.browser}")
            return ""

    def search(self, query, num_results=5):
        search_url = f"https://duckduckgo.com/?q={query}&t=h_&ia=web"
        self.driver.get(search_url)

        # Wait for the search page to load
        time.sleep(2)

        # Parse the page content to find search results
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        results = []
        for link in soup.find_all('a', {'class': 'result__a'}, href=True)[:num_results]:
            results.append(urljoin(search_url, link['href']))

        return results

    def quit(self):
        self.driver.quit()


class GoogleSearch:
    def __init__(self, browser="firefox"):
        self.browser = browser
        if self.browser.lower() == "chrome":
            options = webdriver.ChromeOptions()
            options.headless = True
            options.add_argument("--headless")
            self.driver = webdriver.Chrome(service=webdriver.chrome.service.Service(ChromeDriverManager().install()), options=options)
        elif self.browser.lower() == "firefox" or self.browser.lower() == "gecko":
            options = webdriver.FirefoxOptions()
            options.headless = True
            options.add_argument("--headless")
            self.driver = webdriver.Firefox(service=webdriver.firefox.service.Service(GeckoDriverManager().install(), log_path='/dev/null'), options=options)
        else:
            print(f"Unsupported browser: {self.browser}")
            return ""

    def search(self, query, num_results=10):
        search_url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
        self.driver.get(search_url)

        # Wait for the search page to load
        time.sleep(2)

        # Parse the page content to find search results
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        results = []
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            if "/url?q=" in href:
                # Clean up the href to get the actual URL
                url = href.split("/url?q=")[1].split("&")[0]
                results.append(url)
                if len(results) >= num_results:
                    break

        return results

    def quit(self):
        self.driver.quit()


if __name__ == "__main__":
    # Search on DuckDuckGo for the top 5 results about "Python programming"
    search_term = "Python programming"
    ddg_search = DuckDuckGoSearch(browser='firefox')
    top_results = ddg_search.search(search_term, num_results=5)

    print("Top search results:")
    for i, result_url in enumerate(top_results, start=1):
        print(f"{i}: {result_url}")

    # Crawl the content of the top search results
    crawler = WebCrawler(browser='firefox')
    for url in top_results:
        print(f"\nCrawling URL: {url}")
        pages = crawler.pull_website_content(url, search_term=None, crawl=False)
        for md5, page in pages.items():
            print(f"URL: {page['url']}")
            print(f"Content Type: {page['content_type']}")
            print(f"Content: {page['content'][:500]}...")  # Print the first 500 chars for brevity

    ddg_search.quit()

    # Google Search for the top 5 results about "Python programming"
    google_search = GoogleSearch(browser='firefox')
    top_google_results = google_search.search("Python programming", num_results=5)

    print("Top Google search results:")
    for i, result_url in enumerate(top_google_results, start=1):
        print(f"{i}: {result_url}")

    # Crawl the content of the top search results
    crawler = WebCrawler(browser='firefox')
    for url in top_google_results:
        print(f"\nCrawling URL: {url}")
        pages = crawler.pull_website_content(url, search_term=None, crawl=False)
        for md5, page in pages.items():
            print(f"URL: {page['url']}")
            print(f"Content Type: {page['content_type']}")
            print(f"Content: {page['content'][:500]}...")  # Print the first 500 chars for brevity

    google_search.quit()

