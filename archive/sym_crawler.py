#!/usr/bin/env python3
#
# webcrawler.py

from rich.console import Console
console = Console()
print = console.print
log = console.log

from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import hashlib
import re
from prompt_toolkit import print_formatted_text, ANSI
from prompt_toolkit.utils import get_cwidth
import requests

class WebCrawler:
    def __init__(self, browser="firefox"):
        self.pages = {}
        self.visited_urls = set()
        self.match_count = 0  # Count of matched pages
        self.crawl_count = 0  # Count of crawled pages
        self.discarded_count = 0 # Count discarded pages
        self.browser = browser
        self.base_url = None 
        self.found_links = set()

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
            log(f"Unsupported browser: {self.browser}")
            return ""

    def get_text_from_url(self, url):
        """
        Fetches a webpage from the given URL, extracts and sanitizes the text content.
        
        Parameters:
            url (str): The URL of the webpage to fetch.
        
        Returns:
            str: Sanitized text content of the webpage.
        """
        try:
            # Fetch the content from the URL
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            # Parse the HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for element in soup(['script', 'style']):
                element.decompose()
            
            # Extract text and sanitize
            text = soup.get_text(separator=' ')
            sanitized_text = ' '.join(text.split())  # Normalize whitespace
            
            return sanitized_text
        except requests.RequestException as e:
            return f"Error fetching the URL: {e}"
        except Exception as e:
            return f"Error processing the content: {e}"

    def pull_website_content(self, url=None, search_term=None, crawl=False, depth=None):
        if self.base_url is None:
            self.base_url = url

        self.found_links.add(url)

        self.search_term = search_term
        self.crawl = crawl

        try: 
            self.driver.get(url)
        except Exception as e:
            log(f"Error fetching the website content: {e}")
            return ""

        soup = BeautifulSoup(self.driver.page_source, 'html.parser')

        # Close the WebDriver 
        #self.driver.quit()

        # Remove all script and style elements
        scripts = []
        for script in soup.find_all("script"):
            if script.string is not None:
                scripts.append(script.string)

            elif script.get("src"):
                script_url = script["src"]

                if not script_url.startswith("http"):
                    script_url = requests.compat.urljoin(url, script_url)
                try:
                    script_response = requests.get(script_url)
                    if script_response.status_code == 200:
                        scripts.append(script_response.text)
                except Exception as e:
                    log(e)

        csss = []
        for style in soup.find_all("style"):
            if style.string is not None:
                csss.append(style.string)

        for style_link in soup.find_all("link", rel="stylesheet"):
            css_url = style_link.get("href")

            if css_url:
                if not css_url.startswith("http"):
                    css_url = requests.compat.urljoin(url, css_url)

                try:
                    css_response = requests.get(css_url)
                    if css_response.status_code == 200:
                        csss.append(css_response.text)
                except Exception as e:
                    log(e)

        for link in soup.find_all("a", href=True):
            href_link = link["href"]
            full_url = urljoin(url, href_link)
            self.found_links.add(full_url) 

        # Get the text content
        text = soup.get_text()

        md5_sum = hashlib.md5(text.encode()).hexdigest()

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
            'url': str(url),
            'content_type': self.driver.execute_script("return document.contentType"),
            'content': text,
            'scripts': list(scripts),
            'css': list(csss),
            'matched': bool(matched),
            'links': list(self.found_links)
        }

        # Display a progress update
        self.crawl_count += 1
        #progress = f"\x1b[2KCount: {self.crawl_count} Discarded: {self.discarded_count} Matches: {self.match_count} URL: {url}"
        #log(progress, end='\r')

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

    def close(self):
        """Close the WebDriver and clean up resources."""
        if self.driver:
            self.driver.quit()

    def __enter__(self):
        """Support context manager entry."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Support context manager exit, ensuring resources are closed."""
        self.close()

if __name__ == "__main__":
    # Usage
    # Initialize a WebCrawler object
    crawler = WebCrawler(browser='firefox')

    # Define the URL you want to crawl
    url = "https://books.toscrape.com"

    # Define the search term you're looking for
    search_term = "Python"

    # Pull website content
    pages = crawler.pull_website_content(url, search_term=None, crawl=False, depth=1)

    # Print the pages
    for md5, page in pages.items():
        log(f"URL: {page['url']}")
        log(f"Content Type: {page['content_type']}")
        log(f"Content: {page['content']}")
        log(f"Search Term Matched: {page['matched']}")
        log("\n")
