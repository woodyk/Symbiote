#!/usr/bin/env python3
#
# sym_crawler.py

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
from concurrent.futures import ThreadPoolExecutor
import hashlib
import re
from prompt_toolkit import print_formatted_text, ANSI
from prompt_toolkit.utils import get_cwidth
import requests

class WebCrawler:
    def __init__(self, browser="firefox", use_selenium=True):
        """
        Initializes the WebCrawler object.

        Args:
            browser (str): Browser type for Selenium ("chrome" or "firefox").
            use_selenium (bool): Whether to use Selenium for advanced crawling.
        """
        self.pages = {}
        self.visited_urls = set()
        self.match_count = 0
        self.crawl_count = 0
        self.discarded_count = 0
        self.browser = browser
        self.base_url = None
        self.found_links = set()
        self.use_selenium = use_selenium
        self.session = requests.Session()

        # Set up WebDriver if using Selenium
        if self.use_selenium:
            if self.browser.lower() == "chrome":
                options = ChromeOptions()
                options.add_argument("--headless")
                self.driver = webdriver.Chrome(service=webdriver.chrome.service.Service(ChromeDriverManager().install()), options=options)
            elif self.browser.lower() == "firefox":
                options = FirefoxOptions()
                options.add_argument("--headless")
                self.driver = webdriver.Firefox(service=webdriver.firefox.service.Service(GeckoDriverManager().install(), log_path='/dev/null'), options=options)
            else:
                log(f"Unsupported browser: {self.browser}")
                self.driver = None

    def extract_tables(self, soup):
        """
        Extracts tables from the given BeautifulSoup object.

        Args:
            soup (BeautifulSoup): Parsed HTML content.

        Returns:
            list: A list of dictionaries representing tables.
        """
        tables = []
        for table in soup.find_all("table"):
            table_data = []
            headers = [header.get_text(strip=True) for header in table.find_all("th")]

            for row in table.find_all("tr"):
                cells = row.find_all(["td", "th"])
                row_data = [cell.get_text(strip=True) for cell in cells]

                # Match row data to headers if headers exist
                if headers and len(headers) == len(row_data):
                    table_data.append(dict(zip(headers, row_data)))
                else:
                    table_data.append(row_data)

            # Append parsed table to the list
            tables.append({
                "headers": headers,
                "rows": table_data
            })
        return tables

    def fetch_content_requests(self, url):
        """
        Fetch and parse webpage content using requests and BeautifulSoup.

        Args:
            url (str): URL to fetch content from.

        Returns:
            tuple: BeautifulSoup object and content type.
        """
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'lxml')
            return soup, response.headers.get("Content-Type", "").split(";")[0]
        except requests.RequestException as e:
            log(f"Error fetching the URL with requests: {e}")
            return None, None

    def fetch_content_selenium(self, url):
        """
        Fetch and parse webpage content using Selenium and BeautifulSoup.

        Args:
            url (str): URL to fetch content from.

        Returns:
            tuple: BeautifulSoup object and content type.
        """
        try:
            self.driver.get(url)
            soup = BeautifulSoup(self.driver.page_source, 'lxml')
            content_type = self.driver.execute_script("return document.contentType")
            return soup, content_type
        except Exception as e:
            log(f"Error fetching the URL with Selenium: {e}")
            return None, None

    def fetch_and_clean(self, url):
        """
        Fetch content from a URL and return its cleaned content.

        Args:
            url (str): URL to fetch.

        Returns:
            str: Cleaned content or error message.
        """
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return ' '.join(response.text.split())
        except requests.RequestException:
            return f"Error fetching {url}"

    def fetch_and_clean_assets(self, soup, url):
        """
        Extract scripts, CSS, and links from a webpage.

        Args:
            soup (BeautifulSoup): Parsed HTML content.
            url (str): Base URL of the webpage.

        Returns:
            tuple: (list of scripts, list of CSS, list of links)
        """
        # Collect scripts
        script_urls = [urljoin(url, script["src"]) for script in soup.find_all("script", src=True)]
        scripts = self.fetch_resources(script_urls)

        # Collect CSS files
        css_urls = [urljoin(url, link["href"]) for link in soup.find_all("link", rel="stylesheet", href=True)]
        css_files = self.fetch_resources(css_urls)

        # Collect unique links
        links = set()
        for link in soup.find_all("a", href=True):
            full_url = urljoin(url, link["href"])
            links.add(full_url)

        return scripts, css_files, list(links)

    def fetch_resources(self, urls):
        """
        Fetch multiple resources in parallel.

        Args:
            urls (list): List of URLs to fetch.

        Returns:
            list: List of cleaned content for each URL.
        """
        with ThreadPoolExecutor(max_workers=10) as executor:
            return list(executor.map(self.fetch_and_clean, urls))

    def delay_request(self):
        """
        Introduce a random delay between 1 and 3 seconds to avoid overloading servers.
        """
        time.sleep(random.uniform(1, 3))

    def pull_website_content(self, url=None, search_term=None, crawl=False, depth=None):
        """
        Crawl a website and extract data.

        Args:
            url (str): URL to crawl.
            search_term (str): Term to search for in the content.
            crawl (bool): Whether to recursively crawl links.
            depth (int): Depth of recursive crawling.

        Returns:
            dict: Dictionary of crawled pages and their metadata.
        """
        if self.base_url is None:
            self.base_url = url

        self.found_links.add(url)
        self.search_term = search_term
        self.crawl = crawl

        if self.use_selenium:
            soup, content_type = self.fetch_content_selenium(url)
        else:
            soup, content_type = self.fetch_content_requests(url)

        if not soup:
            return ""

        # Clean and extract text
        for element in soup(['script', 'style']):
            element.decompose()
        text = soup.get_text(separator=' ')
        sanitized_text = ' '.join(text.split())

        # Hash content to detect duplicates
        md5_sum = hashlib.md5(sanitized_text[:10000].encode()).hexdigest()
        if md5_sum in self.pages:
            self.discarded_count += 1
            return ""

        # Fetch assets and links
        scripts, css_files, links = self.fetch_and_clean_assets(soup, url)

        # Extract tables
        tables = self.extract_tables(soup)

        # Check for search term match
        matched = False
        if self.search_term:
            if any(re.search(variation, sanitized_text, re.IGNORECASE) for variation in [self.search_term]):
                matched = True
                self.match_count += 1

        # Store page data
        self.pages[md5_sum] = {
            'url': url,
            'content_type': content_type,
            'content': sanitized_text,
            'scripts': scripts,
            'css': css_files,
            'matched': matched,
            'links': links,
            'tables': tables
        }

        # Crawl deeper if needed
        if self.crawl and (depth is None or depth > 0):
            for link in links:
                if link not in self.visited_urls:
                    self.visited_urls.add(link)
                    self.pull_website_content(link, search_term=self.search_term, crawl=True, depth=depth - 1 if depth else None)

        return self.pages

    def close(self):
        """
        Close the WebDriver and session.
        """
        if self.use_selenium and self.driver:
            self.driver.quit()
        if hasattr(self, 'session'):
            self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


if __name__ == "__main__":
    # Testing the crawler
    crawler = WebCrawler(browser="firefox", use_selenium=False)
    url = "https://books.toscrape.com"
    pages = crawler.pull_website_content(url, search_term="Python", crawl=True, depth=1)

    # Output test results
    for md5, page in pages.items():
        log(f"URL: {page['url']}")
        log(f"Content Type: {page['content_type']}")
        log(f"Matched: {page['matched']}")
        log(f"Links Found: {len(page['links'])}")
        log(f"Tables Extracted: {len(page['tables'])}")
        log(f"Text Content: {page['content']}")
    crawler.close()

