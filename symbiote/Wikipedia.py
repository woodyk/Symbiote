#!/usr/bin/env python3
#
# wikipedia.py

import requests
log(f"Loading WikipediaSearch.")

class WikipediaSearch:
    def __init__(self, language='en'):
        self.session = requests.Session()
        self.url = f'https://{language}.wikipedia.org/w/api.php'

    def search(self, query, results=10):
        params = {
            'action': 'query',
            'list': 'search',
            'srsearch': query,
            'format': 'json',
            'srlimit': results
        }
        
        response = self.session.get(url=self.url, params=params)
        search_results = response.json().get('query', {}).get('search', [])
        
        articles = []
        for article in search_results:
            pageid = article['pageid']
            page_data = self.get_page_by_id(pageid)
            if page_data:
                articles.append(page_data)
        return articles

    def get_page_by_id(self, pageid):
        params = {
            'action': 'query',
            'prop': 'extracts|info',
            'pageids': pageid,
            'inprop': 'url',
            'explaintext': True,
            'format': 'json'
        }
        
        response = self.session.get(url=self.url, params=params)
        pages = response.json().get('query', {}).get('pages', {})
        page = pages.get(str(pageid), {})
        
        return {
            'title': page.get('title', ''),
            'text': page.get('extract', ''),
            'url': page.get('fullurl', '')
        }

if __name__ == "__main__":
    # Example usage:
    wiki_search = WikipediaSearch()
    results = wiki_search.search('Quantum computing', 5)
    for article in results:
        print(f"Title: {article['title']}")
        print(f"URL: {article['url']}")
        print(f"Text: {article['text'][:500]}")  # Printing only the first 500 characters of the text for brevity

