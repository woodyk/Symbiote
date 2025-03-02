#!/usr/bin/env python3
#
# github_tools.py

import os
import requests

class GithubTools:
    def __init__(self, owner, repo):
        self.owner = owner
        self.repo = repo
        self.base_url = f"https://api.github.com/repos"
        self.api_headers = {"Accept": "application/vnd.github.v3+json"}
        self.api_headers = {
            "Accept": "application/vnd.github.v3+json",
            "Authorization": f"token {os.getenv('GITHUB_API_KEY')}"
        }

    def fetch_base_endpoints(self):
        """
        Fetch the base GitHub API endpoints and parse the available search options.
        """
        response = requests.get(self.base_url, headers=self.api_headers)
        if response.status_code == 200:
            self.available_endpoints = response.json()
            print("Successfully fetched base endpoints:")
            for key, value in self.available_endpoints.items():
                print(f"- {key}: {value}")
        else:
            print(f"Failed to fetch base endpoints. HTTP Status: {response.status_code}")

    def get_search_url(self, search_type):
        """
        Get the search URL for a specific type.
        """
        key = f"{search_type}_search_url"
        if key in self.available_endpoints:
            return self.available_endpoints[key]
        else:
            print(f"Search type '{search_type}' is not available.")
            return None

    def perform_search(self, search_type, query, **kwargs):
        """
        Perform a search using the specified search type and query.
        """
        search_url = self.get_search_url(search_type)
        if not search_url:
            return

        # Replace placeholders in the URL
        search_url = search_url.replace("{query}", query)

        # Add additional query parameters
        if kwargs:
            params = "&".join(f"{key}={value}" for key, value in kwargs.items())
            search_url = search_url.replace("{&page,per_page,sort,order}", f"&{params}")
        else:
            search_url = search_url.replace("{&page,per_page,sort,order}", "")

        response = requests.get(search_url, headers=self.api_headers)
        if response.status_code == 200:
            data = response.json()
            print(f"Search results for '{query}' in '{search_type}':")
            print(json.dumps(data, indent=2))
        else:
            print(f"Search failed. HTTP Status: {response.status_code}")

    def search_github_repositories(self, search_term):
        """
        Searches GitHub repositories based on a search term.
        """
        url = f"https://api.github.com/search/repositories?q={search_term}"
        response = requests.get(url, headers=self.api_headers)
        data = response.json()

        # Check if the response contains items
        if 'items' not in data:
            print("No data found or there was an error with the request.")
            return

    def search_code(self, search_term, repo_name=None):
        """
        Searches GitHub for files containing a specific search term.
        If a repository name is provided, it restricts the search to that repository.
        """
        query = f"{search_term}+in:file"
        if repo_name:
            query += f"+repo:{self.owner}/{repo_name}"
        url = f"https://api.github.com/search/code?q={query}"
        
        response = requests.get(url, headers=self.api_headers)
        if response.status_code == 200:
            data = response.json()
            if 'items' not in data or not data['items']:
                print("No files found for the given search term.")
                return

            print(f"Found {len(data['items'])} files containing the term '{search_term}':")
            for item in data['items']:
                print(f"- File Name: {item['name']}")
                print(f"  Repository: {item['repository']['full_name']}")
                print(f"  File Path: {item['path']}")
                print(f"  URL: {item['html_url']}")
        elif response.status_code == 403:
            print("Rate limit exceeded. Try authenticating with a personal access token.")
        else:
            print(f"Failed to fetch code search results. HTTP Status: {response.status_code}")


    def fetch_repo_details(self):
        """
        Fetches repository details from the GitHub API.
        """
        response = requests.get(self.base_url)
        if response.status_code == 200:
            return response.json()
        else:
            return None

    def fetch_readme_contents(self):
        """
        Fetches the README.md file from the repository if it exists.
        """
        url = f"{self.base_url}/contents/README.md"
        response = requests.get(url)
        if response.status_code == 200:
            readme_data = response.json()
            return requests.get(readme_data['download_url']).text
        return "README.md not found."

    def print_repo_details(self):
        """
        Prints selected details from the repository data and contents of README.md.
        """
        repo_data = self.fetch_repo_details()
        readme_contents = self.fetch_readme_contents()

        if repo_data:
            print("Repository Details:")
            print(f"Name: {repo_data['name']}")
            print(f"Owner: {repo_data['owner']['login']}")
            print(f"Description: {repo_data.get('description', 'No description provided')}")
            print(f"Stars: {repo_data['stargazers_count']}")
            print(f"Forks: {repo_data['forks_count']}")
            print(f"Open Issues: {repo_data['open_issues_count']}")
            print(f"Watchers: {repo_data['watchers_count']}")
            print(f"Main Language: {repo_data.get('language', 'Not specified')}")
            print(f"Created At: {repo_data['created_at']}")
            print(f"Updated At: {repo_data['updated_at']}")
            print(f"GitHub URL: {repo_data['html_url']}")
            print("\nREADME.md Contents:")
            print(readme_contents)
        else:
            print("Failed to fetch repository details. Please check the owner and repository name.")

    def search_github_repositories(self, search_term):
        # Prepare the request to GitHub API
        url = f"https://api.github.com/search/repositories?q={search_term}"
        response = requests.get(url)
        data = response.json()

        # Check if the response contains items
        if 'items' not in data:
            print("No data found or there was an error with the request.")
            return

        # Display the repositories
        repositories = data['items']
        if not repositories:
            print("No repositories found for your search.")
        else:
            print(f"Found {len(repositories)} repositories:")
            for repo in repositories:
                print(f"- {repo['name']} by {repo['owner']['login']}")
                print(f"  URL: {repo['html_url']}")
                print(f"  Description: {repo['description'] if repo['description'] else 'No description provided.'}")
                print(f"  Stars: {repo['stargazers_count']}")

def main():
    owner = input("Enter the GitHub owner name: ")
    repo = input("Enter the repository name: ")

    # Initialize the GitHubRepository class
    github_repo = GithubTools(owner, repo)

    while True:
        print("\nMenu:")
        print("1. Fetch and display repository details")
        print("2. Search for repositories")
        print("3. Search for files containing specific content")
        print("4. Fetch and display README.md contents")
        print("5. Exit")
        
        choice = input("Enter your choice (1-5): ")

        if choice == "1":
            print("\nFetching repository details...")
            github_repo.print_repo_details()
        elif choice == "2":
            search_term = input("\nEnter a search term to find repositories: ")
            print(f"\nSearching for repositories with term '{search_term}'...")
            github_repo.search_github_repositories(search_term)
        elif choice == "3":
            search_term = input("\nEnter a term to search for in files: ")
            repo_filter = input("Restrict search to this repository (leave blank for no restriction): ")
            print(f"\nSearching for files with term '{search_term}'...")
            github_repo.search_code(search_term, repo_filter)
        elif choice == "4":
            print("\nFetching README.md contents...")
            readme_contents = github_repo.fetch_readme_contents()
            print("\nREADME.md Contents:")
            print(readme_contents)
        elif choice == "5":
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()

