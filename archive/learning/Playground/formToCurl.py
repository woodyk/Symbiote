#!/usr/bin/env python3
#
# formToCurl.py

from bs4 import BeautifulSoup
import requests
import urllib.parse

def generate_curl_command(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    form = soup.find('form')
    if not form:
        print('No form found on the page.')
        return

    action = form.get('action')
    if not action.startswith('http'):
        action = url + action

    method = form.get('method', 'get').upper()

    fields = form.find_all('input', {'type': 'text'})
    field_values = {}
    for field in fields:
        name = field.get('name')
        value = input(f'Enter value for {name}: ')
        field_values[name] = urllib.parse.quote_plus(value)

    if method == 'GET':
        curl_command = f'curl -X {method} "{action}?"'
        for name, value in field_values.items():
            curl_command += f'{name}={value}&'
    else: # POST
        curl_command = f'curl -X {method} "{action}"'
        for name, value in field_values.items():
            curl_command += f' -d "{name}={value}"'

    print(curl_command)

generate_curl_command('https://www.cia.gov/readingroom/')
