#!/usr/bin/env python3
#
# JsonFileHandler.py

import json
import io
import requests
from urllib.parse import urlparse
import enchant

class JsonlFileHandler:
    def __init__(self, file_url):
        parsed_uri = urlparse(file_url)
        self.netloc = parsed_uri.hostname
        self.path = parsed_uri.path
        self.scheme = parsed_uri.scheme
        self.file_handler = None

    @property
    def is_valid(self):
        """Check whether the file exists."""
        http_status = 0
        try:
            req = requests.Request('HEAD', self.scheme + '://' + self.netloc + '/' + self.path)
            prep = req.prepare()
            sess = requests.Session()
            resp = sess.send(prep, timeout=1, verify=False)
            http_status = resp.status_code
        except Exception as err:
            pass
        return http_status == 200

    def open(self):
        """Open local or remote JSONL file."""
        if self.is_local():
            self.file_handler = io.open(self.path, mode='r', encoding='utf-8')
        else:
            self.file_handler = requests.get(self.path, stream=True).raw

    def readline(self):
        """Read a single line from JSONL file."""
        if self.file_handler is None:
            raise ValueError('Please call open() before attempting to read lines.')
        return self.file_handler.readline().decode('utf-8').strip('\n')

    def close(self):
        """Close the opened JSONL file."""
        if self.file_handler is not None:
            self.file_handler.close()

    def iter_lines(self):
        """Iterate through the JSONL file returning lines."""
        while True:
            line = self.readline()
            if len(line) == 0:
                break
            yield json.loads(line)

    def is_local(self):
        """Determine whether the path points to a local resource."""
        return bool(self.path and not self.scheme)

def main():
    file_handler = JsonlFileHandler('https://wadih.com/1a6271c55f385d3da93c21e2c630b5732a2a5050c328de8e832a066761a9f946.jsonl')
    file_handler.open()

    chat = GoogleKnowledgeGraph()

    for obj in file_handler.iter_lines():
        message = obj.get('message')
        sender = obj.get('sender')
        if sender != 'human':
            continue

        response = chatbot.generate_response(message)
        output_obj = {'message': response, 'sender': 'assistant'}
        print(json.dumps(output_obj))

    file_handler.close()

"""
if __name__ == '__main__':
    main()
"""
