#!/usr/bin/env python3
#
# symbiote/api.py

from flask import Flask, request
from threading import Thread

class SymbioteAPI:
    def __init__(self, obj, debug=False):
        self.schat = obj
        self.app = Flask(__name__)
        self.debug = debug

        @self.app.route('/chat', methods=['POST'])
        def receive_input_post():
            user_input = request.json.get('user_input')
            # Call the chat method from your existing codebase with user_input
            # Return the response as a JSON object
            response = self.schat.chat(user_input=user_input, run=True, enable=False)
            return {'response': response}

        @self.app.route('/chat', methods=['GET'])
        def receive_input_get():
            user_input = request.args.get('user_input')
            # Call the chat method from your existing codebase with user_input
            # Return the response as a JSON object
            response = self.schat.chat(user_input=user_input, run=True, enable=False)
            if response is None:
                response = "OK"

            return {'response': response}

    def start(self):
        if self.debug:
            self.app.run()
        else:
            # Start the API server in a new thread
            #api_thread = Thread(target=self.app.run)
            #api_thread.start()
            self.app.run()

