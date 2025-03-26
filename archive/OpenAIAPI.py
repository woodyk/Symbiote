#!/usr/bin/env python3
#
# OpenAIAPI.py

import os
import time
import json
import base64
import hmac
import hashlib
from datetime import datetime
import requests
from typing import List, Union
import asyncio

class OpenAIAPI:
    BASE_URL = "https://api.openai.com/v1"
    HEADERS = {"Content-Type": "application/json"}
    
    def __init__(self, auth_key: str, organization: str = "", verify: bool = True):
        self._auth_key = auth_key
        self._organization = organization
        self._verify = verify
        self.headers = self.HEADERS

    @property
    def headers(self):
        timestamp = round(time.time())
        signature = base64.b64encode(hmac.new(bytes(self._auth_key, "ascii"), msg=bytes(timestamp, "ascii"), digestmod=hashlib.sha256).digest()).decode("ascii")
        self.headers = dict({**self.HEADERS}, Authorization=f"Bearer {self._auth_key}", "User-Agent" : "OpenAI-Python/v1", "X-Api-Key": self._auth_key, "X-Organization-Id": self._organization, "X-Timestamp": str(timestamp), "X-Signature-Ed25519": signature)
        return self.headers

    def http_call(self, method: str, route: str, payload: dict = {}) -> tuple:
        start_time = time.time()
        url = self.BASE_URL + route
        resp = None
        try:
            if method.lower() == "get":
                resp = requests.get(url, params=payload, headers=self.headers, timeout=30, verify=self._verify)
            elif method.lower() == "post":
                resp = requests.post(url, json=payload, headers=self.headers, timeout=30, verify=self._verify)
            elif method.lower() == "put":
                resp = requests.put(url, json=payload, headers=self.headers, timeout=30, verify=self._verify)
            elif method.lower() == "delete":
                resp = requests.delete(url, headers=self.headers, timeout=30, verify=self._verify)
            else:
                raise ValueError("Method should be one of [GET|POST|PUT|DELETE]")

            response_obj = {
                "duration": round((time.time() - start_time)*1000, 2),
                "status_code": resp.status_code,
                "reason": resp.reason,
                "headers": dict(resp.headers),
                "body": resp.json() if resp.status_code != 204 else {},
            }
            return response_obj, resp.content

        except requests.exceptions.RequestException as err:
            response_obj = {
                "duration": round((time.time() - start_time)*1000, 2),
                "status_code": -1,
                "reason": str(err),
                "headers": {},
                "body": {},
            }
            return response_obj, b""

    def generate_chat_completion(self, model: str, messages: list, temperature: float = 0.7, top_p: float = 1.0, n: int = 1, stream: bool = False, presence_penalty: float = 0.0, frequency_penalty: float = 0.0, stop: str | list = "") -> dict:
        route = f"/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
            "stream": stream,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "stop": stop
        }
        res, _ = self.http_call("POST", route, payload)
        return res["body"]

    def generate_text_completion(self, model: str, prompt: str, temperature: float = 0.7, top_p: float = 1.0, max_tokens: int = 16, stream: bool = False, logprobs: int = None, echo: bool = False, stop: str | list = "", presence_penalty: float = 0.0, frequency_penalty: float = 0.0) -> dict:
        route = f"/engines/{model}/completions"
        payload = {
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": stream,
            "logprobs": logprobs,
            "echo": echo,
            "stop": stop,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty
        }
        res, _ = self.http_call("POST", route, payload)
        return res["body"]

    def generate_image(self, prompt: str, format: str = "url", width: int = 1024, height: int = 1024, samples: int = 1, engine: str = "any") -> dict:
        route = f"/images/generations"
        payload = {
            "prompt": prompt,
            "format": format,
            "width": width,
            "height": height,
            "samples": samples,
            "engine": engine
        }
        res, _ = self.http_call("POST", route, payload)
        return res["body"]

    def edit_image(self, image: str, mask: str, prompt: str, format: str = "url", width: int = 1024, height: int = 1024, samples: int = 1, engine: str = "any") -> dict:
        route = f"/images/edits"
        payload = {
            "image": image,
            "mask": mask,
            "prompt": prompt,
            "format": format,
            "width": width,
            "height": height,
            "samples": samples,
            "engine": engine
        }
        res, _ = self.http_call("POST", route, payload)
        return res["body"]

    def vary_image(self, image: str, format: str = "url", width: int = 1024, height: int = 1024, samples: int = 1, engine: str = "any") -> dict:
        route = f"/images/variations"
        payload = {
            "image": image,
            "format": format,
            "width": width,
            "height": height,
            "samples": samples,
            "engine": engine
        }
        res, _ = self.http_call("POST", route, payload)
        return res["body"]

    def search_images(self, query: str, next_page: str = None) -> dict:
        route = f"/images/search"
        payload = {
            "query": query,
            "next_page": next_page
        }
        res, _ = self.http_call("GET", route, payload)
        return res["body"]

    def audio_transcription(self, file: str, model: str = "whisper-1", language: str = None) -> dict:
        route = f"/audio/translations"
        files = {'file': ("audio_file.mp3", open(file, "rb"))}
        payload = {
            "model": model,
            "language": language
        }
        res, _ = self.http_call("POST", route, payload, files=files)
        return res["body"]

    def audio_translation(self, file: str, source: str, target: str, model: str = "whisper-1") -> dict:
        route = f"/audio/translations"
        files = {'file': ("audio_file.mp3", open(file, "rb"))}
        payload = {
            "source": source,
            "target": target,
            "model": model
        }
        res, _ = self.http_call("POST", route, payload, files=files)
        return res["body"]

    def moderation(self, input: str) -> dict:
        route = f"/moderations"
        payload = {
            "input": input
        }
        res, _ = self.http_call("POST", route, payload)
        return res["body"]

    def fine_tunes(self, action: str, id: str = None, model: str = None, training_files: list = [], validation_files: list = [], description: str = None, organization: str = None, callback_endpoint: str = None) -> dict:
        if action.lower() not in ["create", "retrieve", "list", "cancel", "delete"]:
            raise ValueError("Action should be one of [CREATE|RETRIEVE|LIST|CANCEL|DELETE]")

        route = f"/finetunes/{action}"

        if action.lower() == "create":
            payload = {
                "training_file": training_files,
                "validation_file": validation_files,
                "description": description,
                "organization": organization,
                "callback_endpoint": callback_endpoint
            }
        elif action.lower() == "retrieve":
            route = f"/finetunes/{id}"
            payload = {}
        elif action.lower() in ["list", "cancel", "delete"]:
            payload = {
                "model": model
            }

        res, _ = self.http_call("POST", route, payload)
        return res["body"]

    def embeddings(self, model: str, input: str | list) -> dict:
        if type(input) is not list:
            input = [input]

        route = f"/embeddings"
        payload = {
            "input": input,
            "model": model
        }
        res, _ = self.http_call("POST", route, payload)
        return res["body"]

    def completions(self, model: str, prompt: str, max_tokens: int = 16, temperature: float = 0.7, top_p: float = 1.0, n: int = 1, stream: bool = False, logprobs: int = None, echo: bool = False, stop: str | list = "", presence_penalty: float = 0.0, frequency_penalty: float = 0.0, best_of: int = 1, logit_bias: dict = None, user: str = None) -> dict:
        route = f"/models/{model}/completions"
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
            "stream": stream,
            "logprobs": logprobs,
            "echo": echo,
            "stop": stop,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "best_of": best_of,
            "logit_bias": logit_bias,
            "user": user
        }
        res, _ = self.http_call("POST", route, payload)
        return res["body"]

    def corrections(self, model: str, input: str, instruction: str) -> dict:
        route = f"/models/{model}/corrections"
        payload = {
            "input": input,
            "instruction": instruction
        }
        res, _ = self.http_call("POST", route, payload)
        return res["body"]

    def transcriptions(self, model: str, input: str, instruction: str) -> dict:
        route = f"/models/{model}/transcriptions"
        payload = {
            "input": input,
            "instruction": instruction
        }
        res, _ = self.http_call("POST", route, payload)
        return res["body"]

    def summarizations(self, model: str, input: str, instruction: str) -> dict:
        route = f"/models/{model}/summarizations"
        payload = {
            "input": input,
            "instruction": instruction
        }
        res, _ = self.http_call("POST", route, payload)
        return res["body"]

    def chat_completions(self, model: str, messages: list, temperature: float = 0.7, top_p: float = 1.0, n: int = 1, stream: bool = False, presence_penalty: float = 0.0, frequency_penalty: float = 0.0, stop: str | list = "", user: str = None) -> dict:
        route = f"/models/{model}/chat/completions"
        payload = {
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
            "stream": stream,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "stop": stop,
            "user": user
        }
        res, _ = self.http_call("POST", route, payload)
        return res["body"]

    def generate_audio(self, model: str, prompt: str, voice: str, volume: int = 1, speed: int = 1, pitch: int = 1, sample_rate_hertz: int = 22050) -> dict:
        route = f"/voices/{voice}/versions/{model}/jobs"
        payload = {
            "text": prompt,
            "volume": volume,
            "speed": speed,
            "pitch": pitch,
            "sample_rate_hertz": sample_rate_hertz
        }
        res, _ = self.http_call("POST", route, payload)
        return res["body"]

    def retrieve_audio(self, job_id: str) -> dict:
        route = f"/audios/{job_id}"
        payload = {}
        res, _ = self.http_call("GET", route, payload)
        return res["body"]

    def delete_audio(self, job_id: str) -> dict:
        route = f"/audios/{job_id}"
        payload = {}
        res, _ = self.http_call("DELETE", route, payload)
        return res["body"]

class OpenAIAssistant:
    def __init__(
        self,
        api_instance: OpenAIAPI,
        session_memory_limit: int = 4096,
        model: str = "text-davinci-002",
        temperature: float = 0.7,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
    ):
        self.api_instance = api_instance
        self.session_memory_limit = session_memory_limit
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.reset_session()

    def reset_session(self):
        self.context = ""
        self.current_request = []

    async def converse(
        self,
        prompt: str,
        *,
        streaming: bool = False,
        stop: Union[List[str], str] = None,
    ) -> Union[str, List[str]]:
        """
        Conduct a conversation with the OpenAI Assistant.

        Parameters
        ----------
        prompt : str
            A user's question or statement meant to elicit a response from the OpenAI Assistant.

        streaming : bool, optional
            Whether to receive partial results during the text generation, defaults to False.

        stop : Optional[Union[List[str], str]], optional
            Defines conditions where the text generation stops early. Accepts both strings and lists of strings. Default behavior depends on the model.

        Returns
        -------
        Union[str, List[str]]
            Either a single response string or a list of generated strings in case of streaming.
        """

        # Append user prompt to context
        self.context += f"\nHuman: {prompt}"

        # Reset current request
        self.current_request = [{"role": "system", "content": self.build_introduction()}],

        # Ensure the context fits within the memory limit
        while len(self.context) > self.session_memory_limit:
            self.context = self.context[len(self.context) // 2 :]

        # Prepare and send the API request
        api_params = {
            "model": self.model,
            "prompt": self.context,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }

        if stop is not None:
            api_params["stop"] = stop

        raw_response = await self.api_instance.complete_text(
            prompt=self.context,
            streaming=streaming,
            **api_params,
        )

        response = []
        for entry in raw_response.items():
            response.extend(entry.as_dict().get("choices", [])[0].get("delta", {}).get("content", ""))

        # Remove system introduction from final result
        response = [item for item in response if not item.startswith("You are")]

        # Store the user's latest contribution
        self.context = self.context.rsplit("\nHuman:", 1)[-1]

        return response

    def build_introduction(self) -> str:
        """Construct an initial greeting."""
        return (
            "You are a helpful assistant who responds in a friendly and engaging manner.\n"
            "Your answers are clear, concise, and considerate of the user's needs.\n"
            "Do not reveal sensitive or harmful information.\n"
            "Always double-check your grammar and spelling.\n"
            "Capitalize the beginning of sentences and ending punctuation marks correctly.\n"
            "Never impersonate humans or attempt to deceive users.\n"
            "Ensure your tone matches the level of familiarity expressed by the user.\n"
            "When faced with ambiguity, politely seek clarification.\n"
            "Feel free to utilize humor appropriately but avoid sarcasm and irony.\n"
            "If asked complex questions, focus on delivering valuable insights rather than exhaustive answers.\n"
        )

'''
async def main():
    # Instantiate an OpenAI API instance
    api_instance = OpenAIAPI("<YOUR_API_KEY>")

    # Setup the OpenAI Assistant
    assistant = OpenAIAssistant(
        api_instance=api_instance,
        session_memory_limit=4096,
        model="text-davinci-002",
        temperature=0.7,
    )

    # Interact with the OpenAI Assistant
    user_question = "What is the meaning of life?"
    print(f"> User Question: {user_question}")

    response = await assistant.converse(prompt=user_question, streaming=False)
    print(f"> Response: {response}")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())

'''

'''
OpenAI API Class for Python
This library simplifies access to OpenAI's powerful suite of AI tools through a simple and convenient Python interface. All OpenAI API calls are supported, allowing developers to leverage generative models, image recognition, speech synthesis, and much more directly from their Python projects.

Getting Started
Prerequisites
Install the package via pip:

pip install openai-api-py
Or clone the repository and install locally:

git clone git@github.com:yourusername/openai-api-py.git
cd openai-api-py
pip install .
Create a free account on OpenAI and obtain an API Key. Set up environment variables containing the OpenAI API key:

For Linux/MacOS:

export OPENAI_API_KEY="your-secret-key"
For Windows CMD:

setx OPENAI_API_KEY "your-secret-key"
For PowerShell:

$env:OPENAI_API_KEY = "your-secret-key"
Replace "your-secret-key" with the obtained API Key.

Usage
Text Generation Example
Instantiate an OpenAI API Client and specify the preferred model:

from openai_api import OpenAIClient

client = OpenAIClient("text-davinci-002")
response = client.complete_text(prompt="Once upon a time,")
print(response.result)
Image Generation Example
Generate an image using DALL·E mini:

from openai_api import OpenAIClient

client = OpenAIClient("dalle-mini")
response = client.generate_image(prompt="An adorable cat wearing sunglasses sitting inside a vintage car")
print(response.result)
Speech Synthesis Example
Convert text to speech:

from openai_api import OpenAIClient

client = OpenAIClient("whisper-1")
response = client.convert_speech(text="Hello world, welcome to OpenAI's API.")
print(response.result)
See documentation/examples.ipynb for additional examples showcasing the capabilities of the OpenAI API Class for Python.

Documentation
Detailed documentation is available in Jupyter Notebook format under documentation/examples.ipynb. Alternatively, view the rendered version online at [![Documentation][badge-doc]][link-doc].

Support
Please submit pull requests or report issues on GitHub. We appreciate feedback and contributions!

License
Distributed under the MIT license. See LICENSE for details.
'''
