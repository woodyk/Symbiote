#!/usr/bin/env python3
#
# mrblack2.py

import openai
from openai import AssistantEventHandler
from typing_extensions import override
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from prompt_toolkit import HTML

# Ensure you replace 'your_api_key_here' with your actual OpenAI API key
openai.api_key = 'your_api_key_here'
assistant_id = "asst_cGS0oOCEuRqm0QPO9vVsPw1y"

client = openai.OpenAI()

class MyAssistant:
    def __init__(self, assistant_id):
        self.assistant_id = assistant_id
        self.assistant = self.get_assistant_by_id()
        self.thread = self.create_thread()
        self.response_log = []  # Initialize a list to store responses
        self.chat_history = []

    def get_assistant_by_id(self):
        return client.beta.assistants.retrieve(assistant_id=self.assistant_id)

    def create_thread(self):
        return client.beta.threads.create()

    def add_message_to_thread(self, content):
        try:
            result = client.beta.threads.messages.create(
                thread_id=self.thread.id,
                role="user",
                content=content
            )
        except Exception as e:
            result = e

        return result

    def run_assistant(self, instructions="", thread_id=""):
        class EventHandler(AssistantEventHandler):
            @override
            def on_text_created(self, text) -> None:
                print(f"\033[34massistant:\033[0m ", end="", flush=True)
            
            @override
            def on_text_delta(self, delta, snapshot):
                print(delta.value, end="", flush=True)
            
            def on_tool_call_created(self, tool_call):
                print(f"\n\033[34massistant:\033[0m {tool_call.type}\n", end="", flush=True)
            
            def on_tool_call_delta(self, delta, snapshot):
                if delta.type == 'code_interpreter':
                    if delta.code_interpreter.input:
                        print(delta.code_interpreter.input, end="", flush=True)
                    if delta.code_interpreter.outputs:
                        print(f"\n\033[32moutput >\033[0m\n", flush=True)
                        for output in delta.code_interpreter.outputs:
                            if output.type == "logs":
                                print(f"{output.logs}", flush=True)
                            elif output.type == "file":
                                self.process_file(output.file)

            def process_file(self, file):
                file_id = file.id
                file_name = file.name
                print(f"\nFile received: {file_name} (ID: {file_id})")

                # Perform file analysis or process file as needed
                # For this example, let's simply print the file details
                file_details = client.beta.files.retrieve(file_id=file_id)
                print(f"\nFile details:\n{file_details}")

        print("---")

        try:
            with client.beta.threads.runs.stream(
                thread_id=self.thread.id,
                assistant_id=self.assistant_id,
                instructions=instructions,
                event_handler=EventHandler(),
            ) as stream:
                stream.until_done()
        except Exception as e:
            print(f"error: {e}")
            pass

        print("\n---")

        if thread_id == "":
            message = ""
        else:
            messages = client.beta.threads.messages.list(thread_id)
            message = messages.data[0].content[0].text.value

        return message

    def upload_file(self, file_path):
        response = client.beta.files.create(
            file=open(file_path, 'rb')
        )
        file_id = response.result.id
        return file_id

    def download_file(self, file_id, download_path):
        response = client.beta.files.download(
            file_id=file_id,
            destination=download_path
        )
        return response

    def delete_file(self, file_id):
        response = client.beta.files.delete(file_id=file_id)
        return response

    def standard(self, user_input, model="gpt-4o"):
        self.chat_history.append({"role": "user", "content": user_input})

        try:
            response = client.chat.completions.create(
                model=model,
                messages=self.chat_history,
                stream=True
            )
        except Exception as e:
            print(e)

        response_content = ""

        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="")
                response_content += chunk.choices[0].delta.content

        self.chat_history.append({"role": "assistant", "content": response_content})
        print()

        return response_content

if __name__ == "__main__":
    assistant = MyAssistant(assistant_id)
    session = PromptSession(style=Style.from_dict({'user': '#0088ff', 'assistant': '#ff88ff'}))
    while True:
        user_input = session.prompt(HTML('\n<user>User:</user> '))
        if user_input.lower().strip() == "quit" or user_input.lower().strip() == "exit":
            break

        if user_input.lower().strip() == "upload":
            file_path = session.prompt(HTML('<user>File Path:</user> '))
            file_id = assistant.upload_file(file_path)
            print(f"\nFile uploaded with ID: {file_id}")
        elif user_input.lower().strip() == "download":
            file_id = session.prompt(HTML('<user>File ID:</user> '))
            download_path = session.prompt(HTML('<user>Download Path:</user> '))
            response = assistant.download_file(file_id, download_path)
            if isinstance(response, Exception):
                print(f"\nError: {response}")
            else:
                print(f"\nFile downloaded to: {response}")
        elif user_input.lower().strip() == "delete":
            file_id = session.prompt(HTML('<user>File ID:</user> '))
            response = assistant.delete_file(file_id)
            if isinstance(response, Exception):
                print(f"\nError: {response}")
            else:
                print(f"\nFile deleted")
        else:
            result = assistant.add_message_to_thread(user_input)
            assistant.run_assistant()
            print(result)
            messages = client.beta.threads.messages.list(result.thread_id)
            print(messages.data[0].content[0].text.value)

