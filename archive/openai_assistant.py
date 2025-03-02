#!/usr/bin/env python3
#
# openai_assistant.py

import openai

def create_assistant(name, instructions, tools, model):
    """
    Create an Assistant with the given parameters.
    """
    assistant = openai.Assistant.create(
        name=name,
        instructions=instructions,
        tools=tools,
        model=model
    )
    return assistant

def create_thread():
    """
    Create a new conversation thread.
    """
    thread = openai.Thread.create()
    return thread

def add_message_to_thread(thread_id, user_input):
    """
    Add a user's message to the thread.
    """
    message = openai.Message.create(
        thread_id=thread_id,
        role="user",
        content=user_input
    )
    return message

def run_assistant(thread_id, assistant_id, instructions=None):
    """
    Run the assistant on the thread to get a response.
    """
    run = openai.Run.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
        instructions=instructions
    )
    return run

def get_assistant_response(thread_id):
    """
    Retrieve the assistant's response from the thread.
    """
    messages = openai.Message.list(thread_id=thread_id)
    return messages

def connect_to_openai_assistant(user_input):
    """
    Connect to the OpenAI Assistant and get a response for the user input.
    """
    # Step 1: Create an Assistant
    assistant = create_assistant(
        name="Mr. Black",
        instructions="",
        tools=[{"type": "code_interpreter"}],
        model="gpt-4-1106-preview"
    )

    # Step 2: Create a Thread
    thread = create_thread()

    # Step 3: Add a Message to a Thread
    add_message_to_thread(thread.id, user_input)

    # Step 4: Run the Assistant
    run = run_assistant(thread.id, assistant.id)

    # Step 5: Check the Run status (omitted for brevity)

    # Step 6: Display the Assistant's Response
    response = get_assistant_response(thread.id)

    # Extract and return the assistant's messages
    assistant_messages = [msg['content'] for msg in response['data'] if msg['role'] == 'assistant']
    return assistant_messages

# Example usage
user_input = "I need to solve the equation `3x + 11 = 14`. Can you help me?"
response_messages = connect_to_openai_assistant(user_input)
for message in response_messages:
    print(message)

