#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: openai_interactor.py
# Author: Wadih Khairallah
# Description: 
# Created: 2024-11-29 23:10:20
# Modified: 2024-11-29 23:13:08

from openai import OpenAI
import openai
import inspect
import os
import json
import logging


MODEL_INFO = {
    "gpt-4o": {"context_window": 128000, "max_output_tokens": 16384},
    "gpt-4o-mini": {"context_window": 128000, "max_output_tokens": 16384},
    "o1-preview": {"context_window": 128000, "max_output_tokens": 32768},
    "o1-mini": {"context_window": 128000, "max_output_tokens": 65536},
    "gpt-4": {"context_window": 8192, "max_output_tokens": 8192},
    "gpt-3.5-turbo": {"context_window": 16385, "max_output_tokens": 4096},
}

class OpenAIInteractor:
    def __init__(self, api_key=None, model="gpt-4o-mini", streaming=False, suppress=False):
        """
        Initialize the OpenAI Interactor.

        :param api_key: Your OpenAI API key.
        :param model: The default model to use for OpenAI interactions.
        :param streaming: Whether to enable streaming responses.
        :param suppress: Whether to suppress intermediate outputs.
        """
        self.model = model
        self.streaming = streaming
        self.suppress = suppress
        self.conversation_history = []
        self.tools = []
        self.last_response = None
        self.total_tokens_used = 0

        # Default to reasonable values if MODEL_INFO is missing or model is not found
        default_context_window = 16000
        default_max_output_tokens = 16000

        # Get model-specific limits from MODEL_INFO, or use defaults
        if "MODEL_INFO" in globals() and model in MODEL_INFO:
            self.context_window = MODEL_INFO[model]["context_window"]
            self.max_output_tokens = MODEL_INFO[model]["max_output_tokens"]
        else:
            self.context_window = default_context_window
            self.max_output_tokens = default_max_output_tokens

        self.client = OpenAI()
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")

        self.client.api_key = api_key

        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized OpenAIInteractor with model: {self.model}")
        self.logger.info(f"Context window: {self.context_window}, Max output tokens: {self.max_output_tokens}")

    def add_function(self, external_callable=None, name=None, description=None):
        """
        Add a function schema to enable function calling.

        :param external_callable: A callable function to analyze and add to the tools list.
        :param name: Optional custom name for the function. Defaults to the callable's name.
        :param description: Optional custom description. Defaults to the callable's docstring summary.
        """
        if external_callable is None:
            raise ValueError("You must provide an external_callable to add a function.")

        # Auto-populate details from the callable
        function_name = name or external_callable.__name__
        docstring = inspect.getdoc(external_callable) or ""
        function_description = description or docstring.split("\n")[0] if docstring else "No description provided."

        # Analyze the function signature
        signature = inspect.signature(external_callable)
        properties = {}
        required_params = []

        for param_name, param in signature.parameters.items():
            param_type = (
                "number" if param.annotation in [float, int] else
                "string" if param.annotation == str else
                "boolean" if param.annotation == bool else
                "array" if param.annotation == list else
                "object"
            )
            if param.annotation == inspect.Parameter.empty:
                param_type = "string"
            properties[param_name] = {"type": param_type, "description": f"{param_name} parameter."}
            if param.default == inspect.Parameter.empty:
                required_params.append(param_name)

        function_definition = {
            "type": "function",
            "function": {
                "name": function_name,
                "description": function_description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required_params,
                },
            },
        }

        # Register the function as a tool
        self.tools.append(function_definition)

        # Dynamically attach the callable as an attribute of this instance
        setattr(self, function_name, external_callable)

        self.logger.info(f"Function '{function_name}' added successfully!")

    def _prune_context(self):
        """
        Prune the conversation history to fit within the model's context window.
        Ensures the system message is retained.
        """
        # Convert messages to dictionaries if they are not already
        self.conversation_history = [
            msg if isinstance(msg, dict) else msg.to_dict() for msg in self.conversation_history
        ]

        # Estimate the token count for the current history
        estimated_tokens = sum(len(json.dumps(msg)) for msg in self.conversation_history)

        # Subtract max_output_tokens to leave space for the model's response
        max_prompt_tokens = self.context_window - self.max_output_tokens

        # Prune messages until the total size fits within the remaining context window
        while estimated_tokens > max_prompt_tokens and len(self.conversation_history) > 1:
            removed = self.conversation_history.pop(1)  # Remove oldest user/assistant message
            self.logger.info(f"Pruned message: {removed}")
            estimated_tokens = sum(len(json.dumps(msg)) for msg in self.conversation_history)

        # Ensure the system message remains
        if not any(msg["role"] == "system" for msg in self.conversation_history):
            self.conversation_history.insert(0, {"role": "system", "content": "Default system message."})

    def handle_chat_request(self, user_message, stream=None, tool_choice="auto", max_completion_tokens=None, system_message=None):
        """
        Handle a chat request, dynamically resolving tool calls until a final response is generated.

        :param user_message: The user's input message (string or dictionary).
        :param stream: Whether to enable streaming for this request (overrides class-level setting).
        :param tool_choice: Control tool usage ("none", "auto", "required").
        :param max_completion_tokens: Maximum tokens for the completion.
        :param system_message: Optionally set or update the system message.
        :return: Final assistant response or a generator for streaming.
        """
        use_streaming = stream if stream is not None else self.streaming
        max_output_tokens = max_completion_tokens or self.max_output_tokens

        # Update the system message if provided
        if system_message:
            self.conversation_history = [msg for msg in self.conversation_history if msg["role"] != "system"]
            self.conversation_history.insert(0, {"role": "system", "content": system_message})

        # Ensure user_message is formatted correctly
        if isinstance(user_message, dict) and "content" in user_message:
            if isinstance(user_message["content"], dict):
                user_message["content"] = json.dumps(user_message["content"])  # Convert nested dict to JSON string
        elif isinstance(user_message, str):
            user_message = {"role": "user", "content": user_message}
        else:
            raise ValueError("Invalid format for user_message. Must be a string or dictionary with 'content' key.")

        # Add the user message to the conversation history
        self.conversation_history.append(user_message)

        # Prune the context to fit within the model's limits
        self._prune_context()

        # Configure tools based on tool_choice
        tools = None
        if tool_choice == "auto" and self.tools:
            tools = self.tools
        elif tool_choice == "none":
            tools = None
        elif tool_choice == "required" and not self.tools:
            raise ValueError("Tool usage is required, but no tools are registered.")

        input_data = {
            "model": self.model,
            "messages": self.conversation_history,
            "tools": tools,
            "max_completion_tokens": max_output_tokens,
            "stream": use_streaming,
        }

        try:
            if use_streaming:
                response = self.client.chat.completions.create(**input_data, stream=True)
                return self._stream_response(response)
            else:
                response = self.client.chat.completions.create(**input_data)
                self.total_tokens_used += response.usage.total_tokens
                return self._handle_response(response)
        except Exception as e:
            self.logger.error(f"Error during OpenAI interaction: {e}")
            raise


    def _stream_response(self, response):
        """
        Stream response content and process tool calls incrementally.

        :param response: The streaming response object.
        :yield: Chunks of the assistant's response content or tool call results.
        """
        function_arguments = ""
        function_name = ""
        is_collecting_function_args = False
        tool_call_id = None

        for part in response:
            delta = part.choices[0].delta
            finish_reason = part.choices[0].finish_reason

            if "content" in delta:
                yield delta["content"]

            if delta.tool_calls:
                is_collecting_function_args = True
                tool_call = delta.tool_calls[0]
                tool_call_id = tool_call.id

                if tool_call.function.name:
                    function_name = tool_call.function.name
                if tool_call.function.arguments:
                    function_arguments += tool_call.function.arguments

            if finish_reason == "tool_calls" and is_collecting_function_args:
                # Execute the tool and format the result
                arguments = json.loads(function_arguments)
                tool_result = self._execute_tool(function_name, arguments)

                # Create a tool call result message, ensuring content is serialized
                tool_call_result_message = {
                    "role": "tool",
                    "content": json.dumps(tool_result),  # Ensure content is serialized
                    "tool_call_id": tool_call_id
                }

                # Append to conversation history and resubmit
                self.conversation_history.append({"role": "assistant", "tool_calls": [tool_call.to_dict()]})
                self.conversation_history.append(tool_call_result_message)

                input_data = {
                    "model": self.model,
                    "messages": self.conversation_history,
                    "stream": True,
                }
                response = self.client.chat.completions.create(**input_data, stream=True)

                # Continue streaming from the new response
                yield from self._stream_response(response)
                break


    def _handle_response(self, response):
        """
        Handle non-streaming responses from the API.

        :param response: The API response object.
        :return: The assistant's response or the result of a tool call.
        """
        choice = response.choices[0]

        if choice.finish_reason == "stop":
            # Standard response from the assistant
            assistant_message = choice.message.content
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            return assistant_message

        elif choice.finish_reason == "tool_calls":
            # Handle the tool call
            tool_call = choice.message.tool_calls[0]
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            tool_result = self._execute_tool(tool_name, arguments)

            # Create a message for the tool call result, ensuring content is a JSON string
            tool_call_result_message = {
                "role": "tool",
                "content": json.dumps(tool_result),  # Ensure content is serialized
                "tool_call_id": tool_call.id
            }

            # Append tool call and result to the conversation history
            self.conversation_history.append(choice.message.to_dict())
            self.conversation_history.append(tool_call_result_message)

            # Resubmit the conversation with the tool result included
            input_data = {
                "model": self.model,
                "messages": self.conversation_history,
            }
            response = self.client.chat.completions.create(**input_data)

            # Recursively handle the next response
            return self._handle_response(response)


    def _execute_tool(self, tool_name, arguments):
        """
        Execute the requested tool (function) with the provided arguments.

        :param tool_name: Name of the tool (function) to execute.
        :param arguments: Arguments to pass to the tool.
        :return: The result of the tool execution.
        """
        # Look for the dynamically attached function
        external_callable = getattr(self, tool_name, None)
        if not external_callable:
            raise ValueError(f"Function '{tool_name}' is not defined.")

        # Execute the function with the provided arguments
        return external_callable(**arguments)


def main():
    def get_current_temperature(location: str, unit: str):
        """
        Get the current temperature for a specific location.

        Parameters:
            location (str): The city and state, e.g., "San Francisco, CA".
            unit (str): The temperature unit ("Celsius" or "Fahrenheit").

        Returns:
            dict: The temperature and location.
        """
        return {"location": location, "unit": unit, "temperature": 72}

    def get_rain_probability(location: str):
        """
        Get the probability of rain for a specific location.

        Parameters:
            location (str): The city and state, e.g., "San Francisco, CA".

        Returns:
            dict: The rain probability and location.
        """
        return {"location": location, "rain_probability": 15}

    oai = OpenAIInteractor()
    oai.add_function(external_callable=get_current_temperature)
    oai.add_function(external_callable=get_rain_probability)
    response = oai.handle_chat_request("What's the temperature in San Francisco, CA?")

    print(response)

    response = oai.handle_chat_request({"role": "user", "content": "Calculate the area of a rectangle with width 5 and height 10."})
    print(response)


if __name__ == "__main__":
    main()
