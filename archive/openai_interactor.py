#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: openai_interactor.py
# Author: Wadih Khairallah
# Description: 
# Created: 2024-11-29 23:10:20
# Modified: 2025-03-13 13:26:58

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
        logging.basicConfig(level=logging.WARNING)
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
            "name": function_name,
            "description": function_description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required_params,
                "additionalProperties": False,
            },
        }

        # Register the function
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
        functions = None
        if tool_choice == "auto" and self.tools:
            functions = self.tools
        elif tool_choice == "none":
            functions = None
        elif tool_choice == "required" and not self.tools:
            raise ValueError("Tool usage is required, but no tools are registered.")

        input_data = {
            "model": self.model,
            "messages": self.conversation_history,
            "functions": functions,  # Use 'functions' instead of 'tools' for OpenAI API
            "max_tokens": max_output_tokens,  # Correct parameter name
            "stream": use_streaming,
        }

        try:
            response = self.client.chat.completions.create(**input_data)

            if use_streaming:
                return self._stream_response(response)
            else:
                self.total_tokens_used += response.usage.total_tokens
                return self._handle_response(response)
        except Exception as e:
            self.logger.error(f"Error during OpenAI interaction: {e}")
            raise

    def _stream_response(self, response):
        """
        Stream response content and process function calls incrementally.

        :param response: The streaming response object.
        :yield: Chunks of the assistant's response content or function call results.
        """
        self.logger.debug("Starting to process streaming response.")
        function_arguments = ""
        function_name = ""
        is_collecting_function_args = False

        for part in response:
            self.logger.debug(f"Streaming part: {part}")
            delta = part.choices[0].delta

            # Handle assistant content
            if hasattr(delta, 'content') and delta.content is not None:
                self.logger.debug(f"Received content chunk: {delta.content}")
                yield delta.content

            # Handle function calls
            if hasattr(delta, 'function_call') and delta.function_call is not None:
                function_call = delta.function_call
                if hasattr(function_call, 'name') and function_call.name is not None:
                    function_name = function_call.name
                if hasattr(function_call, 'arguments') and function_call.arguments is not None:
                    function_arguments += function_call.arguments
                is_collecting_function_args = True
                self.logger.debug(f"Function call detected: {function_name} with partial arguments: {function_arguments}")

            # If the function call is complete
            if part.choices[0].finish_reason == "function_call" and is_collecting_function_args:
                self.logger.debug(f"Function call '{function_name}' is complete with arguments: {function_arguments}")
                try:
                    arguments = json.loads(function_arguments) if function_arguments.strip() else {}
                    tool_result = self._execute_tool(function_name, arguments)
                    self.logger.debug(f"Function '{function_name}' executed with result: {tool_result}")

                    function_call_result_message = {
                        "role": "function",
                        "name": function_name,
                        "content": json.dumps(tool_result),
                    }
                    self.conversation_history.append(function_call_result_message)
                    self.logger.debug(f"Appended function result message: {function_call_result_message}")

                    input_data = {
                        "model": self.model,
                        "messages": self.conversation_history,
                        "functions": self.tools,
                        "stream": True,
                    }
                    self.logger.debug(f"Making a new streaming request with input_data: {input_data}")
                    new_response = self.client.chat.completions.create(**input_data)
                    # Start processing the new streaming response
                    yield from self._stream_response(new_response)
                    return  # Ensure we exit after starting the new streaming response
                except Exception as e:
                    error_message = {
                        "error": f"Function '{function_name}' execution failed.",
                        "message": str(e),
                    }
                    self.logger.error(f"Error processing function '{function_name}': {e}")

                    # Return error as function response
                    function_call_result_message = {
                        "role": "function",
                        "name": function_name,
                        "content": json.dumps(error_message),
                    }
                    self.conversation_history.append(function_call_result_message)
                    self.logger.debug(f"Appended error function result message: {function_call_result_message}")

                    # Resubmit conversation with function error
                    input_data = {
                        "model": self.model,
                        "messages": self.conversation_history,
                        "functions": self.tools,
                        "stream": True,
                    }
                    self.logger.debug(f"Making a new streaming request with error input_data: {input_data}")
                    new_response = self.client.chat.completions.create(**input_data)
                    yield from self._stream_response(new_response)
                    return  # Ensure we exit after starting the new streaming response

            # Stop streaming on "stop" finish reason
            if part.choices[0].finish_reason == "stop":
                self.logger.debug("Streaming completed successfully with finish_reason='stop'.")
                break

        self.logger.debug("Finished processing streaming response.")

    def _handle_response(self, response):
        """
        Handle non-streaming responses from the API.

        :param response: The API response object.
        :return: The assistant's response or the result of a function call.
        """
        choice = response.choices[0]

        if choice.finish_reason == "stop":
            assistant_message = choice.message.content
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            return assistant_message

        elif choice.finish_reason == "function_call":
            function_call = choice.message.function_call
            function_name = function_call.name

            try:
                arguments = json.loads(function_call.arguments) if function_call.arguments else {}
                tool_result = self._execute_tool(function_name, arguments)

                function_call_result_message = {
                    "role": "function",
                    "name": function_name,
                    "content": json.dumps(tool_result),
                }
                self.conversation_history.append(function_call_result_message)

                # Resubmit the conversation with the function result
                input_data = {
                    "model": self.model,
                    "messages": self.conversation_history,
                    "functions": self.tools,
                }
                response = self.client.chat.completions.create(**input_data)
                return self._handle_response(response)

            except Exception as e:
                error_message = {
                    "error": f"Function '{function_name}' failed.",
                    "message": str(e),
                }
                self.logger.error(f"Error processing function '{function_name}': {e}")

                # Return error as function response
                function_call_result_message = {
                    "role": "function",
                    "name": function_name,
                    "content": json.dumps(error_message),
                }
                self.conversation_history.append(function_call_result_message)

                # Resubmit conversation with function error
                input_data = {
                    "model": self.model,
                    "messages": self.conversation_history,
                    "functions": self.tools,
                }
                response = self.client.chat.completions.create(**input_data)
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

        # Check the function signature
        signature = inspect.signature(external_callable)
        required_params = signature.parameters
        missing_args = []

        # Validate arguments against the function signature
        for param_name, param in required_params.items():
            if param_name not in arguments:
                if param.default == inspect.Parameter.empty:
                    missing_args.append(param_name)
                else:
                    arguments[param_name] = param.default  # Use default value

        # If arguments are missing, return an error response
        if missing_args:
            error_message = f"Missing required argument(s): {', '.join(missing_args)}"
            self.logger.error(error_message)
            return {
                "error": f"Function '{tool_name}' failed to execute.",
                "missing_arguments": missing_args,
                "message": error_message,
            }

        # Execute the tool
        self.logger.debug(f"Executing '{tool_name}' with arguments: {arguments}")
        return external_callable(**arguments)

# Functions to be registered
def get_current_temperature(location: str, unit: str):
    """
    Get the current temperature for a specific location.

    Parameters:
        location (str): The city and state, e.g., "San Francisco, CA".
        unit (str): The temperature unit ("Celsius" or "Fahrenheit").

    Returns:
        dict: The temperature and location.
    """
    # Simulated temperature retrieval logic
    return {"location": location, "unit": unit, "temperature": 72}

def get_rain_probability(location: str):
    """
    Get the probability of rain for a specific location.

    Parameters:
        location (str): The city and state, e.g., "San Francisco, CA".

    Returns:
        dict: The rain probability and location.
    """
    # Simulated rain probability retrieval logic
    return {"location": location, "rain_probability": 15}

def get_server_status():
    """
    Get the current status of the server.

    Returns:
        dict: A dictionary containing server uptime and load information.
    """
    # Simulated server status retrieval logic
    return {"status": "online", "uptime": "48 hours", "load": "moderate"}

# Main code to test the functionality
if __name__ == "__main__":
    oai = OpenAIInteractor()
    oai.add_function(external_callable=get_current_temperature)
    oai.add_function(external_callable=get_rain_probability)
    oai.add_function(external_callable=get_server_status)

    # Test with streaming
    print("Streaming Mode:")
    response = oai.handle_chat_request(
        {"role": "user", "content": "What is the temperature in San Francisco and tell me about San Fran."},
        stream=True
    )
    for chunk in response:
        print(chunk, end="")  # Consume and print each chunk

    response = oai.handle_chat_request(
        {"role": "user", "content": "What is the current server status.  Also is it going to rain today?"},
    )

    print(response)
