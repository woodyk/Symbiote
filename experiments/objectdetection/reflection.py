#!/usr/bin/env python3
#
# reflection.py

from huggingface_hub import InferenceClient

client = InferenceClient(
    "mattshumer/Reflection-Llama-3.1-70B",
    token="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",
)

for message in client.chat_completion(
	messages=[{"role": "user", "content": "What is the capital of France?"}],
	max_tokens=500,
	stream=True,
):
    print(message.choices[0].delta.content, end="")
