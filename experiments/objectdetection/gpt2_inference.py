#!/usr/bin/env python3
#
# gpt2_inference.py

from huggingface_hub import InferenceClient

client = InferenceClient(
    "mistralai/Mistral-7B-Instruct-v0.3",
    token="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",
)

for message in client.chat_completion(
	messages=[{"role": "user", "content": "What is the capital of France?"}],
	max_tokens=500,
	stream=True,
):
    print(message.choices[0].delta.content, end="")
