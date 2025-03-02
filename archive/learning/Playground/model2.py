#!/usr/bin/env python3
#
# test.py

from transformers import GPT2LMHeadModel, GPT2Tokenizer

def interactWithModel(self, prompt):
    # Load the trained model and tokenizer
    model_dir = self.settings['symbiote_path'] + "learning/index_model"
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)

    # Tokenize the prompt
    inputs = tokenizer.encode(prompt, return_tensors='pt')

    # Generate a response
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2, temperature=0.7)

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response

response = interactWithModel("Hello, symbiote!")
print(response)

