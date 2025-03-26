#!/usr/bin/env python3
#
# sentence_check.py

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def sentence_makes_sense(sentence):
    # Load pre-trained model and tokenizer
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Encode the sentence
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss

    # A lower loss indicates the sentence is more likely to make sense
    return float(loss)

# Example usage:
sentence = "Take the bottle of water and put it on the table."
sentence = "The man hello the talk on the water."
score = sentence_makes_sense(sentence)
print(f"Sense score: {score:.2f} (Lower is better)")

