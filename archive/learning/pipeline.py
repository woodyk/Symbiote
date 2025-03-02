#!/usr/bin/env python3
#
# pipeline.py

from transformers import pipeline, set_seed

generator = pipeline('text-generation', model='/Users/kato/.symbiote/learning/gpt2_finetuned')
set_seed(42)
generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)

