#!/usr/bin/env python3
from elasticsearch import Elasticsearch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import torch
import os
import requests


class symLearn():
    def __init__(self, settings):
        self.settings = settings


    def loadES(self):
        es = Elasticsearch([{'host': 'dockera.vm.sr', 'port': 9200, 'scheme': 'http'}])

tokenizer_dir = os.getcwd() + "/gpt2_finetuned"

# Download the tokenizer files
tokenizer_files = {
    "vocab.json": "https://huggingface.co/gpt2/resolve/main/vocab.json",
    "merges.txt": "https://huggingface.co/gpt2/resolve/main/merges.txt",
    "tokenizer_config.json": "https://huggingface.co/gpt2/resolve/main/tokenizer_config.json",
}

for filename, url in tokenizer_files.items():
    response = requests.get(url)
    with open(os.path.join(tokenizer_dir, filename), "wb") as f:
        f.write(response.content)

# Load the tokenizer files and create a GPT2Tokenizer instance
vocab_file = os.path.join(tokenizer_dir, "vocab.json")
merges_file = os.path.join(tokenizer_dir, "merges.txt")

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer(vocab_file=vocab_file, merges_file=merges_file)
model.save_pretrained("./gpt2_finetuned")

def fetch_data_from_elasticsearch(index_name, query, size=1000):
    results = es.search(index=index_name, body=query, size=size)
    hits = results['hits']['hits']
    documents = [hit['_source'] for hit in hits]
    return documents

index_name = 'symbiote'
query = {
    "query": {
        "match_all": {}
    }
}

documents = fetch_data_from_elasticsearch(index_name, query)

def preprocess_elastic_data(documents):
    text_data = []
    for doc in documents:
        # Extract relevant fields from the document and concatenate them
        addresses = " ".join([f"{addr.get('road', '')} {addr.get('house_number', '')} {addr.get('city', '')} {addr.get('state', '')} {addr.get('postcode', '')}" for addr in doc.get('ADDRESSES', [])])
        artworks = " ".join(doc.get('ARTWORKS', []))
        currencies = " ".join(doc.get('CURRENCIES', []))
        dates = " ".join(doc.get('DATES', []))
        emails = " ".join(doc.get('EMAILS', []))
        events = " ".join(doc.get('EVENTS', []))
        landmarks = " ".join(doc.get('LANDMARKS', []))
        legal = " ".join(doc.get('LEGAL', []))
        localities = " ".join(doc.get('LOCALITIES', []))
        locations = " ".join(doc.get('LOCATIONS', []))
        nationalities = " ".join(doc.get('NATIONALITIES', []))
        organizations = " ".join(doc.get('ORGANIZATIONS', []))
        persons = " ".join(doc.get('PERSONS', []))
        phone_numbers = " ".join(doc.get('PHONE_NUMBERS', []))
        products = " ".join(doc.get('PRODUCTS', []))
        quantities = " ".join(doc.get('QUANTITIES', []))
        times = " ".join(doc.get('TIMES', []))
        title = doc.get('METADATA', {}).get('FileName', '')
        content = doc.get('SUMMARY', '')

        text = f"{title} {content} {addresses} {artworks} {currencies} {dates} {emails} {events} {landmarks} {legal} {localities} {locations} {nationalities} {organizations} {persons} {phone_numbers} {products} {quantities} {times}"
        text_data.append(text)
    return text_data

text_data = preprocess_data(documents)
text_data = "\n".join(text_data)

with open("train_data.txt", "w") as f:
    f.write(text_data)

dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="train_data.txt",
    block_size=128,
)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

training_args = TrainingArguments(
    output_dir="./gpt2_finetuned",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    learning_rate=5e-5,
    weight_decay=0.01,
    gradient_accumulation_steps=4,
    max_grad_norm=1.0,
    report_to=[]
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()
