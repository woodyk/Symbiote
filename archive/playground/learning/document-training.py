#!/usr/bin/env python3
from elasticsearch import Elasticsearch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import torch
import os
import sys
import requests

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

def read_text_files_in_directory(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r") as infile:
                content = infile.read()
                documents.append(content)
    return documents

def preprocess_data(documents):
    text_data = []
    for doc in documents:
        # Extract relevant fields from the document and concatenate them
        # Update this section according to the structure of your text files
        text = doc  # Assuming each document is a single string
        text_data.append(text)
    return text_data

# Read text files from the directory
directory_path = sys.argv[1] 
documents = read_text_files_in_directory(directory_path)

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

with open("gpt2_finetuned/tokenizer_config.json") as f:
    data = {
      "model_max_length": 1024,
      "model_type": "gpt2",
      "padding_side": "right"
    }

    f.write(json.dumps(data))
