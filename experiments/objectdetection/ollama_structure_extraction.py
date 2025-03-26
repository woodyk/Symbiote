#!/usr/bin/env python3
#
# ollama_structure_extraction.py

import ollama
import argparse
import json

def extract_structure(text):
    prompt = f"""
    Analyze the following text and identify any structured data within it. 
    This could include JSON, YAML, CODE, MARKDOWN, XML, or any other structured format.
    If you find structured data, please output it in the following JSON format:
    {{
        "type": "The type of structured data (e.g., JSON, YAML, CODE, MARKDOWN, XML)",
        "content": "The extracted structured content"
    }}
    If multiple structures are found, return an array of such objects.
    If no structured data is found, return an empty array.
    Return only the JSON document with the structurs found.
    Do not provide any further details.

    Text to analyze:
    {text}
    """

    response = ollama.generate(model='phi3.5', prompt=prompt)
    print(response['response'])
    
    try:
        result = json.loads(response['response'])
        return result
    except json.JSONDecodeError:
        print("Error: The model's response was not in the expected JSON format.")
        return []

def main():
    input_text = """
In today's data-driven world, structured data formats play a crucial role in enabling systems to communicate and process information efficiently. Whether it's configuration files, data exchange between services, or even documentation, structured formats like JSON, YAML, CSV, and others are ubiquitous. This document explores these formats with examples.

JSON: JavaScript Object Notation
JSON is a lightweight data interchange format that's easy for humans to read and write, and easy for machines to parse and generate. JSON is widely used in web applications for transmitting data between a server and a client.

json
Copy code
{
    "user": {
        "id": 12345,
        "name": "Alice",
        "email": "alice@example.com",
        "isAdmin": false
    },
    "tasks": [
        {"id": 1, "description": "Buy groceries", "completed": false},
        {"id": 2, "description": "Read a book", "completed": true}
    ]
}
The JSON snippet above represents a user object with a nested list of tasks. This structure is commonly used in REST APIs to transmit data.

YAML: YAML Ain't Markup Language
YAML is a human-readable data serialization standard that can be used in conjunction with all programming languages. It's often used for configuration files and data exchange.

yaml
Copy code
version: 1.2
services:
  web:
    image: "nginx:latest"
    ports:
      - "8080:80"
  database:
    image: "mysql:5.7"
    environment:
      MYSQL_ROOT_PASSWORD: example
      MYSQL_DATABASE: test
      MYSQL_USER: user
      MYSQL_PASSWORD: pass
In this YAML example, a basic configuration for a Docker Compose file is shown. It defines two services: a web server running Nginx and a database service running MySQL.

Code: Python Functions
Code snippets are often included in documentation to demonstrate how to use specific APIs or to provide examples of functions and methods.

python
Copy code
def fibonacci(n):
    a, b = 0, 1
    sequence = []
    while len(sequence) < n:
        sequence.append(a)
        a, b = b, a + b
    return sequence

# Example usage
print(fibonacci(10))  # Output: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
This Python function generates a Fibonacci sequence up to the nth number. Such code snippets are common in programming tutorials and API documentation.

CSV: Comma-Separated Values
CSV is a simple file format used to store tabular data, such as a spreadsheet or database. CSVs are widely used because they are easy to generate and parse.

csv
Copy code
id,name,age,city
1,John Doe,29,New York
2,Jane Smith,34,Los Angeles
3,Bob Johnson,45,Chicago
4,Alice Williams,23,Houston
The above CSV data represents a simple table with four columns: ID, Name, Age, and City. CSV files are often used for exporting data from databases or spreadsheets.

Markdown: Lightweight Markup Language
Markdown is a lightweight markup language that is easy to write and read, with plain text formatting syntax. It is widely used in documentation, especially in README files for GitHub repositories.

markdown
Copy code
# Project Title

## Introduction
This project aims to provide a solution to the problem of **data management** in cloud environments.

### Features
- **Simple** and **intuitive** user interface.
- Supports integration with multiple data sources.
- Provides **real-time** data analysis.

## Installation
To install the project, clone the repository and run the following command:

```bash
pip install -r requirements.txt
Usage
After installation, you can start the application using:

bash
Copy code
python app.py
License
This project is licensed under the MIT License.

vbnet
Copy code

Markdown allows users to format text, include links, images, code snippets, and much more in a simple and readable way. It's especially popular in the open-source community.

---

### **XML: Extensible Markup Language**

XML is a markup language that defines a set of rules for encoding documents in a format that is both human-readable and machine-readable. It's often used in web services, configuration files, and data exchange between systems.

```xml
<note>
  <to>Alice</to>
  <from>Bob</from>
  <heading>Reminder</heading>
  <body>Don't forget our meeting at 3 PM tomorrow.</body>
</note>
The XML snippet above represents a simple note with fields for the recipient, sender, heading, and body. XML is widely used for its versatility and ability to represent complex data structures.

HTML: HyperText Markup Language
HTML is the standard markup language for documents designed to be displayed in a web browser. It can be assisted by technologies such as CSS and JavaScript to enhance the appearance and functionality of web pages.

html
Copy code
<!DOCTYPE html>
<html>
<head>
    <title>Sample Page</title>
</head>
<body>
    <h1>Welcome to My Website</h1>
    <p>This is a paragraph of text on my website.</p>
    <ul>
        <li>Home</li>
        <li>About</li>
        <li>Contact</li>
    </ul>
</body>
</html>
This basic HTML document defines a webpage with a title, a header, a paragraph, and a list. HTML is foundational for creating structured web content.

INI: Initialization File
INI files are simple text files with a structure composed of sections, properties, and values. They are often used for configuration purposes in software applications.

ini
Copy code
[general]
appname = MyApplication
version = 1.0

[user]
name = John Doe
email = johndoe@example.com

[settings]
autosave = true
theme = dark
The INI example above shows a configuration file with sections for general application settings, user details, and specific settings.

TOML: Tom's Obvious, Minimal Language
TOML is a configuration file format that's easy to read due to its simplicity and clarity. It's often used in Rust projects and has become popular as an alternative to JSON and YAML for configuration files.

toml
Copy code
title = "TOML Example"

[owner]
name = "Tom Preston-Werner"
dob = 1979-05-27T07:32:00Z

[database]
server = "192.168.1.1"
ports = [ 8001, 8001, 8002 ]
connection_max = 5000
enabled = true

[servers]
  [servers.alpha]
  ip = "10.0.0.1"
  dc = "eqdc10"

  [servers.beta]
  ip = "10.0.0.2"
  dc = "eqdc20"

This TOML file is a configuration example, demonstrating how you can store various types of data, including strings, integers, arrays, and dates."""

    structures = extract_structure(input_text)

    print(json.dumps(structures, indent=4))

if __name__ == "__main__":
    main()
