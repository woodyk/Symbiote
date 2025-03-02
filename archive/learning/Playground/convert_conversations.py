#!/usr/bin/env python3
#
# convert_conversations.py
# todo:
#   - Add the ability to extract the conversation header from exported conversations
#     and write them to their respective <conversation>.json file

import json

def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def get_messages(data):
    messages = []
    for item in data:
        mapping = item['mapping']
        for key in mapping:
            message = mapping[key]['message']
            if message:
                role = message['author']['role']
                parts = message['content']['parts']
                content = ' '.join(parts)
                create_time = message['create_time']
                messages.append({
                    'role': role,
                    'content': content,
                    'create_time': create_time,
                    'id': key,
                    'message_obj': message
                })
    return messages

def messages_to_json(messages):
    output = []
    for msg in messages:
        output.append({
            'role': msg['role'],
            'content': msg['content']
        })
    return json.dumps(output, indent=2)

# Load the JSON data, provide the correct path to your JSON file
data = load_data("conversations.json")

# Extract messages
messages = get_messages(data)

sorted_messages = sorted(messages, key=lambda x: x['create_time'], reverse=False)

json_out = messages_to_json(sorted_messages)
print(json_out)

# Print the extracted information
#for msg in messages:
#    print(msg['role'])
#    print(msg['content'])
#    print(msg['id'])
