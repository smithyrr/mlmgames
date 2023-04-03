import json

with open('/home/cognitron/codebert/arma3_commands_with_descriptions.json', 'r') as f:
    data = json.load(f)

with open('/home/cognitron/codebert/formatted_data.txt', 'w') as f:
    for d in data:
        example = d['example'].replace('\n', '')
        f.write(f"{d['name']}\n{d['description']}\nExample: {example}\n\n")
