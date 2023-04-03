# Replace the path with the actual path to your JSON file
json_file_path = "/home/cognitron/codebertcustom/arma3_commands_with_descriptions.json"


import json

def save_to_file(data, file_path):
    with open(file_path, 'w') as f:
        for line in data:
            f.write(line + '\n')

# Replace the path with the actual path to your JSON file
json_file_path = "/home/cognitron/codebertcustom/arma3_commands_with_descriptions.json"

with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

code_names = []
descriptions = []

for item in data:
    code_name = item['name']
    description = "Describe the following Arma 3 code: " + item['example']
    code_names.append(code_name)
    descriptions.append(description)

# Replace these paths with the desired paths to save your text files
descriptions_file_path = "/home/cognitron/codebertcustom/arma3/data/ready/descriptions.txt"
code_names_file_path = "/home/cognitron/codebertcustom/arma3/data/ready/code_names.txt"

save_to_file(descriptions, descriptions_file_path)
save_to_file(code_names, code_names_file_path)
