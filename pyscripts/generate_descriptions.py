import json
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

def generate_description(code_example):
    input_text = f"Describe the following Arma 3 code: {code_example}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    attention_mask = torch.ones(input_ids.shape)

    with torch.no_grad():
        output = model.generate(input_ids, max_length=100, num_return_sequences=1, attention_mask=attention_mask)

    description = tokenizer.decode(output[0], skip_special_tokens=True)
    return description

with open("../formatted_arma3_commands_by_functionality.json", "r") as file:
    commands_data = json.load(file)

# Change this part to loop through the list of dictionaries
for category in commands_data:
    for command in category:
        if command["description"] == "No description available.":
            generated_description = generate_description(command["example"])
            command["description"] = generated_description
            print(f"Code example: {command['example']}\nGenerated description: {generated_description}\n")

with open("../arma3_commands_with_descriptions.json", "w") as file:
    json.dump(commands_data, file, indent=4)
