import json
import os

def restructure_json_file(input_file, output_file):
    # Read the content from the original JSON file
    with open(input_file, "r") as file:
        content = file.read()

    # Remove the leading and trailing commas from the content
    content = content.strip(", ")

    # Wrap the content in square brackets to form a JSON array
    formatted_content = f"[{content}]"

    # Parse the formatted content as a JSON object
    formatted_data = json.loads(formatted_content)

    # Save the formatted content to a new JSON file
    with open(output_file, "w") as file:
        json.dump(formatted_data, file, indent=4)

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_path, "../arma3_commands_by_functionality.json")
    output_file = os.path.join(base_path, "../formatted_arma3_commands_by_functionality.json")
    restructure_json_file(input_file, output_file)
