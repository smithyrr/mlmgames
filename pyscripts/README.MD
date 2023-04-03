Scripts:

smalldataset.py:

Python script defining a small dataset for training or evaluation.

websiterip.py:

Python script for downloading and extracting code blocks from websites.

train_with_small_batch_size.py: Python script for training with a small batch size to reduce memory usage.

train_with_gradient_accumulation.py: Python script for training with gradient accumulation to improve stability.

Note: The train_data folder has been added, but the data inside requires preprocessing before use.

generate_description.py Script Summary:

Imports the necessary libraries, including the GPT-2 tokenizer and model from the Hugging Face Transformers library, as well as JSON and PyTorch.

Loads the pretrained GPT-2 tokenizer and model.

Defines a function called generate_description that takes a code example as input and returns a generated description.

Reads the "arma3_commands.json" file and loads its content into a variable called commands_data.

Iterates through the commands_data list and generates a new description for each command with the description "No description available" based on the command's example code.

Prints the code example and the corresponding generated description.

Writes the updated commands_data list to a new JSON file called "arma3_commands_with_descriptions.json".

generate_descriptions.py:

Reads the contents of the arma3_commands_by_functionality.json file.

Removes the leading and trailing commas from the content.

Wraps the content in square brackets ([...]) to make it a valid JSON array.

Parses the formatted content as a JSON object.

Saves the formatted content to a new JSON file named formatted_arma3_commands_by_functionality.json with proper indentation.

Note: This project is a work in progress and is not complete.
