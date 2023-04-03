###used google colab####
#dependencies bellow 
#!pip install torch
#!pip install wandb
#!pip install --upgrade transformers

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Initialize wandb
wandb.login()
wandb.init(project="chatgtparma3")

# ...



# Set random seed for reproducibility
set_seed(42)

# Load the tokenizer and the model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Load your custom dataset
code_names_file = "/content/drive/MyDrive/codebertcustom-main (1)/arma3/data/ready/code_names.txt"
descriptions_file = "/content/drive/MyDrive/codebertcustom-main (1)/arma3/data/ready/descriptions.txt"

with open(code_names_file, "r") as f:
    code_names = f.readlines()

with open(descriptions_file, "r") as f:
    descriptions = f.readlines()

tokenizer.pad_token = tokenizer.eos_token

# Tokenize the data
input_texts = [code.strip() for code in code_names]
target_texts = [desc.strip() for desc in descriptions]

inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True)
targets = tokenizer(target_texts, return_tensors="pt", padding=True, truncation=True)

# Create a dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs["input_ids"])

    def __getitem__(self, idx):
        input_ids = self.inputs["input_ids"][idx]
        attention_mask = self.inputs["attention_mask"][idx]
        target_ids = self.targets["input_ids"][idx]
        target_attention_mask = self.targets["attention_mask"][idx]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": target_ids,
            "labels_attention_mask": target_attention_mask,
        }

dataset = CustomDataset(inputs, targets)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.1, verbose=True, threshold=1e-4)

# Create a DataLoader
data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Set training arguments
training_args = TrainingArguments(
    output_dir="/content/drive/MyDrive/codebertcustom-main (1)",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    save_steps=5000,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=500,
    learning_rate=5e-5,  # pass the learning rate through TrainingArguments
)

# Create a Trainer
# Create a Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
    optimizers=(optimizer, None),  # Pass None as the second value in the tuple
)
# Train the model
trainer.train()
training_logs = trainer.train()
wandb.log(training_logs)
