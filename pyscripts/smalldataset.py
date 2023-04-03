import pandas as pd

df = pd.read_csv("small_dataset.csv")
texts = df["code_snippet"].tolist()

# Tokenize the code snippets
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Fine-tune the model
model.train_model(inputs)
