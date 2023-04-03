from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2, # Use a smaller batch size to fit within your memory
    num_train_epochs=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
