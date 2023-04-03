import wandb
from wandb.sdk import wandb_config

def sweep_config():
    config = wandb_config.Config()
    
    # Define hyperparameters to sweep over
    config.learning_rate = wandb.config.learning_rate = [1e-5, 2e-5, 5e-5]
    config.num_train_epochs = wandb.config.num_train_epochs = [3, 5, 10]
    config.per_device_train_batch_size = wandb.config.per_device_train_batch_size = [2, 4, 8]
    
    return config
