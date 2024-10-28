import torch
import pytorch_lightning as pl
from transformers import T5Tokenizer
from src.model import T5FineTuner
from src.utils import load_data
from src.mode_config import ModeConfig

def train_model(data_dir, learning_rate, batch_size, max_epochs, mode, target_max_length=32):
    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    # Load data
    train_dataloader, val_dataloader, test_dataloader = load_data(data_dir, tokenizer, batch_size)

    # Initialize model
    model = T5FineTuner(
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        learning_rate=learning_rate,
        target_max_length=target_max_length,
        mode=mode
    )

    # Initialize trainer and train
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1  # Set devices=1 for both GPU and CPU
    )
    trainer.fit(model)
    return model
