import torch
import pytorch_lightning as pl
from transformers import T5Tokenizer
from src.model import T5FineTuner
from src.utils import load_data
from src.config import Config  

def train_model():
    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    train_dataloader, val_dataloader, test_dataloader = load_data(
        data_dir=Config.DATA_DIR, 
        tokenizer=tokenizer, 
        batch_size=Config.BATCH_SIZE
    )

    model = T5FineTuner(
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        learning_rate=Config.LEARNING_RATE,
        target_max_length=Config.TARGET_MAX_LENGTH,
        mode=Config.MODE
    )

    trainer = pl.Trainer(
        max_epochs=Config.MAX_EPOCHS,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[
            pl.callbacks.ModelCheckpoint(dirpath="output/checkpoints", save_top_k=1, monitor="val_loss", mode="min"),
            pl.callbacks.EarlyStopping(monitor="val_loss", patience=3, mode="min")
        ],
        logger=pl.loggers.TensorBoardLogger("output/logs", name="T5_FineTuning")
    )
    trainer.fit(model)
    return model

# Execute training if this file is run directly
if __name__ == "__main__":
    train_model()
