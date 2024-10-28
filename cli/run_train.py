import argparse
from src.train import train_model  # Import train_model from src.train
from src.config import Config      # Import Config from src.config

def main():
    parser = argparse.ArgumentParser(description="Train T5 model with entity-aware modes")
    parser.add_argument("--data_dir", type=str, default=Config.DATA_DIR, help="Directory for dataset")
    parser.add_argument("--learning_rate", type=float, default=Config.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--max_epochs", type=int, default=Config.MAX_EPOCHS, help="Maximum epochs")
    parser.add_argument("--mode", type=int, choices=[0, 1, 2, 3], default=Config.MODE, help="Training mode")
    parser.add_argument("--target_max_length", type=int, default=Config.TARGET_MAX_LENGTH, help="Target max length")
    
    args = parser.parse_args()

    train_model(
        data_dir=args.data_dir,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        mode=args.mode,
        target_max_length=args.target_max_length
    )

if __name__ == "__main__":
    main()
