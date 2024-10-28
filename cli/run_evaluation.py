import argparse
from evaluate import evaluate_model
from transformers import T5Tokenizer
from model import T5FineTuner
from utils import load_data

def main():
    parser = argparse.ArgumentParser(description="Evaluate T5 model on test data")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory for test dataset")
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--mode", type=int, choices=[0, 1, 2, 3], default=0, help="Evaluation mode")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint")

    args = parser.parse_args()

    # Load tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    model = T5FineTuner.load_from_checkpoint(args.checkpoint_path, tokenizer=tokenizer, mode=args.mode)

    # Load data
    _, _, test_dataloader = load_data(args.data_dir, tokenizer, args.batch_size)

    # Run evaluation
    bleu_score = evaluate_model(model, test_dataloader)
    print(f"BLEU Score on test set: {bleu_score}")

if __name__ == "__main__":
    main()
