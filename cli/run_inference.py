import argparse
import torch
from transformers import T5Tokenizer
from model import T5FineTuner
from utils import load_data

def main():
    parser = argparse.ArgumentParser(description="Run inference on new data with T5 model")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory for dataset")
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--mode", type=int, choices=[0, 1, 2, 3], default=0, help="Inference mode")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint")

    args = parser.parse_args()

    # Load tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    model = T5FineTuner.load_from_checkpoint(args.checkpoint_path, tokenizer=tokenizer, mode=args.mode)

    # Load data
    _, _, test_dataloader = load_data(args.data_dir, tokenizer, args.batch_size)

    # Run inference
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_dataloader:
            source_token_ids, source_mask, _, _, _, _ = batch
            pred_token_ids = model(source_token_ids, source_mask)
            predictions.extend([tokenizer.decode(ids, skip_special_tokens=True) for ids in pred_token_ids])
    
    # Print predictions
    for i, pred in enumerate(predictions):
        print(f"Sample {i + 1}: {pred}")

if __name__ == "__main__":
    main()
