import sacrebleu
import torch
from transformers import T5Tokenizer

def evaluate_model(model, dataloader):
    tokenizer = model.tokenizer
    model.eval()
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    predictions, references = [], []

    for batch in dataloader:
        source_ids, source_mask, target_ids, _, _, _ = batch
        source_ids = source_ids.to(model.device)
        source_mask = source_mask.to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(input_ids=source_ids, attention_mask=source_mask, max_length=model.target_max_length)
        
        pred_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]
        ref_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in target_ids]
        
        predictions.extend(pred_texts)
        references.extend(ref_texts)
    
    bleu_score = sacrebleu.corpus_bleu(predictions, [references]).score
    print(f"BLEU Score: {bleu_score}")
    return bleu_score
