import sacrebleu
from transformers import T5Tokenizer

def evaluate_model(model, dataloader):
    tokenizer = model.tokenizer
    model.eval()
    predictions, references = [], []

    for batch in dataloader:
        source_ids, source_mask, target_ids, _, _, _ = batch
        with torch.no_grad():
            generated_ids = model(source_ids, source_mask)
        
        pred_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]
        ref_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in target_ids]
        
        predictions.extend(pred_texts)
        references.extend(ref_texts)
    
    bleu_score = sacrebleu.corpus_bleu(predictions, [references]).score
    print(f"BLEU Score: {bleu_score}")
    return bleu_score
