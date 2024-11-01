import torch
import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration, T5Tokenizer
import sacrebleu
from src.mode_config import ModeConfig
from src.loss_functions import entity_aware_loss, ner_auxiliary_loss, placeholder_loss

class T5FineTuner(pl.LightningModule):
    def __init__(self, tokenizer, train_dataloader, val_dataloader, test_dataloader, learning_rate, target_max_length=32, mode=0):
        super(T5FineTuner, self).__init__()
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self._test_dataloader = test_dataloader
        self.model = T5ForConditionalGeneration.from_pretrained(tokenizer.name_or_path)
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.target_max_length = target_max_length
        self.mode = mode  

    def forward(self, source_token_ids, source_mask, target_token_ids=None, target_mask=None, ne_tag_mask=None, training=False):
        if training:
            target_token_ids[target_token_ids == self.tokenizer.pad_token_id] = -100
            output = self.model(input_ids=source_token_ids, attention_mask=source_mask, labels=target_token_ids)
            
            if self.mode == 1 and ne_tag_mask is not None:
                loss = entity_aware_loss(output.logits, target_token_ids, ne_tag_mask, weight_factor=ModeConfig.MODE_1_WEIGHT)

            elif self.mode == 2 and ne_tag_mask is not None:
                base_loss = output.loss
                loss = placeholder_loss(base_loss, ne_tag_mask) * ModeConfig.MODE_2_WEIGHT

            elif self.mode == 3 and ne_tag_mask is not None:
                encoder_outputs = self.model.encoder(input_ids=source_token_ids, attention_mask=source_mask, return_dict=True)
                attention_weights = encoder_outputs.last_hidden_state
                loss = output.loss + ner_auxiliary_loss(attention_weights, ne_tag_mask) * ModeConfig.MODE_3_NER_WEIGHT

            else:
                loss = output.loss * ModeConfig.MODE_0_WEIGHT
            
            return loss
        else:
            predicted_token_ids = self.model.generate(input_ids=source_token_ids, max_length=self.target_max_length)
            return predicted_token_ids

    def training_step(self, batch, batch_idx):
        source_token_ids, source_mask, target_token_ids, target_mask, source_ne_tags, target_ne_tags = batch
        loss = self(source_token_ids, source_mask, target_token_ids, target_mask, ne_tag_mask=source_ne_tags, training=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        source_token_ids, source_mask, target_token_ids, target_mask, source_ne_tags, target_ne_tags = batch
        val_loss = self(source_token_ids, source_mask, target_token_ids, target_mask, ne_tag_mask=source_ne_tags, training=True)
        
        pred_token_ids = self(source_token_ids, source_mask)

        def filter_valid_ids(token_ids):
            return [id for id in token_ids if 0 <= id < self.tokenizer.vocab_size]

        pred_texts = [self.tokenizer.decode(filter_valid_ids(ids.tolist()), skip_special_tokens=True) for ids in pred_token_ids]
        target_texts = [self.tokenizer.decode(filter_valid_ids(ids.tolist()), skip_special_tokens=True) for ids in target_token_ids]
        
        bleu_score = sacrebleu.corpus_bleu(pred_texts, [target_texts]).score
        
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_bleu', bleu_score, prog_bar=True)
        
        return val_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def test_dataloader(self):
        return self._test_dataloader
