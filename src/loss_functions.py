import torch
from torch.nn import CrossEntropyLoss

def entity_aware_loss(logits, labels, ne_tag_mask, weight_factor=2.0):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Basic cross-entropy loss
    loss_fct = CrossEntropyLoss(reduction="none")
    base_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    # Apply extra weight to NE tags in the loss
    entity_weights = torch.ones_like(shift_labels).float()
    entity_weights[ne_tag_mask == 1] *= weight_factor
    weighted_loss = base_loss * entity_weights.view(-1)

    return weighted_loss.mean()

def ner_auxiliary_loss(attention_weights, ne_tag_mask):
    avg_attention = attention_weights.mean(dim=-1)
    ner_loss = torch.mean((avg_attention - ne_tag_mask.float()) ** 2)
    return ner_loss

def placeholder_loss(base_loss, ne_tag_mask):
    return base_loss * torch.mean(ne_tag_mask.float())  
