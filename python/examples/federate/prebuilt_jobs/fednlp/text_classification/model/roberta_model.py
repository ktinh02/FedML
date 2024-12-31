import torch
import torch.nn as nn
from transformers.models.roberta.modeling_roberta import (
   RobertaModel,
   RobertaPreTrainedModel,
)




class RobertaForSequenceClassification(RobertaPreTrainedModel):
   r"""
   **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
       Labels for computing the sequence classification/regression loss.
       Indices should be in ``[0, ..., config.num_labels - 1]``.
       If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
       If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).


   Outputs:
       **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
           Classification or regression loss.
       **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
           Classification (or regression if config.num_labels==1) scores before SoftMax.
       **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
           list of ``torch.FloatTensor`` for each layer.
       **attentions**: (`optional`, returned when ``config.output_attentions=True``)
           list of ``torch.FloatTensor`` representing attention weights for each layer.


   Example usage::
       from transformers import RobertaTokenizer, RobertaForSequenceClassification
       import torch


       tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
       model = RobertaForSequenceClassification.from_pretrained('roberta-base')
       input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
       labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
       outputs = model(input_ids, labels=labels)
       loss, logits = outputs[:2]
   """


   def __init__(self, config, weight=None):
       super(RobertaForSequenceClassification, self).__init__(config)
       self.num_labels = config.num_labels
       self.weight = weight


       self.roberta = RobertaModel(config)
       # Classification head: typically Roberta uses a CLS token at start
       self.dropout = nn.Dropout(config.hidden_dropout_prob)
       self.classifier = nn.Linear(config.hidden_size, config.num_labels)


       self.init_weights()


   def forward(
       self,
       input_ids=None,
       attention_mask=None,
       token_type_ids=None,
       position_ids=None,
       head_mask=None,
       inputs_embeds=None,
       labels=None,
       output_attentions=None,
       output_hidden_states=None,
       return_dict=False,
   ):
       # RoBERTa doesn't use token_type_ids by default; this is kept for compatibility.
       outputs = self.roberta(
           input_ids,
           attention_mask=attention_mask,
           token_type_ids=token_type_ids,
           position_ids=position_ids,
           head_mask=head_mask,
           inputs_embeds=inputs_embeds,
           output_attentions=output_attentions,
           output_hidden_states=output_hidden_states,
           return_dict=return_dict,
       )


       # outputs[0] is the last hidden state: (batch_size, seq_len, hidden_size)
       # For classification tasks, we take the hidden state of the first token (CLS)
       pooled_output = outputs[0][:, 0, :]  # (batch_size, hidden_size)


       pooled_output = self.dropout(pooled_output)
       logits = self.classifier(pooled_output)  # (batch_size, num_labels)


       loss = None
       if labels is not None:
           if self.num_labels == 1:
               # Regression task
               loss_fct = nn.MSELoss()
               loss = loss_fct(logits.view(-1), labels.view(-1))
           else:
               # Classification task
               loss_fct = nn.CrossEntropyLoss(weight=self.weight)
               loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))


       # Return (loss, logits, hidden_states, attentions) to be consistent with Hugging Face output format
       if return_dict:
           # If using return_dict=True, we could return a ModelOutput, but for consistency with the provided code:
           return {
               "loss": loss,
               "logits": logits,
               "hidden_states": outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
               "attentions": outputs.attentions if hasattr(outputs, "attentions") else None,
           }
       else:
           outputs = (logits,) + outputs[1:]
           return ((loss,) + outputs) if loss is not None else outputs



