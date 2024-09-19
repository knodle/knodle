import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel


class TextClassBert(nn.Module):
    def __init__(self, bert_backbone, bert_dropout_rate, num_classes):
        super(TextClassBert, self).__init__()
        self.num_classes = num_classes
        self.bert = AutoModel.from_pretrained(bert_backbone)
        self.drop = nn.Dropout(p=bert_dropout_rate)
        self.out = nn.Linear(self.bert.config.hidden_size, self.num_classes)

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = bert_out[0][:, 0, :]
        final_repr = bert_out['pooler_output']
        output = self.drop(final_repr)
        logits = self.out(output)
        return {'logits': logits, 'cls_repr': cls_repr, 'final_repr': final_repr}
