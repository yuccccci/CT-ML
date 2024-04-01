import math
import torch
import torch.nn as nn
import torch.nn.init as init
from transformers import BertModel, AutoTokenizer, BigBirdModel
import torch.nn as nn
class BertMLClf(nn.Module):
     def __init__(self, args):
         super(BertMLClf, self).__init__()
         self.bert = BertModel.from_pretrained("bert-base-uncased")#bert-base-uncased
         #self.bert = BigBirdModel.from_pretrained("bert-base-uncased")#bert-base-uncased
         self.bert_config = self.bert.config
         out_dims = self.bert_config.hidden_size
         self.dropout = nn.Dropout(0.3)
         self.linear = nn.Linear(out_dims, args.num_tags)

     def forward(self, token_ids, attention_masks, token_type_ids):
         bert_outputs = self.bert(
             input_ids = token_ids,
             attention_mask = attention_masks,
             token_type_ids = token_type_ids,
         )
         seq_out = bert_outputs[1]
         seq_out = self.dropout(seq_out)
         seq_out = self.linear(seq_out)
         return seq_out


class BertMLClf2(nn.Module):
    def __init__(self, args):
        super(BertMLClf2, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')#./bert-base-cased
        self.bert_config = self.bert.config
        out_dims = self.bert_config.hidden_size
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(out_dims, args.num_tags)

    def forward(self, token_ids, attention_masks, token_type_ids):
        bert_outputs = self.bert(
            input_ids=token_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids,
        )
        seq_out = bert_outputs[1]
        seq_out = self.dropout(seq_out)
        seq_out = self.linear(seq_out)
        return seq_out