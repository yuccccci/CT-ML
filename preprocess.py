import sys
import os
from transformers import BertTokenizer,AutoTokenizer
import pandas as pd

class InputExample:
     def __init__(self, set_type, text, labels=None,id=None):
         self.set_type = set_type
         self.text = text
         self.labels = labels
         self.id = id

class BaseFeature:
     def __init__(self, token_ids, attention_masks, token_type_ids):
         # BERT 输入
         self.token_ids = token_ids
         self.attention_masks = attention_masks
         self.token_type_ids = token_type_ids

class BertFeature(BaseFeature):
     def __init__(self, token_ids, attention_masks, token_type_ids, labels=None,id=None):
         super(BertFeature, self).__init__(
             token_ids=token_ids,
             attention_masks=attention_masks,
             token_type_ids=token_type_ids)
         self.labels = labels
         self.id = id

def convert_bert_example(ex_idx, example: InputExample, tokenizer: BertTokenizer,
 max_seq_len,label2id):
     set_type = example.set_type
     raw_text = example.text
     labels = example.labels
     id = example.id
     # 文本元组
     callback_info = (raw_text,)
     callback_labels = labels
     callback_info += (callback_labels,)
     # 转换为one-hot编码
     print(label2id)
     labels = str(labels).strip('(').strip(')').strip('[').strip(']').strip(',').replace('', '').split(',')
     label_ids = [0 for _ in range(len(label2id))]
     print(label_ids)
     for label in labels:
         label_ids[(label2id[int(label)])] = 1
         #label_ids[label2id[label]] = 1
     encode_dict = tokenizer.encode_plus(text=raw_text,
                                         add_special_tokens=True,
                                         max_length=max_seq_len,
                                         truncation_strategy='longest_first',
                                         padding="max_length",
                                         return_token_type_ids=True,
                                         return_attention_mask=True)

     token_ids = encode_dict['input_ids']
     attention_masks = encode_dict['attention_mask']
     token_type_ids = encode_dict['token_type_ids']

     feature = BertFeature(
         token_ids=token_ids,
         attention_masks=attention_masks,
         token_type_ids=token_type_ids,
         labels=label_ids,
         id =id)
     print(feature)
     return feature, callback_info

def convert_examples_to_features(examples, max_seq_len, bert_dir, label2id) :
     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
     features = []
     callback_info = []
     for i, example in enumerate(examples):
         feature, tmp_callback = convert_bert_example(
             ex_idx=i,
             example=example,
             max_seq_len=max_seq_len,
             tokenizer=tokenizer,
             label2id = label2id
         )
         if feature is None:
             continue

         features.append(feature)
         callback_info.append(tmp_callback)
     out = (features,)
     if not len(callback_info):
         return out
     out += (callback_info,)
     return out


def out(file_path, args,bert_dir, label2id, mode, inference_list = []):
     if mode == "infer":
         data = pd.DataFrame({"sentence": inference_list})
         data["labels"] = "[]"
     else:
         data = pd.read_csv(file_path)
     examples = []
     for index, raw in data.iterrows():
         id = raw["id"]
         #labels = eval(str(raw["labels"].lstrip(',')))
         labels = eval(raw["labels"])
         examples.append(InputExample(set_type=mode,
                                         text=raw['sentence'],
                                         labels=labels,
                                         id = id))
     out = convert_examples_to_features(examples, args.max_seq_len, bert_dir, label2id)
     return out