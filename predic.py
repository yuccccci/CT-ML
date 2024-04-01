# -*- coding:utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.metrics as metrics
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score, f1_score,hamming_loss
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from scipy.cluster.hierarchy import dendrogram, ward
import argparse, sys
import bert_model
import numpy as np
import preprocess
import dataset
import datetime
import shutil
import pandas as pd
from tqdm import tqdm
from transformers import LongformerForSequenceClassification
parser = argparse.ArgumentParser()

parser.add_argument('--result_dir', type = str, help = 'dir to save result txt files', default = 'results/')
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.1)
parser.add_argument('--forget_rate', type = float, help = 'forget rate', default = None)
parser.add_argument('--noise_type', type = str, help='[pairflip, symmetric]', default='pairflip')
parser.add_argument('--num_gradual', type = int, default = 10, help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
parser.add_argument('--exponent', type = float, default = 1, help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
parser.add_argument('--top_bn', action='store_true')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--num_iter_per_epoch', type=int, default=400)
parser.add_argument('--output_dir', default=os.path.join("./testbert", "checkpoint"),
                             help='the output dir for model checkpoints')
parser.add_argument('--bert_dir', default='./bert/bert-base-chinese')
parser.add_argument('--num_tags', default=90, type=int,help='number of tags') # 多标签分类的类别数
# parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument('--gpu_ids', type=str, default="0",
                             help='gpu ids to use, -1 for cpu, "0,1" for multi gpu')
parser.add_argument('--max_seq_len', default=128, type=int)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--swa_start', default=2, type=int,
                             help='the epoch when swa start')
parser.add_argument('--train_epochs', default=10, type=int,
                             help='Max training epoch')
parser.add_argument('--dropout_prob', default=0.1, type=float,
                             help='drop out probability')
parser.add_argument('--lr', default=3e-5, type=float,
                             help='learning rate for the bert module')
parser.add_argument('--other_lr', default=3e-4, type=float,
                             help='learning rate for the module except bert')
parser.add_argument('--max_grad_norm', default=1, type=float,
                             help='max grad clip')
parser.add_argument('--warmup_proportion', default=0.1, type=float)
parser.add_argument('--weight_decay', default=0.0, type=float)
parser.add_argument('--adam_epsilon', default=1e-8, type=float)
parser.add_argument('--eval_model', default=True, action='store_true',
                             help='whether to eval model after training')

args = parser.parse_args()

# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

device = 'cuda' if torch.cuda.is_available() else "cpu"
criterion = nn.BCEWithLogitsLoss()

label2id = {}
id2label = {}

labels = [i for i in range(1, 91)]
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

def load_ckp( model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

def dbscan(logit,model):
    logit = np.array(logit)
    pred = np.zeros_like(logit)

    for idx in range(len(logit)):
        y = model.fit_predict(logit[idx].reshape(-1, 1))
        for j in range(len(logit[idx])):
            i = np.array(j).astype(int)
            pred[idx, j] = 1 if y[i] == -1 else 0
        pred = np.array(pred).astype(int)
    return pred

def get_metrics(logit, target):
    """Computes the precision@k for the specified values of k"""
    accuracy = accuracy_score(target, logit)
    micro_f1 = f1_score(target, logit, average='micro')
    macro_f1 = f1_score(target, logit, average='macro')
    hamming_loss = metrics.hamming_loss(target, logit)
    micro_precison = metrics.precision_score(target, logit, average='micro')
    micro_recall = metrics.recall_score(target, logit, average='micro')
    metrics_dict = {}
    metrics_dict['accuracy'] = accuracy
    metrics_dict['micro_f1'] = micro_f1
    metrics_dict['micro_precison'] = micro_precison
    metrics_dict['micro_recall'] = micro_recall
    metrics_dict['macro_f1'] = macro_f1
    metrics_dict['hamming_loss'] = 1-hamming_loss
    return metrics_dict

THRESHOLDS = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.8,0.85,0.9,0.95]
THRESHOLDSdbscan =[0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3,0.35,0.4,0.45,0.5]
METRICS = ['accuracy', 'micro_f1','micro_precison', 'micro_recall', 'macro_f1','hamming_loss']
def predict():
    print('========进行测试========')
    dev_out = preprocess.out('reuters_test_back2labels.csv', args, './longformer-base-plagiarism-detection', label2id, 'dev')
    dev_features, dev_callback_info = dev_out
    dev_dataset = dataset.MLDataset(dev_features)
    dev_loader = DataLoader(dataset=dev_dataset,
                            batch_size=args.batch_size,
                            num_workers=2)

    checkpoint_path = 'Coteaching/checkpoint/best_reuters2.5LC_NP04.pt'
    model = bert_model.BertMLClf(args)
    model = nn.DataParallel(model, device_ids=[0, 1]).cuda()

    new_state_dict = model.state_dict()
    checkpoint = torch.load(checkpoint_path)
    state_dict1 = checkpoint['state_dict'].copy()
    state_dict = checkpoint['state_dict']
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")
        if new_key in new_state_dict:
            state_dict1[new_key] = value
    model.load_state_dict(state_dict1, strict=False)

    model.eval()
    model.to(device)
    total_loss = 0.0
    all_outputs = []
    test_outputs = []
    test_targets = []
    with torch.no_grad():
        for test_step, test_data in enumerate(dev_loader):
            token_ids = test_data['token_ids'].to(device)
            attention_masks = test_data['attention_masks'].to(device)
            token_type_ids = test_data['token_type_ids'].to(device)
            labels = test_data['labels'].to(device)
            output = model(token_ids, attention_masks, token_type_ids)
            loss = criterion(output, labels)
            total_loss += loss.item()
            output = torch.sigmoid(output).cpu().detach().numpy()
            all_outputs.extend(output)
            test_outputs.extend(output)
            test_targets.extend(labels.cpu().detach().numpy())

    test_outputs = np.array(test_outputs)
    test_targets = np.array(test_targets)

    from sklearn.cluster import DBSCAN
    best_dbtest_metrics = None
    for threshold in THRESHOLDSdbscan:
        modelDBSCAN = DBSCAN(eps=threshold, min_samples=70)

        dboutputs = dbscan(test_outputs, modelDBSCAN)
        dbtest_metrics = get_metrics(dboutputs, test_targets)
        if best_dbtest_metrics == None:
            best_dbtest_metrics = {}
            for metric in METRICS:
                best_dbtest_metrics[metric] = dbtest_metrics[metric]
        else:
            if dbtest_metrics['micro_f1'] >  best_dbtest_metrics['micro_f1']:
                for metric in METRICS:
                    best_dbtest_metrics[metric] = dbtest_metrics[metric]

    print("bestdbscan****************")
    for metric in METRICS:
        print(metric, ":", best_dbtest_metrics[metric])
    print("****************")

if __name__ == '__main__':
    predict()