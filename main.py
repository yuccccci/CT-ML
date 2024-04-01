# -*- coding:utf-8 -*-
import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms

import argparse, sys
import numpy as np
import datetime
import shutil
import pandas as pd
from tqdm import tqdm

from loss_llm import loss_coteaching

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', default=os.path.join("./Coteaching", "checkpoint"), help='the output dir for model checkpoints')
parser.add_argument('--result_dir', type = str, help = 'dir to save result txt files', default = 'results/')
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.45)
parser.add_argument('--forget_rate', type = float, help = 'forget rate', default = None)
parser.add_argument('--noise_type', type = str, help='[pairflip, symmetric]', default='pairflip')
parser.add_argument('--num_gradual', type = int, default = 10, help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
parser.add_argument('--exponent', type = float, default = 1, help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
parser.add_argument('--top_bn', action='store_true')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=200)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--num_iter_per_epoch', type=int, default=400)
parser.add_argument('--bert_dir', default='./bert/bert-base-chinese')
parser.add_argument('--num_tags', default=90, type=int,help='number of tags') # 多标签分类的类别数
# parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument('--gpu_ids', type=str, default="0",
                             help='gpu ids to use, -1 for cpu, "0,1" for multi gpu')
parser.add_argument('--max_seq_len', default=128, type=int)
parser.add_argument('--batch_size', default=10, type=int)
parser.add_argument('--swa_start', default=2, type=int,
                             help='the epoch when swa start')
parser.add_argument('--train_epochs', default=15, type=int,
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
parser.add_argument('--interval', default=0.05, type=float)
parser.add_argument('--eval_model', default=True, action='store_true',
                             help='whether to eval model after training')

args = parser.parse_args()

# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Hyper Parameters
batch_size = 128
learning_rate = args.lr

from pprint import pprint
import os
import shutil
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertTokenizer
import sys

import preprocess
import dataset
import bert_model
import pandas as pd

label2id = {}
id2label = {}
labels = [i for i in range(1, 91)]
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label
print(label2id)
print(id2label)
device = 'cuda' if torch.cuda.is_available() else "cpu"
criterion = nn.BCEWithLogitsLoss()
# load dataset

THRESHOLDSdbscan = [0.05, 0.10, 0.13, 0.15, 0.20, 0.23, 0.25, 0.30]
METRICS = ['accuracy', 'micro_f1', 'macro_f1','hamming_loss']

if args.forget_rate is None:
    forget_rate=args.noise_rate
else:
    forget_rate=args.forget_rate


# define drop rate schedule
rate_schedule = np.ones(args.train_epochs)*forget_rate
rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate, args.num_gradual)


def get_metrics(logit, target):
    """Computes the precision@k for the specified values of k"""
    # logit = self.logit_tomultilabel(logit)
    # batch_size = target.size(0)
    accuracy = accuracy_score(target, logit)
    micro_f1 = f1_score(target, logit, average='micro')
    macro_f1 = f1_score(target, logit, average='macro')
    hm = hamming_loss(target, logit)
    metrics_dict = {}
    metrics_dict['accuracy'] = accuracy
    metrics_dict['micro_f1'] = micro_f1
    metrics_dict['macro_f1'] = macro_f1
    metrics_dict['hamming_loss'] = 1 - hm
    return metrics_dict


'''  _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res'''


def dbscan(logit, model):
    logit = np.array(logit)
    pred = np.zeros_like(logit)
    for idx in range(len(logit)):
        y = model.fit_predict(logit[idx].reshape(-1, 1))
        for j in range(len(logit[idx])):
            i = np.array(j).astype(int)
            pred[idx, j] = 1 if y[i] == -1 else 0
        pred = np.array(pred).astype(int)
    return pred

# Train the Model
def save_ckp( state, checkpoint_path):
    torch.save(state, checkpoint_path)


def dev(dev_loader, model):
    model.eval()
    total_loss = 0.0
    dev_outputs = []
    dev_targets = []
    with torch.no_grad():
        for dev_step, dev_data in enumerate(dev_loader):
            token_ids = dev_data['token_ids'].to(device)
            attention_masks = dev_data['attention_masks'].to(device)
            token_type_ids = dev_data['token_type_ids'].to(device)
            labels = dev_data['labels'].to(device)
            outputs = model(token_ids, attention_masks, token_type_ids)
            outputs = torch.sigmoid(outputs)
            dev_outputs.extend(outputs.tolist())
            dev_targets.extend(labels.cpu().detach().numpy().tolist())
    return  dev_outputs, dev_targets

def train(train_loader, dev_loader, epoch, model1, optimizer1, model2, optimizer2,best_dev_micro_f1, best_dev_micro_f12 ):
    print( 'Training ...' )

    for train_step, train_data in enumerate(train_loader):
        #model.train()

        token_ids = train_data['token_ids'].to(device)
        attention_masks = train_data['attention_masks'].to(device)
        token_type_ids = train_data['token_type_ids'].to(device)
        labels = train_data['labels'].to(device)
        ind = train_data['id']

        # Forward + Backward + Optimize
        #model1
        train_outputs1 = model1(token_ids, attention_masks, token_type_ids)

        train_outputs2 = model2(token_ids, attention_masks, token_type_ids)


        loss_1, loss_2 = loss_coteaching(train_outputs1, train_outputs2, labels, args, epoch)



        optimizer1.zero_grad()
        loss_1.backward()

        optimizer2.zero_grad()
        loss_2.backward()

        optimizer1.step()
        optimizer2.step()

        outputs1 = torch.sigmoid(train_outputs1).cpu().detach().numpy()
        outputs1 = (np.array(outputs1) > 0.35).astype(int)
        outputs2 = torch.sigmoid(train_outputs2).cpu().detach().numpy()
        outputs2 = (np.array(outputs2) > 0.35).astype(int)

        train_metrics1 = get_metrics(outputs1, labels.cpu())
        train_metrics2 = get_metrics(outputs2, labels.cpu())
        print("****************")
        print(
            "[train] epoch:{} step:{}/{} loss1:{:.6f} loss2:{:.6f}".format(
                epoch, args.train_epochs, train_step, loss_1.item(), loss_2.item()))
        for metric in METRICS:
            print(metric, "1:", train_metrics1[metric])
        for metric in METRICS:
            print(metric, "2:", train_metrics2[metric])
        print("****************")

        if (train_step) % args.print_freq == 0:
            dev_outputs1, dev_targets1 = dev(dev_loader,model1)
            from sklearn.cluster import DBSCAN
            best_dbtest_metrics1 = None
            for threshold in THRESHOLDSdbscan:
                modelDBSCAN1 = DBSCAN(eps=threshold, min_samples=70)
                dboutputs1 = dbscan(dev_outputs1, modelDBSCAN1)
                dbtest_metrics1 = get_metrics(dboutputs1, dev_targets1)
                if best_dbtest_metrics1 == None:
                    best_dbtest_metrics1 = {}
                    for metric in METRICS:
                        best_dbtest_metrics1[metric] = dbtest_metrics1[metric]
                else:
                    for metric in METRICS:
                        best_dbtest_metrics1[metric] = max(best_dbtest_metrics1[metric], dbtest_metrics1[metric])
            print("bestdbscan1--------------------")
            for metric in METRICS:
                print(metric, ":", best_dbtest_metrics1[metric])
            print("-----------------")
            if best_dbtest_metrics1['micro_f1'] > best_dev_micro_f1:
                print("------------>save the best")
                checkpoint1 = {
                    'epoch': epoch,
                    'state_dict': model1.state_dict(),
                    'optimizer': optimizer1.state_dict(),
                }
                best_dev_micro_f1 = best_dbtest_metrics1['micro_f1']
                checkpoint_path1 = os.path.join(args.output_dir, 'best_reuters2.5LC_NP04.pt')
                save_ckp(checkpoint1, checkpoint_path1)
    return best_dev_micro_f1, best_dev_micro_f12



def main():
    # Data Loader (Input Pipeline)
    train_out = preprocess.out('reuters_train_0.4_repeat.csv', args, '', label2id, 'train')
    features, callback_info = train_out
    train_dataset = dataset.MLDataset(features)
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              sampler=train_sampler,
                              num_workers=2)

    dev_out = preprocess.out('reuters_val_back2labels.csv', args, '', label2id, 'dev')
    dev_features, dev_callback_info = dev_out
    dev_dataset = dataset.MLDataset(dev_features)
    dev_loader = DataLoader(dataset=dev_dataset,
                            batch_size=args.batch_size,
                            num_workers=2)
    # Define models

    bert1 = bert_model.BertMLClf(args)
    bert1 = nn.DataParallel(bert1, device_ids=[0, 1]).cuda()
    optimizer1 = torch.optim.Adam(bert1.parameters(), lr=learning_rate, weight_decay=args.weight_decay)

    bert2 = bert_model.BertMLClf2(args)
    bert2 = nn.DataParallel(bert2, device_ids=[0, 1]).cuda()
    #bert2.cuda()
    optimizer2 = torch.optim.Adam(bert2.parameters(), lr=learning_rate, weight_decay=args.weight_decay)

    epoch = 0
    best_dev_micro_f1 = 0.0
    best_dev_micro_f12 = 0.0
    # training
    for epoch in range(1, args.train_epochs):
        bert1.train()
        bert2.train()

        best_dbtest_metrics1, best_dbtest_metrics2 = train(train_loader, dev_loader, epoch, bert1, optimizer1, bert2, optimizer2, best_dev_micro_f1, best_dev_micro_f12)
        best_dev_micro_f1 = best_dbtest_metrics1
        best_dev_micro_f12 = best_dbtest_metrics2

if __name__=='__main__':
    main()